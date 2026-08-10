# QEMU CXL Type-2 Device Model

这个仓库回答一个具体问题：guest里的普通CUDA程序怎样把设备发现、内存分配、模块加载、kernel launch和数据传输交给host上的真实L40，同时让CXL memory、cache、coherency和backing store成为可观察、可修改的系统状态。

QEMU在这条链路里创建guest能够发现的PCI function，并连接guest地址空间与host进程地址空间。guest写入的是BAR寄存器、BAR内存和对象编号；host侧Concordia与NVIDIA Driver消费的是进程内指针、CUDA handle和真实`CUdeviceptr`。本仓把前一组状态翻译成后一组状态，再把完成状态、返回码和输出字节写回guest。

```text
guest观察到的设备
  PCI config + CXL DVSEC + BAR0/BAR2/BAR4
                  │
                  │ MMIO、BAR内存、对象ID
                  v
QEMU cxl-type2
  ├── 保存guest可见寄存器和MemoryRegion
  ├── 把CUDA命令ID还原成host函数调用
  ├── 保存module/function等guest ID到host handle的映射
  ├── 维护cache、device_mem、coherency和fabric-memory状态
  ├── 调用Concordia/HetGPU → NVIDIA Driver → L40
  └── 向CXLMemSim发送CXL memory request
                  │
                  v
guest读取result、data window和最终计算结果
```

[QEMU Camp教程](https://qemu.gevico.online/tutorial/2026/ch3/qemu-cxlemu/)用这台设备研究两件会在推理过程中汇合的事：kernel怎样在加速器上执行，模型权重、KV cache和workspace怎样到达这些kernel。CUDA命令路径回答前一个问题；BAR memory、CXLMemSim与显式bulk copy回答后一个问题。Kimi correctness要求整条调用链保持CUDA语义，后续TPS实验再改变数据放置、cache、prefetch、backing store和kernel实现。

## CXL概念怎样落到当前源码

[CXL规范与官方资源](https://computeexpresslink.org/cxl-specification/)把Type-2设备定义在带本地加速器计算能力的设备类别中。这类设备可以组合`CXL.io`、`CXL.cache`和`CXL.mem`：

- `CXL.io`承载PCIe风格的发现、配置、中断和MMIO。
- `CXL.cache`让设备按照cache-line一致性协议访问host memory。
- `CXL.mem`让host按照内存访问语义使用device-attached memory。

OS与runtime可以把这些资源组织成类似NUMA的心智模型。程序继续围绕地址、allocation、load/store和一致性推理；物理位置、容量、带宽、延迟、placement和migration policy保留各自差异。这里的“统一”发生在寻址、访问和一致性合同上；一次访问的代价仍由数据所在资源决定。

当前源码把教程需要的能力组合在名为`cxl-type2`的QEMU对象中。准确理解这台设备需要同时读取QOM名称、guest可见DVSEC、BAR布局和运行日志：

| 源码构造 | 当前实现写入的状态 | guest或host怎样消费 |
| --- | --- | --- |
| `cxl_type2_realize()` | 注册BAR0、BAR2和BAR4 | guest枚举PCI function并映射BAR |
| `build_dvsecs()`与`cxl_type2_reset()` | 当前使用`CXL2_TYPE3_DEVICE`编码组件类型 | guest driver根据真实DVSEC与PCI身份建立设备对象 |
| BAR2 `cache_mem` | 复用cache MemoryRegion，起始`CXL_GPU_CMD_REG_SIZE`由CUDA命令handler解释 | guest shim通过`resource2`提交项目自定义CUDA ABI |
| BAR4 `device_mem` | 复用device-attached-memory表示，包含coherent pool和fabric-memory状态 | guest BAR访问、bulk copy和coherency代码消费BAR4 offset |
| `hetgpu-lib` | Concordia `libnvcuda.so`的绝对路径 | QEMU加载host CUDA/HetGPU实现并调用真实Driver |
| `cxlmemsim-addr`与`cxlmemsim-port` | CXLMemSim request/response连接 | cache miss、memory write及相关CXL操作产生server请求 |

这张表说明了当前代码的实现形状：Type-2是项目希望表达的加速器角色；guest实际看到的协议身份由当前DVSEC与BAR实现共同决定。源码位置见[`hw/cxl/cxl_type2.c`](hw/cxl/cxl_type2.c)、[`include/hw/cxl/cxl_type2.h`](include/hw/cxl/cxl_type2.h)和[`include/hw/cxl/cxl_type2_gpu_cmd.h`](include/hw/cxl/cxl_type2_gpu_cmd.h)。

## 一块Kimi数据在链路中的身份

同一批模型字节经过这台设备时会出现多种地址和句柄。allocation返回值、对象表、copy命令与BAR offset共同建立这些身份之间的关系。

| 状态所在层 | 使用的身份 | 消费者 |
| --- | --- | --- |
| guest llama/GGML | guest virtual address、tensor和GGML buffer | 模型加载器、GGML scheduler、CUDA backend |
| guest CUDA shim | CUDA pointer、module/function handle和BAR2 payload | `libcuda.so.1` shim |
| QEMU BAR2 | command、parameter、result和1 MiB data window offset | `cxl_type2_gpu_cmd_read/write()` |
| QEMU BAR4 | `device_mem` offset、coherent-pool offset和fabric range | cache/coherency、bulk transfer、CXLMemSim路径 |
| host HetGPU/NVIDIA | host CUDA context、module/function handle、真实`CUdeviceptr` | Concordia、NVIDIA Driver和L40 kernel |
| CXLMemSim | request address、size、operation、timestamp和payload | backing store、latency/cache模型和统计 |

因此，一条可审计的buffer证据需要记录“哪个guest tensor触发了哪次allocation/copy，它对应哪个BAR offset或真实`CUdeviceptr`，哪个kernel随后消费了它”。这组映射让一个十六进制地址获得所在地址空间和consumer语义。

## 数据怎样进入真实L40与CXL内存模型

当前代码提供三条可以分别观察的路径。

### 普通CUDA copy路径

guest shim把一段数据写进BAR2的1 MiB data window，然后提交`CXL_GPU_CMD_MEM_COPY_HTOD`。QEMU读取`gpu_cmd.data`，调用`hetgpu_memcpy_htod()`，Concordia/NVIDIA Driver把字节写入真实`CUdeviceptr`。

```text
guest buffer
  → BAR2 data window
  → QEMU gpu_cmd.data
  → hetgpu_memcpy_htod(real CUdeviceptr)
  → NVIDIA Driver
  → L40 allocation
```

`MEM_COPY_HTOD/DTOH`代码还包含一个按数值范围执行的BAR4 shadow更新条件：`dev_ptr + size <= device_mem_size`。真实NVIDIA `CUdeviceptr`与BAR4 offset属于不同地址空间，运行证据需要同时记录pointer、`device_mem_size`和该条件的执行结果。普通CUDA copy的直接终点是GPU allocation；BAR4 shadow属于满足该数值条件后执行的附加状态更新。

### 显式BAR4 bulk路径

`CXL_GPU_CMD_BULK_HTOD`同时携带BAR4 offset、真实GPU destination pointer和长度。QEMU从`device_mem + bar4_offset`取字节，再调用`hetgpu_memcpy_htod()`。`BULK_DTOH`沿相反方向写回BAR4。这两个命令在源码中明确建立CXL device-memory表示与真实GPU allocation之间的映射。

```text
BAR4 device_mem offset
  ↔ BULK_HTOD / BULK_DTOH
  ↔ real CUdeviceptr
  ↔ L40 kernel
```

### BAR memory与CXLMemSim路径

BAR2命令窗口之外的cache访问和BAR4 device-memory访问进入`cxl_type2_cache_read/write()`。cache miss与write构造包含operation、address、size、timestamp和payload的请求，发送给CXLMemSim server。coherency、flush、prefetch、bias、DCD、GFAM和MH-SLD命令也在这组设备状态上工作。

当前`cxl_type2_device_mem_read/write()`把BAR4访问转交给同一组cache handler；cache handler会把低于`CXL_GPU_CMD_REG_SIZE`的offset解释成GPU command区域。这个调用关系使BAR4低地址访问与BAR2命令窗口共享同一个offset判定，而bulk命令通过`memory_region_get_ram_ptr(&device_mem)`直接读取或写入BAR4存储。分析低地址BAR4行为时，需要按照这两个具体入口区分MMIO handler路径与bulk direct-memory路径。

```text
guest BAR access
  → QEMU cache/device_mem handler
  → cache-line与coherency状态
  → CXLMemSim request
  → backing store / latency / statistics
```

三条路径在“同一块业务数据”上汇合时，CXL优化才会改变Kimi的数据供给。正式证据需要把Kimi tensor或buffer关联到BAR4/CXLMemSim请求，再关联到消费它的GGML operator或L40 kernel。普通CUDA copy与kernel launch已经覆盖计算路径；显式bulk或可归属的BAR memory request覆盖CXL供数路径。

## 一次kernel调用怎样穿过QEMU

以guest中的`cuLaunchKernel`为例：

```text
llama / GGML CUDA backend
  → guest libcudart调用CUDA Driver API
  → guest libcuda.so.1 shim编码function ID、grid、block、shared memory和参数
  → BAR2 command registers + data window
  → CXL_GPU_CMD_LAUNCH_KERNEL
  → QEMU cxl_type2_gpu_execute_cmd()
  → QEMU function table把guest ID解析成host function handle
  → cxl_hetgpu调用Concordia libnvcuda.so
  → Concordia调用NVIDIA Driver
  → L40执行kernel
  → Driver/HetGPU返回launch与sync状态
  → QEMU写cmd_result和result registers
  → guest shim还原CUresult
  → llama继续执行GGML图
```

context、allocation、module load、function lookup、attribute、occupancy、stream、event、copy和synchronize都沿用这个“guest编码状态—QEMU保存映射—host执行—guest读取完成状态”的基本结构。每类命令的字段合同由两端共享的BAR2 ABI定义。

## 从设备出现到Kimi正确性的证据递进

```text
QEMU注册PCI function与BAR
  → guest PCI枚举和driver probe
  → guest shim映射resource2并读取magic/version
  → QEMU收到有序command sequence
  → HetGPU/Concordia获得真实Driver返回
  → launch、sync与DtoH产生预期数值
  → baseline和concordia完成同一Kimi workload
  → 独立comparator确认输出一致
```

每一步都消费上一步产生的状态。源码与日志的对应关系如下：

| 状态产生点 | 运行观察 | 当前观察确认的事实 |
| --- | --- | --- |
| `cxl_type2_realize()`注册PCI/BAR | realized日志、PCI config、BAR resource | QEMU设备实例和guest可见资源已经建立 |
| guest driver probe | dmesg、sysfs、`cache0`和`mem0` | guest kernel已经为真实PCI/DVSEC创建对象 |
| guest shim映射`resource2` | BAR2 magic/version/capabilities | guest userspace已经进入项目自定义CUDA transport |
| `cxl_type2_gpu_execute_cmd()` | `cmd_begin/cmd_end` sequence与result | 某个CUDA语义已经由QEMU解码并完成 |
| `cxl_hetgpu`与Concordia | Driver return、module/function、launch和sync | host backend已经消费该命令 |
| DtoH数值oracle | 预期字节、tensor或probe结果 | 被覆盖的计算语义已经返回guest |
| BAR4/CXLMemSim关联记录 | buffer、BAR offset、request和consumer | 被命名的数据已经进入CXL供数路径 |
| same-VM paired comparator | 两个case的原始输出与verdict | 当前run manifest声明的Kimi语义已经对齐 |

从某一行继续调试时，先读取这一行的状态产生点与原始日志，再进入下一行的consumer。这样可以把“设备出现”“CUDA可达”“CXL供数”和“Kimi正确”保持为一条连续、可复查的因果链。

## 构建时的全链路

### 构建、组装与运行控制

```text
各组件源码仓
  │
  ├── qemu-cxl-type2 ──build/publish──> QEMU OCI component
  ├── linux-cxl-type2 ────────────────> kernel OCI component
  ├── cxlmemsim ──────────────────────> server + guest shim OCI component
  ├── concordia ──────────────────────> libnvcuda.so OCI component
  └── llama-cpp ──────────────────────> llama/GGML/CUDA userland component
                                                │
                                                v
                                      type2-guest组装器
                                        ├── bzImage
                                        └── initrd.cpio.gz

cxl-lab current run manifest
  │ 固定QEMU、kernel/guest、CXLMemSim、Concordia digest
  │ 固定VM内存、CPU、CXL窗口、模型、oracle和result repository
  v
CNB runner
  ├── 恢复所有OCI component
  ├── 启动CXLMemSim server
  ├── 启动本仓qemu-system-x86_64
  ├── 把type2-guest的bzImage/initrd交给QEMU
  └── 收集guest、QEMU、CXLMemSim、GPU和workload结果
```

`type2-guest`生成QEMU启动所需的`bzImage`与`initrd.cpio.gz`。`cxl-lab`选择exact QEMU component、guest component和相邻组件，并构造一轮可重放运行。设备行为来自本仓构建出的`qemu-system-x86_64`。

### QEMU启动后的请求路径

```text
guest workload
  ├── Kimi：llama-completion → GGML CUDA backend
  └── tiny：CUDA/cuBLAS probe
          │
          v
guest CUDA userland
  libcudart / libcublas / libcublasLt
          │ CUDA Driver API
          v
guest libcuda.so.1 shim（来自CXLMemSim component）
  │ 找到8086:0d92 PCI function
  │ open /sys/bus/pci/devices/<BDF>/resource2
  │ mmap BAR2
  │ 写参数、data window与CXL_GPU_REG_CMD
  v
QEMU cxl-type2（本仓）
  │
  ├── BAR2 register/data protocol
  │     └── cxl_type2_gpu_execute_cmd
  │
  ├── CUDA执行支路
  │     cxl_hetgpu
  │       → dlopen Concordia libnvcuda.so
  │       → CUDA Driver API / HetGPU状态
  │       → NVIDIA Driver
  │       → L40
  │
  └── CXL memory与coherency支路
        cache_mem / device_mem / BAR4 shadow
          → cache、bias、flush、invalidate、bulk transfer
          → 本设备的CXLMemSim TCP request/response
          → backing store与访问统计

结果返回：
L40或CXLMemSim结果
  → QEMU cmd_result/results/data window
  → BAR2
  → guest shim还原CUresult与输出参数
  → CUDA runtime / GGML / llama
```

CXLMemSim与Concordia位于QEMU下游的两条相交支路。CUDA allocation、module、function、launch和synchronize主要进入HetGPU；CXL memory access、cache与backing-store请求进入CXLMemSim。显式bulk copy把BAR4 offset与真实GPU pointer连接起来；普通copy在满足源码中的数值范围条件时更新BAR4 shadow。运行时因此形成两条由具体copy操作汇合的支路。

## 与其他仓库怎样交接状态

| 仓库 | 该仓产生的状态 | 这些状态怎样进入或离开QEMU |
| --- | --- | --- |
| [`cxl-lab`](https://cnb.cool/gevico.online/jensen/cxl-lab) | exact component digest、QEMU参数、guest路径、oracle和result合同 | runner把这些声明展开成QEMU命令行，并收集QEMU日志与运行结果 |
| [`type2-guest`](https://cnb.cool/gevico.online/jensen/type2-guest) | 包含kernel、driver、shim、CUDA userland和workload的`bzImage`与`initrd.cpio.gz` | QEMU把它们作为guest启动输入 |
| [`linux-cxl-type2`](https://cnb.cool/gevico.online/jensen/linux-cxl-type2) | guest `cxl_type2_accel` driver以及`cache0`、`mem0`设备对象 | driver消费QEMU暴露的PCI identity、DVSEC和BAR，创建guest kernel状态 |
| [`cxlmemsim`](https://cnb.cool/gevico.online/jensen/cxlmemsim) | guest `libcuda.so.1`、共享BAR2 ABI与CXLMemSim server | shim向QEMU编码命令；QEMU向server发送带地址和时序的memory request |
| [`concordia`](https://cnb.cool/gevico.online/jensen/concordia) | host `libnvcuda.so`、HetGPU对象生命周期与NVIDIA调用实现 | QEMU通过`hetgpu-lib=<absolute-path>`加载它，并把已解码CUDA操作交给它 |
| [`cxl-models`](https://cnb.cool/gevico.online/jensen/cxl-models) | Kimi分片、SHA256与LFS对象 | runner把模型提供给guest llama；llama产生的CUDA/BAR状态随后进入QEMU |

跨仓目标和correctness阶梯见[`cxl-memsim/AGENTS.md`](https://cnb.cool/gevico.online/jensen/cxl-memsim/-/blob/main/AGENTS.md)。QEMU component怎样进入正式run见[`cxl-lab/manifests/AGENTS.md`](https://cnb.cool/gevico.online/jensen/cxl-lab/-/blob/main/manifests/AGENTS.md)。

## BAR承担什么职责

### BAR0：CXL component registers

BAR0保存CXL component register block。guest `cxl_type2_accel`用它发现CXL能力、HDM decoder和相关寄存器。可选tmatmul硬件控制面也约定在BAR0的独立CSR窗口出现；当前Kimi使用BAR2 CUDA链，tmatmul CSR服务另一条可选硬件控制路径。

### BAR2：guest CUDA命令与cache窗口

QEMU把`cache_mem`注册为PCI BAR2，并在起始范围叠加`cxl_type2_cache_ops`。guest shim直接mmap sysfs `resource2`，因此它看到的就是这块BAR2 `MemoryRegion`。

BAR2起始区域包含：

```text
magic / version / capabilities
command status / command result
parameter registers
result registers
command data window
```

guest写入`CXL_GPU_REG_CMD`后，QEMU进入`cxl_type2_gpu_execute_cmd()`。命令覆盖device/context、allocation、copy、module load、function lookup、attribute、occupancy、kernel launch、sync、case begin/end、cache/coherency与扩展memory操作。

guest和QEMU对协议的定义必须一致：

- guest侧：[`cxlmemsim/qemu_integration/guest_libcuda/cxl_gpu_cmd.h`](https://cnb.cool/gevico.online/jensen/cxlmemsim/-/blob/main/qemu_integration/guest_libcuda/cxl_gpu_cmd.h)
- QEMU侧：[`include/hw/cxl/cxl_type2_gpu_cmd.h`](include/hw/cxl/cxl_type2_gpu_cmd.h)

任何一侧单独增加命令、参数或返回值都会制造ABI断裂。

### 2D DtoD描述符的stream语义

共享描述符协议当前常量版本为`2`。它与CXLMemSim guest shim配对：`CXL_GPU_CMD_MEM_COPY_2D_DTOD`在`PARAM6`携带`stream_wire`。QEMU接受版本`1..2`；版本`2`且该字段能解析为非sentinel stream时，调用`hetgpu_memcpy2d_dtod_async()`，由动态加载的`cuMemcpy2DAsync_v2`在该stream上提交copy。

版本`1`、sentinel stream或无法解析的stream保持既有`hetgpu_memcpy2d_dtod()`同步路径。这个分支保留旧shim兼容性，也使缺失的stream信息保持可见：它不会被推断为任意异步stream。

`copy_driver` trace把这次选择写为两个字段：`stream_forwarded=1`证明stream已经透传到当前copy路径；`implementation`记录实际实现路径，例如同步legacy或异步direct。两者必须一起读取：前者不能单独证明Driver已在目标stream完成copy，后者也不能替代BAR2描述符的输入证据。

### BAR4：device-attached memory与bulk window

QEMU把`device_mem`注册为BAR4。它承载Type-2 device-attached memory、bulk copy的中间窗口、coherent pool、dynamic-capacity和部分shadow/coherency状态。

真实NVIDIA allocation由HetGPU/NVIDIA Driver拥有；QEMU的BAR4/device_mem承担guest可见CXL memory与coherency模型。显式bulk命令携带BAR4 offset和真实GPU pointer，从而建立两者的直接映射；普通copy的shadow更新受`dev_ptr + size <= device_mem_size`条件约束。地址身份由这些操作与对象表确定。

## `cxl_type2_accel`、`cxl-gpu0`与`/dev/cxl_gpu0`

这三个名字来自三层：

```text
cxl_type2_accel
  Linux guest PCI driver名称
  来源：linux-cxl-type2/drivers/cxl/cxl_type2_accel.c

id=cxl-gpu0
  QEMU命令行中的对象ID
  用于QEMU对象图引用和日志辨识

/dev/cxl_gpu0
  导师描述的guest设备路径
  创建规则与UAPI作为独立接口合同继续在evidence中追踪
```

QEMU的`id=cxl-gpu0`给QEMU对象图和日志提供实例名称。Linux设备节点由guest driver的注册代码创建。当前source inventory已经把`/dev/cxl/cache0`和`/dev/cxl/mem0`映射到`cxl_type2_accel`，guest shim则通过PCI `resource2`访问BAR2；`/dev/cxl_gpu0`的创建函数、file operations和userspace ABI继续由设备节点合同追踪。

导师已经确认driver仓是[`vickiegpt/linux-cxl-type2`](https://github.com/vickiegpt/linux-cxl-type2/tree/main)。该来源确定了driver身份；[`cxl-gpu-device-node-contract.md`](https://cnb.cool/gevico.online/jensen/cxl-lab/-/blob/main/docs/evidence/cxl-gpu-device-node-contract.md)保存节点创建者、UAPI与运行证据的后续调查。

## HetGPU与Concordia怎样接入

QEMU属性`hetgpu-lib`接收一个绝对路径。realize阶段初始化`HetGPUState`并加载该动态库。当前正式运行使用Concordia component里的`libnvcuda.so`。

```text
QEMU cxl_type2
  → cxl_hetgpu wrapper
  → dlopen(hetgpu-lib)
  → cuInit / context / allocation / module / function / launch / sync
  → NVIDIA Driver
```

QEMU还维护module和function handle表、paired case epoch、command sequence与返回码。guest把QEMU分配的稳定ID作为opaque token使用，QEMU再把它解析成host pointer。core或日志中的handle问题需要同时检查guest编码、BAR2参数、QEMU表和Concordia/NVIDIA返回值。

### 2D DtoD同步定罪探针

`hetgpu_memcpy2d_dtod()`读取`HETGPU_PROBE_SYNC_BEFORE_2D_DTOD`。值为`1`或`true`时，它在同步2D DtoD copy之前注入`cuCtxSynchronize()`；默认关闭。该门控只用于定罪实验：若强制上下文同步改变结果，它证明待确认的异步stream语义影响了该copy前的可见性。它不证明丢失stream发生在哪一层，也不构成正式路径的同步策略。

## CXLMemSim怎样接入

属性`cxlmemsim-addr`和`cxlmemsim-port`指定server。默认TCP路径由本设备建立request/response连接；QEMU把CXL read/write/fence等请求连同地址、大小、时间戳和数据发送给server，并消费响应状态。

环境变量把transport设为`shm`或`pgas`时，`cxlmemsim_connect()`把`use_shm`置位并跳过本设备的TCP连接，源码注释把共享内存连接交给Type-3设备路径。这个分支的证据入口是transport选择日志与对应Type-3状态；Type-2文件中的`cxl_type2_memsim_request_ext()`在`use_shm`状态下直接返回。

证据沿消费关系递进：TCP连接成功确认QEMU与server控制面可达；一次带地址、大小和操作类型的request/response确认对应CXL操作进入server；buffer与request的关联确认业务数据进入CXL路径；同一buffer被GGML operator或L40 kernel消费后，才能解释CXL状态对推理的影响。CUDA module、launch与Kimi comparator沿HetGPU支路提供相邻证据。

## QEMU component如何发布

```text
exact QEMU source
  + manifests/build-profile.json
  + declared hetGPU gitlink
  + fixed toolchain
      │
      v
scripts/build_component.sh
  → configure
  → qemu-system-x86_64
  → 必需firmware与file manifest
      │
      v
scripts/publish_component.sh
  → immutable OCI digest
      │
      v
cxl-lab candidate → fresh-pull → promote
      │
      v
current run manifest消费新QEMU digest
```

[`manifests/build-profile.json`](manifests/build-profile.json)定义configure与输出；[`manifests/artifact-contract.json`](manifests/artifact-contract.json)定义payload；[`manifests/artifacts/qemu-cxl-type2.json`](manifests/artifacts/qemu-cxl-type2.json)保存当前已晋升制品。会变化的commit与digest由这些manifest持续维护。

## 常用源码入口

- [`hw/cxl/cxl_type2.c`](hw/cxl/cxl_type2.c)：device realize、BAR、CXLMemSim、BAR2命令、case状态和coherency。
- [`hw/cxl/cxl_hetgpu.c`](hw/cxl/cxl_hetgpu.c)：Concordia/NVIDIA动态加载与CUDA操作包装。
- [`include/hw/cxl/cxl_type2.h`](include/hw/cxl/cxl_type2.h)：QEMU设备状态。
- [`include/hw/cxl/cxl_type2_gpu_cmd.h`](include/hw/cxl/cxl_type2_gpu_cmd.h)：QEMU侧BAR2协议。
- [`include/hw/cxl/cxl_hetgpu.h`](include/hw/cxl/cxl_hetgpu.h)：HetGPU状态、handle和操作接口。
- [`hw/cxl/cxl_type2_coherency.c`](hw/cxl/cxl_type2_coherency.c)：BAR coherency与bias状态。

## 常用构建与检查

```bash
bash scripts/verify_source_checkout.sh <expected-source-sha>

mkdir -p build
cd build
../configure --target-list=x86_64-softmmu
ninja -j2 qemu-system-x86_64

./qemu-system-x86_64 -device help | rg 'cxl-type2'
```

正式构建使用CNB exact source和冻结toolchain，并由artifact manifest赋予制品身份。本地`build/`提供快速编译、设备枚举和调试证据。

## 怎样判断证据到了哪一层

```text
-device help出现cxl-type2
  → 设备类型已编入QEMU

QEMU打印Type-2 realized与BAR布局
  → 设备实例化成功

guest出现cache0/mem0
  → linux-cxl-type2 driver完成发现与登记

guest shim打印mapped BAR2 magic/version
  → resource2与QEMU BAR2协议建立

tiny result=1234
  → 最小CUDA计算经过shim、QEMU、HetGPU和L40返回正确值

Kimi baseline/concordia comparator pass
  → 正式same-VM correctness成立
```

每一层产生下一层所需的前置状态：realized提供可枚举设备，driver probe提供guest设备对象，BAR2映射提供CUDA transport，kernel launch与DtoH提供计算结果，Kimi输出比较给出端到端语义结论。

## 继续阅读

- [`AGENTS.md`](AGENTS.md)：本仓稳定职责、source/build和证据边界。
- [`scripts/AGENTS.md`](scripts/AGENTS.md)：component build、publish与fresh-pull入口。
- [`docs/specs/cloud-component-artifact.md`](docs/specs/cloud-component-artifact.md)：QEMU component目标shape。
- [`type2-guest README`](https://cnb.cool/gevico.online/jensen/type2-guest/-/blob/main/README.md)：kernel、shim和workload怎样组成guest。
- [`linux-cxl-type2 README`](https://cnb.cool/gevico.online/jensen/linux-cxl-type2/-/blob/main/README.md)：guest内核怎样绑定QEMU PCI function。
