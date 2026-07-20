# QEMU CXL Type-2 Device Model

这个仓库实现QEMU Camp运行时的host侧CXL Type-2 PCI设备。guest看到的PCI function、BAR0组件寄存器、BAR2命令与cache窗口、BAR4 device-attached memory，以及CUDA命令进入Concordia/HetGPU和CXLMemSim的分流，都在这里建立。

如果把整个工程看成一次请求从guest走到L40再返回，QEMU位于最关键的协议边界：它接收guest shim写入的BAR2状态，把无类型的MMIO寄存器和data window还原成CUDA/内存操作，调用host backend，再把返回码与结果写回guest可见寄存器。

## 先看全图

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

`type2-guest`不编译QEMU。它只生成QEMU要启动的guest文件。`cxl-lab`不实现设备行为。它选择exact QEMU component并构造一轮可重放运行。

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
          → CXLMemSim TCP或共享内存transport
          → backing store与访问统计

结果返回：
L40或CXLMemSim结果
  → QEMU cmd_result/results/data window
  → BAR2
  → guest shim还原CUresult与输出参数
  → CUDA runtime / GGML / llama
```

CXLMemSim与Concordia位于QEMU下游的两条相交支路。CUDA allocation、module、function、launch和synchronize主要进入HetGPU；CXL memory access、cache与backing-store请求进入CXLMemSim。copy路径还会更新QEMU shadow memory并产生coherency事件，所以不能把运行时画成“先CXLMemSim，再Concordia”的单向串联。

## 与其他仓库的责任交界

| 仓库 | 它负责什么 | QEMU从它取得什么 |
| --- | --- | --- |
| [`cxl-lab`](https://cnb.cool/gevico.online/jensen/cxl-lab) | source/artifact/run/result控制、CNB runner、evidence | QEMU exact digest、启动参数、CXLMemSim地址、Concordia绝对路径和guest文件 |
| [`type2-guest`](https://cnb.cool/gevico.online/jensen/type2-guest) | 组装kernel、initramfs、shim、workload、CUDA userland | `bzImage`与`initrd.cpio.gz` |
| [`linux-cxl-type2`](https://cnb.cool/gevico.online/jensen/linux-cxl-type2) | guest `cxl_type2_accel` driver、CXL.cache/mem对象、HDM/DPA | QEMU暴露的PCI function由这个driver在guest中绑定 |
| [`cxlmemsim`](https://cnb.cool/gevico.online/jensen/cxlmemsim) | CXLMemSim server、guest libcuda shim、BAR2共享协议 | QEMU连接server；guest shim与QEMU必须解释同一组寄存器和命令ID |
| [`concordia`](https://cnb.cool/gevico.online/jensen/concordia) | `libnvcuda.so`、Kimi case管理、HetGPU/NVIDIA执行 | QEMU通过`hetgpu-lib=<absolute-path>`加载它 |
| [`cxl-models`](https://cnb.cool/gevico.online/jensen/cxl-models) | Kimi模型身份与LFS对象 | runner把模型只读提供给guest，QEMU不保存模型副本 |

跨仓目标和correctness阶梯见[`cxl-memsim/AGENTS.md`](https://cnb.cool/gevico.online/jensen/cxl-memsim/-/blob/main/AGENTS.md)。QEMU component怎样进入正式run见[`cxl-lab/manifests/AGENTS.md`](https://cnb.cool/gevico.online/jensen/cxl-lab/-/blob/main/manifests/AGENTS.md)。

## BAR承担什么职责

### BAR0：CXL component registers

BAR0保存CXL component register block。guest `cxl_type2_accel`用它发现CXL能力、HDM decoder和相关寄存器。可选tmatmul硬件控制面也约定在BAR0的独立CSR窗口出现；当前Kimi BAR2 CUDA链不依赖该miscdevice。

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

### BAR4：device-attached memory与bulk window

QEMU把`device_mem`注册为BAR4。它承载Type-2 device-attached memory、bulk copy的中间窗口、coherent pool、dynamic-capacity和部分shadow/coherency状态。

真实NVIDIA allocation由HetGPU/NVIDIA Driver拥有；QEMU的BAR4/device_mem还承担guest可见CXL memory与coherency模型。两者通过copy、shadow更新和bar coherency事件关联，地址身份不能凭数值相同就互换。

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
  当前公开driver、QEMU、shim与运行输入没有创建者或UAPI
```

QEMU的`id=`只是对象名。`-device cxl-type2,id=cxl-gpu0`不会让Linux自动创建`/dev/cxl_gpu0`。guest当前由`cxl_type2_accel`创建`/dev/cxl/cache0`和`/dev/cxl/mem0`；guest shim使用PCI `resource2`访问BAR2。

导师已经确认driver仓是[`vickiegpt/linux-cxl-type2`](https://github.com/vickiegpt/linux-cxl-type2/tree/main)。该来源关闭了driver身份问题，仍没有解释`/dev/cxl_gpu0`的创建规则与ABI。完整调查见[`cxl-gpu-device-node-contract.md`](https://cnb.cool/gevico.online/jensen/cxl-lab/-/blob/main/docs/evidence/cxl-gpu-device-node-contract.md)。

## HetGPU与Concordia怎样接入

QEMU属性`hetgpu-lib`接收一个绝对路径。realize阶段初始化`HetGPUState`并加载该动态库。当前正式运行使用Concordia component里的`libnvcuda.so`。

```text
QEMU cxl_type2
  → cxl_hetgpu wrapper
  → dlopen(hetgpu-lib)
  → cuInit / context / allocation / module / function / launch / sync
  → NVIDIA Driver
```

QEMU还维护module和function handle表、paired case epoch、command sequence与返回码。guest拿到的是QEMU分配的稳定ID，不是可在guest直接解引用的host pointer。core或日志中的handle问题需要同时检查guest编码、BAR2参数、QEMU表和Concordia/NVIDIA返回值。

## CXLMemSim怎样接入

属性`cxlmemsim-addr`和`cxlmemsim-port`指定server。默认路径使用TCP；环境声明共享内存transport时走对应模式。QEMU把CXL read/write/fence等请求连同地址、大小、时间戳和数据发送给server，并消费响应状态。

CXLMemSim连接成功只证明QEMU与server控制面可达。它不证明CUDA module加载、kernel执行或Kimi correctness。相反，L40 kernel成功也不能单独证明CXL backing-store语义正确；两类证据必须按各自调用链读取。

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

[`manifests/build-profile.json`](manifests/build-profile.json)定义configure与输出；[`manifests/artifact-contract.json`](manifests/artifact-contract.json)定义payload；[`manifests/artifacts/qemu-cxl-type2.json`](manifests/artifacts/qemu-cxl-type2.json)保存当前已晋升制品。README不复制会变化的commit或digest。

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

正式构建应使用CNB exact source和冻结toolchain。本地`build/`用于快速编译与调试，不提供正式artifact身份。

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

某一层通过只能关闭这一层的未知量。QEMU realized不能代替driver probe，BAR2映射不能代替kernel launch，kernel launch也不能代替Kimi输出比较。

## 继续阅读

- [`AGENTS.md`](AGENTS.md)：本仓稳定职责、source/build和证据边界。
- [`scripts/AGENTS.md`](scripts/AGENTS.md)：component build、publish与fresh-pull入口。
- [`docs/specs/cloud-component-artifact.md`](docs/specs/cloud-component-artifact.md)：QEMU component目标shape。
- [`type2-guest README`](https://cnb.cool/gevico.online/jensen/type2-guest/-/blob/main/README.md)：kernel、shim和workload怎样组成guest。
- [`linux-cxl-type2 README`](https://cnb.cool/gevico.online/jensen/linux-cxl-type2/-/blob/main/README.md)：guest内核怎样绑定QEMU PCI function。
