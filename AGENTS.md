# Project: QEMU CXL Type-2

本仓保存QEMU CXL Type-2设备模型、BAR2命令处理和hetGPU backend bridge。它负责把guest CUDA shim写入的命令解释成host侧设备行为；不拥有CXLMemSim server、Concordia backend、guest kernel、模型文件或最终run specification。

项目目标、正确性层级与实时工程入口由`/home/jensen/Projects/cxl-memsim/AGENTS.md`定义。跨组件exact source从`cxl-lab/manifests/sources.lock.json`读取。本机活跃云端组件checkout是`/home/jensen/Projects/cxl-cloud/qemu-cxl-type2/`；`/home/jensen/Projects/qemu-cxl-type2/`保留既有本地构建现场。

## Cloud Source Authority

CNB `gevico.online/jensen/qemu-cxl-type2`是source primary，GitHub `jensenojs/qemu-cxl-type2`保存同SHA公开mirror。新commit先进入CNB，再以相同SHA进入GitHub；`cxl-lab`的有效控制ref只消费已经同步到两端的exact source。

source迁移只证明公开heads/tags及其可达superproject对象，不证明`.gitmodules`中的外部仓库、QEMU build、Type-2 realization或模型正确性。当前source由本仓Git与cxl-lab source lock共同给出；迁移边界见`docs/specs/cloud-source-authority.md`，执行事实见`docs/evidence/cloud-source-migration.md`。

## Cloud Build Boundary

`manifests/build-profile.json`定义恢复的hetGPU gitlink、configure参数和build目标。正式payload保存binary和串口Type-2运行实际读取的firmware；当前gitlink、source和artifact身份分别从profile、source lock与`manifests/artifacts/qemu-cxl-type2.json`读取。

工具链缺口在`cxl-lab/.ide/Dockerfile`从固定digest派生最小增量层。组件feature、configure参数和输出shape只在本仓profile/spec中决定。source probe不允许fetch、初始化submodule、build或修改checkout。

## Type-2 Evidence Boundary

最小链路是：guest shim → BAR2 → QEMU `cxl-type2` → hetGPU/Concordia backend。`-device help`出现`cxl-type2`只证明设备已编入；realized日志只证明设备实例化；只有guest tiny结果或模型输出对齐才能证明对应计算层。

涉及BAR2寄存器、命令ID、kernel参数布局、module/global查询或async copy时，必须同时检查CXLMemSim guest header和Concordia/backend消费者。禁止通过默认CPU fallback或吞没backend错误让QEMU看似成功。

## Commands

- source probe: `bash scripts/verify_source_checkout.sh <expected-source-sha>`
- candidate local configure: `mkdir -p build && cd build && ../configure --target-list=x86_64-softmmu`
- candidate local build: `ninja -j2 qemu-system-x86_64`
- device presence: `build/qemu-system-x86_64 -device help | grep -i cxl-type2`
- CNB build/publish: event `api_trigger_component_build_publish`
- CNB fresh pull: event `api_trigger_component_fresh_pull`

完整QEMU build应在CNB exact SHA和冻结toolchain上执行。本机若为调试必须复用现有build目录并限制并行，不能把本地结果传播为CNB artifact证据。

## Boundaries

- `build/`、日志、磁盘镜像、initrd和运行目录是生成现场，不提交。
- `subprojects/hetGPU`和`roms/*`是gitlink；source迁移不初始化它们。
- 本仓不复制CXLMemSim、Concordia或kernel的build profile。
- 新CNB任务必须记录repo、branch、exact SHA、event、runner资源、toolchain digest、stage和首个失败日志。
