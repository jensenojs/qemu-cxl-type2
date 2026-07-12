# Project: QEMU CXL Type-2

本仓保存QEMU CXL Type-2设备模型、BAR2命令处理和hetGPU backend bridge。它负责把guest CUDA shim写入的命令解释成host侧设备行为；不拥有CXLMemSim server、Concordia backend、guest kernel、模型文件或最终run specification。

项目目标、正确性层级与当前工程入口由`/home/jensen/Projects/cxl-memsim/AGENTS.md`定义。CNB独立checkout从`gevico.online/jensen/cxl-lab`的有效控制ref `refs/heads/fixed-1p5b-control`读取exact source lock。本机活跃云端组件checkout是`/home/jensen/Projects/cxl-cloud/qemu-cxl-type2/`；`/home/jensen/Projects/qemu-cxl-type2/`保留既有本地构建现场。

## Cloud Source Authority

在`cxl-lab` source lock切换前，CNB `gevico.online/jensen/qemu-cxl-type2`只是candidate，GitHub `jensenojs/qemu-cxl-type2`仍是primary。迁移commit必须先进入CNB，再以相同SHA进入GitHub；双端fresh mirror验证通过后，`cxl-lab`单个cutover commit才使CNB成为primary。

当前功能基线是`49b1a4e0edd7e1605975292fd62b85d2942db80b`。source迁移只证明公开heads/tags及其可达superproject对象，不证明`.gitmodules`中的16个外部仓库、QEMU build、Type-2 realization或模型正确性。局部边界见`docs/specs/cloud-source-authority.md`，执行证据见`docs/evidence/cloud-source-migration.md`。

## Cloud Build Boundary

`manifests/build-profile.json`只保存当前已验证本地入口的候选形状：`configure --target-list=x86_64-softmmu`和`ninja qemu-system-x86_64`。真实CNB build前必须确认`subprojects/hetGPU@67bef2966eed98a4e4cb9634f6310ed0b46d03ed`及其他实际依赖；不得把候选profile称为已验证artifact合同。

工具链缺口在`cxl-lab/.ide/Dockerfile`从固定digest派生最小增量层。组件feature、configure参数和输出shape只在本仓profile/spec中决定。source probe不允许fetch、初始化submodule、build或修改checkout。

## Type-2 Evidence Boundary

最小链路是：guest shim → BAR2 → QEMU `cxl-type2` → hetGPU/Concordia backend。`-device help`出现`cxl-type2`只证明设备已编入；realized日志只证明设备实例化；只有guest tiny结果或模型输出对齐才能证明对应计算层。

涉及BAR2寄存器、命令ID、kernel参数布局、module/global查询或async copy时，必须同时检查CXLMemSim guest header和Concordia/backend消费者。禁止通过默认CPU fallback或吞没backend错误让QEMU看似成功。

## Commands

- source probe: `bash scripts/verify_source_checkout.sh 49b1a4e0edd7e1605975292fd62b85d2942db80b`
- candidate local configure: `mkdir -p build && cd build && ../configure --target-list=x86_64-softmmu`
- candidate local build: `ninja -j2 qemu-system-x86_64`
- device presence: `build/qemu-system-x86_64 -device help | grep -i cxl-type2`

完整QEMU build应在CNB exact SHA和冻结toolchain上执行。本机若为调试必须复用现有build目录并限制并行，不能把本地结果传播为CNB artifact证据。

## Boundaries

- `build/`、日志、磁盘镜像、initrd和运行目录是生成现场，不提交。
- `subprojects/hetGPU`和`roms/*`是gitlink；source迁移不初始化它们。
- 本仓不复制CXLMemSim、Concordia或kernel的build profile。
- 新CNB任务必须记录repo、branch、exact SHA、event、runner资源、toolchain digest、stage和首个失败日志。
