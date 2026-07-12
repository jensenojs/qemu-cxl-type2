# QEMU Type-2 Scripts

本文件约束`scripts/`。组件职责、source边界和候选profile见`../AGENTS.md`与`../docs/specs/`。

```text
verify_source_checkout.sh  验证superproject exact source与clean checkout
build_component.sh         只恢复声明的hetGPU gitlink并构建qemu-system-x86_64
component_artifact.py      生成/验证通用组件manifest与安全归档
publish_component.sh       确定性归档并发布到本仓registry
pull_component.sh          只按digest恢复并重复version/device/ELF检查
```

build脚本只允许初始化`subprojects/hetGPU`的声明commit。十五个ROM/test gitlinks不得被隐式初始化。publish/pull是仅有registry边界，认证来自CNB任务。payload只含QEMU binary，不捆绑系统动态库。

build和fresh pull证明QEMU文件可生成与恢复；`-device help`只证明设备类型注册。它们不证明`cxl-type2` realized、guest节点、tiny、固定1.5B、Kimi或性能。
