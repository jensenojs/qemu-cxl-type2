# QEMU Type-2 Scripts

本文件约束`scripts/`。组件职责、source边界和候选profile见`../AGENTS.md`与`../docs/specs/`。

```text
verify_source_checkout.sh  验证superproject exact source与clean checkout
build_component.sh         只恢复声明的hetGPU gitlink并构建qemu-system-x86_64
component_artifact.py      生成/验证通用组件manifest与安全归档
publish_component.sh       确定性归档并发布到本仓registry
pull_component.sh          只按digest恢复并重复version/device/ELF检查
```

build脚本只允许初始化`subprojects/hetGPU`的声明commit。十五个ROM/test gitlinks不得被隐式初始化。publish/pull是仅有registry边界，认证来自CNB任务。payload保存QEMU binary，以及当前串口Type-2运行实际读取的`bios-256k.bin`、`kvmvapic.bin`和`linuxboot_dma.bin`；不捆绑系统动态库、默认VGA ROM、NIC ROM或整个`pc-bios/`。

build和fresh pull都必须用显式`-L payload/share/qemu -nic none -vga none`把QEMU启动到`-S`暂停边界。退出码124来自外层`timeout`，表示QEMU保持运行直至被测试终止；其他退出码直接暴露缺失firmware或启动错误。消费端同样必须显式传`-L`，不得搜索宿主机QEMU数据目录作为fallback。

build和fresh pull证明QEMU文件可生成与恢复；`-device help`只证明设备类型注册。它们不证明`cxl-type2` realized、guest节点、tiny、固定1.5B、Kimi或性能。
