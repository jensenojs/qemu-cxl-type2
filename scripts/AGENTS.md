# QEMU Type-2 Scripts

本文件约束`scripts/`。组件职责、source边界和候选profile见`../AGENTS.md`与`../docs/specs/`。

```text
verify_source_checkout.sh  验证superproject exact source与clean checkout
build_component.sh         只恢复声明的hetGPU gitlink并构建qemu-system-x86_64

CNB组件构建通过`CCACHE_DIR`消费本仓独立的编译缓存卷，QEMU configure接收由`ccache`包装的C/C++编译器并保存构建前后统计。build目录仍然每次删除重建；cache只减少重复编译，不改变source、gitlink、configure参数或artifact identity。
component_artifact.py      生成/验证通用组件manifest与安全归档
publish_component.sh       确定性归档并发布到本仓registry
verify_component_payload.sh 唯一拥有version、device、paused boot与ELF检查
pull_component.sh          显式接收runtime与不存在的work目录，只按digest恢复，并在artifact contract选择的exact toolchain image中调用同一payload verifier
```

build脚本只允许初始化`subprojects/hetGPU`的声明commit。十五个ROM/test gitlinks不得被隐式初始化。publish/pull是仅有registry边界，认证来自CNB任务。payload保存QEMU binary，以及当前串口Type-2运行实际读取的`bios-256k.bin`、`kvmvapic.bin`和`linuxboot_dma.bin`；不捆绑系统动态库、默认VGA ROM、NIC ROM或整个`pc-bios/`。

build和fresh pull都必须用显式`-L payload/share/qemu -nic none -vga none`把QEMU启动到`-S`暂停边界。退出码124来自外层`timeout`，表示QEMU保持运行直至被测试终止；其他退出码直接暴露缺失firmware或启动错误。消费端同样必须显式传`-L`，不得搜索宿主机QEMU数据目录作为fallback。

publish任务输出唯一`CNB_OUTPUT component-candidate` JSON封装，随后在同一个build task中按digest调用`pull_component.sh`恢复和验证；只有两步都通过才输出`component_build=pass`与`component_fresh_pull=pass`。cxl-lab制品工具验证source、contract、profile和这两个task事实后接受正式`qemu-cxl-type2.json`。历史fresh-pull event固定读取candidate路径，只用于显式旧build迁移或诊断。脚本不直接修改Git manifest。

build和fresh pull证明QEMU文件可生成与恢复；`-device help`只证明设备类型注册。它们不证明`cxl-type2` realized、guest节点、tiny、固定1.5B、Kimi或性能。
