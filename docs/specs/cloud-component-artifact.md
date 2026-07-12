# QEMU Type-2组件制品

## 目标

在CNB exact source与固定工具链中，只恢复Type-2 build需要的hetGPU gitlink，构建`qemu-system-x86_64`，发布本仓OCI并独立fresh pull。该BR2切片不启动guest。

## 预期结构

```text
qemu-cxl-type2/
├── manifests/
│   ├── build-profile.json
│   ├── artifact-contract.json
│   └── artifacts/qemu-cxl-type2.json
├── scripts/
│   ├── AGENTS.md
│   ├── build_component.sh
│   ├── component_artifact.py
│   ├── publish_component.sh
│   └── pull_component.sh
└── tests/test_component_artifact.py
```

payload包含组件自身可执行文件和当前串口Type-2运行实际读取的最小firmware集合：

```text
payload/
├── bin/qemu-system-x86_64
└── share/qemu/
    ├── bios-256k.bin
    ├── kvmvapic.bin
    └── linuxboot_dma.bin
```

动态库由固定consumer环境提供。运行入口使用`-nic none -vga none`，所以默认VGA和NIC ROM不进入制品；不捆绑整个`pc-bios/`。

## 输入和调用链

```text
QEMU source + build profile + hetGPU@67bef296 + toolchain digest
  → configure --target-list=x86_64-softmmu
  → ninja qemu-system-x86_64
  → --version + -device help + ELF检查
  → -L payload/share/qemu + -nic none + -vga none启动到-S暂停边界
  → deterministic archive + ORAS digest
  → independent fresh pull
```

source仓声明的其他十五个gitlink不属于该build profile。脚本不得用`git submodule update --recursive`扩大输入面。

## 验收边界

fresh pull前不提交正式artifact manifest。`-device help`必须含`cxl-type2`；使用payload内firmware启动到`-S`暂停边界，证明当前最小QEMU运行数据完整。两者都不证明设备realize、guest discovery或compute；这些由cxl-lab统一runner验证。

CNB组合任务[`cnb-qij-1jtb6g3nl`](https://cnb.cool/gevico.online/jensen/cxl-lab/-/build/logs/cnb-qij-1jtb6g3nl)已经证明旧单文件制品会在guest启动前报`could not load PC BIOS 'bios-256k.bin'`。本地只读`strace`在`-nic none -vga none`形状下确认只读取上述三个firmware。本次扩展替换旧单文件payload，不增加运行时搜索或fallback。
