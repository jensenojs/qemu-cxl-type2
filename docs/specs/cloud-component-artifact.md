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

payload只含`bin/qemu-system-x86_64`，动态库由固定consumer环境提供。

## 输入和调用链

```text
QEMU source + build profile + hetGPU@67bef296 + toolchain digest
  → configure --target-list=x86_64-softmmu
  → ninja qemu-system-x86_64
  → --version + -device help + ELF检查
  → deterministic archive + ORAS digest
  → independent fresh pull
```

source仓声明的其他十五个gitlink不属于该build profile。脚本不得用`git submodule update --recursive`扩大输入面。

## 验收边界

fresh pull前不提交正式artifact manifest。`-device help`必须含`cxl-type2`，但这只证明类型注册；真实设备realize、guest discovery和compute由cxl-lab统一runner验证。
