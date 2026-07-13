---
status: completed
---

# QEMU CXL Type-2 云端源码权威

## 目标与证据层

BR3。本切片把QEMU Type-2 superproject的公开历史迁入CNB组件仓，并建立独立checkout可理解的source probe和候选build输入。它服务后续QEMU artifact、TCG discovery、tiny、固定1.5B与Kimi路线，但只关闭源码权威。

```text
GitHub primary
  → CNB candidate保存声明refs
  → migration commit同时进入CNB与GitHub
  → 双端fresh mirror/ref map/fsck
  → CNB exact SHA source probe
  → cxl-lab单commit cutover
  → CNB primary / GitHub mirror
```

## 目标结构

```text
qemu-cxl-type2/
├── AGENTS.md
├── docs/specs/cloud-source-authority.md
├── manifests/build-profile.json
├── scripts/verify_source_checkout.sh
└── .cnb.yml
```

公开ref集合在迁移开始时为三个heads：`cloud-type2-tiny-baseline`、`main`、`master`，无tag；规范化ref map SHA256为`58482b1144414ef58dd4ca294f8503e20abe54f420cb2c5f1249a4e321ca3b1b`。功能基线`49b1a4e0edd7e1605975292fd62b85d2942db80b`必须是migration commit祖先。

## 边界

“完整历史”只指上述superproject refs及其可达对象。`.gitmodules`中的16个gitlink不在本次证明范围。候选profile记录当前本地已成立的QEMU target形状；真实CNB build前必须另行冻结所需submodule、toolchain和artifact文件图。

source probe拒绝shallow和dirty checkout，不fetch、不初始化submodule、不build。仓库迁移证据不得关闭QEMU build、Type-2 realized、guest节点、tiny、固定1.5B、Kimi或tps。

## 验收与回滚

- CNB与GitHub heads/tags ref map逐字节一致，fresh mirrors非shallow，`git fsck --full`通过。
- migration commit包含本结构，功能基线可达，CNB exact SHA probe输出`source_probe=pass`。
- `cxl-lab`单个cutover commit更新QEMU URL/commit；cutover前GitHub仍是primary。
- 回滚恢复上一个`cxl-lab` source-lock commit；不改写或force-push两端历史。
