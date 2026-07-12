# QEMU Type-2 源码迁移证据

## 输入

- GitHub公开仓：`https://github.com/jensenojs/qemu-cxl-type2`
- CNB candidate：`https://cnb.cool/gevico.online/jensen/qemu-cxl-type2`
- 功能基线：`49b1a4e0edd7e1605975292fd62b85d2942db80b`
- 初始公开refs：`cloud-type2-tiny-baseline`、`main`、`master`，无tag
- 初始ref map SHA256：`58482b1144414ef58dd4ca294f8503e20abe54f420cb2c5f1249a4e321ca3b1b`

## 迁移过程

先把GitHub公开heads及其可达对象推入空CNB仓，再在`cloud-type2-tiny-baseline`上提交仓内`AGENTS.md`、局部spec、候选build profile和source probe。控制文件commit为`2c5c093ea6229e85e8deac6417412bd51a6b7b06`，先进入CNB，再以同SHA进入GitHub，没有force-push。

双端fresh mirror使用已有本地对象作为只读传输加速源并在clone后dissociate；远端只补充新对象。两个mirror均非shallow，`git fsck --full`返回0，最终三个ref逐字节一致，ref map SHA256均为`9688e2c2f1fb6c2354281b53a88e2a203212e3225be8b38d33db5d3867cb2bd2`。

CNB source probe：

- SN：[`cnb-s3o-1jtatr0ig`](https://cnb.cool/gevico.online/jensen/qemu-cxl-type2/-/build/logs/cnb-s3o-1jtatr0ig)
- exact commit：`2c5c093ea6229e85e8deac6417412bd51a6b7b06`
- runner：amd64、2 CPU、4 GiB
- 结果：`source_probe=pass`、`shallow=false`、`worktree_clean=true`

## 证明范围

该证据证明CNB与GitHub保存相同QEMU superproject refs及可达对象，且CNB任务能获得非shallow、clean的exact checkout。它不证明16个gitlink已恢复，不证明QEMU能构建，不证明`cxl-type2`设备realized，也不证明tiny、固定1.5B或Kimi。

包含本文件的后续commit只增加迁移说明。最终生效source commit与cutover commit由`cxl-lab`有效控制ref记录。
