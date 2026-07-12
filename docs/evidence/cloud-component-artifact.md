# QEMU Type-2组件制品证据

## 证据链

组件构建与发布任务[`cnb-b6o-1jtb38o53`](https://cnb.cool/gevico.online/jensen/qemu-cxl-type2/-/build/logs/cnb-b6o-1jtb38o53)消费source commit `8b072c4e29e6f5660e8504acd7a518720ffff833`和固定toolchain digest `sha256:8732c435c97964f2ba95b42fbbcaf79b3c23feffc3b1f7683531917122f4f59e`。任务以4 CPU、8 GiB运行，稀疏恢复声明的`subprojects/hetGPU` gitlink，完成`qemu-system-x86_64`构建、版本检查、`-device help`中的`cxl-type2`检查和动态依赖记录。

发布结果：

```text
repository=docker.cnb.cool/gevico.online/jensen/qemu-cxl-type2
digest=sha256:1214638923ebfd9e15348dfb1a35b3cfda2dc84f81f667b4e52c2178df9d6f7a
archive_sha256=997cdd9dea069c27d56f6b758ba46563d864e43bb1439d62898749df18ef5858
manifest_sha256=bc238e6034bc6a575387986ce10ad048877d2a2e50ffc43666a1ed97e1af6814
profile_sha256=1a0536adb54c3b64dc00187a2b78b91512f73261f99e25dd813dc9c9c864583e
```

独立任务[`cnb-j3o-1jtb3sa25`](https://cnb.cool/gevico.online/jensen/qemu-cxl-type2/-/build/logs/cnb-j3o-1jtb3sa25)以2 CPU、4 GiB运行，只按上述digest拉取制品。它重复执行合同负向测试、双blob hash、安全恢复、QEMU版本、动态依赖和`cxl-type2`注册检查，输出`component_fresh_pull=pass`。

## 证明边界

该证据证明QEMU二进制可从固定源码生成，并能从新CNB任务按不可变digest恢复。`-device help`只证明设备类型已经注册。它不证明设备realize、guest节点、tiny计算、固定1.5B、Kimi或性能。

## 后续组合暴露的边界

CNB组合任务[`cnb-qij-1jtb6g3nl`](https://cnb.cool/gevico.online/jensen/cxl-lab/-/build/logs/cnb-qij-1jtb6g3nl)按上述digest恢复QEMU后，在guest启动前报`could not load PC BIOS 'bios-256k.bin'`。这说明单独恢复可执行文件不足以形成可启动的system-emulation输入。该digest仍证明旧单文件制品可恢复，但不能继续作为Type-2运行输入；替代制品必须加入spec冻结的三个QEMU自有firmware并重新build、发布和fresh pull。

## 最小firmware替代制品

任务[`cnb-qp8-1jtb7f73m`](https://cnb.cool/gevico.online/jensen/qemu-cxl-type2/-/build/logs/cnb-qp8-1jtb7f73m)消费source commit `4a925a3c11d8810e0bc90832c3cec27c49b073db`，重新构建binary，并把三个QEMU自有firmware加入payload。build阶段用`-L payload/share/qemu -nic none -vga none`启动到`-S`暂停边界；外层timeout终止仍在运行的QEMU。

```text
repository=docker.cnb.cool/gevico.online/jensen/qemu-cxl-type2
digest=sha256:fcef654e02121d9a6c83ab7a9b7edd7b3dbe0655eb49935964e7dffd325e8266
archive_sha256=503a2da7c68c65f68aee63e927413b2c9b0d2c7f9cd5abe381629cc9efb39385
manifest_sha256=365fe9fa9d023f23c9dbbd7e94c90c522c0f15afa83ead17adf6310d47c7beb6
profile_sha256=b1ad5e7891aff13c38422304a2d4bad2f0dbbfd6db9c8aeff18a3fbfd9d72f5c
```

独立任务[`cnb-42t-1jtb7qeap`](https://cnb.cool/gevico.online/jensen/qemu-cxl-type2/-/build/logs/cnb-42t-1jtb7qeap)只按该digest恢复，完成8项合同测试、双blob校验、安全解包、版本与设备注册检查，并再次使用恢复后的firmware启动到暂停边界，输出`component_fresh_pull=pass`。该digest替换旧单文件digest成为正式QEMU运行输入；真实Type-2 realize与guest证据仍由cxl-lab组合任务产生。
