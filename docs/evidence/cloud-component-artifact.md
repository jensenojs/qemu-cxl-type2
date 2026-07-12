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
