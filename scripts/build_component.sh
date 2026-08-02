#!/usr/bin/env bash
set -euo pipefail

readonly ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
readonly PROFILE=$ROOT/manifests/build-profile.json
readonly WORK=$ROOT/.work/component
readonly BUILD=$WORK/build
readonly PAYLOAD=$WORK/payload
: "${CCACHE_DIR:?CCACHE_DIR must name the CNB compiler-cache volume}"
export CCACHE_BASEDIR=$ROOT

[[ -z $(git -C "$ROOT" status --porcelain=v1 --untracked-files=all) ]] || {
    echo 'error: source checkout is dirty' >&2
    exit 1
}

readarray -t profile < <(python3 - "$PROFILE" <<'PY'
import json, sys
p=json.load(open(sys.argv[1]))
print(p["required_gitlinks"]["subprojects/hetGPU"]["commit"])
print(p["parallelism"])
for value in p["configure_args"]:
    print(value)
PY
)
readonly HETGPU_COMMIT=${profile[0]}
readonly PARALLEL=${profile[1]}
readonly INDEX_HETGPU=$(git -C "$ROOT" ls-files -s subprojects/hetGPU | awk '{print $2}')
[[ $INDEX_HETGPU == "$HETGPU_COMMIT" ]]

git -C "$ROOT" submodule update --init --depth 1 subprojects/hetGPU
[[ $(git -C "$ROOT/subprojects/hetGPU" rev-parse HEAD) == "$HETGPU_COMMIT" ]]

[[ $WORK == "$ROOT/.work/component" ]]
rm -rf "$WORK"
mkdir -p "$BUILD" "$PAYLOAD/bin" "$PAYLOAD/libexec/qemu" "$PAYLOAD/share/qemu" "$PAYLOAD/evidence/cuda-api" "$WORK/evidence"

python3 "$ROOT/tests/test_component_artifact.py"
ccache --show-stats | tee "$WORK/evidence/ccache-before.txt"
(
    cd "$BUILD"
    CC='ccache gcc' CXX='ccache g++' "$ROOT/configure" "${profile[@]:2}"
)
ninja -C "$BUILD" -j"$PARALLEL" qemu-system-x86_64 contrib-plugins
ccache --show-stats | tee "$WORK/evidence/ccache-after.txt"

readonly QEMU=$BUILD/qemu-system-x86_64
[[ -x $QEMU ]]
install -m 0755 "$QEMU" "$PAYLOAD/bin/qemu-system-x86_64"
install -m 0755 "$BUILD/contrib/plugins/libhotblocks.so" \
    "$PAYLOAD/libexec/qemu/libhotblocks.so"
install -m 0644 "$ROOT/hw/cxl/cxl_type2.c" "$PAYLOAD/evidence/cuda-api/cxl_type2.c"
install -m 0644 "$ROOT/hw/cxl/cxl_hetgpu.c" "$PAYLOAD/evidence/cuda-api/cxl_hetgpu.c"
install -m 0644 "$ROOT/include/hw/cxl/cxl_type2_gpu_cmd.h" \
    "$PAYLOAD/evidence/cuda-api/cxl_type2_gpu_cmd.h"
for firmware in bios-256k.bin kvmvapic.bin linuxboot_dma.bin; do
    install -m 0644 "$ROOT/pc-bios/$firmware" "$PAYLOAD/share/qemu/$firmware"
done
bash "$ROOT/scripts/verify_component_payload.sh" "$PAYLOAD" "$WORK/evidence"
printf 'hetgpu_commit=%s\n' "$HETGPU_COMMIT" | tee "$WORK/evidence/gitlinks.txt"

printf 'component_build=pass\npayload=%s\n' "$PAYLOAD"
