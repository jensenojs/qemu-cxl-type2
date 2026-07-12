#!/usr/bin/env bash
set -euo pipefail

readonly ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
readonly PROFILE=$ROOT/manifests/build-profile.json
readonly WORK=$ROOT/.work/component
readonly BUILD=$WORK/build
readonly PAYLOAD=$WORK/payload

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
mkdir -p "$BUILD" "$PAYLOAD/bin" "$PAYLOAD/share/qemu" "$WORK/evidence"

python3 "$ROOT/tests/test_component_artifact.py"
(
    cd "$BUILD"
    "$ROOT/configure" "${profile[@]:2}"
)
ninja -C "$BUILD" -j"$PARALLEL" qemu-system-x86_64

readonly QEMU=$BUILD/qemu-system-x86_64
[[ -x $QEMU ]]
install -m 0755 "$QEMU" "$PAYLOAD/bin/qemu-system-x86_64"
for firmware in bios-256k.bin kvmvapic.bin linuxboot_dma.bin; do
    install -m 0644 "$ROOT/pc-bios/$firmware" "$PAYLOAD/share/qemu/$firmware"
done
"$PAYLOAD/bin/qemu-system-x86_64" --version | tee "$WORK/evidence/version.txt"
"$PAYLOAD/bin/qemu-system-x86_64" -device help | tee "$WORK/evidence/device-help.txt"
grep -Fq 'cxl-type2' "$WORK/evidence/device-help.txt"
set +e
timeout 3 "$PAYLOAD/bin/qemu-system-x86_64" \
    -accel tcg,thread=multi \
    -cpu max \
    -S \
    -display none \
    -monitor none \
    -serial none \
    -nic none \
    -vga none \
    -m 256M \
    -L "$PAYLOAD/share/qemu" \
    >"$WORK/evidence/paused-boot.stdout" \
    2>"$WORK/evidence/paused-boot.stderr"
paused_boot_rc=$?
set -e
[[ $paused_boot_rc -eq 124 ]]
printf 'paused_boot=pass\n' | tee "$WORK/evidence/paused-boot.txt"
file "$PAYLOAD/bin/qemu-system-x86_64" | tee "$WORK/evidence/file.txt"
readelf -d "$PAYLOAD/bin/qemu-system-x86_64" | tee "$WORK/evidence/readelf-dynamic.txt"
ldd "$PAYLOAD/bin/qemu-system-x86_64" | tee "$WORK/evidence/ldd.txt"
printf 'hetgpu_commit=%s\n' "$HETGPU_COMMIT" | tee "$WORK/evidence/gitlinks.txt"

printf 'component_build=pass\npayload=%s\n' "$PAYLOAD"
