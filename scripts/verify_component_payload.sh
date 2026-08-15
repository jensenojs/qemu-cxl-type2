#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    printf 'usage: %s PAYLOAD_DIR EVIDENCE_DIR\n' "$0" >&2
    exit 2
fi

readonly PAYLOAD=$(realpath "$1")
readonly EVIDENCE=$2
readonly QEMU=$PAYLOAD/bin/qemu-system-x86_64
readonly APERTURE_GATE=$PAYLOAD/bin/run_cxl_actual_aperture_gate.sh
readonly RUNTIME=$PAYLOAD/lib

[[ -x $QEMU ]]
[[ -x $APERTURE_GATE ]]
[[ -f $RUNTIME/libcapstone.so.4 && -f $RUNTIME/libaio.so.1t64 ]]
mkdir -p "$EVIDENCE"
bash -n "$APERTURE_GATE"
"$APERTURE_GATE" --help >"$EVIDENCE/actual-aperture-gate-help.txt"
LD_LIBRARY_PATH=$RUNTIME "$QEMU" --version >"$EVIDENCE/version.txt"
LD_LIBRARY_PATH=$RUNTIME "$QEMU" -device help >"$EVIDENCE/device-help.txt"
grep -Fq 'cxl-type2' "$EVIDENCE/device-help.txt"
set +e
timeout 3 env LD_LIBRARY_PATH="$RUNTIME" "$QEMU" \
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
    >"$EVIDENCE/paused-boot.stdout" \
    2>"$EVIDENCE/paused-boot.stderr"
readonly paused_boot_rc=$?
set -e
[[ $paused_boot_rc -eq 124 ]]
printf 'paused_boot=pass\n' >"$EVIDENCE/paused-boot.txt"
file "$QEMU" >"$EVIDENCE/file.txt"
readelf -d "$QEMU" >"$EVIDENCE/readelf-dynamic.txt"
LD_LIBRARY_PATH=$RUNTIME ldd "$QEMU" >"$EVIDENCE/ldd.txt"
if grep -Fq 'not found' "$EVIDENCE/ldd.txt"; then
    printf 'error: unresolved QEMU dynamic dependency\n' >&2
    exit 1
fi
printf 'qemu_payload_verification=pass\n'
