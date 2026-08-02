#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    printf 'usage: %s PAYLOAD_DIR EVIDENCE_DIR\n' "$0" >&2
    exit 2
fi

readonly PAYLOAD=$(realpath "$1")
readonly EVIDENCE=$2
readonly QEMU=$PAYLOAD/bin/qemu-system-x86_64

[[ -x $QEMU ]]
mkdir -p "$EVIDENCE"
"$QEMU" --version >"$EVIDENCE/version.txt"
"$QEMU" -device help >"$EVIDENCE/device-help.txt"
grep -Fq 'cxl-type2' "$EVIDENCE/device-help.txt"
set +e
timeout 3 "$QEMU" \
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
ldd "$QEMU" >"$EVIDENCE/ldd.txt"
if grep -Fq 'not found' "$EVIDENCE/ldd.txt"; then
    printf 'error: unresolved QEMU dynamic dependency\n' >&2
    exit 1
fi
printf 'qemu_payload_verification=pass\n'
