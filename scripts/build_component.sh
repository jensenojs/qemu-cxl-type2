#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'usage: %s --profile PROFILE --build-root EXISTING_PATH --work-dir ABSENT_PATH --cache-dir EXISTING_PATH --payload-dir ABSENT_PATH\n' "$0"
}

if [[ ${1:-} == --help ]]; then usage; exit 0; fi
if [[ ${1:-} == --hint ]]; then
    printf 'agent_hint=self=scripts/build_component.sh\nagent_hint=boundary=builds and verifies one QEMU payload; persistent native build state is caller-owned\n'
    exit 0
fi

profile_arg= build_root_arg= work_arg= cache_arg= payload_arg=
while (( $# )); do
    case "$1" in
        --profile|--build-root|--work-dir|--cache-dir|--payload-dir)
            (( $# >= 2 )) || { usage >&2; exit 2; }
            case "$1" in
                --profile) profile_arg=$2 ;;
                --build-root) build_root_arg=$2 ;;
                --work-dir) work_arg=$2 ;;
                --cache-dir) cache_arg=$2 ;;
                --payload-dir) payload_arg=$2 ;;
            esac
            shift 2 ;;
        *) usage >&2; exit 2 ;;
    esac
done
[[ -n $profile_arg && -n $build_root_arg && -n $work_arg && -n $cache_arg && -n $payload_arg ]] || { usage >&2; exit 2; }
[[ $profile_arg == /* && $build_root_arg == /* && $work_arg == /* && $cache_arg == /* && $payload_arg == /* ]] || {
    printf 'paths must be absolute\n' >&2
    exit 2
}

readonly ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
readonly PROFILE=$(realpath "$profile_arg")
[[ $PROFILE == "$ROOT/manifests/build-profile.json" ]] || { printf 'profile does not match the component contract\n' >&2; exit 2; }
[[ -d $build_root_arg && ! -L $build_root_arg ]] || { printf 'build root must be an existing non-symlink directory\n' >&2; exit 2; }
[[ -d $cache_arg && ! -L $cache_arg ]] || { printf 'cache directory must be an existing non-symlink directory\n' >&2; exit 2; }
for path in "$work_arg" "$payload_arg"; do
    [[ ! -e $path && ! -L $path ]] || { printf 'output path must not exist: %s\n' "$path" >&2; exit 2; }
    [[ -d $(dirname "$path") && ! -L $(dirname "$path") ]] || { printf 'output parent must be an existing non-symlink directory: %s\n' "$path" >&2; exit 2; }
done
readonly BUILD_ROOT=$(realpath "$build_root_arg")
readonly WORK=$(realpath -m "$work_arg")
readonly BUILD=$BUILD_ROOT/build
readonly PAYLOAD=$(realpath -m "$payload_arg")
readonly CCACHE_DIR=$(realpath "$cache_arg")
case "$ROOT/" in "$BUILD_ROOT/"*|"$WORK/"*|"$PAYLOAD/"*) printf 'output path cannot be an ancestor of the repository\n' >&2; exit 2;; esac
[[ $BUILD_ROOT != "$ROOT" && $WORK != "$ROOT" && $PAYLOAD != "$ROOT" ]] || { printf 'output path cannot equal repository root\n' >&2; exit 2; }
export CCACHE_DIR
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
[[ -e $ROOT/subprojects/hetGPU/.git ]] || { printf 'required hetGPU gitlink is not materialized\n' >&2; exit 1; }
[[ $(git -C "$ROOT/subprojects/hetGPU" rev-parse HEAD) == "$HETGPU_COMMIT" ]] || {
    printf 'materialized hetGPU commit does not match build profile\n' >&2
    exit 1
}

mkdir "$WORK" "$PAYLOAD"
mkdir -p "$BUILD" "$PAYLOAD/bin" "$PAYLOAD/lib" "$PAYLOAD/libexec/qemu" "$PAYLOAD/share/qemu" "$PAYLOAD/evidence/cuda-api" "$WORK/evidence"

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
for library in libcapstone.so.4 libaio.so.1t64; do
    library_path=$(ldconfig -p | awk -v name="$library" '$1 == name { print $NF; exit }')
    [[ -n $library_path && -f $library_path ]] || {
        printf 'error: required QEMU runtime library is unavailable: %s\n' "$library" >&2
        exit 1
    }
    install -m 0644 "$library_path" "$PAYLOAD/lib/$library"
done
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
