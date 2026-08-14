#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'usage: %s --profile PROFILE --execution-root ABSENT_PATH --cache-root EXISTING_PATH --build-root EXISTING_PATH\n' "$0"
}

if [[ ${1:-} == --help ]]; then usage; exit 0; fi
if [[ ${1:-} == --hint ]]; then
    printf 'agent_hint=self=scripts/run_local_component.sh\nagent_hint=boundary=coordinates the component-owned build and package stages for one managed local execution\n'
    exit 0
fi

profile= execution_root= cache_root= build_root=
while (( $# )); do
    case "$1" in
        --profile|--execution-root|--cache-root|--build-root)
            (( $# >= 2 )) || { usage >&2; exit 2; }
            case "$1" in
                --profile) profile=$2 ;;
                --execution-root) execution_root=$2 ;;
                --cache-root) cache_root=$2 ;;
                --build-root) build_root=$2 ;;
            esac
            shift 2 ;;
        *) usage >&2; exit 2 ;;
    esac
done
[[ -n $profile && -n $execution_root && -n $cache_root && -n $build_root ]] || { usage >&2; exit 2; }
[[ $profile == /* && $execution_root == /* && $cache_root == /* && $build_root == /* ]] || { printf 'paths must be absolute\n' >&2; exit 2; }

readonly ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
[[ $(realpath "$profile") == "$ROOT/manifests/build-profile.json" ]] || { printf 'profile does not match the component contract\n' >&2; exit 2; }
[[ ! -e $execution_root && ! -L $execution_root ]] || { printf 'execution root must not exist: %s\n' "$execution_root" >&2; exit 2; }
[[ -d $cache_root && ! -L $cache_root ]] || { printf 'cache root must be an existing non-symlink directory\n' >&2; exit 2; }
[[ -d $build_root && ! -L $build_root ]] || { printf 'build root must be an existing non-symlink directory\n' >&2; exit 2; }
[[ -d $(dirname "$execution_root") && ! -L $(dirname "$execution_root") ]] || { printf 'execution parent must be an existing non-symlink directory\n' >&2; exit 2; }

readonly EXECUTION=$(realpath -m "$execution_root")
readonly OUTPUTS=$EXECUTION/outputs
mkdir "$EXECUTION" "$OUTPUTS"
bash "$ROOT/scripts/build_component.sh" \
    --profile "$profile" --build-root "$build_root" \
    --work-dir "$EXECUTION/build" --cache-dir "$cache_root" \
    --payload-dir "$OUTPUTS/payload"
bash "$ROOT/scripts/package_component.sh" \
    --payload-dir "$OUTPUTS/payload" \
    --source-commit "$(git -C "$ROOT" rev-parse HEAD)" \
    --work-dir "$EXECUTION/package" \
    --manifest-out "$OUTPUTS/manifest.json" \
    --archive-out "$OUTPUTS/component.tar.zst"
printf 'local_component=pass\noutput_manifest=%s\n' "$OUTPUTS/manifest.json"
