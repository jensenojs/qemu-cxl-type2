#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'usage: %s --payload-dir EXISTING_PATH --source-commit 40_HEX --work-dir ABSENT_PATH --manifest-out ABSENT_PATH --archive-out ABSENT_PATH\n' "$0"
}

if [[ ${1:-} == --help ]]; then usage; exit 0; fi
if [[ ${1:-} == --hint ]]; then
    printf 'agent_hint=self=scripts/package_component.sh\nagent_hint=boundary=creates deterministic artifact bytes without registry access\n'
    exit 0
fi

payload_arg= source_commit= work_arg= manifest_arg= archive_arg=
while (( $# )); do
    case "$1" in
        --payload-dir|--source-commit|--work-dir|--manifest-out|--archive-out)
            (( $# >= 2 )) || { usage >&2; exit 2; }
            case "$1" in
                --payload-dir) payload_arg=$2 ;;
                --source-commit) source_commit=$2 ;;
                --work-dir) work_arg=$2 ;;
                --manifest-out) manifest_arg=$2 ;;
                --archive-out) archive_arg=$2 ;;
            esac
            shift 2 ;;
        *) usage >&2; exit 2 ;;
    esac
done
[[ -n $payload_arg && -n $source_commit && -n $work_arg && -n $manifest_arg && -n $archive_arg ]] || { usage >&2; exit 2; }
[[ $payload_arg == /* && $work_arg == /* && $manifest_arg == /* && $archive_arg == /* ]] || { printf 'paths must be absolute\n' >&2; exit 2; }
[[ $source_commit =~ ^[0-9a-f]{40}$ ]] || { printf 'source commit must be 40 lowercase hex characters\n' >&2; exit 2; }

readonly ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
readonly PROFILE=$ROOT/manifests/build-profile.json
readonly CONTRACT=$ROOT/manifests/artifact-contract.json
[[ $(git -C "$ROOT" rev-parse HEAD) == "$source_commit" ]] || { printf 'source commit does not equal checkout HEAD\n' >&2; exit 1; }
[[ -z $(git -C "$ROOT" status --porcelain=v1 --untracked-files=all) ]] || { printf 'source checkout must be clean\n' >&2; exit 1; }
[[ -d $payload_arg && ! -L $payload_arg ]] || { printf 'payload must be an existing non-symlink directory\n' >&2; exit 2; }
for path in "$work_arg" "$manifest_arg" "$archive_arg"; do
    [[ ! -e $path && ! -L $path ]] || { printf 'output path must not exist: %s\n' "$path" >&2; exit 2; }
    [[ -d $(dirname "$path") && ! -L $(dirname "$path") ]] || { printf 'output parent must be an existing non-symlink directory: %s\n' "$path" >&2; exit 2; }
done
readonly WORK=$(realpath -m "$work_arg")
readonly PAYLOAD=$(realpath "$payload_arg")
readonly MANIFEST_OUT=$(realpath -m "$manifest_arg")
readonly ARCHIVE_OUT=$(realpath -m "$archive_arg")
case "$ROOT/" in "$WORK/"*) printf 'work path cannot be an ancestor of the repository\n' >&2; exit 2;; esac
[[ $WORK != "$ROOT" ]] || { printf 'work path cannot equal repository root\n' >&2; exit 2; }
python3 - "$WORK" "$MANIFEST_OUT" "$ARCHIVE_OUT" <<'PY'
import os, sys
devices = {os.stat(os.path.dirname(path)).st_dev for path in sys.argv[1:]}
if len(devices) != 1:
    raise SystemExit("package work and terminal outputs must share one filesystem")
PY

mkdir "$WORK"
python3 "$ROOT/scripts/component_artifact.py" create-manifest \
    --payload "$PAYLOAD" --profile "$PROFILE" --contract "$CONTRACT" \
    --source-commit "$source_commit" --output "$WORK/manifest.json"

find "$PAYLOAD" -mindepth 1 \( -type f -o -type l \) -printf '%P\0' |
    LC_ALL=C sort -z >"$WORK/archive-files.list"
for run in 1 2; do
    tar --format=posix --directory="$PAYLOAD" --no-recursion \
        --mtime=@0 --owner=0 --group=0 --numeric-owner \
        --pax-option=delete=atime,delete=ctime --null \
        --files-from="$WORK/archive-files.list" -cf "$WORK/component-${run}.tar"
    zstd -q -T1 -19 -f "$WORK/component-${run}.tar" -o "$WORK/component-${run}.tar.zst"
done
cmp "$WORK/component-1.tar.zst" "$WORK/component-2.tar.zst"
python3 "$ROOT/scripts/component_artifact.py" verify-archive \
    --archive "$WORK/component-1.tar" --manifest "$WORK/manifest.json" \
    --extract-dir "$WORK/fresh-payload"
python3 "$ROOT/scripts/component_artifact.py" verify-payload \
    --payload "$WORK/fresh-payload" --manifest "$WORK/manifest.json"

python3 - "$WORK/manifest.json" "$MANIFEST_OUT" "$WORK/component-1.tar.zst" "$ARCHIVE_OUT" <<'PY'
import os, sys
for source, destination in ((sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4])):
    os.replace(source, destination)
PY
printf 'component_package=pass\nmanifest=%s\narchive=%s\n' "$MANIFEST_OUT" "$ARCHIVE_OUT"
