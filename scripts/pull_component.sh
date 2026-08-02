#!/usr/bin/env bash
set -euo pipefail

container_runtime=
work_dir=
while [[ $# -gt 0 ]]; do
    case "$1" in
        --container-runtime)
            [[ $# -ge 2 ]] || { printf 'error: --container-runtime requires a value\n' >&2; exit 2; }
            container_runtime=$2; shift 2 ;;
        --work-dir)
            [[ $# -ge 2 ]] || { printf 'error: --work-dir requires a value\n' >&2; exit 2; }
            work_dir=$2; shift 2 ;;
        --) shift; break ;;
        -*) printf 'error: unknown argument: %s\n' "$1" >&2; exit 2 ;;
        *) break ;;
    esac
done
if [[ $# -ne 2 ]]; then
    printf 'usage: %s --container-runtime docker|podman --work-dir ABSENT_ABSOLUTE_PATH CANDIDATE_JSON OUTPUT_DIR\n' "$0" >&2
    exit 2
fi

case "$container_runtime" in
    docker|podman) ;;
    "") printf 'error: --container-runtime is required\n' >&2; exit 2 ;;
    *) printf 'error: --container-runtime must be docker or podman\n' >&2; exit 2 ;;
esac
[[ -n $work_dir ]] || { printf 'error: --work-dir is required\n' >&2; exit 2; }
command -v "$container_runtime" >/dev/null 2>&1 || {
    printf 'error: missing container runtime: %s\n' "$container_runtime" >&2
    exit 1
}

readonly ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
readonly CANDIDATE=$(realpath "$1")
readonly OUTPUT=$2
readonly CONTRACT=${ROOT}/manifests/artifact-contract.json
[[ $work_dir == /* ]] || { printf 'error: work directory must be absolute\n' >&2; exit 1; }
python3 - "$work_dir" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
if path.exists() or path.is_symlink():
    raise SystemExit(f"work directory must not exist: {path}")
if not path.parent.is_dir():
    raise SystemExit(f"work directory parent must exist: {path.parent}")
for ancestor in (path.parent, *path.parent.parents):
    if ancestor.is_symlink():
        raise SystemExit(f"work directory ancestor must not be a symlink: {ancestor}")
PY
readonly WORK=$(realpath -m "$work_dir")

readarray -t candidate < <(python3 - "$CANDIDATE" <<'PY'
import json, sys
c=json.load(open(sys.argv[1]))
required={"repository","digest","source_commit","profile_sha256","archive_sha256","manifest_sha256"}
if set(c) != required or not c["digest"].startswith("sha256:"):
    raise SystemExit("invalid candidate manifest")
for key in ("repository","digest","source_commit","profile_sha256","archive_sha256","manifest_sha256"):
    print(c[key])
PY
)
readonly REPOSITORY=${candidate[0]}
readonly DIGEST=${candidate[1]}
readonly EXPECTED_ARCHIVE_SHA256=${candidate[4]}
readonly EXPECTED_MANIFEST_SHA256=${candidate[5]}

readarray -t contract < <(python3 - "$CONTRACT" <<'PY'
import json, sys
c=json.load(open(sys.argv[1]))
print(c["artifact_repository"])
print(c["oras_image"])
print(c["toolchain_image"])
PY
)
[[ $REPOSITORY == "${contract[0]}" ]]
readonly ORAS_IMAGE=${contract[1]}
readonly TOOLCHAIN_IMAGE=${contract[2]}
[[ $TOOLCHAIN_IMAGE == *@sha256:* ]] || {
    printf 'error: artifact contract toolchain_image must be pinned by digest\n' >&2
    exit 1
}

mkdir "$WORK"
mkdir "$WORK/pulled"
readonly ORAS_BIN=${WORK}/oras
oras_container=$("$container_runtime" create "$ORAS_IMAGE")
trap '"$container_runtime" rm -f "${oras_container:-}" >/dev/null 2>&1 || true' EXIT
"$container_runtime" cp "${oras_container}:/bin/oras" "$ORAS_BIN"
"$container_runtime" rm "$oras_container" >/dev/null
oras_container=
chmod +x "$ORAS_BIN"

"$ORAS_BIN" pull "${REPOSITORY}@${DIGEST}" --output "$WORK/pulled" \
    --format json >"$WORK/pull.json"

readarray -t outer_files < <(find "$WORK/pulled" -mindepth 1 -maxdepth 1 -type f -printf '%f\n' | LC_ALL=C sort)
[[ ${#outer_files[@]} -eq 2 ]]
[[ ${outer_files[0]} == component.tar.zst ]]
[[ ${outer_files[1]} == manifest.json ]]
printf '%s  %s\n' "$EXPECTED_ARCHIVE_SHA256" "$WORK/pulled/component.tar.zst" | sha256sum --check
printf '%s  %s\n' "$EXPECTED_MANIFEST_SHA256" "$WORK/pulled/manifest.json" | sha256sum --check

zstd -q -d "$WORK/pulled/component.tar.zst" -o "$WORK/component.tar"
python3 "$ROOT/scripts/component_artifact.py" verify-archive \
    --archive "$WORK/component.tar" \
    --manifest "$WORK/pulled/manifest.json" \
    --extract-dir "$OUTPUT"

runtime_args=(run --rm --user "$(id -u):$(id -g)")
if [[ $container_runtime == podman ]]; then
    runtime_args+=(--userns=keep-id --security-opt label=disable)
fi
runtime_args+=(
    --mount "type=bind,src=$OUTPUT,dst=/payload,readonly"
    --mount "type=bind,src=$WORK,dst=/work"
    --mount "type=bind,src=$ROOT/scripts/verify_component_payload.sh,dst=/verify_component_payload.sh,readonly"
    "$TOOLCHAIN_IMAGE"
    bash /verify_component_payload.sh /payload /work/evidence
)
"$container_runtime" "${runtime_args[@]}"

printf 'component_fresh_pull=pass\n'
printf 'artifact_reference=%s@%s\n' "$REPOSITORY" "$DIGEST"
printf 'restored_payload=%s\n' "$OUTPUT"
