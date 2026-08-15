#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'usage: %s --manifest EXISTING_FILE --archive EXISTING_FILE --work-dir ABSENT_PATH --candidate-out ABSENT_PATH\n' "$0"
}

if [[ ${1:-} == --help ]]; then usage; exit 0; fi
if [[ ${1:-} == --hint ]]; then
    printf 'agent_hint=self=scripts/publish_component.sh\nagent_hint=boundary=pushes verified artifact bytes; does not build or package them\n'
    exit 0
fi

manifest_arg= archive_arg= work_arg= candidate_arg=
while (( $# )); do
    case "$1" in
        --manifest|--archive|--work-dir|--candidate-out)
            (( $# >= 2 )) || { usage >&2; exit 2; }
            case "$1" in
                --manifest) manifest_arg=$2 ;;
                --archive) archive_arg=$2 ;;
                --work-dir) work_arg=$2 ;;
                --candidate-out) candidate_arg=$2 ;;
            esac
            shift 2 ;;
        *) usage >&2; exit 2 ;;
    esac
done
[[ -n $manifest_arg && -n $archive_arg && -n $work_arg && -n $candidate_arg ]] || { usage >&2; exit 2; }
[[ $manifest_arg == /* && $archive_arg == /* && $work_arg == /* && $candidate_arg == /* ]] || { printf 'paths must be absolute\n' >&2; exit 2; }
[[ -f $manifest_arg && ! -L $manifest_arg && -f $archive_arg && ! -L $archive_arg ]] || { printf 'manifest and archive must be existing non-symlink files\n' >&2; exit 2; }
for path in "$work_arg" "$candidate_arg"; do
    [[ ! -e $path && ! -L $path ]] || { printf 'output path must not exist: %s\n' "$path" >&2; exit 2; }
    [[ -d $(dirname "$path") && ! -L $(dirname "$path") ]] || { printf 'output parent must be an existing non-symlink directory\n' >&2; exit 2; }
done

readonly ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
readonly CONTRACT=${ROOT}/manifests/artifact-contract.json
readonly MANIFEST=$(realpath "$manifest_arg")
readonly ARCHIVE=$(realpath "$archive_arg")
readonly WORK=$(realpath -m "$work_arg")
readonly CANDIDATE_OUT=$(realpath -m "$candidate_arg")
case "$ROOT/" in "$WORK/"*) printf 'work path cannot be an ancestor of the repository\n' >&2; exit 2;; esac
[[ $WORK != "$ROOT" ]] || { printf 'work path cannot equal repository root\n' >&2; exit 2; }
python3 - "$WORK" "$CANDIDATE_OUT" <<'PY'
import os, sys
if os.stat(os.path.dirname(sys.argv[1])).st_dev != os.stat(os.path.dirname(sys.argv[2])).st_dev:
    raise SystemExit("publish work and candidate output must share one filesystem")
PY

readarray -t contract < <(python3 - "$CONTRACT" <<'PY'
import json, sys
c=json.load(open(sys.argv[1]))
for key in ("artifact_repository", "oras_image", "artifact_type", "archive_media_type", "manifest_media_type"):
    print(c[key])
PY
)
readonly REPOSITORY=${contract[0]}
readonly ORAS_IMAGE=${contract[1]}
readonly ARTIFACT_TYPE=${contract[2]}
readonly ARCHIVE_MEDIA_TYPE=${contract[3]}
readonly MANIFEST_MEDIA_TYPE=${contract[4]}

if [[ -n ${CNB_DOCKER_REGISTRY:-} || -n ${CNB_REPO_SLUG_LOWERCASE:-} ]]; then
    readonly CNB_REPOSITORY=${CNB_DOCKER_REGISTRY:?}/${CNB_REPO_SLUG_LOWERCASE:?}
    [[ $CNB_REPOSITORY == "$REPOSITORY" ]]
fi

mkdir "$WORK"
cp "$MANIFEST" "$WORK/manifest.json"
cp "$ARCHIVE" "$WORK/component.tar.zst"
python3 - "$WORK/manifest.json" "$CONTRACT" <<'PY'
import json, sys
manifest=json.load(open(sys.argv[1]))
contract=json.load(open(sys.argv[2]))
expected=(contract["component"], contract["source_repository"], contract["toolchain_image"])
actual=(manifest.get("component"), manifest.get("source",{}).get("repository"), manifest.get("build",{}).get("toolchain_image"))
if actual != expected:
    raise SystemExit(f"manifest identity does not match artifact contract: actual={actual!r} expected={expected!r}")
PY
zstd -q -d "$WORK/component.tar.zst" -o "$WORK/component.tar"
python3 "$ROOT/scripts/component_artifact.py" verify-archive \
    --archive "$WORK/component.tar" \
    --manifest "$WORK/manifest.json"

readarray -t identity < <(python3 - "$WORK/manifest.json" <<'PY'
import json, sys
m=json.load(open(sys.argv[1]))
print(m["source"]["commit"])
print(m["build"]["profile_sha256"])
PY
)
readonly SOURCE_COMMIT=${identity[0]}
readonly PROFILE_SHA256=${identity[1]}

readonly ORAS_BIN=${WORK}/oras
oras_container=$(docker create "$ORAS_IMAGE")
trap 'docker rm -f "$oras_container" >/dev/null 2>&1 || true' EXIT
docker cp "${oras_container}:/bin/oras" "$ORAS_BIN"
docker rm "$oras_container" >/dev/null
oras_container=
chmod +x "$ORAS_BIN"
"$ORAS_BIN" version

tag_id=${CNB_BUILD_ID:-local}
tag_id=${tag_id//[^a-zA-Z0-9_.-]/-}
readonly TAG="candidate-${tag_id}-${SOURCE_COMMIT:0:12}"
readonly REFERENCE=${REPOSITORY}:${TAG}

(cd "$WORK" && "$ORAS_BIN" push "$REFERENCE" \
    --artifact-type "$ARTIFACT_TYPE" \
    "component.tar.zst:${ARCHIVE_MEDIA_TYPE}" \
    "manifest.json:${MANIFEST_MEDIA_TYPE}" \
    --format json >push.json)

digest=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["digest"])' \
    "$WORK/push.json")
[[ $digest == sha256:* ]]
archive_sha256=$(sha256sum "$ARCHIVE" | awk '{print $1}')
manifest_sha256=$(sha256sum "$MANIFEST" | awk '{print $1}')
python3 - "$REPOSITORY" "$digest" "$SOURCE_COMMIT" "$PROFILE_SHA256" \
    "$archive_sha256" "$manifest_sha256" "$WORK/candidate.json" <<'PY'
import json, sys
keys=("repository","digest","source_commit","profile_sha256","archive_sha256","manifest_sha256")
with open(sys.argv[7], "w", encoding="utf-8") as stream:
    json.dump(dict(zip(keys,sys.argv[1:7])),stream,indent=2,sort_keys=True)
    stream.write("\n")
PY
python3 - "$WORK/candidate.json" "$CANDIDATE_OUT" <<'PY'
import os, sys
os.replace(sys.argv[1], sys.argv[2])
PY

printf 'component_publish=pass\n'
printf 'artifact_reference=%s@%s\n' "$REPOSITORY" "$digest"
printf 'archive_sha256=%s\n' "$archive_sha256"
printf 'manifest_sha256=%s\n' "$manifest_sha256"
printf '%s\n' '=== CNB_OUTPUT_BEGIN component-candidate ==='
python3 - "$CANDIDATE_OUT" <<'PY'
import json, sys
candidate=json.load(open(sys.argv[1]))
json.dump({
    "name":"component-candidate",
    "kind":"component-candidate",
    "transport":"inline-json",
    "payload":candidate,
},sys.stdout,indent=2,sort_keys=True)
print()
PY
printf '%s\n' '=== CNB_OUTPUT_END component-candidate ==='
