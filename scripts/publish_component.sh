#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'usage: %s PAYLOAD_DIR\n' "$0" >&2
    exit 2
fi

readonly ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
readonly PAYLOAD=$(realpath "$1")
readonly PROFILE=${ROOT}/manifests/build-profile.json
readonly CONTRACT=${ROOT}/manifests/artifact-contract.json
readonly WORK=${ROOT}/.work/component/publish
readonly SOURCE_COMMIT=$(git -C "$ROOT" rev-parse HEAD)

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

rm -rf "$WORK"
mkdir -p "$WORK"
python3 "$ROOT/scripts/component_artifact.py" create-manifest \
    --payload "$PAYLOAD" \
    --profile "$PROFILE" \
    --contract "$CONTRACT" \
    --source-commit "$SOURCE_COMMIT" \
    --output "$WORK/manifest.json"

find "$PAYLOAD" -mindepth 1 \( -type f -o -type l \) -printf '%P\0' |
    LC_ALL=C sort -z >"$WORK/archive-files.list"

for run in 1 2; do
    tar --format=posix --directory="$PAYLOAD" --no-recursion \
        --mtime=@0 --owner=0 --group=0 --numeric-owner \
        --pax-option=delete=atime,delete=ctime --null \
        --files-from="$WORK/archive-files.list" \
        -cf "$WORK/component-${run}.tar"
    zstd -q -T1 -19 -f "$WORK/component-${run}.tar" \
        -o "$WORK/component-${run}.tar.zst"
done

cmp "$WORK/component-1.tar.zst" "$WORK/component-2.tar.zst"
mv "$WORK/component-1.tar.zst" "$WORK/component.tar.zst"
python3 "$ROOT/scripts/component_artifact.py" verify-archive \
    --archive "$WORK/component-1.tar" \
    --manifest "$WORK/manifest.json"

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
archive_sha256=$(sha256sum "$WORK/component.tar.zst" | awk '{print $1}')
manifest_sha256=$(sha256sum "$WORK/manifest.json" | awk '{print $1}')
profile_sha256=$(python3 - "$PROFILE" <<'PY'
import hashlib, json, sys
value=json.load(open(sys.argv[1]))
data=(json.dumps(value,sort_keys=True,separators=(",",":"))+"\n").encode()
print(hashlib.sha256(data).hexdigest())
PY
)

python3 - "$REPOSITORY" "$digest" "$SOURCE_COMMIT" "$profile_sha256" \
    "$archive_sha256" "$manifest_sha256" >"$WORK/candidate.json" <<'PY'
import json, sys
keys=("repository","digest","source_commit","profile_sha256","archive_sha256","manifest_sha256")
json.dump(dict(zip(keys,sys.argv[1:])),sys.stdout,indent=2,sort_keys=True)
print()
PY

printf 'component_publish=pass\n'
printf 'artifact_reference=%s@%s\n' "$REPOSITORY" "$digest"
printf 'archive_sha256=%s\n' "$archive_sha256"
printf 'manifest_sha256=%s\n' "$manifest_sha256"
printf '%s\n' '=== COMPONENT_CANDIDATE_JSON_BEGIN ==='
cat "$WORK/candidate.json"
printf '%s\n' '=== COMPONENT_CANDIDATE_JSON_END ==='
