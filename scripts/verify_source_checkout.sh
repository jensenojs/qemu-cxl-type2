#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'usage: %s BASELINE_COMMIT\n' "$0" >&2
    exit 2
fi

readonly baseline=$1
readonly root=$(git rev-parse --show-toplevel)
cd "$root"

readonly head_commit=$(git rev-parse HEAD)
readonly shallow=$(git rev-parse --is-shallow-repository)
if [[ $shallow != false ]]; then
    printf 'source checkout is shallow: %s\n' "$head_commit" >&2
    exit 1
fi

git cat-file -e "${baseline}^{commit}"
if ! git merge-base --is-ancestor "$baseline" "$head_commit"; then
    printf 'baseline %s is not an ancestor of %s\n' "$baseline" "$head_commit" >&2
    exit 1
fi

readonly worktree_status=$(git status --porcelain=v1 --untracked-files=all)
if [[ -n $worktree_status ]]; then
    printf 'source checkout is dirty:\n%s\n' "$worktree_status" >&2
    exit 1
fi

printf 'source_probe=pass\n'
printf 'head_commit=%s\n' "$head_commit"
printf 'baseline_commit=%s\n' "$baseline"
printf 'shallow=%s\n' "$shallow"
printf 'worktree_clean=true\n'
