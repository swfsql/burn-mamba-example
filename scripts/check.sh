#!/usr/bin/env bash
# The check of this repo. CI runs it and skips the two steps that read this box:
# the host paths and the prose.
#
# Each build has the four checkpoint features and the flex backend. The check
# builds one feature set for each front-end: the native binary, the browser page
# and the browser console. It builds no other backend, no bench and no
# `#[ignore]` test.
set -euo pipefail
cd "$(dirname "$0")/.."

checkpoints="mamba1,mamba2,mamba3-siso,mamba3-mimo"
native_features="native,backend-flex,backend-simd,$checkpoints"
page_features="yew,backend-flex,backend-simd,$checkpoints"
console_features="backend-flex,backend-simd,$checkpoints"

# Run a cargo command. Fail when its output holds a warning. A color code can
# come before the word, for example when `CARGO_TERM_COLOR=always`.
no_warnings() {
    local output esc=$'\e'
    output="$("$@" 2>&1)" || {
        printf '%s\n' "$output" >&2
        return 1
    }
    printf '%s\n' "$output"
    if grep -qE "^(${esc}\[[0-9;]*m)*warning" <<<"$output"; then
        echo "error: \`$*\` gave the warnings above." >&2
        return 1
    fi
}

# Fail when a file has more hard STE violations than it has on origin/main. A
# new file starts at zero. The linter is part of the `asd-ste100` skill.
check_prose() {
    local lint="$HOME/.claude/skills/asd-ste100/scripts/ste-lint.py"
    if [ ! -f "$lint" ]; then
        echo "note: this box has no ste-lint.py, so the prose check did not run." >&2
        return 0
    fi
    hard_violations() {
        python3 "$lint" --enable history --summary --baseline 1000000 "$1" | awk 'NR == 2 { print $1 }'
    }
    local base old_dir file before after failed=0
    base="$(git merge-base origin/main HEAD)"
    old_dir="$(mktemp -d)"
    while IFS= read -r file; do
        before=0
        if git cat-file -e "$base:$file" 2>/dev/null; then
            mkdir -p "$old_dir/$(dirname "$file")"
            git show "$base:$file" >"$old_dir/$file"
            before="$(hard_violations "$old_dir/$file")"
        fi
        after="$(hard_violations "$file")"
        if [ "$after" -gt "$before" ]; then
            echo "error: $file has $after hard STE violations. origin/main has $before." >&2
            python3 "$lint" --enable history "$file" >&2 || true
            failed=1
        fi
    done < <(
        {
            git diff --name-only --diff-filter=d "$base" -- '*.rs' '*.md'
            git ls-files --others --exclude-standard -- '*.rs' '*.md'
        } | sort -u
    )
    rm -rf "$old_dir"
    return "$failed"
}

if [ -z "${CI:-}" ]; then
    # This box gives the two roots, so this file names no path of a box.
    repo_root="/$(pwd | cut -d/ -f2)"
    home_root="/$(printf '%s' "$HOME" | cut -d/ -f2)"
    if git grep --untracked -n -I -E "(^|[^[:alnum:]_./-])($repo_root|$home_root)(/|\$)"; then
        echo "error: the lines above hold a host path. Write a path relative to the repo." >&2
        exit 1
    fi
    check_prose
fi

no_warnings cargo check --no-default-features --features "$native_features"
no_warnings cargo test --no-default-features --features "$native_features" -- --test-threads=1
no_warnings cargo doc --no-deps --no-default-features --features "$native_features"
for features in "$page_features" "$console_features"; do
    no_warnings cargo +nightly check --target wasm32-unknown-unknown --no-default-features --features "$features"
done
