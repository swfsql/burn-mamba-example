# 01: The check script, the access hook and the box list

Status: resolved
Blocked by: None (can start immediately)
Spec: `.scratch/docs-overhaul/spec.md`

**What to build:** A clean clone and the CI run the same check:
`scripts/check.sh`. The deploy workflow runs it before the four bundle builds.
A local hook refuses each path outside the access allowlist. `docs/box.tsv`
lists the directories next to this repo that the docs name. The spec gives
each decision in full (Implementation Decisions, ticket 01).

## Acceptance criteria

- [x] `scripts/check.sh` runs the steps of the spec, with three feature sets.
      These are the native set (for `cargo check`, `cargo test` and
      `cargo doc`), the browser page and the browser console. Each cargo step
      fails on a warning.
- [x] The check builds no cuda, no other backend, no bench and no `#[ignore]`
      test.
- [x] The host-path step and the prose ratchet follow
      `../burn-mamba/scripts/check.sh`, and skip when `CI` is set.
- [x] The check passes on this box. If a step gives warnings on the code of
      today, fix the cause, or list it under `## Comments` with the step that
      gives it.
- [x] The deploy workflow runs the check with `CI` set, before the four bundle
      builds.
- [x] `.claude/hooks/access-allowlist.py` is a copy of the burn-mamba hook,
      with the roots of the spec. `.claude/settings.local.json` runs it as a
      `PreToolUse` hook, and keeps its other entries.
- [x] `docs/box.tsv` follows the format of the sibling lists.
      `../burn-stack/scripts/setup-box.sh --root <an empty directory>` reads it
      without a warning about its format.
- [x] `docs/agents/code-review.md` and `docs/agents/testing.md` name the check
      script, not the bacon jobs.

## Notes

- Ticket 02 renames the features. This ticket uses the names of today, and
  ticket 02 updates the check script.
- The hook and its settings are in `.claude/`, which git ignores. They are not
  part of the commit. Name them in the commit message.

## Comments

- **Warnings on the code of today, fixed:**
  - `cargo doc` gave three warnings about doc links. `src/common/hub/mod.rs`
    linked `[wasm]`, which a native build does not have.
    `src/common/tokenizer/mod.rs` linked two private items. The three links are
    now code spans, with the same words. The `[sync]` link next to `[wasm]`
    gave no warning, but it is also a code span now, so that the two names
    match. Ticket 04 rewrites these headers.
  - The host-path step found the absolute home path of this box in
    `.scratch/bump-burn-mamba-main/issues/05-the-cuda-bench-build-does-not-link.md`.
    It is now `~/.local/…`.
  - The nightly `cargo check` of the browser page gave
    `unused dependency pretty_env_logger`: only the native binary uses it. It is
    now optional, and the `native` feature enables it. `Cargo.lock` does not
    change.
- **Color codes:** the deploy workflow sets `CARGO_TERM_COLOR: always`. Then a
  color code comes before each `warning`, and the `^warning` grep of
  `../burn-mamba/scripts/check.sh` finds no warning. A throwaway test crate
  showed it: 0 matches with the plain grep, 2 with the grep of this repo.
  `no_warnings` here skips the color codes. `../burn-mamba/.github/workflows/check.yml`
  sets the same variable, so the CI of burn-mamba does not see a warning. That
  fix is for a burn-mamba session.
- **Toolchains in CI:** the deploy installs stable, then nightly. Nightly stays
  the default, so wasm-pack builds as before. The check step sets
  `RUSTUP_TOOLCHAIN: stable`, so the native steps use stable, as on this box.
  `cargo +nightly` wins over that variable, so the wasm steps use nightly.
- **The hook:** the hub cache root follows `$HF_HOME`, as
  `src/common/hub/sync.rs` does. The spec does not name `$CARGO_HOME/bin`, so
  this copy does not have that root. The burn-mamba copy has it.
- **The box list:** each version is the HEAD of the checkout on this box.
  `burn-mamba` and `burn-stack` have `-`, as in the list of burn-stack.
  `setup-box.sh --root <an empty directory> docs/box.tsv` gave
  `done: created 6, present 0, warnings 0.`
- **The decisions of the maintainer, after the review:**
  - The prose ratchet also reads `.scratch/`, as the Testing Decisions of the
    spec say. `../burn-mamba/scripts/check.sh` skips it. A ticket in the tracker
    of burn-mamba and a ticket in the tracker of burn-stack ask for the same
    change. The spec failed the wider ratchet with two hard violations. Each was
    a name next to a synonym (`Erase` and `delete`, `Verifying` and `check`).
    The names are now code spans.
  - The hook gets the root `$CARGO_HOME/bin`, as in the burn-mamba copy. The
    auto mode of the agent refused that edit of its own hook, so the maintainer
    makes it.
  - `.claude/settings.local.json` lost the three `permissions.allow` entries
    that the hook does not permit. The first is the `Read` rule for the parent
    directory of this repo. The second is `Read` of `~/.cargo/registry`, because
    the cargo home of this box is elsewhere. The third is `Bash(find / …)`.
  - `CLAUDE.md` now has the access allowlist. burn-mamba and burn-stack have
    other users, so they stay read-only. A change to them starts as a ticket in
    their own tracker.
