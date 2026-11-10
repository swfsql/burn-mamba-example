# 01: The check script, the access hook and the box list

Status: ready-for-agent
Blocked by: None (can start immediately)
Spec: `.scratch/docs-overhaul/spec.md`

**What to build:** A clean clone and the CI run the same check:
`scripts/check.sh`. The deploy workflow runs it before the four bundle builds.
A local hook refuses each path outside the access allowlist. `docs/box.tsv`
lists the directories next to this repo that the docs name. The spec gives
each decision in full (Implementation Decisions, ticket 01).

## Acceptance criteria

- [ ] `scripts/check.sh` runs the steps of the spec, with three feature sets.
      These are the native set (for `cargo check`, `cargo test` and
      `cargo doc`), the browser page and the browser console. Each cargo step
      fails on a warning.
- [ ] The check builds no cuda, no other backend, no bench and no `#[ignore]`
      test.
- [ ] The host-path step and the prose ratchet follow
      `../burn-mamba/scripts/check.sh`, and skip when `CI` is set.
- [ ] The check passes on this box. If a step gives warnings on the code of
      today, fix the cause, or list it under `## Comments` with the step that
      gives it.
- [ ] The deploy workflow runs the check with `CI` set, before the four bundle
      builds.
- [ ] `.claude/hooks/access-allowlist.py` is a copy of the burn-mamba hook,
      with the roots of the spec. `.claude/settings.local.json` runs it as a
      `PreToolUse` hook, and keeps its other entries.
- [ ] `docs/box.tsv` follows the format of the sibling lists.
      `../burn-stack/scripts/setup-box.sh --root <an empty directory>` reads it
      without a warning about its format.
- [ ] `docs/agents/code-review.md` and `docs/agents/testing.md` name the check
      script, not the bacon jobs.

## Notes

- Ticket 02 renames the features. This ticket uses the names of today, and
  ticket 02 updates the check script.
- The hook and its settings are in `.claude/`, which git ignores. They are not
  part of the commit. Name them in the commit message.

## Comments
