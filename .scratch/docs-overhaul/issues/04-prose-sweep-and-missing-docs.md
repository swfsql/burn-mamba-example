# 04: The prose sweep of the code docs, and `missing_docs`

Status: ready-for-agent
Blocked by: 03
Spec: `.scratch/docs-overhaul/spec.md`

**What to build:** Each module header and comment describes the code as it is
now, in STE and in the glossary terms. Each public item has a doc, and
`#![warn(missing_docs)]` keeps it so. The logic does not change. The spec gives
the scope and the list of the docs that are out of date (Implementation
Decisions, ticket 04).

## Acceptance criteria

- [ ] Each item of the list of the spec is fixed.
- [ ] No comment or doc has changelog prose. `ste-lint.py --enable history`
      finds none in the changed files.
- [ ] No comment uses a word of an `_Avoid_` line in our own words.
- [ ] No page label says "TODO".
- [ ] No dead commented-out code, and no comment that asks a question.
- [ ] `src/lib.rs` has `#![warn(missing_docs)]`, and each public item has a doc.
- [ ] Each changed string that a test matches keeps the matched part.
- [ ] `scripts/check.sh` passes, with its prose ratchet.

## Comments
