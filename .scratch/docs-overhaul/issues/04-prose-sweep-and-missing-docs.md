# 04: The prose sweep of the code docs, and `missing_docs`

Status: resolved
Blocked by: 03
Spec: `.scratch/docs-overhaul/spec.md`

**What to build:** Each module header and comment describes the code as it is
now, in STE and in the glossary terms. Each public item has a doc, and
`#![warn(missing_docs)]` keeps it so. The logic does not change. The spec gives
the scope and the list of the docs that are out of date (Implementation
Decisions, ticket 04).

## Acceptance criteria

- [x] Each item of the list of the spec is fixed.
- [x] No comment or doc has changelog prose. `ste-lint.py --enable history`
      finds none in the changed files.
- [x] No comment uses a word of an `_Avoid_` line in our own words.
- [x] No page label says "TODO".
- [x] No dead commented-out code, and no comment that asks a question.
- [x] `src/lib.rs` has `#![warn(missing_docs)]`, and each public item has a doc.
- [x] Each changed string that a test matches keeps the matched part.
- [x] `scripts/check.sh` passes, with its prose ratchet.

## Comments

- **The result.** The hard violations of `ste-lint.py --enable history` in the
  `.rs` files went from 117 to 0. `missing_docs` found about 150 public items
  without a doc in the three front-ends. Each one has a doc now. No test
  matches a string, so each string was free to change.
- **The list of the spec:**
  - The header of `store_load.rs` was already correct at the start of this
    ticket. Ticket 03 rewrote it.
  - `bench.sh` also named the dfdx sibling two times. Both mentions are gone.
    The header of `kernels.sh` named the `metal` and `rocm` backends, which
    this repo has no feature for. They are gone.
  - The docs of `MambaWrapper::step` and `add_logits` named an `i` parameter
    that `step` does not have. They describe the code now.
- **Changes outside the strict list**, all in strings or comments:
  - The page labels "Please fetch or upload" (the page cannot upload),
    "Assets not yet loaded", "Awaiting.." and the output placeholder have new
    texts in STE.
  - The four copies of the panic string of the device defaults agree.
  - The commented `on:` triggers of the deploy workflow are deleted.
- **What stays:**
  - The commented dependency lines of `Cargo.toml` (a burn git rev, the local
    paths) stay, as switches for development. `docs/agents/issue-tracker.md`
    uses the local `path` line. The burn rev of the git line can be stale.
  - `src/wasm/mod.rs` keeps the unused `use crate::Precision`, with its
    `#[allow(unused_imports)]`. This ticket changes no code line.
- **Follow-ups that this ticket found** (not done, because they change logic
  or a generated report):
  - The browser page keeps its step interval after the eos token. Each 1ms
    tick then renders the page again. Only `Msg::StopGeneration` drops the
    interval.
  - The repeat penalty reads the positions `i - repeat_last_n` to `i`, which
    is `repeat_last_n + 1` positions. Candle reads `repeat_last_n` positions.
  - The report strings of `bench.sh` and `kernels.sh` still say "one
    chunkwise pass", "one recurrent decode step" and "One whole language model
    per case". The next run writes them into `bench.md` and `kernels.md`.
