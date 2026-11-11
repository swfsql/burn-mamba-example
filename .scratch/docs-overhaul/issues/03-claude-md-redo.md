# 03: A short CLAUDE.md, in the shape of the sibling files

Status: resolved
Blocked by: 02
Spec: `.scratch/docs-overhaul/spec.md`

**What to build:** An agent that starts a session reads a short `CLAUDE.md`
that points to where each kind of knowledge lives. Its sections and its rules
follow `../burn-mamba/CLAUDE.md`. The content of the Architecture and Key
Design Decisions sections moves to the module headers, the glossaries and the
ADRs. The spec gives each decision in full, with the destination of each
section (Implementation Decisions, ticket 03).

## Acceptance criteria

- [x] `CLAUDE.md` has about 200 lines or fewer. Its File Map has 40 lines or
      fewer.
- [x] The sections are the sections of the spec, in that order.
- [x] The rules of the spec are in `CLAUDE.md`. They include the path rule of
      the spec and the access allowlist with the roots of ticket 01.
- [x] `CLAUDE.md` has no backend verdict.
- [x] Each deleted paragraph has its content in a module header, a glossary or
      an ADR, or is listed under `## Comments`.
- [x] The header of the `hf` module holds the checklist to add a checkpoint.
- [x] No doc links to a `CLAUDE.md` anchor that is gone.
- [x] `ste-lint.py` passes on `CLAUDE.md`, and `scripts/check.sh` passes.

## Comments

- Ticket 01 added the access allowlist to the Custom Commands of `CLAUDE.md`.
  Keep its rule on burn-mamba and burn-stack. They have other users, so they
  stay read-only, except their `.scratch/`. A change to them starts as a ticket
  in their own tracker.
  Do not copy the read and write roots of `../burn-mamba/CLAUDE.md` for them.
- **The size.** `CLAUDE.md` has 218 lines, and its File Map has 23. The rules
  that the spec copies from burn-mamba take 59 lines of Custom Commands.
- **Where each deleted section went:**
  - Load path: the header of `src/common/store_load.rs` now gives the steps of
    `build_network`, and the f32, f16 and bf16 checkpoints. The docs of
    `key_remapping` and `tie_lm_head` hold the rest. A new comment on the
    structural rule says that the `mlp` of Mamba-3 needs no other rule.
  - Two run modes: the glossary, and a new doc on `MambaWrapper`.
  - Verifying a backend: the glossary (Run agreement, Reference backend), the
    seam "Sequential against parallel" of `docs/agents/testing.md`, and one
    bullet of the commands. The backend verdicts are gone.
  - Device & precision: ADR-0003, and a new header of `src/common/mod.rs`.
  - Browser asset lifecycle: the glossary, and a new header of
    `src/wasm/yew_ui/mod.rs`. That header also holds the 1ms tick of the
    generation, from the old File Map line of `update.rs`.
  - Glue only: the summary and a Code rule.
  - No candle / tokenizers / hf-hub: ADR-0001. The doc of the pre-tokenizer
    parse in `src/common/tokenizer/mod.rs` holds the verbatim compare of the
    `Split` regex.
  - Deps are git-rev-pinned: "The burn-mamba pin" of
    `docs/agents/issue-tracker.md`. The old paragraph was out of date, because
    `burn` comes from crates.io.
  - Configs are hardcoded: ADR-0002, and a new header of `hf`. The doc of
    `mlp()` holds the rounding of `d_intermediate`.
  - Mamba-3 needs the per-layer `GatedMlp`: deleted.
  - Model choice is runtime data: ADR-0004, and the new header of `hf`, with
    the checklist.
  - `frontend/*/pkg/` is generated, and CI = deploy: a File Map line and a Code
    rule. The checklist names the edits of the workflow and of the check.
- **Other deleted lines:**
  - The `bacon.toml` line. Git ignores that file.
  - The long File Map lines. The headers of `bench.sh`, `kernels.sh`,
    `benches/model.rs`, `src/common/hub/` and `src/common/tokenizer/mod.rs`,
    and `docs/agents/testing.md`, already hold their content.
  - "Write no `tmp.md` and no `files.md`". burn-mamba has no `files.md`, and
    the rule of temporary work puts each temporary file in `tmp/`.
  - "Keep `README.md` and `CLAUDE.md` in sync". "Where to look" points to
    `README.md`, and ticket 05 rewrites it.
  - The live URLs and the download sizes of each file. `README.md` has the
    URLs. One bullet gives the sizes, rounded.
  - The command of the allocations bench. The header comment of
    `benches/allocations.rs` gives it, and the File Map names the file.
  - The `getrandom` line. The File Map line of `.cargo/config.toml` names it.
- **Other files:**
  - `.gitignore` has `tmp/`, for the rule of temporary work. Without it, the
    host-path step of the check reads the reports in `tmp/`.
  - `docs/box.tsv` has `py/state-spaces/mamba`, because "Where to look" names
    it. Its version is the HEAD of the checkout on this box, the same as in the
    burn-mamba list.
  - The example of `docs/agents/code-review.md` names the rule of ADR-0001.
    The "Burn crate first" rule is there now.
  - The header of `store_load.rs` said "both checkpoints". Ticket 04 listed
    this fix. This ticket did it, because it edits that header.
- **Rules that differ from burn-mamba:**
  - The rule on `_Avoid_` cites no ADR. burn-mamba cites its ADR-0008. Here,
    the criteria of ticket 02 hold the same rule. An ADR is not necessary,
    because the rule has no trade-off.
  - The Pushes rule adds "A push to `main` deploys".
- **The memory.** The spec says that the path rule replaces an auto-memory
  with the same rule. The auto-memory of this repo has no such memory, so no
  memory changed. The user gave the rule again at the start of this ticket.
- **A term for `/domain-modeling`.** "Run wrapper" names `MambaWrapper` in
  the code and in the spec, but no glossary defines it.
- **The review.** The two agents found no blocking defect. The fixes:
  - The `hf` header linked `CHECKPOINTS` and `CheckpointSpec`. A `cargo doc`
    with `default = []` has neither item, so it gave three warnings. The check
    builds the docs with the native features only, so it did not see them. The
    header now names the two items in code spans.
  - The new prose used "target" beside "front-end", "load" for the build of
    the network, "assets" for a native run and "decode" for the detokenizer.
    It now uses the glossary terms.
  - "The rev of `Cargo.toml` wins" was true only for burn-mamba. `Cargo.lock`
    also pins burn-stack, so the line names `Cargo.lock`.
  - The manifest test fetches about 1.4GB, so it moved to the commands that
    the user runs.
  - The checklist names the doc of `CHECKPOINTS`, the two Mamba-3 tests, the
    priority comment of `Cargo.toml` and the id lists of `README.md`.
  - The File Map has lines for `.cargo/config.toml` and `allocations.rs`, and
    tells that git ignores `pkg/`.
  - The `store_load.rs` paragraph that this ticket edits had a sentence of 35
    words. It is now three sentences.
- **The review findings that stay**, with the reason:
  - Run agreement is in `CLAUDE.md`, the glossary, `testing.md` and the doc of
    `MambaWrapper`. The spec puts it in each of these places.
  - "A push to `main` deploys" is in the Code rule and in the Pushes rule. An
    agent reads the Pushes rule at the time of a push.
  - The rule "Document each new public item" has no `missing_docs` lint yet.
    Ticket 04 adds the lint.
  - `docs/box.tsv` does not list `../cubecl/`, `../cubek/` and `../papers/`.
    `CLAUDE.md` names them only as roots of the access allowlist, as before.
