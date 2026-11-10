# Docs overhaul: the standards of burn-mamba and burn-stack

Status: ready-for-agent

## Problem Statement

The maintainer wants this repo to follow the standards of `../burn-mamba` and
`../burn-stack`. Their `CLAUDE.md` files set the direction. burn-mamba did the
same work in its commit `b0a17ee3`. Its ticket is
`.scratch/glossary-followups/issues/01-docs-redo.md`.

The docs of this repo do not follow these standards yet:

- `CLAUDE.md` has 384 lines. It repeats the module headers, and parts of it
  are out of date: the remap target `mamba_block`, the `skip_enum_variants`
  reason, `Layer::{norm2,mlp}` and `LayersBuilder::with_mlp`. It calls
  `bacon.toml` canonical, but git ignores that file.
- `README.md` has a backend table that is out of date. It also claims two
  things that the browser page cannot do. The page cannot take an asset from a
  local file, and it cannot save the cached copy to a file.
- The code uses words that the glossary avoids. "Model" names a checkpoint
  (`ModelSpec`, `hf::MODELS`, `MAMBA_MODEL`). "Cache" names both the state of a
  block and the hub cache. `CHUNK_SIZE` names both a download chunk and an SSD
  chunk length.
- Some code docs are out of date or hold changelog prose (see ticket 04).
- The repo has no tracked check script, no access allowlist hook and no
  `docs/box.tsv`. The siblings have all three.

## Solution

The grilling session of this spec already did these parts:

- `GLOSSARY.md`: the new terms `Topology`, `Checkpoint feature`,
  `Reference backend`, `Front-end`, `Hub cache`, `Fetch`, `Erase`, `Load`,
  `Build` and `Unload`. The terms
  Manifest, Sequential run and Parallel run are sharper.
- `docs/adr/0001` to `0004`: the standalone glue, the hardcoded topologies,
  the f32 load, and the compile-time list with the run-time pick.
- A ticket in burn-stack for three terms of the vocab network that no glossary
  defines: `../burn-stack/.scratch/vocab-network-terms/issues/01-vocab-network-terms.md`.

Five tickets do the rest, in this order. Each ticket is blocked by the ticket
before it.

1. A check script, an access allowlist hook and a box list.
2. The renames that the glossary asks for. Breaking changes are permitted.
3. A short `CLAUDE.md`, in the shape of the sibling files.
4. A prose sweep of the code docs, with `#![warn(missing_docs)]`.
5. A short `README.md`.

Tickets 06 and 07 are for later, and need triage.

## User Stories

1. As an agent, I want a short `CLAUDE.md` that points to each kind of knowledge, so that I load only what I need.
2. As an agent, I want the same rules as in the sibling repos, so that I do not learn a second set of rules.
3. As the maintainer, I want one tracked check script, so that a clean clone and the CI run the same checks.
4. As the maintainer, I want the check to build the four checkpoints together, so that it makes few build artifacts.
5. As the maintainer, I want the CI to run the check before the deploy, so that a broken build never goes live.
6. As the maintainer, I want an access allowlist hook, so that an agent cannot read or write a path outside the list without my approval.
7. As an agent, I want each identifier to use the glossary term, so that a search for a term finds the code.
8. As a reader, I want one meaning for each word ("checkpoint", "cache", "chunk"), so that I do not confuse two concepts.
9. As the maintainer, I want a feature for each front-end, so that the feature list shows the three front-ends.
10. As the author of a future web front-end, I want docs without Yew words, so that the vocabulary stays when the implementation changes.
11. As a user, I want a `README.md` that claims only what the app does, so that I trust it.
12. As a user, I want the example outputs to stay, so that I can compare my run with them.
13. As a reader of the code, I want module headers without changelog prose, so that each header describes the code as it is now.
14. As the maintainer, I want a doc on each public item, so that the docs stay complete as the code changes.

## Implementation Decisions

### Ticket 01: the check script, the hook and the box list

- `scripts/check.sh` runs these steps. Each cargo step fails on a warning.
  1. `cargo check`, `cargo test -- --test-threads=1` and `cargo doc --no-deps`,
     all with one native feature set: the native front-end, `backend-flex`,
     `backend-simd` and the four checkpoint features.
  2. `cargo +nightly check --target wasm32-unknown-unknown` for the browser
     page and for the browser console, each with the four checkpoint features.
  3. The host-path step and the prose ratchet of
     `../burn-mamba/scripts/check.sh`. Both skip when `CI` is set.
- The check runs no cuda build, no other backend, no bench and no `#[ignore]`
  test. A build with one checkpoint feature is for a debug session only.
- The deploy workflow runs the check before the four bundle builds.
- The hook is a copy of `../burn-mamba/.claude/hooks/access-allowlist.py`, with
  the roots of this repo:
  - Read and write: this repo and the auto-memory.
  - Read only: `../{burn-mamba,burn-stack,burn,cubecl,cubek}`,
    `../{hf-hub,candle,tokenizers}`, `../py/`, `../papers/`, the cargo
    registry and git checkouts, the rustup toolchains, `~/.claude`,
    `~/.cache/huggingface/hub` and `/tmp`.
  - The approval file is `.claude/access-approved.txt`.
- Git ignores `.claude/`, so the hook and its settings stay local.
- `docs/box.tsv` lists the directories next to this repo that the docs name, in
  the format of the sibling lists. `../burn-stack/scripts/setup-box.sh` reads
  the list of each repo in the root.
- `docs/agents/code-review.md` and `docs/agents/testing.md` name the check
  script, not the bacon jobs. `bacon.toml` stays a local file.

### Ticket 02: the renames

- **The rule.** Each identifier, log line, message and page label that uses a
  word of an `_Avoid_` line in the avoided sense changes to the glossary term.
- **Breaking changes are permitted.** The crate is like a binary, and each
  deploy bundles it. A user can lose the hub cache of the browser.
- **Frozen**: the tensor names of the checkpoints, the HF repo ids, file paths
  and revisions (`refs/pr/1`), and the layout of the native hub cache. Other HF
  tools share that layout.
- The renames include these. The rule decides each other case.

  | Now | After |
  |---|---|
  | `MAMBA_MODEL` | `MAMBA_CHECKPOINT` |
  | `ModelSpec`, `hf::MODELS` | `CheckpointSpec`, `hf::CHECKPOINTS` |
  | `select_model`, `native::models()` | `select_checkpoint`, `build_wrapper()` |
  | `store_load::Checkpoint` | `WeightsSource` |
  | `load_mamba` | `build_network` |
  | `hub::wasm::CHUNK_SIZE` | `DOWNLOAD_CHUNK_LEN` |
  | `hf::mamba3_*::CHUNK_SIZE` | `CHUNK_LEN` |
  | `ModelData`, `ModelSelection`, `Msg::*ModelData*` | `Asset`, `AssetKind`, `Msg::*Asset*` |
  | `wasm/console_ui.rs` | `wasm/console.rs` |
  | "Connected to cache" | "Connected to the hub cache" |

- **Checkpoint ids** name the checkpoint, not the family: `mamba1-130m`,
  `mamba2-130m`, `mamba3-siso-187m` and `mamba3-mimo-187m`. Each id is also its
  checkpoint feature, its `frontend/` directory, its live URL path, its
  `MAMBA_CHECKPOINT` value and its bench case name.
- **Family features**: `mamba1`, `mamba2` and `mamba3` add only the blocks. Each
  checkpoint feature implies its family feature. An entry point without a
  checkpoint feature is a `compile_error!`.
- **Front-end features**: `native`, `console` and `yew`. A wasm build needs
  exactly one of `console` and `yew`, else a `compile_error!`. A future web
  front-end has its own feature.
- **IndexedDB**: the database and the object store take names in our words, for
  example `burn-mamba-example` and `download-chunks`. The zero-padded offsets of
  the keys stay, because they keep the key order numeric.
- Each doc that names a renamed item follows in the same change. This includes
  `CLAUDE.md`, `README.md`, `GLOSSARY.md`, the ADRs and `docs/agents/`. It also
  includes the module headers, the headers of `bench.sh` and `kernels.sh`, and
  the deploy workflow. In `CLAUDE.md` and `README.md`, change only the names.
  Tickets 03 and 05 rewrite these files.
- `bench.md` and `kernels.md` are generated. Do not edit them. The next bench
  run of the user writes the new case names.

### Ticket 03: the `CLAUDE.md` redo

- Before you write, load the `mattpocock-skills:writing-for-agents` and
  `asd-ste100` skills. Read the three glossaries and `docs/adr/`.
- The sections follow `../burn-mamba/CLAUDE.md`:
  1. a summary of about 3 lines,
  2. Build & Test Commands, with the axes front-end × checkpoint × backend,
  3. Rules: Code, Docs and Custom Commands,
  4. Where to look,
  5. Agent skills, in its current `###` form,
  6. File Map.
- `CLAUDE.md` has about 200 lines or fewer. The File Map has 40 lines or fewer.
  It has one line for each directory, and one line for each file whose name
  does not tell what it holds. It shows `common/` apart from `native/` and
  `wasm/`. `wasm/yew_ui/` is one line: the browser page, built with Yew.
- Copy these sibling rules:
  - always load `asd-ste100`,
  - the waiting rule, one agent at a time, and the agent size,
  - `tmp/` with `.discard` tags,
  - memory, commits and pushes,
  - the `dcg` hook,
  - the access allowlist, with the roots of ticket 01.
- The path rule: "Write each path in a repo file, a commit message or a PR text
  relative to the repo. `scripts/check.sh` finds a path of the box." It
  replaces an auto-memory with the same rule.
- The Code rules add: document each new public item, and `cargo check` and
  `cargo doc` give no warnings.
- Do not copy the Notation table, the test-device rule or a
  `docs/source-terms.md`. The checkpoint keys use the words of the official
  code, and `../burn-mamba/docs/source-terms.md` maps them.
- No backend verdicts. flex and ndarray are the reference backends. Run
  agreement against a reference backend checks another backend.
- Move the Architecture and Key Design Decisions sections to their sources.
  Delete a paragraph only after you find its content in a module header, a
  glossary or an ADR. If you find it in none, move it to the module header. If
  it holds a new term or a new decision, add a note under `## Comments`. Do not
  edit the glossary or the ADRs.

  | Section now | Destination |
  |---|---|
  | Load path | the header of `store_load.rs` |
  | Two run modes | the glossary, the docs of the run wrapper |
  | `Verifying a backend` | the glossary, `docs/agents/testing.md` |
  | Device & precision | ADR-0003, the header of `common/mod.rs` |
  | Browser asset lifecycle | the glossary, the header of `wasm/yew_ui/mod.rs` |
  | Glue only | the summary, a Code rule |
  | No candle / tokenizers / hf-hub | ADR-0001 |
  | Deps are git-rev-pinned | `docs/agents/issue-tracker.md` |
  | Configs are hardcoded | ADR-0002, the header of the `hf` module |
  | Mamba-3 needs the per-layer `GatedMlp` | out of date: delete. burn-stack calls it the feed-forward. |
  | Model choice is runtime data | ADR-0004, the header of the `hf` module |
  | `frontend/*/pkg/` is generated | a File Map line |
  | CI = deploy | a Code rule |

- The header of the `hf` module holds the checklist to add a checkpoint. The
  checklist names its spec, its list entry, its features, its manifest, its
  `frontend/` directory and its deploy step.

### Ticket 04: the prose sweep and `missing_docs`

- Load `asd-ste100`. Doc comments and comments are STE-flavored. Log, panic and
  error strings are Strict. Before you change a string, search the tests for a
  match of a part of it.
- The scope is the comments and docs of each `.rs` file, and the headers of
  `bench.sh` and `kernels.sh`. It also includes the comments of `Cargo.toml`
  and of the deploy workflow.
- Fix the docs that are out of date:
  - `wasm/mod.rs`: the TODO to configure the backend. Each entry point already
    does it.
  - `run_parallel`: the doc names a `mamba2_chunk_size` parameter that does not
    exist.
  - The header of `store_load.rs` says "both checkpoints". There are four, and
    the dtypes are f32, f16 and bf16.
  - The header of `benches/model.rs` says "three configurations, two builds".
    There are four configurations and three builds. It also names a "dfdx
    sibling".
  - The header of `hub/mod.rs` has changelog prose about the `hf-hub` fork.
  - The native log says "inference-friendly" and "training-friendly".
  - Each "mixer" in our own words (burn-stack avoids it), and "pathway" for
    Mamba-1.
- Each page label "TODO" changes to a help text that tells what the control
  does, or that it is not available.
- Delete dead commented-out code and comments that ask a question ("idk").
- Add `#![warn(missing_docs)]` and document each public item. The check script
  then stays free of warnings.
- No change of logic.

### Ticket 05: the `README.md` redo

- Load `asd-ste100`. The README is STE-flavored.
- The sections, in order:
  1. an intro and the live demos,
  2. a quick start,
  3. the build: the features, native, the browser,
  4. the backends,
  5. the benchmarks,
  6. how it works,
  7. the credits.
- The backends section has no verdict table. It lists the `backend-*`
  features, names flex and ndarray as the reference backends, and tells how to
  check another backend with run agreement.
- The benchmarks section is short. It links `bench.md`, `kernels.md` and the
  headers of the two scripts.
- How it works is short. It links the module headers and the ADRs.
- The example outputs stay as they are. They are correct at HEAD.
- Delete the two false claims. The "Download to file" link of the page
  downloads the file from the hub, not from the hub cache.
- Describe the browser page with the glossary terms. Name Yew only where a
  command needs the feature.
- The Tests and Development sections go. `CLAUDE.md` and
  `docs/agents/testing.md` hold them.

## Testing Decisions

- These tickets change docs and names, not logic. The seams of
  `docs/agents/testing.md` stay as they are.
- From ticket 01, each ticket passes `scripts/check.sh`.
- Ticket 02 changes names that the browser reads at run time (the IndexedDB
  names, the `frontend/` directories). Only a run in the browser shows that they
  work, and only the user does that run. The deploy workflow builds the four
  bundles.
- The prose ratchet of the check script covers each changed `.rs` and `.md`
  file.
- No bench gate. Ticket 02 changes the case names, but not the timed code.

## Out of Scope

- Ticket 04 of `bump-burn-mamba-main` (the new backend features). The docs
  describe the backends of today.
- The ticket of `bench-host-info`.
- Any change in `../burn-mamba`. The burn-stack ticket above is the only
  change outside this repo.
- A change of logic, other than the renames and the features of ticket 02.

## Further Notes

- Ticket 06 moves the parts of `wasm/yew_ui/model.rs` that do not need Yew
  into a module outside `yew_ui/`. These parts are the asset state, the chunk
  plan and the wrapper builder. A future web front-end can then use them.
- Ticket 07 builds the deploy once for the four checkpoints, if the browser page
  can pick a checkpoint at run time. Then the deploy does not compile each
  family alone (ADR-0004).
- The `bump-burn-mamba-main` spec put each edit of `CLAUDE.md` and `README.md`
  out of its scope, for a later overhaul session. This spec is that overhaul.
