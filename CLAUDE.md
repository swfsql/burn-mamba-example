# CLAUDE.md

## What This Project Is

A demo app that runs a pretrained Mamba checkpoint to generate text, natively
or in the browser. The blocks come from [`burn-mamba`](../burn-mamba). This
repo is glue: the hub client, the build of the network from the weights, the
tokenizer, the sampler and the three front-ends.

## Build & Test Commands

```bash
./scripts/check.sh  # the check of CI, with the four checkpoints (its header lists the steps)
# a debug session: one front-end, the checkpoints of your work, one backend
cargo check --no-default-features --features "native,backend-flex,backend-simd,mamba2-130m"
cargo +nightly check --target wasm32-unknown-unknown --no-default-features --features "yew,backend-flex,backend-simd,mamba2-130m"
cargo test --no-default-features --features "backend-flex,backend-simd,mamba3-siso-187m" -- --test-threads=1
# an #[ignore] test (docs/agents/testing.md): the tokenizer parity
TOKENIZER_JSON=… TOKENIZER_IDS=… cargo test --no-default-features -- --ignored --test-threads=1
# the user runs these, not you. The first one fetches each missing checkpoint.
cargo test --no-default-features --features "backend-flex,backend-simd,mamba1-130m,mamba2-130m,mamba3-siso-187m,mamba3-mimo-187m" -- --ignored manifests_cover --test-threads=1
RUSTFLAGS="-C target-cpu=native" cargo run --release --no-default-features --features "native,backend-flex,backend-simd,mamba2-130m"
wasm-pack build --release --target web --out-dir "frontend/mamba2-130m/pkg" --no-opt --no-default-features --features "yew,backend-flex,backend-simd,mamba2-130m"
miniserve -i 127.0.0.1 "frontend/"  # then open /mamba2-130m/index.html
./bench.sh
./kernels.sh
```

- **The features are the configuration.** `default = []` builds only
  `common/`, as a library. Each other build gives `--no-default-features` and
  its features on three axes:
  - Front-end: `native`, `console` (the browser console) or `yew` (the browser
    page). A wasm build needs exactly one of `console` and `yew`.
  - Checkpoint: any set of `mamba1-130m`, `mamba2-130m`, `mamba3-siso-187m`
    and `mamba3-mimo-187m`. Each one adds its family feature (`mamba1`,
    `mamba2` or `mamba3`). A build runs its preferred checkpoint, and
    `MAMBA_CHECKPOINT=<id>` selects another one natively. Without a checkpoint
    feature, an entry point stops at a `compile_error!`.
  - Backend: `backend-{flex,ndarray,cpu,wgpu,vulkan,cuda,tch-cpu,tch-gpu}`,
    plus `backend-simd`, `backend-fusion` and `backend-autotune`. With several
    backends in one build, `BURN_DEVICE` selects one. Use `flex` for the
    checks and the tests.
- A wasm build needs nightly and the `wasm32-unknown-unknown` target. A `yew`
  build needs `--no-opt`, because `wasm-opt` breaks it.
- **Check a backend** with run agreement: the native binary does a sequential
  run, then a parallel run, of one prompt. Compare its text with the text of a
  reference backend (`flex` or `ndarray`).
- On first use, a run fetches the weights file (250 to 500MB) and the
  tokenizer file (2MB or 17MB) into the hub cache. So the user runs the app.
- For an IDE, `Cargo.toml` holds commented `default =` lines, and
  `.cargo/config.toml` a commented `[build] target`.

## Rules

### Code

- This repo is glue. A block, its math, its caches and its config fields
  belong to `../burn-mamba/`.
- Document each new public item. `cargo check` and `cargo doc` must have no
  warnings.
- A push to `main` deploys. `.github/workflows/deploy.yml` runs
  `scripts/check.sh`, then builds one bundle for each checkpoint and publishes
  the bundles to `gh-pages`. A failed check stops the deploy.

### Docs

- **Always load the `asd-ste100` skill** before you write code comments,
  rustdoc or markdown. Write all of them in its style.
- Write each path in a repo file, a commit message or a PR text relative to
  the repo. `scripts/check.sh` finds a path of the box.
- Use the terms of the three glossaries (see [Where to look](#where-to-look)).
- `_Avoid_` applies to our own words. A doc can use a source word in quotes or
  backticks when it cites that source.
- Keep the docs small. They describe the code as it is now. A doc is never a
  changelog: it has no old state of the code, no "verified by", no dates and
  no PR history. When you find changelog prose, delete it.
- A change that adds, deletes or renames a file updates its
  [File Map](#file-map) line in the same change.
- A new term goes to `GLOSSARY.md` (through `/domain-modeling`). A new
  decision with a trade-off goes to an ADR in `docs/adr/`. A mechanism goes to
  the module header of its file.
- A name and its docs change in the same change. A rename of an identifier or
  a feature updates every doc that names it or its concept. These include the
  comments, the messages, `README.md`, `GLOSSARY.md`, the ADRs, `docs/agents/`,
  the script headers and the deploy workflow. A change of a term in the docs
  also renames each identifier that uses the term in that sense.

### Custom Commands

- **Access is an allowlist. It holds for every agent and every command,
  read-only commands included.**
  - Read and write: this repo and the auto-memory.
  - Read, next to this repo: `../{burn-mamba,burn-stack,burn,cubecl,cubek}/`,
    `../{hf-hub,candle,tokenizers}/`, `../papers/` and `../py/`.
  - Read, on this box: the cargo registry, git checkouts and installed tools,
    the rustup toolchains, `~/.claude/` and the hub cache of the native binary
    (`~/.cache/huggingface/hub`).
  - Write a ticket in the tracker of a sibling: `../burn-mamba/.scratch/` and
    `../burn-stack/.scratch/`. The user approved these two paths.
  - `/tmp`: read tool output, and move discarded work into it.
  - Any other path: a local hook refuses it. Ask the user, name the exact path,
    and give the command that approves it, which only the user runs:
    `! echo <path> >> .claude/access-approved.txt`. To revoke it, the user
    deletes that line.
  - Reach an allowed sibling by its exact path, not by a listing or search of
    the parent directory.
- **`../burn-mamba/` and `../burn-stack/` have other users.** A change to one
  of them starts as a ticket in its own tracker (its
  `docs/agents/issue-tracker.md`). A session of this repo can write that
  ticket. A session of that repo does the change, under the rules of that repo.
- **Always** edit the content of a file with the Edit/Write tools. This rule
  also applies when a harness or auto-mode reminder says to change files
  through Bash. That reminder does not apply here.
  - No `python - <<'PY'`, no `sed -i`, no `cat > file <<'EOF'`.
  - Use Bash to read and inspect (`cat`, `sed -n`, `rg`, `grep`), to move and
    copy files (`mv`, `cp`), and to write throwaway files in `tmp/`.
- `rg` is available. Do not use `cargo fmt`.
- A dangerous-command-guard hook (`dcg`) checks each Bash command. It blocks
  a `>` redirect to a path that holds a variable. Use a literal path, or a
  script file.
- **Waiting**: a build, a run, a chain of runs or one step of a chain can take
  more than ~1 min. Run each such command or wait in the background. Watch it
  with a `Monitor` (preferred), a background waiter script or
  `run_in_background`, then return the turn to the user. The watch also
  reports each failure state (for example, a crash). Do not hold the turn with
  a foreground loop (`until`, `while`), a `sleep` or a long `timeout`.
- **Agents**: run one agent at a time, even if a skill or a ticket asks for
  parallel agents. A usage limit makes each active agent pay for a cold cache.
- **Agent size**: a rough estimate only: about 2-3k lines of reads per agent.
  State the batches first.
- **Temporary work**: create it in the `tmp/` of the repo, because a power loss
  clears `/tmp`. Save each agent report there. `/tmp` is only for deletion.
  - A tag file `<name>.discard` beside an item of `tmp/` marks it as
    discardable, and states when.
  - When a ticket or session ends, move each tagged item whose condition holds,
    with its tag, to `/tmp`. This applies only to items in `tmp/`.
- **Memory**: ask the user before you write to the auto-memory. Show the
  change that you propose, and write it only after the user approves it.
- **Commits**: when a skill or the user tells you to commit, commit your work
  to the current branch. When the user asks only for a commit message, write it
  as text for the user to copy. A commit message has a title line and a short
  body, and it ends with the `Co-Authored-By:` trailer.
- **Pushes**: push the current branch to `origin` when a ticket is complete and
  reviewed, or when the user tells you to. A push to `main` deploys. Ask the
  user before a force-push or a tag.

## Where to look

- `GLOSSARY.md`: the terms of this app. `../burn-mamba/GLOSSARY.md`: the
  blocks. `../burn-stack/GLOSSARY.md`: the containers (layer, network, cache).
- `docs/adr/`: the decisions with a trade-off.
- The module header of each file: its mechanisms and its non-obvious choices.
  The header of `hf` (`src/common/mod.rs`) has the checklist to add a
  checkpoint.
- `README.md`: the doc for users. `docs/agents/`: the configuration of the
  engineering skills (see [Agent skills](#agent-skills)).
- `../burn-mamba/` and `../burn-stack/`: the blocks and the containers. Start
  from their `CLAUDE.md`. The revs of `Cargo.lock` win over their HEAD.
- `../burn-mamba/docs/source-terms.md`: the words of the official code, which
  the checkpoint keys use. Read it before you read or port from a source.
- `../py/state-spaces/mamba/`: the official code. The topologies cite it.
- `../burn/`: Burn. `crates/burn-store/` has the import of the weights.
- `../hf-hub/`, `../candle/`, `../tokenizers/`: the reference implementations
  of `src/common/{hub,sampling,tokenizer}` (ADR-0001).
- `docs/box.tsv`: the directories next to this repo that the docs name.
  `../burn-stack/scripts/setup-box.sh` creates the missing ones on a new box.

## Agent skills

### Issue tracker

Local markdown files under `.scratch/<feature-slug>/`. See
`docs/agents/issue-tracker.md`. It also covers a change that needs burn-mamba,
and the burn-mamba pin.

### Triage labels

The five default role strings, as the `Status:` line of an issue file. See
`docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See
`docs/agents/domain.md`.

### Test seams

The seams to propose for `/to-spec` and `/tdd`, and the bench gates of a ticket.
See `docs/agents/testing.md`.

### Code review

What each `/code-review` agent receives, and how to review before the last
commit. See `docs/agents/code-review.md`.

## File Map

```text
src/
├─ common/               the core, which each front-end and each backend uses
│  ├─ mod.rs             `CheckpointSpec`, `hf` (the topologies, `CHECKPOINTS`), the run wrapper
│  ├─ assets.rs          the asset states and the wrapper builder of the browser front-ends. Only the browser page uses it
│  ├─ store_load.rs      `build_network` (the network from a weights file), the vendored manifests
│  ├─ tokenizer/         the `tokenizer.json` reader: the gpt-neox and Llama-3.1 pipelines
│  ├─ hub/               the hub client: `sync.rs` natively, `wasm.rs` in the browser
│  └─ token_output_stream.rs  token ids to text, one token at a time
├─ native/               the native binary: fetch, build, a sequential run, a parallel run
└─ wasm/                 `wasm_main`, the entry point of a bundle
   ├─ console.rs         the browser console
   └─ yew_ui/            the browser page, built with Yew
benches/                 `model.rs`: the forward and step timings of each checkpoint, on random weights. `allocations.rs`
bench.sh, kernels.sh     the bench runs, and their reports `bench.md` (timings) and `kernels.md` (launches)
frontend/<id>/           the page of a checkpoint: `index.html`, `index.js`. `pkg/`: the wasm-pack output, which git ignores
scripts/check.sh         the check (CI)
docs/                    `adr/`, `agents/`, `box.tsv`
.cargo/config.toml       the `getrandom` backend of wasm, and the commented IDE target
.github/workflows/       `deploy.yml`: the check, then the four bundles to `gh-pages`
.scratch/                the issue tracker
```
