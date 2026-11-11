# 05: A short README.md

Status: resolved
Blocked by: 04
Spec: `.scratch/docs-overhaul/spec.md`

**What to build:** A user reads a short `README.md` that claims only what the
app does. It describes the browser page with the glossary terms, and it has
no backend verdict table. The spec gives the sections and each decision
(Implementation Decisions, ticket 05).

## Acceptance criteria

- [x] The sections are the sections of the spec, in that order.
- [x] The backends section has no verdict table. It names flex and ndarray as
      the reference backends, and tells how to check another backend.
- [x] The two false claims of the spec are gone.
- [x] The example outputs are the same text as before.
- [x] The README has no Tests or Development section.
- [x] Each link and each anchor of the README resolves.
- [x] `scripts/check.sh` passes, with its prose ratchet.

## Comments

- **The result.** The README has 1 hard violation of `ste-lint.py --enable
  history`. `origin/main` has 38. The one that stays is a synonym rotation of
  `start` (in the heading `Quick start`) and `launches` (the kernel launches).
  These words do not name the same action, and `launches` is the term of
  `kernels.sh` and `kernels.md`.
- **The example outputs** are an `###` section under `## Quick start`. Their
  text is the same as before. Their lead-in names no backend other than a
  reference backend, because the README has no backend verdicts.
- **Not in a doc now.** The old table said that `backend-cpu` "may
  stack-overflow". The spec deletes the verdict table, and no other doc holds
  this note.
- **Two docs outside this ticket say less than the code does:**
  - The `Run agreement` entry of `GLOSSARY.md` says "the same text". The
    parallel run of the native binary gives 20 tokens and the sequential run
    gives 80. So the README says that the parallel text is the same as the
    first part of the sequential text. A sharper entry needs
    `/domain-modeling`.
  - The `Asset` entry of `GLOSSARY.md` and the header of
    `src/wasm/yew_ui/mod.rs` say that the page unloads each asset
    separately. When the page has built the two assets, the unload of one
    asset also unloads the other (`Msg::AssetUnload`). The README says this.
