# 06: Split the asset logic from the Yew page

Status: needs-triage
Blocked by: None (can start immediately)
Spec: `.scratch/docs-overhaul/spec.md`

**What to build:** The parts of `src/wasm/yew_ui/model.rs` that do not need Yew
move into a module outside `yew_ui/`. These parts are:

- the state of each asset in the hub cache and in memory,
- the plan of the download chunks,
- the wrapper builder.

A future web front-end can then use them. The Yew page keeps only its messages,
its reducer and its view.

## Questions for the triage

- The name and the place of the new module, for example `src/wasm/assets.rs`.
- Is the module free of `web-sys` too, so that a test can run it natively?

## Comments
