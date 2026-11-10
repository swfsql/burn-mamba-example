# 02: The renames of the glossary

Status: ready-for-agent
Blocked by: 01
Spec: `.scratch/docs-overhaul/spec.md`

**What to build:** Each identifier, log line, message and page label uses the
glossary term for its concept. The checkpoint ids name checkpoints, the family
features add only the blocks, and each front-end has a feature. Breaking
changes are permitted. The spec gives the rule, the rename list, the frozen
names and the new features (Implementation Decisions, ticket 02).

## Acceptance criteria

- [ ] No identifier, log line, message or page label uses a word of an
      `_Avoid_` line of `GLOSSARY.md` in the avoided sense. A source word in
      quotes or backticks that cites its source is permitted.
- [ ] Each rename of the table of the spec is done.
- [ ] The checkpoint ids are `mamba1-130m`, `mamba2-130m`, `mamba3-siso-187m`
      and `mamba3-mimo-187m`. Each id is its checkpoint feature, its
      `frontend/` directory, its `MAMBA_CHECKPOINT` value and its bench case
      name.
- [ ] The features `mamba1`, `mamba2` and `mamba3` add only the blocks. A
      native or wasm build without a checkpoint feature is a `compile_error!`.
- [ ] A wasm build needs exactly one of `console` and `yew`, else a
      `compile_error!`.
- [ ] The IndexedDB database and its object store have names in our words. The
      keys keep their zero-padded offsets.
- [ ] The frozen names of the spec do not change.
- [ ] Each doc that names a renamed item follows in the same change. In
      `CLAUDE.md` and `README.md`, only the names change.
- [ ] The deploy workflow, `bench.sh`, `kernels.sh` and `scripts/check.sh` use
      the new feature names and ids. `bench.md` and `kernels.md` do not change.
- [ ] `scripts/check.sh` passes.

## Notes

- Only a run in the browser shows that the new IndexedDB names and the new
  `frontend/` directories work. List this run under `## Comments`, for the
  user.

## Comments
