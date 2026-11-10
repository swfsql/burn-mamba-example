# Domain Docs

How the engineering skills read the domain documentation of this repo when they explore the code.

## Before exploring, read these

- **`GLOSSARY.md`** at the repo root.
- **`../burn-mamba/GLOSSARY.md`**: it defines the terms of the blocks (plant, head, SSD path, MIMO rank, …).
- **`../burn-stack/GLOSSARY.md`**: it defines the terms of the containers (block, layer, cache, row, segment, …).
- **`docs/adr/`**: read the ADRs that touch the area of your work.

If `GLOSSARY.md` or `docs/adr/` does not exist, **continue silently**. Do not report that it is missing. Do not propose to create it in advance. The `/domain-modeling` skill (through `/grill-with-docs` and `/improve-codebase-architecture`) creates these files when a term or a decision is resolved.

The two sibling glossaries describe the HEAD of their repos. The burn-mamba rev that `Cargo.toml` pins can be older, so the pinned API can use older names. Use the glossary terms in new prose. Keep each name of the pinned API as it is. The pin wins for what builds.

## Before you read or port from a source

Read **`../burn-mamba/docs/source-terms.md`** before you read the official Mamba code or port from it. It maps the words of the Mamba papers, the official code and mamba3-minimal to the burn-mamba terms. The checkpoint keys that `src/common/store_load.rs` remaps (`A_log`, `dt_bias`, `B_norm`, …) use the words of the official code.

## File structure

This repo is single-context. There is no `GLOSSARY-MAP.md`.

```
/
├── GLOSSARY.md
├── docs/adr/
│   └── 0001-<decision-slug>.md
└── src/
```

## Use the glossary's vocabulary

When your output names a domain concept, use the term that a glossary defines. This applies to issue titles, refactor proposals, hypotheses and test names. Do not change to a synonym that the glossary tells you to avoid.

If no glossary has the concept, that is a signal. Either you invent language that the project does not use (think again), or the glossary has a real gap (record it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an ADR, say so explicitly. Do not silently override it:

> _Contradicts ADR-<NNNN> (<title>), but worth reopening because…_
