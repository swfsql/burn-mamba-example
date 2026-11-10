# 03: A short CLAUDE.md, in the shape of the sibling files

Status: ready-for-agent
Blocked by: 02
Spec: `.scratch/docs-overhaul/spec.md`

**What to build:** An agent that starts a session reads a short `CLAUDE.md`
that points to where each kind of knowledge lives. Its sections and its rules
follow `../burn-mamba/CLAUDE.md`. The content of the Architecture and Key
Design Decisions sections moves to the module headers, the glossaries and the
ADRs. The spec gives each decision in full, with the destination of each
section (Implementation Decisions, ticket 03).

## Acceptance criteria

- [ ] `CLAUDE.md` has about 200 lines or fewer. Its File Map has 40 lines or
      fewer.
- [ ] The sections are the sections of the spec, in that order.
- [ ] The rules of the spec are in `CLAUDE.md`, with the path rule of the spec
      and the access allowlist with the roots of ticket 01.
- [ ] `CLAUDE.md` has no backend verdict.
- [ ] Each deleted paragraph has its content in a module header, a glossary or
      an ADR, or is listed under `## Comments`.
- [ ] The header of the `hf` module holds the checklist to add a checkpoint.
- [ ] No doc links to a `CLAUDE.md` anchor that is gone.
- [ ] `ste-lint.py` passes on `CLAUDE.md`, and `scripts/check.sh` passes.

## Comments
