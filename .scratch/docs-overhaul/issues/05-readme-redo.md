# 05: A short README.md

Status: ready-for-agent
Blocked by: 04
Spec: `.scratch/docs-overhaul/spec.md`

**What to build:** A user reads a short `README.md` that claims only what the
app does. It describes the browser page with the glossary terms, and it has
no backend verdict table. The spec gives the sections and each decision
(Implementation Decisions, ticket 05).

## Acceptance criteria

- [ ] The sections are the sections of the spec, in that order.
- [ ] The backends section has no verdict table. It names flex and ndarray as
      the reference backends, and tells how to check another backend.
- [ ] The two false claims of the spec are gone.
- [ ] The example outputs are the same text as before.
- [ ] The README has no Tests or Development section.
- [ ] Each link and each anchor of the README resolves.
- [ ] `scripts/check.sh` passes, with its prose ratchet.

## Comments
