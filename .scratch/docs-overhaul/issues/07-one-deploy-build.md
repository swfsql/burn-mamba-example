# 07: One deploy build for the four checkpoints

Status: needs-triage
Blocked by: 06
Spec: `.scratch/docs-overhaul/spec.md`

**What to build:** The deploy workflow compiles the browser bundle once, with
the four checkpoint features. The browser page lets the user pick a checkpoint
at run time. Then the CI makes fewer builds.

This needs a browser page that can pick a checkpoint at run time. The Yew page
of today runs only the preferred checkpoint. A future web front-end can add the
pick.

## Questions for the triage

- ADR-0004 says that the deploy compiles each family alone. After this ticket,
  no routine build does. Does a check of each family alone move to
  `scripts/check.sh`, or does the maintainer accept the gap?
- The live URLs: one page with a pick, or one page for each checkpoint over
  one bundle?

## Comments
