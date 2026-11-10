# A compile-time list of checkpoints, a run-time pick

Each checkpoint feature adds one entry to a list of checkpoints, and the app
picks one entry at run time. So one build can hold several checkpoints, and the
bench and the checks need fewer builds. A build with one checkpoint feature
compiles only the family of that checkpoint. burn-mamba supports each family
alone, and the app keeps that support.

## Consequences

- Code that depends on the checkpoint reads the spec of the checkpoint, not a
  `#[cfg]`.
- Only a match on the family enum of burn-mamba stays per feature. It has no
  catch-all arm, because with one family the arm is unreachable.
- `scripts/check.sh` builds the four checkpoints together. The deploy workflow
  builds one bundle for each checkpoint, so it compiles each family alone.
