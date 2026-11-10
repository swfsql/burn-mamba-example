# Standalone glue: no `candle`, `tokenizers` or `hf-hub`

The app has its own code to read `tokenizer.json`, to fetch from the hub and
to sample tokens (`src/common/`). It does not use the `tokenizers`, `hf-hub` or
`candle` crates. The app used few of their definitions, and `candle` was a
large dependency that the app did not need. So the app has fewer dependencies.

## Consequences

- A new tokenizer pipeline or a new hub feature needs new code in this repo.
- The tokenizer rejects each pipeline outside the two that it supports, at load
  time. A near-miss would give token ids that look correct but are wrong.
- Before you add a dependency, look for a Burn crate that does the work.
