# Each weight loads as f32

The load casts each tensor of a checkpoint to f32. So one code path serves the
f32, f16 and bf16 checkpoints. The checkpoints from Mamba-2 onwards expect
mixed precision, and the app does not explore it yet. This is a best-effort
choice, and it can change.

## Consequences

An f16 or bf16 checkpoint takes about twice its file size in memory, in the
browser too.
