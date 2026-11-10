# Hardcoded topologies, not a `config.json` reader

The topology of each checkpoint is Rust code in `src/common/mod.rs`. The code
copies the values of the `config.json` of the checkpoint, and the app does not
fetch or read that file. This likely helps the speed of the app. There is no
hard reason against a `config.json` reader, so this decision can change.

## Consequences

- Each constant must agree with its checkpoint. The checkpoint-against-module
  tests find a wrong shape. A wrong value that keeps the shapes (for example, a
  rotation knob) needs its own test, as the Mamba-3 knobs have.
- A `config.json` reader would also need code. The fields of the network that
  `config.json` does not hold, such as the SSD path, stay in code.
