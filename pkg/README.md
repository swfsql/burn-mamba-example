# burn-mamba-example

Click [here](https://swfsql.github.io/burn-mamba-example/) to run a [130m model](https://huggingface.co/state-spaces/mamba-130m/) in your browser.

### Information

Adapted from [huggingface/candle/mamba-minimal](https://github.com/huggingface/candle/blob/fd7c8565646039e35925b8730d27ddad195d7e73/candle-examples/examples/mamba-minimal/), this utilizes [burn-mamba](https://github.com/swfsql/burn-mamba) block definitions.

### Building

##### Native (Console)
```bash
RUSTFLAGS="-C target-cpu=native"
cargo run --release --no-default-features --features "native,ndarray"
```
Notes:
- This will automatically download model weights, load and run them, first in cacheless mode and then in cached mode.
- You can swap `ndarray` for `wgpu`, but as of burn `0.17.0` this is having numerical error for cacheless mode.

##### WASM

Using [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/) and serving with [miniserve](https://github.com/svenstaro/miniserve/?tab=readme-ov-file#how-to-install).

#### Web (Console Log)

```bash
wasm-pack build --release --target web --out-dir "frontend/pkg" \
  --no-default-features --features "ndarray"
miniserve -i 127.0.0.1 "frontend/"
```
Then open the page at [http://127.0.0.1:8080/index.html](http://127.0.0.1:8080/index.html) and open the console logs.
Note: This will automatically download model weights, load and run them, first in cacheless mode and then in cached mode, similarly to the native console one.

#### Web (Yew UI)

```bash
wasm-pack build --release --target web --out-dir "frontend/pkg" --no-opt \
  --no-default-features --features "ndarray,yew"
miniserve -i 127.0.0.1 "frontend/"
```
Then open the page at [http://127.0.0.1:8080/index.html](http://127.0.0.1:8080/index.html).
Note: This won't download anything by default, and you must click buttons to download, load and run the model - which is run in cached mode.

### Dev

For a better IDE development, you may want to change the default features/settings on `Cargo.toml` and `.cargo/config.toml`.
