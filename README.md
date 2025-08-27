# burn-mamba-example

Click [here](https://swfsql.github.io/burn-mamba-example/mamba1) to run a [130m Mamba-1 model](https://huggingface.co/state-spaces/mamba-130m/) or [here](https://swfsql.github.io/burn-mamba-example/mamba2) to run a [130m Mamba-2 model](https://huggingface.co/state-spaces/mamba2-130m/) in your browser.

### Information

Mamba-1 adapted from [huggingface/candle/mamba-minimal](https://github.com/huggingface/candle/blob/fd7c8565646039e35925b8730d27ddad195d7e73/candle-examples/examples/mamba-minimal/) and Mamba-2 adapted from [mamba2-minimal](https://github.com/tommyip/mamba2-minimal). This utilizes [burn-mamba](https://github.com/swfsql/burn-mamba) block definitions.

### Features

- "default" or "empty": nothing is enabled and the "common" mod is exported as a library.
- Target:
  - `native`: local executable.
  - "empty": web console wasm if rustc target is wasm. Can use `yew` for a web wasm UI.
- Model:
  - `mamba1`: Mamba1 model. For executables, only one can be selected.
  - `mamba2`: Mamba2 model. For executables, only one can be selected.
- Burn backend:
  - `ndarray`: used for dev or wasm. Seems entirely correct. Can use `simd` for extra speed.
  - `wgpu`: for webgpu backend. Seemed entirely correct, now errors when loading the model.
  - `cuda`: for webgpu backend. Seems entirely correct.
  - `tch`: for pytorch backend. Seems correct only for cacheless mode (training-friendly).
- Extra burn features:
 - `fusion`: enable the fusion feature. Seems counter-productive for correctness and/or speed, you should sanity-check.

Note: Please check Cargo.toml for more info.

#### Performance

Generation speed (cacheless length of 100, cached length of 400) on a RTX 2060:

- `native,mamba2,ndarray,simd`: cacheless 29 tk/s; cached 2 tk/s
- `native,mamba2,wgpu`: `BufferTooBig(154484736)` error when loading the model.
- `native,mamba2,wgpu,fusion`: `BufferTooBig(154484736)` error when loading the model.
- `native,mamba2,cuda`: cacheless 248 tk/s; cached 91 tk/s
- `native,mamba2,cuda,fusion`: cacheless(garbage) 53 tk/s; cached 60 tk/s
- `native,mamba2,tch`: cacheless 108 tk/s; cached(garbage) 16 tk/s

### Example Outputs

To test for correctness for some backend, I recommend first checking `native`, if cacheless matches against cached, and optionally if they match against the `ndarray` backend. Then even if they don't match, you can guess if the results are sensible, that they return coeherent tokens, don't cause panics, etc.

The following are my results from different backends (native ndarray, native wgpu + cuda, wasm ndarray), with cacheless and cached always matching.

Mamba1:
```
Mamba is the most popular and best-selling game in the world. It has been downloaded more than 1,000 times by over 1 million people worldwide since its release on March 18th 2016...
```

Mamba2:
```
Mamba is the most popular and well-known of all Mambo songs. It was first recorded by a group called The Natives in 1883, but it has been covered many times since then with...
```

### Building Examples

##### Native Mamba2 (Console)
```bash
RUSTFLAGS="-C target-cpu=native"
cargo run --release --no-default-features --features "native,wgpu,mamba2"
```
Notes:
- This will automatically download model weights, load and run them, first in cacheless mode and then in cached mode.

##### WASM

Using [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/), [wasm-opt](https://github.com/brson/wasm-opt-rs?tab=readme-ov-file#installing-the-binary) and serving with [miniserve](https://github.com/svenstaro/miniserve/?tab=readme-ov-file#how-to-install).

#### Web Mamba1 (Console Log)

```bash
wasm-pack build --release --target web --out-dir "frontend/mamba1/pkg" \
  --no-default-features --features "ndarray,simd,mamba1"
miniserve -i 127.0.0.1 "frontend/"
```
Then open the page at [http://127.0.0.1:8080/mamba1/index.html](http://127.0.0.1:8080/mamba1/index.html) and open the console logs.
Note: This will automatically download model weights, load and run them, first in cacheless mode and then in cached mode, similarly to the native console one.

#### Web Mamba2 (Yew UI)

```bash
wasm-pack build --release --target web --out-dir "frontend/mamba2/pkg" --no-opt \
  --no-default-features --features "ndarray,simd,yew,mamba2"
miniserve -i 127.0.0.1 "frontend/"
```
Then open the page at [http://127.0.0.1:8080/mamba2/index.html](http://127.0.0.1:8080/mamba2/index.html).
Nots:
- This won't download anything by default, and you must click buttons to download, load and run the model - which is run in cached mode.
- `wasm-opt` is disabled for `yew` with `wasm-pack build --no-opt`.

### Dev

For a better IDE development, you may want to change the default features/settings on `Cargo.toml` and `.cargo/config.toml` depending on what backend and target you're testing.
