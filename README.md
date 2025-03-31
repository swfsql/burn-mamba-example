# mamba-minimal-burn-example

Click [here](https://swfsql.github.io/mamba-minimal-burn-example/) to run the model in your browser.

### Information

Adapted from [huggingface/candle/mamba-minimal](https://github.com/huggingface/candle/blob/fd7c8565646039e35925b8730d27ddad195d7e73/candle-examples/examples/mamba-minimal/).

### Building

##### Native
```bash
RUSTFLAGS="-C target-cpu=native"
cargo run --release --no-default-features --features "native,ndarray"
```

##### (WIP) WASM
```bash
# no-ui (web console only)
wasm-pack build --release --target web --no-default-features

# yew web ui
wasm-pack build --release --target web --no-default-features --features "wasm_yew_ui"

# serve
http -a 127.0.0.1
```
