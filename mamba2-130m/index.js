import init, { wasm_main } from "./pkg/burn_mamba_example.js";

async function run() {
    await init();
    wasm_main();
}
run();
