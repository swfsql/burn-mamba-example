#[cfg(not(feature = "yew"))]
pub mod console_ui;
#[cfg(feature = "yew")]
pub mod yew_ui;

use wasm_bindgen::prelude::wasm_bindgen;

/// Several checkpoints may be compiled in, but a bundle runs one; without any
/// there would be nothing to run. `mamba3` on its own enables the blocks but
/// neither 187m topology, hence the two sub-features here.
#[cfg(not(any(
    feature = "mamba1",
    feature = "mamba2",
    feature = "mamba3-siso",
    feature = "mamba3-mimo"
)))]
compile_error!(
    "a wasm bundle needs at least one checkpoint feature: \
     `mamba1`, `mamba2`, `mamba3-siso` and/or `mamba3-mimo`"
);

#[allow(unused_imports)]
use crate::Precision;

#[wasm_bindgen]
pub async fn wasm_main() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init_with_level(log::Level::Debug).unwrap();
    log::info!("wasm initialized");

    // TODO: configure the backend to the correct precision (crate::Precision).

    #[cfg(not(feature = "yew"))]
    console_ui::run().await.unwrap();

    #[cfg(feature = "yew")]
    {
        use crate::wasm::yew_ui::Msg;
        let handle = yew::Renderer::<yew_ui::Model>::new().render();
        handle.send_message_batch(vec![Msg::StartConnectApi]);
    }

    log::info!("wasm finished");
}
