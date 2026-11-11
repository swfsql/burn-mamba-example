//! The entry point of a browser bundle: [wasm_main]. A bundle has one
//! front-end, the browser console or the browser page, and runs the preferred
//! checkpoint.

#[cfg(feature = "console")]
pub mod console;
#[cfg(feature = "yew")]
pub mod yew_ui;

use wasm_bindgen::prelude::wasm_bindgen;

/// A build can hold several checkpoints, and a bundle runs one of them. With
/// no checkpoint, it has nothing to run. A family feature alone (`mamba1`,
/// `mamba2`, `mamba3`) adds the blocks, but no checkpoint.
#[cfg(not(any(
    feature = "mamba1-130m",
    feature = "mamba2-130m",
    feature = "mamba3-siso-187m",
    feature = "mamba3-mimo-187m"
)))]
compile_error!(
    "a wasm bundle needs at least one checkpoint feature: \
     `mamba1-130m`, `mamba2-130m`, `mamba3-siso-187m` and/or `mamba3-mimo-187m`"
);

/// A bundle runs one front-end: the browser console or the browser page.
#[cfg(not(any(feature = "console", feature = "yew")))]
compile_error!("a wasm bundle needs one front-end feature: `console` or `yew`");
#[cfg(all(feature = "console", feature = "yew"))]
compile_error!("a wasm bundle needs exactly one front-end feature, but `console` and `yew` are both on");

#[allow(unused_imports)]
use crate::Precision;

/// The entry point of the bundle. `index.js` calls it.
///
/// It sets the panic hook and the logger, then starts the front-end of the
/// bundle. The front-end sets the f32 and i32 defaults of the device before it
/// builds the network.
#[wasm_bindgen]
pub async fn wasm_main() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init_with_level(log::Level::Debug).unwrap();
    log::info!("wasm initialized");

    #[cfg(feature = "console")]
    console::run().await.unwrap();

    #[cfg(feature = "yew")]
    {
        use crate::wasm::yew_ui::Msg;
        let handle = yew::Renderer::<yew_ui::Model>::new().render();
        handle.send_message_batch(vec![Msg::StartConnectApi]);
    }

    log::info!("wasm finished");
}
