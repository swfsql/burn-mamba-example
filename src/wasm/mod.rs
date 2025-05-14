#[cfg(not(feature = "yew"))]
pub mod console_ui;
#[cfg(feature = "yew")]
pub mod yew_ui;

use wasm_bindgen::prelude::wasm_bindgen;

use crate::Precision;

#[cfg(feature = "ndarray")]
type MyBackend = burn::backend::NdArray<Precision, i32>;
#[cfg(feature = "wgpu")]
type MyBackend = burn::backend::Wgpu<Precision, i32>;

#[wasm_bindgen]
pub async fn wasm_main() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init_with_level(log::Level::Debug).unwrap();
    log::info!("wasm initialized");

    #[cfg(not(feature = "yew"))]
    console_ui::run::<MyBackend>().await.unwrap();

    #[cfg(feature = "yew")]
    {
        use crate::wasm::yew_ui::Msg;
        let handle = yew::Renderer::<yew_ui::Model<MyBackend>>::new().render();
        handle.send_message_batch(vec![Msg::StartConnectApi]);
    }

    log::info!("wasm finished");
}
