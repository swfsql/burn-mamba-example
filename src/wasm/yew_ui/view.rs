//! The view of the browser page: [Model::view] renders the page with Bulma.

use super::Msg;
pub use super::model::{self, Model};
use crate::common::assets::{Asset, AssetKind};
use yew::prelude::*;

impl model::Model {
    /// Renders the page: the input box, the output box, the generation
    /// controls, an asset card for each asset, and the state of the hub
    /// client.
    pub fn view(&self, ctx: &Context<Self>) -> Html {
        let link = ctx.link();

        let navbar = html_nested! {
            <nav
                class="navbar has-background-light is-fixed-bottom"
                role="navigation"
                aria_label="main navigation"
            >
            <div class="navbar-brand"/>
            <div class="navbar-menu">
                <div class="navbar-end">
                    <div class="navbar-item">
                        if self.hub_api.is_exactly_connected() {
                            <span class="tag is-success is-light">
                                {"Connected to the hub cache"}
                            </span>
                        } else if self.hub_api.is_exactly_disconnected() {
                            <span class="tag is-danger is-light">
                                {"Disconnected from the hub cache"}
                            </span>
                        } else {
                            <span class="tag is-loading is-light">
                                {"Connecting.."}
                            </span>
                        }
                    </div>
                </div>
            </div>
            </nav>
        };
        let input = html_nested! {
            <div class="tile is-child">
            <label class="label">{"Your Input"}</label>
            <textarea
                class="textarea"
                name="user-input"
                rows="4"
                placeholder="Insert here the start of the text.."
                value={self.input.clone()}
                oninput={link.callback(|e: InputEvent| Msg::InputUpdate(value_from_event(e)))}
            />
            <label class="help">{"The generation continues this text. An edit applies at the next start."}</label>
            </div>
        };
        let output = html_nested! {
            <div class="tile is-child">
            <label class="label">{"Generated Content"}</label>
            <textarea
                class="textarea"
                name="mamba-output"
                rows="8"
                placeholder="The generated text shows here.."
                value={self.output.clone()}
            />
            <label class="help">{"The input text, then the generated text."}</label>
            </div>
        };
        let controls = html_nested! {
            <div class="tile is-parent">
            <div class="tile is-child">
                if self.assets.wrapper.is_some() {
                    if self.is_generating {
                        <button
                            class="button"
                            onclick={link.callback(|_| Msg::StopGeneration)}
                            >
                            {"Pause Generation"}
                        </button>
                        <label class="help">{"Click to pause generation"}</label>
                    } else if self.is_reset {
                        <button
                            class="button"
                            onclick={link.callback(|_| Msg::StartGeneration)}
                            >
                            {"Start Generation"}
                        </button>
                        <label class="help">{"Click to start generation"}</label>
                    } else {
                        <button
                            class="button"
                            onclick={link.callback(|_| Msg::ResumeGeneration)}
                            >
                            {"Resume Generation"}
                        </button>
                        <label class="help">{"Click to resume generation"}</label>
                    }
                } else {
                    <button
                        class="button"
                        disabled={true}
                        >
                        {"Start Generation"}
                    </button>
                    <label class="help">{"Load the two assets first"}</label>
                }
            </div>
            <div class="tile is-child">
                if self.assets.wrapper.is_some() {
                    if self.is_generating {
                        <button
                            class="button"
                            disabled={true}
                            >
                            {"Reset Caches"}
                        </button>
                        <label class="help">{"Generation in progress"}</label>
                    } else if self.is_reset {
                        <button
                            class="button"
                            >
                            {"Reset Caches"}
                        </button>
                        <label class="help">{"Already reset"}</label>
                    } else {
                        <button
                            class="button"
                            onclick={link.callback(|_| Msg::ResetCaches)}
                            >
                            {"Reset Caches"}
                        </button>
                        <label class="help">{"Click to reset caches"}</label>
                    }
                } else {
                    <button
                        class="button"
                        disabled={true}
                        >
                        {"Reset Caches"}
                    </button>
                    <label class="help">{"Load the two assets first"}</label>
                }
            </div>
            </div>
        };

        let tokenizer_card = asset_card(link, &self.assets.tokenizer, AssetKind::Tokenizer);
        let weights_card = asset_card(link, &self.assets.weights, AssetKind::Weights);

        let assets = html_nested! {
            <div class="tile is-child is-vertical">
            {tokenizer_card}
            {weights_card}
            </div>
        };
        let section = html_nested! {
            <section class="section">
            <div class="container">
                <div class="tile is-ancestor">
                    <div class="tile is-vertical">
                        <div class="tile is-parent is-vertical">
                            <div class="tile is-child">
                                <h2 class="subtitle">{"Content"}</h2>
                            </div>
                            {input}
                            {output}
                            {controls}
                        </div>
                        <div class="tile is-vertical">
                            <div class="tile is-child">
                            <h2 class="subtitle">{"Assets"}</h2>
                            </div>
                            {assets}
                        </div>
                    </div>
                </div>
            </div>
            </section>
        };

        html! {<>
            {navbar}
            {section}
        </>
        }
    }
}

fn asset_card<M: yew::Component<Message = Msg>>(
    link: &yew::html::Scope<M>,
    asset: &Asset,
    kind: AssetKind,
) -> Html {
    let label = &asset.label;
    let hub_cache = &asset.hub_cache;
    let load = &asset.load;
    let total_bytes = hub_cache
        .fetching
        .metadata
        .as_ref()
        .map(|md| md.size)
        .unwrap_or_default();
    let total_bytes_human = &humansize::format_size(total_bytes, humansize::DECIMAL);
    let current_download_chunk = hub_cache.fetching.current_download_chunk;
    let total_download_chunks = hub_cache.fetching.plan.len();
    let file_url = asset.config.file_url().0;
    let file_path = &asset.config.file_path().0;
    let data_load = match (load.is_checking, load.is_done, load.is_busy) {
        (_checking @ true, _, _) => html_nested! {<>
            <button class="button is-outline is-loading" disabled={true}>
                {"Checking"}
            </button>
            <label class="help">
                {"Checking.."}
            </label>
        </>
        },
        (_checking @ false, _loaded @ false, _loading @ false) => {
            if !hub_cache.is_busy && hub_cache.is_done {
                html_nested! {<>
                    <button
                        class="button is-danger"
                        onclick={link.callback(move |_| Msg::StartAssetLoad(kind))}
                    >
                        {"Not loaded"}
                    </button>
                    <label class="help">
                        {"Click to load from the hub cache"}
                    </label>
                </>
                }
            } else {
                html_nested! {<>
                    <button class="button is-outlined is-danger" disabled={true}>
                        {"Not loaded"}
                    </button>
                    <label class="help">
                        {"Fetch it first"}
                    </label>
                </>
                }
            }
        }
        (false, false, _loading @ true) => html_nested! {<>
            <button class="button is-loading" disabled={true}>
                {"Loading"}
            </button>
            <label class="help">
                {format!("Loading ({total_bytes_human})..")}
            </label>
        </>
        },
        (false, _loaded @ true, false) => html_nested! {<>
            <button
                class="button is-success is-light"
                onclick={link.callback(move |_| Msg::AssetUnload(kind))}
            >
                {format!("Loaded ({total_bytes_human})..")}
            </button>
            <label class="help">
                {"Click to unload"}
            </label>
        </>
        },
        // The match needs this arm, but no render shows it: an unload runs in
        // one update.
        (false, _loaded @ true, _unloading @ true) => html_nested! {<>
            <button class="button is-loading" disabled={true}>
                {"Unloading"}
            </button>
            <label class="help">
                {format!("Unloading ({total_bytes_human})..")}
            </label>
        </>
        },
    };

    let data_hub_cache = match (hub_cache.is_checking, hub_cache.is_done, hub_cache.is_busy) {
        (_checking @ true, _, _) => html_nested! {<>
            <button class="button is-outline is-loading" disabled={true}>
                {"Checking"}
            </button>
            <label class="help">
                {"Checking.."}
            </label>
        </>
        },
        (_checking @ false, _cached @ false, _fetching @ false) => html_nested! {<>
            <button
                class="button"
                onclick={link.callback(move |_| Msg::StartAssetFetch(kind))}
            >
                {"Not cached"}
            </button>
            <label class="help">
                {format!("Click to fetch ({total_bytes_human})")}
            </label>
        </>
        },
        (false, false, _fetching @ true) => html_nested! {<>
            <button class="button is-loading" disabled={true}>
                {"Fetching"}
            </button>
            <label class="help">
                {format!("Fetching {total_bytes_human} ({current_download_chunk}/{total_download_chunks})")}
            </label>
        </>
        },
        (false, _cached @ true, false) => html_nested! {<>
            <button
                class="button is-success is-light"
                onclick={link.callback(move |_| Msg::StartAssetErase(kind))}
            >
                {format!("Cached ({total_bytes_human})")}
            </button>
            <label class="help">
                {"Click to erase from the hub cache"}
            </label>
        </>
        },
        (false, _cached @ true, _erasing @ true) => html_nested! {<>
            <button class="button is-loading" disabled={true}>
                {"Erasing"}
            </button>
            <label class="help">
                {format!("Erasing {total_bytes_human}..")}
            </label>
        </>
        },
    };
    html_nested! {
        <div class="tile is-parent is-vertical">
        <div class="tile is-child">
            <label class="label">
                {label}
            </label>
        </div>
        <div class="tile is-parent">
            <div class="tile is-child">
                {data_load}
            </div>
            <div class="tile is-child">
                {data_hub_cache}
            </div>
            <div class="tile is-child">
                <a
                    class="button"
                    href={file_url}
                    download={file_path.clone()}
                    target="_blank"
                >
                    {"Download to file"}
                </a>
                <label class="help">
                    {"Click to download or open in new tab"}
                </label>
            </div>
            <div class="tile is-child">
                <button
                    class="button"
                    disabled={true}
                    onclick={link.callback(move |_| Msg::StartAssetUpload(kind))}
                >
                    {"Load from file"}
                </button>
                <label class="help">
                    {"Not available"}
                </label>
            </div>
        </div>
        </div>
    }
}

/// The text of the text area that sent the input event `e`.
pub fn value_from_event(e: InputEvent) -> String {
    use wasm_bindgen::JsCast;
    let event: Event = e.dyn_into().unwrap();
    let event_target = event.target().unwrap();
    let target: indexed_db_futures::web_sys::HtmlTextAreaElement = event_target.dyn_into().unwrap();
    target.value()
}
