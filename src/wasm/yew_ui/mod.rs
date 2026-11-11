//! The browser page, built with Yew. [Msg] is the whole protocol of the page,
//! [update] is its reducer, and [view] renders the [Model] with Bulma.
//!
//! Each [Asset](model::Asset), the tokenizer file or the weights file, has two
//! separate states: its [HubCacheState](model::HubCacheState) in IndexedDB, and
//! its [Load](model::Load) in memory. So the page can fetch, erase, load and
//! unload each asset separately.
//!
//! - [ApiRepo::check](crate::hub::wasm::ApiRepo::check) gives a
//!   [DownloadChunkList]. It is the plan of the fetch, and also its progress
//!   bar.
//! - [MambaWrapperBuilder::with](model::MambaWrapperBuilder::with) builds the
//!   tokenizer or the network from the loaded bytes of an asset, and drops the
//!   bytes.
//! - The page makes the [Wrapper](model::Wrapper) only after it builds the
//!   tokenizer and the network.
//!
//! The generation does one step for each tick of a 1ms interval, so the
//! browser can paint between two steps.

pub mod model;
pub mod update;
pub mod view;

use self::model::AssetKind;
use crate::hub::wasm::{Api, DownloadChunkList};
use crate::hub::{HubError, Metadata};
pub use model::{Connection, Model};
use yew::prelude::*;

/// A message of the browser page. [Model::update] handles each one.
pub enum Msg {
    // The hub client, and the assets: check, fetch, load, build.
    /// Starts the connection of the hub client: `fetch`, and the hub cache in
    /// IndexedDB.
    StartConnectApi,
    /// Ends the connection of the hub client. Then the page checks each
    /// asset.
    FinishConnectApi(Api),
    /// The connection of the hub client failed.
    FailConnectApi(HubError),
    /// Not implemented: the update only logs an error. No code sends it.
    StartDisconnectApi,
    /// Not implemented: the update only logs an error. No code sends it.
    FinishDisconnectApi,
    /// Not implemented: the update only logs an error. No code sends it.
    FailDisconnectApi,
    /// Starts the check of an asset: its metadata, and its download chunks in
    /// the hub cache.
    StartAssetCheck(AssetKind),
    /// Ends the check of an asset.
    FinishAssetCheck(AssetKind, Metadata, DownloadChunkList),
    /// Not implemented: the update panics. No code sends it, because a failed
    /// check panics before.
    FailAssetCheck,
    /// Starts the fetch of an asset: each download chunk that is not in the
    /// hub cache.
    StartAssetFetch(AssetKind),
    /// Ends one download chunk of the fetch of an asset: a fetch, or a skip of
    /// a cached download chunk. The page shows the progress of the fetch from
    /// it.
    FinishAssetFetchSingle(AssetKind, usize),
    /// The fetch of one download chunk of an asset failed.
    FailAssetFetchSingle(AssetKind, usize, HubError),
    /// Ends the fetch of an asset: each download chunk is in the hub cache.
    FinishAssetFetch(AssetKind),
    /// Not available: the update only logs an error. Its button on the page is
    /// disabled.
    StartAssetUpload(AssetKind),
    /// Not implemented: the update only logs an error. No code sends it.
    FinishAssetUpload(AssetKind),
    /// Not implemented: the update only logs an error. No code sends it.
    FailAssetUpload,
    /// Starts the load of an asset: it reads the asset from the hub cache into
    /// memory, as bytes.
    StartAssetLoad(AssetKind),
    /// Ends the load of an asset, with its bytes. Then the build of the asset
    /// starts.
    FinishAssetLoad(AssetKind, Vec<u8>),
    /// The load of an asset failed.
    FailAssetLoad(AssetKind, HubError),
    /// Unloads an asset: it frees the bytes of the asset, and the tokenizer or
    /// the network that the page built from them.
    ///
    /// When the page has a [model::Wrapper], the unload drops it. So the
    /// unload of one asset also unloads the other asset. A running generation
    /// stops first.
    AssetUnload(AssetKind),
    /// Starts the erase of an asset from the hub cache. The erase frees the
    /// storage of the browser.
    StartAssetErase(AssetKind),
    /// Ends the erase of an asset.
    FinishAssetErase(AssetKind),
    /// The erase of an asset failed. Then the page checks the asset again.
    FailAssetErase(AssetKind, HubError),
    /// Starts the build of the tokenizer or the network from the loaded bytes
    /// of an asset.
    StartAssetBuild(AssetKind),
    /// Ends the build of an asset.
    FinishAssetBuild(AssetKind),
    /// Not implemented: the update panics. No code sends it, because a failed
    /// build panics before.
    FailAssetBuild,
    /// Makes the [model::Wrapper] when the page built the tokenizer and the
    /// network. The generation uses the wrapper.
    TryAssembleWrapper,

    // The input.
    /// The new text of the input box.
    InputUpdate(String),

    // The generation.
    /// Starts the generation from the text of the input box. The caches must
    /// be zero.
    StartGeneration,
    /// Does one step of the generation. An interval of `TICK_MILLIS` (1ms)
    /// sends it, so the browser can paint between two steps.
    StepGeneration,
    /// Pauses the generation.
    StopGeneration,
    /// Continues the generation from the last caches.
    ResumeGeneration,
    /// Sets the caches to zero, and makes a new sampler.
    ResetCaches,
}

impl Component for model::Model {
    type Message = Msg;
    type Properties = ();

    fn create(_ctx: &Context<Self>) -> Self {
        Self::default()
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        Model::view(self, ctx)
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        Model::update(self, ctx, msg)
    }
}
