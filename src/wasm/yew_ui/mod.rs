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

pub enum Msg {
    // Todo,

    // fetching, loading, building
    /// Starts the huggingface api connection (fetch + IndexedDB hub cache).
    StartConnectApi,
    /// Concludes the huggingface api connection (fetch + IndexedDB hub cache).
    FinishConnectApi(Api),
    FailConnectApi(HubError),
    /// Starts the huggingface api disconnection (fetch + IndexedDB hub cache).
    StartDisconnectApi,
    /// Concludes the huggingface api disconnection (fetch + IndexedDB hub cache).
    FinishDisconnectApi,
    FailDisconnectApi,
    /// Starts checking information about an asset (size, etc).
    StartAssetCheck(AssetKind),
    /// Concludes checking information about an asset (size, etc).
    FinishAssetCheck(AssetKind, Metadata, DownloadChunkList),
    FailAssetCheck,
    /// Starts fetching an asset.
    StartAssetFetch(AssetKind),
    /// Concludes fetching a single download chunk of an asset.
    /// This is useful to state about the fetching progress.
    FinishAssetFetchSingle(AssetKind, usize),
    FailAssetFetchSingle(AssetKind, usize, HubError),
    /// Concludes fetching an asset (all download chunks).
    FinishAssetFetch(AssetKind),
    /// Starts uploading an asset.
    /// This is an alternative to the "fetch and load" mechanism.
    StartAssetUpload(AssetKind),
    /// Concludes uploading an asset.
    FinishAssetUpload(AssetKind),
    FailAssetUpload,
    /// Starts loading an asset.
    /// The goal is to have bytes into the memory.
    StartAssetLoad(AssetKind),
    /// Concludes loading an asset.
    FinishAssetLoad(AssetKind, Vec<u8>),
    FailAssetLoad(AssetKind, HubError),
    /// Unloads an asset.
    /// The goal is to clear memory usage.
    /// If the tokenizer or the network was built, it also gets unbuilt.
    /// The other asset may also get unbuilt as a result of this action.
    AssetUnload(AssetKind),
    /// Starts erasing an asset from the hub cache.
    /// The goal is to free HDD data.
    StartAssetErase(AssetKind),
    /// Concludes erasing an asset from the hub cache.
    FinishAssetErase(AssetKind),
    FailAssetErase(AssetKind, HubError),
    /// Starts building the tokenizer or the network from the loaded asset.
    /// This is when the data stops being raw bytes and become tensors (etc) instead.
    StartAssetBuild(AssetKind),
    /// Concludes building the tokenizer or the network from the loaded asset.
    FinishAssetBuild(AssetKind),
    FailAssetBuild,
    /// If the tokenizer and the network are built, puts them in a [model::Wrapper],
    /// which the generation then uses.
    TryAssembleWrapper,

    // user input
    /// What the user has as inserted to the input textarea.
    InputUpdate(String),

    // inference
    /// Starts the generation.
    /// This can only be used from a zero (clean) initial cache.
    StartGeneration,
    /// Ask for a single inference step.
    /// The goal is to avoid freezing the rendering by adding a small delay between the steps.
    StepGeneration,
    /// Stops (or pause) the generation.
    StopGeneration,
    /// Resumes the generation.
    /// The last caches are used instead of a zero (clean) one.
    ResumeGeneration,
    /// Resets the last caches into a zero (clean) one.
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
