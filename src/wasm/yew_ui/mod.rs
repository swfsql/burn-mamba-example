pub mod model;
pub mod update;
pub mod view;

use self::model::ModelSelection;
use crate::hub::wasm::{Api, ChunkList};
use crate::hub::{HubError, Metadata};
pub use model::{Connection, Model};
use yew::prelude::*;

pub enum Msg {
    // Todo,

    // fetching, loading, building
    /// Starts the huggingface api connection (fetch + IndexedDB cache).
    StartConnectApi,
    /// Concludes the huggingface api connection (fetch + IndexedDB cache).
    FinishConnectApi(Api),
    FailConnectApi(HubError),
    /// Starts the huggingface api disconnection (fetch + IndexedDB cache).
    StartDisconnectApi,
    /// Concludes the huggingface api disconnection (fetch + IndexedDB cache).
    FinishDisconnectApi,
    FailDisconnectApi,
    /// Starts checking information about the data of a model (size, etc).
    StartModelDataCheck(ModelSelection),
    /// Concludes checking information about the data of a model (size, etc).
    FinishModelDataCheck(ModelSelection, Metadata, ChunkList),
    FailModelDataCheck,
    /// Starts fetching a model data.
    StartModelDataFetch(ModelSelection),
    /// Concludes fetching a single chunk of a model data.
    /// This is useful to state about the fetching progress.
    FinishModelDataFetchSingle(ModelSelection, usize),
    FailModelDataFetchSingle(ModelSelection, usize, HubError),
    /// Concludes fetching a model data (all chunks).
    FinishModelDataFetch(ModelSelection),
    /// Starts uploading a model data.
    /// This is an alternative to the "fetch and cache read" mechanism.
    StartModelDataUpload(ModelSelection),
    /// Concludes uploading a model data.
    FinishModelDataUpload(ModelSelection),
    FailModelDataUpload,
    /// Starts loading (reading) a model data.
    /// The goal is to have bytes into the memory.
    StartModelDataLoad(ModelSelection),
    /// Concludes loading (reading) a model data.
    FinishModelDataLoad(ModelSelection, Vec<u8>),
    FailModelDataLoad(ModelSelection, HubError),
    /// Unloads a model data.
    /// The goal is to clear memory usage.
    /// If the model was built, it also get's unbuilt.
    /// Other models may also get unbuilt as a result of this action.
    ModelDataUnload(ModelSelection),
    /// Starts erasing a model data from the cache.
    /// The goal is to free HDD data.
    StartModelDataErase(ModelSelection),
    /// Concludes erasing a model data from the cache.
    FinishModelDataErase(ModelSelection),
    FailModelDataErase(ModelSelection, HubError),
    /// Starts building a model from the model data.
    /// This is when the data stops being raw bytes and become tensors (etc) instead.
    StartModelBuild(ModelSelection),
    /// Concludes building a model from the model data.
    FinishModelBuild(ModelSelection),
    FailModelBuild,
    /// If all required models are built, we move to the next step of being to use the models
    /// for inference (etc).
    TryFinilizeModelsBuilding,

    // user input
    /// What the user has as inserted to the input textarea.
    InputUpdate(String),

    // inference
    /// Starts the models inference.
    /// This can only be used from a zero (clean) initial cache.
    StartGeneration,
    /// Ask for a single inference step.
    /// The goal is to avoid freezing the rendering by adding a small delay between the steps.
    StepGeneration,
    /// Stops (or pause) the models inference.
    StopGeneration,
    /// Resumes the models inference.
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
