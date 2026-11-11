//! The state of the browser page: the [Model], with its two [Asset]s, the
//! connection of the hub client and the run wrapper.

use crate::hub::wasm::{Api, ApiRepo, DownloadChunkList};
use crate::hub::{
    Endpoint, FilePath, FileUrl, HubError, Metadata, Repo, RepoId, RepoType, RevisionPath,
    UrlTemplate,
};
use crate::tokenizer::Tokenizer;
use crate::{CheckpointSpec, LogitsProcessorWrapper, MambaWrapper, WeightsSource, build_network, hf};
use burn::prelude::*;
use burn_mamba::prelude::*;

/// The checkpoint of this page.
///
/// A bundle can hold several checkpoints, and the page runs one of them. So
/// it takes the preferred checkpoint (see [hf::preferred]).
fn active() -> &'static CheckpointSpec {
    hf::preferred().expect("no checkpoint feature is enabled in this build")
}

/// The state of the browser page: its assets, its run wrapper and its
/// generation.
pub struct Model {
    // The device and the checkpoint.
    /// The device of the network. [Model::default] sets its f32 and i32
    /// defaults.
    pub device: Device,
    /// The checkpoint that the asset cards fetch and the generation runs.
    pub spec: &'static CheckpointSpec,

    // The hub client and the assets.
    /// The hub client. It checks, fetches, loads and erases the assets.
    pub hub_api: Connection<Api>,
    /// The tokenizer file: its state in the hub cache, its load and its
    /// loaded bytes.
    pub tokenizer: Asset,
    /// The weights file: its state in the hub cache, its load and its loaded
    /// bytes.
    pub weights: Asset,
    /// Builds the tokenizer and the network from the loaded bytes. It holds
    /// them until the page makes the run wrapper.
    pub wrapper_builder: MambaWrapperBuilder,

    // The run wrapper.
    /// The run wrapper, with its caches and its sampler. It is `None` until
    /// the page builds the tokenizer and the network.
    pub wrapper: Option<Wrapper>,

    // The generation.
    /// The text of the input box.
    pub input: String,
    /// Whether the generation can differ from the input box, after an edit of
    /// the input. No code sets it to `true`.
    pub is_input_dirty: bool,
    /// Whether the caches are zero. Only then can the generation start.
    pub is_reset: bool,
    /// Whether the generation runs.
    pub is_generating: bool,
    /// The interval that sends [super::Msg::StepGeneration] after a start or a
    /// resume of the generation. [super::Msg::StopGeneration] drops it, and
    /// the drop stops the interval. At the eos token, the interval continues.
    pub generation_callback_interval: Option<gloo_timers::callback::Interval>,
    /// The position of the next step in [Self::tokens].
    pub step: usize,
    /// The token ids of the prompt, then the generated token ids.
    pub tokens: Vec<usize>,
    /// The text so far: the prompt, then the generated text.
    pub output: String,
    /// The token the network uses to signal the end of the generation.
    pub eos_token: usize,
}

impl Model {
    /// The asset of `kind`.
    pub fn select(&self, kind: &AssetKind) -> &Asset {
        match kind {
            AssetKind::Tokenizer => &self.tokenizer,
            AssetKind::Weights => &self.weights,
        }
    }

    /// The asset of `kind`, as a mutable reference.
    pub fn select_mut(&mut self, kind: &AssetKind) -> &mut Asset {
        match kind {
            AssetKind::Tokenizer => &mut self.tokenizer,
            AssetKind::Weights => &mut self.weights,
        }
    }
}

impl Default for Model {
    fn default() -> Self {
        let mut device = Device::default();
        {
            device
                .configure((crate::PRECISION_FLOAT_D_TYPE, crate::PRECISION_INT_D_TYPE))
                .expect("failed to set the f32 and i32 defaults of the device");
        }

        let spec = active();

        Self {
            // The device and the checkpoint.
            device,
            spec,
            // The hub client and the assets.
            hub_api: Connection::Disconnected,
            tokenizer: Asset::new(
                "Tokenizer".into(),
                AssetConfig::Huggingface(HuggingfaceConfig {
                    endpoint: Endpoint::default(),
                    url_template: UrlTemplate::default(),
                    repo_id: RepoId(spec.tokenizer_repo_id.into()),
                    repo_type: RepoType::Model,
                    revision: RevisionPath::default(),
                    filepath: FilePath(spec.file_path_tokenizer_json.into()),
                }),
            ),
            weights: Asset::new(
                spec.display_name.into(),
                AssetConfig::Huggingface(HuggingfaceConfig {
                    endpoint: Endpoint::default(),
                    url_template: UrlTemplate::default(),
                    repo_id: RepoId(spec.repo_id.into()),
                    repo_type: RepoType::Model,
                    revision: RevisionPath(spec.revision_path.into()),
                    filepath: FilePath(spec.file_path_model_safetensors.into()),
                }),
            ),
            wrapper_builder: MambaWrapperBuilder::default(),

            // The run wrapper.
            wrapper: None,

            // The generation.
            input: "Mamba is the".into(),
            is_input_dirty: false,
            is_reset: true,
            is_generating: false,
            generation_callback_interval: None,
            step: 0,
            tokens: vec![],
            output: "".into(),
            eos_token: 0,
        }
    }
}

/// Builds the tokenizer and the network of the checkpoint from the loaded
/// bytes of each asset. It holds them until both exist.
pub struct MambaWrapperBuilder {
    /// The checkpoint to build.
    pub spec: Option<&'static CheckpointSpec>,
    /// The built tokenizer.
    pub tokenizer: Option<Tokenizer>,
    /// The built network.
    pub mamba: Option<MambaVocabNet>,
}

impl Default for MambaWrapperBuilder {
    fn default() -> Self {
        MambaWrapperBuilder {
            spec: Some(active()),
            tokenizer: None,
            mamba: None,
        }
    }
}

impl MambaWrapperBuilder {
    /// Whether the builder holds the tokenizer and the network.
    pub fn is_ready(&self) -> bool {
        self.tokenizer.is_some() && self.mamba.is_some()
    }
    /// Makes the [Wrapper]. It panics when [Self::is_ready] is false.
    pub fn build(self) -> Wrapper {
        self.into()
    }
    /// Builds the tokenizer or the network from `data`, the loaded bytes of
    /// the asset `kind`. It panics when the build fails.
    pub fn with(&mut self, kind: &AssetKind, data: Vec<u8>, device: &Device) {
        match kind {
            AssetKind::Tokenizer => {
                let tokenizer = Tokenizer::from_bytes(&data).unwrap();
                self.tokenizer = Some(tokenizer);
            }
            AssetKind::Weights => {
                let mamba = {
                    let timing = web_time::Instant::now();
                    log::info!("building the network");

                    let spec = self.spec.expect("missing checkpoint spec");
                    let mamba =
                        build_network(WeightsSource::Bytes(data), (spec.topology)(), device)
                            .unwrap();
                    log::info!("built the network in {}ms", timing.elapsed().as_millis());

                    mamba
                };
                self.mamba = Some(mamba);
            }
        }
    }
    /// Takes each part from `self`, else from `other`.
    pub fn merge(self, other: Self) -> Self {
        Self {
            spec: self.spec.or(other.spec),
            tokenizer: self.tokenizer.or(other.tokenizer),
            mamba: self.mamba.or(other.mamba),
        }
    }
}

impl From<MambaWrapperBuilder> for Wrapper {
    fn from(value: MambaWrapperBuilder) -> Self {
        match (value.spec, value.tokenizer, value.mamba) {
            (Some(s), Some(t), Some(m)) => {
                let run_wrapper = MambaWrapper::new(s, t, m);
                Wrapper::new(run_wrapper)
            }
            (None, _, _) => panic!("missing checkpoint spec"),
            (_, None, _) => panic!("missing tokenizer"),
            (_, _, None) => panic!("missing network"),
        }
    }
}

/// The state of a connection, with its client `T`.
pub enum Connection<T> {
    /// No connection.
    Disconnected,
    /// The connection is in progress.
    Connecting,
    /// The connection is open.
    Connected(T),
    /// The disconnection is in progress. No code makes this state.
    Disconnecting(T),
}

impl<T> Connection<T> {
    /// Whether the state is [Self::Connected]. A state that is not connected
    /// can be a state other than [Self::Disconnected].
    pub fn is_exactly_connected(&self) -> bool {
        matches!(self, Self::Connected(_))
    }
    /// Whether the state is [Self::Disconnected]. A state that is not
    /// disconnected can be a state other than [Self::Connected].
    pub fn is_exactly_disconnected(&self) -> bool {
        matches!(self, Self::Disconnected)
    }
    /// The client, when the state is [Self::Connected].
    pub fn as_connected(&self) -> Option<&T> {
        if let Self::Connected(connected) = &self {
            Some(connected)
        } else {
            None
        }
    }
}

/// The run wrapper of the page, with its caches and its sampler.
pub struct Wrapper {
    /// The run wrapper.
    pub run_wrapper: MambaWrapper,
    /// The caches after the last step.
    pub caches: MambaCaches,
    /// The sampler.
    pub processor: LogitsProcessorWrapper,
}

impl Wrapper {
    /// Makes the wrapper, with zero caches and a new sampler.
    pub fn new(run_wrapper: MambaWrapper) -> Self {
        let caches = run_wrapper.empty_caches(1).unwrap();
        Self {
            run_wrapper,
            caches,
            processor: LogitsProcessorWrapper::new(299792458, None, None, 1.1, 1024),
        }
    }
}

/// One asset of the page: the tokenizer file or the weights file. It has two
/// separate states: its [HubCacheState] in the hub cache, and its [Load] in
/// memory.
#[derive(Clone, Debug, PartialEq)]
pub struct Asset {
    /// The name on the asset card.
    pub label: String,
    /// The source of the asset.
    pub config: AssetConfig,
    /// The state of the asset in memory.
    pub load: Load,
    /// The state of the asset in the hub cache.
    pub hub_cache: HubCacheState,
}

impl Asset {
    /// Makes an asset in the check state: the page checks it first.
    pub fn new(label: String, config: AssetConfig) -> Self {
        Self {
            label,
            config,
            load: Load::default(),
            hub_cache: HubCacheState::default(),
        }
    }
}

/// Which of the two assets.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AssetKind {
    /// The tokenizer file.
    Tokenizer,
    /// The weights file of the checkpoint.
    Weights,
}

/// The source of an asset.
#[derive(Clone, Debug, PartialEq)]
pub enum AssetConfig {
    /// A file of the HuggingFace hub.
    Huggingface(HuggingfaceConfig),
    /// Not available: each method panics (`todo!`). No code makes it.
    Custom(CustomConfig),
}

impl AssetConfig {
    /// The handle on the repo of the asset.
    pub fn api_repo(&self, api: &Api) -> ApiRepo {
        match &self {
            AssetConfig::Custom(_) => {
                todo!()
            }
            AssetConfig::Huggingface(hf) => hf.api_repo(api),
        }
    }

    /// The URL of the file of the asset.
    pub fn file_url(&self) -> FileUrl {
        match &self {
            AssetConfig::Custom(_) => {
                todo!()
            }
            AssetConfig::Huggingface(hf) => hf.file_url(),
        }
    }

    /// The path of the file of the asset in its repo.
    pub fn file_path(&self) -> &FilePath {
        match &self {
            AssetConfig::Custom(_) => {
                todo!()
            }
            AssetConfig::Huggingface(hf) => &hf.filepath,
        }
    }

    /// Requests the metadata of the file of the asset from the hub.
    pub async fn metadata(&self, api: &Api) -> Result<Metadata, HubError> {
        match &self {
            AssetConfig::Custom(_) => {
                todo!()
            }
            AssetConfig::Huggingface(hf) => hf.metadata(api).await,
        }
    }
    /// Plans the download chunks of the file of the asset (see
    /// [ApiRepo::check]).
    pub async fn check(
        &self,
        api: &Api,
        metadata: &Metadata,
    ) -> Result<DownloadChunkList, HubError> {
        match &self {
            AssetConfig::Custom(_) => {
                todo!()
            }
            AssetConfig::Huggingface(hf) => hf.check(api, metadata).await,
        }
    }
}

/// A file of the HuggingFace hub: the hub, the repo, the revision and the
/// path of the file.
#[derive(Clone, Debug, PartialEq)]
pub struct HuggingfaceConfig {
    endpoint: Endpoint,
    url_template: UrlTemplate,
    repo_id: RepoId,
    repo_type: RepoType,
    revision: RevisionPath,
    filepath: FilePath,
}

impl HuggingfaceConfig {
    /// The handle on the repo of the file, at its revision.
    pub fn api_repo(&self, api: &Api) -> ApiRepo {
        let repo = Repo::with_revision(self.repo_id.clone(), self.repo_type, self.revision.clone());
        api.repo(repo)
    }

    /// The URL of the file.
    pub fn file_url(&self) -> FileUrl {
        let repo = Repo::with_revision(self.repo_id.clone(), self.repo_type, self.revision.clone());
        self.url_template
            .url(&self.endpoint, &repo, &self.revision, &self.filepath)
    }

    /// Requests the metadata of the file from the hub.
    pub async fn metadata(&self, api: &Api) -> Result<Metadata, HubError> {
        let api_repo = self.api_repo(api);
        let file_url = api_repo.url(&self.filepath);
        api.metadata(&file_url).await
    }
    /// Plans the download chunks of the file (see [ApiRepo::check]).
    pub async fn check(
        &self,
        api: &Api,
        metadata: &Metadata,
    ) -> Result<DownloadChunkList, HubError> {
        self.api_repo(api).check(metadata).await
    }
}

/// Not available. No code makes it.
#[derive(Clone, Debug, PartialEq)]
pub struct CustomConfig {}

/// The state of an asset in memory.
#[derive(Clone, Debug, PartialEq)]
pub struct Load {
    /// Whether the check of the asset runs.
    pub is_checking: bool,
    /// Whether the page loaded the asset and built the tokenizer or the
    /// network from it.
    pub is_done: bool,
    /// Whether a load or a build of the asset runs.
    pub is_busy: bool,
    /// The loaded bytes. The build takes them, so they are empty after it.
    pub data: Vec<u8>,
}

impl Default for Load {
    fn default() -> Self {
        Self {
            is_checking: true,
            is_busy: false,
            is_done: false,
            data: vec![],
        }
    }
}

/// The state of an asset in the hub cache.
#[derive(Clone, Debug, PartialEq)]
pub struct HubCacheState {
    /// Whether the check of the asset runs.
    pub is_checking: bool,
    /// Whether the hub cache holds each download chunk of the asset.
    pub is_done: bool,
    /// Whether a fetch or an erase of the asset runs.
    pub is_busy: bool,
    /// The plan of the fetch, and its progress.
    pub fetching: FetchState,
}

impl Default for HubCacheState {
    fn default() -> Self {
        Self {
            is_checking: true,
            is_done: Default::default(),
            is_busy: Default::default(),
            fetching: Default::default(),
        }
    }
}

/// The plan of a fetch, and its progress.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FetchState {
    /// The count of the download chunks that the fetch handled, cached ones
    /// included.
    pub current_download_chunk: usize,
    /// The metadata of the file, after the check.
    pub metadata: Option<Metadata>,
    /// The download chunks of the file. An [Ok] entry is in the hub cache.
    pub download_chunk_list: DownloadChunkList,
}
