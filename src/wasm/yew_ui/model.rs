use crate::hub::wasm::{Api, ApiRepo, DownloadChunkList};
use crate::hub::{
    Endpoint, FilePath, FileUrl, HubError, Metadata, Repo, RepoId, RepoType, RevisionPath,
    UrlTemplate,
};
use crate::tokenizer::Tokenizer;
use crate::{CheckpointSpec, LogitsProcessorWrapper, MambaWrapper, WeightsSource, build_network, hf};
use burn::prelude::*;
use burn_mamba::prelude::*;

/// The checkpoint this page runs.
///
/// A bundle may carry several; the page drives one, so it takes the
/// highest-priority compiled-in checkpoint (see [hf::preferred]).
fn active() -> &'static CheckpointSpec {
    hf::preferred().expect("no checkpoint feature is enabled in this build")
}

pub struct Model {
    // general data
    /// Backend device.
    pub device: Device,
    /// Which checkpoint the asset cards fetch and the generation runs.
    pub spec: &'static CheckpointSpec,

    // fetching, loading, building
    /// Can check the hub cache, fetch and load data.
    pub hub_api: Connection<Api>,
    /// Stores hub cache and load status information, and also loaded bytes data.
    pub tokenizer: Asset,
    /// Stores hub cache and load status information, and also loaded bytes data.
    pub weights: Asset,
    /// Consumes loaded bytes data to build the tokenizer and the network.
    pub wrapper_builder: MambaWrapperBuilder,

    // built tokenizer and network
    /// The run wrapper with its caches and its sampler, once the tokenizer and
    /// the network are built.
    pub wrapper: Option<Wrapper>,

    // inference-related data
    /// Current user input.
    pub input: String,
    /// Whether the ongoing generation possibly no longer reflects the (new) user input.
    pub is_input_dirty: bool,
    pub is_reset: bool,
    pub is_generating: bool,
    pub generation_callback_interval: Option<gloo_timers::callback::Interval>,
    //
    /// Current token step index (for logits selection).
    pub step: usize,
    /// Tokens being (at first) introduced into or (later) produced by the generation.
    pub tokens: Vec<usize>,
    /// Current generation result (token concatenation from each generation step).
    pub output: String,
    /// The token the network uses to signal the end of the generation.
    pub eos_token: usize,
}

impl Model {
    pub fn select(&self, kind: &AssetKind) -> &Asset {
        match kind {
            AssetKind::Tokenizer => &self.tokenizer,
            AssetKind::Weights => &self.weights,
        }
    }

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
                .expect("Failed to install fp32/i32 device defaults");
        }

        let spec = active();

        Self {
            // general data
            device,
            spec,
            // fetching, loading, building
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

            // built tokenizer and network
            wrapper: None,

            // inference-related data
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

pub struct MambaWrapperBuilder {
    pub spec: Option<&'static CheckpointSpec>,
    pub tokenizer: Option<Tokenizer>,
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
    pub fn is_ready(&self) -> bool {
        self.tokenizer.is_some() && self.mamba.is_some()
    }
    pub fn build(self) -> Wrapper {
        self.into()
    }
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
            (_, _, None) => panic!("missing mamba"),
        }
    }
}

pub enum Connection<T> {
    Disconnected,
    Connecting,
    Connected(T),
    Disconnecting(T),
}

impl<T> Connection<T> {
    /// Note: not connected does not implies disconnected.
    pub fn is_exactly_connected(&self) -> bool {
        matches!(self, Self::Connected(_))
    }
    /// Note: not disconnected does not implies connected.
    pub fn is_exactly_disconnected(&self) -> bool {
        matches!(self, Self::Disconnected)
    }
    pub fn as_connected(&self) -> Option<&T> {
        if let Self::Connected(connected) = &self {
            Some(connected)
        } else {
            None
        }
    }
}

pub struct Wrapper {
    pub run_wrapper: MambaWrapper,
    pub caches: MambaCaches,
    pub processor: LogitsProcessorWrapper,
}

impl Wrapper {
    pub fn new(run_wrapper: MambaWrapper) -> Self {
        let caches = run_wrapper.empty_caches(1).unwrap();
        Self {
            run_wrapper,
            caches,
            processor: LogitsProcessorWrapper::new(299792458, None, None, 1.1, 1024),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Asset {
    pub label: String,
    pub config: AssetConfig,
    pub load: Load,
    pub hub_cache: HubCacheState,
}

impl Asset {
    pub fn new(label: String, config: AssetConfig) -> Self {
        Self {
            label,
            config,
            load: Load::default(),
            hub_cache: HubCacheState::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AssetKind {
    Tokenizer,
    Weights,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AssetConfig {
    Huggingface(HuggingfaceConfig),
    Custom(CustomConfig),
}

impl AssetConfig {
    pub fn api_repo(&self, api: &Api) -> ApiRepo {
        match &self {
            AssetConfig::Custom(_) => {
                todo!()
            }
            AssetConfig::Huggingface(hf) => hf.api_repo(api),
        }
    }

    pub fn file_url(&self) -> FileUrl {
        match &self {
            AssetConfig::Custom(_) => {
                todo!()
            }
            AssetConfig::Huggingface(hf) => hf.file_url(),
        }
    }

    pub fn file_path(&self) -> &FilePath {
        match &self {
            AssetConfig::Custom(_) => {
                todo!()
            }
            AssetConfig::Huggingface(hf) => &hf.filepath,
        }
    }

    pub async fn metadata(&self, api: &Api) -> Result<Metadata, HubError> {
        match &self {
            AssetConfig::Custom(_) => {
                todo!()
            }
            AssetConfig::Huggingface(hf) => hf.metadata(api).await,
        }
    }
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
    pub fn api_repo(&self, api: &Api) -> ApiRepo {
        let repo = Repo::with_revision(self.repo_id.clone(), self.repo_type, self.revision.clone());
        api.repo(repo)
    }

    pub fn file_url(&self) -> FileUrl {
        let repo = Repo::with_revision(self.repo_id.clone(), self.repo_type, self.revision.clone());
        self.url_template
            .url(&self.endpoint, &repo, &self.revision, &self.filepath)
    }

    pub async fn metadata(&self, api: &Api) -> Result<Metadata, HubError> {
        let api_repo = self.api_repo(api);
        let file_url = api_repo.url(&self.filepath);
        api.metadata(&file_url).await
    }
    pub async fn check(
        &self,
        api: &Api,
        metadata: &Metadata,
    ) -> Result<DownloadChunkList, HubError> {
        self.api_repo(api).check(metadata).await
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CustomConfig {}

#[derive(Clone, Debug, PartialEq)]
pub struct Load {
    pub is_checking: bool,
    pub is_done: bool,
    pub is_busy: bool,
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

#[derive(Clone, Debug, PartialEq)]
pub struct HubCacheState {
    pub is_checking: bool,
    pub is_done: bool,
    pub is_busy: bool,
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

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FetchState {
    pub current_download_chunk: usize,
    pub metadata: Option<Metadata>,
    pub download_chunk_list: DownloadChunkList, // pub total_chunk: usize,
                                                // pub total_bytes: usize,
}
