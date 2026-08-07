use crate::hub::wasm::{Api, ApiRepo, ChunkList};
use crate::hub::{
    Endpoint, FilePath, FileUrl, HubError, Metadata, Repo, RepoId, RepoType, RevisionPath,
    UrlTemplate,
};
use crate::tokenizer::Tokenizer;
use crate::{Checkpoint, LogitsProcessorWrapper, MambaWrapper, load_mamba};
use burn::prelude::*;
use burn_mamba::prelude::*;

/// The single checkpoint this bundle was built for.
///
/// A wasm build enables exactly one model feature, so this alias resolves once
/// and every repo id / config / display name below reads through it. Enabling
/// two collides on the name, which is the intended compile error.
#[cfg(feature = "mamba1")]
use crate::hf::mamba1_130m as active;
#[cfg(feature = "mamba2")]
use crate::hf::mamba2_130m as active;
#[cfg(feature = "mamba3-mimo")]
use crate::hf::mamba3_mimo_187m as active;
#[cfg(feature = "mamba3-siso")]
use crate::hf::mamba3_siso_187m as active;

pub struct Model {
    // general data
    /// Backend device.
    pub device: Device,

    // fetching, loading, building
    /// Can check the cache, fetch and load data.
    pub cache_api: Connection<Api>,
    /// Stores cache and load status information, and also loaded bytes data.
    pub tokenizer: ModelData,
    /// Stores cache and load status information, and also loaded bytes data.
    pub mamba: ModelData,
    /// Consumes loaded bytes data to partially build the required models.
    pub models_wrapper_builder: MambaWrapperBuilder,

    // built models
    /// Models that are built and ready to use for inference.
    pub models_wrapper: Option<Wrapper>,

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
    /// The token the model uses to signal the end of the generation.
    pub eos_token: usize,
}

impl Model {
    pub fn select(&self, selection: &ModelSelection) -> &ModelData {
        match selection {
            ModelSelection::Tokenizer => &self.tokenizer,
            ModelSelection::Mamba => &self.mamba,
        }
    }

    pub fn select_mut(&mut self, selection: &ModelSelection) -> &mut ModelData {
        match selection {
            ModelSelection::Tokenizer => &mut self.tokenizer,
            ModelSelection::Mamba => &mut self.mamba,
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

        Self {
            // general data
            device,
            // fetching, loading, building
            cache_api: Connection::Disconnected,
            tokenizer: ModelData::new(
                "Tokenizer".into(),
                ModelDataConfig::Huggingface(HuggingfaceConfig {
                    endpoint: Endpoint::default(),
                    url_template: UrlTemplate::default(),
                    repo_id: RepoId(active::tokenizer_source::REPO_ID.into()),
                    repo_type: RepoType::Model,
                    revision: RevisionPath::default(),
                    filepath: FilePath(active::tokenizer_source::FILE_PATH_TOKENIZER_JSON.into()),
                }),
            ),
            mamba: ModelData::new(
                active::DISPLAY_NAME.into(),
                ModelDataConfig::Huggingface(HuggingfaceConfig {
                    endpoint: Endpoint::default(),
                    url_template: UrlTemplate::default(),
                    repo_id: RepoId(active::REPO_ID.into()),
                    repo_type: RepoType::Model,
                    revision: RevisionPath(active::REVISION_PATH.into()),
                    filepath: FilePath(active::FILE_PATH_MODEL_SAFETENSORS.into()),
                }),
            ),
            models_wrapper_builder: MambaWrapperBuilder::default(),

            // built models
            models_wrapper: None,

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
    pub tokenizer: Option<Tokenizer>,
    pub mamba: Option<MambaVocabNet>,
    pub mamba_config: Option<MambaVocabNetConfig>,
}

impl Default for MambaWrapperBuilder {
    fn default() -> Self {
        MambaWrapperBuilder {
            tokenizer: None,
            mamba: None,
            mamba_config: Some(active::config()),
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
    pub fn with(&mut self, selection: &ModelSelection, data: Vec<u8>, device: &Device) {
        match selection {
            ModelSelection::Tokenizer => {
                let tokenizer = Tokenizer::from_bytes(&data).unwrap();
                self.tokenizer = Some(tokenizer);
            }
            ModelSelection::Mamba => {
                let mamba = {
                    let timing = web_time::Instant::now();
                    log::info!("initializing and loading mamba model");

                    let mamba_config = self.mamba_config.clone().expect("missing mamba config");
                    let mamba = load_mamba(Checkpoint::Bytes(data), mamba_config, device).unwrap();
                    log::info!(
                        "mamba initialized and loaded in {}ms",
                        timing.elapsed().as_millis()
                    );

                    mamba
                };
                self.mamba = Some(mamba);
            }
        }
    }
    pub fn merge(self, other: Self) -> Self {
        Self {
            tokenizer: self.tokenizer.or(other.tokenizer),
            mamba: self.mamba.or(other.mamba),
            mamba_config: self.mamba_config.or(other.mamba_config),
        }
    }
}

impl From<MambaWrapperBuilder> for Wrapper {
    fn from(value: MambaWrapperBuilder) -> Self {
        match (value.tokenizer, value.mamba, value.mamba_config) {
            (Some(t), Some(m), Some(c)) => {
                let models = MambaWrapper::new(t, m, c);
                Wrapper::new(models)
            }
            (None, _, _) => panic!("missing tokenizer"),
            (_, None, _) => panic!("missing mamba"),
            (_, _, None) => panic!("missing mamba config"),
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
    pub models: MambaWrapper,
    pub caches: MambaCaches,
    pub processor: LogitsProcessorWrapper,
}

impl Wrapper {
    pub fn new(models: MambaWrapper) -> Self {
        let caches = models.empty_caches(1).unwrap();
        Self {
            models,
            caches,
            processor: LogitsProcessorWrapper::new(299792458, None, None, 1.1, 1024),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelData {
    pub label: String,
    pub config: ModelDataConfig,
    pub load: Load,
    pub cache: Cache,
}

impl ModelData {
    pub fn new(label: String, config: ModelDataConfig) -> Self {
        Self {
            label,
            config,
            load: Load::default(),
            cache: Cache::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ModelSelection {
    Tokenizer,
    Mamba,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ModelDataConfig {
    Huggingface(HuggingfaceConfig),
    Custom(CustomConfig),
}

impl ModelDataConfig {
    pub fn api_repo(&self, api: &Api) -> ApiRepo {
        match &self {
            ModelDataConfig::Custom(_) => {
                todo!()
            }
            ModelDataConfig::Huggingface(hf) => hf.api_repo(api),
        }
    }

    pub fn file_url(&self) -> FileUrl {
        match &self {
            ModelDataConfig::Custom(_) => {
                todo!()
            }
            ModelDataConfig::Huggingface(hf) => hf.file_url(),
        }
    }

    pub fn file_path(&self) -> &FilePath {
        match &self {
            ModelDataConfig::Custom(_) => {
                todo!()
            }
            ModelDataConfig::Huggingface(hf) => &hf.filepath,
        }
    }

    pub async fn metadata(&self, api: &Api) -> Result<Metadata, HubError> {
        match &self {
            ModelDataConfig::Custom(_) => {
                todo!()
            }
            ModelDataConfig::Huggingface(hf) => hf.metadata(api).await,
        }
    }
    pub async fn check(&self, api: &Api, metadata: &Metadata) -> Result<ChunkList, HubError> {
        match &self {
            ModelDataConfig::Custom(_) => {
                todo!()
            }
            ModelDataConfig::Huggingface(hf) => hf.check(api, metadata).await,
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
    pub async fn check(&self, api: &Api, metadata: &Metadata) -> Result<ChunkList, HubError> {
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
pub struct Cache {
    pub is_checking: bool,
    pub is_done: bool,
    pub is_busy: bool,
    pub fetching: CacheFetch,
}

impl Default for Cache {
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
pub struct CacheFetch {
    pub current_chunk: usize,
    pub metadata: Option<Metadata>,
    pub chunk_list: ChunkList, // pub total_chunk: usize,
                               // pub total_bytes: usize,
}
