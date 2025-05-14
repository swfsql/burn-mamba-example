// use crate::safetensors_load;
use crate::{
    LogitsProcessorWrapper, MambaBlockCaches, MambaModel, MambaModelConfig, MambaWrapper, hf,
};
use burn::prelude::*;
use hf_hub::{
    Repo, RepoType,
    api::wasm::{Api, ApiRepo, Metadata, UrlTemplate},
    types::{Endpoint, FilePath, FileUrl, RepoId, RevisionPath, TmpFileBlobKeyList},
};
use tokenizers::Tokenizer;

#[cfg(feature = "mamba1")]
use crate::safetensors_load_mamba1;
#[cfg(feature = "mamba1")]
use burn_mamba::{mamba1, mamba1_block};

#[cfg(feature = "mamba2")]
use crate::safetensors_load_mamba2;
#[cfg(feature = "mamba2")]
use burn_mamba::{mamba2, mamba2_block};

pub struct Model<B: Backend> {
    // general data
    /// Backend device.
    pub device: B::Device,

    // fetching, loading, building
    /// Can check the cache, fetch and load data.
    pub cache_api: Connection<Api>,
    /// Stores cache and load status information, and also loaded bytes data.
    pub tokenizer: ModelData,
    /// Stores cache and load status information, and also loaded bytes data.
    pub mamba: ModelData,
    /// Consumes loaded bytes data to partially build the required models.
    pub models_wrapper_builder: MambaWrapperBuilder<B>,

    // built models
    /// Models that are built and ready to use for inference.
    pub models_wrapper: Option<Wrapper<B>>,

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

impl<B: Backend> Model<B> {
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

impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        Self {
            // general data
            device: <B::Device>::default(),

            // fetching, loading, building
            cache_api: Connection::Disconnected,
            tokenizer: ModelData::new(
                "Tokenizer".into(),
                ModelDataConfig::Huggingface(HuggingfaceConfig {
                    endpoint: Endpoint::default(),
                    url_template: UrlTemplate::default(),
                    repo_id: RepoId(hf::tokenizer::REPO_ID.into()),
                    repo_type: RepoType::Model,
                    revision: RevisionPath::default(),
                    filepath: FilePath(hf::tokenizer::FILE_PATH_TOKENIZER_JSON.into()),
                }),
            ),
            #[cfg(feature = "mamba1")]
            mamba: ModelData::new(
                "Mamba-130m".into(),
                ModelDataConfig::Huggingface(HuggingfaceConfig {
                    endpoint: Endpoint::default(),
                    url_template: UrlTemplate::default(),
                    repo_id: RepoId(hf::mamba1_130m::REPO_ID.into()),
                    repo_type: RepoType::Model,
                    revision: RevisionPath(hf::mamba1_130m::REVISION_PATH.into()),
                    filepath: FilePath(hf::mamba1_130m::FILE_PATH_MODEL_SAFETENSORS.into()),
                }),
            ),
            #[cfg(feature = "mamba2")]
            mamba: ModelData::new(
                "Mamba2-130m".into(),
                ModelDataConfig::Huggingface(HuggingfaceConfig {
                    endpoint: Endpoint::default(),
                    url_template: UrlTemplate::default(),
                    repo_id: RepoId(hf::mamba2_130m::REPO_ID.into()),
                    repo_type: RepoType::Model,
                    revision: RevisionPath(hf::mamba2_130m::REVISION_PATH.into()),
                    filepath: FilePath(hf::mamba2_130m::FILE_PATH_MODEL_SAFETENSORS.into()),
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

pub struct MambaWrapperBuilder<B: Backend> {
    pub tokenizer: Option<Tokenizer>,
    pub mamba: Option<MambaModel<B>>,
    pub mamba_config: Option<MambaModelConfig>,
}

impl<B: Backend> Default for MambaWrapperBuilder<B> {
    fn default() -> Self {
        MambaWrapperBuilder {
            tokenizer: None,
            mamba: None,
            #[cfg(feature = "mamba1")]
            mamba_config: Some(MambaModelConfig::Mamba1(mamba1::Mamba1Config::new(
                hf::mamba1_130m::N_LAYER,
                hf::mamba1_130m::VOCAB_SIZE,
                hf::mamba1_130m::PAD_VOCAB_SIZE_MULTIPLE,
                mamba1_block::Mamba1BlockConfig::new(hf::mamba1_130m::D_MODEL),
                true,
            ))),
            #[cfg(feature = "mamba2")]
            mamba_config: Some(MambaModelConfig::Mamba2(mamba2::Mamba2Config::new(
                hf::mamba2_130m::N_LAYER,
                hf::mamba2_130m::VOCAB_SIZE,
                hf::mamba2_130m::PAD_VOCAB_SIZE_MULTIPLE,
                mamba2_block::Mamba2BlockConfig::new(hf::mamba2_130m::D_MODEL),
                true,
            ))),
        }
    }
}

impl<B: Backend> MambaWrapperBuilder<B> {
    pub fn is_ready(&self) -> bool {
        self.tokenizer.is_some() && self.mamba.is_some()
    }
    pub fn build(self) -> Wrapper<B> {
        self.into()
    }
    pub fn with(&mut self, selection: &ModelSelection, data: Vec<u8>, device: &B::Device) {
        match selection {
            ModelSelection::Tokenizer => {
                let tokenizer = tokenizers::Tokenizer::from_bytes(data)
                    .map_err(anyhow::Error::msg)
                    .unwrap();
                self.tokenizer = Some(tokenizer);
            }
            #[cfg(feature = "mamba1")]
            ModelSelection::Mamba => {
                let mamba = {
                    let timing = web_time::Instant::now();
                    log::info!("initializing and loading mamba model");

                    #[allow(irrefutable_let_patterns)]
                    let MambaModelConfig::Mamba1(config) =
                        self.mamba_config.clone().expect("missing mamba config")
                    else {
                        unreachable!()
                    };
                    let mamba = safetensors_load_mamba1::<B>(&data, config, &device).unwrap();
                    log::info!(
                        "mamba initialized and loaded in {}ms",
                        timing.elapsed().as_millis()
                    );

                    mamba
                };
                self.mamba = Some(MambaModel::Mamba1(mamba));
            }
            #[cfg(feature = "mamba2")]
            ModelSelection::Mamba => {
                let mamba = {
                    let timing = web_time::Instant::now();
                    log::info!("initializing and loading mamba model");

                    #[allow(irrefutable_let_patterns)]
                    let MambaModelConfig::Mamba2(config) =
                        self.mamba_config.clone().expect("missing mamba config")
                    else {
                        unreachable!()
                    };
                    let mamba = safetensors_load_mamba2::<B>(&data, config, &device).unwrap();
                    log::info!(
                        "mamba initialized and loaded in {}ms",
                        timing.elapsed().as_millis()
                    );

                    mamba
                };
                self.mamba = Some(MambaModel::Mamba2(mamba));
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

impl<B: Backend> From<MambaWrapperBuilder<B>> for Wrapper<B> {
    fn from(value: MambaWrapperBuilder<B>) -> Self {
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

pub struct Wrapper<B: Backend> {
    pub models: MambaWrapper<B>,
    pub caches: MambaBlockCaches<B>,
    pub processor: LogitsProcessorWrapper,
}

impl<B: Backend> Wrapper<B> {
    pub fn new(models: MambaWrapper<B>) -> Self {
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

    pub async fn metadata(&self, api: &Api) -> Option<Metadata> {
        match &self {
            ModelDataConfig::Custom(_) => {
                todo!()
            }
            ModelDataConfig::Huggingface(hf) => hf.metadata(api).await,
        }
    }
    pub async fn check(&self, api: &Api, metadata: &Metadata) -> Option<TmpFileBlobKeyList> {
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

    pub async fn metadata(&self, api: &Api) -> Option<Metadata> {
        let api_repo = self.api_repo(api);
        let file_url = api_repo.url(&self.filepath);
        let metadata = api.metadata(&file_url).await.unwrap();
        Some(metadata)
    }
    pub async fn check(&self, api: &Api, metadata: &Metadata) -> Option<TmpFileBlobKeyList> {
        let repo = Repo::new(self.repo_id.clone(), self.repo_type);
        let api_repo = api.repo(repo);
        let check = api_repo.check(&self.filepath, metadata).await.unwrap();
        Some(check)
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
    pub chunk_list: TmpFileBlobKeyList, // pub total_chunk: usize,
                                        // pub total_bytes: usize,
}
