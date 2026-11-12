//! The asset logic of a browser front-end. It holds the state of each asset in
//! the hub cache and in memory, the plan of its download chunks, and the
//! wrapper builder. The module needs no Yew and no `web-sys`, so a native test
//! runs it.
//!
//! Each [Asset], the tokenizer file or the weights file, has two separate
//! states: its [HubCacheState] in the hub cache, and its [Load] in memory. So a
//! front-end can fetch, erase, load and unload each asset separately.
//!
//! - The check of an asset gives a [DownloadChunkPlan] (see
//!   [plan_download_chunks](crate::hub::plan_download_chunks)). It is the plan
//!   of the fetch, and also its progress bar.
//! - [MambaWrapperBuilder::with] builds the tokenizer or the network from the
//!   loaded bytes of an asset, and drops the bytes.
//! - [Assets] makes the [Wrapper] only after it builds the tokenizer and the
//!   network.
//!
//! The methods of [Asset] and [Assets] hold the state changes. A front-end
//! starts each operation of the hub client, and gives its result to one of
//! these methods. The methods that take the hub client of the browser compile
//! only for wasm. [Connection] holds the state of the connection of that hub
//! client.

use crate::hub::{
    DownloadChunkKey, DownloadChunkPlan, Endpoint, FilePath, FileUrl, Metadata, Repo, RepoId,
    RepoType, RevisionPath, UrlTemplate,
};
#[cfg(target_arch = "wasm32")]
use crate::hub::{
    HubError,
    wasm::{Api, ApiRepo},
};
use crate::tokenizer::Tokenizer;
use crate::{CheckpointSpec, LogitsProcessorWrapper, MambaWrapper, WeightsSource, build_network};
use burn::prelude::*;
use burn_mamba::prelude::*;

/// The two assets of one checkpoint, the wrapper builder and the run wrapper.
pub struct Assets {
    /// The tokenizer file: its state in the hub cache, its load and its
    /// loaded bytes.
    pub tokenizer: Asset,
    /// The weights file: its state in the hub cache, its load and its loaded
    /// bytes.
    pub weights: Asset,
    /// Builds the tokenizer and the network from the loaded bytes. It holds
    /// them until [Assets] makes the run wrapper.
    pub wrapper_builder: MambaWrapperBuilder,
    /// The run wrapper, with its caches and its sampler. It is `None` until
    /// [Assets] builds the tokenizer and the network.
    pub wrapper: Option<Wrapper>,
}

impl Assets {
    /// Makes the two assets of the checkpoint `spec`, each in the check state,
    /// and an empty wrapper builder.
    pub fn new(spec: &'static CheckpointSpec) -> Self {
        Self {
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
            wrapper_builder: MambaWrapperBuilder::new(spec),
            wrapper: None,
        }
    }

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

    /// Unloads the asset `kind`.
    ///
    /// With a run wrapper, the unload drops the run wrapper, and marks both
    /// assets as not loaded. Without a run wrapper, the unload frees only the
    /// built part and the loaded bytes of the asset `kind`.
    pub fn unload(&mut self, kind: &AssetKind) {
        if self.wrapper.is_some() {
            self.wrapper = None;
            self.tokenizer.load.is_done = false;
            self.weights.load.is_done = false;
            return;
        }
        match kind {
            AssetKind::Tokenizer => self.wrapper_builder.tokenizer = None,
            AssetKind::Weights => self.wrapper_builder.mamba = None,
        }
        let load = &mut self.select_mut(kind).load;
        load.data = Vec::new();
        load.is_done = false;
    }

    /// Gives the loaded bytes of the asset `kind` to the wrapper builder, which
    /// builds the tokenizer or the network on `device`. It panics when the
    /// build fails.
    pub fn build(&mut self, kind: &AssetKind, device: &Device) {
        let data = std::mem::take(&mut self.select_mut(kind).load.data);
        self.wrapper_builder.with(kind, data, device);
    }

    /// Makes the run wrapper when the wrapper builder holds the tokenizer and
    /// the network. Then a new wrapper builder for the same spec replaces the
    /// used one, so a load after an unload builds again.
    pub fn try_assemble_wrapper(&mut self) {
        if !self.wrapper_builder.is_ready() {
            return;
        }
        let spec = self.wrapper_builder.spec.expect("missing checkpoint spec");
        let builder = std::mem::replace(&mut self.wrapper_builder, MambaWrapperBuilder::new(spec));
        self.wrapper = Some(builder.build());
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

impl MambaWrapperBuilder {
    /// Makes an empty builder for the checkpoint `spec`.
    pub fn new(spec: &'static CheckpointSpec) -> Self {
        Self {
            spec: Some(spec),
            tokenizer: None,
            mamba: None,
        }
    }
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

/// The run wrapper of a browser front-end, with its caches and its sampler.
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

/// One asset of a checkpoint: the tokenizer file or the weights file. It has
/// two separate states: its [HubCacheState] in the hub cache, and its [Load] in
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
    /// Makes an asset in the check state: a front-end checks it first.
    pub fn new(label: String, config: AssetConfig) -> Self {
        Self {
            label,
            config,
            load: Load::default(),
            hub_cache: HubCacheState::default(),
        }
    }

    /// Ends the check: it stores `metadata` and `plan`. The hub cache state is
    /// done only when the hub cache holds each download chunk of `plan`.
    pub fn finish_check(&mut self, metadata: Metadata, plan: DownloadChunkPlan) {
        self.hub_cache.is_checking = false;
        self.load.is_checking = false;
        self.hub_cache.fetching.metadata = Some(metadata);
        self.hub_cache.is_done = plan.iter().all(Result::is_ok);
        self.hub_cache.fetching.plan = plan;
    }

    /// Starts the fetch, with a progress of zero. Returns a copy of the plan,
    /// for the fetch.
    pub fn start_fetch(&mut self) -> DownloadChunkPlan {
        self.hub_cache.is_busy = true;
        self.hub_cache.fetching.current_download_chunk = 0;
        self.hub_cache.fetching.plan.clone()
    }

    /// Ends the entry `i` of the plan: a fetch, or a skip of a cached download
    /// chunk. The entry becomes a cached entry, and the progress increases by
    /// one.
    pub fn finish_fetch_download_chunk(&mut self, i: usize) {
        self.hub_cache.fetching.current_download_chunk += 1;
        let entry = &mut self.hub_cache.fetching.plan[i];
        if let Err(key) = entry.clone() {
            *entry = Ok(key);
        }
    }

    /// The fetch of one download chunk failed. The fetch stops.
    pub fn fail_fetch(&mut self) {
        self.hub_cache.is_busy = false;
        self.hub_cache.is_done = false;
    }

    /// Ends the fetch: the hub cache holds each download chunk.
    pub fn finish_fetch(&mut self) {
        self.hub_cache.is_busy = false;
        self.hub_cache.is_done = true;
    }

    /// Starts the load. Returns the keys of the plan, for the load. It panics
    /// when the hub cache does not hold a download chunk of the plan.
    pub fn start_load(&mut self) -> Vec<DownloadChunkKey> {
        self.load.is_busy = true;
        self.cached_keys()
    }

    /// Ends the load: it stores `data`, the loaded bytes. The build is next.
    pub fn finish_load(&mut self, data: Vec<u8>) {
        self.load.data = data;
    }

    /// The load failed. The load stops.
    pub fn fail_load(&mut self) {
        self.load.is_busy = false;
    }

    /// Ends the build: the asset is loaded.
    pub fn finish_build(&mut self) {
        self.load.is_busy = false;
        self.load.is_done = true;
    }

    /// Starts the erase. Returns the keys of the plan, for the erase. It panics
    /// when a load runs, or when the hub cache does not hold a download chunk
    /// of the plan.
    pub fn start_erase(&mut self) -> Vec<DownloadChunkKey> {
        assert!(!self.load.is_busy, "an erase cannot start during a load");
        self.hub_cache.is_busy = true;
        self.cached_keys()
    }

    /// Ends the erase: it marks each download chunk of the plan as not
    /// cached.
    pub fn finish_erase(&mut self) {
        self.hub_cache.is_busy = false;
        self.hub_cache.is_done = false;
        for entry in self.hub_cache.fetching.plan.iter_mut() {
            let (Ok(key) | Err(key)) = entry.clone();
            *entry = Err(key);
        }
    }

    /// The erase failed. The asset returns to the check state, so the
    /// front-end must check it again.
    pub fn fail_erase(&mut self) {
        self.hub_cache.is_busy = false;
        self.hub_cache.is_done = false;
        self.hub_cache.is_checking = true;
    }

    /// The keys of the plan. It panics when the hub cache does not hold a
    /// download chunk of the plan.
    fn cached_keys(&self) -> Vec<DownloadChunkKey> {
        self.hub_cache
            .fetching
            .plan
            .iter()
            .cloned()
            .collect::<Result<Vec<_>, _>>()
            .unwrap_or_else(|missing| {
                panic!(
                    "the hub cache does not hold the download chunk {}",
                    missing.key
                )
            })
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
    #[cfg(target_arch = "wasm32")]
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
    #[cfg(target_arch = "wasm32")]
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
    #[cfg(target_arch = "wasm32")]
    pub async fn check(
        &self,
        api: &Api,
        metadata: &Metadata,
    ) -> Result<DownloadChunkPlan, HubError> {
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
    #[cfg(target_arch = "wasm32")]
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
    #[cfg(target_arch = "wasm32")]
    pub async fn metadata(&self, api: &Api) -> Result<Metadata, HubError> {
        let api_repo = self.api_repo(api);
        let file_url = api_repo.url(&self.filepath);
        api.metadata(&file_url).await
    }
    /// Plans the download chunks of the file (see [ApiRepo::check]).
    #[cfg(target_arch = "wasm32")]
    pub async fn check(
        &self,
        api: &Api,
        metadata: &Metadata,
    ) -> Result<DownloadChunkPlan, HubError> {
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
    /// Whether the front-end loaded the asset and built the tokenizer or the
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
    pub plan: DownloadChunkPlan,
}

/// The state changes of the assets, without a hub client, a weights file or a
/// tokenizer file. Each test follows one rule of the asset lifecycle.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::hf;
    use crate::hub::{BlobHash, RevisionHash};

    /// A checkpoint of each family of the build. A family feature compiles the
    /// specs of its family, also without a checkpoint feature.
    const FAMILY_SPECS: &[&CheckpointSpec] = &[
        #[cfg(feature = "mamba1")]
        &hf::mamba1_130m::SPEC,
        #[cfg(feature = "mamba2")]
        &hf::mamba2_130m::SPEC,
        #[cfg(feature = "mamba3")]
        &hf::mamba3_siso_187m::SPEC,
    ];

    /// The size of the file of each test asset, in bytes.
    const SIZE: usize = 30;

    fn metadata() -> Metadata {
        Metadata {
            commit_hash: RevisionHash::default(),
            etag: BlobHash("etag".into()),
            size: SIZE,
        }
    }

    fn key(start: usize, end: usize) -> DownloadChunkKey {
        DownloadChunkKey::new("prefix", start, end, SIZE)
    }

    /// The hub cache holds the first download chunk, and not the other two.
    fn partial_plan() -> DownloadChunkPlan {
        vec![Ok(key(0, 10)), Err(key(10, 20)), Err(key(20, 30))]
    }

    /// The hub cache holds each download chunk.
    fn full_plan() -> DownloadChunkPlan {
        vec![Ok(key(0, 10)), Ok(key(10, 20)), Ok(key(20, 30))]
    }

    /// The weights asset of a new [Assets], after a check that gives `plan`.
    fn checked_asset(plan: DownloadChunkPlan) -> Asset {
        let mut asset = Assets::new(FAMILY_SPECS[0]).weights;
        asset.finish_check(metadata(), plan);
        asset
    }

    #[test]
    fn a_finished_check_stores_the_metadata_and_the_plan() {
        let asset = checked_asset(partial_plan());
        assert!(!asset.hub_cache.is_checking);
        assert!(!asset.load.is_checking);
        assert_eq!(asset.hub_cache.fetching.metadata, Some(metadata()));
        assert_eq!(asset.hub_cache.fetching.plan, partial_plan());
    }

    #[test]
    fn a_finished_check_marks_the_hub_cache_as_done_only_when_it_holds_each_download_chunk() {
        assert!(!checked_asset(partial_plan()).hub_cache.is_done);
        assert!(checked_asset(full_plan()).hub_cache.is_done);
    }

    /// The fetch handles each entry of the plan. The first entry is already in
    /// the hub cache, and stays a cached entry.
    #[test]
    fn a_fetched_download_chunk_becomes_a_cached_entry_of_the_plan() {
        let mut asset = checked_asset(partial_plan());
        assert_eq!(asset.start_fetch(), partial_plan());
        assert!(asset.hub_cache.is_busy);
        assert_eq!(asset.hub_cache.fetching.current_download_chunk, 0);

        asset.finish_fetch_download_chunk(0);
        asset.finish_fetch_download_chunk(1);
        assert_eq!(asset.hub_cache.fetching.current_download_chunk, 2);
        assert_eq!(
            asset.hub_cache.fetching.plan,
            vec![Ok(key(0, 10)), Ok(key(10, 20)), Err(key(20, 30))]
        );
    }

    #[test]
    fn a_failed_fetch_ends_the_busy_state() {
        let mut asset = checked_asset(partial_plan());
        asset.start_fetch();
        asset.fail_fetch();
        assert!(!asset.hub_cache.is_busy);
        assert!(!asset.hub_cache.is_done);
    }

    #[test]
    fn a_finished_fetch_ends_the_busy_state() {
        let mut asset = checked_asset(partial_plan());
        asset.start_fetch();
        for i in 0..3 {
            asset.finish_fetch_download_chunk(i);
        }
        asset.finish_fetch();
        assert!(!asset.hub_cache.is_busy);
        assert!(asset.hub_cache.is_done);
        assert_eq!(asset.hub_cache.fetching.plan, full_plan());
    }

    #[test]
    fn a_load_collects_the_keys_of_the_plan() {
        let mut asset = checked_asset(full_plan());
        assert_eq!(asset.start_load(), vec![key(0, 10), key(10, 20), key(20, 30)]);
        assert!(asset.load.is_busy);

        asset.finish_load(vec![1, 2, 3]);
        assert_eq!(asset.load.data, vec![1, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "the hub cache does not hold the download chunk")]
    fn a_load_panics_when_the_hub_cache_does_not_hold_a_download_chunk() {
        checked_asset(partial_plan()).start_load();
    }

    #[test]
    fn a_failed_load_ends_the_busy_state() {
        let mut asset = checked_asset(full_plan());
        asset.start_load();
        asset.fail_load();
        assert!(!asset.load.is_busy);
        assert!(!asset.load.is_done);
    }

    #[test]
    fn a_finished_erase_marks_each_download_chunk_of_the_plan_as_not_cached() {
        let mut asset = checked_asset(full_plan());
        assert_eq!(asset.start_erase(), vec![key(0, 10), key(10, 20), key(20, 30)]);
        assert!(asset.hub_cache.is_busy);

        asset.finish_erase();
        assert!(!asset.hub_cache.is_busy);
        assert!(!asset.hub_cache.is_done);
        assert_eq!(
            asset.hub_cache.fetching.plan,
            vec![Err(key(0, 10)), Err(key(10, 20)), Err(key(20, 30))]
        );
    }

    #[test]
    fn a_failed_erase_puts_the_asset_back_into_the_check_state() {
        let mut asset = checked_asset(full_plan());
        asset.start_erase();
        asset.fail_erase();
        assert!(asset.hub_cache.is_checking);
        assert!(!asset.hub_cache.is_busy);
        assert!(!asset.hub_cache.is_done);
    }

    /// A `tokenizer.json` of one token, with the gpt-neox pipeline.
    const TOKENIZER_JSON: &str = r#"{
        "pre_tokenizer": {"type": "ByteLevel"},
        "decoder": {"type": "ByteLevel"},
        "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}
    }"#;

    /// The topology of the first spec of [FAMILY_SPECS], with one layer and a
    /// small vocabulary. So a test builds a small network.
    fn small_topology() -> MambaVocabNetConfig {
        let mut topology = (FAMILY_SPECS[0].topology)();
        topology.shape.stack.n_real_layers = 1;
        topology.shape.vocab_size = 64;
        topology
    }

    /// The first spec of [FAMILY_SPECS], with [small_topology].
    static SMALL_SPEC: CheckpointSpec = CheckpointSpec {
        topology: small_topology,
        ..*FAMILY_SPECS[0]
    };

    /// The assets of [SMALL_SPEC], with the tokenizer built from
    /// [TOKENIZER_JSON].
    fn assets_with_tokenizer(device: &Device) -> Assets {
        let mut assets = Assets::new(&SMALL_SPEC);
        assets.tokenizer.finish_load(TOKENIZER_JSON.into());
        assets.build(&AssetKind::Tokenizer, device);
        assets.tokenizer.finish_build();
        assets
    }

    /// The assets of [SMALL_SPEC], with the run wrapper.
    fn assets_with_wrapper(device: &Device) -> Assets {
        let mut assets = assets_with_tokenizer(device);
        assets.wrapper_builder.mamba = Some(small_topology().init(device));
        assets.weights.finish_build();
        assets.try_assemble_wrapper();
        assert!(assets.wrapper.is_some());
        assets
    }

    #[test]
    fn a_build_gives_the_loaded_bytes_of_the_asset_to_the_wrapper_builder() {
        let device = Device::default();
        let mut assets = Assets::new(&SMALL_SPEC);
        assets.tokenizer.finish_load(TOKENIZER_JSON.into());
        assets.build(&AssetKind::Tokenizer, &device);
        assert!(assets.tokenizer.load.data.is_empty());
        assert!(assets.wrapper_builder.tokenizer.is_some());

        assets.tokenizer.finish_build();
        assert!(!assets.tokenizer.load.is_busy);
        assert!(assets.tokenizer.load.is_done);
    }

    #[test]
    fn the_run_wrapper_comes_only_when_the_builder_holds_the_tokenizer_and_the_network() {
        let device = Device::default();
        let mut assets = assets_with_tokenizer(&device);
        assets.try_assemble_wrapper();
        assert!(assets.wrapper.is_none());
        assert!(assets.wrapper_builder.tokenizer.is_some());

        assets.wrapper_builder.mamba = Some(small_topology().init(&device));
        assets.try_assemble_wrapper();
        assert!(assets.wrapper.is_some());
    }

    /// After the run wrapper, the builder is empty, and it builds for the same
    /// spec. So a load after an unload builds again.
    #[test]
    fn a_new_wrapper_builder_for_the_same_spec_replaces_the_used_one() {
        let device = Device::default();
        let mut assets = assets_with_wrapper(&device);
        let builder = &assets.wrapper_builder;
        assert!(builder.spec.is_some_and(|spec| std::ptr::eq(spec, &SMALL_SPEC)));
        assert!(builder.tokenizer.is_none());
        assert!(builder.mamba.is_none());

        assets.unload(&AssetKind::Tokenizer);
        assets.tokenizer.finish_load(TOKENIZER_JSON.into());
        assets.build(&AssetKind::Tokenizer, &device);
        assert!(assets.wrapper_builder.tokenizer.is_some());
    }

    #[test]
    fn an_unload_with_a_run_wrapper_drops_it_and_marks_both_assets_as_not_loaded() {
        let device = Device::default();
        let mut assets = assets_with_wrapper(&device);
        assets.unload(&AssetKind::Weights);
        assert!(assets.wrapper.is_none());
        assert!(!assets.tokenizer.load.is_done);
        assert!(!assets.weights.load.is_done);
    }

    /// The assets of [SMALL_SPEC], with the tokenizer and the network built,
    /// but with no run wrapper yet.
    fn assets_with_both_parts(device: &Device) -> Assets {
        let mut assets = assets_with_tokenizer(device);
        assets.wrapper_builder.mamba = Some(small_topology().init(device));
        assets.weights.finish_build();
        assets
    }

    #[test]
    fn an_unload_of_the_weights_without_a_run_wrapper_frees_only_the_network() {
        let device = Device::default();
        let mut assets = assets_with_both_parts(&device);
        assets.unload(&AssetKind::Weights);
        assert!(assets.wrapper_builder.mamba.is_none());
        assert!(!assets.weights.load.is_done);
        assert!(assets.wrapper_builder.tokenizer.is_some());
        assert!(assets.tokenizer.load.is_done);
    }

    #[test]
    fn an_unload_of_the_tokenizer_without_a_run_wrapper_frees_only_the_tokenizer() {
        let device = Device::default();
        let mut assets = assets_with_both_parts(&device);
        assets.unload(&AssetKind::Tokenizer);
        assert!(assets.wrapper_builder.tokenizer.is_none());
        assert!(!assets.tokenizer.load.is_done);
        assert!(assets.wrapper_builder.mamba.is_some());
        assert!(assets.weights.load.is_done);
    }

    /// Each asset holds loaded bytes, and no build ran.
    #[test]
    fn an_unload_without_a_run_wrapper_frees_only_the_loaded_bytes_of_its_asset() {
        let mut assets = Assets::new(&SMALL_SPEC);
        assets.tokenizer.finish_load(vec![1, 2, 3]);
        assets.weights.finish_load(vec![4, 5, 6]);

        assets.unload(&AssetKind::Tokenizer);
        assert!(assets.tokenizer.load.data.is_empty());
        assert_eq!(assets.weights.load.data, vec![4, 5, 6]);

        assets.unload(&AssetKind::Weights);
        assert!(assets.weights.load.data.is_empty());
    }
}
