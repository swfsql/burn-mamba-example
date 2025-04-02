use burn::prelude::*;
use burn_mamba_example::{LogitsProcessorWrapper, MambaWrapper, hf, safetensors_load};
use hf_hub::types::FilePath;
use hf_hub::{
    Repo, RepoType,
    api::sync::Api,
    types::{RepoId, RevisionPath},
};
use log::info;

#[cfg(feature = "ndarray")]
type MyBackend = burn::backend::NdArray<f32, i32>;
#[cfg(feature = "wgpu")]
type MyBackend = burn::backend::Wgpu<f32, i32>;

fn main() -> anyhow::Result<()> {
    let start = std::time::Instant::now();

    let () = pretty_env_logger::formatted_timed_builder()
        .filter(Some("burn_mamba_example"), log::LevelFilter::Info)
        .init();
    info!("init");

    let api = Api::new()?;
    let tokenizer_filename = api
        .model(RepoId(hf::tokenizer::REPO_ID.into()))
        .get(&FilePath(hf::tokenizer::FILE_PATH_TOKENIZER_JSON.into()))?;
    info!(
        "tokenizer {} path: {tokenizer_filename:?}",
        hf::tokenizer::FILE_PATH_TOKENIZER_JSON
    );

    let repo = api.repo(Repo::with_revision(
        RepoId(hf::mamba_130m::REPO_ID.into()),
        RepoType::Model,
        RevisionPath(hf::mamba_130m::REVISION_PATH.into()),
    ));
    let mamba_filename = repo.get(&FilePath(
        hf::mamba_130m::FILE_PATH_MODEL_SAFETENSORS.into(),
    ))?;
    info!(
        "mamba {} path: {mamba_filename:?}",
        hf::mamba_130m::FILE_PATH_MODEL_SAFETENSORS
    );
    info!("retrieved the files in {:?}", start.elapsed());

    let tokenizer =
        tokenizers::Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;

    let device: <MyBackend as Backend>::Device = Default::default();

    let start = std::time::Instant::now();
    info!("started loading the model");
    let mamba_safetensors_bytes = {
        let f = std::fs::File::open(mamba_filename)?;
        unsafe { memmap2::MmapOptions::new().map(&f)? }
    };
    let mamba_config = burn_mamba::MambaConfig::new(
        hf::mamba_130m::N_LAYER,
        hf::mamba_130m::PADDED_VOCAB_SIZE,
        burn_mamba::MambaBlockConfig::new(hf::mamba_130m::D_MODEL),
        true,
    );
    let mamba =
        safetensors_load::<MyBackend>(&mamba_safetensors_bytes, mamba_config.clone(), &device)?;
    info!("loaded the model in {:?}", start.elapsed());

    let mut models = MambaWrapper::new(tokenizer, mamba, mamba_config);
    let mut processor = LogitsProcessorWrapper::new(299792458, None, None, 1.1, 1024);

    info!("running cacheless (training-friendly)");
    models.run_cacheless("Mamba is the", 10, &mut processor)?;
    println!();

    info!("running cached (inference-friendly)");
    models.run_cached("Mamba is the", 40, &mut processor)?;
    println!();

    info!("finished (success)");
    Ok(())
}
