use burn::prelude::*;
use burn_mamba_example::{
    LogitsProcessorWrapper, MambaModel, MambaModelConfig, MambaWrapper, Precision, hf,
};
use hf_hub::types::FilePath;
use hf_hub::{
    Repo, RepoType,
    api::sync::Api,
    types::{RepoId, RevisionPath},
};
use log::info;

#[cfg(feature = "ndarray")]
type MyBackend = burn::backend::NdArray<Precision, i32>;
#[cfg(feature = "tch")]
type MyBackend = burn::backend::LibTorch<Precision, i8>;
#[cfg(feature = "wgpu")]
type MyBackend = burn::backend::Wgpu<Precision, i32>;

fn main() -> anyhow::Result<()> {
    let () = pretty_env_logger::formatted_timed_builder()
        .filter(Some("burn_mamba_example"), log::LevelFilter::Info)
        .init();
    info!("init");

    #[cfg(feature = "mamba1")]
    let mut models = models_mamba1::<MyBackend>()?;
    #[cfg(feature = "mamba2")]
    let mut models = models_mamba2::<MyBackend>()?;

    let start = std::time::Instant::now();
    info!("running cacheless (training-friendly)");
    let mut processor = LogitsProcessorWrapper::new(299792458, None, None, 1.1, 1024);
    let chunk_size = 4;
    models.run_cacheless("Mamba is the", 10, &mut processor, Some(chunk_size))?;
    println!();
    info!("ran in {}ms", start.elapsed().as_millis());

    let start = std::time::Instant::now();
    let sample_len = 40;
    info!("running cached (inference-friendly)");
    let mut processor = LogitsProcessorWrapper::new(299792458, None, None, 1.1, 1024);
    models.run_cached("Mamba is the", sample_len, &mut processor)?;
    println!();
    let elapsed = start.elapsed().as_millis();
    info!(
        "mamba model generated {sample_len} tokens in {}ms ({} token/s)",
        elapsed,
        (sample_len * 1000) as f32 / elapsed as f32
    );

    info!("finished (success)");
    Ok(())
}

#[cfg(feature = "mamba1")]
fn models_mamba1<B: Backend>() -> anyhow::Result<MambaWrapper<B>> {
    use burn_mamba::{mamba1, mamba1_block};
    use burn_mamba_example::safetensors_load_mamba1;

    let start = std::time::Instant::now();

    let api = Api::new()?;
    let tokenizer_filename = api
        .model(RepoId(hf::tokenizer::REPO_ID.into()))
        .get(&FilePath(hf::tokenizer::FILE_PATH_TOKENIZER_JSON.into()))?;
    info!(
        "tokenizer {} path: {tokenizer_filename:?}",
        hf::tokenizer::FILE_PATH_TOKENIZER_JSON
    );

    let repo = api.repo(Repo::with_revision(
        RepoId(hf::mamba1_130m::REPO_ID.into()),
        RepoType::Model,
        RevisionPath(hf::mamba1_130m::REVISION_PATH.into()),
    ));
    let mamba_filename = repo.get(&FilePath(
        hf::mamba1_130m::FILE_PATH_MODEL_SAFETENSORS.into(),
    ))?;
    info!(
        "mamba {} path: {mamba_filename:?}",
        hf::mamba1_130m::FILE_PATH_MODEL_SAFETENSORS
    );
    info!("retrieved the files in {:?}", start.elapsed());

    let tokenizer =
        tokenizers::Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;

    let device: <B as Backend>::Device = Default::default();

    let start = std::time::Instant::now();
    info!("started loading the model");
    let mamba_safetensors_bytes = {
        let f = std::fs::File::open(mamba_filename)?;
        unsafe { memmap2::MmapOptions::new().map(&f)? }
    };
    let mamba_config = mamba1::Mamba1Config::new(
        hf::mamba1_130m::N_LAYER,
        hf::mamba1_130m::VOCAB_SIZE,
        hf::mamba1_130m::PAD_VOCAB_SIZE_MULTIPLE,
        mamba1_block::Mamba1BlockConfig::new(hf::mamba1_130m::D_MODEL),
        true,
    );
    let mamba =
        safetensors_load_mamba1::<B>(&mamba_safetensors_bytes, mamba_config.clone(), &device)?;
    info!("loaded the model in {:?}", start.elapsed());

    let models = MambaWrapper::new(
        tokenizer,
        MambaModel::Mamba1(mamba),
        MambaModelConfig::Mamba1(mamba_config),
    );

    return Ok(models);
}

#[cfg(feature = "mamba2")]
fn models_mamba2<B: Backend>() -> anyhow::Result<MambaWrapper<B>> {
    use burn_mamba::{mamba2, mamba2_block};
    use burn_mamba_example::safetensors_load_mamba2;

    let start = std::time::Instant::now();

    let device: <B as Backend>::Device = Default::default();

    let api = Api::new()?;
    let tokenizer_filename = api
        .model(RepoId(hf::tokenizer::REPO_ID.into()))
        .get(&FilePath(hf::tokenizer::FILE_PATH_TOKENIZER_JSON.into()))?;
    info!(
        "tokenizer {} path: {tokenizer_filename:?}",
        hf::tokenizer::FILE_PATH_TOKENIZER_JSON
    );
    let tokenizer =
        tokenizers::Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;
    // let mut processor = LogitsProcessorWrapper::new(0, Some(1.0), Some(1.0), 1.0, 1024);

    let models = {
        let repo = api.repo(Repo::with_revision(
            RepoId(hf::mamba2_130m::REPO_ID.into()),
            RepoType::Model,
            RevisionPath(hf::mamba2_130m::REVISION_PATH.into()),
        ));
        let mamba_filename = repo.get(&FilePath(
            hf::mamba2_130m::FILE_PATH_MODEL_SAFETENSORS.into(),
        ))?;
        info!(
            "mamba {} path: {mamba_filename:?}",
            hf::mamba2_130m::FILE_PATH_MODEL_SAFETENSORS
        );
        info!("retrieved the files in {:?}", start.elapsed());

        let start = std::time::Instant::now();
        info!("started loading the model");
        let mamba_safetensors_bytes = {
            let f = std::fs::File::open(mamba_filename)?;
            unsafe { memmap2::MmapOptions::new().map(&f)? }
        };
        let mamba_config = mamba2::Mamba2Config::new(
            hf::mamba2_130m::N_LAYER,
            hf::mamba2_130m::VOCAB_SIZE,
            hf::mamba2_130m::PAD_VOCAB_SIZE_MULTIPLE,
            mamba2_block::Mamba2BlockConfig::new(hf::mamba2_130m::D_MODEL)
                .with_is_norm_before_gate(false),
            true,
        );
        let mamba =
            safetensors_load_mamba2::<B>(&mamba_safetensors_bytes, mamba_config.clone(), &device)?;
        info!("loaded the model in {:?}", start.elapsed());

        let models = MambaWrapper::new(
            tokenizer,
            MambaModel::Mamba2(mamba),
            MambaModelConfig::Mamba2(mamba_config),
        );

        models
    };

    return Ok(models);
}
