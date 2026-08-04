#[allow(unused_imports)]
use crate::Precision;
use crate::{LogitsProcessorWrapper, MambaWrapper, hf};
use burn::prelude::*;
use hf_hub::types::FilePath;
use hf_hub::{
    Repo, RepoType,
    api::sync::Api,
    types::{RepoId, RevisionPath},
};
use log::info;

pub fn main() -> anyhow::Result<()> {
    let () = pretty_env_logger::formatted_timed_builder()
        .filter(Some("burn_mamba_example"), log::LevelFilter::Info)
        .init();
    info!("init");

    #[cfg(feature = "mamba1")]
    let mut models = models_mamba1()?;
    #[cfg(feature = "mamba2")]
    let mut models = models_mamba2()?;

    info!("running in sequential mode (inference-friendly)");
    let sample_len = 80;
    let mut processor = LogitsProcessorWrapper::new(299792458, None, None, 1.1, 1024);
    let (sample_len, start) = models.run_sequential("Mamba is the", sample_len, &mut processor)?;
    println!();
    let elapsed = start.unwrap().elapsed().as_millis();
    info!(
        "mamba model generated {sample_len} tokens in {}ms ({} token/s)",
        elapsed,
        (sample_len * 1000) as f32 / elapsed as f32
    );

    info!("running in parallel mode (training-friendly)");
    let sample_len = 20;
    let mut processor = LogitsProcessorWrapper::new(299792458, None, None, 1.1, 1024);
    let (sample_len, start) =
        models.run_parallel("Mamba is the", sample_len, &mut processor)?;
    println!();
    let elapsed = start.unwrap().elapsed().as_millis();
    let total_sample_len = (1 + sample_len) * sample_len / 2;
    info!(
        "mamba model generated {total_sample_len} total tokens in {}ms ({} token/s)",
        elapsed,
        (total_sample_len * 1000) as f32 / elapsed as f32
    );

    info!("finished (success)");
    Ok(())
}

#[cfg(feature = "mamba1")]
fn models_mamba1() -> anyhow::Result<MambaWrapper> {
    use crate::safetensors_load_mamba1;

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

    let mut device: Device = Default::default();
    {
        device
            .configure((crate::PRECISION_FLOAT_D_TYPE, crate::PRECISION_INT_D_TYPE))
            .expect("Failed to install fp32/i32 device defaults");
    }

    let start = std::time::Instant::now();
    info!("started loading the model");
    let mamba_safetensors_bytes = {
        let f = std::fs::File::open(mamba_filename)?;
        unsafe { memmap2::MmapOptions::new().map(&f)? }
    };

    let mamba_config = hf::mamba1_130m::config();
    
    let mamba =
        safetensors_load_mamba1(&mamba_safetensors_bytes, mamba_config.clone(), &device)?;
    info!("loaded the model in {:?}", start.elapsed());

    let models = MambaWrapper::new(
        tokenizer,
        mamba,
        mamba_config,
    );

    return Ok(models);
}

#[cfg(feature = "mamba2")]
fn models_mamba2() -> anyhow::Result<MambaWrapper>
{
    use crate::safetensors_load_mamba2;

    let start = std::time::Instant::now();

    let mut device: Device = Default::default();
    {
        device
            .configure((crate::PRECISION_FLOAT_D_TYPE, crate::PRECISION_INT_D_TYPE))
            .expect("Failed to install fp32/i32 device defaults");
    }

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
        let mamba_config = hf::mamba2_130m::config();
        let mamba =
            safetensors_load_mamba2(&mamba_safetensors_bytes, mamba_config.clone(), &device)?;
        info!("loaded the model in {:?}", start.elapsed());

        let models = MambaWrapper::new(
            tokenizer,
            mamba,
            mamba_config,
        );

        models
    };

    return Ok(models);
}
