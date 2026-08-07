#[allow(unused_imports)]
use crate::Precision;
use crate::hub::sync::Api;
use crate::hub::{FilePath, Repo, RepoId, RepoType, RevisionPath};
use crate::tokenizer::Tokenizer;
use crate::{Checkpoint, LogitsProcessorWrapper, MambaWrapper, hf, load_mamba};
use burn::prelude::*;
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
    #[cfg(feature = "mamba3-siso")]
    let mut models = models_mamba3_siso()?;
    #[cfg(feature = "mamba3-mimo")]
    let mut models = models_mamba3_mimo()?;

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
    models(
        hf::mamba1_130m::REPO_ID,
        hf::mamba1_130m::REVISION_PATH,
        hf::mamba1_130m::FILE_PATH_MODEL_SAFETENSORS,
        hf::mamba1_130m::tokenizer_source::REPO_ID,
        hf::mamba1_130m::tokenizer_source::FILE_PATH_TOKENIZER_JSON,
        hf::mamba1_130m::config(),
    )
}

#[cfg(feature = "mamba2")]
fn models_mamba2() -> anyhow::Result<MambaWrapper> {
    models(
        hf::mamba2_130m::REPO_ID,
        hf::mamba2_130m::REVISION_PATH,
        hf::mamba2_130m::FILE_PATH_MODEL_SAFETENSORS,
        hf::mamba2_130m::tokenizer_source::REPO_ID,
        hf::mamba2_130m::tokenizer_source::FILE_PATH_TOKENIZER_JSON,
        hf::mamba2_130m::config(),
    )
}

#[cfg(feature = "mamba3-siso")]
fn models_mamba3_siso() -> anyhow::Result<MambaWrapper> {
    models(
        hf::mamba3_siso_187m::REPO_ID,
        hf::mamba3_siso_187m::REVISION_PATH,
        hf::mamba3_siso_187m::FILE_PATH_MODEL_SAFETENSORS,
        hf::mamba3_siso_187m::tokenizer_source::REPO_ID,
        hf::mamba3_siso_187m::tokenizer_source::FILE_PATH_TOKENIZER_JSON,
        hf::mamba3_siso_187m::config(),
    )
}

#[cfg(feature = "mamba3-mimo")]
fn models_mamba3_mimo() -> anyhow::Result<MambaWrapper> {
    models(
        hf::mamba3_mimo_187m::REPO_ID,
        hf::mamba3_mimo_187m::REVISION_PATH,
        hf::mamba3_mimo_187m::FILE_PATH_MODEL_SAFETENSORS,
        hf::mamba3_mimo_187m::tokenizer_source::REPO_ID,
        hf::mamba3_mimo_187m::tokenizer_source::FILE_PATH_TOKENIZER_JSON,
        hf::mamba3_mimo_187m::config(),
    )
}

/// Downloads (or reuses) the tokenizer and the checkpoint, then builds the model.
fn models(
    repo_id: &str,
    revision_path: &str,
    model_file: &str,
    tokenizer_repo_id: &str,
    tokenizer_file: &str,
    mamba_config: burn_mamba::prelude::MambaVocabNetConfig,
) -> anyhow::Result<MambaWrapper> {
    let start = std::time::Instant::now();

    let api = Api::new()?;
    let tokenizer_filename = api
        .model(RepoId(tokenizer_repo_id.into()))
        .get(&FilePath(tokenizer_file.into()))?;
    info!("tokenizer {tokenizer_file} path: {tokenizer_filename:?}");

    let repo = api.repo(Repo::with_revision(
        RepoId(repo_id.into()),
        RepoType::Model,
        RevisionPath(revision_path.into()),
    ));
    let mamba_filename = repo.get(&FilePath(model_file.into()))?;
    info!("mamba {model_file} path: {mamba_filename:?}");
    info!("retrieved the files in {:?}", start.elapsed());

    let tokenizer = Tokenizer::from_file(tokenizer_filename)?;

    let mut device: Device = Default::default();
    {
        device
            .configure((crate::PRECISION_FLOAT_D_TYPE, crate::PRECISION_INT_D_TYPE))
            .expect("Failed to install fp32/i32 device defaults");
    }

    let start = std::time::Instant::now();
    info!("started loading the model");
    let mamba = load_mamba(
        Checkpoint::File(mamba_filename),
        mamba_config.clone(),
        &device,
    )?;
    info!("loaded the model in {:?}", start.elapsed());

    Ok(MambaWrapper::new(tokenizer, mamba, mamba_config))
}
