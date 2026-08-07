#[allow(unused_imports)]
use crate::Precision;
use crate::hub::sync::Api;
use crate::hub::{FilePath, Repo, RepoId, RepoType, RevisionPath};
use crate::tokenizer::Tokenizer;
use crate::{Checkpoint, LogitsProcessorWrapper, MambaWrapper, ModelSpec, hf, load_mamba};
use burn::prelude::*;
use log::info;

/// Several checkpoints may be compiled in, but the binary runs one; without any
/// there would be nothing to run. `mamba3` on its own enables the blocks but
/// neither 187m topology, hence the two sub-features here.
#[cfg(not(any(
    feature = "mamba1",
    feature = "mamba2",
    feature = "mamba3-siso",
    feature = "mamba3-mimo"
)))]
compile_error!(
    "the `native` binary needs at least one checkpoint feature: \
     `mamba1`, `mamba2`, `mamba3-siso` and/or `mamba3-mimo`"
);

/// Picks the checkpoint to run out of everything compiled in: `MAMBA_MODEL`
/// (a [ModelSpec::id]) when set, else the highest-priority one.
pub fn select_model() -> anyhow::Result<&'static ModelSpec> {
    match std::env::var("MAMBA_MODEL") {
        Ok(id) => hf::by_id(id.trim()).ok_or_else(|| {
            anyhow::anyhow!(
                "MAMBA_MODEL={id:?} is not compiled into this binary; available: {:?}",
                hf::ids()
            )
        }),
        Err(_) => hf::preferred()
            .ok_or_else(|| anyhow::anyhow!("no checkpoint feature is enabled in this build")),
    }
}

pub fn main() -> anyhow::Result<()> {
    let () = pretty_env_logger::formatted_timed_builder()
        .filter(Some("burn_mamba_example"), log::LevelFilter::Info)
        .init();
    info!("init");

    let model = select_model()?;
    info!(
        "running {} (id {:?}); compiled-in models, by priority: {:?}",
        model.display_name,
        model.id,
        hf::ids()
    );
    let mut models = models(model)?;

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
    let (sample_len, start) = models.run_parallel("Mamba is the", sample_len, &mut processor)?;
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

/// Downloads (or reuses) the tokenizer and the checkpoint, then builds the model.
///
/// Takes the checkpoint to build, so a binary carrying several can build any of
/// them (see [hf::MODELS]).
pub fn models(model: &'static ModelSpec) -> anyhow::Result<MambaWrapper> {
    let start = std::time::Instant::now();

    let api = Api::new()?;
    let tokenizer_file = model.file_path_tokenizer_json;
    let tokenizer_filename = api
        .model(RepoId(model.tokenizer_repo_id.into()))
        .get(&FilePath(tokenizer_file.into()))?;
    info!("tokenizer {tokenizer_file} path: {tokenizer_filename:?}");

    let repo = api.repo(Repo::with_revision(
        RepoId(model.repo_id.into()),
        RepoType::Model,
        RevisionPath(model.revision_path.into()),
    ));
    let model_file = model.file_path_model_safetensors;
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
    let mamba = load_mamba(Checkpoint::File(mamba_filename), (model.config)(), &device)?;
    info!("loaded the model in {:?}", start.elapsed());

    Ok(MambaWrapper::new(model, tokenizer, mamba))
}
