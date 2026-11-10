#[allow(unused_imports)]
use crate::Precision;
use crate::hub::sync::Api;
use crate::hub::{FilePath, Repo, RepoId, RepoType, RevisionPath};
use crate::tokenizer::Tokenizer;
use crate::{CheckpointSpec, LogitsProcessorWrapper, MambaWrapper, WeightsSource, build_network, hf};
use burn::prelude::*;
use log::info;

/// Several checkpoints may be compiled in, but the binary runs one; without any
/// there would be nothing to run. A family feature on its own (`mamba1`,
/// `mamba2`, `mamba3`) enables the blocks but no checkpoint.
#[cfg(not(any(
    feature = "mamba1-130m",
    feature = "mamba2-130m",
    feature = "mamba3-siso-187m",
    feature = "mamba3-mimo-187m"
)))]
compile_error!(
    "the `native` binary needs at least one checkpoint feature: \
     `mamba1-130m`, `mamba2-130m`, `mamba3-siso-187m` and/or `mamba3-mimo-187m`"
);

/// Picks the checkpoint to run out of everything compiled in: `MAMBA_CHECKPOINT`
/// (a [CheckpointSpec::id]) when set, else the highest-priority one.
pub fn select_checkpoint() -> anyhow::Result<&'static CheckpointSpec> {
    match std::env::var("MAMBA_CHECKPOINT") {
        Ok(id) => hf::by_id(id.trim()).ok_or_else(|| {
            anyhow::anyhow!(
                "MAMBA_CHECKPOINT={id:?} is not compiled into this binary; available: {:?}",
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

    let spec = select_checkpoint()?;
    info!(
        "running {} (id {:?}); compiled-in checkpoints, by priority: {:?}",
        spec.display_name,
        spec.id,
        hf::ids()
    );
    let mut wrapper = build_wrapper(spec)?;

    info!("running in sequential mode (inference-friendly)");
    let sample_len = 80;
    let mut processor = LogitsProcessorWrapper::new(299792458, None, None, 1.1, 1024);
    let (sample_len, start) = wrapper.run_sequential("Mamba is the", sample_len, &mut processor)?;
    println!();
    let elapsed = start.unwrap().elapsed().as_millis();
    info!(
        "the network generated {sample_len} tokens in {}ms ({} token/s)",
        elapsed,
        (sample_len * 1000) as f32 / elapsed as f32
    );

    info!("running in parallel mode (training-friendly)");
    let sample_len = 20;
    let mut processor = LogitsProcessorWrapper::new(299792458, None, None, 1.1, 1024);
    let (sample_len, start) = wrapper.run_parallel("Mamba is the", sample_len, &mut processor)?;
    println!();
    let elapsed = start.unwrap().elapsed().as_millis();
    let total_sample_len = (1 + sample_len) * sample_len / 2;
    info!(
        "the network generated {total_sample_len} total tokens in {}ms ({} token/s)",
        elapsed,
        (total_sample_len * 1000) as f32 / elapsed as f32
    );

    info!("finished (success)");
    Ok(())
}

/// Fetches (or reuses) the tokenizer and the weights of the checkpoint, then
/// builds the network.
///
/// Takes the checkpoint to build, so a binary carrying several can build any of
/// them (see [hf::CHECKPOINTS]).
pub fn build_wrapper(spec: &'static CheckpointSpec) -> anyhow::Result<MambaWrapper> {
    let start = std::time::Instant::now();

    let api = Api::new()?;
    let tokenizer_file = spec.file_path_tokenizer_json;
    let tokenizer_filename = api
        .model_repo(RepoId(spec.tokenizer_repo_id.into()))
        .get(&FilePath(tokenizer_file.into()))?;
    info!("tokenizer {tokenizer_file} path: {tokenizer_filename:?}");

    let repo = api.repo(Repo::with_revision(
        RepoId(spec.repo_id.into()),
        RepoType::Model,
        RevisionPath(spec.revision_path.into()),
    ));
    let weights_file = spec.file_path_model_safetensors;
    let mamba_filename = repo.get(&FilePath(weights_file.into()))?;
    info!("mamba {weights_file} path: {mamba_filename:?}");
    info!("retrieved the files in {:?}", start.elapsed());

    let tokenizer = Tokenizer::from_file(tokenizer_filename)?;

    let mut device: Device = Default::default();
    {
        device
            .configure((crate::PRECISION_FLOAT_D_TYPE, crate::PRECISION_INT_D_TYPE))
            .expect("Failed to install fp32/i32 device defaults");
    }

    let start = std::time::Instant::now();
    info!("started the build of the network");
    let mamba = build_network(WeightsSource::File(mamba_filename), (spec.topology)(), &device)?;
    info!("built the network in {:?}", start.elapsed());

    Ok(MambaWrapper::new(spec, tokenizer, mamba))
}
