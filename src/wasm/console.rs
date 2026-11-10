use crate::hub::wasm::Api;
use crate::hub::{FilePath, Repo, RepoId, RepoType, RevisionPath};
use crate::tokenizer::Tokenizer;
use crate::{CheckpointSpec, LogitsProcessorWrapper, MambaWrapper, WeightsSource, build_network, hf};
use burn::prelude::*;

pub async fn run() -> anyhow::Result<()> {
    let spec = hf::preferred()
        .ok_or_else(|| anyhow::anyhow!("no checkpoint feature is enabled in this build"))?;
    log::info!(
        "running {} (id {:?}); compiled-in checkpoints, by priority: {:?}",
        spec.display_name,
        spec.id,
        hf::ids()
    );
    let mut wrapper = build_wrapper(spec).await?;

    let prompt = "Mamba is the";
    let sample_len = 30;
    let mut output = String::new();

    log::info!("running the network");
    let mut timing = web_time::Instant::now();
    let mut last_elapsed = timing.elapsed().as_millis();
    let mut processor = LogitsProcessorWrapper::new(299792458, None, None, 1.1, 1024);

    // sequential run
    let mut i: usize = 0;
    {
        let (mut tokens, eos_token) = wrapper.reset_prompt(prompt)?;

        // adds the first token to the output (as if it were an implicit output)
        if let Some(t) = tokens.first() {
            if let Some(t) = wrapper.tokenizer.next_token(*t as u32) {
                output += &t;
            }
        }

        // initial caches
        let mut caches = wrapper.empty_caches(1)?;

        while i < sample_len {
            let (next_logits, next_caches) = wrapper.step(tokens[i], Some(caches))?;
            caches = next_caches;
            if i == 0 {
                // reset after the network generates the first token, for a better approximation
                timing = web_time::Instant::now();
            }

            let this_elapsed = timing.elapsed().as_millis();
            if this_elapsed > last_elapsed + 1000 {
                last_elapsed = this_elapsed;
                log::info!("(generation still running..): {output}");
            }

            let next_token = processor.add_logits(i, &mut tokens, next_logits)?;
            if next_token == eos_token {
                break;
            }

            // if the token has some valid representation, print it
            if let Some(t) = wrapper.tokenizer.next_token(next_token as u32) {
                output += &t;
            };

            i += 1;
        }
        if let Some(rest) = wrapper.tokenizer.decode_rest() {
            output += &rest;
        }
    }
    let elapsed = timing.elapsed().as_millis();
    log::info!(
        "the network generated {} tokens in {}ms ({} token/s)",
        i,
        elapsed,
        ((i - 1) * 1000) as f32 / elapsed as f32
    );
    log::info!("{output}");

    Ok(())
}

/// Fetches (or reuses the hub cache of) the tokenizer and the weights of the
/// checkpoint, then builds the network.
///
/// Takes the checkpoint to build, so a bundle carrying several can build any of
/// them (see [hf::CHECKPOINTS]).
pub async fn build_wrapper(spec: &'static CheckpointSpec) -> anyhow::Result<MambaWrapper> {
    let api = Api::new().await?;

    let tokenizer = {
        let timing = web_time::Instant::now();
        let bytes = api
            .model_repo(RepoId(spec.tokenizer_repo_id.into()))
            .get_bytes(&FilePath(spec.file_path_tokenizer_json.into()))
            .await?;
        log::info!(
            "tokenizer data loaded in {}ms",
            timing.elapsed().as_millis()
        );

        let timing = web_time::Instant::now();
        let tokenizer = Tokenizer::from_bytes(&bytes)?;
        log::info!("built the tokenizer in {}ms", timing.elapsed().as_millis());
        tokenizer
    };

    let mut device: Device = Default::default();
    {
        device
            .configure((crate::PRECISION_FLOAT_D_TYPE, crate::PRECISION_INT_D_TYPE))
            .expect("Failed to install fp32/i32 device defaults");
    }

    let mamba = {
        let timing = web_time::Instant::now();
        let bytes = api
            .repo(Repo::with_revision(
                RepoId(spec.repo_id.into()),
                RepoType::Model,
                RevisionPath(spec.revision_path.into()),
            ))
            .get_bytes(&FilePath(spec.file_path_model_safetensors.into()))
            .await?;
        log::info!("mamba data loaded in {}ms", timing.elapsed().as_millis());

        let timing = web_time::Instant::now();
        log::info!("building the network");
        let mamba = build_network(WeightsSource::Bytes(bytes), (spec.topology)(), &device)?;
        log::info!("built the network in {}ms", timing.elapsed().as_millis());
        mamba
    };

    Ok(MambaWrapper::new(spec, tokenizer, mamba))
}
