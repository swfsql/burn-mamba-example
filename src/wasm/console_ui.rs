use crate::hub::wasm::Api;
use crate::hub::{FilePath, Repo, RepoId, RepoType, RevisionPath};
use crate::tokenizer::Tokenizer;
use crate::{Checkpoint, LogitsProcessorWrapper, MambaWrapper, ModelSpec, hf, load_mamba};
use burn::prelude::*;

pub async fn run() -> anyhow::Result<()> {
    let model = hf::preferred()
        .ok_or_else(|| anyhow::anyhow!("no checkpoint feature is enabled in this build"))?;
    log::info!(
        "running {} (id {:?}); compiled-in models, by priority: {:?}",
        model.display_name,
        model.id,
        hf::ids()
    );
    let mut models = models(model).await?;

    let prompt = "Mamba is the";
    let sample_len = 30;
    let mut output = String::new();

    log::info!("Running mamba model");
    let mut timing = web_time::Instant::now();
    let mut last_elapsed = timing.elapsed().as_millis();
    let mut processor = LogitsProcessorWrapper::new(299792458, None, None, 1.1, 1024);

    // sequential run
    let mut i: usize = 0;
    {
        let (mut tokens, eos_token) = models.reset_prompt(prompt)?;

        // gets first token (as if it were an implicit output)
        if let Some(t) = tokens.first() {
            if let Some(t) = models.tokenizer.next_token(*t as u32) {
                output += &t;
            }
        }

        // initial caches
        let mut caches = models.empty_caches(1)?;

        while i < sample_len {
            let (next_logits, next_caches) = models.step(tokens[i], Some(caches))?;
            caches = next_caches;
            if i == 0 {
                // reset after the first token gets generated to get a better approximation
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
            if let Some(t) = models.tokenizer.next_token(next_token as u32) {
                output += &t;
            };

            i += 1;
        }
        if let Some(rest) = models.tokenizer.decode_rest() {
            output += &rest;
        }
    }
    let elapsed = timing.elapsed().as_millis();
    log::info!(
        "mamba model generated {} tokens in {}ms ({} token/s)",
        i,
        elapsed,
        ((i - 1) * 1000) as f32 / elapsed as f32
    );
    log::info!("{output}");

    Ok(())
}

/// Fetches (or reuses the IndexedDB cache of) the tokenizer and the checkpoint,
/// then builds the model.
///
/// Takes the checkpoint to build, so a bundle carrying several can build any of
/// them (see [hf::MODELS]).
pub async fn models(model: &'static ModelSpec) -> anyhow::Result<MambaWrapper> {
    let api = Api::new().await?;

    let tokenizer = {
        let timing = web_time::Instant::now();
        let bytes = api
            .model(RepoId(model.tokenizer_repo_id.into()))
            .get_bytes(&FilePath(model.file_path_tokenizer_json.into()))
            .await?;
        log::info!(
            "tokenizer data loaded in {}ms",
            timing.elapsed().as_millis()
        );

        let timing = web_time::Instant::now();
        let tokenizer = Tokenizer::from_bytes(&bytes)?;
        log::info!("tokenizer loaded in {}ms", timing.elapsed().as_millis());
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
                RepoId(model.repo_id.into()),
                RepoType::Model,
                RevisionPath(model.revision_path.into()),
            ))
            .get_bytes(&FilePath(model.file_path_model_safetensors.into()))
            .await?;
        log::info!("mamba data loaded in {}ms", timing.elapsed().as_millis());

        let timing = web_time::Instant::now();
        log::info!("initializing and loading mamba model");
        let mamba = load_mamba(Checkpoint::Bytes(bytes), (model.config)(), &device)?;
        log::info!(
            "mamba initialized and loaded in {}ms",
            timing.elapsed().as_millis()
        );
        mamba
    };

    Ok(MambaWrapper::new(model, tokenizer, mamba))
}
