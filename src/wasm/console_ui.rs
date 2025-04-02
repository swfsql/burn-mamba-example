use crate::hf;
use crate::{LogitsProcessorWrapper, MambaWrapper, safetensors_load};
use burn::prelude::*;
use hf_hub::{
    Repo, RepoType,
    api::wasm::Api,
    types::{FilePath, RepoId, RevisionPath},
};

pub async fn run<B: Backend>() -> anyhow::Result<()> {
    let api = Api::new().await?;

    let tokenizer_filename = {
        let timing = web_time::Instant::now();
        let filename = api
            .model(RepoId(hf::tokenizer::REPO_ID.into()))
            .get(&FilePath(hf::tokenizer::FILE_PATH_TOKENIZER_JSON.into()))
            .await?;
        log::info!(
            "finished downloading/checking tokenizer in {}ms",
            timing.elapsed().as_millis()
        ); // 4s/2s
        filename
    };

    let mamba_filename = {
        let timing = web_time::Instant::now();
        let repo = api.repo(Repo::with_revision(
            RepoId(hf::mamba_130m::REPO_ID.into()),
            RepoType::Model,
            RevisionPath(hf::mamba_130m::REVISION_PATH.into()),
        ));
        let filename = repo
            .get(&FilePath(
                hf::mamba_130m::FILE_PATH_MODEL_SAFETENSORS.into(),
            ))
            .await?;
        log::info!(
            "finished downloading/checking the mamba model in {}ms",
            timing.elapsed().as_millis()
        ); // ~180s/2s
        filename
    };

    let tokenizer = {
        let timing = web_time::Instant::now();
        log::info!("loading tokenizer data");
        let tokenizer = api.load_bytes(&tokenizer_filename).await.unwrap();
        log::info!(
            "tokenizer data loaded in {}ms",
            timing.elapsed().as_millis()
        ); // ~100ms

        let timing = web_time::Instant::now();
        log::info!("loading tokenizer");
        let tokenizer = tokenizers::Tokenizer::from_bytes(tokenizer).map_err(anyhow::Error::msg)?;
        log::info!("tokenizer loaded in {}ms", timing.elapsed().as_millis()); // ~200ms
        tokenizer
    };

    let device: B::Device = Default::default();

    let mamba_config = burn_mamba::MambaConfig::new(
        hf::mamba_130m::N_LAYER,
        hf::mamba_130m::PADDED_VOCAB_SIZE,
        burn_mamba::MambaBlockConfig::new(hf::mamba_130m::D_MODEL),
        true,
    );
    let mamba = {
        let timing = web_time::Instant::now();
        log::info!("loading mamba data");
        let mamba_bytes = api.load_bytes(&mamba_filename).await.unwrap();
        log::info!("mamba data loaded in {}ms", timing.elapsed().as_millis()); // ~2-3s

        let timing = web_time::Instant::now();
        log::info!("initializing and loading mamba model");
        let mamba = safetensors_load::<B>(&mamba_bytes, mamba_config.clone(), &device)?;
        log::info!(
            "mamba initialized and loaded in {}ms",
            timing.elapsed().as_millis()
        );

        mamba
    };

    let mut models = MambaWrapper::new(tokenizer, mamba, mamba_config);
    let mut processor = LogitsProcessorWrapper::new(299792458, None, None, 1.1, 1024);

    let prompt = "Mamba is the";
    let sample_len = 30;
    let mut output = String::new();

    log::info!("Running mamba model");
    let timing = web_time::Instant::now();
    let mut last_elapsed = timing.elapsed().as_millis();

    // cached run
    let mut i: usize = 0;
    {
        let (mut tokens, eos_token) = models.reset_prompt(prompt)?;

        // gets first token (as if it were an implicit output)
        if let Some(t) = tokens.first() {
            if let Some(t) = models.tokenizer.next_token(*t as u32)? {
                output += &t;
            }
        }

        // initial caches
        let mut caches = models.empty_caches(1)?;

        while i < sample_len {
            let this_elapsed = timing.elapsed().as_millis();
            if this_elapsed > last_elapsed + 1000 {
                last_elapsed = this_elapsed;
                log::info!("(generation still running..): {output}");
            }

            let next_logits = models.step(tokens[i], Some(&mut caches))?;
            let next_token = processor.add_logits(i, &mut tokens, next_logits)?;
            if next_token == eos_token {
                break;
            }

            // if the token has some valid representation, print it
            if let Some(t) = models.tokenizer.next_token(next_token as u32)? {
                output += &t;
            }

            i += 1;
        }
        if let Some(rest) = models.tokenizer.decode_rest().map_err(anyhow::Error::msg)? {
            output += &rest;
        }
    }
    let elapsed = timing.elapsed().as_millis();
    log::info!(
        "mamba model generated {} tokens in {}ms ({} token/s)",
        i,
        elapsed,
        (i * 1000) as f32 / elapsed as f32
    );
    log::info!("{output}");

    Ok(())
}
