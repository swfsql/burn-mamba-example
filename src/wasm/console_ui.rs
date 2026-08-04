use crate::{LogitsProcessorWrapper, MambaWrapper, hf};
use burn::prelude::*;
use hf_hub::{
    Repo, RepoType,
    api::wasm::Api,
    types::{FilePath, RepoId, RevisionPath},
};

pub async fn run() -> anyhow::Result<()> {
    #[cfg(feature = "mamba1")]
    let mut models = models_mamba1().await?;
    #[cfg(feature = "mamba2")]
    let mut models = models_mamba2().await?;

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
            if let Some(t) = models.tokenizer.next_token(*t as u32)? {
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
            if let Some(t) = models.tokenizer.next_token(next_token as u32)? {
                output += &t;
            };

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
        ((i - 1) * 1000) as f32 / elapsed as f32
    );
    log::info!("{output}");

    Ok(())
}

#[cfg(feature = "mamba1")]
pub async fn models_mamba1() -> anyhow::Result<MambaWrapper> {
    use crate::safetensors_load_mamba1;

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
            RepoId(hf::mamba1_130m::REPO_ID.into()),
            RepoType::Model,
            RevisionPath(hf::mamba1_130m::REVISION_PATH.into()),
        ));
        let filename = repo
            .get(&FilePath(
                hf::mamba1_130m::FILE_PATH_MODEL_SAFETENSORS.into(),
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

    let mut device: Device = Default::default();
    {
        device
            .configure((crate::PRECISION_FLOAT_D_TYPE, crate::PRECISION_INT_D_TYPE))
            .expect("Failed to install fp32/i32 device defaults");
    }
    
    let mamba_config = hf::mamba1_130m::config();
    let mamba = {
        let timing = web_time::Instant::now();
        log::info!("loading mamba data");
        let mamba_bytes = api.load_bytes(&mamba_filename).await.unwrap();
        log::info!("mamba data loaded in {}ms", timing.elapsed().as_millis()); // ~2-3s

        let timing = web_time::Instant::now();
        log::info!("initializing and loading mamba model");
        let mamba = safetensors_load_mamba1(&mamba_bytes, mamba_config.clone(), &device)?;
        log::info!(
            "mamba initialized and loaded in {}ms",
            timing.elapsed().as_millis()
        );

        mamba
    };

    let models = MambaWrapper::new(
        tokenizer,
        mamba,
        mamba_config,
    );

    Ok(models)
}

#[cfg(feature = "mamba2")]
pub async fn models_mamba2() -> anyhow::Result<MambaWrapper> {
    use crate::safetensors_load_mamba2;

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
            RepoId(hf::mamba2_130m::REPO_ID.into()),
            RepoType::Model,
            RevisionPath(hf::mamba2_130m::REVISION_PATH.into()),
        ));
        let filename = repo
            .get(&FilePath(
                hf::mamba2_130m::FILE_PATH_MODEL_SAFETENSORS.into(),
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

    let mut device: Device = Default::default();
    {
        device
            .configure((crate::PRECISION_FLOAT_D_TYPE, crate::PRECISION_INT_D_TYPE))
            .expect("Failed to install fp32/i32 device defaults");
    }

    let mamba_config = hf::mamba2_130m::config();
    let mamba = {
        let timing = web_time::Instant::now();
        log::info!("loading mamba data");
        let mamba_bytes = api.load_bytes(&mamba_filename).await.unwrap();
        log::info!("mamba data loaded in {}ms", timing.elapsed().as_millis()); // ~2-3s

        let timing = web_time::Instant::now();
        log::info!("initializing and loading mamba model");
        let mamba = safetensors_load_mamba2(&mamba_bytes, mamba_config.clone(), &device)?;
        log::info!(
            "mamba initialized and loaded in {}ms",
            timing.elapsed().as_millis()
        );

        mamba
    };

    let models = MambaWrapper::new(
        tokenizer,
        mamba,
        mamba_config,
    );

    Ok(models)
}
