mod safetensors_load;
pub mod token_output_stream;

use burn::prelude::*;
use burn_mamba;
use candle_transformers::generation::LogitsProcessor;
pub use safetensors_load::{load_param, safetensors_load};
use token_output_stream::*;
use tokenizers::Tokenizer;

pub mod hf {
    pub mod tokenizer {
        #[allow(unused_imports)]
        use hf_hub::types::{FilePath, RepoId};

        /// A [RepoId].
        pub const REPO_ID: &str = "EleutherAI/gpt-neox-20b";
        /// A [FilePath].
        pub const FILE_PATH_TOKENIZER_JSON: &str = "tokenizer.json";
    }

    pub mod mamba_130m {
        #[allow(unused_imports)]
        use hf_hub::types::{FilePath, RepoId, RevisionPath};

        /// A [RepoId].
        pub const REPO_ID: &str = "state-spaces/mamba-130m";
        /// A [RevisionPath].
        pub const REVISION_PATH: &str = "refs/pr/1";
        /// A [FilePath].
        pub const FILE_PATH_CONFIG_JSON: &str = "config.json";
        /// A [FilePath].
        pub const FILE_PATH_MODEL_SAFETENSORS: &str = "model.safetensors";

        pub const PADDED_VOCAB_SIZE: usize = 50280;
        pub const N_LAYER: usize = 24;
        pub const D_MODEL: usize = 768;
    }
}

pub struct MambaWrapper<B: Backend> {
    pub tokenizer: TokenOutputStream,
    pub mamba: burn_mamba::Mamba<B>,
    pub mamba_config: burn_mamba::MambaConfig,
}

pub struct LogitsProcessorWrapper {
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl<B: Backend> MambaWrapper<B> {
    pub fn new(
        tokenizer: Tokenizer,
        mamba: burn_mamba::Mamba<B>,
        mamba_config: burn_mamba::MambaConfig,
    ) -> Self {
        Self {
            tokenizer: TokenOutputStream::new(tokenizer),
            mamba,
            mamba_config,
        }
    }

    /// Clears the [Tokenizer] and returns the `prompt` as a list of Vocab tokens
    /// and also the eos token.
    pub fn reset_prompt(&mut self, prompt: &str) -> anyhow::Result<(Vec<usize>, usize)> {
        self.tokenizer.clear();
        let tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the </s> token"),
        };
        Ok((
            tokens.into_iter().map(|e| e as usize).collect(),
            eos_token as usize,
        ))
    }

    /// Initializes a list of empty (zero, null) [burn_mamba::step::MambaBlockCache] for a cached run.
    pub fn empty_caches(
        &self,
        batch: usize,
    ) -> anyhow::Result<Vec<burn_mamba::step::MambaBlockCache<B>>> {
        let device = self.mamba.embedding.weight.device();
        let len = self.mamba.layers.len();
        let mut caches = Vec::with_capacity(len);
        for _ in 0..len {
            let cache = burn_mamba::step::MambaBlockCacheConfig::new(
                batch,
                self.mamba_config.mamba_block.clone(),
            )
            .init(&device);
            caches.push(cache);
        }
        Ok(caches)
    }

    /// Reset and make up to `sample_len - 1` cacheless (training-friendly) calls to generate up to `sample_len - 1` tokens.
    pub fn run_cacheless(
        &mut self,
        prompt: &str,
        sample_len: usize,
        logits_processor_config: &mut LogitsProcessorWrapper,
    ) -> anyhow::Result<()> {
        use std::io::Write;
        let (mut tokens, eos_token) = self.reset_prompt(prompt)?;
        let device = self.mamba.embedding.weight.device();

        // prints the first token (if present), as this is used as *input* to the model
        if let Some(t) = tokens.first() {
            if let Some(t) = self.tokenizer.next_token(*t as u32)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut i = 0;
        'outer: while i < sample_len {
            let input: Tensor<B, 1, Int> = Tensor::from_data(tokens.as_slice(), &device);
            let input = input.unsqueeze();

            let logits_list = self.mamba.forward(input);
            let full_shape = logits_list.dims();
            let shape = (full_shape[2],);

            let logits_list = logits_list.into_data().to_vec::<f32>().unwrap();

            // logits contains an output for each timestep
            let logits_list = logits_list
                .chunks_exact(hf::mamba_130m::PADDED_VOCAB_SIZE)
                .skip(i)
                .map(|chunk: &[f32]| {
                    candle_core::Tensor::from_slice(chunk, shape, &candle_core::Device::Cpu)
                })
                .collect::<Result<Vec<_>, _>>()?;

            //

            for logits in logits_list.into_iter() {
                let next_token = logits_processor_config.add_logits(i, &mut tokens, logits)?;
                if next_token as usize == eos_token {
                    break 'outer;
                }

                // if the token has some valid representation, print it
                if let Some(t) = self.tokenizer.next_token(next_token as u32)? {
                    #[allow(unused_imports)]
                    use std::io::Write;
                    print!("{t}");
                    std::io::stdout().flush()?;
                }
                i += 1;
            }
        }
        if let Some(rest) = self.tokenizer.decode_rest().map_err(anyhow::Error::msg)? {
            print!("{rest}");
        }
        Ok(())
    }

    /// Reset and make up to `sample_len - 1` cached (inference-friendly) calls to generate up to `sample_len - 1` tokens.
    pub fn run_cached(
        &mut self,
        prompt: &str,
        sample_len: usize,
        logits_processor_config: &mut LogitsProcessorWrapper,
    ) -> anyhow::Result<()> {
        use std::io::Write;
        let (mut tokens, eos_token) = self.reset_prompt(prompt)?;

        // prints the first token (if present), as this is used as *input* to the model
        if let Some(t) = tokens.first() {
            if let Some(t) = self.tokenizer.next_token(*t as u32)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut caches = self.empty_caches(1)?;

        let mut i = 0;
        while i < sample_len {
            let next_logits = self.step(tokens[i], Some(&mut caches))?;
            let next_token = logits_processor_config.add_logits(i, &mut tokens, next_logits)?;
            if next_token == eos_token {
                break;
            }

            // if the token has some valid representation, print it
            if let Some(t) = self.tokenizer.next_token(next_token as u32)? {
                #[allow(unused_imports)]
                use std::io::Write;
                print!("{t}");
                std::io::stdout().flush()?;
            }

            i += 1;
        }
        if let Some(rest) = self.tokenizer.decode_rest().map_err(anyhow::Error::msg)? {
            print!("{rest}");
        }
        Ok(())
    }

    /// Make a cached call to generate a logits.
    ///
    /// `i` is the i-th call. For the first call, `i` should be `0`.
    pub fn step(
        &self,
        input: usize,
        mut caches: Option<&mut Vec<burn_mamba::step::MambaBlockCache<B>>>,
    ) -> anyhow::Result<candle_core::Tensor> {
        let device = self.mamba.embedding.weight.device();
        let input = Tensor::from_data([input], &device);
        let caches_owned = std::mem::take(&mut caches).unwrap();
        let (logits, new_caches) = self.mamba.step(input, caches_owned.clone());
        *caches_owned = new_caches;

        let full_shape = logits.dims();
        let shape = (full_shape[1],);

        let logits = logits.into_data().to_vec::<f32>().unwrap();

        let logits = candle_core::Tensor::from_vec(logits, shape, &candle_core::Device::Cpu)?
            .to_dtype(candle_core::DType::F32)?;
        Ok(logits)
    }
}

impl LogitsProcessorWrapper {
    pub fn new(
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            logits_processor,
            repeat_penalty,
            repeat_last_n,
        }
    }

    /// Add logits that represents a token.
    ///
    /// `i` is the i-th call. For the first call, `i` should be `0`.
    pub fn add_logits(
        &mut self,
        i: usize,
        tokens: &mut Vec<usize>,
        logits: candle_core::Tensor,
    ) -> anyhow::Result<usize> {
        let logits = if self.repeat_penalty == 1. {
            logits
        } else {
            let start_at = i.saturating_sub(self.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                self.repeat_penalty,
                tokens[start_at..i + 1]
                    .iter()
                    .map(|e| *e as u32)
                    .collect::<Vec<u32>>()
                    .as_slice(),
            )?
        };

        let next_token;
        if i + 1 < tokens.len() {
            // don't try to predict the next token (it was pre-defined)
            // also don't increment the "tokens" list (this token was already part of the list)
            next_token = tokens[i + 1];

            // should it still sample? idk
            // let _discarded_token = logits_processor.sample(&logits)?;
        } else {
            // try to predict the next token
            next_token = self.logits_processor.sample(&logits)? as usize;
            // add the token to the "tokens" list
            tokens.push(next_token as usize);
            // *generated_tokens += 1;
        }
        Ok(next_token)
    }
}
