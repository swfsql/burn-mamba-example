mod safetensors_load;
pub mod token_output_stream;

#[cfg(feature = "mamba1")]
use burn_mamba::{mamba1, mamba1_block};
#[cfg(feature = "mamba1")]
pub use safetensors_load::safetensors_load_mamba1;

#[cfg(feature = "mamba2")]
use burn_mamba::{mamba2, mamba2_block};
#[cfg(feature = "mamba2")]
pub use safetensors_load::safetensors_load_mamba2;

#[allow(unused_imports)]
use burn::prelude::*;
#[allow(unused_imports)]
use token_output_stream::*;
#[allow(unused_imports)]
use tokenizers::Tokenizer;

use burn::tensor::DType;
use candle_transformers::generation::LogitsProcessor;
pub use safetensors_load::load_param_f32_to_f32;

pub type Precision = f32;
pub const PRECISION_D_TYPE: DType = DType::F32;
pub const CANDLE_PRECISION_D_TYPE: candle_core::DType = candle_core::DType::F32;

pub mod hf {
    pub mod tokenizer {
        #[allow(unused_imports)]
        use hf_hub::types::{FilePath, RepoId};

        /// A [RepoId].
        pub const REPO_ID: &str = "EleutherAI/gpt-neox-20b";
        /// A [FilePath].
        pub const FILE_PATH_TOKENIZER_JSON: &str = "tokenizer.json";
    }

    #[cfg(feature = "mamba1")]
    pub mod mamba1_130m {
        #[allow(unused_imports)]
        use hf_hub::types::{FilePath, RepoId, RevisionPath};

        /// A [RepoId].
        pub const REPO_ID: &str = "state-spaces/mamba-130m";
        /// A [RevisionPath].
        ///
        /// Safetensor PR conversion made by a bot.
        pub const REVISION_PATH: &str = "refs/pr/1";
        /// A [FilePath].
        pub const FILE_PATH_CONFIG_JSON: &str = "config.json";
        /// A [FilePath].
        pub const FILE_PATH_MODEL_SAFETENSORS: &str = "model.safetensors";

        pub const VOCAB_SIZE: usize = 50277;
        pub const PAD_VOCAB_SIZE_MULTIPLE: usize = 8;
        pub const N_LAYER: usize = 24;
        pub const D_MODEL: usize = 768;
    }

    #[cfg(feature = "mamba2")]
    pub mod mamba2_130m {
        #[allow(unused_imports)]
        use hf_hub::types::{FilePath, RepoId, RevisionPath};

        /// A [RepoId].
        pub const REPO_ID: &str = "state-spaces/mamba2-130m";
        /// A [RevisionPath].
        ///
        /// Safetensor PR conversion made by a bot.
        pub const REVISION_PATH: &str = "refs/pr/1";
        /// A [FilePath].
        pub const FILE_PATH_CONFIG_JSON: &str = "config.json";
        /// A [FilePath].
        pub const FILE_PATH_MODEL_SAFETENSORS: &str = "model.safetensors";

        pub const VOCAB_SIZE: usize = 50277;
        pub const PAD_VOCAB_SIZE_MULTIPLE: usize = 16;
        pub const N_LAYER: usize = 24;
        pub const D_MODEL: usize = 768;
    }
}

#[cfg(any(feature = "mamba1", feature = "mamba2"))]
pub struct MambaWrapper<B: Backend> {
    pub tokenizer: TokenOutputStream,
    pub mamba: MambaModel<B>,
    pub mamba_config: MambaModelConfig,
}

pub struct LogitsProcessorWrapper {
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

#[cfg(any(feature = "mamba1", feature = "mamba2"))]
impl<B: Backend> MambaWrapper<B> {
    pub fn new(tokenizer: Tokenizer, mamba: MambaModel<B>, mamba_config: MambaModelConfig) -> Self {
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
    pub fn empty_caches(&self, batch: usize) -> anyhow::Result<MambaBlockCaches<B>> {
        let device = self.mamba.device();
        let caches = MambaBlockCaches::empty_caches(batch, &self.mamba_config, &device);
        Ok(caches)
    }

    /// Reset and make up to `sample_len - 1` cacheless (training-friendly) calls to generate up to `sample_len - 1` tokens.
    /// Returns how many tokens and the instant after the first token got generated.
    ///
    /// `mamba2_chunk_size`: Chunk size for Mamba2 selective scan. Defaults to 256. No effect for Mamba1.
    pub fn run_cacheless(
        &mut self,
        prompt: &str,
        sample_len: usize,
        logits_processor_config: &mut LogitsProcessorWrapper,
        mamba2_chunk_size: Option<usize>,
    ) -> anyhow::Result<(usize, Option<std::time::Instant>)> {
        use std::io::Write;
        let (mut tokens, eos_token) = self.reset_prompt(prompt)?;
        let device = self.mamba.device();

        // prints the first token (if present), as this is used as *input* to the model
        if let Some(t) = tokens.first() {
            if let Some(t) = self.tokenizer.next_token(*t as u32)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut instant = None;
        let mut i = 0;
        'outer: while i < sample_len {
            let input: Tensor<B, 1, Int> = Tensor::from_data(tokens.as_slice(), &device);
            let input = input.unsqueeze();

            let logits_list = self.mamba.forward(input, mamba2_chunk_size);
            if i == 0 {
                instant = Some(std::time::Instant::now());
            }

            let full_shape = logits_list.dims();
            let shape = (full_shape[2],);

            let logits_list = logits_list.into_data().to_vec::<Precision>().unwrap();

            let padded_vocab_size = {
                #[cfg(feature = "mamba1")]
                use hf::mamba1_130m as m;

                #[cfg(feature = "mamba2")]
                use hf::mamba2_130m as m;

                if m::VOCAB_SIZE % m::PAD_VOCAB_SIZE_MULTIPLE == 0 {
                    m::VOCAB_SIZE
                } else {
                    ((m::VOCAB_SIZE / m::PAD_VOCAB_SIZE_MULTIPLE) + 1) * m::PAD_VOCAB_SIZE_MULTIPLE
                }
            };

            // logits contains an output for each timestep
            let logits_list = logits_list
                .chunks_exact(padded_vocab_size)
                .skip(i)
                .map(|chunk: &[_]| {
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
        Ok((i, instant))
    }

    /// Reset and make up to `sample_len - 1` cached (inference-friendly) calls to generate up to `sample_len - 1` tokens.
    /// Returns how many tokens and the instant after the first token got generated.
    pub fn run_cached(
        &mut self,
        prompt: &str,
        sample_len: usize,
        logits_processor_config: &mut LogitsProcessorWrapper,
    ) -> anyhow::Result<(usize, Option<std::time::Instant>)> {
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

        let mut instant = None;
        let mut i = 0;
        while i < sample_len {
            let next_logits = self.step(tokens[i], Some(&mut caches))?;
            if i == 0 {
                instant = Some(std::time::Instant::now());
            }
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
        Ok((i, instant))
    }

    /// Make a cached call to generate a logits.
    ///
    /// `i` is the i-th call. For the first call, `i` should be `0`.
    pub fn step(
        &self,
        input: usize,
        mut caches: Option<&mut MambaBlockCaches<B>>,
    ) -> anyhow::Result<candle_core::Tensor> {
        let device = self.mamba.device();
        let input = Tensor::from_data([input], &device);
        let caches_owned = std::mem::take(&mut caches).unwrap();

        let (logits, new_caches) = self.mamba.step(input, caches_owned.clone());
        assert_eq!([1, self.padded_vocab_size()], logits.dims());
        *caches_owned = new_caches;

        let shape = (self.padded_vocab_size(),);

        let logits = logits
            .cast(PRECISION_D_TYPE)
            .into_data()
            .to_vec::<Precision>()
            .unwrap();

        let logits = candle_core::Tensor::from_vec(logits, shape, &candle_core::Device::Cpu)?
            .to_dtype(CANDLE_PRECISION_D_TYPE)?;
        Ok(logits)
    }

    pub fn padded_vocab_size(&self) -> usize {
        self.mamba_config.padded_vocab_size()
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

#[derive(Clone, Debug)]
pub enum MambaVersion {
    #[cfg(feature = "mamba1")]
    Mamba1,
    #[cfg(feature = "mamba2")]
    Mamba2,
}

#[cfg(any(feature = "mamba1", feature = "mamba2"))]
#[derive(Debug)]
pub enum MambaModel<B: Backend> {
    #[cfg(feature = "mamba1")]
    Mamba1(mamba1::Mamba1<B>),
    #[cfg(feature = "mamba2")]
    Mamba2(mamba2::Mamba2<B>),
}

#[cfg(any(feature = "mamba1", feature = "mamba2"))]
#[derive(Clone, Debug)]
pub enum MambaModelConfig {
    #[cfg(feature = "mamba1")]
    Mamba1(mamba1::Mamba1Config),
    #[cfg(feature = "mamba2")]
    Mamba2(mamba2::Mamba2Config),
}

#[cfg(any(feature = "mamba1", feature = "mamba2"))]
#[derive(Clone, Debug)]
pub enum MambaBlockCaches<B: Backend> {
    #[cfg(feature = "mamba1")]
    Mamba1(Vec<mamba1_block::step::Mamba1BlockCache<B>>),
    #[cfg(feature = "mamba2")]
    Mamba2(Vec<mamba2_block::Mamba2BlockCache<B>>),
}

#[cfg(any(feature = "mamba1", feature = "mamba2"))]
impl<B: Backend> MambaModel<B> {
    pub fn version(&self) -> MambaVersion {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(_m) => MambaVersion::Mamba1,
            #[cfg(feature = "mamba2")]
            Self::Mamba2(_m) => MambaVersion::Mamba2,
        }
    }
    pub fn device(&self) -> B::Device {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(m) => m.embedding.weight.device(),
            #[cfg(feature = "mamba2")]
            Self::Mamba2(m) => m.embedding.weight.device(),
        }
    }
    pub fn layers_len(&self) -> usize {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(m) => m.layers.len(),
            #[cfg(feature = "mamba2")]
            Self::Mamba2(m) => m.layers.len(),
        }
    }
    /// `mamba2_chunk_size`: Chunk size for Mamba2 selective scan. Defaults to 256. No effect for Mamba1.
    pub fn forward(&self, x: Tensor<B, 2, Int>, mamba2_chunk_size: Option<usize>) -> Tensor<B, 3> {
        let res = match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(m) => m.forward(x),
            #[cfg(feature = "mamba2")]
            Self::Mamba2(m) => {
                let (y, _cache) = m.forward(x, mamba2_chunk_size);
                y
            }
        };
        let _mamba2_chunk_size = mamba2_chunk_size;
        res
    }
    pub fn step(
        &self,
        x: Tensor<B, 1, Int>,
        caches: MambaBlockCaches<B>,
    ) -> (Tensor<B, 2>, MambaBlockCaches<B>) {
        #[allow(irrefutable_let_patterns)]
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(m) => {
                let MambaBlockCaches::Mamba1(caches) = caches else {
                    unreachable!()
                };
                let (logits, new_caches) = m.step(x, caches.clone());
                let new_caches = MambaBlockCaches::Mamba1(new_caches);
                (logits, new_caches)
            }
            #[cfg(feature = "mamba2")]
            Self::Mamba2(m) => {
                let MambaBlockCaches::Mamba2(caches) = caches else {
                    unreachable!()
                };
                let (logits, new_caches) = m.step(x, caches.clone());
                let new_caches = MambaBlockCaches::Mamba2(new_caches);
                (logits, new_caches)
            }
        }
    }
}

#[cfg(any(feature = "mamba1", feature = "mamba2"))]
impl MambaModelConfig {
    pub fn version(&self) -> MambaVersion {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(_config) => MambaVersion::Mamba1,
            #[cfg(feature = "mamba2")]
            Self::Mamba2(_config) => MambaVersion::Mamba2,
        }
    }
    pub fn padded_vocab_size(&self) -> usize {
        #[allow(irrefutable_let_patterns)]
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(_config) => {
                use hf::mamba1_130m as m;
                if m::VOCAB_SIZE % m::PAD_VOCAB_SIZE_MULTIPLE == 0 {
                    m::VOCAB_SIZE
                } else {
                    ((m::VOCAB_SIZE / m::PAD_VOCAB_SIZE_MULTIPLE) + 1) * m::PAD_VOCAB_SIZE_MULTIPLE
                }
            }

            #[cfg(feature = "mamba2")]
            Self::Mamba2(_config) => {
                use hf::mamba2_130m as m;
                if m::VOCAB_SIZE % m::PAD_VOCAB_SIZE_MULTIPLE == 0 {
                    m::VOCAB_SIZE
                } else {
                    ((m::VOCAB_SIZE / m::PAD_VOCAB_SIZE_MULTIPLE) + 1) * m::PAD_VOCAB_SIZE_MULTIPLE
                }
            }
        }
    }
}

#[cfg(any(feature = "mamba1", feature = "mamba2"))]
impl<B: Backend> MambaBlockCaches<B> {
    pub fn empty_caches(batch: usize, mamba_config: &MambaModelConfig, device: &B::Device) -> Self {
        let mamba_version = mamba_config.version();
        #[allow(irrefutable_let_patterns)]
        match mamba_version {
            #[cfg(feature = "mamba1")]
            MambaVersion::Mamba1 => {
                let MambaModelConfig::Mamba1(config) = mamba_config else {
                    unreachable!()
                };
                let len = config.n_layer;
                let mut caches = Vec::with_capacity(len);
                for _ in 0..len {
                    let cache = mamba1_block::step::Mamba1BlockCacheConfig::new(
                        batch,
                        config.mamba_block.clone(),
                    )
                    .init::<B>(&device);
                    caches.push(cache);
                }
                MambaBlockCaches::Mamba1(caches)
            }
            #[cfg(feature = "mamba2")]
            MambaVersion::Mamba2 => {
                let MambaModelConfig::Mamba2(config) = mamba_config else {
                    unreachable!()
                };
                let len = config.n_layer;
                let mut caches = Vec::with_capacity(len);
                for _ in 0..len {
                    let cache = mamba2_block::Mamba2BlockCacheConfig::new(
                        batch,
                        config.mamba_block.clone(),
                    )
                    .init::<B>(&device);
                    caches.push(cache);
                }
                MambaBlockCaches::Mamba2(caches)
            }
        }
    }
}
