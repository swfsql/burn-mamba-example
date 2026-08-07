pub mod hub;
pub mod sampling;
#[cfg(any(feature = "mamba1", feature = "mamba2"))]
mod store_load;
pub mod token_output_stream;
pub mod tokenizer;

#[cfg(any(feature = "mamba1", feature = "mamba2"))]
pub use store_load::{Checkpoint, load_mamba};

#[allow(unused_imports)]
use burn::prelude::*;
#[allow(unused_imports)]
use token_output_stream::*;
#[allow(unused_imports)]
use tokenizer::Tokenizer;

#[allow(unused_imports)]
use burn_mamba::prelude::*;
use sampling::LogitsProcessor;

#[allow(unused_imports)]
pub type Precision = f32;
use burn::tensor::{FloatDType, IntDType};
pub const PRECISION_FLOAT_D_TYPE: FloatDType = FloatDType::F32;
pub const PRECISION_INT_D_TYPE: IntDType = IntDType::I32;

pub mod hf {
    pub mod tokenizer {
        #[allow(unused_imports)]
        use crate::hub::{FilePath, RepoId};

        /// A [RepoId].
        pub const REPO_ID: &str = "EleutherAI/gpt-neox-20b";
        /// A [FilePath].
        pub const FILE_PATH_TOKENIZER_JSON: &str = "tokenizer.json";
    }

    #[cfg(feature = "mamba1")]
    pub mod mamba1_130m {
        #[allow(unused_imports)]
        use crate::hub::{FilePath, RepoId, RevisionPath};
        use burn_mamba::mamba1;
        use burn_mamba::prelude::*;

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

        pub fn config() -> MambaVocabNetConfig {
            MambaVocabNetConfig::Mamba1 {
                n_real_layers: N_LAYER, // 24
                n_virtual_layers: None,
                vocab_size: VOCAB_SIZE, // 50277
                pad_vocab_size_multiple: PAD_VOCAB_SIZE_MULTIPLE, // 8
                missing_lm_head: true,
                ignore_first_residual: false,
                ignore_last_residual: false,
                residuals: ResidualsConfig::Standard,
                mamba_block: mamba1::prelude::Mamba1Config::new(
                    D_MODEL, // 768
                )
                .with_state_rank(16) // default
                .with_conv_kernel(4) // default
                .with_expand(2) // default
                .with_has_proj_bias(false) // default
                .with_has_conv_bias(true) // default
            }
        }
    }

    #[cfg(feature = "mamba2")]
    pub mod mamba2_130m {
        #[allow(unused_imports)]
        use crate::hub::{FilePath, RepoId, RevisionPath};
        use burn_mamba::mamba2;
        use burn_mamba::prelude::*;

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

        pub fn config() -> MambaVocabNetConfig {
            MambaVocabNetConfig::Mamba2 {
                n_real_layers: N_LAYER, // 24
                n_virtual_layers: None,
                vocab_size: VOCAB_SIZE, // 50277
                pad_vocab_size_multiple: PAD_VOCAB_SIZE_MULTIPLE, // 16
                missing_lm_head: true,
                ignore_first_residual: false,
                ignore_last_residual: false,
                residuals: ResidualsConfig::Standard,
                mamba_block: mamba2::prelude::Mamba2Config::new(
                    D_MODEL, // 768
                )
                .with_state_rank(128) // default
                .with_conv_kernel(4) // default
                .with_expand(2) // default
                .with_per_head_dim(64) // default; n_heads = 768*2/64 = 24
                .with_ngroups(1) // default
                .with_is_norm_before_gate(false)
                .with_has_proj_bias(false) // default
                .with_has_conv_bias(true) // default
            }
        }
    }
}

#[cfg(any(feature = "mamba1", feature = "mamba2"))]
pub struct MambaWrapper {
    pub tokenizer: TokenOutputStream,
    pub mamba: MambaVocabNet,
    pub mamba_config: MambaVocabNetConfig,
}

pub struct LogitsProcessorWrapper {
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

#[cfg(any(feature = "mamba1", feature = "mamba2"))]
impl MambaWrapper
{
    pub fn new(tokenizer: Tokenizer, mamba: MambaVocabNet, mamba_config: MambaVocabNetConfig) -> Self {
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
        let tokens = self.tokenizer.tokenizer().encode(prompt);
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
    pub fn empty_caches(&self, batch: usize) -> anyhow::Result<MambaCaches> {
        let device = device(&self.mamba);
        let caches = empty_caches(batch, &self.mamba_config, &device);
        Ok(caches)
    }

    /// Reset and make up to `sample_len - 1` parallel (training-friendly) calls to generate up to `sample_len - 1` tokens.
    /// Returns how many tokens and the instant after the first token got generated.
    ///
    /// `mamba2_chunk_size`: Chunk size for Mamba2 selective scan. Defaults to 256. No effect for Mamba1.
    pub fn run_parallel(
        &mut self,
        prompt: &str,
        sample_len: usize,
        logits_processor_config: &mut LogitsProcessorWrapper,
    ) -> anyhow::Result<(usize, Option<std::time::Instant>)> {
        use std::io::Write;
        let (mut tokens, eos_token) = self.reset_prompt(prompt)?;
        let device = device(&self.mamba);

        // prints the first token (if present), as this is used as *input* to the model
        if let Some(t) = tokens.first() {
            if let Some(t) = self.tokenizer.next_token(*t as u32) {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut instant = None;
        let mut i = 0;
        'outer: while i < sample_len {
            let input: Tensor<1, Int> = Tensor::from_data(tokens.as_slice(), &device);
            let input = input.unsqueeze();

            let ssd_path = ssd_path(&self.mamba_config);
            let (logits_list, _caches) = self.mamba.forward(input, None, ssd_path);
            if i == 0 {
                instant = Some(std::time::Instant::now());
            }

            let logits_list = logits_list.into_data().to_vec::<Precision>().unwrap();

            // logits contains an output for each timestep
            let logits_list = logits_list
                .chunks_exact(self.padded_vocab_size())
                .skip(i)
                .map(<[Precision]>::to_vec)
                .collect::<Vec<_>>();

            //

            for logits in logits_list.into_iter() {
                let next_token = logits_processor_config.add_logits(i, &mut tokens, logits)?;
                if next_token == eos_token {
                    break 'outer;
                }

                // if the token has some valid representation, print it
                if let Some(t) = self.tokenizer.next_token(next_token as u32) {
                    #[allow(unused_imports)]
                    use std::io::Write;
                    print!("{t}");
                    std::io::stdout().flush()?;
                }
                i += 1;
            }
        }
        if let Some(rest) = self.tokenizer.decode_rest() {
            print!("{rest}");
        }
        Ok((i, instant))
    }

    /// Reset and make up to `sample_len - 1` sequential (inference-friendly) calls to generate up to `sample_len - 1` tokens.
    /// Returns how many tokens and the instant after the first token got generated.
    pub fn run_sequential(
        &mut self,
        prompt: &str,
        sample_len: usize,
        logits_processor_config: &mut LogitsProcessorWrapper,
    ) -> anyhow::Result<(usize, Option<std::time::Instant>)> {
        use std::io::Write;
        let (mut tokens, eos_token) = self.reset_prompt(prompt)?;

        // prints the first token (if present), as this is used as *input* to the model
        if let Some(t) = tokens.first() {
            if let Some(t) = self.tokenizer.next_token(*t as u32) {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut caches = self.empty_caches(1)?;

        let mut instant = None;
        let mut i = 0;
        while i < sample_len {
            let (next_logits, new_caches) = self.step(tokens[i], Some(caches))?;
            caches = new_caches;
            if i == 0 {
                instant = Some(std::time::Instant::now());
            }
            let next_token = logits_processor_config.add_logits(i, &mut tokens, next_logits)?;
            if next_token == eos_token {
                break;
            }

            // if the token has some valid representation, print it
            if let Some(t) = self.tokenizer.next_token(next_token as u32) {
                #[allow(unused_imports)]
                use std::io::Write;
                print!("{t}");
                std::io::stdout().flush()?;
            }

            i += 1;
        }
        if let Some(rest) = self.tokenizer.decode_rest() {
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
        caches: Option<MambaCaches>,
    ) -> anyhow::Result<(Vec<Precision>, MambaCaches)> {
        let device = device(&self.mamba);
        let input = Tensor::from_data([input], &device);

        let (logits, new_caches) = self.mamba.step(input, caches, None, None);
        assert_eq!([1, self.padded_vocab_size()], logits.dims());

        let logits = logits
            .cast(PRECISION_FLOAT_D_TYPE)
            .into_data()
            .to_vec::<Precision>()
            .unwrap();

        Ok((logits, new_caches))
    }

    pub fn padded_vocab_size(&self) -> usize {
        padded_vocab_size(&self.mamba_config)
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
        mut logits: Vec<Precision>,
    ) -> anyhow::Result<usize> {
        if self.repeat_penalty != 1. {
            let start_at = i.saturating_sub(self.repeat_last_n);
            let context = tokens[start_at..i + 1]
                .iter()
                .map(|e| *e as u32)
                .collect::<Vec<u32>>();
            sampling::apply_repeat_penalty(&mut logits, self.repeat_penalty, &context);
        }

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
            tokens.push(next_token);
            // *generated_tokens += 1;
        }
        Ok(next_token)
    }
}

#[cfg(any(feature = "mamba1", feature = "mamba2"))]
pub fn device(model: &MambaVocabNet) -> Device {
    match model {
        #[cfg(feature = "mamba1")]
        MambaVocabNet::Mamba1(m) => m.embedding.weight.device(),
        #[cfg(feature = "mamba2")]
        MambaVocabNet::Mamba2(m) => m.embedding.weight.device(),
    }
}

#[cfg(any(feature = "mamba1", feature = "mamba2"))]
#[allow(irrefutable_let_patterns)]
pub fn padded_vocab_size(config: &MambaVocabNetConfig) -> usize {
    let (vocab_size, pad_vocab_size_multiple) =
    match config {
        #[cfg(feature = "mamba1")]
        MambaVocabNetConfig::Mamba1{vocab_size, pad_vocab_size_multiple, ..} => {
            (vocab_size, pad_vocab_size_multiple)
        }
        #[cfg(feature = "mamba2")]
        MambaVocabNetConfig::Mamba2{vocab_size, pad_vocab_size_multiple, ..} => {
            (vocab_size, pad_vocab_size_multiple)
        }
    };

    if vocab_size % pad_vocab_size_multiple == 0 {
        *vocab_size
    } else {
        ((vocab_size / pad_vocab_size_multiple) + 1) * pad_vocab_size_multiple
    }
}

#[cfg(any(feature = "mamba1", feature = "mamba2"))]
pub fn empty_caches(batch: usize, mamba_config: &MambaVocabNetConfig, device: &Device) -> MambaCaches {
    match mamba_config {
        #[cfg(feature = "mamba1")]
        MambaVocabNetConfig::Mamba1 {n_real_layers, mamba_block, ..} => {
            let caches = Mamba1CachesConfig::new_from_block_config(*n_real_layers, batch, mamba_block.clone())
                .init(device);
            MambaCaches::Mamba1(caches)
        }
        #[cfg(feature = "mamba2")]
        MambaVocabNetConfig::Mamba2 {n_real_layers, mamba_block, ..} => {
            let caches =
            Mamba2CachesConfig::new_from_block_config(*n_real_layers, batch, mamba_block.clone())
                .init(device);
            MambaCaches::Mamba2(caches)
        }
    }
}

#[cfg(any(feature = "mamba1", feature = "mamba2"))]
pub fn ssd_path(mamba_config: &MambaVocabNetConfig) -> MambaSsdPath {
    match mamba_config {
        #[cfg(feature = "mamba1")]
        MambaVocabNetConfig::Mamba1 {..} => MambaSsdPath::Mamba1,
        #[cfg(feature = "mamba2")]
        MambaVocabNetConfig::Mamba2 {..} => MambaSsdPath::Mamba2(Mamba2SsdPath::SerialRecalculated(None)),
    }
}
