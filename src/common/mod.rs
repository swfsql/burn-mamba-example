pub mod hub;
pub mod sampling;
#[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
mod store_load;
pub mod token_output_stream;
pub mod tokenizer;

#[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
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

/// One compiled-in checkpoint: everything an entry point needs to fetch, build
/// and label a model, as plain data.
///
/// Any number of model features may be enabled at once; [hf::MODELS] is the
/// resulting runtime list (highest priority first) and [hf::preferred] is what
/// an entry point that runs a single model picks.
#[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
#[derive(Clone, Copy, Debug)]
pub struct ModelSpec {
    /// Selection name; matches the cargo feature and the `frontend/` directory.
    pub id: &'static str,
    /// Human-readable name, for logs and the browser UI's asset card.
    pub display_name: &'static str,
    /// A [hub::RepoId].
    pub repo_id: &'static str,
    /// A [hub::RevisionPath].
    pub revision_path: &'static str,
    /// A [hub::FilePath].
    pub file_path_model_safetensors: &'static str,
    /// A [hub::RepoId], of the repo this checkpoint's tokenizer comes from.
    pub tokenizer_repo_id: &'static str,
    /// A [hub::FilePath].
    pub file_path_tokenizer_json: &'static str,
    /// The hardcoded topology, mirroring the checkpoint's `config.json`.
    pub config: fn() -> MambaVocabNetConfig,
    /// The scan the parallel path takes, including the chunk length the
    /// checkpoint was trained with.
    pub ssd_path: fn() -> MambaSsdPath,
}

pub mod hf {
    #[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
    use crate::ModelSpec;

    /// Every checkpoint compiled into this build, **highest priority first**.
    ///
    /// The order is the priority: mamba3-mimo > mamba3-siso > mamba2 > mamba1.
    #[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
    pub const MODELS: &[&ModelSpec] = &[
        #[cfg(feature = "mamba3-mimo")]
        &mamba3_mimo_187m::SPEC,
        #[cfg(feature = "mamba3-siso")]
        &mamba3_siso_187m::SPEC,
        #[cfg(feature = "mamba2")]
        &mamba2_130m::SPEC,
        #[cfg(feature = "mamba1")]
        &mamba1_130m::SPEC,
    ];

    /// The checkpoint to use when exactly one model should run: the
    /// highest-priority entry of [MODELS].
    ///
    /// [None] only for a build with no checkpoint feature — `mamba3` on its own
    /// enables the blocks but neither 187m topology.
    #[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
    pub fn preferred() -> Option<&'static ModelSpec> {
        MODELS.first().copied()
    }

    /// Looks a compiled-in checkpoint up by [ModelSpec::id].
    #[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
    pub fn by_id(id: &str) -> Option<&'static ModelSpec> {
        MODELS.iter().copied().find(|model| model.id == id)
    }

    /// The [ModelSpec::id] of every compiled-in checkpoint, for error messages.
    #[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
    pub fn ids() -> Vec<&'static str> {
        MODELS.iter().map(|model| model.id).collect()
    }

    /// The tokenizer of the Mamba-1 / Mamba-2 130m checkpoints.
    pub mod tokenizer {
        #[allow(unused_imports)]
        use crate::hub::{FilePath, RepoId};

        /// A [RepoId].
        pub const REPO_ID: &str = "EleutherAI/gpt-neox-20b";
        /// A [FilePath].
        pub const FILE_PATH_TOKENIZER_JSON: &str = "tokenizer.json";
    }

    /// The tokenizer of the Mamba-3 187m checkpoints (`vocab_size: 128256`).
    ///
    /// Their model cards name `meta-llama/Llama-3.1-8B`, whose repo is gated —
    /// the hub client sends no credentials, and the deployed wasm page has none
    /// to send. This is an ungated mirror carrying the same `tokenizer.json`
    /// pipeline and the same 128256 ids.
    #[cfg(feature = "mamba3")]
    pub mod tokenizer_llama31 {
        #[allow(unused_imports)]
        use crate::hub::{FilePath, RepoId};

        /// A [RepoId].
        pub const REPO_ID: &str = "unsloth/Meta-Llama-3.1-8B";
        /// A [FilePath].
        pub const FILE_PATH_TOKENIZER_JSON: &str = "tokenizer.json";
    }

    #[cfg(feature = "mamba1")]
    pub mod mamba1_130m {
        #[allow(unused_imports)]
        use crate::hub::{FilePath, RepoId, RevisionPath};
        use burn_mamba::mamba1;
        use burn_mamba::prelude::*;

        /// This checkpoint's entry in [super::MODELS].
        pub const SPEC: crate::ModelSpec = crate::ModelSpec {
            id: "mamba1",
            display_name: DISPLAY_NAME,
            repo_id: REPO_ID,
            revision_path: REVISION_PATH,
            file_path_model_safetensors: FILE_PATH_MODEL_SAFETENSORS,
            tokenizer_repo_id: tokenizer_source::REPO_ID,
            file_path_tokenizer_json: tokenizer_source::FILE_PATH_TOKENIZER_JSON,
            config,
            ssd_path,
        };

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

        /// Where this checkpoint's `tokenizer.json` comes from.
        pub use super::tokenizer as tokenizer_source;

        /// Shown on the browser UI's asset card.
        pub const DISPLAY_NAME: &str = "Mamba-130m";

        pub const VOCAB_SIZE: usize = 50277;
        pub const PAD_VOCAB_SIZE_MULTIPLE: usize = 8;
        pub const N_LAYER: usize = 24;
        pub const D_MODEL: usize = 768;

        pub fn config() -> MambaVocabNetConfig {
            MambaVocabNetConfig::Mamba1 {
                n_real_layers: N_LAYER, // 24
                n_virtual_layers: None,
                vocab_size: VOCAB_SIZE,                           // 50277
                pad_vocab_size_multiple: PAD_VOCAB_SIZE_MULTIPLE, // 8
                missing_lm_head: true,
                ignore_first_residual: false,
                ignore_last_residual: false,
                residuals: ResidualsConfig::Standard,
                // The 130m checkpoints are mixer-only (`d_intermediate: 0`).
                mlp: None,
                mamba_block: mamba1::prelude::Mamba1Config::new(
                    D_MODEL, // 768
                )
                .with_state_rank(16) // default
                .with_conv_kernel(4) // default
                .with_expand(2) // default
                .with_has_proj_bias(false) // default
                .with_has_conv_bias(true), // default
            }
        }

        pub fn ssd_path() -> MambaSsdPath {
            MambaSsdPath::Mamba1
        }
    }

    #[cfg(feature = "mamba2")]
    pub mod mamba2_130m {
        #[allow(unused_imports)]
        use crate::hub::{FilePath, RepoId, RevisionPath};
        use burn_mamba::mamba2;
        use burn_mamba::prelude::*;

        /// This checkpoint's entry in [super::MODELS].
        pub const SPEC: crate::ModelSpec = crate::ModelSpec {
            id: "mamba2",
            display_name: DISPLAY_NAME,
            repo_id: REPO_ID,
            revision_path: REVISION_PATH,
            file_path_model_safetensors: FILE_PATH_MODEL_SAFETENSORS,
            tokenizer_repo_id: tokenizer_source::REPO_ID,
            file_path_tokenizer_json: tokenizer_source::FILE_PATH_TOKENIZER_JSON,
            config,
            ssd_path,
        };

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

        /// Where this checkpoint's `tokenizer.json` comes from.
        pub use super::tokenizer as tokenizer_source;

        /// Shown on the browser UI's asset card.
        pub const DISPLAY_NAME: &str = "Mamba2-130m";

        pub const VOCAB_SIZE: usize = 50277;
        pub const PAD_VOCAB_SIZE_MULTIPLE: usize = 16;
        pub const N_LAYER: usize = 24;
        pub const D_MODEL: usize = 768;

        pub fn config() -> MambaVocabNetConfig {
            MambaVocabNetConfig::Mamba2 {
                n_real_layers: N_LAYER, // 24
                n_virtual_layers: None,
                vocab_size: VOCAB_SIZE,                           // 50277
                pad_vocab_size_multiple: PAD_VOCAB_SIZE_MULTIPLE, // 16
                missing_lm_head: true,
                ignore_first_residual: false,
                ignore_last_residual: false,
                residuals: ResidualsConfig::Standard,
                // The 130m checkpoints are mixer-only (`d_intermediate: 0`).
                mlp: None,
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
                .with_has_conv_bias(true), // default
            }
        }

        pub fn ssd_path() -> MambaSsdPath {
            MambaSsdPath::Mamba2(Mamba2SsdPath::SerialRecalculated(None))
        }
    }

    /// Shared shape of the two official Mamba-3 187m checkpoints.
    ///
    /// Both are 12 layers of `Mamba3` mixer + SwiGLU MLP over the Llama-3.1
    /// vocabulary, and differ only in `mimo_rank`, `d_intermediate` and the
    /// checkpoint's `chunk_size`. The fields below mirror their `config.json`;
    /// `is_outproj_norm: false` and `attn_layer_idx: []` in both, so there is no
    /// `out_norm` and no attention layer to model.
    #[cfg(feature = "mamba3")]
    macro_rules! mamba3_187m {
        (
            $module:ident,
            id = $id:literal,
            repo = $repo:literal,
            display_name = $display_name:literal,
            d_intermediate = $d_intermediate:expr,
            mimo_rank = $mimo_rank:expr,
            chunk_size = $chunk_size:expr,
        ) => {
            pub mod $module {
                #[allow(unused_imports)]
                use crate::hub::{FilePath, RepoId, RevisionPath};
                use burn_mamba::mamba3;
                use burn_mamba::modules::GatedMlpConfig;
                use burn_mamba::prelude::*;

                /// This checkpoint's entry in [super::MODELS] — only listed
                /// there under this topology's own feature, since `mamba3`
                /// alone picks neither.
                pub const SPEC: crate::ModelSpec = crate::ModelSpec {
                    id: $id,
                    display_name: DISPLAY_NAME,
                    repo_id: REPO_ID,
                    revision_path: REVISION_PATH,
                    file_path_model_safetensors: FILE_PATH_MODEL_SAFETENSORS,
                    tokenizer_repo_id: tokenizer_source::REPO_ID,
                    file_path_tokenizer_json: tokenizer_source::FILE_PATH_TOKENIZER_JSON,
                    config,
                    ssd_path,
                };

                /// A [RepoId].
                pub const REPO_ID: &str = $repo;
                /// A [RevisionPath].
                ///
                /// Safetensor PR conversion made by a bot — `main` carries only
                /// `pytorch_model.bin`, which `burn-store` cannot read.
                pub const REVISION_PATH: &str = "refs/pr/1";
                /// A [FilePath].
                pub const FILE_PATH_CONFIG_JSON: &str = "config.json";
                /// A [FilePath].
                pub const FILE_PATH_MODEL_SAFETENSORS: &str = "model.safetensors";

                /// Where this checkpoint's `tokenizer.json` comes from.
                pub use super::tokenizer_llama31 as tokenizer_source;

                /// Shown on the browser UI's asset card.
                pub const DISPLAY_NAME: &str = $display_name;

                pub const VOCAB_SIZE: usize = 128256;
                pub const PAD_VOCAB_SIZE_MULTIPLE: usize = 16;
                pub const N_LAYER: usize = 12;
                pub const D_MODEL: usize = 768;
                pub const D_INTERMEDIATE: usize = $d_intermediate;
                pub const MIMO_RANK: usize = $mimo_rank;
                /// The SSD chunk length the checkpoint was trained with.
                pub const CHUNK_SIZE: usize = $chunk_size;

                /// The per-layer SwiGLU feed-forward. Note `GatedMlpConfig`
                /// rounds `d_intermediate` up to a multiple of 128, which is why
                /// the MIMO checkpoint's `fc1` is `[2·1280, 768]` while its
                /// `config.json` says `1264`.
                pub fn mlp() -> GatedMlpConfig {
                    GatedMlpConfig::new(D_MODEL, D_INTERMEDIATE)
                }

                pub fn config() -> MambaVocabNetConfig {
                    MambaVocabNetConfig::Mamba3 {
                        n_real_layers: N_LAYER, // 12
                        n_virtual_layers: None,
                        vocab_size: VOCAB_SIZE, // 128256
                        pad_vocab_size_multiple: PAD_VOCAB_SIZE_MULTIPLE, // 16
                        missing_lm_head: true,  // tie_embeddings: true
                        ignore_first_residual: false,
                        ignore_last_residual: false,
                        residuals: ResidualsConfig::Standard,
                        mlp: Some(mlp()),
                        mamba_block: mamba3::prelude::Mamba3Config::new(
                            D_MODEL, // 768
                        )
                        .with_state_rank(128) // default
                        .with_expand(2) // default
                        .with_per_head_dim(64) // default; n_heads = 768*2/64 = 24
                        .with_ngroups(1) // default
                        .with_mimo_rank(MIMO_RANK)
                        .with_rope_fraction(0.5) // default
                        .with_a_floor(1e-4) // default
                        .with_dt_min(1e-3) // default
                        .with_dt_max(0.1) // default
                        .with_dt_init_floor(1e-4) // default
                        .with_has_outproj_norm(false) // is_outproj_norm: false
                        .with_has_proj_bias(false) // default
                        // Identical values and grads either way; `false` suits
                        // the CPU backends this demo decodes on (flex natively,
                        // and in the browser), where the specialised per-token
                        // kernel is markedly slower than the plain matmul.
                        .with_siso_specialization_decode(false),
                    }
                }

                pub fn ssd_path() -> MambaSsdPath {
                    MambaSsdPath::Mamba3(Mamba3SsdPath::SerialRecalculated(Some(CHUNK_SIZE)))
                }
            }
        };
    }

    #[cfg(feature = "mamba3")]
    mamba3_187m!(
        mamba3_siso_187m,
        id = "mamba3-siso",
        repo = "state-spaces/mamba3-siso-187m",
        display_name = "Mamba3-SISO-187m",
        d_intermediate = 1536,
        mimo_rank = 1,
        chunk_size = 64,
    );

    #[cfg(feature = "mamba3")]
    mamba3_187m!(
        mamba3_mimo_187m,
        id = "mamba3-mimo",
        repo = "state-spaces/mamba3-mimo-187m",
        display_name = "Mamba3-MIMO-187m",
        d_intermediate = 1264,
        mimo_rank = 4,
        chunk_size = 16,
    );
}

#[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
pub struct MambaWrapper {
    /// Which checkpoint this is — the run modes read the topology and the scan
    /// path from it, so a build carrying several models keeps them apart.
    pub spec: &'static ModelSpec,
    pub tokenizer: TokenOutputStream,
    pub mamba: MambaVocabNet,
    pub mamba_config: MambaVocabNetConfig,
}

pub struct LogitsProcessorWrapper {
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

#[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
impl MambaWrapper {
    pub fn new(spec: &'static ModelSpec, tokenizer: Tokenizer, mamba: MambaVocabNet) -> Self {
        Self {
            spec,
            tokenizer: TokenOutputStream::new(tokenizer),
            mamba,
            mamba_config: (spec.config)(),
        }
    }

    /// The end-of-sequence token names of the supported tokenizers, tried in
    /// order. `EleutherAI/gpt-neox-20b` spells it `<|endoftext|>` and
    /// `meta-llama/Llama-3.1-8B` spells it `<|end_of_text|>`; neither vocabulary
    /// contains the other's name, so probing is unambiguous.
    const EOS_TOKENS: [&str; 2] = ["<|endoftext|>", "<|end_of_text|>"];

    /// Clears the [Tokenizer] and returns the `prompt` as a list of Vocab tokens
    /// and also the eos token.
    pub fn reset_prompt(&mut self, prompt: &str) -> anyhow::Result<(Vec<usize>, usize)> {
        self.tokenizer.clear();
        let tokens = self.tokenizer.tokenizer().encode(prompt);
        let eos_token = Self::EOS_TOKENS
            .iter()
            .find_map(|name| self.tokenizer.get_token(name))
            .ok_or_else(|| {
                anyhow::anyhow!("cannot find an eos token among {:?}", Self::EOS_TOKENS)
            })?;
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

            let ssd_path = (self.spec.ssd_path)();
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

#[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
pub fn device(model: &MambaVocabNet) -> Device {
    match model {
        #[cfg(feature = "mamba1")]
        MambaVocabNet::Mamba1(m) => m.embedding.weight.device(),
        #[cfg(feature = "mamba2")]
        MambaVocabNet::Mamba2(m) => m.embedding.weight.device(),
        #[cfg(feature = "mamba3")]
        MambaVocabNet::Mamba3(m) => m.embedding.weight.device(),
    }
}

#[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
#[allow(irrefutable_let_patterns)]
pub fn padded_vocab_size(config: &MambaVocabNetConfig) -> usize {
    let (vocab_size, pad_vocab_size_multiple) = match config {
        #[cfg(feature = "mamba1")]
        MambaVocabNetConfig::Mamba1 {
            vocab_size,
            pad_vocab_size_multiple,
            ..
        } => (vocab_size, pad_vocab_size_multiple),
        #[cfg(feature = "mamba2")]
        MambaVocabNetConfig::Mamba2 {
            vocab_size,
            pad_vocab_size_multiple,
            ..
        } => (vocab_size, pad_vocab_size_multiple),
        #[cfg(feature = "mamba3")]
        MambaVocabNetConfig::Mamba3 {
            vocab_size,
            pad_vocab_size_multiple,
            ..
        } => (vocab_size, pad_vocab_size_multiple),
    };

    if vocab_size % pad_vocab_size_multiple == 0 {
        *vocab_size
    } else {
        ((vocab_size / pad_vocab_size_multiple) + 1) * pad_vocab_size_multiple
    }
}

#[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
pub fn empty_caches(
    batch: usize,
    mamba_config: &MambaVocabNetConfig,
    device: &Device,
) -> MambaCaches {
    match mamba_config {
        #[cfg(feature = "mamba1")]
        MambaVocabNetConfig::Mamba1 {
            n_real_layers,
            mamba_block,
            ..
        } => {
            let caches = Mamba1CachesConfig::new_from_block_config(
                *n_real_layers,
                batch,
                mamba_block.clone(),
            )
            .init(device);
            MambaCaches::Mamba1(caches)
        }
        #[cfg(feature = "mamba2")]
        MambaVocabNetConfig::Mamba2 {
            n_real_layers,
            mamba_block,
            ..
        } => {
            let caches = Mamba2CachesConfig::new_from_block_config(
                *n_real_layers,
                batch,
                mamba_block.clone(),
            )
            .init(device);
            MambaCaches::Mamba2(caches)
        }
        #[cfg(feature = "mamba3")]
        MambaVocabNetConfig::Mamba3 {
            n_real_layers,
            mamba_block,
            ..
        } => {
            // Mamba-3 caches are pathway-tagged, and the supplied cache is what
            // selects the pathway. Single-SSD is the one a missing cache would
            // have defaulted to, and the one `step` decodes through.
            use burn_mamba::mamba3::single_ssd::prelude::Mamba3SingleSsdCachesConfig;
            let caches = Mamba3SingleSsdCachesConfig::new_from_block_config(
                *n_real_layers,
                batch,
                mamba_block.clone(),
            )
            .init(device);
            MambaCaches::Mamba3(Mamba3Caches::SingleSsd(caches))
        }
    }
}

/// Checks the compiled-in checkpoint list, whatever set of model features this
/// build happens to enable.
#[cfg(all(test, any(feature = "mamba1", feature = "mamba2", feature = "mamba3")))]
mod tests {
    use super::*;

    /// Highest priority first, and every [ModelSpec::id] resolvable — that
    /// ordering is what the entry points take when they run a single model.
    #[test]
    fn models_are_listed_by_descending_priority() {
        const PRIORITY: [&str; 4] = ["mamba3-mimo", "mamba3-siso", "mamba2", "mamba1"];

        let ids = hf::ids();
        let ranks: Vec<usize> = ids
            .iter()
            .map(|id| {
                PRIORITY
                    .iter()
                    .position(|known| known == id)
                    .unwrap_or_else(|| panic!("{id:?} is missing from the priority list"))
            })
            .collect();
        assert!(
            ranks.windows(2).all(|pair| pair[0] < pair[1]),
            "{ids:?} is not ordered by descending priority"
        );

        for id in &ids {
            assert_eq!(hf::by_id(id).map(|model| model.id), Some(*id));
        }
        assert_eq!(hf::preferred().map(|model| model.id), ids.first().copied());
    }

    /// The two Mamba-3 checkpoints were trained with different chunk lengths, so
    /// each one's scan must carry its own — a build with both compiled in still
    /// keeps them apart.
    #[cfg(feature = "mamba3")]
    #[test]
    fn each_mamba3_checkpoint_keeps_its_own_chunk_size() {
        let checkpoints = [
            (hf::mamba3_siso_187m::SPEC, hf::mamba3_siso_187m::CHUNK_SIZE),
            (hf::mamba3_mimo_187m::SPEC, hf::mamba3_mimo_187m::CHUNK_SIZE),
        ];
        assert_ne!(checkpoints[0].1, checkpoints[1].1);

        for (spec, chunk_size) in checkpoints {
            let path = (spec.ssd_path)();
            let MambaSsdPath::Mamba3(Mamba3SsdPath::SerialRecalculated(Some(got))) = path else {
                panic!("{}: expected a chunked Mamba-3 scan, got {path:?}", spec.id)
            };
            assert_eq!(got, chunk_size, "{}", spec.id);
        }
    }
}
