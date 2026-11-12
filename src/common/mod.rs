//! The core of the app. Each front-end and each backend uses it: the
//! checkpoints ([hf]), the build of the network, the tokenizer, the sampler and
//! the run wrapper.
//!
//! The backend is a run-time [Device]. Burn sends each tensor op to the backend
//! of its device, so no type takes a backend generic. `BURN_DEVICE` selects one
//! of the compiled-in backends. Before it builds a network, each entry point
//! calls `Device::default()`, then `configure` with [PRECISION_FLOAT_D_TYPE]
//! and [PRECISION_INT_D_TYPE]: f32 and i32 (ADR-0003).

#[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
pub mod assets;
pub mod hub;
pub mod sampling;
#[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
mod store_load;
pub mod token_output_stream;
pub mod tokenizer;

#[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
pub use store_load::{WeightsSource, build_network, tie_lm_head};

#[allow(unused_imports)]
use burn::prelude::*;
#[allow(unused_imports)]
use token_output_stream::*;
#[allow(unused_imports)]
use tokenizer::Tokenizer;

#[allow(unused_imports)]
use burn_mamba::prelude::*;
use sampling::LogitsProcessor;

/// The Rust type of a float of the network: f32. It agrees with
/// [PRECISION_FLOAT_D_TYPE] (ADR-0003).
#[allow(unused_imports)]
pub type Precision = f32;
use burn::tensor::{FloatDType, IntDType};
/// The float dtype of the device and of the loaded weights: f32 (ADR-0003).
pub const PRECISION_FLOAT_D_TYPE: FloatDType = FloatDType::F32;
/// The int dtype of the device: i32 (ADR-0003).
pub const PRECISION_INT_D_TYPE: IntDType = IntDType::I32;

/// One compiled-in checkpoint, as plain data: the values that an entry point
/// needs to fetch its files, build its network and show its name.
///
/// A build can have any set of checkpoint features. The list
/// [hf::CHECKPOINTS] holds the checkpoints of the build, highest priority
/// first. The function [hf::preferred] gives its first entry.
#[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
#[derive(Clone, Copy, Debug)]
pub struct CheckpointSpec {
    /// The checkpoint id. It is also the checkpoint feature, the `frontend/`
    /// directory, the `MAMBA_CHECKPOINT` value and the bench case name.
    pub id: &'static str,
    /// The name for the logs and for the asset card of the browser page.
    pub display_name: &'static str,
    /// The hub repo of the weights file, as a [hub::RepoId].
    pub repo_id: &'static str,
    /// The revision of the weights file, as a [hub::RevisionPath].
    pub revision_path: &'static str,
    /// The weights file in its repo, as a [hub::FilePath].
    pub file_path_model_safetensors: &'static str,
    /// The hub repo of the tokenizer file, as a [hub::RepoId].
    pub tokenizer_repo_id: &'static str,
    /// The tokenizer file in its repo, as a [hub::FilePath].
    pub file_path_tokenizer_json: &'static str,
    /// The hardcoded topology. It copies the `config.json` of the checkpoint
    /// (ADR-0002).
    pub topology: fn() -> MambaVocabNetConfig,
    /// The scan that a forward uses. For a Mamba-3 checkpoint, it holds the
    /// chunk length of the training of the checkpoint.
    pub ssd_path: fn() -> MambaSsdPath,
}

pub mod hf {
    //! The checkpoints of the app: the hub files and the hardcoded topology of
    //! each checkpoint (ADR-0002), and `CHECKPOINTS`, the checkpoints of this
    //! build (ADR-0004).
    //!
    //! Each value of a topology copies the `config.json` of its checkpoint, or
    //! the signature of the official code where `config.json` has no value. A
    //! comment beside the value names its source. Two checkpoints with one
    //! tokenizer can still differ: `pad_vocab_size_multiple` is 8 for
    //! `mamba1-130m` and 16 for `mamba2-130m`.
    //!
    //! Code that depends on the checkpoint reads its `CheckpointSpec`. A value
    //! that the topology does not hold, such as the chunk length of a Mamba-3
    //! checkpoint, goes into the spec.
    //!
    //! # Add a checkpoint
    //!
    //! 1. The spec: a module here with `SPEC`, `topology()` and `ssd_path()`. A
    //!    new tokenizer also needs a module here, and the tokenizer reader must
    //!    support its pipeline.
    //! 2. The list entry: an entry of `CHECKPOINTS` at its priority. Its id
    //!    goes into the doc of `CHECKPOINTS` and into the priority test of this
    //!    file. A Mamba-3 checkpoint also goes into the two Mamba-3 tests.
    //! 3. The features: its checkpoint feature in `Cargo.toml`, which adds its
    //!    family feature, and its id in the priority comment there. Its id goes
    //!    into the `compile_error!` blocks of `src/native/mod.rs`,
    //!    `src/wasm/mod.rs` and `benches/model.rs`.
    //! 4. The manifest: its tensors in the tests of `src/common/store_load.rs`,
    //!    in a `*_MANIFEST` and a `VENDORED` entry.
    //! 5. The `frontend/<id>/` directory, with an `index.html` and an
    //!    `index.js`.
    //! 6. The deploy step: a bundle build in `.github/workflows/deploy.yml`, and
    //!    its id in the loop of that workflow.
    //! 7. Its id in the lists of `scripts/check.sh`, `bench.sh` and
    //!    `kernels.sh`. Its id and its live demo in `README.md`.

    #[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
    use crate::CheckpointSpec;

    /// Every checkpoint compiled into this build, **highest priority first**.
    ///
    /// The order is the priority: mamba3-mimo-187m > mamba3-siso-187m >
    /// mamba2-130m > mamba1-130m.
    ///
    /// A family feature compiles the spec of each checkpoint of its family, so
    /// the tests can check each of them. Only a checkpoint feature adds an entry
    /// to this list, and an entry point runs only an entry of this list.
    #[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
    pub const CHECKPOINTS: &[&CheckpointSpec] = &[
        #[cfg(feature = "mamba3-mimo-187m")]
        &mamba3_mimo_187m::SPEC,
        #[cfg(feature = "mamba3-siso-187m")]
        &mamba3_siso_187m::SPEC,
        #[cfg(feature = "mamba2-130m")]
        &mamba2_130m::SPEC,
        #[cfg(feature = "mamba1-130m")]
        &mamba1_130m::SPEC,
    ];

    /// The first entry of [CHECKPOINTS]: the checkpoint with the highest
    /// priority.
    ///
    /// It is [None] only in a build with no checkpoint feature. A family feature
    /// alone adds the blocks, but no checkpoint.
    #[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
    pub fn preferred() -> Option<&'static CheckpointSpec> {
        CHECKPOINTS.first().copied()
    }

    /// Looks a compiled-in checkpoint up by [CheckpointSpec::id].
    #[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
    pub fn by_id(id: &str) -> Option<&'static CheckpointSpec> {
        CHECKPOINTS.iter().copied().find(|spec| spec.id == id)
    }

    /// The [CheckpointSpec::id] of every compiled-in checkpoint, for error
    /// messages.
    #[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
    pub fn ids() -> Vec<&'static str> {
        CHECKPOINTS.iter().map(|spec| spec.id).collect()
    }

    /// The layer stack of the reference `MixerModel` (`mixer_seq_simple.py` of
    /// `mamba_ssm`), which the four checkpoints share. This function sets each
    /// field, so a later change in burn-stack cannot change a checkpoint.
    #[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
    fn reference_stack(
        n_layer: usize,
        mlp: Option<burn_mamba::prelude::GatedMlpConfig>,
    ) -> burn_mamba::prelude::NetworkShape {
        use burn_mamba::prelude::*;
        NetworkShape {
            n_real_layers: n_layer,
            // `MixerModel.forward` runs each of its `n_layer` layers once.
            n_virtual_layers: None,
            // The reference has no gradient horizon.
            grad_horizon: None,
            // The reference has no class latents.
            class_latents: Vec::new(),
            // `Block.forward` (`block.py`) adds the residual of each layer.
            // `MixerModel.forward` adds the residual of the last layer before
            // `norm_f`.
            ignore_first_residual: false,
            ignore_last_residual: false,
            // `residual = hidden_states + residual`: one additive residual
            // stream.
            residuals: ResidualsConfig::Standard,
            mlp,
            // The reference has no untied parameters.
            untied: Vec::new(),
            // The load writes each weight of the checkpoint. The flag
            // embedding, which no checkpoint holds, keeps its zero init.
            init: None,
        }
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
    /// Their model cards name `meta-llama/Llama-3.1-8B`. That repo is gated: it
    /// needs credentials. The hub client sends none, and the browser page has
    /// none to send. This repo is an ungated mirror, with the same
    /// `tokenizer.json` pipeline and the same 128256 ids.
    #[cfg(feature = "mamba3")]
    pub mod tokenizer_llama31 {
        #[allow(unused_imports)]
        use crate::hub::{FilePath, RepoId};

        /// A [RepoId].
        pub const REPO_ID: &str = "unsloth/Meta-Llama-3.1-8B";
        /// A [FilePath].
        pub const FILE_PATH_TOKENIZER_JSON: &str = "tokenizer.json";
    }

    /// The `state-spaces/mamba-130m` checkpoint, of the Mamba-1 family.
    #[cfg(feature = "mamba1")]
    pub mod mamba1_130m {
        #[allow(unused_imports)]
        use crate::hub::{FilePath, RepoId, RevisionPath};
        use burn_mamba::mamba1;
        use burn_mamba::prelude::*;

        /// The spec of this checkpoint. [super::CHECKPOINTS] lists it only with
        /// its checkpoint feature, `mamba1-130m`.
        pub const SPEC: crate::CheckpointSpec = crate::CheckpointSpec {
            id: "mamba1-130m",
            display_name: DISPLAY_NAME,
            repo_id: REPO_ID,
            revision_path: REVISION_PATH,
            file_path_model_safetensors: FILE_PATH_MODEL_SAFETENSORS,
            tokenizer_repo_id: tokenizer_source::REPO_ID,
            file_path_tokenizer_json: tokenizer_source::FILE_PATH_TOKENIZER_JSON,
            topology,
            ssd_path,
        };

        /// A [RepoId].
        pub const REPO_ID: &str = "state-spaces/mamba-130m";
        /// A [RevisionPath].
        ///
        /// The pull request where a bot converted the weights to safetensors.
        pub const REVISION_PATH: &str = "refs/pr/1";
        /// A [FilePath].
        pub const FILE_PATH_CONFIG_JSON: &str = "config.json";
        /// A [FilePath].
        pub const FILE_PATH_MODEL_SAFETENSORS: &str = "model.safetensors";

        /// The source of the `tokenizer.json` of this checkpoint.
        pub use super::tokenizer as tokenizer_source;

        /// The name on the asset card of the browser page.
        pub const DISPLAY_NAME: &str = "Mamba-130m";

        /// `vocab_size` in `config.json`.
        pub const VOCAB_SIZE: usize = 50277;
        /// `pad_vocab_size_multiple` in `config.json`.
        pub const PAD_VOCAB_SIZE_MULTIPLE: usize = 8;
        /// `n_layer` in `config.json`.
        pub const N_LAYER: usize = 24;
        /// `d_model` in `config.json`.
        pub const D_MODEL: usize = 768;

        /// The topology of this checkpoint. A comment beside each value names
        /// its source.
        pub fn topology() -> MambaVocabNetConfig {
            MambaVocabNetConfig {
                shape: VocabShape {
                    vocab_size: VOCAB_SIZE, // vocab_size: 50277
                    // `config.json` has no `d_intermediate`, and
                    // `MambaConfig.d_intermediate = 0`: no feed-forward.
                    stack: super::reference_stack(N_LAYER, None), // n_layer: 24
                    pad_vocab_size_multiple: PAD_VOCAB_SIZE_MULTIPLE, // pad_vocab_size_multiple: 8
                    // `config.json` has no `tie_embeddings`, and
                    // `MambaConfig.tie_embeddings = True`.
                    missing_lm_head: true,
                },
                // `ssm_cfg: {}`, so the block is `Mamba` (`mamba_simple.py`)
                // with the keyword values of its signature.
                block: MambaBlockConfig::Mamba1(
                    mamba1::prelude::Mamba1Config::new(
                        D_MODEL, // d_model: 768
                    )
                    .with_state_rank(16) // d_state=16
                    .with_conv_kernel(4) // d_conv=4
                    .with_expand(2) // expand=2
                    .with_d_inner(Some(2 * D_MODEL)) // d_inner = int(expand * d_model)
                    .with_dt_rank(Some(D_MODEL.div_ceil(16))) // dt_rank="auto": ceil(d_model / 16)
                    .with_has_proj_bias(false) // bias=False
                    .with_has_conv_bias(true) // conv_bias=True
                    // The checkpoint holds each parameter as one tensor.
                    .with_untied(Vec::new()),
                ),
            }
        }

        /// The scan of a forward. Mamba-1 has no SSD, so it has no SSD path.
        pub fn ssd_path() -> MambaSsdPath {
            MambaSsdPath::Mamba1
        }
    }

    /// The `state-spaces/mamba2-130m` checkpoint, of the Mamba-2 family.
    #[cfg(feature = "mamba2")]
    pub mod mamba2_130m {
        #[allow(unused_imports)]
        use crate::hub::{FilePath, RepoId, RevisionPath};
        use burn_mamba::mamba2;
        use burn_mamba::prelude::*;

        /// The spec of this checkpoint. [super::CHECKPOINTS] lists it only with
        /// its checkpoint feature, `mamba2-130m`.
        pub const SPEC: crate::CheckpointSpec = crate::CheckpointSpec {
            id: "mamba2-130m",
            display_name: DISPLAY_NAME,
            repo_id: REPO_ID,
            revision_path: REVISION_PATH,
            file_path_model_safetensors: FILE_PATH_MODEL_SAFETENSORS,
            tokenizer_repo_id: tokenizer_source::REPO_ID,
            file_path_tokenizer_json: tokenizer_source::FILE_PATH_TOKENIZER_JSON,
            topology,
            ssd_path,
        };

        /// A [RepoId].
        pub const REPO_ID: &str = "state-spaces/mamba2-130m";
        /// A [RevisionPath].
        ///
        /// The pull request where a bot converted the weights to safetensors.
        pub const REVISION_PATH: &str = "refs/pr/1";
        /// A [FilePath].
        pub const FILE_PATH_CONFIG_JSON: &str = "config.json";
        /// A [FilePath].
        pub const FILE_PATH_MODEL_SAFETENSORS: &str = "model.safetensors";

        /// The source of the `tokenizer.json` of this checkpoint.
        pub use super::tokenizer as tokenizer_source;

        /// The name on the asset card of the browser page.
        pub const DISPLAY_NAME: &str = "Mamba2-130m";

        /// `vocab_size` in `config.json`.
        pub const VOCAB_SIZE: usize = 50277;
        /// `pad_vocab_size_multiple` in `config.json`.
        pub const PAD_VOCAB_SIZE_MULTIPLE: usize = 16;
        /// `n_layer` in `config.json`.
        pub const N_LAYER: usize = 24;
        /// `d_model` in `config.json`.
        pub const D_MODEL: usize = 768;

        /// The topology of this checkpoint. A comment beside each value names
        /// its source.
        pub fn topology() -> MambaVocabNetConfig {
            MambaVocabNetConfig {
                shape: VocabShape {
                    vocab_size: VOCAB_SIZE, // vocab_size: 50277
                    // `d_intermediate: 0`: no feed-forward.
                    stack: super::reference_stack(N_LAYER, None), // n_layer: 24
                    pad_vocab_size_multiple: PAD_VOCAB_SIZE_MULTIPLE, // pad_vocab_size_multiple: 16
                    missing_lm_head: true, // tie_embeddings: true
                },
                // `ssm_cfg: {"layer": "Mamba2"}`, so the block is `Mamba2`
                // (`mamba2.py`) with the keyword values of its signature.
                block: MambaBlockConfig::Mamba2(
                    mamba2::prelude::Mamba2Config::new(
                        D_MODEL, // d_model: 768
                    )
                    .with_state_rank(128) // d_state=128
                    .with_conv_kernel(4) // d_conv=4
                    .with_expand(2) // expand=2
                    .with_per_head_dim(64) // headdim=64, so nheads = 768*2/64 = 24
                    .with_ngroups(1) // ngroups=1
                    .with_is_norm_before_gate(false) // norm_before_gate=False
                    .with_dt_limit((0.0, f64::INFINITY)) // dt_limit=(0.0, float("inf"))
                    .with_has_proj_bias(false) // bias=False
                    .with_has_conv_bias(true) // conv_bias=True
                    // `Mamba2` has no learnable initial state.
                    .with_has_learnable_init_state(false)
                    // The checkpoint holds each parameter as one tensor.
                    .with_untied(Vec::new()),
                ),
            }
        }

        /// The scan of a forward: the `SerialRecalculated` SSD path. With
        /// `None`, burn-mamba uses its optimal chunk length for the
        /// dimensions of the block.
        pub fn ssd_path() -> MambaSsdPath {
            MambaSsdPath::Mamba2(Mamba2SsdPath::SerialRecalculated(None))
        }
    }

    /// Writes the module of one of the two official Mamba-3 187m checkpoints.
    ///
    /// Each checkpoint has 12 layers of a `Mamba3` block and a SwiGLU
    /// feed-forward, over the Llama-3.1 vocabulary. The two differ only in
    /// `mimo_rank`, `d_intermediate` and the chunk length (`chunk_size` in
    /// `config.json`). The values below copy their `config.json`. Both have
    /// `is_outproj_norm: false` and `attn_layer_idx: []`, so the topology has
    /// no `out_norm` and no attention layer.
    #[cfg(feature = "mamba3")]
    macro_rules! mamba3_187m {
        (
            $module:ident,
            id = $id:literal,
            repo = $repo:literal,
            display_name = $display_name:literal,
            d_intermediate = $d_intermediate:expr,
            mimo_rank = $mimo_rank:expr,
            chunk_len = $chunk_len:expr,
        ) => {
            #[doc = concat!("The `", $repo, "` checkpoint, of the Mamba-3 family.")]
            pub mod $module {
                #[allow(unused_imports)]
                use crate::hub::{FilePath, RepoId, RevisionPath};
                use burn_mamba::mamba3;
                use burn_mamba::prelude::*;

                #[doc = concat!(
                    "The spec of this checkpoint. [super::CHECKPOINTS] lists it only with ",
                    "its checkpoint feature, `", $id, "`. The family feature `mamba3` alone ",
                    "adds no checkpoint."
                )]
                pub const SPEC: crate::CheckpointSpec = crate::CheckpointSpec {
                    id: $id,
                    display_name: DISPLAY_NAME,
                    repo_id: REPO_ID,
                    revision_path: REVISION_PATH,
                    file_path_model_safetensors: FILE_PATH_MODEL_SAFETENSORS,
                    tokenizer_repo_id: tokenizer_source::REPO_ID,
                    file_path_tokenizer_json: tokenizer_source::FILE_PATH_TOKENIZER_JSON,
                    topology,
                    ssd_path,
                };

                /// A [RepoId].
                pub const REPO_ID: &str = $repo;
                /// A [RevisionPath].
                ///
                /// The pull request where a bot converted the weights to
                /// safetensors. `main` holds only `pytorch_model.bin`, and
                /// `burn-store` cannot read that file.
                pub const REVISION_PATH: &str = "refs/pr/1";
                /// A [FilePath].
                pub const FILE_PATH_CONFIG_JSON: &str = "config.json";
                /// A [FilePath].
                pub const FILE_PATH_MODEL_SAFETENSORS: &str = "model.safetensors";

                /// The source of the `tokenizer.json` of this checkpoint.
                pub use super::tokenizer_llama31 as tokenizer_source;

                /// The name on the asset card of the browser page.
                pub const DISPLAY_NAME: &str = $display_name;

                /// `vocab_size` in `config.json`.
                pub const VOCAB_SIZE: usize = 128256;
                /// `pad_vocab_size_multiple` in `config.json`.
                pub const PAD_VOCAB_SIZE_MULTIPLE: usize = 16;
                /// `n_layer` in `config.json`.
                pub const N_LAYER: usize = 12;
                /// `d_model` in `config.json`.
                pub const D_MODEL: usize = 768;
                /// `d_intermediate` in `config.json`: the width of the
                /// feed-forward, before [mlp] rounds it to a multiple of 128.
                pub const D_INTERMEDIATE: usize = $d_intermediate;
                /// `mimo_rank` in `config.json`. For the SISO checkpoint
                /// (`is_mimo: false`), `Mamba3.__init__` sets it to 1.
                pub const MIMO_RANK: usize = $mimo_rank;
                /// The chunk length of the SSD scan in the training of the
                /// checkpoint (`chunk_size` in `config.json`).
                pub const CHUNK_LEN: usize = $chunk_len;

                /// The SwiGLU feed-forward of each layer: the reference
                /// `GatedMLP` (`mlp.py`). It rounds `d_intermediate` to the next
                /// multiple of 128. So the `fc1` of the MIMO checkpoint is
                /// `[2·1280, 768]`, and its `config.json` says `1264`.
                pub fn mlp() -> GatedMlpConfig {
                    GatedMlpConfig {
                        d_model: D_MODEL,               // d_model: 768
                        d_intermediate: D_INTERMEDIATE, // d_intermediate
                        multiple_of: 128,               // multiple_of=128
                        has_proj_bias: false,           // bias=False
                    }
                }

                /// The topology of this checkpoint. A comment beside each value
                /// names its source.
                pub fn topology() -> MambaVocabNetConfig {
                    MambaVocabNetConfig {
                        shape: VocabShape {
                            vocab_size: VOCAB_SIZE, // vocab_size: 128256
                            stack: super::reference_stack(N_LAYER, Some(mlp())), // n_layer: 12
                            pad_vocab_size_multiple: PAD_VOCAB_SIZE_MULTIPLE, // pad_vocab_size_multiple: 16
                            missing_lm_head: true, // tie_embeddings: true
                        },
                        block: MambaBlockConfig::Mamba3(
                            mamba3::prelude::Mamba3Config::new(
                                D_MODEL, // d_model: 768
                            )
                            .with_state_rank(128) // d_state: 128
                            .with_expand(2) // expand: 2
                            .with_per_head_dim(64) // headdim: 64, so nheads = 768*2/64 = 24
                            .with_ngroups(1) // ngroups: 1
                            .with_mimo_rank(MIMO_RANK) // is_mimo, mimo_rank (the macro argument)
                            // The rotation of `mamba3.py` turns pairs of state
                            // entries, one angle for each pair.
                            .with_rotation(RotationKind::Complex2D)
                            .with_rotation_fraction(0.5) // rope_fraction: 0.5
                            // The angle of one step is `Δ·π·tanh(ϑ)` (the
                            // `angle_dt` kernel of `mamba_ssm`).
                            .with_rotation_range(1.0)
                            // One recurrence step for each token.
                            .with_micro_steps(1)
                            // The trapezoid of `mamba3.py` has one tap, at
                            // `t−1` (`k_state`, `v_state`).
                            .with_tap_pattern(TapPattern::HorizontalCarryOver)
                            // `_A = -softplus(dd_A)`: the decay comes from a
                            // projection of the token.
                            .with_gain(Gain::Projected)
                            // `mamba3.py` has no tropical register.
                            .with_tropical(Tropical::None)
                            // `mamba3.py` has no learnable initial state.
                            .with_has_learnable_init_state(false)
                            .with_a_floor(1e-4) // A_floor: 0.0001
                            // `DT = softplus(dd_dt + dt_bias)`, with no clamp.
                            .with_dt_limit((0.0, f64::INFINITY))
                            .with_dt_min(1e-3) // dt_min: 0.001
                            .with_dt_max(0.1) // dt_max: 0.1
                            .with_dt_init_floor(1e-4) // dt_init_floor: 0.0001
                            .with_has_outproj_norm(false) // is_outproj_norm: false
                            .with_has_proj_bias(false) // `in_proj` and `out_proj` have `bias=False`
                            // The checkpoint holds each parameter as one tensor.
                            .with_untied(Vec::new())
                            // This option does not change the values or the
                            // gradients. This app decodes on flex, natively and
                            // in the browser. On flex, the specialized per-token
                            // kernel is about 3× slower than the plain matmul.
                            .with_siso_specialization_decode(false),
                        ),
                    }
                }

                /// The scan of a forward: the `SerialRecalculated` SSD path, at
                /// the chunk length of the training ([CHUNK_LEN]).
                pub fn ssd_path() -> MambaSsdPath {
                    MambaSsdPath::Mamba3(Mamba3SsdPath::SerialRecalculated(Some(CHUNK_LEN)))
                }
            }
        };
    }

    #[cfg(feature = "mamba3")]
    mamba3_187m!(
        mamba3_siso_187m,
        id = "mamba3-siso-187m",
        repo = "state-spaces/mamba3-siso-187m",
        display_name = "Mamba3-SISO-187m",
        d_intermediate = 1536, // d_intermediate: 1536
        // `is_mimo: false`, and then `Mamba3.__init__` sets `mimo_rank = 1`.
        mimo_rank = 1,
        chunk_len = 64, // chunk_size: 64
    );

    #[cfg(feature = "mamba3")]
    mamba3_187m!(
        mamba3_mimo_187m,
        id = "mamba3-mimo-187m",
        repo = "state-spaces/mamba3-mimo-187m",
        display_name = "Mamba3-MIMO-187m",
        d_intermediate = 1264, // d_intermediate: 1264
        mimo_rank = 4,         // is_mimo: true, mimo_rank: 4
        chunk_len = 16,        // chunk_size: 16
    );
}

/// The run wrapper: the tokenizer and the network of one checkpoint, with the
/// two run modes over the same weights.
///
/// - [MambaWrapper::run_sequential] is a sequential run: one
///   [MambaWrapper::step] for each token, with the caches. The browser page
///   and the browser console call [MambaWrapper::step] directly.
/// - [MambaWrapper::run_parallel] is a parallel run: for each new token, one
///   forward over the whole sequence so far, with no cache.
///
/// The native binary does both runs of one prompt, for run agreement.
#[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
pub struct MambaWrapper {
    /// The checkpoint of the network. A parallel run reads its scan
    /// ([CheckpointSpec::ssd_path]). So a build with several checkpoints keeps
    /// their scans apart.
    pub spec: &'static CheckpointSpec,
    /// The tokenizer, with the stream of the token ids of the run.
    pub tokenizer: TokenOutputStream,
    /// The network.
    pub mamba: MambaVocabNet,
    /// The topology of the network. It gives the padded vocabulary size.
    pub topology: MambaVocabNetConfig,
}

/// The sampler of the run wrapper: a [LogitsProcessor] and a repeat penalty.
pub struct LogitsProcessorWrapper {
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

#[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
impl MambaWrapper {
    /// Makes the run wrapper of the checkpoint `spec`, from its tokenizer and
    /// its network.
    pub fn new(spec: &'static CheckpointSpec, tokenizer: Tokenizer, mamba: MambaVocabNet) -> Self {
        Self {
            spec,
            tokenizer: TokenOutputStream::new(tokenizer),
            mamba,
            topology: (spec.topology)(),
        }
    }

    /// The end-of-sequence token names of the supported tokenizers, tried in
    /// order. `EleutherAI/gpt-neox-20b` spells it `<|endoftext|>` and
    /// `meta-llama/Llama-3.1-8B` spells it `<|end_of_text|>`. Each vocabulary
    /// holds only its own name, so the first name that it holds is its eos
    /// token.
    const EOS_TOKENS: [&str; 2] = ["<|endoftext|>", "<|end_of_text|>"];

    /// Clears the token ids of [Self::tokenizer]. Returns the token ids of
    /// `prompt` and the id of the eos token.
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

    /// The zero caches of the network for a batch of `batch` rows, for a
    /// sequential run. See [empty_caches].
    pub fn empty_caches(&self, batch: usize) -> anyhow::Result<MambaCaches> {
        Ok(empty_caches(batch, &self.mamba))
    }

    /// A parallel run of `prompt`. It prints the text to stdout.
    ///
    /// For each new token, it does one forward over the whole sequence so far,
    /// with no cache. The forward uses the scan of [Self::spec]. The run
    /// stops at the eos token. It also stops before a forward when at least
    /// `sample_len` positions gave a token.
    ///
    /// It returns two values:
    ///
    /// - the count of the positions that gave a token other than the eos
    ///   token, prompt positions included,
    /// - the instant after the first forward.
    pub fn run_parallel(
        &mut self,
        prompt: &str,
        sample_len: usize,
        logits_processor_config: &mut LogitsProcessorWrapper,
    ) -> anyhow::Result<(usize, Option<std::time::Instant>)> {
        use std::io::Write;
        let (mut tokens, eos_token) = self.reset_prompt(prompt)?;
        let device = device(&self.mamba);

        // No position gives the first token as an output, so print it here.
        if let Some(t) = tokens.first() {
            if let Some(t) = self.tokenizer.next_token(*t as u32) {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut instant = None;
        let mut i = 0;
        'outer: while i < sample_len {
            let input = token_pairs(&tokens, 1, &device);

            let ssd_path = (self.spec.ssd_path)();
            let (logits_list, _caches) = self.mamba.forward(input, None, ssd_path, None, None);
            if i == 0 {
                instant = Some(std::time::Instant::now());
            }

            let logits_list = self.vocab_logits(logits_list);
            let logits_list = logits_list.into_data().try_to_vec::<Precision>().unwrap();

            // The logits have one row for each position. Skip the positions
            // that already gave a token.
            let logits_list = logits_list
                .chunks_exact(self.padded_vocab_size())
                .skip(i)
                .map(<[Precision]>::to_vec)
                .collect::<Vec<_>>();

            for logits in logits_list.into_iter() {
                let next_token = logits_processor_config.add_logits(i, &mut tokens, logits)?;
                if next_token == eos_token {
                    break 'outer;
                }

                // Print the new text, if the token completes some.
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

    /// A sequential run of `prompt`. It prints the text to stdout.
    ///
    /// It does one [Self::step] for each token, the prompt tokens included,
    /// with the caches. The run stops at the eos token, or after `sample_len`
    /// steps.
    ///
    /// It returns two values:
    ///
    /// - the count of the steps that gave a token other than the eos token,
    ///   prompt steps included,
    /// - the instant after the first step.
    pub fn run_sequential(
        &mut self,
        prompt: &str,
        sample_len: usize,
        logits_processor_config: &mut LogitsProcessorWrapper,
    ) -> anyhow::Result<(usize, Option<std::time::Instant>)> {
        use std::io::Write;
        let (mut tokens, eos_token) = self.reset_prompt(prompt)?;

        // No step gives the first token as an output, so print it here.
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

            // Print the new text, if the token completes some.
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

    /// One step of the network for the token id `input`. Returns the vocab
    /// logits and the caches after the step.
    ///
    /// `caches` holds the caches after the step before. With `None`, the step
    /// starts from zero caches.
    pub fn step(
        &self,
        input: usize,
        caches: Option<MambaCaches>,
    ) -> anyhow::Result<(Vec<Precision>, MambaCaches)> {
        let device = device(&self.mamba);
        let input = token_pairs(&[input], 1, &device).squeeze_dim(1);

        let (logits, new_caches) = self.mamba.step(input, caches, None);
        let logits = self.vocab_logits(logits);
        assert_eq!([1, self.padded_vocab_size()], logits.dims());

        let logits = logits
            .cast(PRECISION_FLOAT_D_TYPE)
            .into_data()
            .try_to_vec::<Precision>()
            .unwrap();

        Ok((logits, new_caches))
    }

    /// The width of the vocab logits. The flag logit is not part of it.
    pub fn padded_vocab_size(&self) -> usize {
        padded_vocab_size(&self.topology)
    }

    /// Drops the flag logit, which is the last column of the logits of the
    /// network. So the sampler reads only the vocab logits, and each id that it
    /// picks is an id of the padded vocabulary.
    fn vocab_logits<const D: usize>(&self, logits: Tensor<D>) -> Tensor<D> {
        let padded_vocab = self.padded_vocab_size();
        assert_eq!(
            logits.dims()[D - 1],
            padded_vocab + 1,
            "the logits must hold the vocab logits and one flag logit"
        );
        logits.narrow(D - 1, 0, padded_vocab)
    }
}

impl LogitsProcessorWrapper {
    /// Makes the sampler. [LogitsProcessor::new] reads `seed`, `temp` and
    /// `top_p`.
    ///
    /// `repeat_penalty` applies to the logit of each token at the positions
    /// `i - repeat_last_n` to `i` of [Self::add_logits] (see
    /// [sampling::apply_repeat_penalty]). A penalty of 1.0 applies no penalty.
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

    /// Gives the token at the position `i + 1`, from the `logits` of the
    /// position `i`. The first call has `i = 0`.
    ///
    /// Inside the prompt, it gives the next token of the prompt. After the
    /// prompt, it samples a token and appends it to `tokens`.
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
            // The prompt gives the next token, and `tokens` already holds it.
            next_token = tokens[i + 1];
        } else {
            // After the prompt: sample the next token, and append it.
            next_token = self.logits_processor.sample(&logits)? as usize;
            tokens.push(next_token);
        }
        Ok(next_token)
    }
}

/// The device of the network: the device of its embedding.
#[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
pub fn device(mamba: &MambaVocabNet) -> Device {
    mamba.embedding.weight.device()
}

/// The vocabulary size of `topology`, rounded up to a multiple of its
/// `pad_vocab_size_multiple`. It is the row count of the embedding, and the
/// width of the vocab logits.
#[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
pub fn padded_vocab_size(topology: &MambaVocabNetConfig) -> usize {
    let VocabShape {
        vocab_size,
        pad_vocab_size_multiple,
        ..
    } = &topology.shape;
    vocab_size.div_ceil(*pad_vocab_size_multiple) * pad_vocab_size_multiple
}

/// The input of a network for the token ids `ids`: a `[batch, sequence, 2]`
/// tensor of `(id, flag)` pairs. `ids` holds the `batch` rows one after the
/// other. The flag of each pair is 0, so the flag embedding adds nothing to a
/// token. The `[batch, 2]` input of a step is this tensor for one id per row,
/// without its sequence axis.
#[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
pub fn token_pairs(ids: &[usize], batch: usize, device: &Device) -> Tensor<3, Int> {
    assert!(
        batch > 0 && ids.len().is_multiple_of(batch),
        "the {} ids do not make {batch} rows of equal length",
        ids.len()
    );
    let pairs: Vec<i32> = ids.iter().flat_map(|&id| [id as i32, 0]).collect();
    Tensor::from_data(TensorData::new(pairs, [batch, ids.len() / batch, 2]), device)
}

/// The zero caches of `mamba` for a batch of `batch` rows: the caches that a
/// step with no cache starts from. The zero-cache constructor of the layer
/// stack gives them, for each family. For Mamba-3, they are single-SSD caches.
#[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
pub fn empty_caches(batch: usize, mamba: &MambaVocabNet) -> MambaCaches {
    let [_padded_vocab, d_model] = mamba.embedding.weight.dims();
    // The constructor reads only the shape and the device of its input.
    let x_bd: Tensor<2> = Tensor::zeros([batch, d_model], &device(mamba));
    mamba.layers.zero_caches_2d(&x_bd)
}

/// Checks the list of the compiled-in checkpoints, whatever set of checkpoint
/// features the build has.
#[cfg(all(test, any(feature = "mamba1", feature = "mamba2", feature = "mamba3")))]
mod tests {
    use super::*;

    /// The list holds the highest priority first, and [hf::by_id] finds each
    /// [CheckpointSpec::id]. An entry point that runs one checkpoint takes the
    /// first entry.
    #[test]
    fn checkpoints_are_listed_by_descending_priority() {
        const PRIORITY: [&str; 4] = [
            "mamba3-mimo-187m",
            "mamba3-siso-187m",
            "mamba2-130m",
            "mamba1-130m",
        ];

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
            assert_eq!(hf::by_id(id).map(|spec| spec.id), Some(*id));
        }
        assert_eq!(hf::preferred().map(|spec| spec.id), ids.first().copied());
    }

    /// The two Mamba-3 checkpoints used two different chunk lengths in their
    /// training. So the scan of each checkpoint must hold its own chunk length,
    /// also in a build with both checkpoints.
    #[cfg(feature = "mamba3")]
    #[test]
    fn each_mamba3_checkpoint_keeps_its_own_chunk_len() {
        let checkpoints = [
            (hf::mamba3_siso_187m::SPEC, hf::mamba3_siso_187m::CHUNK_LEN),
            (hf::mamba3_mimo_187m::SPEC, hf::mamba3_mimo_187m::CHUNK_LEN),
        ];
        assert_ne!(checkpoints[0].1, checkpoints[1].1);

        for (spec, chunk_len) in checkpoints {
            let path = (spec.ssd_path)();
            let MambaSsdPath::Mamba3(Mamba3SsdPath::SerialRecalculated(Some(got))) = path else {
                panic!("{}: expected a chunked Mamba-3 scan, got {path:?}", spec.id)
            };
            assert_eq!(got, chunk_len, "{}", spec.id);
        }
    }

    /// Each Mamba-3 checkpoint turns its state as the reference implementation
    /// does. It has no mechanism that the reference block does not have. A
    /// sequential run and a parallel run share these values, so run agreement
    /// cannot show a wrong value. The expected values come from the reference,
    /// not from burn-mamba:
    ///
    /// - The rotation turns pairs of state entries, one angle for each pair
    ///   (`num_rope_angles` of `mamba3.py`): `Complex2D`.
    /// - `rope_fraction: 0.5` in `config.json`.
    /// - The angle of one step is `Δ·π·tanh(ϑ)` (the `angle_dt` kernel), so the
    ///   rotation range is 1.0.
    /// - One recurrence step for each token, with one tap at `t−1`
    ///   (`k_state`, `v_state`).
    /// - The decay comes from a projection of the token: `_A = -softplus(dd_A)`.
    /// - No tropical register, and no learnable initial state.
    #[cfg(feature = "mamba3")]
    #[test]
    fn each_mamba3_checkpoint_keeps_the_reference_rotation() {
        use burn_mamba::mamba3::prelude::{Gain, RotationKind, TapPattern, Tropical};

        for spec in [hf::mamba3_siso_187m::SPEC, hf::mamba3_mimo_187m::SPEC] {
            let id = spec.id;
            let topology = (spec.topology)();
            #[allow(irrefutable_let_patterns)]
            let MambaBlockConfig::Mamba3(block) = &topology.block else {
                panic!("{id}: expected a Mamba-3 block config")
            };
            assert_eq!(block.rotation, RotationKind::Complex2D, "{id}: rotation kind");
            assert_eq!(block.rotation_fraction, 0.5, "{id}: rotation fraction");
            assert_eq!(block.rotation_range, 1.0, "{id}: rotation range");
            assert_eq!(block.micro_steps, 1, "{id}: micro-steps");
            assert_eq!(block.tap_pattern, TapPattern::HorizontalCarryOver, "{id}: tap pattern");
            assert_eq!(block.gain, Gain::Projected, "{id}: gain");
            assert_eq!(block.tropical, Tropical::None, "{id}: tropical register");
            assert!(!block.has_learnable_init_state, "{id}: learnable initial state");
        }
    }
}
