//! Loading the HF checkpoints into a [MambaVocabNet] through `burn-store`.
//!
//! The whole import is declarative: a [SafetensorsStore] gets a key remapping
//! (`backbone.…` → the Burn module paths) plus an adapter chain that transposes
//! `Linear` weights (PyTorch `[out, in]` → Burn `[in, out]`) and casts every float
//! tensor to [crate::Precision]. That last cast is why both checkpoints share one
//! code path even though mamba-130m stores f32 and mamba2-130m stores f16.

use burn::module::Param;
use burn::prelude::*;
use burn::store::{
    FloatCastAdapter, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
};
use burn_mamba::prelude::*;

/// Where the checkpoint bytes come from.
pub enum Checkpoint {
    /// A `model.safetensors` on disk; memory-mapped by the store.
    #[cfg(not(target_arch = "wasm32"))]
    File(std::path::PathBuf),
    /// Already-downloaded bytes (the browser has no filesystem).
    Bytes(Vec<u8>),
}

impl Checkpoint {
    fn store(self) -> SafetensorsStore {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            Checkpoint::File(path) => SafetensorsStore::from_file(path),
            Checkpoint::Bytes(bytes) => SafetensorsStore::from_bytes(Some(bytes)),
        }
    }
}

/// Builds the model from `mamba_config` and overwrites every parameter from the
/// checkpoint, then ties the LM head to the (transposed) embedding.
pub fn load_mamba(
    checkpoint: Checkpoint,
    mamba_config: MambaVocabNetConfig,
    device: &Device,
) -> anyhow::Result<MambaVocabNet> {
    let mut mamba: MambaVocabNet = mamba_config.init(device);

    let mut store = checkpoint
        .store()
        // `MambaVocabNet` is an enum module: drop the `Mamba1`/`Mamba2` variant
        // segment so the checkpoint paths line up with the inner network.
        .skip_enum_variants(true)
        .with_from_adapter(
            PyTorchToBurnAdapter.chain(FloatCastAdapter::to(crate::PRECISION_FLOAT_D_TYPE.into())),
        );
    for (from, to) in key_remapping(&mamba_config) {
        store = store.with_key_remapping(from, to);
    }

    let result = mamba
        .load_from(&mut store)
        .map_err(|e| anyhow::anyhow!("failed to load the mamba checkpoint: {e}"))?;
    if !result.unused.is_empty() {
        log::warn!(
            "{} checkpoint tensor(s) went unused: {:?}",
            result.unused.len(),
            result.unused
        );
    }
    log::info!("loaded {} tensors from the checkpoint", result.applied.len());

    tie_lm_head(&mut mamba, device);

    Ok(mamba)
}

/// The `safetensors name → Burn module path` rewrites, applied in order.
///
/// The structural rules run first (they strip the `backbone.` prefix and expand
/// `layers.{i}.mixer` into `layers.real_layers.{i}.mamba_block`); the
/// per-parameter renames then operate on the already-rewritten paths.
fn key_remapping(config: &MambaVocabNetConfig) -> Vec<(&'static str, &'static str)> {
    let mut rules: Vec<(&'static str, &'static str)> = vec![
        // backbone.layers.{i}.mixer.X -> layers.real_layers.{i}.mamba_block.X
        (
            r"^backbone\.layers\.(\d+)\.mixer\.",
            "layers.real_layers.$1.mamba_block.",
        ),
        // backbone.layers.{i}.X -> layers.real_layers.{i}.X
        (r"^backbone\.layers\.(\d+)\.", "layers.real_layers.$1."),
        // backbone.X -> X
        (r"^backbone\.", ""),
        // `norm_f`, the per-layer pre-norm and (Mamba-2) the mixer's gated norm all
        // name their scale `gamma`, where the checkpoint says `weight`.
        (r"(^|\.)norm(_f)?\.weight$", "${1}norm${2}.gamma"),
    ];

    #[allow(irrefutable_let_patterns)]
    match config {
        #[cfg(feature = "mamba1")]
        MambaVocabNetConfig::Mamba1 { .. } => {
            rules.push((r"\.mamba_block\.A_log$", ".mamba_block.a_log"));
            rules.push((r"\.mamba_block\.D$", ".mamba_block.d"));
        }
        #[cfg(feature = "mamba2")]
        MambaVocabNetConfig::Mamba2 { .. } => {
            rules.push((r"\.mamba_block\.A_log$", ".mamba_block.a_log_h"));
            rules.push((r"\.mamba_block\.D$", ".mamba_block.d_h"));
            rules.push((r"\.mamba_block\.dt_bias$", ".mamba_block.dt_bias_h"));
        }
    }
    rules
}

/// The checkpoints tie the LM head to the embedding (`missing_lm_head: true`),
/// so the head is the transposed embedding table.
fn tie_lm_head(mamba: &mut MambaVocabNet, device: &Device) {
    let embedding_weight = match mamba {
        #[cfg(feature = "mamba1")]
        MambaVocabNet::Mamba1(m) => m.embedding.weight.val(),
        #[cfg(feature = "mamba2")]
        MambaVocabNet::Mamba2(m) => m.embedding.weight.val(),
    };

    let weight = embedding_weight.swap_dims(0, 1);
    // `from_data(into_data())` forces the transposed view to be contiguous.
    let weight: Tensor<2> = Tensor::from_data(weight.into_data(), device);
    let lm_head = Some(burn::nn::Linear {
        weight: Param::from_tensor(weight),
        bias: None,
    });

    match mamba {
        #[cfg(feature = "mamba1")]
        MambaVocabNet::Mamba1(m) => m.lm_head = lm_head,
        #[cfg(feature = "mamba2")]
        MambaVocabNet::Mamba2(m) => m.lm_head = lm_head,
    }
}
