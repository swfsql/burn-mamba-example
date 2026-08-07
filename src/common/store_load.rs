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
    log::info!(
        "loaded {} tensors from the checkpoint",
        result.applied.len()
    );

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
        // `norm_f`, the per-layer pre-norm, the Mamba-3 feed-forward's `norm2`
        // and (Mamba-2) the mixer's gated norm all name their scale `gamma`,
        // where the checkpoint says `weight`.
        (r"(^|\.)norm(_f|2)?\.weight$", "${1}norm${2}.gamma"),
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
        #[cfg(feature = "mamba3")]
        MambaVocabNetConfig::Mamba3 { .. } => {
            // Mamba-3 has no `A_log`: A is data-dependent, projected out of
            // `in_proj` rather than stored.
            rules.push((r"\.mamba_block\.D$", ".mamba_block.d_h"));
            rules.push((r"\.mamba_block\.dt_bias$", ".mamba_block.dt_bias_h"));
            rules.push((r"\.mamba_block\.B_bias$", ".mamba_block.b_bias_hmr"));
            rules.push((r"\.mamba_block\.C_bias$", ".mamba_block.c_bias_hmr"));
            // The QK-norms are `B_norm`/`C_norm`, which the generic `norm.weight`
            // rule above deliberately does not reach (it anchors on `.norm`).
            rules.push((
                r"\.mamba_block\.B_norm\.weight$",
                ".mamba_block.b_norm.gamma",
            ));
            rules.push((
                r"\.mamba_block\.C_norm\.weight$",
                ".mamba_block.c_norm.gamma",
            ));
            // MIMO-only; absent from the SISO checkpoint, where the rules simply
            // match nothing.
            rules.push((r"\.mamba_block\.mimo_x$", ".mamba_block.mimo_x_hmp"));
            rules.push((r"\.mamba_block\.mimo_z$", ".mamba_block.mimo_z_hmp"));
            rules.push((r"\.mamba_block\.mimo_o$", ".mamba_block.mimo_o_hmp"));
        }
    }
    rules
}

/// The checkpoints tie the LM head to the embedding (`missing_lm_head: true`),
/// so the head is the transposed embedding table.
///
/// Public because it is the second half of building a runnable model: anything
/// that assembles one without a checkpoint — `benches/model.rs` builds the same
/// topology on random weights — must materialise the head the same way, or it
/// measures `VocabNetwork::apply_lm_head`'s transpose-per-call fallback instead.
pub fn tie_lm_head(mamba: &mut MambaVocabNet, device: &Device) {
    let embedding_weight = match mamba {
        #[cfg(feature = "mamba1")]
        MambaVocabNet::Mamba1(m) => m.embedding.weight.val(),
        #[cfg(feature = "mamba2")]
        MambaVocabNet::Mamba2(m) => m.embedding.weight.val(),
        #[cfg(feature = "mamba3")]
        MambaVocabNet::Mamba3(m) => m.embedding.weight.val(),
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
        #[cfg(feature = "mamba3")]
        MambaVocabNet::Mamba3(m) => m.lm_head = lm_head,
    }
}

/// Checks [`key_remapping`] against the real checkpoint manifests.
///
/// A remapping bug is otherwise only discoverable by downloading ~750MB and
/// watching the load fail, so the checkpoints' `model.safetensors` headers are
/// vendored here — every tensor of layer 0 plus the two model-level ones, read
/// from `refs/pr/1` of each repo. Each name is pushed through the same rules the
/// store uses and must land exactly on a parameter the built module exposes,
/// with a matching shape.
#[cfg(all(test, feature = "mamba3"))]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// `(checkpoint name, checkpoint shape)` for one layer plus the globals.
    type Manifest = &'static [(&'static str, &'static [usize])];

    const SISO_MANIFEST: Manifest = &[
        ("backbone.embedding.weight", &[128256, 768]),
        ("backbone.norm_f.weight", &[768]),
        ("backbone.layers.0.norm.weight", &[768]),
        ("backbone.layers.0.norm2.weight", &[768]),
        ("backbone.layers.0.mixer.B_bias", &[24, 1, 128]),
        ("backbone.layers.0.mixer.B_norm.weight", &[128]),
        ("backbone.layers.0.mixer.C_bias", &[24, 1, 128]),
        ("backbone.layers.0.mixer.C_norm.weight", &[128]),
        ("backbone.layers.0.mixer.D", &[24]),
        ("backbone.layers.0.mixer.dt_bias", &[24]),
        ("backbone.layers.0.mixer.in_proj.weight", &[3432, 768]),
        ("backbone.layers.0.mixer.out_proj.weight", &[768, 1536]),
        ("backbone.layers.0.mlp.fc1.weight", &[3072, 768]),
        ("backbone.layers.0.mlp.fc2.weight", &[768, 1536]),
    ];

    const MIMO_MANIFEST: Manifest = &[
        ("backbone.embedding.weight", &[128256, 768]),
        ("backbone.norm_f.weight", &[768]),
        ("backbone.layers.0.norm.weight", &[768]),
        ("backbone.layers.0.norm2.weight", &[768]),
        ("backbone.layers.0.mixer.B_bias", &[24, 4, 128]),
        ("backbone.layers.0.mixer.B_norm.weight", &[128]),
        ("backbone.layers.0.mixer.C_bias", &[24, 4, 128]),
        ("backbone.layers.0.mixer.C_norm.weight", &[128]),
        ("backbone.layers.0.mixer.D", &[24]),
        ("backbone.layers.0.mixer.dt_bias", &[24]),
        ("backbone.layers.0.mixer.in_proj.weight", &[4200, 768]),
        ("backbone.layers.0.mixer.mimo_o", &[24, 4, 64]),
        ("backbone.layers.0.mixer.mimo_x", &[24, 4, 64]),
        ("backbone.layers.0.mixer.mimo_z", &[24, 4, 64]),
        ("backbone.layers.0.mixer.out_proj.weight", &[768, 1536]),
        ("backbone.layers.0.mlp.fc1.weight", &[2560, 768]),
        ("backbone.layers.0.mlp.fc2.weight", &[768, 1280]),
    ];

    /// One layer and a token-sized vocabulary: the parameter *paths* do not
    /// depend on either, and this keeps the test from allocating the real 187M
    /// parameters. The embedding is excluded from the shape comparison for the
    /// same reason.
    #[allow(irrefutable_let_patterns)]
    fn one_layer(mut config: MambaVocabNetConfig) -> MambaVocabNetConfig {
        let MambaVocabNetConfig::Mamba3 {
            ref mut n_real_layers,
            ref mut vocab_size,
            ..
        } = config
        else {
            panic!("expected a Mamba-3 config")
        };
        *n_real_layers = 1;
        *vocab_size = 64;
        config
    }

    /// The checkpoint names, pushed through [`key_remapping`] exactly as
    /// `SafetensorsStore` applies it (every pattern, in order).
    fn remapped(manifest: Manifest, config: &MambaVocabNetConfig) -> BTreeMap<String, Vec<usize>> {
        let patterns = burn::store::KeyRemapper::from_patterns(key_remapping(config))
            .expect("the remapping patterns must compile")
            .to_regex_pairs();
        manifest
            .iter()
            .map(|(name, shape)| {
                let mut path = name.to_string();
                for (regex, replacement) in &patterns {
                    if regex.is_match(&path) {
                        path = regex.replace_all(&path, replacement.as_str()).to_string();
                    }
                }
                // `PyTorchToBurnAdapter` transposes `Linear` weights. Inside a
                // layer every rank-2 tensor is one; `embedding.weight` is the
                // only rank-2 tensor outside, and is left alone.
                let shape = if shape.len() == 2 && name.contains(".layers.") {
                    vec![shape[1], shape[0]]
                } else {
                    shape.to_vec()
                };
                (path, shape)
            })
            .collect()
    }

    /// The parameters the built module actually exposes to the store.
    fn module_params(config: &MambaVocabNetConfig) -> BTreeMap<String, Vec<usize>> {
        let device: Device = Default::default();
        let model = config.init(&device);
        model
            .collect(None, None, true)
            .into_iter()
            .map(|snapshot| (snapshot.full_path(), snapshot.shape.to_vec()))
            .collect()
    }

    fn check(manifest: Manifest, config: MambaVocabNetConfig, embedding_rows: usize) {
        let config = one_layer(config);
        let mut expected = remapped(manifest, &config);
        // Restore the shrunken vocabulary for the comparison.
        assert_eq!(
            expected.insert("embedding.weight".into(), vec![64, 768]),
            Some(vec![embedding_rows, 768]),
            "the embedding must remap to `embedding.weight`"
        );

        let actual = module_params(&config);
        assert_eq!(
            expected.keys().collect::<Vec<_>>(),
            actual.keys().collect::<Vec<_>>(),
            "every checkpoint tensor must remap onto a module parameter, and \
             every parameter of a layer must be covered by the checkpoint"
        );
        for (path, want) in &expected {
            assert_eq!(&actual[path], want, "shape mismatch at {path}");
        }
    }

    #[test]
    fn siso_checkpoint_keys_match_the_module() {
        check(SISO_MANIFEST, crate::hf::mamba3_siso_187m::config(), 128256);
    }

    #[test]
    fn mimo_checkpoint_keys_match_the_module() {
        check(MIMO_MANIFEST, crate::hf::mamba3_mimo_187m::config(), 128256);
    }
}
