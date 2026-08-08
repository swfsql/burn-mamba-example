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
        MambaVocabNetConfig::Mamba3 { mamba_block, .. } => {
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
            // The MIMO projections exist only at `mimo_rank > 1`: the SISO
            // checkpoint stores none and `Mamba3` holds `None` in their place,
            // so the rules are conditioned on the same thing the tensors are.
            if mamba_block.mimo_rank > 1 {
                rules.push((r"\.mamba_block\.mimo_x$", ".mamba_block.mimo_x_hmp"));
                rules.push((r"\.mamba_block\.mimo_z$", ".mamba_block.mimo_z_hmp"));
                rules.push((r"\.mamba_block\.mimo_o$", ".mamba_block.mimo_o_hmp"));
            }
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

/// Checks [`key_remapping`] against the real checkpoint manifests: nothing on
/// either side of the load may be left dangling.
///
/// A remapping bug is otherwise only discoverable by downloading ~1.1GB and
/// watching the load fail, so every compiled-in checkpoint's `model.safetensors`
/// header is vendored here — the model-level tensors plus the whole of layer 0,
/// read from `refs/pr/1` of each repo. Each name is pushed through the same
/// rules the store uses, and the resulting set must be *exactly* the set of
/// parameters the built module exposes, with matching shapes: a checkpoint
/// tensor landing on no parameter would be silently dropped (`load_from` only
/// logs those), and a parameter no tensor fills would keep its random init.
/// [`remapping_rules_all_fire`] closes the third gap — a rule matching nothing,
/// which is a rule that has stopped describing the checkpoint — and
/// [`manifests_cover_the_whole_checkpoint`] the fourth: a manifest that is not
/// the whole file.
#[cfg(all(test, any(feature = "mamba1", feature = "mamba2", feature = "mamba3")))]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    /// `(checkpoint tensor name, checkpoint shape)`.
    type Manifest = &'static [(&'static str, &'static [usize])];

    /// One checkpoint's vendored `model.safetensors` header.
    struct Vendored {
        /// The checkpoint it was read from.
        spec: &'static crate::ModelSpec,
        /// Everything outside `backbone.layers.`, plus the whole of layer 0.
        /// Every layer repeats layer 0's names and shapes, which is what lets
        /// one layer stand for all of them.
        manifest: Manifest,
        /// How many times layer 0 repeats.
        n_layers: usize,
    }

    #[cfg(feature = "mamba1")]
    const MAMBA1_MANIFEST: Manifest = &[
        ("backbone.embedding.weight", &[50280, 768]),
        ("backbone.norm_f.weight", &[768]),
        ("backbone.layers.0.norm.weight", &[768]),
        ("backbone.layers.0.mixer.A_log", &[1536, 16]),
        ("backbone.layers.0.mixer.D", &[1536]),
        ("backbone.layers.0.mixer.conv1d.bias", &[1536]),
        ("backbone.layers.0.mixer.conv1d.weight", &[1536, 1, 4]),
        ("backbone.layers.0.mixer.dt_proj.bias", &[1536]),
        ("backbone.layers.0.mixer.dt_proj.weight", &[1536, 48]),
        ("backbone.layers.0.mixer.in_proj.weight", &[3072, 768]),
        ("backbone.layers.0.mixer.out_proj.weight", &[768, 1536]),
        ("backbone.layers.0.mixer.x_proj.weight", &[80, 1536]),
    ];

    #[cfg(feature = "mamba2")]
    const MAMBA2_MANIFEST: Manifest = &[
        ("backbone.embedding.weight", &[50288, 768]),
        ("backbone.norm_f.weight", &[768]),
        ("backbone.layers.0.norm.weight", &[768]),
        ("backbone.layers.0.mixer.A_log", &[24]),
        ("backbone.layers.0.mixer.D", &[24]),
        ("backbone.layers.0.mixer.conv1d.bias", &[1792]),
        ("backbone.layers.0.mixer.conv1d.weight", &[1792, 1, 4]),
        ("backbone.layers.0.mixer.dt_bias", &[24]),
        ("backbone.layers.0.mixer.in_proj.weight", &[3352, 768]),
        ("backbone.layers.0.mixer.norm.weight", &[1536]),
        ("backbone.layers.0.mixer.out_proj.weight", &[768, 1536]),
    ];

    #[cfg(feature = "mamba3")]
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

    #[cfg(feature = "mamba3")]
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

    /// Every checkpoint this build compiled in, whatever set of model features
    /// that is.
    const VENDORED: &[Vendored] = &[
        #[cfg(feature = "mamba1")]
        Vendored {
            spec: &crate::hf::mamba1_130m::SPEC,
            manifest: MAMBA1_MANIFEST,
            n_layers: crate::hf::mamba1_130m::N_LAYER,
        },
        #[cfg(feature = "mamba2")]
        Vendored {
            spec: &crate::hf::mamba2_130m::SPEC,
            manifest: MAMBA2_MANIFEST,
            n_layers: crate::hf::mamba2_130m::N_LAYER,
        },
        #[cfg(feature = "mamba3")]
        Vendored {
            spec: &crate::hf::mamba3_siso_187m::SPEC,
            manifest: SISO_MANIFEST,
            n_layers: crate::hf::mamba3_siso_187m::N_LAYER,
        },
        #[cfg(feature = "mamba3")]
        Vendored {
            spec: &crate::hf::mamba3_mimo_187m::SPEC,
            manifest: MIMO_MANIFEST,
            n_layers: crate::hf::mamba3_mimo_187m::N_LAYER,
        },
    ];

    /// The vocabulary the comparison module is built with. Small enough to keep
    /// the test from allocating the real 130M/187M parameters, and a multiple of
    /// every checkpoint's `pad_vocab_size_multiple`, so it survives the padding
    /// unchanged.
    const TEST_VOCAB: usize = 64;

    /// [`key_remapping`] applied to a manifest.
    struct Remapped {
        /// `module path → shape`, as the store would see it.
        params: BTreeMap<String, Vec<usize>>,
        /// `(rule pattern, how many names it rewrote)`, in [`key_remapping`]
        /// order.
        rule_hits: Vec<(&'static str, usize)>,
    }

    /// One layer and a token-sized vocabulary: the parameter *paths* depend on
    /// neither, so the comparison runs on a model small enough to build.
    fn one_layer(mut config: MambaVocabNetConfig) -> MambaVocabNetConfig {
        let (n_real_layers, vocab_size) = match &mut config {
            #[cfg(feature = "mamba1")]
            MambaVocabNetConfig::Mamba1 {
                n_real_layers,
                vocab_size,
                ..
            } => (n_real_layers, vocab_size),
            #[cfg(feature = "mamba2")]
            MambaVocabNetConfig::Mamba2 {
                n_real_layers,
                vocab_size,
                ..
            } => (n_real_layers, vocab_size),
            #[cfg(feature = "mamba3")]
            MambaVocabNetConfig::Mamba3 {
                n_real_layers,
                vocab_size,
                ..
            } => (n_real_layers, vocab_size),
        };
        *n_real_layers = 1;
        *vocab_size = TEST_VOCAB;
        config
    }

    /// The checkpoint names, pushed through [`key_remapping`] exactly as
    /// `SafetensorsStore` applies it (every pattern, in order).
    ///
    /// Two names remapping onto one parameter is a panic: the second would
    /// overwrite the first, leaving whichever lost silently unloaded.
    fn remapped(manifest: Manifest, config: &MambaVocabNetConfig) -> Remapped {
        let rules = key_remapping(config);
        let patterns = burn::store::KeyRemapper::from_patterns(rules.clone())
            .expect("the remapping patterns must compile")
            .to_regex_pairs();
        let mut rule_hits: Vec<(&'static str, usize)> =
            rules.iter().map(|(from, _)| (*from, 0)).collect();

        let mut params: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        let mut sources: BTreeMap<String, &'static str> = BTreeMap::new();
        for (name, shape) in manifest {
            let mut path = name.to_string();
            for (index, (regex, replacement)) in patterns.iter().enumerate() {
                if regex.is_match(&path) {
                    rule_hits[index].1 += 1;
                    path = regex.replace_all(&path, replacement.as_str()).to_string();
                }
            }
            // `PyTorchToBurnAdapter` transposes `Linear` weights, and inside a
            // layer a rank-2 `….weight` is exactly one: the norms are rank 1,
            // `conv1d.weight` rank 3, and the bare `Param`s (`A_log`, `B_bias`,
            // `mimo_*`) carry no `.weight` suffix. `embedding.weight` is the
            // only rank-2 weight outside a layer, and is left alone.
            let shape =
                if shape.len() == 2 && name.ends_with(".weight") && name.contains(".layers.") {
                    vec![shape[1], shape[0]]
                } else {
                    shape.to_vec()
                };
            if let Some(first) = sources.insert(path.clone(), name) {
                panic!("{first:?} and {name:?} both remap onto {path:?}");
            }
            params.insert(path, shape);
        }
        Remapped { params, rule_hits }
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

    /// Neither side of the load may hold anything the other does not: every
    /// checkpoint tensor lands on a parameter, and every parameter is filled.
    #[test]
    fn checkpoints_and_modules_hold_the_same_tensors() {
        for vendored in VENDORED {
            let id = vendored.spec.id;
            let full = (vendored.spec.config)();
            let config = one_layer(full.clone());
            let mut expected = remapped(vendored.manifest, &config).params;

            // The one shape the shrunken topology cannot be compared against
            // directly — and the checkpoint's own row count is the check on
            // `pad_vocab_size_multiple`, which differs between models.
            let embedding = expected
                .get_mut("embedding.weight")
                .unwrap_or_else(|| panic!("{id}: the embedding must remap to `embedding.weight`"));
            assert_eq!(
                embedding[0],
                crate::padded_vocab_size(&full),
                "{id}: the config's padded vocabulary must match the checkpoint's embedding"
            );
            embedding[0] = TEST_VOCAB;

            let actual = module_params(&config);
            let expected_keys: BTreeSet<&str> = expected.keys().map(String::as_str).collect();
            let actual_keys: BTreeSet<&str> = actual.keys().map(String::as_str).collect();
            let dangling: Vec<&str> = expected_keys.difference(&actual_keys).copied().collect();
            let unfilled: Vec<&str> = actual_keys.difference(&expected_keys).copied().collect();
            assert!(
                dangling.is_empty(),
                "{id}: checkpoint tensor(s) that reach no module parameter: {dangling:?}"
            );
            assert!(
                unfilled.is_empty(),
                "{id}: module parameter(s) that no checkpoint tensor fills: {unfilled:?}"
            );
            for (path, want) in &expected {
                assert_eq!(&actual[path], want, "{id}: shape mismatch at {path}");
            }
        }
    }

    /// A rule that rewrites nothing is dead weight in [`key_remapping`] — or,
    /// worse, a rename that silently stopped matching. This holds for every
    /// checkpoint with no exception, which is why the MIMO rules are gated on
    /// `mimo_rank` rather than on the Mamba-3 variant.
    #[test]
    fn remapping_rules_all_fire() {
        for vendored in VENDORED {
            let id = vendored.spec.id;
            let config = one_layer((vendored.spec.config)());
            let unused: Vec<&str> = remapped(vendored.manifest, &config)
                .rule_hits
                .into_iter()
                .filter(|(_, hits)| *hits == 0)
                .map(|(pattern, _)| pattern)
                .collect();
            assert!(
                unused.is_empty(),
                "{id}: remapping rule(s) that rewrite nothing: {unused:?}"
            );
        }
    }

    /// Confirms each vendored manifest really is its whole checkpoint header:
    /// the file must hold exactly the manifest's model-level tensors plus one
    /// copy of layer 0 per layer, at the same shapes. Without this the offline
    /// checks above only prove that the *transcribed* part of the checkpoint
    /// loads.
    ///
    /// Ignored by default: it reads `model.safetensors` from the same
    /// `~/.cache/huggingface` a native run uses, **downloading** the ~1.1GB of
    /// checkpoints that are not already there.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "needs the real model.safetensors of every compiled-in checkpoint"]
    fn manifests_cover_the_whole_checkpoint() {
        use crate::hub::{FilePath, Repo, RepoId, RepoType, RevisionPath};
        use std::io::Read;

        let api = crate::hub::sync::Api::new().expect("a hub client");
        for vendored in VENDORED {
            let spec = vendored.spec;
            let path = api
                .repo(Repo::with_revision(
                    RepoId(spec.repo_id.into()),
                    RepoType::Model,
                    RevisionPath(spec.revision_path.into()),
                ))
                .get(&FilePath(spec.file_path_model_safetensors.into()))
                .unwrap_or_else(|e| panic!("{}: {e}", spec.id));

            // safetensors: an 8-byte little-endian header length, then that many
            // bytes of JSON — `name → {dtype, shape, data_offsets}`.
            let mut file = std::fs::File::open(&path).expect("the cached checkpoint");
            let mut length = [0u8; 8];
            file.read_exact(&mut length).expect("a header length");
            let mut header = vec![0u8; u64::from_le_bytes(length) as usize];
            file.read_exact(&mut header).expect("a header");
            let header: BTreeMap<String, serde_json::Value> =
                serde_json::from_slice(&header).expect("a safetensors header");

            let mut found: BTreeMap<String, Vec<usize>> = BTreeMap::new();
            for (name, entry) in header {
                if name == "__metadata__" {
                    continue;
                }
                let shape = entry["shape"]
                    .as_array()
                    .expect("a shape")
                    .iter()
                    .map(|dim| dim.as_u64().expect("a dimension") as usize)
                    .collect();
                found.insert(name, shape);
            }

            let mut expected: BTreeMap<String, Vec<usize>> = BTreeMap::new();
            for (name, shape) in vendored.manifest {
                match name.strip_prefix("backbone.layers.0.") {
                    Some(rest) => {
                        for layer in 0..vendored.n_layers {
                            expected
                                .insert(format!("backbone.layers.{layer}.{rest}"), shape.to_vec());
                        }
                    }
                    None => {
                        expected.insert(name.to_string(), shape.to_vec());
                    }
                }
            }

            let expected_keys: BTreeSet<&str> = expected.keys().map(String::as_str).collect();
            let found_keys: BTreeSet<&str> = found.keys().map(String::as_str).collect();
            let missing: Vec<&str> = expected_keys.difference(&found_keys).copied().collect();
            let extra: Vec<&str> = found_keys.difference(&expected_keys).copied().collect();
            assert!(
                missing.is_empty(),
                "{}: the manifest claims tensor(s) the checkpoint does not have: {missing:?}",
                spec.id
            );
            assert!(
                extra.is_empty(),
                "{}: the checkpoint holds tensor(s) the manifest does not cover: {extra:?}",
                spec.id
            );
            for (name, want) in &expected {
                assert_eq!(&found[name], want, "{}: shape mismatch at {name}", spec.id);
            }
        }
    }
}
