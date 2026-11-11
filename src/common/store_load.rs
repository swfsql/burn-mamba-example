//! The build of a [MambaVocabNet] from the weights file of a checkpoint,
//! through `burn-store`.
//!
//! [build_network] makes the network from its topology, applies the store to
//! it, then calls [tie_lm_head].
//!
//! The whole import is declarative. A [SafetensorsStore] has a key remapping
//! (`backbone.…` → the Burn module paths) and an adapter chain. The chain
//! transposes `Linear` weights (PyTorch `[out, in]` → Burn `[in, out]`) and casts
//! every float tensor to [crate::Precision]. With that cast, the f32, f16 and
//! bf16 checkpoints share one code path (ADR-0003).
//!
//! The load is strict. Each parameter of the network must receive a checkpoint
//! tensor, with one exception: [`ALLOWED_MISSING`], the parameters of the flag
//! channel of the vocab network. No HF checkpoint holds them, so they keep
//! their init values. The load only logs a checkpoint tensor that reaches no
//! parameter.

use burn::module::Param;
use burn::prelude::*;
use burn::store::{
    FloatCastAdapter, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
};
use burn_mamba::prelude::*;

/// Where the bytes of the weights file of a checkpoint come from.
pub enum WeightsSource {
    /// A `model.safetensors` file on disk. The store maps it into memory.
    #[cfg(not(target_arch = "wasm32"))]
    File(std::path::PathBuf),
    /// Bytes that are already in memory (the browser has no filesystem).
    Bytes(Vec<u8>),
}

impl WeightsSource {
    fn store(self) -> SafetensorsStore {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            WeightsSource::File(path) => SafetensorsStore::from_file(path),
            WeightsSource::Bytes(bytes) => SafetensorsStore::from_bytes(Some(bytes)),
        }
    }
}

/// The parameters of the network that no checkpoint tensor fills: the flag
/// embedding, and the weight and the bias of the flag head. The load keeps
/// their init values. The flag embedding is zero at init, and each token has
/// the flag 0. The run wrapper drops the flag logit, so no output reads the
/// flag head.
const ALLOWED_MISSING: &[&str] = &[
    "flag_embedding",
    "flag_head.weight",
    "flag_head.bias",
];

/// Builds the network from `topology`, writes each parameter from the
/// checkpoint, then ties the LM head to the transposed embedding.
///
/// The parameters of the flag channel keep their init values. No checkpoint
/// holds them.
pub fn build_network(
    weights: WeightsSource,
    topology: MambaVocabNetConfig,
    device: &Device,
) -> anyhow::Result<MambaVocabNet> {
    let mut mamba: MambaVocabNet = topology.init(device);

    let mut store = weights
        .store()
        // `MambaBlock` is an enum module: drop its `Mamba1`/`Mamba2`/`Mamba3`
        // variant segment, so the checkpoint paths line up with the block.
        .skip_enum_variants(true)
        // The store returns the missing parameters, and does not fail on them.
        // The check below fails on each one that is not in `ALLOWED_MISSING`.
        .allow_partial(true)
        .with_from_adapter(
            PyTorchToBurnAdapter.chain(FloatCastAdapter::to(crate::PRECISION_FLOAT_D_TYPE.into())),
        );
    for (from, to) in key_remapping(&topology) {
        store = store.with_key_remapping(from, to);
    }

    let result = mamba
        .load_from(&mut store)
        .map_err(|e| anyhow::anyhow!("failed to load the mamba checkpoint: {e}"))?;
    let unfilled: Vec<&str> = result
        .missing
        .iter()
        .map(|(path, _containers)| path.as_str())
        .filter(|path| !ALLOWED_MISSING.contains(path))
        .collect();
    if !unfilled.is_empty() {
        anyhow::bail!("the checkpoint has no tensor for the module parameter(s): {unfilled:?}");
    }
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
/// The structural rules run first. They remove the `backbone.` prefix and
/// expand `layers.{i}.mixer` into `layers.real_layers.{i}.block`. The
/// per-parameter renames then operate on the rewritten paths. The family of the
/// block config selects the per-parameter renames.
fn key_remapping(topology: &MambaVocabNetConfig) -> Vec<(&'static str, &'static str)> {
    let mut rules: Vec<(&'static str, &'static str)> = vec![
        // backbone.layers.{i}.mixer.X -> layers.real_layers.{i}.block.X
        (
            r"^backbone\.layers\.(\d+)\.mixer\.",
            "layers.real_layers.$1.block.",
        ),
        // backbone.layers.{i}.X -> layers.real_layers.{i}.X. The `mlp` of
        // Mamba-3 needs no other rule.
        (r"^backbone\.layers\.(\d+)\.", "layers.real_layers.$1."),
        // backbone.X -> X
        (r"^backbone\.", ""),
        // The checkpoint names the scale of a norm `weight`, and the network
        // names it `gamma`. This rule covers `norm_f`, the pre-norm of each
        // layer, the `norm2` of the Mamba-3 feed-forward and the gated norm of
        // the Mamba-2 block.
        (r"(^|\.)norm(_f|2)?\.weight$", "${1}norm${2}.gamma"),
    ];

    match &topology.block {
        #[cfg(feature = "mamba1")]
        MambaBlockConfig::Mamba1(_) => {
            rules.push((r"\.block\.A_log$", ".block.a_log"));
            rules.push((r"\.block\.D$", ".block.d"));
        }
        #[cfg(feature = "mamba2")]
        MambaBlockConfig::Mamba2(_) => {
            rules.push((r"\.block\.A_log$", ".block.a_log_h"));
            rules.push((r"\.block\.D$", ".block.d_h"));
            rules.push((r"\.block\.dt_bias$", ".block.dt_bias_h"));
        }
        #[cfg(feature = "mamba3")]
        MambaBlockConfig::Mamba3(block) => {
            // Mamba-3 has no `A_log`. A comes from a projection of the token
            // (`in_proj`), not from a parameter.
            rules.push((r"\.block\.D$", ".block.d_h"));
            rules.push((r"\.block\.dt_bias$", ".block.dt_bias_h"));
            rules.push((r"\.block\.B_bias$", ".block.b_bias_hmr"));
            rules.push((r"\.block\.C_bias$", ".block.c_bias_hmr"));
            // The norms of B and C are `B_norm` and `C_norm`. The `norm.weight`
            // rule above does not match them: it needs `norm` at the start of
            // the name or after a `.`.
            rules.push((r"\.block\.B_norm\.weight$", ".block.b_norm.gamma"));
            rules.push((r"\.block\.C_norm\.weight$", ".block.c_norm.gamma"));
            // The MIMO projections exist only at `mimo_rank > 1`. The SISO
            // checkpoint holds none of them, and `Mamba3` holds `None` in
            // their place. So the rules have the condition of the tensors.
            if block.mimo_rank > 1 {
                rules.push((r"\.block\.mimo_x$", ".block.mimo_x_hmp"));
                rules.push((r"\.block\.mimo_z$", ".block.mimo_z_hmp"));
                rules.push((r"\.block\.mimo_o$", ".block.mimo_o_hmp"));
            }
        }
    }
    rules
}

/// The checkpoints tie the LM head to the embedding (`missing_lm_head: true`),
/// so the head is the transposed embedding table.
///
/// It is public because it is the second half of the build of a runnable
/// network. A build without a checkpoint must also call it. `benches/model.rs`
/// builds the same topology on random weights. Without this head, it measures
/// the tied head of the vocab network, which transposes the embedding at each
/// call.
pub fn tie_lm_head(mamba: &mut MambaVocabNet, device: &Device) {
    let weight = mamba.embedding.weight.val().swap_dims(0, 1);
    // `from_data(into_data())` forces the transposed view to be contiguous.
    let weight: Tensor<2> = Tensor::from_data(weight.into_data(), device);
    mamba.lm_head = Some(burn::nn::Linear {
        weight: Param::from_tensor(weight),
        bias: None,
    });
}

/// Checks [`key_remapping`] against the manifests of the checkpoints. No tensor
/// and no parameter of the load may stay without a match.
///
/// Without these tests, a remapping bug shows only after a fetch of about
/// 1.1GB, when the load fails. So this file holds the manifest of each
/// checkpoint, from `refs/pr/1` of its repo. The tests push each name through
/// the rules of the store. The result must be *exactly* the parameters of the
/// built network minus [`ALLOWED_MISSING`], with the same shapes. So the tests
/// find these gaps:
///
/// - A checkpoint tensor that reaches no parameter. The load only logs it, so
///   its values are lost.
/// - A parameter outside [`ALLOWED_MISSING`] that no tensor fills. The load
///   fails, but only after the fetch.
/// - A rule that matches nothing ([`remapping_rules_all_fire`]). Such a rule
///   does not describe the checkpoint.
/// - A manifest that is not the whole file
///   ([`manifests_cover_the_whole_checkpoint`]).
#[cfg(all(test, any(feature = "mamba1", feature = "mamba2", feature = "mamba3")))]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    /// `(checkpoint tensor name, checkpoint shape)`.
    type Manifest = &'static [(&'static str, &'static [usize])];

    /// The vendored manifest of one checkpoint.
    struct Vendored {
        /// The checkpoint of the manifest.
        spec: &'static crate::CheckpointSpec,
        /// The tensors outside `backbone.layers.`, and the tensors of layer 0.
        /// Each layer repeats the names and the shapes of layer 0, so layer 0
        /// stands for each layer.
        manifest: Manifest,
        /// The count of the layers.
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

    /// Each checkpoint of the families of this build, whatever set of family
    /// features the build has.
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

    /// The vocabulary size of the test network. It is small, so the test does
    /// not allocate the 130M or 187M parameters of a checkpoint. It is a
    /// multiple of each `pad_vocab_size_multiple`, so the padding does not
    /// change it.
    const TEST_VOCAB: usize = 64;

    /// [`key_remapping`] applied to a manifest.
    struct Remapped {
        /// `parameter path → shape`, as the store sees it.
        params: BTreeMap<String, Vec<usize>>,
        /// `(rule pattern, how many names it rewrote)`, in [`key_remapping`]
        /// order.
        rule_hits: Vec<(&'static str, usize)>,
    }

    /// One layer and a small vocabulary. The parameter *paths* do not depend on
    /// them, so the test builds a small network.
    fn one_layer(mut topology: MambaVocabNetConfig) -> MambaVocabNetConfig {
        topology.shape.stack.n_real_layers = 1;
        topology.shape.vocab_size = TEST_VOCAB;
        topology
    }

    /// The checkpoint names, pushed through [`key_remapping`] as
    /// `SafetensorsStore` applies it: each pattern, in order.
    ///
    /// Two names that remap onto one parameter cause a panic. In a load, the
    /// second tensor writes over the first, and the first has no effect.
    fn remapped(manifest: Manifest, topology: &MambaVocabNetConfig) -> Remapped {
        let rules = key_remapping(topology);
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
            // `PyTorchToBurnAdapter` transposes the `Linear` weights. Inside a
            // layer, each rank-2 `….weight` is a `Linear` weight. The norms
            // have rank 1, `conv1d.weight` has rank 3, and the bare `Param`s
            // (`A_log`, `B_bias`, `mimo_*`) have no `.weight` suffix. Outside
            // the layers, the only rank-2 weight is `embedding.weight`, and the
            // adapter does not transpose it.
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

    /// The parameters that the built network shows to the store.
    fn module_params(topology: &MambaVocabNetConfig) -> BTreeMap<String, Vec<usize>> {
        let device: Device = Default::default();
        let network = topology.init(&device);
        network
            .collect(None, None, true)
            .into_iter()
            .map(|tensor| (tensor.name, tensor.shape.to_vec()))
            .collect()
    }

    /// Neither side of the load may hold anything the other does not. Each
    /// checkpoint tensor reaches a parameter. A checkpoint tensor fills each
    /// parameter that is not in [`ALLOWED_MISSING`].
    #[test]
    fn checkpoints_and_modules_hold_the_same_tensors() {
        for vendored in VENDORED {
            let id = vendored.spec.id;
            let full = (vendored.spec.topology)();
            let topology = one_layer(full.clone());
            let mut expected = remapped(vendored.manifest, &topology).params;

            // The test cannot compare this shape with the small topology. The
            // row count of the checkpoint checks `pad_vocab_size_multiple`,
            // which differs between checkpoints.
            let embedding = expected
                .get_mut("embedding.weight")
                .unwrap_or_else(|| panic!("{id}: the embedding must remap to `embedding.weight`"));
            assert_eq!(
                embedding[0],
                crate::padded_vocab_size(&full),
                "{id}: the topology's padded vocabulary must match the checkpoint's embedding"
            );
            embedding[0] = TEST_VOCAB;

            let actual = module_params(&topology);
            let expected_keys: BTreeSet<&str> = expected.keys().map(String::as_str).collect();
            let actual_keys: BTreeSet<&str> = actual.keys().map(String::as_str).collect();
            let dangling: Vec<&str> = expected_keys.difference(&actual_keys).copied().collect();
            let unfilled: BTreeSet<&str> = actual_keys.difference(&expected_keys).copied().collect();
            let allowed: BTreeSet<&str> = ALLOWED_MISSING.iter().copied().collect();
            assert!(
                dangling.is_empty(),
                "{id}: checkpoint tensor(s) that reach no module parameter: {dangling:?}"
            );
            assert_eq!(
                unfilled, allowed,
                "{id}: the module parameters that no checkpoint tensor fills must be `ALLOWED_MISSING`"
            );
            for (path, want) in &expected {
                assert_eq!(&actual[path], want, "{id}: shape mismatch at {path}");
            }
        }
    }

    /// Each rule of [`key_remapping`] rewrites a name of each checkpoint. A rule
    /// that rewrites nothing has no use, or it is a rename that does not match
    /// the checkpoint. The test has no exception, so the condition of the MIMO
    /// rules is `mimo_rank`, not only the Mamba-3 family.
    #[test]
    fn remapping_rules_all_fire() {
        for vendored in VENDORED {
            let id = vendored.spec.id;
            let topology = one_layer((vendored.spec.topology)());
            let unused: Vec<&str> = remapped(vendored.manifest, &topology)
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

    /// Checks that each manifest covers the whole header of its weights file.
    /// The file must hold exactly two sets of tensors, with the same shapes.
    /// These are the tensors of the manifest outside the layers, and one copy
    /// of layer 0 for each layer.
    /// Without this test, the tests above show only that the *copied* part of
    /// the checkpoint loads.
    ///
    /// The test is `#[ignore]`. It reads `model.safetensors` from the hub cache
    /// of the native binary, and it **fetches** each missing weights file
    /// (about 1.1GB in total).
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

            // A safetensors file starts with the header length (8 bytes,
            // little-endian), then the JSON header:
            // `name → {dtype, shape, data_offsets}`.
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
