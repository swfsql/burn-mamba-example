//! A self-contained reader for HuggingFace `tokenizer.json` files, covering the
//! two byte-level BPE pipelines this demo needs.
//!
//! `EleutherAI/gpt-neox-20b` (the Mamba-1/Mamba-2 130m tokenizer):
//!
//! `NFC normalizer → ByteLevel pre-tokenizer → BPE model → ByteLevel decoder`
//!
//! `meta-llama/Llama-3.1-8B` (the Mamba-3 187m tokenizer):
//!
//! `no normalizer → Sequence[Split(regex), ByteLevel(use_regex: false)]
//!    → BPE model (ignore_merges) → Sequence[ByteLevel, TemplateProcessing]
//!    → ByteLevel decoder`
//!
//! plus the `added_tokens` table in both cases. The two differ in more than
//! configuration — see [`byte_level::split_words_llama3`] for the pre-tokenizer
//! and [`bpe::Bpe::tokenize`] for `ignore_merges`.
//!
//! This replaces the `tokenizers` crate. Anything the file asks for outside those
//! pipelines is rejected at load time rather than silently ignored.

mod bpe;
mod byte_level;

use bpe::Bpe;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use unicode_normalization::UnicodeNormalization;

/// A loaded byte-level BPE tokenizer.
#[derive(Debug)]
pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    tokens: HashMap<u32, String>,
    /// Added tokens, longest content first, so scanning yields the longest match.
    added: Vec<(String, u32)>,
    /// First byte of every added token's content, to skip most scan positions.
    added_first_bytes: [bool; 256],
    special_ids: HashSet<u32>,
    normalization: Normalization,
    pre_tokenization: PreTokenization,
    /// Ids the post-processor prepends to every sequence (Llama-3's
    /// `<|begin_of_text|>`; empty for gpt-neox). See [`Self::encode`].
    prefix_ids: Vec<u32>,
    bpe: Bpe,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Normalization {
    None,
    Nfc,
    Nfd,
    Nfkc,
    Nfkd,
}

/// Which pre-tokenizer regex splits text into words before the BPE runs.
#[derive(Clone, Copy, Debug, PartialEq)]
enum PreTokenization {
    /// A single `ByteLevel` doing both the split and the byte mapping.
    Gpt2,
    /// `Split(regex)` for the words, then `ByteLevel` for the mapping only.
    Llama3,
}

/// The one `Split` pattern this crate implements, from
/// `meta-llama/Llama-3.1-8B`. Compared verbatim: a pattern that differs by so
/// much as a quantifier tokenizes differently, and silently accepting it would
/// produce plausible-looking but wrong ids.
const LLAMA3_SPLIT_PATTERN: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
);

impl Tokenizer {
    /// Parses a `tokenizer.json`.
    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        let json: TokenizerJson = serde_json::from_slice(bytes)?;

        if json.model.kind != "BPE" {
            anyhow::bail!("unsupported tokenizer model {:?}, expected BPE", json.model.kind);
        }
        for (field, value) in [
            ("continuing_subword_prefix", &json.model.continuing_subword_prefix),
            ("end_of_word_suffix", &json.model.end_of_word_suffix),
            ("unk_token", &json.model.unk_token),
        ] {
            if let Some(value) = value {
                anyhow::bail!("unsupported BPE {field}: {value:?}");
            }
        }
        if json.model.byte_fallback {
            anyhow::bail!("unsupported BPE byte_fallback");
        }

        let pre_tokenization = parse_pre_tokenizer(json.pre_tokenizer.as_ref())?;

        // As a *decoder* `ByteLevel` only maps the alphabet back to bytes — it
        // reads neither `add_prefix_space` nor `use_regex` — so both are accepted
        // here even though the pre-tokenizer stage is strict about them.
        match &json.decoder {
            Some(c) if c.kind == "ByteLevel" => {}
            other => anyhow::bail!("unsupported decoder: {:?}", other.as_ref().map(|c| &c.kind)),
        }

        let prefix_tokens = parse_post_processor(json.post_processor.as_ref())?;

        let normalization = match json.normalizer.as_ref().map(|n| n.kind.as_str()) {
            None => Normalization::None,
            Some("NFC") => Normalization::Nfc,
            Some("NFD") => Normalization::Nfd,
            Some("NFKC") => Normalization::Nfkc,
            Some("NFKD") => Normalization::Nfkd,
            Some(other) => anyhow::bail!("unsupported normalizer: {other:?}"),
        };

        let vocab = json.model.vocab;
        let merges = json
            .model
            .merges
            .iter()
            .map(MergeJson::pair)
            .collect::<anyhow::Result<Vec<_>>>()?;
        let bpe = Bpe::new(merges, &vocab, json.model.ignore_merges);

        let mut tokens: HashMap<u32, String> =
            vocab.iter().map(|(t, id)| (*id, t.clone())).collect();
        let mut added: Vec<(String, u32)> = Vec::with_capacity(json.added_tokens.len());
        let mut special_ids = HashSet::new();
        let mut added_first_bytes = [false; 256];
        for token in json.added_tokens {
            if let Some(byte) = token.content.as_bytes().first() {
                added_first_bytes[*byte as usize] = true;
            }
            tokens.insert(token.id, token.content.clone());
            if token.special {
                special_ids.insert(token.id);
            }
            added.push((token.content, token.id));
        }
        // Longest first: `find_added` then reports the longest match at a position.
        added.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        // Resolved only now: the post-processor names its special tokens by
        // content, and they live in `added_tokens` rather than the base vocab.
        let prefix_ids = prefix_tokens
            .iter()
            .map(|token| {
                vocab
                    .get(token)
                    .copied()
                    .or_else(|| added.iter().find(|(c, _)| c == token).map(|(_, id)| *id))
                    .ok_or_else(|| {
                        anyhow::anyhow!("post_processor special token {token:?} is not in the vocabulary")
                    })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        Ok(Self {
            vocab,
            tokens,
            added,
            added_first_bytes,
            special_ids,
            normalization,
            pre_tokenization,
            prefix_ids,
            bpe,
        })
    }

    /// Parses a `tokenizer.json` from disk.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_file(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    /// Number of entries in the base vocabulary (added tokens excluded).
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Looks up a token's id, added tokens included.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab
            .get(token)
            .copied()
            .or_else(|| self.added.iter().find(|(c, _)| c == token).map(|(_, id)| *id))
    }

    /// Looks up a token's text, added tokens included.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.tokens.get(&id).map(|s| s.as_str())
    }

    /// Encodes `text` into token ids, including whatever the post-processor
    /// prepends (Llama-3's `<|begin_of_text|>`; nothing for gpt-neox).
    ///
    /// The prefix is part of the encoding rather than an opt-in: the reference
    /// `AutoTokenizer` applies the post-processor by default, so this is the
    /// input distribution the checkpoints were trained and evaluated under.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let text = self.normalize(text);

        let mut ids = self.prefix_ids.clone();
        let mut chunk_start = 0;
        let mut pos = 0;
        while pos < text.len() {
            let Some((len, id)) = self.find_added(&text[pos..]) else {
                // Not an added token: advance one character.
                pos += text[pos..].chars().next().map_or(1, char::len_utf8);
                continue;
            };
            self.encode_chunk(&text[chunk_start..pos], &mut ids);
            ids.push(id);
            pos += len;
            chunk_start = pos;
        }
        self.encode_chunk(&text[chunk_start..], &mut ids);

        ids
    }

    /// Decodes token ids back into text, optionally dropping special tokens.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> String {
        let tokens = ids
            .iter()
            .filter(|id| !(skip_special_tokens && self.special_ids.contains(id)))
            .filter_map(|id| self.id_to_token(*id));
        byte_level::decode_bytes(tokens)
    }

    fn normalize(&self, text: &str) -> String {
        match self.normalization {
            Normalization::None => text.to_string(),
            Normalization::Nfc => text.nfc().collect(),
            Normalization::Nfd => text.nfd().collect(),
            Normalization::Nfkc => text.nfkc().collect(),
            Normalization::Nfkd => text.nfkd().collect(),
        }
    }

    /// The longest added token that `text` starts with, as `(byte length, id)`.
    fn find_added(&self, text: &str) -> Option<(usize, u32)> {
        if !self.added_first_bytes[*text.as_bytes().first()? as usize] {
            return None;
        }
        self.added
            .iter()
            .find(|(content, _)| text.starts_with(content.as_str()))
            .map(|(content, id)| (content.len(), *id))
    }

    /// Pre-tokenizes then BPE-encodes a stretch of text holding no added token.
    fn encode_chunk(&self, text: &str, ids: &mut Vec<u32>) {
        let words = match self.pre_tokenization {
            PreTokenization::Gpt2 => byte_level::split_words(text),
            PreTokenization::Llama3 => byte_level::split_words_llama3(text),
        };
        for word in words {
            let word = byte_level::encode_bytes(word);
            ids.extend(self.bpe.tokenize(&word, &self.vocab));
        }
    }
}

/// Identifies the pre-tokenizer as one of the two supported shapes.
///
/// gpt-neox declares a lone `ByteLevel` that both splits and maps; Llama-3
/// declares `Sequence[Split(Regex), ByteLevel(use_regex: false)]`, where only the
/// `Split` splits and the `ByteLevel` maps.
fn parse_pre_tokenizer(component: Option<&ComponentJson>) -> anyhow::Result<PreTokenization> {
    let Some(c) = component else {
        anyhow::bail!("unsupported pre_tokenizer: none");
    };
    match c.kind.as_str() {
        "ByteLevel" => {
            if c.add_prefix_space {
                anyhow::bail!("unsupported ByteLevel pre_tokenizer with add_prefix_space");
            }
            if !c.use_regex {
                anyhow::bail!("unsupported ByteLevel pre_tokenizer without use_regex");
            }
            Ok(PreTokenization::Gpt2)
        }
        "Sequence" => {
            let [split, byte_level] = c.pretokenizers.as_slice() else {
                anyhow::bail!(
                    "unsupported pre_tokenizer Sequence of {} stages, expected Split + ByteLevel",
                    c.pretokenizers.len()
                );
            };
            if split.kind != "Split" || byte_level.kind != "ByteLevel" {
                anyhow::bail!(
                    "unsupported pre_tokenizer Sequence [{:?}, {:?}]",
                    split.kind,
                    byte_level.kind
                );
            }
            if split.behavior.as_deref() != Some("Isolated") {
                anyhow::bail!("unsupported Split behavior: {:?}", split.behavior);
            }
            if split.invert {
                anyhow::bail!("unsupported inverted Split pre_tokenizer");
            }
            let pattern = split
                .pattern
                .as_ref()
                .and_then(|p| p.regex.as_deref())
                .ok_or_else(|| anyhow::anyhow!("Split pre_tokenizer without a Regex pattern"))?;
            if pattern != LLAMA3_SPLIT_PATTERN {
                anyhow::bail!("unsupported Split pattern: {pattern:?}");
            }
            // This ByteLevel only maps bytes; splitting already happened.
            if byte_level.add_prefix_space {
                anyhow::bail!("unsupported ByteLevel pre_tokenizer with add_prefix_space");
            }
            if byte_level.use_regex {
                anyhow::bail!("a Split pre_tokenizer must be followed by ByteLevel use_regex: false");
            }
            Ok(PreTokenization::Llama3)
        }
        other => anyhow::bail!("unsupported pre_tokenizer: {other:?}"),
    }
}

/// The special tokens the post-processor prepends to a single sequence.
///
/// `ByteLevel` here only trims offsets, which this crate never reads, and
/// contributes no tokens. `TemplateProcessing` may prepend specials (Llama-3's
/// `<|begin_of_text|>`); anything it would *append*, or any template that does
/// not start with the sequence placeholder after its prefix, is rejected rather
/// than half-applied.
fn parse_post_processor(component: Option<&ComponentJson>) -> anyhow::Result<Vec<String>> {
    let Some(c) = component else {
        return Ok(Vec::new());
    };
    match c.kind.as_str() {
        "ByteLevel" => Ok(Vec::new()),
        "TemplateProcessing" => {
            let mut prefix = Vec::new();
            let mut seen_sequence = false;
            for piece in &c.single {
                match piece {
                    TemplatePieceJson::SpecialToken { id } => {
                        if seen_sequence {
                            anyhow::bail!(
                                "unsupported post_processor: special token {id:?} after the sequence"
                            );
                        }
                        prefix.push(id.clone());
                    }
                    TemplatePieceJson::Sequence { .. } => seen_sequence = true,
                }
            }
            if !seen_sequence {
                anyhow::bail!("unsupported post_processor: template has no sequence placeholder");
            }
            Ok(prefix)
        }
        "Sequence" => {
            let mut prefix = Vec::new();
            for inner in &c.processors {
                prefix.extend(parse_post_processor(Some(inner))?);
            }
            Ok(prefix)
        }
        other => anyhow::bail!("unsupported post_processor: {other:?}"),
    }
}

#[derive(Debug, Deserialize)]
struct TokenizerJson {
    #[serde(default)]
    added_tokens: Vec<AddedTokenJson>,
    #[serde(default)]
    normalizer: Option<ComponentJson>,
    #[serde(default)]
    pre_tokenizer: Option<ComponentJson>,
    #[serde(default)]
    post_processor: Option<ComponentJson>,
    #[serde(default)]
    decoder: Option<ComponentJson>,
    model: ModelJson,
}

#[derive(Debug, Deserialize)]
struct AddedTokenJson {
    id: u32,
    content: String,
    #[serde(default)]
    special: bool,
}

/// The shared shape of the `normalizer` / `pre_tokenizer` / `post_processor` /
/// `decoder` entries; only the fields this pipeline cares about are read.
#[derive(Debug, Deserialize)]
struct ComponentJson {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    add_prefix_space: bool,
    #[serde(default = "default_true")]
    use_regex: bool,
    /// Children of a `Sequence` pre_tokenizer.
    #[serde(default)]
    pretokenizers: Vec<ComponentJson>,
    /// Children of a `Sequence` post_processor.
    #[serde(default)]
    processors: Vec<ComponentJson>,
    /// `Split` only.
    #[serde(default)]
    pattern: Option<PatternJson>,
    /// `Split` only — how the match itself is treated relative to the pieces.
    #[serde(default)]
    behavior: Option<String>,
    /// `Split` only — match the complement of the pattern.
    #[serde(default)]
    invert: bool,
    /// `TemplateProcessing` only: the single-sequence template.
    #[serde(default)]
    single: Vec<TemplatePieceJson>,
}

/// A `Split` pattern. Only the `Regex` form is understood (`String` would be a
/// literal delimiter, which neither supported tokenizer uses).
#[derive(Debug, Deserialize)]
struct PatternJson {
    #[serde(rename = "Regex")]
    #[serde(default)]
    regex: Option<String>,
}

/// One entry of a `TemplateProcessing` template. The placeholder's own `id`
/// ("A"/"B") only matters for pair templates, which are rejected upstream.
#[derive(Debug, Deserialize)]
enum TemplatePieceJson {
    SpecialToken { id: String },
    Sequence {},
}

#[derive(Debug, Deserialize)]
struct ModelJson {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    unk_token: Option<String>,
    #[serde(default)]
    continuing_subword_prefix: Option<String>,
    #[serde(default)]
    end_of_word_suffix: Option<String>,
    #[serde(default)]
    byte_fallback: bool,
    /// Emit a pre-tokenized word that is itself a vocabulary entry as that id,
    /// bypassing the merge table (set by Llama-3).
    #[serde(default)]
    ignore_merges: bool,
    vocab: HashMap<String, u32>,
    merges: Vec<MergeJson>,
}

/// Merges are `"left right"` in v1 files and `["left", "right"]` in newer ones.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum MergeJson {
    Joined(String),
    Pair([String; 2]),
}

impl MergeJson {
    fn pair(&self) -> anyhow::Result<(&str, &str)> {
        match self {
            MergeJson::Pair([left, right]) => Ok((left, right)),
            MergeJson::Joined(joined) => joined
                .split_once(' ')
                .ok_or_else(|| anyhow::anyhow!("malformed merge rule {joined:?}")),
        }
    }
}

fn default_true() -> bool {
    true
}

/// Whole-pipeline parity against the reference `tokenizers` crate.
///
/// The unit tests above pin the pieces; this pins the composition on a real
/// `tokenizer.json`, which is too large to vendor. Point `TOKENIZER_JSON` at one
/// and the expectations in `TOKENIZER_IDS` (a JSON object of `text -> ids`
/// produced by `tokenizers`) are replayed:
///
/// ```sh
/// TOKENIZER_JSON=… TOKENIZER_IDS=… cargo test --no-default-features -- --ignored
/// ```
#[cfg(all(test, not(target_arch = "wasm32")))]
mod parity_tests {
    use super::*;

    #[test]
    #[ignore = "needs a downloaded tokenizer.json; see the module docs"]
    fn encodes_like_the_reference() {
        let Ok(path) = std::env::var("TOKENIZER_JSON") else {
            panic!("set TOKENIZER_JSON to a tokenizer.json path");
        };
        let expectations = std::env::var("TOKENIZER_IDS").expect("set TOKENIZER_IDS");
        let expected: HashMap<String, Vec<u32>> =
            serde_json::from_slice(&std::fs::read(expectations).unwrap()).unwrap();

        let tokenizer = Tokenizer::from_file(path).expect("the pipeline must be supported");
        let mut failures = 0;
        for (text, want) in &expected {
            let got = tokenizer.encode(text);
            if &got != want {
                eprintln!("{text:?}\n  want {want:?}\n  got  {got:?}");
                failures += 1;
            }
            // Round-tripping must also recover the text (specials dropped).
            let back = tokenizer.decode(&got, true);
            if &back != text {
                eprintln!("{text:?} decoded back to {back:?}");
                failures += 1;
            }
        }
        assert_eq!(failures, 0, "{failures} case(s) disagreed with the reference");
    }
}
