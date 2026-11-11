//! A reader of HuggingFace `tokenizer.json` files. It covers the two byte-level
//! BPE tokenizer pipelines that the app needs. The app does not use the
//! `tokenizers` crate (ADR-0001).
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
//! Both pipelines also have the `added_tokens` table. The two differ in more
//! than the values of their components. See `byte_level::split_words_llama3`
//! for the pre-tokenizer, and `bpe::Bpe::tokenize` for `ignore_merges`.
//!
//! When it parses the file, the reader rejects each component type and each
//! component value outside these two pipelines. It ignores only the JSON
//! fields that it does not read.

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
    /// The added tokens, longest content first. So a scan finds the longest
    /// match.
    added: Vec<(String, u32)>,
    /// The first byte of the content of each added token. The scan skips each
    /// position that starts with another byte.
    added_first_bytes: [bool; 256],
    special_ids: HashSet<u32>,
    normalization: Normalization,
    pre_tokenization: PreTokenization,
    /// The ids that the post-processor puts in front of each sequence:
    /// `<|begin_of_text|>` for Llama-3, none for gpt-neox. See
    /// [`Self::encode`].
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

/// The one `Split` pattern that this crate supports, from
/// `meta-llama/Llama-3.1-8B`. The reader compares the pattern of the file with
/// it, character by character. A pattern with one other quantifier splits the
/// text differently, and gives ids that look correct but are wrong.
const LLAMA3_SPLIT_PATTERN: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
);

impl Tokenizer {
    /// Parses a `tokenizer.json`.
    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        let json: TokenizerJson = serde_json::from_slice(bytes)?;

        if json.model.kind != "BPE" {
            anyhow::bail!(
                "unsupported tokenizer model {:?}, expected BPE",
                json.model.kind
            );
        }
        for (field, value) in [
            (
                "continuing_subword_prefix",
                &json.model.continuing_subword_prefix,
            ),
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

        // As a *decoder*, `ByteLevel` only maps the alphabet back to bytes. It
        // reads neither `add_prefix_space` nor `use_regex`. So the decoder
        // accepts each value of the two fields, but the pre-tokenizer stage
        // does not.
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

        // The reader resolves the prefix ids only now. The post-processor names
        // its special tokens by their content, and they are in `added_tokens`,
        // not in the base vocabulary.
        let prefix_ids = prefix_tokens
            .iter()
            .map(|token| {
                vocab
                    .get(token)
                    .copied()
                    .or_else(|| added.iter().find(|(c, _)| c == token).map(|(_, id)| *id))
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "post_processor special token {token:?} is not in the vocabulary"
                        )
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
        self.vocab.get(token).copied().or_else(|| {
            self.added
                .iter()
                .find(|(c, _)| c == token)
                .map(|(_, id)| *id)
        })
    }

    /// Looks up a token's text, added tokens included.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.tokens.get(&id).map(|s| s.as_str())
    }

    /// Encodes `text` into token ids. The ids start with the prefix of the
    /// post-processor: `<|begin_of_text|>` for Llama-3, none for gpt-neox.
    ///
    /// The prefix is always part of the encoding. The reference
    /// `AutoTokenizer` applies the post-processor by default. So the
    /// checkpoints saw this input in their training and in their evaluation.
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

    /// Decodes token ids into text. With `skip_special_tokens`, it drops the
    /// special tokens.
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

    /// Pre-tokenizes, then BPE-encodes, a part of the text with no added token.
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

/// Finds which of the two supported pre-tokenizers the file declares.
///
/// gpt-neox declares one `ByteLevel` that splits and maps. Llama-3 declares
/// `Sequence[Split(Regex), ByteLevel(use_regex: false)]`: the `Split` splits,
/// and the `ByteLevel` only maps.
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
            // This ByteLevel only maps bytes. The Split before it splits the
            // text.
            if byte_level.add_prefix_space {
                anyhow::bail!("unsupported ByteLevel pre_tokenizer with add_prefix_space");
            }
            if byte_level.use_regex {
                anyhow::bail!(
                    "the ByteLevel after a Split pre_tokenizer must have use_regex: false"
                );
            }
            Ok(PreTokenization::Llama3)
        }
        other => anyhow::bail!("unsupported pre_tokenizer: {other:?}"),
    }
}

/// The special tokens that the post-processor puts in front of a single
/// sequence.
///
/// Here `ByteLevel` only trims offsets, and this crate never reads offsets. So
/// it adds no token. `TemplateProcessing` can put special tokens in front, as
/// the `<|begin_of_text|>` of Llama-3. The function rejects a template that
/// adds a token *after* the sequence, or a template with no sequence
/// placeholder. It does not apply such a template in part.
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

/// The shared shape of the `normalizer`, `pre_tokenizer`, `post_processor` and
/// `decoder` entries. The reader reads only the fields that the supported
/// pipelines use.
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
    /// `Split` only: where the match goes, relative to the pieces.
    #[serde(default)]
    behavior: Option<String>,
    /// `Split` only: match the complement of the pattern.
    #[serde(default)]
    invert: bool,
    /// `TemplateProcessing` only: the single-sequence template.
    #[serde(default)]
    single: Vec<TemplatePieceJson>,
}

/// A `Split` pattern. The reader supports only the `Regex` form. The `String`
/// form is a literal delimiter, and no supported tokenizer uses it.
#[derive(Debug, Deserialize)]
struct PatternJson {
    #[serde(rename = "Regex")]
    #[serde(default)]
    regex: Option<String>,
}

/// One entry of a `TemplateProcessing` template. The `id` of the placeholder
/// ("A" or "B") matters only in a pair template. The reader reads only the
/// single-sequence template.
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
    /// Give a pre-tokenized word that is a vocabulary entry as its id, with no
    /// merge. Llama-3 sets it.
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

/// The parity of the whole pipeline with the reference `tokenizers` crate.
///
/// The unit tests of `bpe` and `byte_level` check the parts. This test checks
/// the whole pipeline on a real `tokenizer.json`, which is too large to
/// vendor. Set `TOKENIZER_JSON` to the path of one. Set `TOKENIZER_IDS` to a
/// JSON object of `text -> ids` from `tokenizers`. Then run the test:
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
            // The decode of the ids must give the text again, with no special
            // token.
            let back = tokenizer.decode(&got, true);
            if &back != text {
                eprintln!("{text:?} decoded back to {back:?}");
                failures += 1;
            }
        }
        assert_eq!(
            failures, 0,
            "{failures} case(s) disagreed with the reference"
        );
    }
}
