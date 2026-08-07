//! A self-contained reader for HuggingFace `tokenizer.json` files, covering the
//! byte-level BPE pipeline that `EleutherAI/gpt-neox-20b` (the tokenizer this demo
//! uses for both Mamba checkpoints) declares:
//!
//! `NFC normalizer → ByteLevel pre-tokenizer → BPE model → ByteLevel decoder`,
//! plus the `added_tokens` table.
//!
//! This replaces the `tokenizers` crate. Anything the file asks for outside that
//! pipeline is rejected at load time rather than silently ignored.

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
        for (stage, component) in [
            ("pre_tokenizer", &json.pre_tokenizer),
            ("decoder", &json.decoder),
        ] {
            match component {
                Some(c) if c.kind == "ByteLevel" => {
                    if c.add_prefix_space {
                        anyhow::bail!("unsupported ByteLevel {stage} with add_prefix_space");
                    }
                    if !c.use_regex {
                        anyhow::bail!("unsupported ByteLevel {stage} without use_regex");
                    }
                }
                other => anyhow::bail!("unsupported {stage}: {:?}", other.as_ref().map(|c| &c.kind)),
            }
        }
        // The ByteLevel post-processor only trims offsets, which this crate never reads,
        // and it contributes no special tokens — so `add_special_tokens` is a no-op here.
        match &json.post_processor {
            None => {}
            Some(c) if c.kind == "ByteLevel" => {}
            Some(c) => anyhow::bail!("unsupported post_processor: {:?}", c.kind),
        }

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
        let bpe = Bpe::new(merges, &vocab);

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

        Ok(Self {
            vocab,
            tokens,
            added,
            added_first_bytes,
            special_ids,
            normalization,
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

    /// Encodes `text` into token ids.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let text = self.normalize(text);

        let mut ids = Vec::new();
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
        for word in byte_level::split_words(text) {
            let word = byte_level::encode_bytes(word);
            ids.extend(self.bpe.tokenize(&word, &self.vocab));
        }
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
