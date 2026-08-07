//! Byte-pair encoding over the byte-level alphabet.
//!
//! Matches `tokenizers`' `models::bpe::BPE` for the configurations used by
//! `EleutherAI/gpt-neox-20b` and `meta-llama/Llama-3.1-8B`: no dropout, no
//! unknown token, no continuing-subword prefix, no end-of-word suffix, and the
//! optional `ignore_merges` short-circuit that Llama-3 sets.

use std::collections::HashMap;

/// Rank (merge priority) and resulting id of a merge rule.
#[derive(Clone, Copy, Debug)]
struct Merge {
    rank: u32,
    new_id: u32,
}

/// The merge table, keyed by the pair of ids being joined.
#[derive(Debug, Default)]
pub struct Bpe {
    merges: HashMap<(u32, u32), Merge>,
    /// `model.ignore_merges`: emit a whole pre-tokenized word that is itself a
    /// vocabulary entry as that single id, without consulting the merge table.
    ignore_merges: bool,
}

impl Bpe {
    /// From the `model.merges` entries of a `tokenizer.json`, each `"<left> <right>"`,
    /// resolved against `vocab`. Entries naming an unknown piece are skipped, the way
    /// `tokenizers` does when a merge cannot be resolved.
    pub fn new<'a>(
        merges: impl IntoIterator<Item = (&'a str, &'a str)>,
        vocab: &HashMap<String, u32>,
        ignore_merges: bool,
    ) -> Self {
        let mut table = HashMap::new();
        for (rank, (left, right)) in merges.into_iter().enumerate() {
            let (Some(a), Some(b)) = (vocab.get(left), vocab.get(right)) else {
                continue;
            };
            let joined = format!("{left}{right}");
            let Some(new_id) = vocab.get(&joined) else {
                continue;
            };
            table.entry((*a, *b)).or_insert(Merge {
                rank: rank as u32,
                new_id: *new_id,
            });
        }
        Self {
            merges: table,
            ignore_merges,
        }
    }

    /// Tokenizes one pre-tokenized word (already in the byte-level alphabet).
    ///
    /// Repeatedly applies the lowest-ranked applicable merge, breaking ties towards
    /// the left-most pair — the same order `tokenizers`' merge heap produces.
    /// Characters absent from `vocab` are dropped (there is no unknown token).
    ///
    /// Under `ignore_merges` a word present in the vocabulary is emitted as-is
    /// first. This is not an optimisation: for Llama-3 the merge table can reach
    /// a *different* segmentation of a word that also exists whole, so skipping
    /// the check changes ids.
    pub fn tokenize(&self, word: &str, vocab: &HashMap<String, u32>) -> Vec<u32> {
        if self.ignore_merges
            && let Some(id) = vocab.get(word)
        {
            return vec![*id];
        }

        let mut symbols: Vec<u32> = Vec::with_capacity(word.len());
        let mut buf = [0u8; 4];
        for c in word.chars() {
            match vocab.get(c.encode_utf8(&mut buf) as &str) {
                Some(id) => symbols.push(*id),
                None => log::warn!("byte-level char {c:?} is missing from the vocabulary"),
            }
        }

        while symbols.len() > 1 {
            let mut best: Option<(usize, Merge)> = None;
            for i in 0..symbols.len() - 1 {
                let Some(merge) = self.merges.get(&(symbols[i], symbols[i + 1])) else {
                    continue;
                };
                if best.is_none_or(|(_, b)| merge.rank < b.rank) {
                    best = Some((i, *merge));
                }
            }
            let Some((i, merge)) = best else { break };
            symbols[i] = merge.new_id;
            symbols.remove(i + 1);
        }

        symbols
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_vocab() -> HashMap<String, u32> {
        ["a", "b", "c", "ab", "bc", "abc"]
            .into_iter()
            .enumerate()
            .map(|(i, s)| (s.to_string(), i as u32))
            .collect()
    }

    fn toy() -> (HashMap<String, u32>, Bpe) {
        let vocab = toy_vocab();
        // "b c" outranks "a b", so "abc" must merge right-to-left first.
        let bpe = Bpe::new([("b", "c"), ("a", "b"), ("a", "bc")], &vocab, false);
        (vocab, bpe)
    }

    #[test]
    fn merges_by_rank_not_position() {
        let (vocab, bpe) = toy();
        assert_eq!(bpe.tokenize("abc", &vocab), vec![vocab["abc"]]);
        assert_eq!(bpe.tokenize("ab", &vocab), vec![vocab["ab"]]);
        assert_eq!(bpe.tokenize("a", &vocab), vec![vocab["a"]]);
        assert_eq!(bpe.tokenize("", &vocab), Vec::<u32>::new());
    }

    #[test]
    fn unknown_chars_are_dropped() {
        let (vocab, bpe) = toy();
        assert_eq!(bpe.tokenize("azb", &vocab), vec![vocab["ab"]]);
    }

    /// `ignore_merges` must take the whole-word id even when the merge table
    /// would have segmented the word differently — the two disagree here, which
    /// is exactly why the flag is not a mere fast path.
    #[test]
    fn ignore_merges_prefers_the_whole_word() {
        let vocab = toy_vocab();
        // Only "a"+"b" is reachable by merging, so without the flag "abc"
        // segments as ["ab", "c"]; the whole word "abc" is in the vocab though.
        let merges = [("a", "b")];
        let plain = Bpe::new(merges, &vocab, false);
        assert_eq!(plain.tokenize("abc", &vocab), vec![vocab["ab"], vocab["c"]]);

        let ignoring = Bpe::new(merges, &vocab, true);
        assert_eq!(ignoring.tokenize("abc", &vocab), vec![vocab["abc"]]);
        // Words absent from the vocabulary still go through the merge table.
        assert_eq!(ignoring.tokenize("ab", &vocab), vec![vocab["ab"]]);
    }
}
