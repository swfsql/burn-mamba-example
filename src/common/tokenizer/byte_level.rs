//! The "byte level" boundary: the reversible byte↔char alphabet shared by every
//! GPT-2-descended BPE, plus the two word-splitting regexes this crate supports,
//! reimplemented as hand-rolled scanners.
//!
//! [`split_words`] is `tokenizers`' `pre_tokenizers::byte_level::ByteLevel` with
//! `add_prefix_space: false`, `use_regex: true` — what `EleutherAI/gpt-neox-20b`
//! asks for. [`split_words_llama3`] is the `Split` pre-tokenizer of
//! `meta-llama/Llama-3.1-8B`, whose `ByteLevel` stage then does the byte mapping
//! only (`use_regex: false`).
//!
//! Both are ordered alternations that between them match every character, so a
//! left-to-right "first alternative that matches wins" scan reproduces the
//! regex crate's `find_iter` exactly — including the `?`/`*` backtracking, which
//! is spelled out by hand where it bites.

use std::collections::HashMap;
use std::sync::OnceLock;

/// The 256 printable chars every byte is mapped to, indexed by that byte.
///
/// Bytes that are already printable ASCII/Latin-1 map to themselves; the rest are
/// shifted into the `U+0100..` range so the BPE never sees whitespace or control
/// characters.
pub fn bytes_char() -> &'static [char; 256] {
    static TABLE: OnceLock<[char; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = ['\0'; 256];
        let mut assigned = [false; 256];
        for b in (b'!'..=b'~').chain(0xA1..=0xAC).chain(0xAE..=0xFF) {
            table[b as usize] = b as char;
            assigned[b as usize] = true;
        }
        let mut n = 0u32;
        for b in 0..=255usize {
            if !assigned[b] {
                // Safety-free: 256 + n stays well below the surrogate range.
                table[b] = char::from_u32(256 + n).unwrap();
                n += 1;
            }
        }
        table
    })
}

/// The inverse of [bytes_char].
pub fn char_bytes() -> &'static HashMap<char, u8> {
    static TABLE: OnceLock<HashMap<char, u8>> = OnceLock::new();
    TABLE.get_or_init(|| {
        bytes_char()
            .iter()
            .enumerate()
            .map(|(b, c)| (*c, b as u8))
            .collect()
    })
}

/// Encodes `s`'s UTF-8 bytes into the byte-level alphabet.
pub fn encode_bytes(s: &str) -> String {
    let table = bytes_char();
    s.bytes().map(|b| table[b as usize]).collect()
}

/// Decodes byte-level `tokens` back into a string.
///
/// Each token is decoded independently; a token containing any char outside the
/// alphabet (an added token such as `<|endoftext|>` or a run of raw spaces) falls
/// back to its own UTF-8 bytes. The concatenation is then decoded lossily, which is
/// what lets a partially-generated multi-byte character surface as `U+FFFD`.
pub fn decode_bytes<'a>(tokens: impl IntoIterator<Item = &'a str>) -> String {
    let table = char_bytes();
    let mut bytes: Vec<u8> = Vec::new();
    for token in tokens {
        let mut decoded = Vec::with_capacity(token.len());
        let complete = token.chars().all(|c| match table.get(&c) {
            Some(b) => {
                decoded.push(*b);
                true
            }
            None => false,
        });
        if complete {
            bytes.extend(decoded);
        } else {
            bytes.extend_from_slice(token.as_bytes());
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Splits `text` the way the GPT-2 pre-tokenizer regex does:
///
/// ```text
/// 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
/// ```
///
/// The alternatives are ordered, and together they match every character, so a
/// left-to-right scan reproduces `find_iter` exactly — no regex engine (and no
/// lookahead support) required.
///
/// `\p{L}` is approximated by [char::is_alphabetic] and `\p{N}` by
/// [char::is_numeric]; the two agree for every script this 130m English LM was
/// trained on, and `is_alphabetic` only additionally covers combining marks used
/// by some Indic scripts.
pub fn split_words(text: &str) -> Vec<&str> {
    const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];

    let mut out = Vec::new();
    let mut start = 0;

    while start < text.len() {
        let rest = &text[start..];

        // 's|'t|'re|'ve|'m|'ll|'d
        if let Some(c) = CONTRACTIONS.iter().find(|c| rest.starts_with(**c)) {
            out.push(&rest[..c.len()]);
            start += c.len();
            continue;
        }

        let mut chars = rest.char_indices();
        let (_, first) = chars.next().expect("non-empty remainder");
        // The optional single leading space of ` ?\p{L}+` / ` ?\p{N}+` / ` ?[^...]+`.
        let (offset, head) = if first == ' ' {
            match chars.next() {
                Some((i, c)) => (i, Some(c)),
                None => (1, None),
            }
        } else {
            (0, Some(first))
        };

        // ` ?\p{L}+` | ` ?\p{N}+` | ` ?[^\s\p{L}\p{N}]+`
        if let Some(head) = head {
            let class: Option<fn(char) -> bool> = if head.is_alphabetic() {
                Some(|c: char| c.is_alphabetic())
            } else if head.is_numeric() {
                Some(|c: char| c.is_numeric())
            } else if !head.is_whitespace() {
                Some(|c: char| !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric())
            } else {
                None
            };
            if let Some(class) = class {
                let mut end = offset;
                for (i, c) in rest[offset..].char_indices() {
                    if class(c) {
                        end = offset + i + c.len_utf8();
                    } else {
                        break;
                    }
                }
                out.push(&rest[..end]);
                start += end;
                continue;
            }
        }

        // `\s+(?!\S)` then `\s+`: consume the whitespace run, but hand the last
        // character back when a non-whitespace follows (that is what the negative
        // lookahead makes the greedy `\s+` backtrack to).
        let mut run: Vec<usize> = Vec::new();
        for (i, c) in rest.char_indices() {
            if c.is_whitespace() {
                run.push(i + c.len_utf8());
            } else {
                break;
            }
        }
        debug_assert!(!run.is_empty(), "every char is covered by some alternative");
        let followed_by_non_space = run.last().copied().unwrap() < rest.len();
        let end = if followed_by_non_space && run.len() > 1 {
            run[run.len() - 2]
        } else {
            run[run.len() - 1]
        };
        out.push(&rest[..end]);
        start += end;
    }

    out
}

/// The longest run of leading characters satisfying `pred`, in bytes.
fn run_len(s: &str, pred: impl Fn(char) -> bool) -> usize {
    let mut end = 0;
    for (i, c) in s.char_indices() {
        if pred(c) {
            end = i + c.len_utf8();
        } else {
            break;
        }
    }
    end
}

fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}
fn is_number(c: char) -> bool {
    c.is_numeric()
}
/// `[^\s\p{L}\p{N}]` — punctuation and symbols.
fn is_punct(c: char) -> bool {
    !c.is_whitespace() && !is_letter(c) && !is_number(c)
}
fn is_newline(c: char) -> bool {
    c == '\r' || c == '\n'
}

/// Splits `text` the way the Llama-3 pre-tokenizer regex does:
///
/// ```text
/// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}
///   | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
/// ```
///
/// The differences from [`split_words`] that actually change token ids: the
/// contractions are **case-insensitive**, a word may absorb any single leading
/// non-letter (not just a space), digits are cut into groups of **at most
/// three**, and a whitespace run ending in a newline is emitted whole rather
/// than one-character-at-a-time.
///
/// Character-class caveats are the same as [`split_words`]' (`\p{L}` via
/// [char::is_alphabetic], `\p{N}` via [char::is_numeric]).
pub fn split_words_llama3(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let rest = &text[start..];
        let end = llama3_match(rest);
        debug_assert!(end > 0, "every char is covered by some alternative");
        out.push(&rest[..end]);
        start += end;
    }
    out
}

/// Byte length of the first Llama-3 alternative matching at the head of `rest`.
fn llama3_match(rest: &str) -> usize {
    // Listed in regex order: earlier alternatives win outright.
    const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];

    // (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if rest.starts_with('\'') {
        for c in CONTRACTIONS {
            if rest.get(..c.len()).is_some_and(|h| h.eq_ignore_ascii_case(c)) {
                return c.len();
            }
        }
    }

    let mut chars = rest.char_indices();
    let (_, c0) = chars.next().expect("non-empty remainder");
    let c1 = chars.next().map(|(_, c)| c);

    // [^\r\n\p{L}\p{N}]?\p{L}+ — the optional head is greedy but backtracks when
    // no letter follows it, so the no-head form has to be tried separately.
    if !is_newline(c0) && !is_letter(c0) && !is_number(c0) && c1.is_some_and(is_letter) {
        let off = c0.len_utf8();
        return off + run_len(&rest[off..], is_letter);
    }
    if is_letter(c0) {
        return run_len(rest, is_letter);
    }

    // \p{N}{1,3}
    if is_number(c0) {
        let mut len = 0;
        for (n, (i, c)) in rest.char_indices().enumerate() {
            if n == 3 || !is_number(c) {
                break;
            }
            len = i + c.len_utf8();
        }
        return len;
    }

    // ` ?[^\s\p{L}\p{N}]+[\r\n]*` — same backtracking shape as above.
    if c0 == ' ' && c1.is_some_and(is_punct) {
        let punct = run_len(&rest[1..], is_punct);
        return 1 + punct + run_len(&rest[1 + punct..], is_newline);
    }
    if is_punct(c0) {
        let punct = run_len(rest, is_punct);
        return punct + run_len(&rest[punct..], is_newline);
    }

    // Only whitespace can reach here.
    let ws = run_len(rest, is_whitespace_char);

    // `\s*[\r\n]+` — `\s*` backtracks until `[\r\n]+` can match, so this ends at
    // the run's **last** newline (e.g. "  \n  x" yields "  \n", not "  \n  ").
    let mut last_newline_end = 0;
    for (i, c) in rest[..ws].char_indices() {
        if is_newline(c) {
            last_newline_end = i + c.len_utf8();
        }
    }
    if last_newline_end > 0 {
        return last_newline_end;
    }

    // `\s+(?!\S)` then `\s+`: hand the final character back when a non-space
    // follows, unless the run is a single character (there `\s+` cannot give up
    // its only char, so the lookahead alternative fails and the bare `\s+` wins).
    if ws < rest.len() {
        let mut prev_end = 0;
        for (i, c) in rest[..ws].char_indices() {
            if i + c.len_utf8() == ws {
                break;
            }
            prev_end = i + c.len_utf8();
        }
        if prev_end > 0 {
            return prev_end;
        }
    }
    ws
}

fn is_whitespace_char(c: char) -> bool {
    c.is_whitespace()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_alphabet_is_a_bijection() {
        let table = bytes_char();
        assert_eq!(char_bytes().len(), 256);
        for b in 0..=255usize {
            assert_eq!(char_bytes()[&table[b]], b as u8);
        }
        assert_eq!(table[b' ' as usize], 'Ġ');
        assert_eq!(table[b'a' as usize], 'a');
    }

    #[test]
    fn splits_like_the_gpt2_regex() {
        assert_eq!(split_words("Mamba is the"), vec!["Mamba", " is", " the"]);
        assert_eq!(split_words("Hello, world!"), vec!["Hello", ",", " world", "!"]);
        // A lone space before a word attaches to the word; extra spaces stay behind.
        assert_eq!(split_words("a  b"), vec!["a", " ", " b"]);
        // Trailing whitespace is kept whole (nothing follows it).
        assert_eq!(split_words("a  "), vec!["a", "  "]);
        assert_eq!(split_words("don't stop"), vec!["don", "'t", " stop"]);
        assert_eq!(split_words("x1 22"), vec!["x", "1", " 22"]);
        // `\s+(?!\S)` hands the last whitespace char back to the next match.
        assert_eq!(split_words("\n\nhi"), vec!["\n", "\n", "hi"]);
        assert_eq!(split_words(""), Vec::<&str>::new());
    }

    #[test]
    fn byte_round_trip() {
        let s = "héllo — ok";
        assert_eq!(decode_bytes([encode_bytes(s).as_str()]), s);
    }

    /// Every expectation here was produced by the reference implementation —
    /// `tokenizers` 0.22.1 driving `meta-llama/Llama-3.1-8B`'s `tokenizer.json`
    /// via `pre_tokenizer.pre_tokenize_str` — and then byte-level-decoded back to
    /// plain text. They are not hand-derived from reading the regex.
    #[test]
    fn splits_like_the_llama3_regex() {
        #[rustfmt::skip]
        let cases: &[(&str, &[&str])] = &[
            ("Mamba is the",      &["Mamba", " is", " the"]),
            ("Hello, world!",     &["Hello", ",", " world", "!"]),
            ("a  b",              &["a", " ", " b"]),
            ("a  ",               &["a", "  "]),
            ("  ",                &["  "]),
            // Contractions are case-insensitive here, unlike GPT-2's.
            ("don't stop",        &["don", "'t", " stop"]),
            ("DON'T STOP",        &["DON", "'T", " STOP"]),
            ("It'S a Test'RE",    &["It", "'S", " a", " Test", "'RE"]),
            // Digits come in groups of at most three, and a space before a digit
            // is left on its own (the ` ?…` head only precedes letters/punct).
            ("x1 22",             &["x", "1", " ", "22"]),
            ("1234567",           &["123", "456", "7"]),
            ("The year 2024 was", &["The", " year", " ", "202", "4", " was"]),
            // A whitespace run ending in a newline is emitted whole...
            ("\n\nhi",            &["\n\n", "hi"]),
            ("a\n\n\nb",          &["a", "\n\n\n", "b"]),
            ("foo\r\nbar",        &["foo", "\r\n", "bar"]),
            // ...but only up to its *last* newline.
            ("  \n  x",           &["  \n", " ", " x"]),
            ("\t\tx",             &["\t", "\tx"]),
            // Any single non-letter may head a word, not just a space.
            ("(hello)",           &["(hello", ")"]),
            ("héllo — ok",        &["héllo", " —", " ok"]),
            ("",                  &[]),
        ];
        for (input, expected) in cases {
            assert_eq!(&split_words_llama3(input), expected, "input {input:?}");
        }
    }

    /// The two splitters must stay distinct — a build that quietly routed the
    /// Llama-3 tokenizer through the GPT-2 scanner would still produce plausible
    /// ids, just wrong ones.
    #[test]
    fn the_two_splitters_disagree() {
        assert_eq!(split_words("\n\nhi"), vec!["\n", "\n", "hi"]);
        assert_eq!(split_words_llama3("\n\nhi"), vec!["\n\n", "hi"]);
        assert_eq!(split_words("1234567"), vec!["1234567"]);
        assert_eq!(split_words_llama3("1234567"), vec!["123", "456", "7"]);
    }
}
