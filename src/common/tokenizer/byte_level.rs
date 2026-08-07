//! The GPT-2 "byte level" boundary: the reversible byte↔char alphabet and the
//! word-splitting regex, reimplemented as a hand-rolled scanner.
//!
//! Equivalent to `tokenizers`' `pre_tokenizers::byte_level::ByteLevel` with
//! `add_prefix_space: false`, `use_regex: true` (which is what
//! `EleutherAI/gpt-neox-20b`'s `tokenizer.json` asks for).

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
}
