//! The "byte level" part of the tokenizer. It has the reversible byte↔char
//! alphabet of each BPE that comes from GPT-2. It also has the two word-split
//! regexes that this crate supports, as hand-written scanners.
//!
//! [`split_words`] is `pre_tokenizers::byte_level::ByteLevel` of `tokenizers`,
//! with `add_prefix_space: false` and `use_regex: true`.
//! `EleutherAI/gpt-neox-20b` asks for it. [`split_words_llama3`] is the `Split`
//! pre-tokenizer of `meta-llama/Llama-3.1-8B`. After it, the `ByteLevel` stage
//! only maps the bytes (`use_regex: false`).
//!
//! Each regex is an ordered alternation, and its alternatives match each
//! character. So a scan from left to right, where the first alternative that
//! matches wins, gives exactly the matches of `find_iter` of the regex crate.
//! The scan also copies the `?` and `*` backtracking. The code shows it by hand
//! where it changes a match.

use std::collections::HashMap;
use std::sync::OnceLock;

/// The printable char of each of the 256 bytes, indexed by the byte.
///
/// A byte that is a printable ASCII or Latin-1 char maps to itself. Each other
/// byte maps into the `U+0100..` range. So the BPE never sees whitespace or
/// control characters.
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
                // The unwrap cannot fail: 256 + n stays far below the
                // surrogate range.
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

/// Decodes byte-level `tokens` into a string.
///
/// The function decodes each token separately. A token with a char outside the
/// alphabet gives its own UTF-8 bytes. An added token such as `<|endoftext|>`
/// is an example, and a run of raw spaces is another. Then the function
/// decodes all the bytes lossily. So a multi-byte character that is not
/// complete shows as `U+FFFD`.
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
/// The alternatives are ordered, and together they match each character. So a
/// scan from left to right gives exactly the matches of `find_iter`, with no
/// regex engine and no lookahead.
///
/// [char::is_alphabetic] approximates `\p{L}`, and [char::is_numeric]
/// approximates `\p{N}`. They agree with the two classes for each script of
/// the training text of the 130m checkpoints. The only other chars that
/// `is_alphabetic` covers are the combining marks of some Indic scripts.
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

        // `\s+(?!\S)` then `\s+`: take the whitespace run. When a
        // non-whitespace char follows, leave the last char for the next match.
        // The negative lookahead makes the greedy `\s+` backtrack by that one
        // char.
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
/// `[^\s\p{L}\p{N}]`: punctuation and symbols.
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
/// The differences from [`split_words`] that change the token ids:
///
/// - The contractions are **case-insensitive**.
/// - A word can take any one non-letter in front, not only a space.
/// - The digits split into groups of **at most three**.
/// - A whitespace run that ends in a newline stays whole. It does not split
///   one character at a time.
///
/// The character classes have the same approximations as in [`split_words`]:
/// [char::is_alphabetic] for `\p{L}`, and [char::is_numeric] for `\p{N}`.
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
            if rest
                .get(..c.len())
                .is_some_and(|h| h.eq_ignore_ascii_case(c))
            {
                return c.len();
            }
        }
    }

    let mut chars = rest.char_indices();
    let (_, c0) = chars.next().expect("non-empty remainder");
    let c1 = chars.next().map(|(_, c)| c);

    // [^\r\n\p{L}\p{N}]?\p{L}+: the optional head is greedy, but it backtracks
    // when no letter follows it. So the code tries the form with no head
    // separately.
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

    // ` ?[^\s\p{L}\p{N}]+[\r\n]*`: the same backtracking as above.
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

    // `\s*[\r\n]+`: `\s*` backtracks until `[\r\n]+` can match. So the match
    // ends at the **last** newline of the run. For example, "  \n  x" gives
    // "  \n", not "  \n  ".
    let mut last_newline_end = 0;
    for (i, c) in rest[..ws].char_indices() {
        if is_newline(c) {
            last_newline_end = i + c.len_utf8();
        }
    }
    if last_newline_end > 0 {
        return last_newline_end;
    }

    // `\s+(?!\S)` then `\s+`: when a non-space follows, leave the last char
    // for the next match. A run of one char is the exception. There `\s+`
    // cannot leave its only char, so the lookahead alternative fails and the
    // bare `\s+` matches.
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
        assert_eq!(
            split_words("Hello, world!"),
            vec!["Hello", ",", " world", "!"]
        );
        // One space before a word goes with the word. The other spaces stay
        // apart.
        assert_eq!(split_words("a  b"), vec!["a", " ", " b"]);
        // Whitespace at the end stays whole, because nothing follows it.
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

    /// The reference implementation gave each expected value here: `tokenizers`
    /// 0.22.1, with the `tokenizer.json` of `meta-llama/Llama-3.1-8B`, through
    /// `pre_tokenizer.pre_tokenize_str`. Then a byte-level decode gave the plain
    /// text. No value comes from a manual reading of the regex.
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
            // Digits come in groups of at most three. A space before a digit
            // stays alone, because the ` ?…` head goes only before letters or
            // punctuation.
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

    /// The two splitters must stay different. A build that sends the Llama-3
    /// tokenizer through the GPT-2 scanner gives ids that look correct, but are
    /// wrong.
    #[test]
    fn the_two_splitters_disagree() {
        assert_eq!(split_words("\n\nhi"), vec!["\n", "\n", "hi"]);
        assert_eq!(split_words_llama3("\n\nhi"), vec!["\n\n", "hi"]);
        assert_eq!(split_words("1234567"), vec!["1234567"]);
        assert_eq!(split_words_llama3("1234567"), vec!["123", "456", "7"]);
    }
}
