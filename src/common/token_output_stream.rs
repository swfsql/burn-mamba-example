//! The token output stream of
//! [candle-examples](https://github.com/huggingface/candle/blob/main/candle-examples/src/token_output_stream.rs),
//! with [crate::tokenizer::Tokenizer] in place of the `tokenizers` crate.

use crate::tokenizer::Tokenizer;

/// A tokenizer with a stream of token ids. It gives the text of each new token
/// when the token comes, not only at the end.
pub struct TokenOutputStream {
    tokenizer: Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl TokenOutputStream {
    /// Makes an empty stream with `tokenizer`.
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    /// Returns the tokenizer, and drops the stream.
    pub fn into_inner(self) -> Tokenizer {
        self.tokenizer
    }

    fn decode(&self, tokens: &[u32]) -> String {
        self.tokenizer.decode(tokens, true)
    }

    /// Appends `token` to the stream. Returns the new text when the decoded
    /// text grows and ends with a letter. Else it returns `None`, and a later
    /// token gives the text.
    // https://github.com/huggingface/text-generation-inference/blob/5ba53d44a18983a4de32d122f4cb46f4a17d9ef6/server/text_generation_server/models/model.py#L68
    pub fn next_token(&mut self, token: u32) -> Option<String> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)
        };
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..]);
        if text.len() > prev_text.len() && text.chars().last().unwrap().is_alphabetic() {
            let text = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Some(text.1.to_string())
        } else {
            None
        }
    }

    /// The text of the last tokens, which [Self::next_token] held back.
    pub fn decode_rest(&self) -> Option<String> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)
        };
        let text = self.decode(&self.tokens[self.prev_index..]);
        if text.len() > prev_text.len() {
            let text = text.split_at(prev_text.len());
            Some(text.1.to_string())
        } else {
            None
        }
    }

    /// The text of all the tokens of the stream.
    pub fn decode_all(&self) -> String {
        self.decode(&self.tokens)
    }

    /// The id of the token `token_s`, added tokens included.
    pub fn get_token(&self, token_s: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token_s)
    }

    /// The tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Removes all the tokens of the stream.
    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }
}
