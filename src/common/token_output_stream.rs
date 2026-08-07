//! Adapted from [candle-examples](https://github.com/huggingface/candle/blob/main/candle-examples/src/token_output_stream.rs),
//! rewired onto [crate::tokenizer::Tokenizer].

use crate::tokenizer::Tokenizer;

/// This is a wrapper around a tokenizer to ensure that tokens can be returned to the user in a
/// streaming way rather than having to wait for the full decoding.
pub struct TokenOutputStream {
    tokenizer: Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl TokenOutputStream {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    pub fn into_inner(self) -> Tokenizer {
        self.tokenizer
    }

    fn decode(&self, tokens: &[u32]) -> String {
        self.tokenizer.decode(tokens, true)
    }

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

    pub fn decode_all(&self) -> String {
        self.decode(&self.tokens)
    }

    pub fn get_token(&self, token_s: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token_s)
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }
}
