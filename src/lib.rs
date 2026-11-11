//! A demo app that runs a pretrained Mamba checkpoint to generate text,
//! natively or in the browser.
//!
//! - [common]: the core, which each front-end and each backend uses.
//! - `native`: the native binary (the `native` feature).
//! - `wasm`: the entry point of a browser bundle (the `wasm32` target).

#![warn(missing_docs)]

pub mod common;

#[cfg(feature = "native")]
pub mod native;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

pub use common::*;
