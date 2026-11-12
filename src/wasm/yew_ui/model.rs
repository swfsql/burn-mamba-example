//! The state of the browser page: the [Model]. It holds the [Assets] of the
//! checkpoint, the connection of the hub client and the state of the
//! generation.

use crate::common::assets::{Assets, Connection};
use crate::hub::wasm::Api;
use crate::{CheckpointSpec, hf};
use burn::prelude::*;

/// The state of the browser page: its assets, its run wrapper and its
/// generation.
pub struct Model {
    // The device and the checkpoint.
    /// The device of the network. [Model::default] sets its f32 and i32
    /// defaults.
    pub device: Device,
    /// The checkpoint that the asset cards fetch and the generation runs.
    pub spec: &'static CheckpointSpec,

    // The hub client and the assets.
    /// The hub client. It checks, fetches, loads and erases the assets.
    pub hub_api: Connection<Api>,
    /// The two assets of [Self::spec], the wrapper builder and the run
    /// wrapper.
    pub assets: Assets,

    // The generation.
    /// The text of the input box.
    pub input: String,
    /// Whether the generation can differ from the input box, after an edit of
    /// the input. No code sets it to `true`.
    pub is_input_dirty: bool,
    /// Whether the caches are zero. Only then can the generation start.
    pub is_reset: bool,
    /// Whether the generation runs.
    pub is_generating: bool,
    /// The interval that sends [super::Msg::StepGeneration] after a start or a
    /// resume of the generation. [super::Msg::StopGeneration] drops it, and
    /// the drop stops the interval. At the eos token, the interval continues.
    pub generation_callback_interval: Option<gloo_timers::callback::Interval>,
    /// The position of the next step in [Self::tokens].
    pub step: usize,
    /// The token ids of the prompt, then the generated token ids.
    pub tokens: Vec<usize>,
    /// The text so far: the prompt, then the generated text.
    pub output: String,
    /// The token the network uses to signal the end of the generation.
    pub eos_token: usize,
}

impl Default for Model {
    fn default() -> Self {
        let mut device = Device::default();
        {
            device
                .configure((crate::PRECISION_FLOAT_D_TYPE, crate::PRECISION_INT_D_TYPE))
                .expect("failed to set the f32 and i32 defaults of the device");
        }

        // A bundle can hold several checkpoints, and the page runs one of
        // them: the preferred checkpoint.
        let spec = hf::preferred().expect("no checkpoint feature is enabled in this build");

        Self {
            // The device and the checkpoint.
            device,
            spec,
            // The hub client and the assets.
            hub_api: Connection::Disconnected,
            assets: Assets::new(spec),

            // The generation.
            input: "Mamba is the".into(),
            is_input_dirty: false,
            is_reset: true,
            is_generating: false,
            generation_callback_interval: None,
            step: 0,
            tokens: vec![],
            output: "".into(),
            eos_token: 0,
        }
    }
}
