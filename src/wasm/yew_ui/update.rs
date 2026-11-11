//! The reducer of the browser page: [Model::update] handles each [Msg].

use super::Msg;
use super::model::AssetKind;
pub use super::model::{self, Connection, Model};
use crate::hub::wasm::{Api, DownloadChunkKey};
use yew::prelude::*;

/// The interval between two steps of the generation, in milliseconds.
const TICK_MILLIS: u32 = 1;

impl model::Model {
    /// Handles `msg`: it changes the state of the page, and can send more
    /// messages. Returns `true` when the page must render again.
    pub fn update(&mut self, ctx: &Context<Self>, msg: Msg) -> bool {
        match msg {
            // The hub client, and the assets: check, fetch, load, build.
            Msg::StartConnectApi => {
                assert!(self.hub_api.is_exactly_disconnected());
                self.hub_api = Connection::Connecting;
                ctx.link().send_future(async {
                    match Api::new().await {
                        Ok(ok) => Msg::FinishConnectApi(ok),
                        Err(err) => Msg::FailConnectApi(err),
                    }
                });
                true
            }
            Msg::FinishConnectApi(api) => {
                self.hub_api = Connection::Connected(api);
                if self.tokenizer.hub_cache.is_checking {
                    ctx.link()
                        .send_message(Msg::StartAssetCheck(AssetKind::Tokenizer));
                }
                if self.weights.hub_cache.is_checking {
                    ctx.link()
                        .send_message(Msg::StartAssetCheck(AssetKind::Weights));
                }
                true
            }
            Msg::FailConnectApi(err) => {
                self.hub_api = Connection::Disconnected;
                log::error!("failed to connect the hub client: {err}");
                true
            }
            Msg::StartDisconnectApi => {
                log::error!("Msg::StartDisconnectApi is not implemented");
                false
            }
            Msg::FinishDisconnectApi => {
                log::error!("Msg::FinishDisconnectApi is not implemented");
                false
            }
            Msg::FailDisconnectApi => {
                log::error!("Msg::FailDisconnectApi is not implemented");
                false
            }
            Msg::StartAssetCheck(kind) => {
                let asset = self.select(&kind);
                let api = self.hub_api.as_connected().unwrap().clone();
                // The asset holds no loaded bytes, so its clone is small.
                assert!(asset.load.data.is_empty());
                let asset = asset.clone();
                ctx.link().send_future(async move {
                    let metadata = asset.config.metadata(&api).await.unwrap();
                    let download_chunk_list = asset.config.check(&api, &metadata).await.unwrap();
                    Msg::FinishAssetCheck(kind, metadata, download_chunk_list)
                });
                false
            }
            Msg::FinishAssetCheck(kind, metadata, download_chunk_list) => {
                let asset = self.select_mut(&kind);
                asset.hub_cache.is_checking = false;
                asset.load.is_checking = false;
                asset.hub_cache.fetching.metadata = Some(metadata);
                asset.hub_cache.is_done = download_chunk_list.iter().all(|c| c.is_ok());
                asset.hub_cache.fetching.download_chunk_list = download_chunk_list;
                true
            }
            Msg::FailAssetCheck => {
                todo!()
            }
            Msg::StartAssetFetch(kind) => {
                let api = self.hub_api.as_connected().unwrap().clone();
                let asset = self.select_mut(&kind);
                asset.hub_cache.is_busy = true;
                asset.hub_cache.fetching.current_download_chunk = 0;

                let download_chunks = asset.hub_cache.fetching.download_chunk_list.clone();
                let link = ctx.link().clone();
                let api_repo = asset.config.api_repo(&api);
                let url = asset.config.file_url();
                ctx.link().send_future(async move {
                    for (i, download_chunk) in download_chunks.into_iter().enumerate() {
                        if let Err(download_chunk) = download_chunk {
                            match api_repo.fetch_download_chunk(&url, &download_chunk).await {
                                Ok(()) => {}
                                Err(err) => {
                                    return Msg::FailAssetFetchSingle(kind, i, err);
                                }
                            }
                        }
                        link.send_message(Msg::FinishAssetFetchSingle(kind, i));
                    }
                    Msg::FinishAssetFetch(kind)
                });
                true
            }
            Msg::FinishAssetFetchSingle(kind, i) => {
                let asset = self.select_mut(&kind);
                asset.hub_cache.fetching.current_download_chunk += 1;
                let item = &mut asset.hub_cache.fetching.download_chunk_list[i];
                if item.is_err() {
                    *item = Ok(item.clone().unwrap_err());
                }
                true
            }
            Msg::FailAssetFetchSingle(kind, i, err) => {
                log::error!("failed to fetch the download chunk {i} of the asset {kind:?}: {err}");
                let asset = self.select_mut(&kind);
                asset.hub_cache.is_busy = false;
                asset.hub_cache.is_done = false;
                true
            }
            Msg::FinishAssetFetch(kind) => {
                let asset = self.select_mut(&kind);
                asset.hub_cache.is_busy = false;
                asset.hub_cache.is_done = true;
                true
            }
            Msg::StartAssetUpload(_kind) => {
                log::error!("Msg::StartAssetUpload is not implemented");
                false
            }
            Msg::FinishAssetUpload(_kind) => {
                log::error!("Msg::FinishAssetUpload is not implemented");
                false
            }
            Msg::FailAssetUpload => {
                log::error!("Msg::FailAssetUpload is not implemented");
                false
            }
            Msg::StartAssetLoad(kind) => {
                let api = self.hub_api.as_connected().unwrap().clone();
                let asset = self.select_mut(&kind);
                asset.load.is_busy = true;
                let download_chunk_keys: Vec<DownloadChunkKey> = asset
                    .hub_cache
                    .fetching
                    .download_chunk_list
                    .iter()
                    .cloned()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                ctx.link().send_future(async move {
                    match api.load_bytes(&download_chunk_keys).await {
                        Ok(ok) => Msg::FinishAssetLoad(kind, ok),
                        Err(err) => Msg::FailAssetLoad(kind, err),
                    }
                });
                true
            }
            Msg::FinishAssetLoad(kind, data) => {
                let asset = self.select_mut(&kind);
                asset.load.data = data;
                ctx.link().send_message(Msg::StartAssetBuild(kind));
                false
            }
            Msg::FailAssetLoad(kind, err) => {
                log::error!("failed to load the asset {kind:?}: {err:?}");
                let asset = self.select_mut(&kind);
                asset.load.is_busy = false;
                true
            }
            Msg::AssetUnload(kind) => {
                // Stop a running generation first, then unload.
                if self.is_generating {
                    ctx.link().send_message_batch(vec![
                        Msg::StopGeneration,
                        Msg::ResetCaches,
                        Msg::AssetUnload(kind),
                    ]);
                    return false;
                }

                // Free the memory of the generation.
                self.tokens.clear();
                self.tokens.shrink_to_fit();
                self.output.shrink_to_fit();

                // Drop the run wrapper: the built tokenizer and network.
                if self.wrapper.is_some() {
                    self.wrapper = None;
                    self.tokenizer.load.is_done = false;
                    self.weights.load.is_done = false;
                }
                // With no run wrapper, free the built part and the loaded
                // bytes of this asset.
                else if let AssetKind::Tokenizer = kind {
                    self.wrapper_builder.tokenizer = None;
                    self.tokenizer.load.data.clear();
                    self.tokenizer.load.data.shrink_to_fit();
                    self.tokenizer.load.is_done = false;
                } else if let AssetKind::Weights = kind {
                    self.wrapper_builder.mamba = None;
                    self.weights.load.data.clear();
                    self.weights.load.data.shrink_to_fit();
                    self.weights.load.is_done = false;
                };

                true
            }
            Msg::StartAssetErase(kind) => {
                let api = self.hub_api.as_connected().unwrap().clone();
                let asset = self.select_mut(&kind);
                assert!(!asset.load.is_busy);
                asset.hub_cache.is_busy = true;
                let download_chunk_keys = asset
                    .hub_cache
                    .fetching
                    .download_chunk_list
                    .iter()
                    .cloned()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();

                ctx.link().send_future(async move {
                    match api.delete_bytes(&download_chunk_keys).await {
                        Ok(()) => Msg::FinishAssetErase(kind),
                        Err(err) => Msg::FailAssetErase(kind, err),
                    }
                });

                asset.hub_cache.is_busy = true;
                true
            }
            Msg::FinishAssetErase(kind) => {
                let asset = self.select_mut(&kind);
                asset.hub_cache.is_busy = false;
                asset.hub_cache.is_done = false;

                // Mark each download chunk as not cached.
                for download_chunk_key in asset.hub_cache.fetching.download_chunk_list.iter_mut() {
                    let owned = download_chunk_key.clone().unwrap();
                    *download_chunk_key = Err(owned);
                }

                true
            }
            Msg::FailAssetErase(kind, err) => {
                log::error!("failed to erase the asset {kind:?}: {err:?}");
                let asset = self.select_mut(&kind);
                asset.hub_cache.is_busy = false;
                asset.hub_cache.is_done = false;
                // Check the asset again. When the erase removed a part of the
                // asset, the button then offers a fetch.
                asset.hub_cache.is_checking = true;
                ctx.link().send_message(Msg::StartAssetCheck(kind));
                true
            }
            Msg::StartAssetBuild(kind) => {
                let asset = self.select_mut(&kind);
                let data = std::mem::take(&mut asset.load.data);
                self.wrapper_builder.with(&kind, data, &self.device);
                ctx.link().send_message(Msg::FinishAssetBuild(kind));
                false
            }
            Msg::FinishAssetBuild(kind) => {
                let asset = self.select_mut(&kind);
                asset.load.is_busy = false;
                asset.load.is_done = true;
                ctx.link().send_message(Msg::TryAssembleWrapper);
                true
            }
            Msg::FailAssetBuild => {
                todo!()
            }
            Msg::TryAssembleWrapper => {
                // When the builder holds the tokenizer and the network, move
                // them into the run wrapper.
                if self.wrapper_builder.is_ready() {
                    let builder = std::mem::take(&mut self.wrapper_builder);
                    let wrapper = builder.build();
                    self.wrapper = Some(wrapper);
                }
                true
            }

            // The input.
            Msg::InputUpdate(new) => match (self.input == new, self.is_reset) {
                (_same @ true, _reset) => false,
                (_same @ false, _reset @ false) => {
                    self.input = new;
                    true
                }
                (_same @ false, _reset @ true) => {
                    self.input = new;
                    false
                }
            },

            // The generation.
            Msg::StartGeneration => {
                assert!(self.is_reset);
                assert!(!self.is_input_dirty);
                assert!(!self.is_generating);
                self.is_generating = true;
                self.is_reset = false;
                self.output.clear();
                self.step = 0;
                let wrapper = self.wrapper.as_mut().unwrap();
                let (tokens, eos_token) = wrapper.run_wrapper.reset_prompt(&self.input).unwrap();
                // The sampler is new: `Wrapper::new` or `Msg::ResetCaches` made
                // it.
                self.tokens = tokens;
                self.eos_token = eos_token;

                // No step gives the first token as an output, so add it here.
                if let Some(t) = self.tokens.first() {
                    if let Some(t) = wrapper.run_wrapper.tokenizer.next_token(*t as u32) {
                        self.output += &t;
                    }
                }

                let link = ctx.link().clone();
                let interval = gloo_timers::callback::Interval::new(TICK_MILLIS, move || {
                    link.send_message(Msg::StepGeneration)
                });
                self.generation_callback_interval = Some(interval);

                true
            }

            Msg::StepGeneration => {
                if !self.is_generating {
                    return true;
                }
                let wrapper = self.wrapper.as_mut().unwrap();
                let run_wrapper = &mut wrapper.run_wrapper;
                let (next_logits, next_caches) = run_wrapper
                    .step(self.tokens[self.step] as usize, Some(wrapper.caches.clone()))
                    .unwrap();
                wrapper.caches = next_caches;
                let next_token = wrapper
                    .processor
                    .add_logits(self.step, &mut self.tokens, next_logits)
                    .unwrap();

                // Add the new text, if the token completes some.
                if let Some(t) = run_wrapper.tokenizer.next_token(next_token as u32) {
                    self.output += &t;
                }
                self.step += 1;

                if next_token == self.eos_token {
                    self.is_generating = false;

                    if let Some(rest) = run_wrapper.tokenizer.decode_rest() {
                        self.output += &rest;
                    }

                    true
                } else {
                    true
                }
            }
            Msg::StopGeneration => {
                self.is_generating = false;
                self.generation_callback_interval = None;
                true
            }
            Msg::ResumeGeneration => {
                assert!(!self.is_input_dirty);
                self.is_generating = true;
                let link = ctx.link().clone();
                let interval = gloo_timers::callback::Interval::new(TICK_MILLIS, move || {
                    link.send_message(Msg::StepGeneration)
                });
                self.generation_callback_interval = Some(interval);
                true
            }
            Msg::ResetCaches => {
                assert!(!self.is_generating);
                let wrapper = self.wrapper.as_mut().unwrap();
                wrapper.caches = wrapper.run_wrapper.empty_caches(1).unwrap();
                wrapper.processor =
                    crate::LogitsProcessorWrapper::new(299792458, None, None, 1.1, 1024);
                self.is_reset = true;
                self.is_input_dirty = false;
                true
            }
        }
    }
}
