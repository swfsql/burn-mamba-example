use super::Msg;
use super::model::AssetKind;
pub use super::model::{self, Connection, Model};
use crate::hub::wasm::{Api, DownloadChunkKey};
use yew::prelude::*;

const TICK_MILLIS: u32 = 1;

impl model::Model {
    pub fn update(&mut self, ctx: &Context<Self>, msg: Msg) -> bool {
        match msg {
            // Msg::Todo => {
            //     todo!()
            // }

            // fetching, loading, building
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
                log::error!("failed to connect to the api: {err}");
                true
            }
            Msg::StartDisconnectApi => {
                log::error!("Msg::StartDisconnectApi not yet implemented");
                false
            }
            Msg::FinishDisconnectApi => {
                log::error!("Msg::FinishDisconnectApi not yet implemented");
                false
            }
            Msg::FailDisconnectApi => {
                log::error!("Msg::FailDisconnectApi not yet implemented");
                false
            }
            Msg::StartAssetCheck(kind) => {
                let asset = self.select(&kind);
                let api = self.hub_api.as_connected().unwrap().clone();
                // assert that a lot of data won't be cloned
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
                log::error!("failed to fetch download chunk {i} for {kind:?}; err: {err}");
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
                log::error!("Msg::StartAssetUpload not yet implemented");
                false
            }
            Msg::FinishAssetUpload(_kind) => {
                log::error!("Msg::FinishAssetUpload not yet implemented");
                false
            }
            Msg::FailAssetUpload => {
                log::error!("Msg::FailAssetUpload not yet implemented");
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
                log::error!("failed to load data for {kind:?}; err: {err:?}");
                let asset = self.select_mut(&kind);
                asset.load.is_busy = false;
                true
            }
            Msg::AssetUnload(kind) => {
                // stop any in-progress generation
                if self.is_generating {
                    ctx.link().send_message_batch(vec![
                        Msg::StopGeneration,
                        Msg::ResetCaches,
                        Msg::AssetUnload(kind),
                    ]);
                    return false;
                }

                // clear generation-related memory
                self.tokens.clear();
                self.tokens.shrink_to_fit();
                self.output.shrink_to_fit();

                // clear the built tokenizer and network
                if self.wrapper.is_some() {
                    self.wrapper = None;
                    self.tokenizer.load.is_done = false;
                    self.weights.load.is_done = false;
                }
                // in case the wrapper wasn't assembled yet,
                // clear build and data-load memory
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

                // set the download chunks to uncached
                for download_chunk_key in asset.hub_cache.fetching.download_chunk_list.iter_mut() {
                    let owned = download_chunk_key.clone().unwrap();
                    *download_chunk_key = Err(owned);
                }

                true
            }
            Msg::FailAssetErase(kind, err) => {
                log::error!("failed to erase data for {kind:?}; err: {err:?}");
                let asset = self.select_mut(&kind);
                asset.hub_cache.is_busy = false;
                asset.hub_cache.is_done = false;
                // re-check if the data is cached or not
                //
                // note: if partial data has been erased, then the button will switch to
                // the "click to fetch" option.
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
                // consume the built tokenizer and network if they are both ready
                if self.wrapper_builder.is_ready() {
                    let builder = std::mem::take(&mut self.wrapper_builder);
                    let wrapper = builder.build();
                    self.wrapper = Some(wrapper);
                }
                true
            }

            // user input
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

            // inference
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
                // TODO: reset preprocessor
                self.tokens = tokens;
                self.eos_token = eos_token;

                // gets first token (as if it were an implicit output)
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

                // if the token has some valid representation, print it
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
                // for state in wrapper.states.iter_mut() {
                //     ResetParams::<f32, Cpu>::try_reset_params(state).unwrap();
                // }
                wrapper.processor =
                    crate::LogitsProcessorWrapper::new(299792458, None, None, 1.1, 1024);
                self.is_reset = true;
                self.is_input_dirty = false;
                true
            }
        }
    }
}
