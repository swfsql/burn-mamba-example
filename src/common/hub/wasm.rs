//! The hub client of the browser: `fetch` range requests, and a hub cache of
//! download chunks in IndexedDB.
//!
//! The client fetches a weights file of 500MB as ranges of
//! [DOWNLOAD_CHUNK_LEN] bytes. So the page can show the progress, continue
//! after a reload, and erase the file. The hub cache of the browser has this
//! layout:
//!
//! ```text
//! db "burn-mamba-example" v1
//!   store "download-chunks"
//!     key "huggingface/hub/models--EleutherAI--gpt-neox-20b/blobs/<etag>/<start>-<end>"
//!     value  Uint8Array
//! ```
//!
//! Each offset in a key has zeros in front, to the width of the file size. So
//! the lexicographic order of the keys in IndexedDB is the numeric order.

use super::{
    BlobHash, DownloadChunkKey, DownloadChunkPlan, Endpoint, FilePath, FileUrl, HubError,
    Metadata, Repo, RepoId, RepoType, RevisionHash, UrlTemplate, clean_etag,
    plan_download_chunks, size_from_content_range,
};
use indexed_db_futures::prelude::*;
use indexed_db_futures::{IdbDatabase, IdbQuerySource, idb_transaction::IdbTransaction};
use js_sys::Uint8Array;
use std::rc::Rc;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    DomException, Headers, IdbCursorDirection, IdbKeyRange, IdbTransactionMode, Request,
    RequestInit, Response,
};

/// The length of a download chunk: the bytes of one range request.
pub const DOWNLOAD_CHUNK_LEN: usize = 10 * 1024 * 1024;

const DB_NAME: &str = "burn-mamba-example";
const DB_VERSION: u32 = 1;
const KEY_ROOT: &str = "huggingface/hub";
const DOWNLOAD_CHUNK_STORE: &str = "download-chunks";

/// A hub client, with the open connection to IndexedDB.
#[derive(Clone)]
pub struct Api {
    endpoint: Endpoint,
    url_template: UrlTemplate,
    db: Rc<IdbDatabase>,
    download_chunk_len: usize,
}

impl Api {
    /// Opens the database of the hub cache. On the first use, it also creates
    /// the database.
    pub async fn new() -> Result<Self, HubError> {
        let mut request = IdbDatabase::open_u32(DB_NAME, DB_VERSION)
            .map_err(|e| dom_err("opening the hub cache database", e))?;
        request.set_on_upgrade_needed(Some(
            |event: &IdbVersionChangeEvent| -> Result<(), JsValue> {
                let existing: Vec<String> = event.db().object_store_names().collect();
                if !existing.iter().any(|name| name == DOWNLOAD_CHUNK_STORE) {
                    log::info!("creating the IndexedDB object store {DOWNLOAD_CHUNK_STORE}");
                    event.db().create_object_store(DOWNLOAD_CHUNK_STORE)?;
                }
                Ok(())
            },
        ));
        let db = request
            .await
            .map_err(|e| dom_err("opening the hub cache database", e))?;

        Ok(Self {
            endpoint: Endpoint::default(),
            url_template: UrlTemplate::default(),
            db: Rc::new(db),
            download_chunk_len: DOWNLOAD_CHUNK_LEN,
        })
    }

    /// A handle on one repository.
    pub fn repo(&self, repo: Repo) -> ApiRepo {
        ApiRepo {
            api: self.clone(),
            repo,
        }
    }

    /// A handle on one repository of the hub type "model", at its default
    /// revision.
    pub fn model_repo(&self, repo_id: RepoId) -> ApiRepo {
        self.repo(Repo::new(repo_id, RepoType::Model))
    }

    /// Resolves the etag and the size of a file, with a range request of 1
    /// byte.
    ///
    /// `fetch` follows the redirect of the hub to its CDN, and shows only the
    /// headers of the last response. So the commit (`x-repo-commit`) is
    /// usually absent here. The hub cache of the browser does not need it,
    /// because its keys use the etag.
    pub async fn metadata(&self, url: &FileUrl) -> Result<Metadata, HubError> {
        log::info!("requesting metadata for {}", url.0);
        let response = fetch(&url.0, Some((0, 0))).await?;
        let headers = response.headers();

        let etag = header(&headers, "x-linked-etag")
            .or_else(|| header(&headers, "etag"))
            .map(|etag| clean_etag(&etag))
            .ok_or_else(|| HubError::Header {
                name: "etag",
                url: url.0.clone(),
            })?;
        let size = header(&headers, "content-range")
            .as_deref()
            .and_then(size_from_content_range)
            .ok_or_else(|| HubError::Header {
                name: "content-range",
                url: url.0.clone(),
            })?;
        let commit_hash = header(&headers, "x-repo-commit").unwrap_or_default();

        Ok(Metadata {
            commit_hash: RevisionHash(commit_hash),
            etag: BlobHash(etag),
            size,
        })
    }

    /// Loads the cached download chunks, and joins them into the whole file.
    pub async fn load_bytes(
        &self,
        download_chunks: &[DownloadChunkKey],
    ) -> Result<Vec<u8>, HubError> {
        let total = download_chunks.last().map(|c| c.end).unwrap_or(0);
        let mut bytes = Vec::with_capacity(total);
        for download_chunk in download_chunks {
            let stored = self
                .get_download_chunk(download_chunk)
                .await?
                .ok_or_else(|| HubError::HubCache {
                    context: download_chunk.key.clone(),
                    message: "the download chunk is missing from the hub cache".into(),
                })?;
            bytes.extend_from_slice(&stored.to_vec());
        }
        Ok(bytes)
    }

    /// Erases the given download chunks from the hub cache.
    pub async fn delete_bytes(&self, download_chunks: &[DownloadChunkKey]) -> Result<(), HubError> {
        for download_chunk in download_chunks {
            let key = &download_chunk.key;
            let tx = self.transaction(IdbTransactionMode::Readwrite)?;
            let store = tx
                .object_store(DOWNLOAD_CHUNK_STORE)
                .map_err(|e| dom_err(key, e))?;
            store
                .delete_owned(key.as_str())
                .map_err(|e| dom_err(key, e))?
                .await
                .map_err(|e| dom_err(key, e))?;
            tx.await.into_result().map_err(|e| dom_err(key, e))?;
        }
        Ok(())
    }

    fn transaction(&self, mode: IdbTransactionMode) -> Result<IdbTransaction<'_>, HubError> {
        self.db
            .transaction_on_one_with_mode(DOWNLOAD_CHUNK_STORE, mode)
            .map_err(|e| dom_err(DOWNLOAD_CHUNK_STORE, e))
    }

    async fn get_download_chunk(
        &self,
        download_chunk: &DownloadChunkKey,
    ) -> Result<Option<Uint8Array>, HubError> {
        let key = &download_chunk.key;
        let tx = self.transaction(IdbTransactionMode::Readonly)?;
        let store = tx
            .object_store(DOWNLOAD_CHUNK_STORE)
            .map_err(|e| dom_err(key, e))?;
        let value = store
            .get_owned(key.as_str())
            .map_err(|e| dom_err(key, e))?
            .await
            .map_err(|e| dom_err(key, e))?;
        tx.await.into_result().map_err(|e| dom_err(key, e))?;
        Ok(value.map(Uint8Array::from))
    }

    async fn put_download_chunk(
        &self,
        download_chunk: &DownloadChunkKey,
        data: &Uint8Array,
    ) -> Result<(), HubError> {
        let key = &download_chunk.key;
        let tx = self.transaction(IdbTransactionMode::Readwrite)?;
        let store = tx
            .object_store(DOWNLOAD_CHUNK_STORE)
            .map_err(|e| dom_err(key, e))?;
        store
            .put_key_val_owned(key.as_str(), data)
            .map_err(|e| dom_err(key, e))?
            .await
            .map_err(|e| dom_err(key, e))?;
        tx.await.into_result().map_err(|e| dom_err(key, e))?;
        Ok(())
    }

    /// The keys already cached for one blob, in ascending offset order.
    async fn cached_download_chunks(
        &self,
        blob_prefix: &str,
        total_size: usize,
    ) -> Result<Vec<DownloadChunkKey>, HubError> {
        // Two empty ranges, at the offset 0 and at the end, bound the keys of
        // this blob. No key of another blob is between them.
        let first = DownloadChunkKey::new(blob_prefix, 0, 0, total_size);
        let last = DownloadChunkKey::new(blob_prefix, total_size, total_size, total_size);
        let range = IdbKeyRange::bound(
            &JsValue::from_str(&first.key),
            &JsValue::from_str(&last.key),
        )
        .map_err(|e| HubError::HubCache {
            context: blob_prefix.into(),
            message: format!("{e:?}"),
        })?;

        let tx = self.transaction(IdbTransactionMode::Readonly)?;
        let store = tx
            .object_store(DOWNLOAD_CHUNK_STORE)
            .map_err(|e| dom_err(blob_prefix, e))?;
        let cursor = store
            .open_key_cursor_with_range_and_direction_owned(range, IdbCursorDirection::Next)
            .map_err(|e| dom_err(blob_prefix, e))?
            .await
            .map_err(|e| dom_err(blob_prefix, e))?;
        let keys = match cursor {
            Some(cursor) => cursor
                .into_vec(0)
                .await
                .map_err(|e| dom_err(blob_prefix, e))?,
            None => vec![],
        };
        tx.await
            .into_result()
            .map_err(|e| dom_err(blob_prefix, e))?;

        Ok(keys
            .into_iter()
            .filter_map(|key| DownloadChunkKey::parse(key.as_string()?))
            .collect())
    }
}

/// A handle on one repository. It makes the URLs of its files, plans their
/// download chunks and fetches them.
#[derive(Clone)]
pub struct ApiRepo {
    api: Api,
    repo: Repo,
}

impl ApiRepo {
    /// The URL of `filename` in this repo.
    pub fn url(&self, filename: &FilePath) -> FileUrl {
        self.api.url_template.url(
            &self.api.endpoint,
            &self.repo,
            &self.repo.revision,
            filename,
        )
    }

    /// Plans the download chunks of a file. It reads the cached keys of the
    /// blob from IndexedDB, then gives them to [plan_download_chunks].
    pub async fn check(&self, metadata: &Metadata) -> Result<DownloadChunkPlan, HubError> {
        let blob_prefix = self.blob_prefix(&metadata.etag);
        let size = metadata.size;
        let cached = self.api.cached_download_chunks(&blob_prefix, size).await?;
        Ok(plan_download_chunks(
            &blob_prefix,
            size,
            self.api.download_chunk_len,
            cached,
        ))
    }

    /// Returns a whole file. It plans the download chunks, fetches the missing
    /// ones, and joins all of them. The caller cannot see the progress, so the
    /// browser page calls [Self::check] and [Self::fetch_download_chunk]
    /// itself.
    pub async fn get_bytes(&self, filename: &FilePath) -> Result<Vec<u8>, HubError> {
        let url = self.url(filename);
        let metadata = self.api.metadata(&url).await?;
        let plan = self.check(&metadata).await?;

        let mut keys = Vec::with_capacity(plan.len());
        for download_chunk in plan {
            let download_chunk = match download_chunk {
                Ok(cached) => cached,
                Err(missing) => {
                    self.fetch_download_chunk(&url, &missing).await?;
                    missing
                }
            };
            keys.push(download_chunk);
        }
        self.api.load_bytes(&keys).await
    }

    /// Fetches one download chunk of the plan, and puts it in the hub cache.
    pub async fn fetch_download_chunk(
        &self,
        url: &FileUrl,
        download_chunk: &DownloadChunkKey,
    ) -> Result<(), HubError> {
        log::info!("fetching download chunk {}", download_chunk.key);
        // HTTP ranges are inclusive on both ends.
        let response = fetch(&url.0, Some((download_chunk.start, download_chunk.end - 1))).await?;
        let buffer = JsFuture::from(response.array_buffer().map_err(|e| HubError::Request {
            url: url.0.clone(),
            message: format!("{e:?}"),
        })?)
        .await
        .map_err(|e| HubError::Request {
            url: url.0.clone(),
            message: format!("{e:?}"),
        })?;
        let data = Uint8Array::new(&buffer);

        if data.length() as usize != download_chunk.len() {
            return Err(HubError::ShortRead {
                expected: download_chunk.len(),
                got: data.length() as usize,
            });
        }
        self.api.put_download_chunk(download_chunk, &data).await
    }

    /// `huggingface/hub/models--…/blobs/<etag>`.
    fn blob_prefix(&self, etag: &BlobHash) -> String {
        format!("{KEY_ROOT}/{}/blobs/{}", self.repo.folder_name(), etag.0)
    }
}

/// A GET request. With `range`, it asks only for that inclusive byte range.
async fn fetch(url: &str, range: Option<(usize, usize)>) -> Result<Response, HubError> {
    let request_err = |e: JsValue| HubError::Request {
        url: url.to_string(),
        message: format!("{e:?}"),
    };

    let init = RequestInit::new();
    init.set_method("GET");
    if let Some((start, end)) = range {
        let headers = Headers::new().map_err(request_err)?;
        headers
            .set("Range", &format!("bytes={start}-{end}"))
            .map_err(request_err)?;
        init.set_headers(&headers);
    }

    let request = Request::new_with_str_and_init(url, &init).map_err(request_err)?;
    let window = web_sys::window().ok_or_else(|| HubError::Request {
        url: url.to_string(),
        message: "no browser window".into(),
    })?;
    let response = JsFuture::from(window.fetch_with_request(&request))
        .await
        .map_err(request_err)?;
    let response: Response = response.dyn_into().map_err(request_err)?;

    if !response.ok() {
        return Err(HubError::Request {
            url: url.to_string(),
            message: format!("status {}", response.status()),
        });
    }
    Ok(response)
}

fn header(headers: &Headers, name: &str) -> Option<String> {
    headers.get(name).ok().flatten()
}

fn dom_err(context: &str, error: DomException) -> HubError {
    HubError::HubCache {
        context: context.into(),
        message: format!("{}: {}", error.name(), error.message()),
    }
}
