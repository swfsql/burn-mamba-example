//! Browser hub client: `fetch` range requests, with each download chunk cached
//! in IndexedDB.
//!
//! A 500MB checkpoint is fetched as [DOWNLOAD_CHUNK_LEN]-sized ranges so the page
//! can show progress, resume after a reload, and erase the file again. The hub
//! cache of the browser has this layout:
//!
//! ```text
//! db "burn-mamba-example" v1
//!   store "download-chunks"
//!     key "huggingface/hub/models--EleutherAI--gpt-neox-20b/blobs/<etag>/<start>-<end>"
//!     value  Uint8Array
//! ```
//!
//! The offsets in a key are zero-padded to the width of the file size so that the
//! lexicographic order IndexedDB uses for keys matches the numeric one.

use super::{
    BlobHash, Endpoint, FilePath, FileUrl, HubError, Metadata, Repo, RepoId, RepoType,
    RevisionHash, UrlTemplate, clean_etag, size_from_content_range,
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

/// How much of a file is fetched (and cached) per request.
pub const DOWNLOAD_CHUNK_LEN: usize = 10 * 1024 * 1024;

const DB_NAME: &str = "burn-mamba-example";
const DB_VERSION: u32 = 1;
const KEY_ROOT: &str = "huggingface/hub";
const DOWNLOAD_CHUNK_STORE: &str = "download-chunks";

/// One cached range of one blob.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DownloadChunkKey {
    /// The IndexedDB key.
    pub key: String,
    /// Offset of the first byte, within the whole file.
    pub start: usize,
    /// Offset just past the last byte.
    pub end: usize,
}

impl DownloadChunkKey {
    fn new(blob_prefix: &str, start: usize, end: usize, total_size: usize) -> Self {
        let width = total_size.checked_ilog10().unwrap_or(0) as usize + 1;
        Self {
            key: format!("{blob_prefix}/{start:0width$}-{end:0width$}"),
            start,
            end,
        }
    }

    /// Parses back a key listed by IndexedDB; `None` if it is not one of ours.
    fn parse(key: String) -> Option<Self> {
        let (_prefix, offsets) = key.rsplit_once('/')?;
        let (start, end) = offsets.split_once('-')?;
        let start = start.parse().ok()?;
        let end = end.parse().ok()?;
        Some(Self { key, start, end })
    }

    /// Number of bytes this download chunk holds.
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The download chunks a file is made of: [Ok] when already cached, [Err] when
/// still to fetch.
pub type DownloadChunkList = Vec<Result<DownloadChunkKey, DownloadChunkKey>>;

/// A hub client holding the open IndexedDB connection.
#[derive(Clone)]
pub struct Api {
    endpoint: Endpoint,
    url_template: UrlTemplate,
    db: Rc<IdbDatabase>,
    download_chunk_len: usize,
}

impl Api {
    /// Opens (and, on first use, creates) the hub cache database.
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

    /// Resolves a file's etag and size, with a 1-byte range request.
    ///
    /// `fetch` transparently follows the hub's redirect to its CDN and only
    /// exposes the final response's headers, so `x-repo-commit` is usually lost
    /// here — it is not needed, since the hub cache of the browser is keyed by
    /// etag.
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

    /// Concatenates the cached download chunks back into the whole file.
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
        // Two empty ranges — one at offset 0, one at the end — bracket every real
        // download chunk key of this blob and nothing else.
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
            // A zero-length entry would stall the walk in `check`.
            .filter(|download_chunk| !download_chunk.is_empty())
            .collect())
    }
}

/// A repository handle: builds URLs, plans download chunks and fetches them.
#[derive(Clone)]
pub struct ApiRepo {
    api: Api,
    repo: Repo,
}

impl ApiRepo {
    /// The download URL of `filename` in this repo.
    pub fn url(&self, filename: &FilePath) -> FileUrl {
        self.api.url_template.url(
            &self.api.endpoint,
            &self.repo,
            &self.repo.revision,
            filename,
        )
    }

    /// Plans the download chunk list for a file: which ranges are cached and
    /// which are not.
    ///
    /// Cached download chunks are reused only when their start offset is exactly
    /// where the walk currently is, so a hub cache written with a different
    /// download chunk length still yields a valid (if partly redundant) plan.
    pub async fn check(&self, metadata: &Metadata) -> Result<DownloadChunkList, HubError> {
        let blob_prefix = self.blob_prefix(&metadata.etag);
        let size = metadata.size;
        let cached = self.api.cached_download_chunks(&blob_prefix, size).await?;

        let mut download_chunks = DownloadChunkList::new();
        let mut cached_iter = cached.into_iter().peekable();
        let mut offset = 0;
        while offset < size {
            // Drop cached download chunks starting before where we are (a leftover
            // from a previous, differently-sized plan).
            while cached_iter.peek().is_some_and(|c| c.start < offset) {
                cached_iter.next();
            }
            if cached_iter.peek().is_some_and(|c| c.start == offset) {
                let cached = cached_iter.next().expect("just peeked");
                offset = cached.end;
                download_chunks.push(Ok(cached));
                continue;
            }
            // Stop the download chunk to fetch right where the next cached one
            // begins; every remaining cached one now starts strictly after
            // `offset`.
            let limit = cached_iter.peek().map_or(size, |c| c.start.min(size));
            let end = (offset + self.api.download_chunk_len).min(limit);
            download_chunks.push(Err(DownloadChunkKey::new(&blob_prefix, offset, end, size)));
            offset = end;
        }

        Ok(download_chunks)
    }

    /// Fetches a whole file: plans its download chunks, fetches the missing ones
    /// and concatenates everything. The intermediate progress is not observable,
    /// so the browser page drives [Self::check] / [Self::fetch_download_chunk]
    /// itself.
    pub async fn get_bytes(&self, filename: &FilePath) -> Result<Vec<u8>, HubError> {
        let url = self.url(filename);
        let metadata = self.api.metadata(&url).await?;
        let download_chunks = self.check(&metadata).await?;

        let mut keys = Vec::with_capacity(download_chunks.len());
        for download_chunk in download_chunks {
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

    /// Fetches one planned download chunk and stores it.
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

/// A GET, optionally restricted to an inclusive byte range.
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
