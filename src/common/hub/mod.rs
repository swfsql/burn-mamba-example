//! A small client of the HuggingFace hub. The app does not use the `hf-hub`
//! crate (ADR-0001).
//!
//! The client does only what the app needs. It resolves the metadata of a file
//! (commit, etag, size), and it fetches the file. Natively, `sync`
//! keeps the hub cache on disk. In the browser, `wasm` keeps the hub cache in
//! IndexedDB, as download chunks.
//!
//! The two hub clients share the types below. The download chunk types are
//! shared types too: [DownloadChunkKey], [DownloadChunkPlan] and
//! [plan_download_chunks]. They compile on each target, so a native test runs
//! them. Only the hub client of the browser uses them.

#[cfg(not(target_arch = "wasm32"))]
pub mod sync;
#[cfg(target_arch = "wasm32")]
pub mod wasm;

use std::fmt;

/// An error of a request to the hub, or of an access to the hub cache.
#[derive(Debug)]
pub enum HubError {
    /// The request failed (transport, DNS, CORS, a status that is not a
    /// success…).
    Request {
        /// The URL of the request.
        url: String,
        /// The error text of the transport, or the status.
        message: String,
    },
    /// The response has no header with this name, or its value is not usable.
    Header {
        /// The name of the header.
        name: &'static str,
        /// The URL of the request.
        url: String,
    },
    /// A read or a write of the hub cache (the filesystem or IndexedDB)
    /// failed.
    HubCache {
        /// The target of the access: a path, an IndexedDB key, or a short
        /// description of the operation.
        context: String,
        /// The error text of the filesystem or of IndexedDB.
        message: String,
    },
    /// A response gave a different number of bytes than the request asked
    /// for.
    ShortRead {
        /// The number of bytes that the request asked for.
        expected: usize,
        /// The number of bytes in the response.
        got: usize,
    },
}

impl fmt::Display for HubError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Request { url, message } => write!(f, "request to {url} failed: {message}"),
            Self::Header { name, url } => {
                write!(f, "response from {url} is missing a usable {name} header")
            }
            Self::HubCache { context, message } => {
                write!(f, "hub cache error ({context}): {message}")
            }
            Self::ShortRead { expected, got } => {
                write!(f, "expected {expected} bytes, received {got}")
            }
        }
    }
}

impl std::error::Error for HubError {}

/// A hub base URL. Eg. `"https://huggingface.co"`.
#[derive(Clone, Debug, PartialEq)]
pub struct Endpoint(pub String);

impl Default for Endpoint {
    fn default() -> Self {
        Self("https://huggingface.co".into())
    }
}

/// A repository name. Eg. `"EleutherAI/gpt-neox-20b"`.
#[derive(Clone, Debug, PartialEq)]
pub struct RepoId(pub String);

/// The kind of a repository of the hub.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RepoType {
    /// A repository of the hub type "model". Each checkpoint is in one.
    Model,
    /// A repository of the hub type "dataset".
    Dataset,
    /// A repository of the hub type "space".
    Space,
}

impl RepoType {
    /// The path prefix the hub uses for this kind of repo.
    fn prefix(&self) -> &'static str {
        match self {
            RepoType::Model => "models",
            RepoType::Dataset => "datasets",
            RepoType::Space => "spaces",
        }
    }
}

/// A branch, tag or ref. Eg. `"main"`, `"refs/pr/1"`.
#[derive(Clone, Debug, PartialEq)]
pub struct RevisionPath(pub String);

impl Default for RevisionPath {
    fn default() -> Self {
        Self("main".into())
    }
}

impl RevisionPath {
    /// Percent-escaped for use inside a URL. Eg. `"refs%2Fpr%2F1"`.
    pub fn url(&self) -> String {
        self.0.replace('/', "%2F")
    }
}

/// A resolved commit hash.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RevisionHash(pub String);

/// A file within a repo. Eg. `"tokenizer.json"`.
#[derive(Clone, Debug, PartialEq)]
pub struct FilePath(pub String);

/// The content hash (etag) of a file's blob.
#[derive(Clone, Debug, PartialEq)]
pub struct BlobHash(pub String);

/// The URL of one file of the hub, at one revision.
#[derive(Clone, Debug, PartialEq)]
pub struct FileUrl(pub String);

/// A repository at a given revision.
#[derive(Clone, Debug, PartialEq)]
pub struct Repo {
    /// The name of the repository.
    pub repo_id: RepoId,
    /// The kind of the repository.
    pub repo_type: RepoType,
    /// The branch, tag or ref.
    pub revision: RevisionPath,
}

impl Repo {
    /// At the default branch (`"main"`).
    pub fn new(repo_id: RepoId, repo_type: RepoType) -> Self {
        Self::with_revision(repo_id, repo_type, RevisionPath::default())
    }

    /// At the branch, tag or ref `revision`.
    pub fn with_revision(repo_id: RepoId, repo_type: RepoType, revision: RevisionPath) -> Self {
        Self {
            repo_id,
            repo_type,
            revision,
        }
    }

    /// The repo's URL fragment: `"{repo_id}"` for models, `"datasets/{repo_id}"`…
    pub fn url(&self) -> String {
        match self.repo_type {
            RepoType::Model => self.repo_id.0.clone(),
            other => format!("{}/{}", other.prefix(), self.repo_id.0),
        }
    }

    /// The directory name of the repo in the hub cache. Eg.
    /// `"models--EleutherAI--gpt-neox-20b"`.
    pub fn folder_name(&self) -> String {
        format!("{}--{}", self.repo_type.prefix(), self.repo_id.0).replace('/', "--")
    }
}

/// The pattern that makes a [FileUrl] from an [Endpoint], a [Repo] and a
/// [FilePath].
#[derive(Clone, Debug, PartialEq)]
pub struct UrlTemplate(pub String);

impl Default for UrlTemplate {
    /// `"{endpoint}/{repo_id}/resolve/{revision}/{filename}"`.
    fn default() -> Self {
        Self("{endpoint}/{repo_id}/resolve/{revision}/{filename}".into())
    }
}

impl UrlTemplate {
    /// The URL of `filename` in `repo` at `revision`.
    pub fn url(
        &self,
        endpoint: &Endpoint,
        repo: &Repo,
        revision: &RevisionPath,
        filename: &FilePath,
    ) -> FileUrl {
        FileUrl(
            self.0
                .replace("{endpoint}", &endpoint.0)
                .replace("{repo_id}", &repo.url())
                .replace("{revision}", &revision.url())
                .replace("{filename}", &filename.0),
        )
    }
}

/// The data that the hub gives about a file before a fetch.
#[derive(Clone, Debug, PartialEq)]
pub struct Metadata {
    /// The commit of the revision at the time of the request. It can be empty
    /// in the browser, because `fetch` does not show the redirect that has the
    /// `x-repo-commit` header.
    pub commit_hash: RevisionHash,
    /// The content hash of the blob. It is the key of the blob in the hub
    /// cache.
    pub etag: BlobHash,
    /// The file size, in bytes.
    pub size: usize,
}

/// Parses the total size out of a `content-range: bytes 0-0/12345` header.
fn size_from_content_range(value: &str) -> Option<usize> {
    value.rsplit('/').next()?.trim().parse().ok()
}

/// Removes the quotes that the hub puts around an etag. They are not part of
/// the hash.
fn clean_etag(value: &str) -> String {
    value.replace('"', "")
}

/// The key and the byte range of one download chunk of one blob.
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
    /// The key of the bytes `start..end` of the blob `blob_prefix`, for a file
    /// of `total_size` bytes. Each offset has zeros in front, to the width of
    /// `total_size`.
    pub fn new(blob_prefix: &str, start: usize, end: usize, total_size: usize) -> Self {
        let width = total_size.checked_ilog10().unwrap_or(0) as usize + 1;
        Self {
            key: format!("{blob_prefix}/{start:0width$}-{end:0width$}"),
            start,
            end,
        }
    }

    /// Parses a key from IndexedDB. Returns `None` for a key of another
    /// format.
    pub fn parse(key: String) -> Option<Self> {
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

    /// Whether this download chunk holds no byte.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The plan of a file: its download chunks, in order. An [Ok] entry is in the
/// hub cache. An [Err] entry is not, so the client must fetch it.
pub type DownloadChunkPlan = Vec<Result<DownloadChunkKey, DownloadChunkKey>>;

/// Plans the download chunks of a file of `size` bytes: the ranges in the hub
/// cache, and the ranges to fetch.
///
/// `cached` holds the cached keys of the blob `blob_prefix`, in ascending
/// offset order. The function ignores a cached key of length zero. A range to
/// fetch has `download_chunk_len` bytes or less.
///
/// The plan uses a cached download chunk only when it starts at the offset of
/// the walk. So a hub cache with another download chunk length still gives a
/// valid plan, but the plan can fetch some bytes again.
pub fn plan_download_chunks(
    blob_prefix: &str,
    size: usize,
    download_chunk_len: usize,
    cached: Vec<DownloadChunkKey>,
) -> DownloadChunkPlan {
    // A key of length zero holds no byte. The walk would add it as an entry
    // of the plan.
    let mut cached_iter = cached
        .into_iter()
        .filter(|download_chunk| !download_chunk.is_empty())
        .peekable();
    let mut plan = DownloadChunkPlan::new();
    let mut offset = 0;
    while offset < size {
        // Skip the cached download chunks that start before the offset.
        // They come from a plan with another download chunk length.
        while cached_iter.peek().is_some_and(|c| c.start < offset) {
            cached_iter.next();
        }
        if cached_iter.peek().is_some_and(|c| c.start == offset) {
            let cached = cached_iter.next().expect("just peeked");
            offset = cached.end;
            plan.push(Ok(cached));
            continue;
        }
        // End the download chunk to fetch where the next cached one starts.
        // Each remaining cached download chunk starts after `offset`.
        let limit = cached_iter.peek().map_or(size, |c| c.start.min(size));
        let end = (offset + download_chunk_len).min(limit);
        plan.push(Err(DownloadChunkKey::new(blob_prefix, offset, end, size)));
        offset = end;
    }
    plan
}

/// The plan of a file, from its cached download chunks. Each expected plan
/// comes from a walk by hand of the rules of [plan_download_chunks].
#[cfg(test)]
mod tests {
    use super::*;

    const PREFIX: &str = "huggingface/hub/models--org--name/blobs/etag";

    /// A key of [PREFIX], for a file of `size` bytes.
    fn key(start: usize, end: usize, size: usize) -> DownloadChunkKey {
        DownloadChunkKey::new(PREFIX, start, end, size)
    }

    /// Fails when an entry of `plan` holds no byte, or does not start at the
    /// end of the entry before it. The first entry starts at 0, and the last
    /// entry ends at `size`.
    fn assert_contiguous(plan: &DownloadChunkPlan, size: usize) {
        let mut offset = 0;
        for entry in plan {
            let (Ok(key) | Err(key)) = entry;
            assert_eq!(key.start, offset, "a gap or an overlap before {key:?}");
            assert!(!key.is_empty(), "an entry of length zero: {key:?}");
            offset = key.end;
        }
        assert_eq!(offset, size, "the plan does not end at the file size");
    }

    /// Each offset of a key has zeros in front, to the width of the file size.
    /// [DownloadChunkKey::parse] reads the key back.
    #[test]
    fn a_key_holds_its_offsets_with_zeros_in_front() {
        let key = DownloadChunkKey::new("prefix", 5, 10, 1000);
        assert_eq!(key.key, "prefix/0005-0010");
        assert_eq!(DownloadChunkKey::parse(key.key.clone()), Some(key));
        assert_eq!(DownloadChunkKey::parse("prefix/5_10".into()), None);
    }

    /// Each entry has the download chunk length, and the last entry ends at
    /// the file size.
    #[test]
    fn an_empty_hub_cache_gives_only_entries_to_fetch() {
        let plan = plan_download_chunks(PREFIX, 25, 10, vec![]);
        assert_contiguous(&plan, 25);
        assert_eq!(
            plan,
            vec![Err(key(0, 10, 25)), Err(key(10, 20, 25)), Err(key(20, 25, 25))]
        );
    }

    #[test]
    fn a_full_hub_cache_gives_only_cached_entries() {
        let cached = vec![key(0, 10, 25), key(10, 20, 25), key(20, 25, 25)];
        let plan = plan_download_chunks(PREFIX, 25, 10, cached);
        assert_contiguous(&plan, 25);
        assert_eq!(
            plan,
            vec![Ok(key(0, 10, 25)), Ok(key(10, 20, 25)), Ok(key(20, 25, 25))]
        );
    }

    /// The cached keys come from a plan with a download chunk length of 15.
    /// The plan uses them, then fetches the rest with a length of 10.
    #[test]
    fn cached_keys_of_another_download_chunk_length_give_a_valid_plan() {
        let cached = vec![key(0, 15, 45), key(15, 30, 45)];
        let plan = plan_download_chunks(PREFIX, 45, 10, cached);
        assert_contiguous(&plan, 45);
        assert_eq!(
            plan,
            vec![
                Ok(key(0, 15, 45)),
                Ok(key(15, 30, 45)),
                Err(key(30, 40, 45)),
                Err(key(40, 45, 45)),
            ]
        );
    }

    /// The key `10-20` starts inside the cached key `0-15`. The plan does not
    /// use it, and fetches its bytes again.
    #[test]
    fn a_cached_key_that_overlaps_the_entry_before_it_is_not_used() {
        let cached = vec![key(0, 15, 30), key(10, 20, 30)];
        let plan = plan_download_chunks(PREFIX, 30, 10, cached);
        assert_contiguous(&plan, 30);
        assert_eq!(
            plan,
            vec![Ok(key(0, 15, 30)), Err(key(15, 25, 30)), Err(key(25, 30, 30))]
        );
    }

    /// The entry to fetch at the offset 10 ends at 13, the start of the cached
    /// key.
    #[test]
    fn a_cached_key_inside_a_gap_ends_the_entry_to_fetch_before_it() {
        let cached = vec![key(13, 23, 30)];
        let plan = plan_download_chunks(PREFIX, 30, 10, cached);
        assert_contiguous(&plan, 30);
        assert_eq!(
            plan,
            vec![
                Err(key(0, 10, 30)),
                Err(key(10, 13, 30)),
                Ok(key(13, 23, 30)),
                Err(key(23, 30, 30)),
            ]
        );
    }

    #[test]
    fn a_file_of_size_zero_gives_an_empty_plan() {
        assert_eq!(plan_download_chunks(PREFIX, 0, 10, vec![]), vec![]);
        assert_eq!(plan_download_chunks(PREFIX, 0, 10, vec![key(0, 0, 0)]), vec![]);
    }

    /// The cached keys hold three keys of length zero. They are at the offset
    /// 0, between two cached keys, and at the file size.
    #[test]
    fn a_cached_key_of_length_zero_has_no_effect_on_the_plan() {
        let cached = vec![
            key(0, 0, 25),
            key(0, 10, 25),
            key(10, 10, 25),
            key(10, 20, 25),
            key(25, 25, 25),
        ];
        let plan = plan_download_chunks(PREFIX, 25, 10, cached);
        assert_contiguous(&plan, 25);
        assert_eq!(
            plan,
            vec![Ok(key(0, 10, 25)), Ok(key(10, 20, 25)), Err(key(20, 25, 25))]
        );
    }
}
