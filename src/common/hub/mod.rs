//! A minimal HuggingFace hub client, replacing the `hf-hub` fork this demo used
//! to depend on.
//!
//! Only what the demo needs is implemented: resolve a file's metadata (commit,
//! etag, size) and get its bytes, with an on-disk cache natively
//! ([sync]) and a chunked IndexedDB cache in the browser ([wasm]).
//!
//! Both back-ends share the naming below, and the browser one keeps `hf-hub`'s
//! IndexedDB database/store/key layout so caches written by earlier builds of this
//! app are still recognised.

#[cfg(not(target_arch = "wasm32"))]
pub mod sync;
#[cfg(target_arch = "wasm32")]
pub mod wasm;

use std::fmt;

/// Everything that can go wrong while talking to the hub or to a cache.
#[derive(Debug)]
pub enum HubError {
    /// The request itself failed (transport, DNS, CORS, non-success status…).
    Request { url: String, message: String },
    /// A response header the hub is expected to send was missing or unusable.
    Header { name: &'static str, url: String },
    /// The local cache (filesystem or IndexedDB) could not be read or written.
    Cache { context: String, message: String },
    /// A range request returned a different number of bytes than asked for.
    ShortRead { expected: usize, got: usize },
}

impl fmt::Display for HubError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Request { url, message } => write!(f, "request to {url} failed: {message}"),
            Self::Header { name, url } => {
                write!(f, "response from {url} is missing a usable {name} header")
            }
            Self::Cache { context, message } => write!(f, "cache error ({context}): {message}"),
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

/// What kind of repository is being addressed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RepoType {
    Model,
    Dataset,
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

/// A fully-resolved download URL.
#[derive(Clone, Debug, PartialEq)]
pub struct FileUrl(pub String);

/// A repository at a given revision.
#[derive(Clone, Debug, PartialEq)]
pub struct Repo {
    pub repo_id: RepoId,
    pub repo_type: RepoType,
    pub revision: RevisionPath,
}

impl Repo {
    /// At the default branch (`"main"`).
    pub fn new(repo_id: RepoId, repo_type: RepoType) -> Self {
        Self::with_revision(repo_id, repo_type, RevisionPath::default())
    }

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

    /// The cache directory name. Eg. `"models--EleutherAI--gpt-neox-20b"`.
    pub fn folder_name(&self) -> String {
        format!("{}--{}", self.repo_type.prefix(), self.repo_id.0).replace('/', "--")
    }
}

/// How a [FileUrl] is built out of an [Endpoint], a [Repo] and a [FilePath].
#[derive(Clone, Debug, PartialEq)]
pub struct UrlTemplate(pub String);

impl Default for UrlTemplate {
    /// `"{endpoint}/{repo_id}/resolve/{revision}/{filename}"`.
    fn default() -> Self {
        Self("{endpoint}/{repo_id}/resolve/{revision}/{filename}".into())
    }
}

impl UrlTemplate {
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

/// What the hub reports about a file before it is downloaded.
#[derive(Clone, Debug, PartialEq)]
pub struct Metadata {
    /// The commit the revision currently resolves to. May be empty in the
    /// browser, where the redirect that carries `x-repo-commit` is not visible
    /// to `fetch`.
    pub commit_hash: RevisionHash,
    /// The blob's content hash, used as its cache key.
    pub etag: BlobHash,
    /// The file size, in bytes.
    pub size: usize,
}

/// Parses the total size out of a `content-range: bytes 0-0/12345` header.
fn size_from_content_range(value: &str) -> Option<usize> {
    value.rsplit('/').next()?.trim().parse().ok()
}

/// The hub quotes etags; the quotes are not part of the hash.
fn clean_etag(value: &str) -> String {
    value.replace('"', "")
}
