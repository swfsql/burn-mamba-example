//! A small client of the HuggingFace hub. The app does not use the `hf-hub`
//! crate (ADR-0001).
//!
//! The client does only what the app needs. It resolves the metadata of a file
//! (commit, etag, size), and it gets the bytes of the file. Natively, `sync`
//! keeps the hub cache on disk. In the browser, `wasm` keeps the hub cache in
//! IndexedDB, as download chunks.
//!
//! The two hub clients share the types below.

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
