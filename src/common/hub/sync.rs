//! The blocking hub client of the native binary.
//!
//! Its hub cache has the standard layout of `~/.cache/huggingface/hub`. So the
//! client uses the files that other HuggingFace tools fetched, and other tools
//! use the files that it fetched:
//!
//! ```text
//! <hub cache>/models--EleutherAI--gpt-neox-20b/
//!   refs/main                    -> the commit hash
//!   blobs/<etag>                 -> the file contents
//!   snapshots/<commit>/<file>    -> a symlink to the blob
//! ```

use super::{
    BlobHash, Endpoint, FilePath, FileUrl, HubError, Metadata, Repo, RepoId, RepoType,
    RevisionHash, UrlTemplate, clean_etag, size_from_content_range,
};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const USER_AGENT: &str = concat!(env!("CARGO_PKG_NAME"), "/", env!("CARGO_PKG_VERSION"));
/// The size of one read from the socket. After each read, the client writes
/// the bytes to the file, and logs the progress at each 10%.
const READ_BUFFER_LEN: usize = 8 * 1024 * 1024;

/// A hub client bound to an endpoint and a hub cache directory.
#[derive(Clone, Debug)]
pub struct Api {
    endpoint: Endpoint,
    url_template: UrlTemplate,
    hub_cache_dir: PathBuf,
    agent: ureq::Agent,
    /// The agent of the metadata requests. Only the redirect response of the
    /// hub has the `x-repo-commit` and `x-linked-etag` headers. The response of
    /// the CDN does not have them.
    no_redirect_agent: ureq::Agent,
}

impl Api {
    /// A client for `https://huggingface.co`. Its hub cache is `$HF_HOME/hub`,
    /// else `~/.cache/huggingface/hub`.
    pub fn new() -> Result<Self, HubError> {
        Ok(Self {
            endpoint: Endpoint::default(),
            url_template: UrlTemplate::default(),
            hub_cache_dir: default_hub_cache_dir(),
            agent: ureq::AgentBuilder::new().user_agent(USER_AGENT).build(),
            no_redirect_agent: ureq::AgentBuilder::new()
                .user_agent(USER_AGENT)
                .redirects(0)
                .build(),
        })
    }

    /// Overrides the hub cache directory.
    pub fn with_hub_cache_dir(mut self, hub_cache_dir: PathBuf) -> Self {
        self.hub_cache_dir = hub_cache_dir;
        self
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

    /// Resolves the commit, the etag and the size of a file, with no fetch of
    /// the file.
    ///
    /// The function follows each relative redirect. The hub uses them between
    /// its own hosts. Then the hub sends an absolute redirect to the CDN. That
    /// redirect response has the headers that the function reads, so the
    /// function does *not* follow it.
    fn metadata(&self, url: &FileUrl) -> Result<Metadata, HubError> {
        let mut current = url.0.clone();
        let response = loop {
            let response = self
                .no_redirect_agent
                .get(&current)
                .set("Range", "bytes=0-0")
                .call()
                .map_err(|e| HubError::Request {
                    url: current.clone(),
                    message: e.to_string(),
                })?;

            let is_redirect = matches!(response.status(), 301 | 302 | 303 | 307 | 308);
            let location = response.header("location").map(str::to_string);
            match (is_redirect, location) {
                // A relative redirect stays on the hub: follow it and keep looking.
                (true, Some(location)) if location.starts_with('/') => {
                    current = join_relative(&current, &location);
                }
                _ => break response,
            }
        };

        let etag = response
            .header("x-linked-etag")
            .or_else(|| response.header("etag"))
            .map(clean_etag)
            .ok_or(HubError::Header {
                name: "etag",
                url: current.clone(),
            })?;
        // The header is absent when the hub does not send the response. The
        // client uses it only for the name of the snapshot directory, so an
        // empty value is permitted.
        let commit_hash = response
            .header("x-repo-commit")
            .unwrap_or_default()
            .to_string();
        let size = response
            .header("x-linked-size")
            .and_then(|s| s.trim().parse().ok())
            .or_else(|| {
                response
                    .header("content-range")
                    .and_then(size_from_content_range)
            })
            .ok_or(HubError::Header {
                name: "content-range",
                url: current.clone(),
            })?;

        Ok(Metadata {
            commit_hash: RevisionHash(commit_hash),
            etag: BlobHash(etag),
            size,
        })
    }
}

/// A handle on one repository. It makes the URLs of its files, and fetches
/// them.
#[derive(Clone, Debug)]
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

    /// Returns a local path to `filename`, and fetches it if the hub cache
    /// misses.
    pub fn get(&self, filename: &FilePath) -> Result<PathBuf, HubError> {
        if let Some(cached) = self.cached_pointer(filename) {
            log::info!("{} is already cached at {cached:?}", filename.0);
            return Ok(cached);
        }

        let url = self.url(filename);
        let metadata = self.api.metadata(&url)?;

        let blob = self.blob_path(&metadata.etag);
        if !blob.exists() {
            log::info!(
                "fetching {} ({:.1} MiB) from {}",
                filename.0,
                metadata.size as f64 / (1024. * 1024.),
                url.0
            );
            self.fetch_to(&url, &blob, metadata.size)?;
        }

        self.write_pointer(filename, &metadata.commit_hash, &blob)?;
        Ok(blob)
    }

    /// Writes the body of `url` into `blob`, through a temporary file. So an
    /// interrupted fetch never leaves a truncated blob.
    fn fetch_to(&self, url: &FileUrl, blob: &Path, size: usize) -> Result<(), HubError> {
        let response = self
            .api
            .agent
            .get(&url.0)
            .call()
            .map_err(|e| HubError::Request {
                url: url.0.clone(),
                message: e.to_string(),
            })?;

        create_dir_all(blob.parent().expect("blob paths have a parent"))?;
        let temp = blob.with_extension("incomplete");
        let mut file = std::fs::File::create(&temp).map_err(|e| hub_cache_err(&temp, e))?;

        let mut reader = response.into_reader();
        let mut buffer = vec![0u8; READ_BUFFER_LEN];
        let mut written = 0usize;
        let mut next_report = 0usize;
        loop {
            let read = reader.read(&mut buffer).map_err(|e| HubError::Request {
                url: url.0.clone(),
                message: e.to_string(),
            })?;
            if read == 0 {
                break;
            }
            file.write_all(&buffer[..read])
                .map_err(|e| hub_cache_err(&temp, e))?;
            written += read;
            if size > 0 && written * 10 / size > next_report {
                next_report = written * 10 / size;
                log::info!("fetched {}%", next_report * 10);
            }
        }
        file.flush().map_err(|e| hub_cache_err(&temp, e))?;
        drop(file);

        if size > 0 && written != size {
            let _ = std::fs::remove_file(&temp);
            return Err(HubError::ShortRead {
                expected: size,
                got: written,
            });
        }

        std::fs::rename(&temp, blob).map_err(|e| hub_cache_err(blob, e))?;
        Ok(())
    }

    /// `<hub cache>/<folder_name>`.
    fn repo_dir(&self) -> PathBuf {
        self.api.hub_cache_dir.join(self.repo.folder_name())
    }

    fn ref_path(&self) -> PathBuf {
        self.repo_dir().join("refs").join(&self.repo.revision.0)
    }

    fn blob_path(&self, etag: &BlobHash) -> PathBuf {
        self.repo_dir().join("blobs").join(&etag.0)
    }

    /// The path of `filename` in the hub cache. It is `None` when no earlier
    /// call resolved the revision, or when the snapshot does not hold the file.
    fn cached_pointer(&self, filename: &FilePath) -> Option<PathBuf> {
        let commit = std::fs::read_to_string(self.ref_path()).ok()?;
        let pointer = self
            .repo_dir()
            .join("snapshots")
            .join(commit.trim())
            .join(&filename.0);
        pointer.exists().then_some(pointer)
    }

    /// Records `refs/<revision>` and links `snapshots/<commit>/<filename>` at the blob.
    fn write_pointer(
        &self,
        filename: &FilePath,
        commit: &RevisionHash,
        blob: &Path,
    ) -> Result<(), HubError> {
        if commit.0.is_empty() {
            // Without a commit, the snapshot directory has no name. The caller
            // still receives the blob path, but a later call cannot find the
            // blob from the revision.
            return Ok(());
        }

        let ref_path = self.ref_path();
        create_dir_all(ref_path.parent().expect("ref paths have a parent"))?;
        std::fs::write(&ref_path, commit.0.trim()).map_err(|e| hub_cache_err(&ref_path, e))?;

        let pointer = self
            .repo_dir()
            .join("snapshots")
            .join(&commit.0)
            .join(&filename.0);
        if pointer.exists() {
            return Ok(());
        }
        create_dir_all(pointer.parent().expect("pointer paths have a parent"))?;
        symlink_or_copy(blob, &pointer)
    }
}

/// `$HF_HOME/hub`, else `~/.cache/huggingface/hub`, else a local `.cache`.
fn default_hub_cache_dir() -> PathBuf {
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        return PathBuf::from(hf_home).join("hub");
    }
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_default();
    home.join(".cache").join("huggingface").join("hub")
}

/// Resolves a same-host redirect target against the URL it came from.
fn join_relative(base: &str, location: &str) -> String {
    // `base` has the form "https://host/path…". Keep the part up to the host.
    match base
        .find("://")
        .and_then(|i| base[i + 3..].find('/').map(|j| i + 3 + j))
    {
        Some(path_start) => format!("{}{}", &base[..path_start], location),
        None => format!("{base}{location}"),
    }
}

fn symlink_or_copy(blob: &Path, pointer: &Path) -> Result<(), HubError> {
    #[cfg(unix)]
    let linked = std::os::unix::fs::symlink(blob, pointer).is_ok();
    #[cfg(windows)]
    let linked = std::os::windows::fs::symlink_file(blob, pointer).is_ok();
    #[cfg(not(any(unix, windows)))]
    let linked = false;

    if linked {
        return Ok(());
    }
    std::fs::copy(blob, pointer)
        .map(|_| ())
        .map_err(|e| hub_cache_err(pointer, e))
}

fn create_dir_all(path: &Path) -> Result<(), HubError> {
    std::fs::create_dir_all(path).map_err(|e| hub_cache_err(path, e))
}

fn hub_cache_err(path: &Path, error: std::io::Error) -> HubError {
    HubError::HubCache {
        context: path.display().to_string(),
        message: error.to_string(),
    }
}
