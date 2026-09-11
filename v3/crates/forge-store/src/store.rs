use crate::reader::read_segment_verified_into;
use forge_core::{ForgeError, SegmentKey};
use std::path::{Component, Path, PathBuf};

pub const DEFAULT_MAX_SEGMENT_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Clone, Debug)]
pub struct StoreConfig {
    pub max_segment_bytes: u64,
    pub direct_cache_hint: bool,
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            max_segment_bytes: DEFAULT_MAX_SEGMENT_BYTES,
            direct_cache_hint: false,
        }
    }
}

fn validate_relative_path(p: &str) -> Result<(), ForgeError> {
    let path = Path::new(p);
    if path.is_absolute() {
        return Err(ForgeError::InvalidRelativePath { path: p.to_string() });
    }
    for c in path.components() {
        match c {
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(ForgeError::InvalidRelativePath { path: p.to_string() });
            }
            _ => {}
        }
    }
    Ok(())
}

pub struct SegmentStore {
    root: PathBuf,
    config: StoreConfig,
}

impl SegmentStore {
    pub fn open(root: impl Into<PathBuf>, config: StoreConfig) -> Result<Self, ForgeError> {
        let root = root.into();
        if !root.is_dir() {
            return Err(ForgeError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("store root not found: {}", root.display()),
            )));
        }
        Ok(Self { root, config })
    }

    pub fn config(&self) -> &StoreConfig {
        &self.config
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn read_verified(
        &self,
        relative_path: &str,
        expected: SegmentKey,
        dst: &mut Vec<u8>,
    ) -> Result<SegmentRead, ForgeError> {
        validate_relative_path(relative_path)?;
        let path = self.root.join(relative_path);
        read_segment_verified_into(&path, expected, self.config.max_segment_bytes, dst)
    }
}
