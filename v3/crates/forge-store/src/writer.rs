use crate::format::{encode_header, SegmentHeader};
use forge_core::{ForgeError, SegmentKey};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

fn tmp_sibling(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".tmp");
    path.with_file_name(name)
}

fn fsync_parent_dir(path: &Path) -> Result<(), ForgeError> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let dir = File::open(parent)?;
    dir.sync_all()?;
    Ok(())
}

pub fn write_segment(
    path: &Path,
    key: SegmentKey,
    payload: &[u8],
) -> Result<[u8; 32], ForgeError> {
    let checksum = *blake3::hash(payload).as_bytes();
    let header = SegmentHeader {
        key,
        payload_len: payload.len() as u64,
        checksum,
    };
    let tmp = tmp_sibling(path);
    {
        let mut f = File::create(&tmp)?;
        f.write_all(&encode_header(&header))?;
        f.write_all(payload)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    fsync_parent_dir(path)?;
    Ok(checksum)
}

pub fn write_manifest_atomic(path: &Path, bytes: &[u8]) -> Result<(), ForgeError> {
    let tmp = tmp_sibling(path);
    {
        let mut f = File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    fsync_parent_dir(path)?;
    Ok(())
}
