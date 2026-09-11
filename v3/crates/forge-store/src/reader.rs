use crate::format::{decode_header, HEADER_LEN, HEADER_LEN_U64};
use forge_core::{ForgeError, SegmentKey};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SegmentRead {
    pub bytes_payload: u64,
    pub logical_bytes_read: u64,
    pub checksum: [u8; 32],
}

pub fn probe_segment(path: &Path) -> Result<crate::SegmentHeader, ForgeError> {
    let f = File::open(path)?;
    let mut buf = [0u8; HEADER_LEN];
    f.read_exact_at(&mut buf, 0)?;
    decode_header(buf, &path.display().to_string())
}

pub fn read_segment_verified_into(
    path: &Path,
    expected: SegmentKey,
    max_payload_bytes: u64,
    dst: &mut Vec<u8>,
) -> Result<SegmentRead, ForgeError> {
    let f = File::open(path)?;
    let mut header_buf = [0u8; HEADER_LEN];
    f.read_exact_at(&mut header_buf, 0)?;
    let header = decode_header(header_buf, &path.display().to_string())?;

    if header.key != expected {
        return Err(ForgeError::KeyMismatch {
            path: path.display().to_string(),
            header: header.key,
            expected,
        });
    }

    let payload_len = header.payload_len;
    let effective_limit = max_payload_bytes.min(usize::MAX as u64);
    if payload_len > effective_limit {
        return Err(ForgeError::SegmentTooLarge {
            key: expected,
            bytes: payload_len,
            limit: effective_limit,
        });
    }

    let required = HEADER_LEN_U64
        .checked_add(payload_len)
        .ok_or(ForgeError::SizeOverflow { payload_len })?;

    let file_size = f.metadata()?.len();
    if file_size < required {
        return Err(ForgeError::TruncatedSegment {
            key: expected,
            required,
            file_size,
        });
    }

    dst.clear();
    dst.try_reserve_exact(payload_len as usize)
        .map_err(|_| ForgeError::AllocFailed { key: expected, bytes: payload_len })?;
    dst.resize(payload_len as usize, 0);
    f.read_exact_at(dst.as_mut_slice(), HEADER_LEN_U64)?;

    let actual = *blake3::hash(dst.as_slice()).as_bytes();
    if actual != header.checksum {
        return Err(ForgeError::ChecksumMismatch { key: expected });
    }

    Ok(SegmentRead {
        bytes_payload: payload_len,
        logical_bytes_read: required,
        checksum: header.checksum,
    })
}

pub fn read_segment_verified(
    path: &Path,
    expected: SegmentKey,
    max_payload_bytes: u64,
) -> Result<(Vec<u8>, SegmentRead), ForgeError> {
    let mut v = Vec::new();
    let r = read_segment_verified_into(path, expected, max_payload_bytes, &mut v)?;
    Ok((v, r))
}
