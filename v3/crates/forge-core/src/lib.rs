use serde::{Deserialize, Serialize};

pub type LayerId = u16;
pub type ExpertId = u16;
pub type ShardId = u16;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct ExpertKey {
    pub layer: LayerId,
    pub expert: ExpertId,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct SegmentKey {
    pub layer: LayerId,
    pub expert: ExpertId,
    pub shard: ShardId,
}

impl SegmentKey {
    pub const fn whole(layer: LayerId, expert: ExpertId) -> Self {
        Self { layer, expert, shard: 0 }
    }
}

impl From<SegmentKey> for ExpertKey {
    fn from(k: SegmentKey) -> Self {
        Self { layer: k.layer, expert: k.expert }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Quantization {
    F32,
    F16,
    Q8_0,
    Synthetic,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SegmentDescriptor {
    pub key: SegmentKey,
    pub relative_path: String,
    pub byte_len: u64,
    pub checksum_blake3: [u8; 32],
    pub quantization: Quantization,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelManifest {
    pub format_version: u32,
    pub model_id: String,
    pub router_seed: u64,
    pub num_layers: LayerId,
    pub num_experts_per_layer: ExpertId,
    pub expert_bytes: u64,
    pub segments: Vec<SegmentDescriptor>,
}

#[derive(Debug, thiserror::Error)]
pub enum ForgeError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("bad magic: {0:?}")]
    BadMagic([u8; 4]),
    #[error("unsupported segment format version: {0}")]
    UnsupportedVersion(u32),
    #[error("checksum mismatch for segment {key:?}")]
    ChecksumMismatch { key: SegmentKey },
    #[error("payload length mismatch for {key:?}: expected={expected}, actual={actual}")]
    LengthMismatch { key: SegmentKey, expected: u64, actual: u64 },
    #[error("segment key mismatch at {path}: header={header:?}, expected={expected:?}")]
    KeyMismatch { path: String, header: SegmentKey, expected: SegmentKey },
    #[error("reserved header bytes non-zero at {path}")]
    ReservedNonZero { path: String },
    #[error("manifest error: {0}")]
    Manifest(String),
    #[error("segment not found: {0:?}")]
    NotFound(SegmentKey),
    #[error("segment too large for this process: {key:?}, {bytes} bytes (limit {limit})")]
    SegmentTooLarge { key: SegmentKey, bytes: u64, limit: u64 },
    #[error("truncated segment {key:?}: required={required} bytes, file_size={file_size} bytes")]
    TruncatedSegment { key: SegmentKey, required: u64, file_size: u64 },
    #[error("segment size overflow: header payload_len={payload_len}")]
    SizeOverflow { payload_len: u64 },
    #[error("allocation failed for {key:?}: {bytes} bytes")]
    AllocFailed { key: SegmentKey, bytes: u64 },
    #[error("invalid relative path for store: {path}")]
    InvalidRelativePath { path: String },
}
