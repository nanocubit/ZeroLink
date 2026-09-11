#[cfg(not(unix))]
compile_error!("forge-store requires a Unix-like target (uses std::os::unix::fs::FileExt::read_exact_at)");

mod format;
mod reader;
mod store;
mod synth;
mod writer;

pub use format::{HEADER_LEN, HEADER_LEN_U64, MAGIC, FORMAT_VERSION, SegmentHeader};
pub use reader::{probe_segment, read_segment_verified, read_segment_verified_into, SegmentRead};
pub use store::{SegmentStore, StoreConfig, DEFAULT_MAX_SEGMENT_BYTES};
pub use synth::synth_expert_payload;
pub use writer::{write_manifest_atomic, write_segment};
