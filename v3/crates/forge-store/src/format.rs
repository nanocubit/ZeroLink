use forge_core::{ForgeError, SegmentKey};

pub const MAGIC: [u8; 4] = *b"AFSG";
pub const FORMAT_VERSION: u32 = 1;
pub const HEADER_LEN: usize = 56;
pub const HEADER_LEN_U64: u64 = HEADER_LEN as u64;

const _: () = assert!(HEADER_LEN == 56);
const _: () = assert!(HEADER_LEN == 4 + 4 + 2 + 2 + 2 + 2 + 8 + 32);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SegmentHeader {
    pub key: SegmentKey,
    pub payload_len: u64,
    pub checksum: [u8; 32],
}

pub(crate) fn encode_header(h: &SegmentHeader) -> [u8; HEADER_LEN] {
    let mut b = [0u8; HEADER_LEN];
    b[0..4].copy_from_slice(&MAGIC);
    b[4..8].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    b[8..10].copy_from_slice(&h.key.layer.to_le_bytes());
    b[10..12].copy_from_slice(&h.key.expert.to_le_bytes());
    b[12..14].copy_from_slice(&h.key.shard.to_le_bytes());
    b[14..16].copy_from_slice(&0u16.to_le_bytes());
    b[16..24].copy_from_slice(&h.payload_len.to_le_bytes());
    b[24..56].copy_from_slice(&h.checksum);
    b
}

pub(crate) fn decode_header(buf: [u8; HEADER_LEN], path: &str) -> Result<SegmentHeader, ForgeError> {
    let magic: [u8; 4] = buf[0..4].try_into().unwrap();
    if magic != MAGIC {
        return Err(ForgeError::BadMagic(magic));
    }
    let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    if version != FORMAT_VERSION {
        return Err(ForgeError::UnsupportedVersion(version));
    }
    let layer = u16::from_le_bytes(buf[8..10].try_into().unwrap());
    let expert = u16::from_le_bytes(buf[10..12].try_into().unwrap());
    let shard = u16::from_le_bytes(buf[12..14].try_into().unwrap());
    let reserved = u16::from_le_bytes(buf[14..16].try_into().unwrap());
    if reserved != 0 {
        return Err(ForgeError::ReservedNonZero { path: path.to_string() });
    }
    let payload_len = u64::from_le_bytes(buf[16..24].try_into().unwrap());
    let mut checksum = [0u8; 32];
    checksum.copy_from_slice(&buf[24..56]);
    Ok(SegmentHeader {
        key: SegmentKey { layer, expert, shard },
        payload_len,
        checksum,
    })
}
