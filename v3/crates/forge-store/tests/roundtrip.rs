use forge_core::{ExpertKey, ForgeError, SegmentKey};
use forge_store::{
    probe_segment, read_segment_verified, synth_expert_payload, write_segment,
    SegmentStore, StoreConfig, FORMAT_VERSION, HEADER_LEN, HEADER_LEN_U64, MAGIC,
};

fn store_at_dir(p: &std::path::Path) -> SegmentStore {
    SegmentStore::open(p, StoreConfig::default()).unwrap()
}

#[test]
fn header_len_is_stable() {
    assert_eq!(HEADER_LEN, 56);
    assert_eq!(HEADER_LEN_U64, 56);
    assert_eq!(MAGIC, b"AFSG");
    assert_eq!(FORMAT_VERSION, 1);
}

#[test]
fn roundtrip_through_store_is_byte_exact() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("seg.bin");
    let key = SegmentKey::whole(3, 17);
    let payload = synth_expert_payload(0xDEADBEEF, ExpertKey { layer: 3, expert: 17 }, 4096);
    write_segment(&path, key, &payload).unwrap();
    let store = store_at_dir(&dir.path());
    let mut buf = Vec::new();
    let r = store.read_verified("seg.bin", key, &mut buf).unwrap();
    assert_eq!(buf, payload);
    assert_eq!(r.bytes_payload, 4096);
    assert_eq!(r.logical_bytes_read, 56 + 4096);
}

#[test]
fn buffer_is_reused_across_reads() {
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("a.bin");
    let b = dir.path().join("b.bin");
    write_segment(&a, SegmentKey::whole(0, 0), &vec![0xAAu8; 100]).unwrap();
    write_segment(&b, SegmentKey::whole(0, 1), &vec![0xBBu8; 50]).unwrap();
    let store = store_at_dir(&dir.path());
    let mut buf = Vec::with_capacity(256);
    let cap_before = buf.capacity();
    let ptr_before = buf.as_ptr();
    store.read_verified("a.bin", SegmentKey::whole(0, 0), &mut buf).unwrap();
    assert_eq!(buf, vec![0xAAu8; 100]);
    assert_eq!(buf.capacity(), cap_before);
    assert_eq!(buf.as_ptr(), ptr_before);
    store.read_verified("b.bin", SegmentKey::whole(0, 1), &mut buf).unwrap();
    assert_eq!(buf, vec![0xBBu8; 50]);
    assert_eq!(buf.capacity(), cap_before);
    assert_eq!(buf.as_ptr(), ptr_before);
}

#[test]
fn payload_is_deterministic_across_calls() {
    let k = ExpertKey { layer: 1, expert: 2 };
    assert_eq!(synth_expert_payload(7, k, 1024), synth_expert_payload(7, k, 1024));
    assert_ne!(synth_expert_payload(7, k, 1024), synth_expert_payload(8, k, 1024));
    assert_ne!(
        synth_expert_payload(7, k, 1024),
        synth_expert_payload(7, ExpertKey { layer: 1, expert: 3 }, 1024)
    );
}

#[test]
fn corrupted_payload_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("seg.bin");
    let key = SegmentKey::whole(0, 0);
    write_segment(&path, key, &vec![0xAAu8; 1024]).unwrap();
    use std::io::{Seek, SeekFrom, Write};
    let mut f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    f.seek(SeekFrom::Start(HEADER_LEN_U64 + 10)).unwrap();
    f.write_all(&[0x55]).unwrap();
    f.sync_all().unwrap();
    let store = store_at_dir(&dir.path());
    let mut buf = Vec::new();
    let err = store.read_verified("seg.bin", key, &mut buf).unwrap_err();
    assert!(matches!(err, ForgeError::ChecksumMismatch { .. }));
}

#[test]
fn key_mismatch_is_typed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("seg.bin");
    write_segment(&path, SegmentKey::whole(0, 0), &[0u8; 64]).unwrap();
    let store = store_at_dir(&dir.path());
    let mut buf = Vec::new();
    let err = store
        .read_verified("seg.bin", SegmentKey::whole(9, 9), &mut buf)
        .unwrap_err();
    assert!(matches!(err, ForgeError::KeyMismatch { .. }));
}

#[test]
fn oversized_header_is_rejected_before_allocation() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("seg.bin");
    let key = SegmentKey::whole(0, 0);
    write_segment(&path, key, &[0u8; 128]).unwrap();
    use std::io::{Seek, SeekFrom, Write};
    let mut f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    f.seek(SeekFrom::Start(16)).unwrap();
    f.write_all(&u64::MAX.to_le_bytes()).unwrap();
    f.sync_all().unwrap();
    let store = SegmentStore::open(
        dir.path(),
        StoreConfig {
            max_segment_bytes: 4096,
            direct_cache_hint: false,
        },
    )
    .unwrap();
    let mut buf = Vec::new();
    let err = store.read_verified("seg.bin", key, &mut buf).unwrap_err();
    match err {
        ForgeError::SegmentTooLarge { bytes, limit, .. } => {
            assert_eq!(bytes, u64::MAX);
            assert_eq!(limit, 4096);
        }
        other => panic!("expected SegmentTooLarge, got {:?}", other),
    }
}

#[test]
fn truncated_file_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("seg.bin");
    let key = SegmentKey::whole(0, 0);
    write_segment(&path, key, &vec![0x11u8; 4096]).unwrap();
    let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    f.set_len(HEADER_LEN_U64 + 2048).unwrap();
    drop(f);
    let store = store_at_dir(&dir.path());
    let mut buf = Vec::new();
    let err = store.read_verified("seg.bin", key, &mut buf).unwrap_err();
    match err {
        ForgeError::TruncatedSegment { required, file_size, .. } => {
            assert_eq!(required, HEADER_LEN_U64 + 4096);
            assert_eq!(file_size, HEADER_LEN_U64 + 2048);
        }
        other => panic!("expected TruncatedSegment, got {:?}", other),
    }
}

#[test]
fn bad_magic_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("seg.bin");
    write_segment(&path, SegmentKey::whole(0, 0), &[0u8; 32]).unwrap();
    use std::io::{Seek, SeekFrom, Write};
    let mut f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    f.seek(SeekFrom::Start(0)).unwrap();
    f.write_all(b"XXXX").unwrap();
    f.sync_all().unwrap();
    let store = store_at_dir(&dir.path());
    let mut buf = Vec::new();
    let err = store
        .read_verified("seg.bin", SegmentKey::whole(0, 0), &mut buf)
        .unwrap_err();
    assert!(matches!(err, ForgeError::BadMagic));
}

#[test]
fn reserved_bytes_must_be_zero() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("seg.bin");
    write_segment(&path, SegmentKey::whole(0, 0), &[0u8; 32]).unwrap();
    use std::io::{Seek, SeekFrom, Write};
    let mut f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    f.seek(SeekFrom::Start(14)).unwrap();
    f.write_all(&1u16.to_le_bytes()).unwrap();
    f.sync_all().unwrap();
    let store = store_at_dir(&dir.path());
    let mut buf = Vec::new();
    let err = store
        .read_verified("seg.bin", SegmentKey::whole(0, 0), &mut buf)
        .unwrap_err();
    assert!(matches!(err, ForgeError::ReservedNonZero { .. }));
}

#[test]
fn probe_sees_identity_and_length_without_payload() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("seg.bin");
    let key = SegmentKey::whole(2, 5);
    write_segment(&path, key, &[0u8; 777]).unwrap();
    let h = probe_segment(&path).unwrap();
    assert_eq!(h.key, key);
    assert_eq!(h.payload_len, 777);
}

#[test]
fn atomic_write_leaves_no_tmp_on_success() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("seg.bin");
    write_segment(&path, SegmentKey::whole(0, 0), &[0u8; 16]).unwrap();
    let entries: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0], "seg.bin");
}

#[test]
fn relative_path_with_parent_dir_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let store = store_at_dir(&dir.path());
    let mut buf = Vec::new();
    let err = store
        .read_verified("../etc/passwd", SegmentKey::whole(0, 0), &mut buf)
        .unwrap_err();
    assert!(matches!(err, ForgeError::InvalidRelativePath { .. }));
}

#[test]
fn absolute_path_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let store = store_at_dir(&dir.path());
    let mut buf = Vec::new();
    let err = store
        .read_verified("/etc/passwd", SegmentKey::whole(0, 0), &mut buf)
        .unwrap_err();
    assert!(matches!(err, ForgeError::InvalidRelativePath { .. }));
}
