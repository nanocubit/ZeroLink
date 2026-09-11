use forge_core::ExpertKey;

pub fn synth_expert_payload(seed: u64, key: ExpertKey, len: usize) -> Vec<u8> {
    let mut h = blake3::Hasher::new();
    h.update(b"archi-forge/synthetic-expert/v1");
    h.update(&seed.to_le_bytes());
    h.update(&key.layer.to_le_bytes());
    h.update(&key.expert.to_le_bytes());
    let mut xof = h.finalize_xof();
    let mut out = vec![0u8; len];
    xof.fill(&mut out);
    out
}
