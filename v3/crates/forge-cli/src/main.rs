use clap::{Args, Parser, Subcommand};
use forge_core::{ExpertKey, ForgeError, ModelManifest, Quantization, SegmentDescriptor, SegmentKey};
use forge_store::{probe_segment, synth_expert_payload, write_manifest_atomic, write_segment, SegmentStore, StoreConfig, DEFAULT_MAX_SEGMENT_BYTES, FORMAT_VERSION};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "forge", version, about = "ARCHI-Forge segmented MoE runtime")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    BuildFixture(BuildFixture),
    Verify(Verify),
}

#[derive(Args)]
struct BuildFixture {
    #[arg(long)] out: PathBuf,
    #[arg(long, default_value_t = 8)] layers: u16,
    #[arg(long, default_value_t = 32)] experts: u16,
    #[arg(long, default_value_t = 65536)] expert_bytes: usize,
    #[arg(long, default_value_t = 0xA2C117E5u64)] seed: u64,
    #[arg(long, default_value = "synthetic-moe-v1")] model_id: String,
}

#[derive(Args)]
struct Verify {
    #[arg(long)] manifest: PathBuf,
    #[arg(long, default_value_t = DEFAULT_MAX_SEGMENT_BYTES)] max_segment_bytes: u64,
}

fn main() -> Result<(), ForgeError> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::BuildFixture(a) => build_fixture(a),
        Cmd::Verify(a) => verify(a),
    }
}

fn build_fixture(a: BuildFixture) -> Result<(), ForgeError> {
    let root = &a.out;
    let seg_dir = root.join("segments");
    std::fs::create_dir_all(&seg_dir)?;

    let capacity = (a.layers as usize) * (a.experts as usize);
    let mut segments = Vec::with_capacity(capacity);
    for layer in 0..a.layers {
        for expert in 0..a.experts {
            let ekey = ExpertKey { layer, expert };
            let skey = SegmentKey::whole(layer, expert);
            let payload = synth_expert_payload(a.seed, ekey, a.expert_bytes);
            let name = format!("l{:03}_e{:03}_s000.bin", layer, expert);
            let path = seg_dir.join(&name);
            let checksum = write_segment(&path, skey, &payload)?;
            segments.push(SegmentDescriptor {
                key: skey,
                relative_path: format!("segments/{}", name),
                byte_len: payload.len() as u64,
                checksum_blake3: checksum,
                quantization: Quantization::Synthetic,
            });
        }
    }

    let manifest = ModelManifest {
        format_version: FORMAT_VERSION,
        model_id: a.model_id,
        router_seed: a.seed,
        num_layers: a.layers,
        num_experts_per_layer: a.experts,
        expert_bytes: a.expert_bytes as u64,
        segments,
    };
    let json = serde_json::to_vec_pretty(&manifest)
        .map_err(|e| ForgeError::Manifest(e.to_string()))?;
    write_manifest_atomic(&root.join("manifest.json"), &json)?;

    eprintln!(
        "wrote {} segments ({} bytes payload each) to {}",
        manifest.segments.len(),
        a.expert_bytes,
        root.display()
    );
    Ok(())
}

fn verify(a: Verify) -> Result<(), ForgeError> {
    let manifest_path = &a.manifest;
    let root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let bytes = std::fs::read(manifest_path)?;
    let manifest: ModelManifest = serde_json::from_slice(&bytes)
        .map_err(|e| ForgeError::Manifest(e.to_string()))?;

    let store = SegmentStore::open(
        root,
        StoreConfig {
            max_segment_bytes: a.max_segment_bytes,
            direct_cache_hint: false,
        },
    )?;

    let mut total_payload = 0u64;
    let mut total_logical = 0u64;
    let mut buf = Vec::new();

    for d in &manifest.segments {
        let abs = root.join(&d.relative_path);
        let hdr = probe_segment(&abs)?;
        if hdr.key != d.key {
            return Err(ForgeError::KeyMismatch {
                path: abs.display().to_string(),
                header: hdr.key,
                expected: d.key,
            });
        }
        if hdr.payload_len != d.byte_len {
            return Err(ForgeError::LengthMismatch {
                key: d.key,
                expected: d.byte_len,
                actual: hdr.payload_len,
            });
        }
        if hdr.checksum != d.checksum_blake3 {
            return Err(ForgeError::Manifest(format!(
                "manifest checksum disagrees with segment header for {:?}",
                d.key
            )));
        }
        let read = store.read_verified(&d.relative_path, d.key, &mut buf)?;
        if read.bytes_payload != d.byte_len {
            return Err(ForgeError::LengthMismatch {
                key: d.key,
                expected: d.byte_len,
                actual: read.bytes_payload,
            });
        }
        total_payload += read.bytes_payload;
        total_logical += read.logical_bytes_read;
    }

    eprintln!(
        "verified {}/{} segments OK (payload: {} bytes, logical: {} bytes including headers)",
        manifest.segments.len(),
        manifest.segments.len(),
        total_payload,
        total_logical
    );
    Ok(())
}
