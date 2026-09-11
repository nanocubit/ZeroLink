# ARCHI-Forge v3

Segmented MoE storage runtime for ZeroLink.

## Quickstart

```bash
cargo test --workspace
cargo run --release -p forge-cli -- build-fixture --out /tmp/forge-moe
cargo run --release -p forge-cli -- verify --manifest /tmp/forge-moe/manifest.json
```

## Routing Trace

```bash
# Synthetic trace (2500 tokens, 8 layers, top-2)
./target/release/forge bench-read \
  --manifest /tmp/forge-moe/manifest.json \
  --trace reports/phase-b/traces/routing-trace-synthetic-2500.jsonl \
  --direct \
  --out reports/phase-b/bench-synthetic.jsonl

# Real trace from TinyMoE (Colab)
python3 notebooks/extract_routing_trace.py
cargo run --release -p forge-cli -- bench-read \
  --manifest /tmp/forge-moe/manifest.json \
  --trace routing-trace-tinymoe-real.jsonl \
  --direct \
  --out reports/phase-b/bench-real.jsonl
```
