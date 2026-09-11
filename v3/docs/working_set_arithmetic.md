# Working Set Arithmetic for MoE Models

## Units convention

Throughout this document:

- **MiB / GiB / TiB** = binary units, `2^20 / 2^30 / 2^40` bytes.
- All calculations divide by `1,048,576`, `1,073,741,824`, `1,099,511,627,776`.

Do not read "MB" as `10^6` anywhere below. The byte counts in the code
snippets are exact; the MiB/GiB/TiB values are the byte counts divided by
the binary unit.

## Criterion

**Storage is NOT a bottleneck if:**

```
live_set_per_layer < SLC_effective_size
```

Where:
- `live_set_per_layer` = bytes occupied by the **distinct** experts activated
  over a reuse window W, in one MoE layer.
- `SLC_effective_size` = System Level Cache size of the target machine,
  minus contention from GPU/ANE/DMA. Roughly 50–75% of total SLC.

If the live set fits, cold reads are paid once per expert over the window,
not once per token.

**Caveat:** SLC is shared between CPU, GPU, Neural Engine, DMA, and video
encoders. A single CPU process does not get the entire SLC. For planning
purposes, treat effective capacity as 50–75% of the documented total.

## Formulas

### Expert size (FFN)

```
params_per_expert  = 2 × hidden_size × moe_intermediate_size
bytes_per_param    = 2 (BF16) | 1 (INT8) | 0.5 (INT4)
expert_size_bytes  = params_per_expert × bytes_per_param
```

### Per-token traversal (lower bound on live set, not the live set itself)

```
per_token_traversal_per_layer = (num_experts_per_tok + num_shared_experts)
×¥ expert_size_bytes
```

This is how many bytes a **single token** reads in one layer. It is not
the working set. It is the floor of the live set: even a window of one
token already touches `top_k + shared` experts.

### Live set per layer

```
live_set_per_layer = | { e : e ∈ experts(t), for some t ∈ window W } |
×¥ expert_size_bytes
```

**Upper bound:**

```
live_set_per_layer ≤ min(W × (top_k + shared), num_experts + shared)
×¥ expert_size_bytes
```

**Measured (from trace):** compute per-position window set sizes over a
real routing trace.

```python
# live_set over W=10 for each layer, from a JSONL trace
# format: {"layer": int, "pos": int, "experts": [int, ...], "weights": [...]}
import json

W = 10
per_layer = {}
with open("routing-trace.jsonl") as f:
    for line in f:
        r = json.loads(line)
        per_layer.setdefault(r["layer"], []).append(r)

for layer in sorted(per_layer):
    recs = sorted(per_layer[layer], key=lambda x: x["pos"])
    counts = []
    for i in range(len(recs)):
        lo = max(0, i - W)
        w = set()
        for j in range(lo, i + 1):
            w.update(recs[j]["experts"])
        counts.append(len(w))
    print(f"layer {layer}: min={min(counts)} "
          f"mean={sum(counts)/len(counts):.2f} max={max(counts)}")
```

The measured live set is mean × expert_size_bytes for a representative
"typical" window, and max × expert_size_bytes for the worst window.

### Cumulative live set

```
cumulative_live_set = live_set_per_layer × num_moe_layers
```

This is not a concurrent working set. Layers are processed
sequentially. The number represents how many distinct expert bytes are
touched across the whole forward pass, if layers did not share cache
lines. It is useful as an upper bound on total memory footprint of the
forward pass, not as the SLC residency requirement for any single point
in time.

## TinyMoE-100m-2x8

```python
num_experts = 8
num_experts_per_tok = 2
num_shared_experts = 0
hidden_size = 512
moe_intermediate_size = 1024
num_moe_layers = 10

# Expert size (BF16)
params_per_expert = 2 * 512 * 1024          # 1,048,576 params
expert_size_bytes = 1048576 * 2             # 2,097,152 bytes = 2 MiB

# Per-token traversal, per layer
per_token_traversal_per_layer = 2 * 2097152  # 4,194,304 bytes = 4 MiB

# Live set: not yet computed from trace over W=10.
# Known from trace:
#   - top-3 by frequency: {2, 6, 7}
#   - distinct experts over full 1000-token trace: 7
#     {0, 1, 2, 3, 4, 6, 7} — expert 5 absent
# Estimate at mean window size 3:
live_set_per_layer_estimate = 3 * 2097152   # 6,291,456 bytes = 6 MiB

# Cumulative live set (estimate)
cumulative_live_set_estimate = 6291456 * 10  # 62,914,560 bytes = 60 MiB
```

### Result

- Per-token traversal per layer: **4 MiB**
- Live set per layer: **~6 MiB** (estimate, top-3 frequency; not yet computed over W=10)
- Cumulative live set: **~60 MiB** (estimate)
- Apple SLC: **8 MiB** (M1/M2), **24 MiB** (M1/M2 Pro), **48 MiB** (M1 Max), **96 MiB** (M1 Ultra / M3 Max)
- **Conclusion:** Likely fits in M1 Max SLC (48 MiB) with headroom. Does not fit in M1/M2 base SLC (8 MiB) if the effective capacity is 4–6 MiB.

### Caveats

- Live set over W=10 not computed from trace. The snippet above is ready to run and takes seconds.
- Top-3 frequency (3 experts) ≠ distinct experts over W=10. These are different statistics.
- SLC contention: on M1 base (8 MiB total), effective CPU capacity is probably 4–6 MiB.

## Ling-3.0-tiny

```python
# Config from HF config.json
# https://huggingface.co/inclusionAI/Ling-3.0-tiny/blob/main/config.json
num_experts = 128
num_experts_per_tok = 8
num_shared_experts = 1
hidden_size = 1536
moe_intermediate_size = 512         # confirmed, not a typo
num_hidden_layers = 24
first_k_dense_replace = 1           # layer 0 is dense
num_moe_layers = 23                 # layers 1..23 are MoE

# Expert size (BF16)
params_per_expert = 2 * 1536 * 512            # 1,572,864 params
expert_size_bytes = 1572864 * 2               # 3,145,728 bytes = 3 MiB

# Per-token traversal, per layer
per_token_traversal_per_layer = 9 * 3145728   # 28,311,552 bytes = 27 MiB

# Live set: interval (not measured)
# Lower bound: 9 experts = top-8 + shared
# Upper bound: min(10 * 9, 128 + 1) = 90 experts
live_set_per_layer_lower = 9  * 3145728       #  28,311,552 bytes =  27 MiB
live_set_per_layer_upper = 90 * 3145728       # 283,115,520 bytes = 270 MiB

# Cumulative live set over all 23 MoE layers
cumulative_live_set_lower = 28311552  * 23    #  651,165,696 bytes = 0.607 GiB
cumulative_live_set_upper = 283115520 * 23    # 6,511,656,960 bytes = 6.064 GiB
```

### Result

- Per-token traversal per layer: **27 MiB**
- Live set per layer: **[27 MiB, 270 MiB]** (interval, not measured)
- Cumulative live set: **[0.61 GiB, 6.06 GiB]** (interval, not measured)
- Apple SLC: **8–96 MiB**
- RAM: **4–16 GiB** typical for consumer Macs
- **Conclusion:** Does not fit in SLC on any Mac. Fits in RAM on 8+ GiB machines. Cold reads paid once per layer over the window, but may thrash on 4 GiB RAM.

### Caveats

- `moe_intermediate_size = 512` confirmed in config.json.
- Live set is an interval, not a point estimate. Actual value depends on `p_window_10` for this model, which is not yet measured.
- If `p_window_10 ≈ 0.9`, live set likely 20–40 experts (60–120 MiB).
- If `p_window_10 ≈ 0.5`, live set approaches the upper bound (90 experts, 270 MiB).
- The 23-vs-24 layer count depends on `first_k_dense_replace = 1`. Verify against config.json before quoting the cumulative numbers.

## Qwen3.5-35B-A3B

```python
# Config from HF. num_moe_layers estimated, needs config.json verification.
num_experts = 256
num_experts_per_tok = 8
num_shared_experts = 1
hidden_size = 5120
moe_intermediate_size = 14336
num_moe_layers = 48                 # ESTIMATED

# Expert size (BF16)
params_per_expert = 2 * 5120 * 14336          # 146,800,640 params
expert_size_bytes = 146800640 * 2             # 293,601,280 bytes = 280 MiB

# Per-token traversal, per layer
per_token_traversal_per_layer = 9 * 293601280 # 2,642,411,520 bytes = 2.46 GiB

# Live set: interval (not measured)
# Lower bound: 9 experts
# Upper bound: min(10 * 9, 256 + 1) = 90 experts
live_set_per_layer_lower =  9 * 293601280     #  2,642,411,520 bytes =  2.46 GiB
live_set_per_layer_upper = 90 * 293601280     # 26,424,115,200 bytes = 24.61 GiB

# Cumulative live set over all 48 MoE layers
cumulative_live_set_lower =  2642411520 * 48  #   126,835,752,960 bytes =  118.1 GiB
cumulative_live_set_upper = 26424115200 * 48  # 1,268,357,529,600 bytes = 1153.2 GiB ≈ 1.13 TiB
```

### Result

- Per-token traversal per layer: **2.46 GiB**
- Live set per layer: **[2.46 GiB, 24.6 GiB]** (interval, not measured)
- Cumulative live set: **[118 GiB, 1.13 TiB]** (interval, not measured)
- Apple SLC: **8–96 MiB**
- RAM: **16–128 GiB** on high-end Macs
- **Conclusion:** Does not fit in SLC or RAM on any current Mac. Storage IS the bottleneck; cold reads are paid per token or per batch, not once per layer.

### Caveats

- `num_moe_layers = 48` estimated. Needs verification against Qwen/Qwen3.5-35B-A3B config.json.
- Live set not measured. A trace is needed for any point estimate inside the interval.
- On a machine that cannot hold the interval lower bound in RAM (2.46 GiB per layer), the interval math still holds, but the practical cost is dominated by the working set of whatever subset is resident.

## Summary Table

| Model | Per-token Traversal / Layer | Live Set / Layer (W=10) | Cumulative Live Set | Fits in SLC 8 MiB (M1)? | Fits in SLC 48 MiB (M1 Max)? | Fits in RAM 8 GiB? | Storage Bottleneck? |
|-------|-----------------------------|--------------------------|---------------------|--------------------------|-------------------------------|---------------------|---------------------|
| TinyMoE-100m | 4 MiB | ~6 MiB (estimate, top-3) | ~60 MiB (estimate) | ❌ | ⚠️ (tight, SLC contention) | ✅ | No, likely fits in M1 Max SLC |
| Ling-3.0-tiny | 27 MiB | [27 MiB, 270 MiB] (not measured) | [0.61 GiB, 6.06 GiB] (not measured) | ❌ | ❌ | ⚠️ (tight on 4 GiB) | No, fits in RAM on 8+ GiB |
| Qwen3.5-35B-A3B | 2.46 GiB | [2.46 GiB, 24.6 GiB] (not measured) | [118 GiB, 1.13 TiB] (not measured) | ❌ | ❌ | ❌ | **Yes** |

## Apple SLC Numbers

| Chip | SLC | Notes |
|------|-----|-------|
| M1 | 8 MiB | Base model |
| M1 Pro | 24 MiB | |
| M1 Max | 48 MiB | |
| M1 Ultra | 96 MiB | 2×¹ M1 Max |
| M2 | 8 MiB | Base model |
| M2 Pro | 24 MiB | |
| M2 Max | 48 MiB | Not officially documented in all sources |
| M3 Max | 96 MiB | |

**Note:** Apple does not use the term "L3 cache" for M-series. The SLC
is shared between CPU, GPU, Neural Engine, DMA, and video encoders.
Effective capacity for a single CPU process is likely 50–75% of the
documented total. This applies to the "Fits in SLC?" column in the
summary table above.

## Implications

### Small MoE (TinyMoE-100m)

- Live set per layer estimated at ~6 MiB, likely fits in M1 Max SLC (48 MiB) with headroom, does not comfortably fit in M1/M2 base SLC (8 MiB).
- Cold reads paid once per layer over the window, not per token.
- `p_window_10 ≈ 0.98` measured on the 1000-token trace — this is the actual measurement available. The live-set-over-W=10 count is not yet computed but bounded from above by the number of distinct experts in the full trace (7).
- Storage optimizations (packed layout, fd-cache, parallel reads) have minimal impact on M1 Max and larger.

### Medium MoE (Ling-3.0-tiny)

- Live set per layer in [27 MiB, 270 MiB], exceeds SLC on every Mac, fits in RAM on 8+ GiB machines.
- Cold reads paid once per layer over the window if RAM is sufficient. May thrash on 4 GiB machines.
- `p_window_10` predicted high (0.8–0.95), not measured. Trace extraction script is ready.
- Storage optimization may matter moderately on 4 GiB RAM, minimally on 8+ GiB.

### Large MoE (Qwen3.5-35B-A3B, Mixtral 8x7B)

- Live set per layer in [2.46 GiB, 24.6 GiB], exceeds both SLC and RAM on consumer machines.
- Cold reads paid per token or per batch.
- `p_window_10` expected lower than small MoE (0.3–0.7 by analogy, not measured).
- Storage optimization is critical for inference latency.

## Next Steps

1. **Compute live_set over W=10 for TinyMoE.** The snippet above runs in seconds against the existing routing-trace.jsonl. This removes the last "estimate" from the small-MoE row.
2. **Run Ling trace extraction in Colab.** Script is ready. Gives measured `p_window_10` and the actual live set interval endpoint.
3. **Verify Qwen3.5 config.json** to confirm or correct `num_moe_layers = 48` and `moe_intermediate_size`.
4. **Optional: trace on Qwen3.5-35B-A3B or Mixtral 8x7B.** Needed only if a point estimate inside the large-MoE interval is required for a decision. The interval alone is sufficient for the "storage is a bottleneck" classification.
5. **Update falsification.md:** Q2 is now partially answered by the working-set criterion. Small MoE: not a bottleneck. Medium MoE: borderline. Large MoE: bottleneck. Reframe Q2 around the live-set vs SLC/RAM comparison.

## Caveats

1. **TinyMoE live_set over W=10 not yet computed.** Top-3 frequency (3 experts) is not the same statistic as distinct experts per window. The snippet above closes this.
2. **Ling and Qwen3.5 numbers are intervals, not point estimates.** Actual values depend on routing patterns that have not been measured for these models.
3. **SLC contention.** SLC is shared. Effective capacity for a CPU process is likely 50–75% of the documented total.
4. **BF16 assumed. INT4 reduces expert size by 4×¹; INT8 by 2×¹.** The arithmetic scales linearly. If the target deployment uses INT4, all live-set numbers shrink by a factor of 4 and the "Fits in SLC?" column changes accordingly.
5. **Qwen3.5 num_moe_layers = 48 estimated.** Needs config.json verification before the cumulative numbers are quoted.
6. **Units are binary (MiB/GiB/TiB) throughout.** Byte counts in code snippets are exact. Do not read "27 MiB" as 27,000,000 bytes.

## References

- TinyMoE config: `FlameF0X/TinyMoE-100m-2x8`
- Ling-3.0-tiny config: `inclusionAI/Ling-3.0-tiny`
- Qwen3.5-35B-A3B config: `Qwen/Qwen3.5-35B-A3B`
- Apple Silicon cache sizes: Apple Silicon CPU Optimization Guide (Tables 3-1, 3-2)
- Routing trace and analysis methodology: `docs/falsification.md`, Phase B reports
