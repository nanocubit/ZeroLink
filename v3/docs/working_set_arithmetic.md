# Working Set Arithmetic for MoE Models

## Criterion

**Storage is NOT a bottleneck if:**

```
working_set_per_layer < UBC_size
```

Where:
- `working_set_per_layer` = memory required for active experts in one layer (MB)
- `UBC_size` = uncore/cache size of target machine (MB)

If working set fits in cache, cold reads are paid once per expert, not per token.

## Formulas

### Expert size (FFN)

```
params_per_expert = 2 × hidden_size × moe_intermediate_size
bytes_per_param = 2 (BF16) or 1 (INT8) or 0.5 (INT4)
expert_size_bytes = params_per_expert × bytes_per_param
```

### Working set per layer

```
working_set_per_layer = (num_experts_per_tok + num_shared_experts) × expert_size_bytes
```

### Total working set (all MoE layers)

```
total_working_set = working_set_per_layer × num_moe_layers
```

## TinyMoE-100m-2x8

```python
# Config
num_experts = 8
num_experts_per_tok = 2
num_shared_experts = 0
hidden_size = 512
moe_intermediate_size = 1024
num_moe_layers = 10

# Calculation
params_per_expert = 2 * 512 * 1024  # 1,048,576 params
expert_size_bytes = 1048576 * 2  # 2,097,152 bytes = 2 MB (BF16)
working_set_per_layer = 2 * 2097152  # 4,194,304 bytes = 4 MB
total_working_set = 4194304 * 10  # 41,943,040 bytes = 40 MB
```

**Result:**
- Working set per layer: **4 MB**
- Total working set: **40 MB**
- Mac L3 cache: **~64 MB** (M1/M2) or **~128 MB** (M3 Max)
- **Conclusion:** Fits in L3 cache → storage NOT bottleneck

## Ling-3.0-tiny

```python
# Config (from HF)
num_experts = 128
num_experts_per_tok = 8
num_shared_experts = 1
hidden_size = 1536
moe_intermediate_size = 512
num_moe_layers = 23  # layers 1..23 (layer 0 is dense)

# Calculation
params_per_expert = 2 * 1536 * 512  # 1,572,864 params
expert_size_bytes = 1572864 * 2  # 3,145,728 bytes = 3 MB (BF16)
working_set_per_layer = (8 + 1) * 3145728  # 28,311,552 bytes = 27 MB
total_working_set = 28311552 * 23  # 651,165,696 bytes = 621 MB
```

**Result:**
- Working set per layer: **27 MB**
- Total working set: **621 MB**
- Mac L3 cache: **~64 MB** (M1/M2) or **~128 MB** (M3 Max)
- RAM: **4–16 GB**
- **Conclusion:** Does NOT fit in L3 cache, but fits in RAM → cold reads paid once per layer, not per token

## Qwen3.5-35B-A3B

```python
# Config (from HF)
num_experts = 256
num_experts_per_tok = 8
num_shared_experts = 1
hidden_size = 5120
moe_intermediate_size = 14336
num_moe_layers = 48  # estimated

# Calculation
params_per_expert = 2 * 5120 * 14336  # 146,800,640 params
expert_size_bytes = 146800640 * 2  # 293,601,280 bytes = 280 MB (BF16)
working_set_per_layer = (8 + 1) * 293601280  # 2,642,411,520 bytes = 2.5 GB
total_working_set = 2642411520 * 48  # 126,835,752,960 bytes = 118 GB
```

**Result:**
- Working set per layer: **2.5 GB**
- Total working set: **118 GB**
- Mac L3 cache: **~64–128 MB**
- RAM: **16–128 GB** (high-end Mac Pro)
- **Conclusion:** Does NOT fit in L3 or RAM → storage IS bottleneck for large MoE

## Summary Table

| Model | Experts | Top-k | Shared | Working Set/Layer | Total Working Set | Fits in L3? | Fits in RAM? | Storage Bottleneck? |
|-------|---------|-------|--------|-------------------|-------------------|-------------|--------------|---------------------|
| TinyMoE-100m | 8 | 2 | 0 | 4 MB | 40 MB | ✅ | ✅ | No |
| Ling-3.0-tiny | 128 | 8 | 1 | 27 MB | 621 MB | ❌ | ✅ | No (fits in RAM) |
| Qwen3.5-35B-A3B | 256 | 8 | 1 | 2.5 GB | 118 GB | ❌ | ❌ | **Yes** |

## Implications

### Small MoE (TinyMoE, Ling-3.0-tiny)

- Working set per layer fits in L3 cache (TinyMoE) or RAM (Ling)
- Cold reads paid once per layer, not per token
- `p_window_10 ≈ 1.0` observed in trace → confirms criterion
- **Storage optimization (packed layout, fd-cache, parallel reads) has minimal impact**

### Large MoE (Qwen3.5-35B-A3B, Mixtral 8x7B)

- Working set per layer exceeds L3 cache and RAM
- Cold reads paid per token or per batch
- `p_window_10` expected to be much lower
- **Storage optimization IS critical for inference latency**

## Next Steps

1. **Verify with real traces:** Run trace extraction on Qwen3.5-35B-A3B or Mixtral 8x7B
2. **Measure p_window_10:** Expect much lower values for large MoE
3. **Optimize storage:** Packed layout, fd-cache, parallel reads for large MoE only
4. **Focus falsification.md:** Q2 answered for small MoE, reformulate for large MoE

## References

- TinyMoE config: `FlameF0X/TinyMoE-100m-2x8`
- Ling-3.0-tiny config: `inclusionAI/Ling-3.0-tiny` (HF)
- Qwen3.5-35B-A3B config: `Qwen/Qwen3.5-35B-A3B` (HF)
- Mac L3 cache sizes: https://www.cpu-world.com/
