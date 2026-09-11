#!/usr/bin/env python3
"""
Extract routing trace from FlameF0X/TinyMoE-100m-2x8
Outputs JSONL compatible with forge bench-read --trace
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

MODEL_ID = "FlameF0X/TinyMoE-100m-2x8"
OUTPUT = "routing-trace-tinymoe-real.jsonl"
NUM_TOKENS = 2000
PROMPT = "The future of sparse mixture of experts models is"

def main():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    traces = defaultdict(list)
    token_counter = 0

    def make_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            topk = torch.topk(logits, k=2, dim=-1)
            experts = topk.indices.detach().cpu()
            for pos in range(experts.shape[1]):
                exp_list = sorted(experts[0, pos].tolist())
                traces[layer_idx].append((token_counter + pos, exp_list))
        return hook

    hooks = []
    for name, module in model.named_modules():
        if "router" in name.lower() or "gate" in name.lower():
            parts = name.split(".")
            layer_idx = None
            for p in parts:
                if p.isdigit():
                    layer_idx = int(p)
                    break
            if layer_idx is not None:
                print(f"Hooking layer {layer_idx}: {name}")
                h = module.register_forward_hook(make_hook(layer_idx))
                hooks.append(h)

    if not hooks:
        print("WARNING: no router modules found. Check model structure.")

    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)

    print(f"Generating {NUM_TOKENS} tokens...")
    with torch.no_grad():
        generated = inputs["input_ids"]
        for step in range(NUM_TOKENS):
            outputs = model(generated)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            token_counter += 1
            if (step + 1) % 200 == 0:
                print(f"  {step+1}/{NUM_TOKENS}")

    for h in hooks:
        h.remove()

    print(f"Writing {OUTPUT}...")
    with open(OUTPUT, "w") as f:
        all_records = []
        for layer, items in traces.items():
            for tok_seq, experts in items:
                all_records.append({
                    "token_seq": tok_seq,
                    "layer": layer,
                    "experts": experts
                })
        all_records.sort(key=lambda x: (x["token_seq"], x["layer"]))
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    print(f"Done. Wrote {len(all_records)} records to {OUTPUT}")

if __name__ == "__main__":
    main()
