"""Stage B validation: load the Gemma4 checkpoint and verify every text weight.

Loads only ``model.language_model.*`` tensors from the safetensors file and
checks every shape against the config-derived layer specs, plus the top-level
PLE tensors and the tied lm_head. Exit code 0 means all shapes match.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from miniservellm.model_adapter.adapters.gemma4_hf_model import (
    load_gemma4_config,
    load_gemma4_text_weights,
)
from miniservellm.model_adapter.gemma4_config import build_gemma4_layer_specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models" / "gemma-4-E2B-it"))
    args = parser.parse_args()
    model_dir = Path(args.model).expanduser()

    config = load_gemma4_config(model_dir)
    specs = build_gemma4_layer_specs(config)
    sliding = [s for s in specs if s.is_sliding]
    full = [s for s in specs if s.is_full]
    print(f"checkpoint={model_dir}")
    print(f"layers={len(specs)} sliding={len(sliding)} full={len(full)}")
    print(f"full_layer_indices={[s.layer_idx for s in full]}")
    print(f"head_dims: sliding={sliding[0].head_dim} full={full[0].head_dim}")
    print(
        f"rope: sliding theta={sliding[0].rope_theta} type={sliding[0].rope_type} "
        f"rotary_dim={sliding[0].rotary_dim}; "
        f"full theta={full[0].rope_theta} type={full[0].rope_type} rotary_dim={full[0].rotary_dim}"
    )

    weights = load_gemma4_text_weights(
        model_dir,
        device="cpu",
        dtype=torch.bfloat16,
    )

    failures: list[str] = []

    def check(name: str, actual: tuple[int, ...], expected: tuple[int, ...]) -> None:
        status = "ok" if actual == expected else "MISMATCH"
        print(f"{name:60s} {str(actual):20s} {status}")
        if actual != expected:
            failures.append(name)

    hidden = 1536
    check("embed_tokens", tuple(weights.embed_tokens.shape), (262144, hidden))
    check("lm_head (tied storage)", tuple(weights.lm_head.shape), (262144, hidden))
    check("final_norm", tuple(weights.final_norm.shape), (hidden,))
    check("embed_tokens_per_layer", tuple(weights.embed_tokens_per_layer.shape), (262144, len(specs) * 256))
    check("per_layer_model_projection", tuple(weights.per_layer_model_projection.shape), (len(specs) * 256, hidden))
    check("per_layer_projection_norm", tuple(weights.per_layer_projection_norm.shape), (256,))

    for lw, spec in zip(weights.layers, specs):
        idx = spec.layer_idx
        prefix = f"layers.{idx}"
        q_width = spec.num_attention_heads * spec.head_dim
        kv_width = spec.num_key_value_heads * spec.head_dim
        wide = 2 if spec.intermediate_size == 12288 else 1
        expected_inter = 6144 * wide
        check(f"{prefix}.q_proj", tuple(lw.q_proj.shape), (q_width, hidden))
        check(f"{prefix}.k_proj", tuple(lw.k_proj.shape), (kv_width, hidden))
        check(f"{prefix}.v_proj", tuple(lw.v_proj.shape), (kv_width, hidden))
        check(f"{prefix}.o_proj", tuple(lw.o_proj.shape), (hidden, q_width))
        check(f"{prefix}.q_norm", tuple(lw.q_norm.shape), (spec.head_dim,))
        check(f"{prefix}.k_norm", tuple(lw.k_norm.shape), (spec.head_dim,))
        if lw.v_norm is not None:
            failures.append(f"{prefix}.v_norm should be None (official v_norm is scale-free)")
        check(f"{prefix}.input_layernorm", tuple(lw.input_layernorm.shape), (hidden,))
        check(f"{prefix}.post_attention_layernorm", tuple(lw.post_attention_layernorm.shape), (hidden,))
        check(f"{prefix}.pre_feedforward_layernorm", tuple(lw.pre_feedforward_layernorm.shape), (hidden,))
        check(f"{prefix}.post_feedforward_layernorm", tuple(lw.post_feedforward_layernorm.shape), (hidden,))
        check(f"{prefix}.post_per_layer_input_norm", tuple(lw.post_per_layer_input_norm.shape), (hidden,))
        check(f"{prefix}.gate_proj", tuple(lw.gate_proj.shape), (expected_inter, hidden))
        check(f"{prefix}.up_proj", tuple(lw.up_proj.shape), (expected_inter, hidden))
        check(f"{prefix}.down_proj", tuple(lw.down_proj.shape), (hidden, expected_inter))
        check(f"{prefix}.per_layer_input_gate", tuple(lw.per_layer_input_gate.shape), (256, hidden))
        check(f"{prefix}.per_layer_projection", tuple(lw.per_layer_projection.shape), (hidden, 256))
        check(f"{prefix}.layer_scalar", tuple(lw.layer_scalar.shape), (1,))

    shared_tail = [s.layer_idx for s in specs if s.intermediate_size == 12288]
    print(f"\ndouble_wide_mlp_layers: {len(shared_tail)} [{shared_tail[0]}..{shared_tail[-1]}]")
    total_params = sum(t.numel() for t in [
        weights.embed_tokens,
        weights.embed_tokens_per_layer,
        weights.per_layer_model_projection,
        weights.per_layer_projection_norm,
        weights.final_norm,
    ]) + sum(
        getattr(lw, name).numel()
        for lw in weights.layers
        for name in (
            "q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm",
            "input_layernorm", "post_attention_layernorm",
            "pre_feedforward_layernorm", "post_feedforward_layernorm",
            "post_per_layer_input_norm", "gate_proj", "up_proj", "down_proj",
            "per_layer_input_gate", "per_layer_projection", "layer_scalar",
        )
    )
    print(f"text_params={total_params / 1e9:.2f}B")

    if failures:
        print(f"\nFAILED: {len(failures)} mismatches")
        for item in failures[:30]:
            print(f"  - {item}")
        raise SystemExit(1)
    print("\nALL SHAPES OK")


if __name__ == "__main__":
    main()
