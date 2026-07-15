"""Load Qwen2 HF safetensors directly into MLX arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import mlx.core as mx
import torch
from safetensors import safe_open

from miniservellm.config import ModelConfig
from miniservellm.mlx.qwen2 import MLXLayerWeights, MLXWeights


def _to_mx(tensor: torch.Tensor) -> mx.array:
    """Convert a CPU torch tensor to FP16 MLX storage."""
    return mx.array(tensor.float().numpy()).astype(mx.float16)


def load_qwen2_weights(model_path: str | Path, config: ModelConfig) -> MLXWeights:
    """Load Qwen2 weights from a local HF safetensors file.

    The source model is BF16; MLX backend intentionally stores FP16 to match the
    existing MPS inference benchmark.
    """
    model_path = str(model_path)
    tensors: Dict[str, torch.Tensor] = {}
    with safe_open(model_path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensors[name] = handle.get_tensor(name)

    def get(name: str) -> torch.Tensor:
        try:
            return tensors[name]
        except KeyError as exc:
            raise KeyError(f"Missing Qwen2 tensor: {name}") from exc

    layers = []
    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}"
        q = get(f"{prefix}.self_attn.q_proj.weight")
        k = get(f"{prefix}.self_attn.k_proj.weight")
        v = get(f"{prefix}.self_attn.v_proj.weight")
        q_bias = get(f"{prefix}.self_attn.q_proj.bias")
        k_bias = get(f"{prefix}.self_attn.k_proj.bias")
        v_bias = get(f"{prefix}.self_attn.v_proj.bias")
        gate = get(f"{prefix}.mlp.gate_proj.weight")
        up = get(f"{prefix}.mlp.up_proj.weight")
        layers.append(
            MLXLayerWeights(
                qkv_proj=_to_mx(torch.cat([q, k, v], dim=0)),
                qkv_bias=_to_mx(torch.cat([q_bias, k_bias, v_bias], dim=0)),
                o_proj=_to_mx(get(f"{prefix}.self_attn.o_proj.weight")),
                gate_up_proj=_to_mx(torch.cat([gate, up], dim=0)),
                down_proj=_to_mx(get(f"{prefix}.mlp.down_proj.weight")),
                input_layernorm=_to_mx(get(f"{prefix}.input_layernorm.weight")),
                post_attention_layernorm=_to_mx(get(f"{prefix}.post_attention_layernorm.weight")),
            )
        )

    embed = _to_mx(get("model.embed_tokens.weight"))
    return MLXWeights(
        embed_tokens=embed,
        layers=layers,
        final_norm=_to_mx(get("model.norm.weight")),
        lm_head=embed,
    )
