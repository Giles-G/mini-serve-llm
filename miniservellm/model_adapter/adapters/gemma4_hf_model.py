"""Direct safetensors access for the Gemma4 text backbone.

The official checkpoint stores the text trunk under
``model.language_model.*`` next to vision/audio tensors. Loading the full
multimodal module would waste memory, so this module maps checkpoint keys
straight into :class:`Gemma4TextWeights` without instantiating any
``torch.nn.Module`` tree. Tensors are moved to the target device as they are
read, which keeps peak memory at roughly one copy of the text weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from safetensors import safe_open

from miniservellm.model_adapter.adapters.gemma4_adapter import (
    Gemma4LayerWeights,
    Gemma4TextWeights,
)
from miniservellm.model_adapter.gemma4_config import build_gemma4_layer_specs

TEXT_PREFIX = "model.language_model."


def load_gemma4_config(model_dir: str | Path) -> SimpleNamespace:
    """Load ``config.json`` into nested namespaces (outer + ``text_config``)."""
    config_path = Path(model_dir).expanduser() / "config.json"
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if "text_config" in raw and isinstance(raw["text_config"], dict):
        raw["text_config"] = SimpleNamespace(**raw["text_config"])
    return SimpleNamespace(**raw)


def _read(weights_file: Any, key: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = weights_file.get_tensor(key)
    return tensor.to(device=device, dtype=dtype)


def load_gemma4_text_weights(
    model_dir: str | Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> Gemma4TextWeights:
    """Extract the text backbone into :class:`Gemma4TextWeights`.

    ``lm_head`` is tied in the E2B checkpoint, so it shares storage with
    ``embed_tokens`` instead of allocating a second full-vocab tensor.
    """
    path = Path(model_dir).expanduser()
    device = torch.device(device)
    specs = build_gemma4_layer_specs(load_gemma4_config(path))
    weights_file = safe_open(str(path / "model.safetensors"), framework="pt")

    def read(key: str) -> torch.Tensor:
        return _read(weights_file, f"{TEXT_PREFIX}{key}", device, dtype)

    embed_tokens = read("embed_tokens.weight")
    layers: list[Gemma4LayerWeights] = []
    for spec in specs:
        idx = spec.layer_idx
        layers.append(
            Gemma4LayerWeights(
                q_proj=read(f"layers.{idx}.self_attn.q_proj.weight"),
                k_proj=read(f"layers.{idx}.self_attn.k_proj.weight"),
                v_proj=read(f"layers.{idx}.self_attn.v_proj.weight"),
                o_proj=read(f"layers.{idx}.self_attn.o_proj.weight"),
                q_norm=read(f"layers.{idx}.self_attn.q_norm.weight"),
                k_norm=read(f"layers.{idx}.self_attn.k_norm.weight"),
                v_norm=None,
                input_layernorm=read(f"layers.{idx}.input_layernorm.weight"),
                post_attention_layernorm=read(f"layers.{idx}.post_attention_layernorm.weight"),
                pre_feedforward_layernorm=read(f"layers.{idx}.pre_feedforward_layernorm.weight"),
                post_feedforward_layernorm=read(f"layers.{idx}.post_feedforward_layernorm.weight"),
                post_per_layer_input_norm=read(f"layers.{idx}.post_per_layer_input_norm.weight"),
                gate_proj=read(f"layers.{idx}.mlp.gate_proj.weight"),
                up_proj=read(f"layers.{idx}.mlp.up_proj.weight"),
                down_proj=read(f"layers.{idx}.mlp.down_proj.weight"),
                per_layer_input_gate=read(f"layers.{idx}.per_layer_input_gate.weight"),
                per_layer_projection=read(f"layers.{idx}.per_layer_projection.weight"),
                layer_spec=spec,
                layer_scalar=read(f"layers.{idx}.layer_scalar"),
            )
        )

    return Gemma4TextWeights(
        embed_tokens=embed_tokens,
        # tie_word_embeddings: the checkpoint has no independent lm_head tensor.
        lm_head=embed_tokens,
        final_norm=read("norm.weight"),
        embed_tokens_per_layer=read("embed_tokens_per_layer.weight"),
        per_layer_model_projection=read("per_layer_model_projection.weight"),
        per_layer_projection_norm=read("per_layer_projection_norm.weight"),
        layers=layers,
    )
