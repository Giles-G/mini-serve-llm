"""Normalized Gemma 4 text configuration.

This module deliberately keeps Gemma 4's per-layer structure explicit instead
of flattening it into the Qwen2-style scalar ``ModelConfig`` fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from miniservellm.config import ModelConfig


@dataclass(frozen=True)
class Gemma4LayerSpec:
    layer_idx: int
    attention_type: str
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rotary_dim: int
    rope_type: str
    rope_theta: float
    sliding_window: Optional[int]
    kv_source_layer: Optional[int]
    intermediate_size: int
    use_double_wide_mlp: bool

    @property
    def is_sliding(self) -> bool:
        return self.attention_type == "sliding_attention"

    @property
    def is_full(self) -> bool:
        return self.attention_type == "full_attention"


def _get(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def get_text_config(hf_config: Any) -> Any:
    """Return the nested text config used by Gemma4 multimodal checkpoints."""
    text_config = _get(hf_config, "text_config")
    return text_config if text_config is not None else hf_config


def _rope_parameters(text_config: Any, attention_type: str) -> dict[str, Any]:
    rope_parameters = _get(text_config, "rope_parameters", {}) or {}
    if not isinstance(rope_parameters, dict):
        raise TypeError("Gemma4 rope_parameters must be a mapping")

    # New Gemma4 configs store one entry per attention type. Accept the
    # single-dict form as a compatibility fallback for text-only checkpoints.
    selected = rope_parameters.get(attention_type, rope_parameters)
    if not isinstance(selected, dict):
        raise TypeError(f"Invalid RoPE parameters for {attention_type}")
    return selected


def _layer_types(text_config: Any) -> list[str]:
    layer_types = _get(text_config, "layer_types")
    if layer_types is None:
        pattern = int(_get(text_config, "sliding_window_pattern", 6))
        layers = int(_get(text_config, "num_hidden_layers"))
        layer_types = [
            "full_attention" if (i + 1) % pattern == 0 else "sliding_attention"
            for i in range(layers)
        ]
        if layer_types:
            layer_types[-1] = "full_attention"
    layer_types = list(layer_types)
    if not layer_types:
        raise ValueError("Gemma4 layer_types must not be empty")
    return layer_types


def _per_layer_config(text_config: Any) -> dict[int, Any]:
    values = _get(text_config, "per_layer_config", {}) or {}
    if isinstance(values, dict):
        return {int(k): v for k, v in values.items()}
    return {}


def build_gemma4_layer_specs(hf_config: Any) -> list[Gemma4LayerSpec]:
    text_config = get_text_config(hf_config)
    layer_types = _layer_types(text_config)
    num_layers = int(_get(text_config, "num_hidden_layers"))
    if len(layer_types) != num_layers:
        raise ValueError(
            f"Gemma4 layer_types length {len(layer_types)} != num_hidden_layers {num_layers}"
        )

    hidden_size = int(_get(text_config, "hidden_size"))
    default_q_heads = int(_get(text_config, "num_attention_heads"))
    default_kv_heads = int(_get(text_config, "num_key_value_heads", default_q_heads))
    default_head_dim = int(
        _get(text_config, "head_dim", hidden_size // default_q_heads)
    )
    global_head_dim = int(_get(text_config, "global_head_dim", default_head_dim))
    global_kv_heads = _get(text_config, "num_global_key_value_heads")
    base_intermediate = int(_get(text_config, "intermediate_size"))
    wide = bool(_get(text_config, "use_double_wide_mlp", False))
    sliding_window = _get(text_config, "sliding_window")
    overrides = _per_layer_config(text_config)
    shared_count = int(_get(text_config, "num_kv_shared_layers", 0) or 0)

    specs: list[Gemma4LayerSpec] = []
    for idx, attention_type in enumerate(layer_types):
        if attention_type not in {"sliding_attention", "full_attention"}:
            raise ValueError(f"Unsupported Gemma4 attention type: {attention_type}")

        override = overrides.get(idx, {})
        head_dim = int(
            _get(
                override,
                "head_dim",
                global_head_dim if attention_type == "full_attention" else default_head_dim,
            )
        )
        kv_heads = int(
            _get(
                override,
                "num_key_value_heads",
                global_kv_heads
                if attention_type == "full_attention" and global_kv_heads is not None
                else default_kv_heads,
            )
        )
        q_heads = int(_get(override, "num_attention_heads", default_q_heads))
        if q_heads % kv_heads != 0:
            raise ValueError(f"Layer {idx}: query heads must divide KV heads")

        rope = _rope_parameters(text_config, attention_type)
        rotary_factor = float(_get(rope, "partial_rotary_factor", 1.0))
        rotary_dim = int(head_dim * rotary_factor)

        # The E-series uses wider FFN projections in the KV-shared tail.
        first_shared = num_layers - shared_count if shared_count > 0 else num_layers
        is_shared_tail = idx >= first_shared and idx > 0
        layer_intermediate = base_intermediate * (2 if wide and is_shared_tail else 1)
        # The exact source-layer mapping is an execution/cache concern and
        # must be derived from the official Gemma4 implementation. Do not
        # infer it from the count alone at config-normalization time.
        kv_source = None

        specs.append(
            Gemma4LayerSpec(
                layer_idx=idx,
                attention_type=attention_type,
                num_attention_heads=q_heads,
                num_key_value_heads=kv_heads,
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                rope_type=str(_get(rope, "rope_type", "default")),
                rope_theta=float(_get(rope, "rope_theta", 10000.0)),
                sliding_window=int(sliding_window) if attention_type == "sliding_attention" and sliding_window else None,
                kv_source_layer=kv_source,
                intermediate_size=layer_intermediate,
                use_double_wide_mlp=wide and is_shared_tail,
            )
        )
    return specs


def convert_gemma4_config(hf_config: Any) -> ModelConfig:
    text_config = get_text_config(hf_config)
    specs = build_gemma4_layer_specs(hf_config)
    first = specs[0]
    return ModelConfig(
        model_type="gemma4",
        vocab_size=int(_get(text_config, "vocab_size")),
        hidden_size=int(_get(text_config, "hidden_size")),
        intermediate_size=int(_get(text_config, "intermediate_size")),
        num_hidden_layers=int(_get(text_config, "num_hidden_layers")),
        num_attention_heads=int(_get(text_config, "num_attention_heads")),
        num_key_value_heads=int(_get(text_config, "num_key_value_heads", first.num_key_value_heads)),
        head_dim=first.head_dim,
        max_position_embeddings=int(_get(text_config, "max_position_embeddings", 131072)),
        rope_theta=first.rope_theta,
        rms_norm_eps=float(_get(text_config, "rms_norm_eps", 1e-6)),
        hidden_activation=str(_get(text_config, "hidden_activation", "gelu_pytorch_tanh")),
        tie_word_embeddings=bool(_get(text_config, "tie_word_embeddings", True)),
        final_logit_softcapping=_get(text_config, "final_logit_softcapping"),
        sliding_window=_get(text_config, "sliding_window"),
        hidden_size_per_layer_input=int(_get(text_config, "hidden_size_per_layer_input", 0) or 0),
        vocab_size_per_layer_input=int(_get(text_config, "vocab_size_per_layer_input", 0) or 0),
        num_kv_shared_layers=int(_get(text_config, "num_kv_shared_layers", 0) or 0),
        use_double_wide_mlp=bool(_get(text_config, "use_double_wide_mlp", False)),
        layer_specs=specs,
    )
