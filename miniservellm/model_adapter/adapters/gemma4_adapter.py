"""Gemma 4 E2B text-only model adapter.

The adapter intentionally keeps Gemma4 weights separate from the Qwen2
``TransformerWeights`` layout. E2B has PLE, per-layer attention shapes and
additional normalization layers, so flattening it into the Qwen structure
would lose information needed by the eventual Gemma4 text runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from miniservellm.config import ModelConfig
from miniservellm.model_adapter.gemma4_config import (
    Gemma4LayerSpec,
    convert_gemma4_config,
    get_text_config,
)


def _import_transformers():
    try:
        import transformers  # type: ignore
        return transformers
    except ImportError as exc:
        raise ImportError("Please install transformers: pip install transformers") from exc


def _clone_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to("cpu").contiguous()


def _module(root: Any, *paths: str) -> Any:
    for path in paths:
        current = root
        try:
            for part in path.split("."):
                current = current[int(part)] if part.isdigit() else getattr(current, part)
            return current
        except (AttributeError, IndexError, KeyError, TypeError):
            continue
    raise AttributeError(f"Could not find any module path: {paths}")


def _weight(root: Any, *paths: str) -> torch.Tensor:
    module = _module(root, *paths)
    if not hasattr(module, "weight"):
        raise AttributeError(f"Module has no weight: {paths}")
    return _clone_cpu(module.weight)


def _optional_weight(root: Any, *paths: str) -> Optional[torch.Tensor]:
    try:
        module = _module(root, *paths)
    except AttributeError:
        return None
    weight = getattr(module, "weight", None)
    return _clone_cpu(weight) if weight is not None else None


@dataclass
class Gemma4LayerWeights:
    q_proj: torch.Tensor
    k_proj: torch.Tensor
    v_proj: torch.Tensor
    o_proj: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    v_norm: Optional[torch.Tensor]
    input_layernorm: torch.Tensor
    post_attention_layernorm: torch.Tensor
    pre_feedforward_layernorm: torch.Tensor
    post_feedforward_layernorm: torch.Tensor
    post_per_layer_input_norm: torch.Tensor
    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor
    per_layer_input_gate: torch.Tensor
    per_layer_projection: torch.Tensor
    layer_spec: Gemma4LayerSpec
    layer_scalar: Optional[torch.Tensor] = None


@dataclass
class Gemma4TextWeights:
    embed_tokens: torch.Tensor
    lm_head: torch.Tensor
    final_norm: torch.Tensor
    embed_tokens_per_layer: torch.Tensor
    per_layer_model_projection: torch.Tensor
    per_layer_projection_norm: torch.Tensor
    layers: list[Gemma4LayerWeights]


class Gemma4Adapter:
    """Load and extract the text backbone of a Gemma4 checkpoint."""

    def load_tokenizer(self, model_name_or_path: str, trust_remote_code: bool = False) -> Any:
        transformers = _import_transformers()
        # AutoProcessor is needed by the full multimodal checkpoint, while the
        # text-only runner only consumes its tokenizer component.
        try:
            return transformers.AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
                local_files_only=True,
            )
        except (OSError, ValueError):
            processor = transformers.AutoProcessor.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
                local_files_only=True,
            )
            if not hasattr(processor, "tokenizer"):
                raise ValueError("Gemma4 processor does not expose a tokenizer")
            return processor.tokenizer

    def load_hf_config(self, model_name_or_path: str, trust_remote_code: bool = False) -> Any:
        transformers = _import_transformers()
        return transformers.AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )

    def convert_hf_config(self, hf_config: Any) -> ModelConfig:
        return convert_gemma4_config(hf_config)

    def load_hf_model(
        self,
        model_name_or_path: str,
        device: str = "cpu",
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ) -> Any:
        transformers = _import_transformers()
        kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": True,
        }
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        # E2B is dense. AutoModelForCausalLM selects Gemma4ForCausalLM for a
        # text-only checkpoint and avoids loading vision/audio modules.
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **kwargs,
        )
        model.to(device)
        model.eval()
        return model

    @staticmethod
    def _text_root(hf_model: Any) -> Any:
        candidates = (
            "model.language_model",
            "language_model",
            "model",
            "base_model.model",
        )
        for path in candidates:
            try:
                return _module(hf_model, path)
            except AttributeError:
                continue
        raise AttributeError("Could not locate Gemma4 text model root")

    def extract_weights(self, hf_model: Any) -> Gemma4TextWeights:
        root = self._text_root(hf_model)
        config = getattr(hf_model, "config", None)
        if config is None:
            raise ValueError("Gemma4 HF model has no config")
        model_config = convert_gemma4_config(config)

        embed_tokens = _weight(root, "embed_tokens")
        final_norm = _weight(root, "norm")
        lm_head_module = _module(hf_model, "lm_head", "model.lm_head")
        lm_head = _clone_cpu(lm_head_module.weight)

        ple_table = _weight(root, "embed_tokens_per_layer")
        ple_projection = _weight(root, "per_layer_model_projection")
        ple_projection_norm = _weight(root, "per_layer_projection_norm")

        layers = _module(root, "layers")
        extracted_layers: list[Gemma4LayerWeights] = []
        for idx, spec in enumerate(model_config.layer_specs):
            layer = layers[idx]
            extracted_layers.append(
                Gemma4LayerWeights(
                    q_proj=_weight(layer, "self_attn.q_proj"),
                    k_proj=_weight(layer, "self_attn.k_proj"),
                    v_proj=_weight(layer, "self_attn.v_proj"),
                    o_proj=_weight(layer, "self_attn.o_proj"),
                    q_norm=_weight(layer, "self_attn.q_norm"),
                    k_norm=_weight(layer, "self_attn.k_norm"),
                    v_norm=_optional_weight(layer, "self_attn.v_norm"),
                    input_layernorm=_weight(layer, "input_layernorm"),
                    post_attention_layernorm=_weight(layer, "post_attention_layernorm"),
                    pre_feedforward_layernorm=_weight(layer, "pre_feedforward_layernorm"),
                    post_feedforward_layernorm=_weight(layer, "post_feedforward_layernorm"),
                    post_per_layer_input_norm=_weight(layer, "post_per_layer_input_norm"),
                    gate_proj=_weight(layer, "mlp.gate_proj"),
                    up_proj=_weight(layer, "mlp.up_proj"),
                    down_proj=_weight(layer, "mlp.down_proj"),
                    per_layer_input_gate=_weight(layer, "per_layer_input_gate"),
                    per_layer_projection=_weight(layer, "per_layer_projection"),
                    layer_spec=spec,
                    layer_scalar=_clone_cpu(layer.layer_scalar)
                    if hasattr(layer, "layer_scalar") else None,
                )
            )

        return Gemma4TextWeights(
            embed_tokens=embed_tokens,
            lm_head=lm_head,
            final_norm=final_norm,
            embed_tokens_per_layer=ple_table,
            per_layer_model_projection=ple_projection,
            per_layer_projection_norm=ple_projection_norm,
            layers=extracted_layers,
        )

    def move_weights_to_device(
        self,
        weights: Gemma4TextWeights,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Gemma4TextWeights:
        def move(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if tensor is None else tensor.to(device=device, dtype=dtype).contiguous()

        def move_layer(layer: Gemma4LayerWeights) -> Gemma4LayerWeights:
            values = {
                name: move(getattr(layer, name))
                for name in (
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "q_norm", "k_norm", "v_norm",
                    "input_layernorm", "post_attention_layernorm",
                    "pre_feedforward_layernorm", "post_feedforward_layernorm",
                    "post_per_layer_input_norm",
                    "gate_proj", "up_proj", "down_proj",
                    "per_layer_input_gate", "per_layer_projection",
                    "layer_scalar",
                )
            }
            return Gemma4LayerWeights(layer_spec=layer.layer_spec, **values)

        return Gemma4TextWeights(
            embed_tokens=move(weights.embed_tokens),
            lm_head=move(weights.lm_head),
            final_norm=move(weights.final_norm),
            embed_tokens_per_layer=move(weights.embed_tokens_per_layer),
            per_layer_model_projection=move(weights.per_layer_model_projection),
            per_layer_projection_norm=move(weights.per_layer_projection_norm),
            layers=[move_layer(layer) for layer in weights.layers],
        )
