"""Model adapter selection based on Hugging Face model_type."""

from __future__ import annotations

from typing import Any

from miniservellm.model_adapter.adapters.gemma4_adapter import Gemma4Adapter
from miniservellm.model_adapter.adapters.qwen2_adapter import Qwen2Adapter


def _model_type(config: Any) -> str:
    value = config.get("model_type") if isinstance(config, dict) else getattr(config, "model_type", "")
    return str(value).lower()


def create_model_adapter(hf_config: Any):
    """Create the adapter for a parsed HF config.

    Gemma4 multimodal configs expose ``model_type=gemma4`` and a nested
    ``text_config``. Text-only configs may expose ``gemma4_text`` directly.
    """
    model_type = _model_type(hf_config)
    if model_type in {"qwen2", "qwen2.5", "qwen2_5"}:
        return Qwen2Adapter()
    if model_type in {"gemma4", "gemma4_text"}:
        return Gemma4Adapter()
    raise ValueError(f"Unsupported model_type: {model_type!r}")

