"""Batch=1 MLX Qwen2 greedy inference runner."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import mlx.core as mx

from miniservellm.config import ModelConfig
from miniservellm.mlx.adapter import load_qwen2_weights
from miniservellm.mlx.qwen2 import MLXKVCache, MLXQwen2


@dataclass
class MLXGenerationResult:
    generated_token_ids: List[int]
    prefill_seconds: float
    decode_seconds: float

    @property
    def decode_tok_s(self) -> float:
        decode_tokens = max(0, len(self.generated_token_ids) - 1)
        return decode_tokens / self.decode_seconds if self.decode_seconds > 0 else 0.0


class MLXQwen2Runner:
    """Independent Apple Silicon FP16 runner; does not use Paged KV or Scheduler."""

    def __init__(self, model_config: ModelConfig, safetensors_path: str | Path):
        self.model_config = model_config
        self.weights = load_qwen2_weights(safetensors_path, model_config)
        self.model = MLXQwen2(model_config, self.weights)
        mx.eval(self.weights.embed_tokens)

    def generate_greedy(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int,
        eos_token_id: int | None = None,
        disable_eos: bool = False,
    ) -> MLXGenerationResult:
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        if max_new_tokens <= 0:
            return MLXGenerationResult([], 0.0, 0.0)

        capacity = len(prompt_token_ids) + max_new_tokens
        cache = MLXKVCache(self.model_config, capacity)
        prompt = mx.array([prompt_token_ids], dtype=mx.uint32)

        prefill_start = time.perf_counter()
        logits = self.model.forward(prompt, cache)
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_token)
        prefill_seconds = time.perf_counter() - prefill_start

        generated = [int(next_token.item())]
        decode_start = time.perf_counter()
        while len(generated) < max_new_tokens:
            if not disable_eos and eos_token_id is not None and generated[-1] == eos_token_id:
                break
            logits = self.model.forward(next_token.reshape(1, 1), cache)
            next_token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(next_token)
            generated.append(int(next_token.item()))
        decode_seconds = time.perf_counter() - decode_start

        return MLXGenerationResult(generated, prefill_seconds, decode_seconds)
