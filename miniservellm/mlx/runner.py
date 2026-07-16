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

    def generate_greedy_batch(
        self,
        prompt_token_ids: List[List[int]],
        max_new_tokens: int,
        eos_token_id: int | None = None,
        disable_eos: bool = False,
        compiled_decode: bool = False,
        use_custom_attn: bool = False,
    ) -> List[MLXGenerationResult]:
        """Same-length batch greedy baseline for MLX throughput experiments."""
        if not prompt_token_ids:
            return []
        lengths = {len(prompt) for prompt in prompt_token_ids}
        if len(lengths) != 1:
            raise ValueError("generate_greedy_batch requires prompts with equal token length")
        if max_new_tokens <= 0:
            return [MLXGenerationResult([], 0.0, 0.0) for _ in prompt_token_ids]

        batch_size = len(prompt_token_ids)
        if (compiled_decode or use_custom_attn) and batch_size != 1:
            raise ValueError("compiled_decode/use_custom_attn currently supports batch_size=1 only")
        if compiled_decode and use_custom_attn:
            raise ValueError("compiled_decode and use_custom_attn are mutually exclusive")
        capacity = next(iter(lengths)) + max_new_tokens
        cache = MLXKVCache(self.model_config, capacity, batch_size=batch_size)
        prompt = mx.array(prompt_token_ids, dtype=mx.uint32)

        prefill_start = time.perf_counter()
        logits = self.model.forward(prompt, cache)
        next_tokens = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_tokens)
        prefill_seconds = time.perf_counter() - prefill_start

        compiled_step = None
        if compiled_decode:
            compiled_step = self.model.get_compiled_decode_fn()

        generated = [[int(token)] for token in next_tokens.tolist()]
        decode_start = time.perf_counter()
        while len(generated[0]) < max_new_tokens:
            if not disable_eos and eos_token_id is not None:
                if all(tokens[-1] == eos_token_id for tokens in generated):
                    break
            if use_custom_attn:
                next_tokens, cache.k_layers, cache.v_layers = self.model._decode_greedy_step(
                    next_tokens.reshape(1, 1), mx.array(cache.offset, dtype=mx.int32), cache.k_layers, cache.v_layers, use_custom_attn=True
                )
                cache.advance(1)
            elif compiled_decode:
                assert compiled_step is not None
                next_tokens, cache.k_layers, cache.v_layers = compiled_step(
                    next_tokens.reshape(1, 1), mx.array(cache.offset, dtype=mx.int32), cache.k_layers, cache.v_layers
                )
                cache.advance(1)
            else:
                logits = self.model.forward(next_tokens.reshape(batch_size, 1), cache)
                next_tokens = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(next_tokens)
            for index, token in enumerate(next_tokens.tolist()):
                if disable_eos or eos_token_id is None or generated[index][-1] != eos_token_id:
                    generated[index].append(int(token))
        decode_seconds = time.perf_counter() - decode_start
        return [
            MLXGenerationResult(tokens, prefill_seconds, decode_seconds)
            for tokens in generated
        ]

    def generate_greedy(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int,
        eos_token_id: int | None = None,
        disable_eos: bool = False,
        compiled_decode: bool = False,
        use_custom_attn: bool = False,
    ) -> MLXGenerationResult:
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        if compiled_decode and use_custom_attn:
            raise ValueError("compiled_decode and use_custom_attn are mutually exclusive")
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

        compiled_step = None
        if compiled_decode:
            compiled_step = self.model.get_compiled_decode_fn()

        generated = [int(next_token.item())]
        decode_start = time.perf_counter()
        while len(generated) < max_new_tokens:
            if not disable_eos and eos_token_id is not None and generated[-1] == eos_token_id:
                break
            if use_custom_attn:
                next_token, cache.k_layers, cache.v_layers = self.model._decode_greedy_step(
                    next_token.reshape(1, 1), mx.array(cache.offset, dtype=mx.int32), cache.k_layers, cache.v_layers, use_custom_attn=True
                )
                cache.advance(1)
            elif compiled_decode:
                assert compiled_step is not None
                next_token, cache.k_layers, cache.v_layers = compiled_step(
                    next_token.reshape(1, 1), mx.array(cache.offset, dtype=mx.int32), cache.k_layers, cache.v_layers
                )
                cache.advance(1)
            else:
                logits = self.model.forward(next_token.reshape(1, 1), cache)
                next_token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(next_token)
            generated.append(int(next_token.item()))
        decode_seconds = time.perf_counter() - decode_start

        return MLXGenerationResult(generated, prefill_seconds, decode_seconds)
