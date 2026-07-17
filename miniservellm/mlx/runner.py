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

    def prefill_batch(
        self,
        prompt_token_ids: List[List[int]],
        max_new_tokens: int,
    ) -> tuple[MLXKVCache, mx.array]:
        """Prefill an equal-length request batch and return cache plus [B, vocab] logits.

        The serving engine owns request lifecycle and sampling; this primitive only
        performs the model-side operation needed to admit one fixed batch.
        """
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        lengths = {len(prompt) for prompt in prompt_token_ids}
        if len(lengths) != 1:
            raise ValueError("prefill_batch requires prompts with equal token length")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than 0")

        batch_size = len(prompt_token_ids)
        capacity = next(iter(lengths)) + max_new_tokens
        cache = MLXKVCache(self.model_config, capacity, batch_size=batch_size)
        prompt = mx.array(prompt_token_ids, dtype=mx.uint32)
        logits = self.model.forward(prompt, cache)[:, -1, :]
        return cache, logits

    def decode_batch(self, token_ids: mx.array, cache: MLXKVCache) -> mx.array:
        """Decode one token for every row of an existing fixed MLX batch."""
        if token_ids.ndim != 1 or token_ids.shape[0] != cache.batch_size:
            raise ValueError("token_ids must have shape [batch_size]")
        return self.model.forward(token_ids.reshape(cache.batch_size, 1), cache)[:, -1, :]

    def prefill_dynamic_batch(
        self,
        prompt_token_ids: List[List[int]],
        max_new_tokens: int,
    ) -> tuple[MLXKVCache, mx.array]:
        """Prefill unequal prompts into one per-slot-offset cache.

        Each row is prefetched independently to preserve its true causal length,
        then copied into its row of a shared batch cache. Decode thereafter is
        batched. This is the correctness-first phase-2 path; phase 5 will replace
        these row copies with masked padded prefill and batched KV scatter.
        """
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than 0")
        if any(not prompt for prompt in prompt_token_ids):
            raise ValueError("prompt_token_ids must not contain empty prompts")

        batch_size = len(prompt_token_ids)
        capacity = max(len(prompt) for prompt in prompt_token_ids) + max_new_tokens
        batch_cache = MLXKVCache(self.model_config, capacity, batch_size=batch_size)
        logits_rows: List[mx.array] = []
        for row, prompt_ids in enumerate(prompt_token_ids):
            row_cache = MLXKVCache(self.model_config, capacity)
            logits = self.model.forward(mx.array([prompt_ids], dtype=mx.uint32), row_cache)[:, -1, :]
            logits_rows.append(logits)
            for layer_idx in range(self.model_config.num_hidden_layers):
                start = mx.array([row, 0, 0, 0], dtype=mx.uint32)
                batch_cache.k_layers[layer_idx] = mx.slice_update(
                    batch_cache.k_layers[layer_idx], row_cache.k_layers[layer_idx], start, axes=(0, 1, 2, 3)
                )
                batch_cache.v_layers[layer_idx] = mx.slice_update(
                    batch_cache.v_layers[layer_idx], row_cache.v_layers[layer_idx], start, axes=(0, 1, 2, 3)
                )
        batch_cache.set_slot_offsets([len(prompt) for prompt in prompt_token_ids])
        return batch_cache, mx.concatenate(logits_rows, axis=0)

    def prefill_dynamic_slots(
        self,
        cache: MLXKVCache,
        slot_indices: List[int],
        prompt_token_ids: List[List[int]],
    ) -> mx.array:
        """Prefill requests independently and install their state in existing slots.

        The shared cache keeps a fixed batch shape. Only the supplied rows are
        replaced, which lets the serving engine refill a completed slot without
        disturbing other requests that are still decoding.
        """
        if len(slot_indices) != len(prompt_token_ids) or not slot_indices:
            raise ValueError("slot_indices and prompt_token_ids must be non-empty and equally sized")
        if len(set(slot_indices)) != len(slot_indices):
            raise ValueError("slot_indices must be unique")
        if any(slot < 0 or slot >= cache.batch_size for slot in slot_indices):
            raise ValueError("slot index is outside cache batch size")
        if any(not prompt for prompt in prompt_token_ids):
            raise ValueError("prompt_token_ids must not contain empty prompts")
        if any(len(prompt) >= cache.max_context for prompt in prompt_token_ids):
            raise ValueError("prompt leaves no decode capacity in the shared cache")

        logits_rows: List[mx.array] = []
        for slot, prompt_ids in zip(slot_indices, prompt_token_ids):
            row_cache = MLXKVCache(self.model_config, cache.max_context)
            logits = self.model.forward(
                mx.array([prompt_ids], dtype=mx.uint32), row_cache
            )[:, -1, :]
            logits_rows.append(logits)
            for layer_idx in range(self.model_config.num_hidden_layers):
                start = mx.array([slot, 0, 0, 0], dtype=mx.uint32)
                cache.k_layers[layer_idx] = mx.slice_update(
                    cache.k_layers[layer_idx],
                    row_cache.k_layers[layer_idx],
                    start,
                    axes=(0, 1, 2, 3),
                )
                cache.v_layers[layer_idx] = mx.slice_update(
                    cache.v_layers[layer_idx],
                    row_cache.v_layers[layer_idx],
                    start,
                    axes=(0, 1, 2, 3),
                )
            cache.slot_offsets[slot] = len(prompt_ids)
        return mx.concatenate(logits_rows, axis=0)

    def decode_static_batch(
        self,
        token_ids: mx.array,
        cache: MLXKVCache,
        compiled_decode: bool = False,
        greedy: bool = False,
    ) -> mx.array:
        """Decode a full batch sharing one KV cursor without dynamic masking.

        When ``greedy=True`` and ``compiled_decode=True``, uses an argmax-fused
        compiled graph that returns ``[batch]`` token ids instead of
        ``[batch, vocab]`` logits, avoiding logits materialisation.
        """
        if token_ids.ndim != 1 or token_ids.shape[0] != cache.batch_size:
            raise ValueError("token_ids must have shape [batch_size]")
        if len(set(cache.slot_offsets)) != 1:
            raise ValueError("decode_static_batch requires equal slot offsets")
        if compiled_decode:
            if greedy:
                compiled_step = self.model.get_compiled_static_decode_greedy_fn(
                    cache.batch_size, cache.max_context, cache.k_layers[0].dtype
                )
                next_tokens, cache.k_layers, cache.v_layers = compiled_step(
                    token_ids.reshape(cache.batch_size, 1),
                    mx.array(cache.offset, dtype=mx.int32),
                    cache.k_layers,
                    cache.v_layers,
                )
                cache.advance(1)
                return next_tokens
            compiled_step = self.model.get_compiled_static_decode_fn(
                cache.batch_size, cache.max_context, cache.k_layers[0].dtype
            )
            logits, cache.k_layers, cache.v_layers = compiled_step(
                token_ids.reshape(cache.batch_size, 1),
                mx.array(cache.offset, dtype=mx.int32),
                cache.k_layers,
                cache.v_layers,
            )
            cache.advance(1)
            return logits
        return self.decode_batch(token_ids, cache)

    def decode_dynamic_batch(
        self,
        token_ids: mx.array,
        cache: MLXKVCache,
        active_rows: List[bool],
        compiled_decode: bool = False,
    ) -> mx.array:
        """Decode one token using each active row's independent KV offset."""
        if token_ids.ndim != 1 or token_ids.shape[0] != cache.batch_size:
            raise ValueError("token_ids must have shape [batch_size]")
        if compiled_decode:
            compiled_step = self.model.get_compiled_dynamic_decode_fn(
                cache.batch_size, cache.max_context, cache.k_layers[0].dtype
            )
            logits, cache.k_layers, cache.v_layers = compiled_step(
                token_ids.reshape(cache.batch_size, 1),
                mx.array(cache.slot_offsets, dtype=mx.uint32),
                mx.array(active_rows, dtype=mx.bool_),
                cache.k_layers,
                cache.v_layers,
            )
            cache.advance_slots(active_rows)
            return logits
        hidden = self.model.forward_hidden_dynamic_decode(
            token_ids.reshape(cache.batch_size, 1), cache, active_rows
        )
        return self.model.project_last_hidden(hidden)[:, -1, :]

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
