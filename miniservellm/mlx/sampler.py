"""MLX sampling implementation matching the public SamplingParams contract."""

from __future__ import annotations

from typing import Iterable

import mlx.core as mx

from miniservellm.scheduler.request import SamplingParams


def _top_p_filter(logits: mx.array, top_p: float) -> mx.array:
    if top_p >= 1.0:
        return logits
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    order = mx.argsort(-logits, axis=-1)
    sorted_logits = mx.take_along_axis(logits, order, axis=-1)
    probs = mx.softmax(sorted_logits.astype(mx.float32), axis=-1)
    cumulative = mx.cumsum(probs, axis=-1)
    remove = cumulative > top_p
    remove = mx.concatenate([mx.zeros_like(remove[..., :1]), remove[..., :-1]], axis=-1)
    sorted_logits = mx.where(remove, mx.array(float("-inf"), dtype=logits.dtype), sorted_logits)
    filtered = mx.full(logits.shape, float("-inf"), dtype=logits.dtype)
    return mx.put_along_axis(filtered, order, sorted_logits, axis=-1)


class MLXSampler:
    def __init__(self, seed: int | None = None):
        if seed is not None:
            mx.random.seed(seed)

    def sample(
        self,
        logits: mx.array,
        params: SamplingParams,
        token_history: Iterable[int] = (),
    ) -> mx.array:
        """Sample [B, vocab] logits and return [B] token ids on device."""
        if params.temperature <= 0.0:
            return mx.argmax(logits, axis=-1)

        work = logits.astype(mx.float32) / float(params.temperature)
        if params.repetition_penalty != 1.0:
            raise NotImplementedError(
                "MLX repetition_penalty is not implemented yet; use 1.0 for this backend."
            )

        if 0 < params.top_k < work.shape[-1]:
            values = mx.topk(work, params.top_k, axis=-1)
            threshold = values[..., -1:]
            work = mx.where(work < threshold, mx.array(float("-inf"), dtype=work.dtype), work)
        if params.top_p < 1.0:
            work = _top_p_filter(work, params.top_p)
        return mx.random.categorical(work)
