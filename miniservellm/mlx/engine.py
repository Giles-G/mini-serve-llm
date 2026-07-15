"""Single-request MLX inference engine sharing Request and SamplingParams semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import mlx.core as mx

from miniservellm.config import ModelConfig
from miniservellm.mlx.qwen2 import MLXKVCache
from miniservellm.mlx.runner import MLXQwen2Runner
from miniservellm.mlx.sampler import MLXSampler
from miniservellm.scheduler.request import FinishReason, Request, RequestStatus, SamplingParams


@dataclass
class MLXStepEvent:
    kind: str
    request_id: str
    token_id: Optional[int] = None


class MLXInferenceEngine:
    """Feature-complete single-active-request engine backed by MLX.

    Queued requests are accepted using the existing Request model. The first
    implementation executes them FIFO because MLX continuous batching has not
    yet been profiled for heterogeneous request lengths.
    """

    def __init__(
        self,
        runner: MLXQwen2Runner,
        tokenizer,
        model_config: ModelConfig,
        eos_token_id: Optional[int] = None,
        stream_interval: int = 1,
        sampler_seed: Optional[int] = None,
    ):
        if stream_interval <= 0:
            raise ValueError("stream_interval must be greater than 0")
        self.runner = runner
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.eos_token_id = eos_token_id
        self.stream_interval = stream_interval
        self.sampler = MLXSampler(sampler_seed)
        self.requests_by_id: Dict[str, Request] = {}
        self._waiting: List[str] = []
        self._active_id: Optional[str] = None
        self._caches: Dict[str, MLXKVCache] = {}
        self._prefix_cache: Dict[tuple[int, ...], tuple[MLXKVCache, mx.array]] = {}
        self._next_request_idx = 0

    def add_request(
        self,
        text: Optional[str] = None,
        prompt_token_ids: Optional[List[int]] = None,
        sampling_params: Optional[SamplingParams] = None,
        max_new_tokens: int = 32,
    ) -> str:
        if prompt_token_ids is None:
            if text is None:
                raise ValueError("Either text or prompt_token_ids must be provided.")
            if self.tokenizer is None:
                raise ValueError("Tokenizer is required for text requests.")
            if hasattr(self.tokenizer, "apply_chat_template"):
                text = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            prompt_token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than 0")
        rid = f"mlx_req_{self._next_request_idx}"
        self._next_request_idx += 1
        req = Request(
            request_id=rid,
            prompt_token_ids=list(prompt_token_ids),
            max_new_tokens=max_new_tokens,
            sampling_params=sampling_params or SamplingParams(),
        )
        self.requests_by_id[rid] = req
        self._waiting.append(rid)
        return rid

    def abort_request(self, request_id: str) -> None:
        req = self.get_request(request_id)
        if req.status == RequestStatus.FINISHED:
            return
        req.mark_aborted()
        self._finish(req)

    def get_request(self, request_id: str) -> Request:
        return self.requests_by_id[request_id]

    def has_pending_work(self) -> bool:
        return bool(self._waiting) or self._active_id is not None

    def _activate_next(self) -> Optional[Request]:
        while self._waiting:
            rid = self._waiting.pop(0)
            req = self.requests_by_id[rid]
            if req.status != RequestStatus.FINISHED:
                self._active_id = rid
                return req
        return None

    def _finish(self, req: Request) -> None:
        self._caches.pop(req.request_id, None)
        if self._active_id == req.request_id:
            self._active_id = None

    def _prefill(self, req: Request) -> int:
        req.status = RequestStatus.RUNNING_PREFILL
        prefix_key = tuple(req.prompt_token_ids)
        cached = self._prefix_cache.get(prefix_key)
        required_capacity = req.total_prompt_tokens() + req.max_new_tokens
        if cached is not None and cached[0].max_context >= required_capacity:
            cache, logits = cached
            cache = cache.clone()
        else:
            cache = MLXKVCache(
                self.model_config,
                max_context=req.total_prompt_tokens() + req.max_new_tokens,
            )
            prompt = mx.array([req.prompt_token_ids], dtype=mx.uint32)
            hidden = self.runner.model.forward_hidden(prompt, cache)
            logits = self.runner.model.project_last_hidden(hidden).reshape(1, -1)
            mx.eval(logits)
            # Functional arrays are immutable: later Decode slice_update operations
            # return new tensors, so this snapshot remains valid for exact prompt hits.
            self._prefix_cache[prefix_key] = (cache.clone(), logits)
        mx.eval(logits)
        token = self.sampler.sample(logits, req.sampling_params, req.generated_token_ids)
        mx.eval(token)
        token_id = int(token.item())
        req.num_prompt_tokens_processed = req.total_prompt_tokens()
        req.status = RequestStatus.RUNNING_DECODE
        self._caches[req.request_id] = cache
        return token_id

    def step(self, callback: Optional[Callable[[str, List[int]], None]] = None) -> List[MLXStepEvent]:
        req = self.get_request(self._active_id) if self._active_id else self._activate_next()
        if req is None:
            return []
        events: List[MLXStepEvent] = []

        if req.status == RequestStatus.RUNNING_PREFILL:
            # Defensive: normal path always moves directly from WAITING to prefill below.
            pass
        if req.num_prompt_tokens_processed == 0:
            token_id = self._prefill(req)
            req.append_generated_token(token_id)
            events.append(MLXStepEvent("prefill_sampled_first_token", req.request_id, token_id))
        else:
            cache = self._caches[req.request_id]
            input_id = req.last_token_id_for_decode_input()
            hidden = self.runner.model.forward_hidden(mx.array([[input_id]], dtype=mx.uint32), cache)
            logits = self.runner.model.project_last_hidden(hidden).reshape(1, -1)
            mx.eval(logits)
            token = self.sampler.sample(logits, req.sampling_params, req.generated_token_ids)
            mx.eval(token)
            token_id = int(token.item())
            req.append_generated_token(token_id)
            events.append(MLXStepEvent("decode_token", req.request_id, token_id))

        if callback and req.generated_token_ids and len(req.generated_token_ids) % self.stream_interval == 0:
            callback(req.request_id, req.generated_token_ids[-self.stream_interval :])

        last = req.generated_token_ids[-1]
        if self.eos_token_id is not None and last == self.eos_token_id:
            req.mark_finished_eos()
        elif not req.can_decode_more():
            req.mark_finished_max_new_tokens()
        if req.status == RequestStatus.FINISHED:
            if callback and len(req.generated_token_ids) % self.stream_interval:
                callback(req.request_id, req.generated_token_ids[-(len(req.generated_token_ids) % self.stream_interval) :])
            events.append(MLXStepEvent("request_finished", req.request_id))
            self._finish(req)
        return events

    def run_until_all_finished(self, callback: Optional[Callable[[str, List[int]], None]] = None) -> None:
        while self.has_pending_work():
            self.step(callback)

    def get_text(self, request_id: str) -> str:
        return self.tokenizer.decode(self.get_request(request_id).generated_token_ids, skip_special_tokens=True)

    def get_full_text(self, request_id: str) -> str:
        return self.tokenizer.decode(self.get_request(request_id).all_token_ids(), skip_special_tokens=True)
