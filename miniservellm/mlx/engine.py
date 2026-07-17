"""Fixed-slot continuous-batching MLX inference engine."""

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
    """MLX engine with fixed-shape slots and continuous request refill.

    The active cache always has ``max_batch_size`` rows. A finished or aborted
    request releases its row immediately; a waiting request that fits the cache
    capacity is prefetched directly into that row on the next engine tick.
    """

    def __init__(
        self,
        runner: MLXQwen2Runner,
        tokenizer,
        model_config: ModelConfig,
        eos_token_id: Optional[int] = None,
        stream_interval: int = 1,
        sampler_seed: Optional[int] = None,
        max_batch_size: int = 1,
        prompt_bucket_multiple: int = 64,
        compiled_decode: bool = False,
        batch_mode: str = "auto",
    ):
        if stream_interval <= 0:
            raise ValueError("stream_interval must be greater than 0")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be greater than 0")
        if prompt_bucket_multiple <= 0:
            raise ValueError("prompt_bucket_multiple must be greater than 0")
        if batch_mode not in {"static", "continuous", "auto"}:
            raise ValueError("batch_mode must be one of: static, continuous, auto")
        self.runner = runner
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.eos_token_id = eos_token_id
        self.stream_interval = stream_interval
        self.sampler = MLXSampler(sampler_seed)
        self.max_batch_size = max_batch_size
        self.prompt_bucket_multiple = prompt_bucket_multiple
        self.compiled_decode = compiled_decode
        self.batch_mode = batch_mode
        self.static_fast_path_ticks = 0
        self.dynamic_fallback_ticks = 0
        self.requests_by_id: Dict[str, Request] = {}
        self._waiting: List[str] = []
        self._slot_request_ids: List[Optional[str]] = [None] * max_batch_size
        self._active_cache: Optional[MLXKVCache] = None
        self._active_next_tokens: Optional[mx.array] = None
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
        if request_id in self._waiting:
            self._waiting.remove(request_id)
            return
        for slot, active_id in enumerate(self._slot_request_ids):
            if active_id == request_id:
                self._release_slot(slot)
                return

    def get_request(self, request_id: str) -> Request:
        return self.requests_by_id[request_id]

    def has_pending_work(self) -> bool:
        return bool(self._waiting) or any(rid is not None for rid in self._slot_request_ids)

    def _prompt_bucket(self, prompt_len: int) -> int:
        """Return the upper token-length bucket used for admission."""
        multiple = self.prompt_bucket_multiple
        return ((prompt_len + multiple - 1) // multiple) * multiple

    def _active_slot_count(self) -> int:
        return sum(rid is not None for rid in self._slot_request_ids)

    def _free_slot_indices(self) -> List[int]:
        return [slot for slot, rid in enumerate(self._slot_request_ids) if rid is None]

    def _release_slot(self, slot: int) -> None:
        self._slot_request_ids[slot] = None

    def _take_waiting_requests(self, limit: int, capacity: Optional[int] = None) -> List[Request]:
        """Select compatible waiting requests, exact matching in static mode."""
        def fits(candidate: Request) -> bool:
            return capacity is None or candidate.total_prompt_tokens() + candidate.max_new_tokens <= capacity

        def matches_batch(first: Request, candidate: Request) -> bool:
            if self.batch_mode == "static":
                return (
                    candidate.total_prompt_tokens() == first.total_prompt_tokens()
                    and candidate.max_new_tokens == first.max_new_tokens
                )
            return self._prompt_bucket(candidate.total_prompt_tokens()) == self._prompt_bucket(
                first.total_prompt_tokens()
            )

        first: Optional[Request] = None
        skipped: List[str] = []
        while self._waiting:
            request_id = self._waiting.pop(0)
            candidate = self.get_request(request_id)
            if candidate.status == RequestStatus.FINISHED:
                continue
            if fits(candidate):
                first = candidate
                break
            skipped.append(request_id)
        if first is None:
            self._waiting = skipped
            return []

        admitted = [first]
        remaining: List[str] = skipped
        for rid in self._waiting:
            candidate = self.get_request(rid)
            if (
                len(admitted) < limit
                and candidate.status != RequestStatus.FINISHED
                and fits(candidate)
                and matches_batch(first, candidate)
            ):
                admitted.append(candidate)
            else:
                remaining.append(rid)
        self._waiting = remaining
        return admitted

    def _clear_active_cache(self) -> None:
        self._slot_request_ids = [None] * self.max_batch_size
        self._active_cache = None
        self._active_next_tokens = None

    def _can_use_static_fast_path(self) -> bool:
        """Return whether every slot shares one cursor and remains active."""
        if self.batch_mode == "continuous" or self._active_cache is None:
            return False
        if self._active_slot_count() != self.max_batch_size:
            return False
        if len(set(self._active_cache.slot_offsets)) != 1:
            return False
        return all(request_id is not None for request_id in self._slot_request_ids)

    def _sample_active_slots(self, active_logits: mx.array, active_slots: List[int]) -> mx.array:
        """Batch-sample all active slots in one GPU op.

        Groups by sampling strategy: greedy (temperature<=0) rows use argmax;
        the rest share a per-row temperature categorical. This avoids one
        ``mx.eval`` per slot.
        """
        params_list = [
            self.get_request(self._slot_request_ids[s]).sampling_params
            for s in active_slots
        ]
        if all(p.temperature <= 0.0 for p in params_list):
            return mx.argmax(active_logits, axis=-1)
        temps = mx.array([p.temperature for p in params_list], dtype=mx.float32)
        work = active_logits.astype(mx.float32) / temps[:, None]
        return mx.random.categorical(work)

    def _emit_and_finish(
        self,
        req: Request,
        event_kind: str,
        token_id: int,
        events: List[MLXStepEvent],
        callback: Optional[Callable[[str, List[int]], None]],
    ) -> None:
        req.append_generated_token(token_id)
        events.append(MLXStepEvent(event_kind, req.request_id, token_id))
        if callback and len(req.generated_token_ids) % self.stream_interval == 0:
            callback(req.request_id, req.generated_token_ids[-self.stream_interval :])

        if self.eos_token_id is not None and token_id == self.eos_token_id:
            req.mark_finished_eos()
        elif not req.can_decode_more():
            req.mark_finished_max_new_tokens()
        if req.status == RequestStatus.FINISHED:
            remainder = len(req.generated_token_ids) % self.stream_interval
            if callback and remainder:
                callback(req.request_id, req.generated_token_ids[-remainder:])
            events.append(MLXStepEvent("request_finished", req.request_id))

    def _install_prefill_slots(
        self,
        slots: List[int],
        requests: List[Request],
        callback: Optional[Callable[[str, List[int]], None]],
        event_kind: str,
    ) -> List[MLXStepEvent]:
        assert self._active_cache is not None and self._active_next_tokens is not None
        logits = self.runner.prefill_dynamic_slots(
            self._active_cache, slots, [req.prompt_token_ids for req in requests]
        )
        # Batch-sample all prefill rows in one GPU op (lazy: logits stay on GPU)
        params_list = [req.sampling_params for req in requests]
        if all(p.temperature <= 0.0 for p in params_list):
            tokens = mx.argmax(logits, axis=-1)
        else:
            temps = mx.array([p.temperature for p in params_list], dtype=mx.float32)
            work = logits.astype(mx.float32) / temps[:, None]
            tokens = mx.random.categorical(work)
        mx.eval(tokens)
        token_ids = tokens.tolist()

        next_tokens = self._active_next_tokens.tolist()
        events: List[MLXStepEvent] = []
        for idx, (slot, req) in enumerate(zip(slots, requests)):
            token_id = token_ids[idx]
            req.num_prompt_tokens_processed = req.total_prompt_tokens()
            req.status = RequestStatus.RUNNING_DECODE
            self._slot_request_ids[slot] = req.request_id
            self._emit_and_finish(req, "prefill_sampled_first_token", token_id, events, callback)
            next_tokens[slot] = token_id
            if event_kind:
                events.append(MLXStepEvent(event_kind, req.request_id))
            if req.status == RequestStatus.FINISHED:
                self._release_slot(slot)
        self._active_next_tokens = mx.array(next_tokens, dtype=mx.uint32)
        return events

    def _start_cache(
        self,
        requests: List[Request],
        callback: Optional[Callable[[str, List[int]], None]],
    ) -> List[MLXStepEvent]:
        capacity = max(req.total_prompt_tokens() + req.max_new_tokens for req in requests)
        self._active_cache = MLXKVCache(self.model_config, capacity, batch_size=self.max_batch_size)
        self._active_next_tokens = mx.zeros((self.max_batch_size,), dtype=mx.uint32)
        return self._install_prefill_slots(list(range(len(requests))), requests, callback, "")

    def _refill_free_slots(
        self,
        callback: Optional[Callable[[str, List[int]], None]],
    ) -> List[MLXStepEvent]:
        if self.batch_mode == "static":
            return []
        assert self._active_cache is not None
        slots = self._free_slot_indices()
        if not slots:
            return []
        requests = self._take_waiting_requests(len(slots), self._active_cache.max_context)
        if not requests:
            return []
        return self._install_prefill_slots(slots[: len(requests)], requests, callback, "slot_refilled")

    def step(self, callback: Optional[Callable[[str, List[int]], None]] = None) -> List[MLXStepEvent]:
        """Run one fixed-shape decode tick and refill all slots released by it."""
        if self._active_cache is None:
            admitted = self._take_waiting_requests(self.max_batch_size)
            if not admitted:
                return []
            return self._start_cache(admitted, callback)

        assert self._active_next_tokens is not None
        events: List[MLXStepEvent] = []
        active_rows = [rid is not None for rid in self._slot_request_ids]
        if any(active_rows):
            if self._can_use_static_fast_path():
                all_greedy = self.compiled_decode and all(
                    self.get_request(rid).sampling_params.temperature <= 0.0
                    for rid in self._slot_request_ids if rid is not None
                )
                if all_greedy:
                    # Greedy fast path: argmax fused in compiled graph,
                    # returns [batch] tokens directly — no logits materialisation.
                    tokens = self.runner.decode_static_batch(
                        self._active_next_tokens,
                        self._active_cache,
                        compiled_decode=True,
                        greedy=True,
                    )
                    self.static_fast_path_ticks += 1
                    mx.eval(tokens)
                    token_ids = tokens.tolist()
                    for slot, request_id in enumerate(self._slot_request_ids):
                        req = self.get_request(request_id)
                        token_id = token_ids[slot]
                        self._emit_and_finish(req, "decode_token", token_id, events, callback)
                        if req.status == RequestStatus.FINISHED:
                            self._release_slot(slot)
                    self._active_next_tokens = tokens.astype(mx.uint32)
                else:
                    logits = self.runner.decode_static_batch(
                        self._active_next_tokens,
                        self._active_cache,
                        compiled_decode=self.compiled_decode,
                    )
                    self.static_fast_path_ticks += 1
                    active_slots = [s for s, rid in enumerate(self._slot_request_ids) if rid is not None]
                    if active_slots:
                        active_logits = logits[mx.array(active_slots)]
                        sampled = self._sample_active_slots(active_logits, active_slots)
                        mx.eval(sampled)
                        sampled_ids = sampled.tolist()
                        next_tokens = self._active_next_tokens.tolist()
                        for idx, slot in enumerate(active_slots):
                            req = self.get_request(self._slot_request_ids[slot])
                            token_id = sampled_ids[idx]
                            self._emit_and_finish(req, "decode_token", token_id, events, callback)
                            next_tokens[slot] = token_id
                            if req.status == RequestStatus.FINISHED:
                                self._release_slot(slot)
                        self._active_next_tokens = mx.array(next_tokens, dtype=mx.uint32)
            else:
                logits = self.runner.decode_dynamic_batch(
                    self._active_next_tokens,
                    self._active_cache,
                    active_rows,
                    compiled_decode=self.compiled_decode,
                )
                self.dynamic_fallback_ticks += 1
                active_slots = [s for s, rid in enumerate(self._slot_request_ids) if rid is not None]
                if active_slots:
                    active_logits = logits[mx.array(active_slots)]
                    sampled = self._sample_active_slots(active_logits, active_slots)
                    mx.eval(sampled)
                    sampled_ids = sampled.tolist()
                    next_tokens = self._active_next_tokens.tolist()
                    for idx, slot in enumerate(active_slots):
                        req = self.get_request(self._slot_request_ids[slot])
                        token_id = sampled_ids[idx]
                        self._emit_and_finish(req, "decode_token", token_id, events, callback)
                        next_tokens[slot] = token_id
                        if req.status == RequestStatus.FINISHED:
                            self._release_slot(slot)
                    self._active_next_tokens = mx.array(next_tokens, dtype=mx.uint32)

        events.extend(self._refill_free_slots(callback))
        if self._active_slot_count() == 0:
            self._clear_active_cache()
        return events

    @property
    def decode_path_stats(self) -> tuple[int, int]:
        """Return ``(static_fast_path_ticks, dynamic_fallback_ticks)``."""
        return self.static_fast_path_ticks, self.dynamic_fallback_ticks

    def run_until_all_finished(self, callback: Optional[Callable[[str, List[int]], None]] = None) -> None:
        while self.has_pending_work():
            self.step(callback)

    def get_text(self, request_id: str) -> str:
        return self.tokenizer.decode(self.get_request(request_id).generated_token_ids, skip_special_tokens=True)

    def get_full_text(self, request_id: str) -> str:
        return self.tokenizer.decode(self.get_request(request_id).all_token_ids(), skip_special_tokens=True)
