"""Batching 策略

第三阶段在第二阶段基础上增加：
- prefill_chunk_sizes：为每个 prefill 请求分配本轮的 token 预算
- max_prefill_tokens_per_step：限制每步 prefill 的总 token 数
- 两轮 round-robin 分配策略，避免长请求饥饿
"""

from __future__ import annotations

from dataclasses import dataclass, field
from miniservellm.scheduler.request import Request


@dataclass
class BatchPlan:
    """一次 engine.step() 的执行计划

    第三阶段增加 prefill_chunk_sizes，记录每个 prefill 请求
    在本轮被分配的 token 数。

    Attributes:
        prefill_requests: 本轮要执行 prefill 的请求列表
        decode_requests: 本轮要执行 decode 的请求列表
        prefill_chunk_sizes: request_id -> 本轮分配的 prefill token 数
    """
    prefill_requests: list[Request]
    decode_requests: list[Request]
    prefill_chunk_sizes: dict[str, int] = field(default_factory=dict)

    @property
    def total_size(self) -> int:
        """本轮总请求数"""
        return len(self.prefill_requests) + len(self.decode_requests)


class BatchingStrategy:
    """构造 BatchPlan 的策略

    第三阶段支持：
    - decode-first：优先调度 decode 请求，剩余 batch 空位再填 prefill
    - prefill token budget：全局预算在所有 prefill 请求间 round-robin 分配
    - 防饥饿：第一轮每个请求至少分配 1 token

    Attributes:
        max_batch_size: 最大 batch 大小（prefill + decode 请求总数）
        decode_first: 是否 decode 优先
        max_prefill_tokens_per_step: 每步 prefill 的总 token 预算
    """

    def __init__(
        self,
        max_batch_size: int = 4,
        decode_first: bool = True,
        max_prefill_tokens_per_step: int = 128,
    ):
        self.max_batch_size = max_batch_size
        self.decode_first = decode_first
        self.max_prefill_tokens_per_step = max_prefill_tokens_per_step

    def _allocate_prefill_budget(self, prefill_requests: list[Request]) -> dict[str, int]:
        """在 prefill 请求间分配 token 预算

        两轮分配策略：
        1. 第一轮：每个请求至少给 1 token，避免饥饿
        2. 第二轮：round-robin 将剩余预算分给还有剩余 token 的请求

        Args:
            prefill_requests: 本轮的 prefill 请求列表

        Returns:
            request_id -> 本轮分配的 chunk token 数
        """
        if not prefill_requests:
            return {}

        remaining_budget = self.max_prefill_tokens_per_step
        chunk_sizes: dict[str, int] = {}

        # 每个请求的 cap = min(chunk_size, 剩余未处理 token 数)
        caps = {
            req.request_id: min(req.chunk_size, req.remaining_prefill_tokens())
            for req in prefill_requests
        }

        # 第一轮：每个请求至少给 1 token，避免饥饿
        for req in prefill_requests:
            rid = req.request_id
            if remaining_budget <= 0:
                chunk_sizes[rid] = 0
                continue

            if caps[rid] > 0:
                chunk_sizes[rid] = 1
                remaining_budget -= 1
            else:
                chunk_sizes[rid] = 0

        # 第二轮：round-robin 分配剩余预算
        made_progress = True
        while remaining_budget > 0 and made_progress:
            made_progress = False
            for req in prefill_requests:
                if remaining_budget <= 0:
                    break
                rid = req.request_id
                if chunk_sizes[rid] < caps[rid]:
                    chunk_sizes[rid] += 1
                    remaining_budget -= 1
                    made_progress = True

        return chunk_sizes

    def build_batch(self, request_queue) -> BatchPlan:
        """从请求队列中构造一个 BatchPlan

        Args:
            request_queue: RequestQueue 实例

        Returns:
            当前 step 的 BatchPlan，包含 prefill/decode 请求和 chunk 分配
        """
        prefill_requests = []
        decode_requests = []
        remaining = self.max_batch_size

        if self.decode_first:
            # decode 优先：先取已有 KV Cache 的请求继续推进
            decode_requests = request_queue.pop_decode_candidates(remaining)
            remaining -= len(decode_requests)

            # 如果 batch 还有空位，再补充新的 prefill 请求
            if remaining > 0:
                prefill_requests = request_queue.pop_prefill_candidates(remaining)
        else:
            # prefill 优先：用于实验不同调度策略
            prefill_requests = request_queue.pop_prefill_candidates(remaining)
            remaining -= len(prefill_requests)

            if remaining > 0:
                decode_requests = request_queue.pop_decode_candidates(remaining)

        # 为 prefill 请求分配 token 预算
        prefill_chunk_sizes = self._allocate_prefill_budget(prefill_requests)

        return BatchPlan(
            prefill_requests=prefill_requests,
            decode_requests=decode_requests,
            prefill_chunk_sizes=prefill_chunk_sizes,
        )
