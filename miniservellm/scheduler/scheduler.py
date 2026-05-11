"""Scheduler

第三阶段在第二阶段基础上增加 max_prefill_tokens_per_step 参数，
传递给 BatchingStrategy 控制 chunked prefill 的每步 token 预算。
"""

from __future__ import annotations

from miniservellm.scheduler.batching import BatchingStrategy, BatchPlan


class Scheduler:
    """请求调度器

    每轮调用 BatchingStrategy 生成 BatchPlan，包含 prefill/decode 请求
    以及每个 prefill 请求的 chunk token 分配。

    Attributes:
        request_queue: 请求队列
        batching_strategy: batching 策略实例
    """

    def __init__(
        self,
        request_queue,
        max_batch_size: int = 4,
        decode_first: bool = True,
        max_prefill_tokens_per_step: int = 128,
    ):
        self.request_queue = request_queue
        self.batching_strategy = BatchingStrategy(
            max_batch_size=max_batch_size,
            decode_first=decode_first,
            max_prefill_tokens_per_step=max_prefill_tokens_per_step,
        )

    def schedule(self) -> BatchPlan:
        """生成下一轮 engine.step() 的执行计划"""
        return self.batching_strategy.build_batch(self.request_queue)
