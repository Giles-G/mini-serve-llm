"""Scheduler

Scheduler 是调度入口，负责每轮调用 batching strategy 生成 BatchPlan。
后续可以在这里加入优先级、公平性、最大 token budget 等策略。
"""

from miniservellm.scheduler.batching import BatchingStrategy, BatchPlan


class Scheduler:
    """请求调度器

    Attributes:
        request_queue: 请求队列
        batching_strategy: batching 策略实例
    """

    def __init__(self, request_queue, max_batch_size: int = 4, decode_first: bool = True):
        self.request_queue = request_queue
        self.batching_strategy = BatchingStrategy(
            max_batch_size=max_batch_size,
            decode_first=decode_first,
        )

    def schedule(self) -> BatchPlan:
        """生成下一轮 engine.step() 的执行计划"""
        return self.batching_strategy.build_batch(self.request_queue)
