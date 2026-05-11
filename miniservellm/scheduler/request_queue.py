"""请求队列

第三阶段在第二阶段基础上增加 prefilling_queue，
支持 chunked prefill 中"未完成 prefill 但正在处理"的请求暂存。
队列结构：waiting_queue -> prefilling_queue -> decoding_queue -> finished_requests。
"""

from __future__ import annotations

from collections import deque
from miniservellm.scheduler.request import Request


class RequestQueue:
    """推理请求队列

    维护四类请求集合：
    - waiting_queue: 新到达、等待进入 prefill 的请求
    - prefilling_queue: 正在 chunked prefill 但尚未完成的请求
    - decoding_queue: 已完成 prefill、等待继续 decode 的请求
    - finished_requests: 已完成请求列表

    Attributes:
        waiting_queue: 等待 prefill 的请求队列
        prefilling_queue: 正在 prefill 的请求队列
        decoding_queue: 等待 decode 的请求队列
        finished_requests: 已完成请求列表
    """

    def __init__(self):
        self.waiting_queue: deque[Request] = deque()
        self.prefilling_queue: deque[Request] = deque()
        self.decoding_queue: deque[Request] = deque()
        self.finished_requests: list[Request] = []

    def add_request(self, request: Request) -> None:
        """添加新请求到 waiting 队列"""
        request.status = "waiting"
        self.waiting_queue.append(request)

    def _admit_waiting_to_prefill(self) -> None:
        """将 waiting 队列中的请求提升到 prefilling 状态

        当 scheduler 请求 prefill 候选时调用，
        将所有 waiting 请求转为 prefilling 状态。
        """
        while self.waiting_queue:
            req = self.waiting_queue.popleft()
            req.mark_prefilling()
            self.prefilling_queue.append(req)

    def pop_prefill_candidates(self, n: int) -> list[Request]:
        """取出最多 n 个等待 prefill 的请求

        先将 waiting 队列的请求提升为 prefilling 状态，
        再从 prefilling 队列中取出。

        Args:
            n: 最多取出的请求数

        Returns:
            prefill 候选请求列表
        """
        if n <= 0:
            return []

        self._admit_waiting_to_prefill()

        out = []
        while self.prefilling_queue and len(out) < n:
            out.append(self.prefilling_queue.popleft())
        return out

    def pop_decode_candidates(self, n: int) -> list[Request]:
        """取出最多 n 个等待 decode 的请求

        Args:
            n: 最多取出的请求数

        Returns:
            decode 候选请求列表
        """
        if n <= 0:
            return []

        out = []
        while self.decoding_queue and len(out) < n:
            out.append(self.decoding_queue.popleft())
        return out

    def requeue_for_prefill(self, request: Request) -> None:
        """将未完成 prefill 的请求重新放回 prefilling 队列"""
        request.mark_prefilling()
        self.prefilling_queue.append(request)

    def requeue_for_decode(self, request: Request) -> None:
        """将请求放入 decode 队列

        两种调用场景：
        1. prefill 完成后首次进入 decode 阶段（首次入队）
        2. decode 每步完成后未结束，重新放回等待下一步（重新入队）
        """
        request.mark_decoding()
        self.decoding_queue.append(request)

    def mark_finished(self, request: Request) -> None:
        """标记请求完成并移入 finished_requests"""
        request.mark_finished()
        self.finished_requests.append(request)

    def has_pending(self) -> bool:
        """是否仍有等待执行或正在处理的请求"""
        return bool(self.waiting_queue or self.prefilling_queue or self.decoding_queue)

    def all_finished_requests(self) -> list[Request]:
        """返回所有已完成请求"""
        return list(self.finished_requests)

    def num_waiting(self) -> int:
        """等待 prefill 的请求数量"""
        return len(self.waiting_queue)

    def num_prefilling(self) -> int:
        """正在 prefill 的请求数量"""
        return len(self.prefilling_queue)

    def num_decoding(self) -> int:
        """等待 decode 的请求数量"""
        return len(self.decoding_queue)

    def num_finished(self) -> int:
        """已完成请求数量"""
        return len(self.finished_requests)
