"""请求队列

维护 waiting、decoding 和 finished 三类请求集合。
Scheduler 每轮从队列中取出一批请求组成 BatchPlan。
"""

from collections import deque
from miniservellm.scheduler.request import Request


class RequestQueue:
    """推理请求队列

    Attributes:
        waiting_queue: 等待 prefill 的请求队列
        decoding_queue: 已完成 prefill、等待继续 decode 的请求队列
        finished_requests: 已完成请求字典，request_id -> Request
    """

    def __init__(self):
        self.waiting_queue = deque()
        self.decoding_queue = deque()
        self.finished_requests: dict[str, Request] = {}

    def add_request(self, request: Request) -> None:
        """添加新请求到 waiting 队列"""
        request.status = "WAITING"
        self.waiting_queue.append(request)

    def has_pending(self) -> bool:
        """是否仍有等待执行或正在解码的请求"""
        return len(self.waiting_queue) > 0 or len(self.decoding_queue) > 0

    def pop_prefill_candidates(self, limit: int) -> list[Request]:
        """取出最多 limit 个等待 prefill 的请求"""
        selected = []
        while self.waiting_queue and len(selected) < limit:
            req = self.waiting_queue.popleft()
            req.status = "PREFILLING"
            selected.append(req)
        return selected

    def pop_decode_candidates(self, limit: int) -> list[Request]:
        """取出最多 limit 个等待 decode 的请求"""
        selected = []
        while self.decoding_queue and len(selected) < limit:
            req = self.decoding_queue.popleft()
            req.status = "DECODING"
            selected.append(req)
        return selected

    def requeue_for_decode(self, request: Request):
        """将未完成请求重新放回 decode 队列"""
        if not request.finished:
            request.status = "DECODING"
            self.decoding_queue.append(request)

    def mark_finished(self, request: Request):
        """标记请求完成并移入 finished_requests"""
        request.status = "FINISHED"
        request.finished = True
        self.finished_requests[request.request_id] = request

    def num_waiting(self) -> int:
        """等待 prefill 的请求数量"""
        return len(self.waiting_queue)

    def num_decoding(self) -> int:
        """等待 decode 的请求数量"""
        return len(self.decoding_queue)

    def num_finished(self) -> int:
        """已完成请求数量"""
        return len(self.finished_requests)

    def all_finished_requests(self) -> list[Request]:
        """返回所有已完成请求"""
        return list(self.finished_requests.values())
