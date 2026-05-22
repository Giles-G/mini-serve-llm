"""请求队列

第五阶段：使用 waiting / running_prefill / running_decode / finished 四个列表。
"""

from __future__ import annotations

from miniservellm.scheduler.request import Request


class RequestQueue:
    def __init__(self) -> None:
        self.waiting: list[Request] = []
        self.running_prefill: list[Request] = []
        self.running_decode: list[Request] = []
        self.finished: list[Request] = []

    def add_waiting(self, req: Request) -> None:
        self.waiting.append(req)

    def remove_waiting(self, req: Request) -> None:
        self.waiting = [r for r in self.waiting if r.request_id != req.request_id]

    def remove_running_prefill(self, req: Request) -> None:
        self.running_prefill = [r for r in self.running_prefill if r.request_id != req.request_id]

    def remove_running_decode(self, req: Request) -> None:
        self.running_decode = [r for r in self.running_decode if r.request_id != req.request_id]

    def has_pending_work(self) -> bool:
        return bool(self.waiting or self.running_prefill or self.running_decode)
