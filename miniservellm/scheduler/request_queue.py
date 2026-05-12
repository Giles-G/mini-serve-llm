"""请求队列

第四阶段简化队列结构：
- waiting: 新到达的请求
- active_prefill: 正在 prefill 的请求
- active_decode: 正在 decode 的请求

调度器直接从 active_prefill / active_decode 中选择请求，
执行器通过设置 request.status 来管理状态转换。
"""

from __future__ import annotations

from miniservellm.scheduler.request import Request


class RequestQueue:
    """推理请求队列

    Attributes:
        waiting: 新到达、等待进入 prefill 的请求列表
        active_prefill: 正在 prefill 的请求列表
        active_decode: 正在 decode 的请求列表
    """

    def __init__(self):
        self.waiting: list[Request] = []
        self.active_prefill: list[Request] = []
        self.active_decode: list[Request] = []

    def add_new_request(self, req: Request) -> None:
        """添加新请求到 waiting 列表"""
        self.waiting.append(req)

    def promote_waiting_to_prefill(self) -> None:
        """将 waiting 列表中的请求提升为 prefill 状态"""
        if not self.waiting:
            return
        moved = self.waiting
        self.waiting = []
        for req in moved:
            req.status = "prefilling"
            self.active_prefill.append(req)

    def remove_finished(self) -> None:
        """从 active_prefill 和 active_decode 中移除已完成的请求"""
        self.active_prefill = [
            r for r in self.active_prefill if r.status != "finished"
        ]
        self.active_decode = [
            r for r in self.active_decode if r.status != "finished"
        ]

    def has_pending(self) -> bool:
        """是否仍有等待执行或正在处理的请求"""
        return bool(self.waiting or self.active_prefill or self.active_decode)

    def num_waiting(self) -> int:
        return len(self.waiting)

    def num_prefilling(self) -> int:
        return len(self.active_prefill)

    def num_decoding(self) -> int:
        return len(self.active_decode)

    def num_finished(self) -> int:
        """已完成请求数量（从 request_index 中统计）"""
        return 0  # 由 engine.request_index 统计
