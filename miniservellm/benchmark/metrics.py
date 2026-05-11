"""请求性能指标统计

第三阶段在第二阶段基础上增加 prefill_cursor / prefill_done / chunk_size 指标，
方便观察 chunked prefill 的效果。
"""

from __future__ import annotations

from miniservellm.runtime.outputs import MetricsSummary


class RequestMetrics:
    """请求性能指标统计器"""

    def __init__(self, request):
        self.request = request

    def summary(self) -> MetricsSummary:
        """生成单个请求的性能指标摘要

        Returns:
            MetricsSummary 包含 request_id、状态、prompt/output token 数、
            prefill 进度、chunk_size、TTFT 和端到端延迟
        """
        ttft = None
        e2e_latency = None

        # TTFT = 首 token 时间 - 请求到达时间
        if self.request.first_token_time is not None:
            ttft = self.request.first_token_time - self.request.arrival_time

        # E2E latency = 完成时间 - 请求到达时间
        if self.request.finish_time is not None:
            e2e_latency = self.request.finish_time - self.request.arrival_time

        return MetricsSummary(
            request_id=self.request.request_id,
            status=self.request.status,
            prompt_tokens=len(self.request.prompt_token_ids),
            output_tokens=len(self.request.generated_token_ids),
            prefill_cursor=self.request.prefill_cursor,
            prefill_done=self.request.prefill_done,
            chunk_size=self.request.chunk_size,
            ttft=ttft,
            e2e_latency=e2e_latency,
        )


def summarize_requests(requests: list):
    """批量汇总多个请求的性能指标

    Args:
        requests: 已完成请求列表

    Returns:
        每个请求的指标摘要列表
    """
    return [RequestMetrics(req).summary() for req in requests]
