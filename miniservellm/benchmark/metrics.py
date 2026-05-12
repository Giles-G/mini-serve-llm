"""请求性能指标统计

第四阶段适配新的 Request 字段（prefill_offset 替代 prefill_cursor）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RequestMetricsSummary:
    """请求性能指标摘要

    Attributes:
        request_id: 请求唯一标识
        status: 请求当前状态
        prompt_tokens: prompt token 数
        output_tokens: 生成 token 数
        prefill_offset: 当前 prefill 偏移量
        prefill_done: prefill 是否完成
        chunk_size: chunk 大小
        ttft: 首 token 延迟（秒）
        e2e_latency: 端到端延迟（秒）
    """

    request_id: str
    status: str
    prompt_tokens: int
    output_tokens: int
    prefill_offset: int
    prefill_done: bool
    chunk_size: int
    ttft: float | None
    e2e_latency: float | None


class RequestMetrics:
    """请求性能指标统计器"""

    def __init__(self, request):
        self.request = request

    def summary(self) -> RequestMetricsSummary:
        """生成单个请求的性能指标摘要"""
        ttft = None
        e2e_latency = None

        if self.request.first_token_time is not None:
            ttft = self.request.first_token_time - self.request.arrival_time

        if self.request.finish_time is not None:
            e2e_latency = self.request.finish_time - self.request.arrival_time

        return RequestMetricsSummary(
            request_id=self.request.request_id,
            status=self.request.status,
            prompt_tokens=len(self.request.prompt_token_ids),
            output_tokens=len(self.request.generated_token_ids),
            prefill_offset=self.request.prefill_offset,
            prefill_done=self.request.is_prefill_done(),
            chunk_size=self.request.chunk_size,
            ttft=ttft,
            e2e_latency=e2e_latency,
        )


def summarize_requests(requests: list) -> list[RequestMetricsSummary]:
    """批量汇总多个请求的性能指标"""
    return [RequestMetrics(req).summary() for req in requests]
