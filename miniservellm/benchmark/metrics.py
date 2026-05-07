"""请求性能指标统计

第二阶段支持批量请求统计：既可以统计单个 Request，
也可以一次性汇总多个完成请求。
"""


class RequestMetrics:
    """请求性能指标统计器"""

    def __init__(self, request):
        self.request = request

    def summary(self):
        """生成单个请求的性能指标摘要

        Returns:
            包含 request_id、状态、prompt token 数、输出 token 数、TTFT 和端到端延迟的字典
        """
        ttft = None
        e2e_latency = None

        # TTFT = 首 token 时间 - 请求到达时间
        if self.request.first_token_time is not None:
            ttft = self.request.first_token_time - self.request.arrival_time

        # E2E latency = 完成时间 - 请求到达时间
        if self.request.finish_time is not None:
            e2e_latency = self.request.finish_time - self.request.arrival_time

        return {
            "request_id": self.request.request_id,
            "status": self.request.status,
            "prompt_tokens": len(self.request.prompt_token_ids),
            "output_tokens": len(self.request.generated_token_ids),
            "ttft": ttft,
            "e2e_latency": e2e_latency,
        }


def summarize_requests(requests: list):
    """批量汇总多个请求的性能指标

    Args:
        requests: 已完成请求列表

    Returns:
        每个请求的指标摘要列表
    """
    summaries = [RequestMetrics(req).summary() for req in requests]
    return summaries
