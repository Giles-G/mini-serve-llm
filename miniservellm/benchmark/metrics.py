"""请求性能指标统计

计算 TTFT（Time To First Token）和端到端延迟等推理性能指标。
"""


class RequestMetrics:
    """请求性能指标统计器

    Attributes:
        request: 已完成的请求对象，包含时间戳信息
    """

    def __init__(self, request):
        self.request = request

    def summary(self) -> dict:
        """生成请求的性能指标摘要

        Returns:
            包含以下指标的字典：
            - request_id: 请求 ID
            - prompt_tokens: prompt 的 token 数
            - output_tokens: 生成的 token 数
            - ttft: 首 token 延迟（秒），prefill 阶段耗时
            - e2e_latency: 端到端延迟（秒），从请求到达到完成
        """
        ttft = None
        e2e_latency = None

        # TTFT = 首 token 产出时间 - 请求到达时间
        if self.request.first_token_time is not None:
            ttft = self.request.first_token_time - self.request.arrival_time

        # 端到端延迟 = 完成时间 - 请求到达时间
        if self.request.finish_time is not None:
            e2e_latency = self.request.finish_time - self.request.arrival_time

        return {
            "request_id": self.request.request_id,
            "prompt_tokens": len(self.request.prompt_token_ids),
            "output_tokens": len(self.request.generated_token_ids),
            "ttft": ttft,
            "e2e_latency": e2e_latency,
        }
