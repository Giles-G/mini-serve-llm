class RequestMetrics:
    def __init__(self, request):
        self.request = request

    def summary(self):
        ttft = None
        e2e_latency = None

        if self.request.first_token_time is not None:
            ttft = self.request.first_token_time - self.request.arrival_time

        if self.request.finish_time is not None:
            e2e_latency = self.request.finish_time - self.request.arrival_time

        return {
            "request_id": self.request.request_id,
            "prompt_tokens": len(self.request.prompt_token_ids),
            "output_tokens": len(self.request.generated_token_ids),
            "ttft": ttft,
            "e2e_latency": e2e_latency,
        }
