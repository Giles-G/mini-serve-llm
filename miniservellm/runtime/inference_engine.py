import time


class InferenceEngine:
    def __init__(self, tokenizer_adapter, prefill_executor, decode_executor):
        self.tokenizer_adapter = tokenizer_adapter
        self.prefill_executor = prefill_executor
        self.decode_executor = decode_executor

    def generate(self, request) -> str:
        # prefill: 输入整段 prompt，产出第一个 token，并得到 past_key_values
        self.prefill_executor.run(request)

        finished = (
            len(request.generated_token_ids) >= request.sampling_params.max_new_tokens
            or request.generated_token_ids[-1] in request.sampling_params.stop_token_ids
        )

        # decode: 后续逐 token 解码
        while not finished:
            _, finished = self.decode_executor.step(request)

        request.finish_time = time.time()
        return self.tokenizer_adapter.decode(request.generated_token_ids)
