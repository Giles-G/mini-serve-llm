"""推理引擎

串联 Prefill 和 Decode 两个阶段，完成从 prompt 到生成文本的全流程。
"""

import time


class InferenceEngine:
    """推理引擎，协调 prefill 和 decode 执行器

    Attributes:
        tokenizer_adapter: tokenizer 适配器，用于解码生成的 token ids
        prefill_executor: Prefill 执行器，处理 prompt 首次前向
        decode_executor: Decode 执行器，逐 token 自回归解码
    """

    def __init__(self, tokenizer_adapter, prefill_executor, decode_executor):
        self.tokenizer_adapter = tokenizer_adapter
        self.prefill_executor = prefill_executor
        self.decode_executor = decode_executor

    def generate(self, request) -> str:
        """执行完整的文本生成流程

        流程：
        1. Prefill：输入整段 prompt，产出第一个 token 并得到 KV Cache
        2. Decode：循环逐 token 解码，直到满足终止条件
        3. 将生成的 token ids 解码为文本

        Args:
            request: 请求对象，包含 prompt、采样参数等

        Returns:
            生成的文本字符串
        """
        # prefill: 输入整段 prompt，产出第一个 token，并得到 past_key_values
        self.prefill_executor.run(request)

        # 检查首 token 是否已满足终止条件
        finished = (
            len(request.generated_token_ids) >= request.sampling_params.max_new_tokens
            or request.generated_token_ids[-1] in request.sampling_params.stop_token_ids
        )

        # decode: 后续逐 token 解码，直到终止
        while not finished:
            _, finished = self.decode_executor.step(request)

        # 记录完成时间，用于计算端到端延迟
        request.finish_time = time.time()

        # 将生成的 token ids 解码为文本
        return self.tokenizer_adapter.decode(request.generated_token_ids)
