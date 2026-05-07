"""批量请求 benchmark 辅助函数

构造多个 Request，加入第二阶段 engine，运行到全部完成后返回输出和指标。
"""

import uuid

from miniservellm.scheduler.request import Request, SamplingParams
from miniservellm.benchmark.metrics import summarize_requests


def build_requests(tokenizer_adapter, prompts: list[str], max_new_tokens: int = 32):
    """根据 prompt 列表构造 Request 列表

    Args:
        tokenizer_adapter: tokenizer 适配器
        prompts: 用户输入列表
        max_new_tokens: 每个请求最大生成 token 数

    Returns:
        Request 列表
    """
    requests = []
    tokenizer = tokenizer_adapter.tokenizer

    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)

    for prompt in prompts:
        prompt_text = tokenizer_adapter.build_prompt(prompt)
        prompt_token_ids = tokenizer_adapter.encode(prompt_text)

        req = Request(
            request_id=str(uuid.uuid4()),
            prompt=prompt_text,
            prompt_token_ids=prompt_token_ids,
            sampling_params=SamplingParams(
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_k=0,
                top_p=1.0,
                stop_token_ids=stop_token_ids,
            ),
        )
        requests.append(req)

    return requests


def run_batch_requests(engine, tokenizer_adapter, prompts: list[str], max_new_tokens: int = 32):
    """运行多个请求，模拟 continuous batching

    Args:
        engine: 第二阶段 InferenceEngine
        tokenizer_adapter: tokenizer 适配器
        prompts: 用户输入列表
        max_new_tokens: 每个请求最大生成 token 数

    Returns:
        (outputs, metrics) 元组
    """
    requests = build_requests(tokenizer_adapter, prompts, max_new_tokens=max_new_tokens)

    # 多请求持续进入队列
    for req in requests:
        engine.add_request(req)

    # 逐 step 推进直到全部完成
    finished_requests = engine.run_until_complete()

    outputs = []
    for req in finished_requests:
        outputs.append(
            {
                "request_id": req.request_id,
                "text": tokenizer_adapter.decode(req.generated_token_ids),
            }
        )

    metrics = summarize_requests(finished_requests)
    return outputs, metrics
