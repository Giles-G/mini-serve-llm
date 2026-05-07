"""单请求 benchmark 辅助函数

用于在第二阶段 engine 上跑一个请求，方便和第一阶段单请求结果对比。
"""

import uuid

from miniservellm.scheduler.request import Request, SamplingParams
from miniservellm.benchmark.metrics import RequestMetrics


def run_single_request(engine, tokenizer_adapter, prompt: str, max_new_tokens: int = 32):
    """运行单个请求并返回生成文本和指标

    Args:
        engine: 第二阶段 InferenceEngine
        tokenizer_adapter: tokenizer 适配器
        prompt: 用户输入文本
        max_new_tokens: 最大生成 token 数

    Returns:
        (text, metrics) 元组
    """
    prompt_text = tokenizer_adapter.build_prompt(prompt)
    prompt_token_ids = tokenizer_adapter.encode(prompt_text)

    # EOS 作为停止 token
    stop_token_ids = []
    tokenizer = tokenizer_adapter.tokenizer
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)

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

    # 通过第二阶段 engine 的队列接口执行
    engine.add_request(req)
    finished_requests = engine.run_until_complete()
    final_req = finished_requests[0]

    text = tokenizer_adapter.decode(final_req.generated_token_ids)
    metrics = RequestMetrics(final_req).summary()
    return text, metrics
