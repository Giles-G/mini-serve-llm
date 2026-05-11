"""Chunked Prefill + Paged KV Cache Demo

第三阶段核心演示脚本，展示：
- Chunked Prefill：长 prompt 分多个 chunk 逐步 prefill
- Paged KV Cache 元数据：BlockAllocator 管理 block 分配/扩容/回收
- Decode-first 调度：优先推进 decode 请求，剩余 batch slot 给 prefill
- Step 级事件输出：PREFILL_PROGRESS / PREFILL_TO_DECODE / DECODE_TOKEN / FINISHED

运行方式：
    python scripts/run_chunked_prefill_demo.py
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import torch

# 将项目根目录加入 Python 搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from miniservellm.config import ModelConfig
from miniservellm.model_adapter.hf_loader import HFLoader
from miniservellm.model_adapter.tokenizer_adapter import TokenizerAdapter
from miniservellm.model_adapter.hf_model_runner import HFModelRunner

from miniservellm.runtime.sampler import Sampler
from miniservellm.runtime.chunked_prefill import ChunkedPrefillExecutor
from miniservellm.runtime.decode import DecodeExecutor
from miniservellm.runtime.inference_engine import InferenceEngine

from miniservellm.cache.block_allocator import BlockAllocator
from miniservellm.cache.kv_cache_manager import KVCacheManager

from miniservellm.scheduler.request import Request, SamplingParams
from miniservellm.scheduler.request_queue import RequestQueue
from miniservellm.scheduler.scheduler import Scheduler

from miniservellm.benchmark.metrics import summarize_requests


def pick_device():
    """自动选择可用设备：cuda > mps > cpu"""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str):
    """根据设备选择数据类型"""
    if device == "cpu":
        return "float32"
    return "float16"


def build_engine():
    """构建推理引擎

    加载模型、tokenizer，组装 ChunkedPrefillExecutor + DecodeExecutor +
    BlockAllocator + KVCacheManager + Scheduler + InferenceEngine。
    """
    device = pick_device()
    dtype = pick_dtype(device)

    model_cfg = ModelConfig(
        device=device,
        dtype=dtype,
    )

    print(
        f"[INFO] device={model_cfg.device}, dtype={model_cfg.dtype}, model={model_cfg.model_name}"
    )

    loader = HFLoader()

    tokenizer = loader.load_tokenizer(
        model_cfg.model_name,
        trust_remote_code=model_cfg.trust_remote_code,
    )
    model = loader.load_model(
        model_cfg.model_name,
        device=model_cfg.device,
        dtype=model_cfg.dtype,
        trust_remote_code=model_cfg.trust_remote_code,
    )

    tokenizer_adapter = TokenizerAdapter(tokenizer)
    model_runner = HFModelRunner(model, model_cfg.device)

    # BlockAllocator：管理 paged KV cache 的 block 资源
    block_allocator = BlockAllocator(
        num_blocks=1024,
        block_size=16,
    )
    kv_cache_manager = KVCacheManager(block_allocator=block_allocator)

    sampler = Sampler()
    # 使用 ChunkedPrefillExecutor 替代第二阶段的 PrefillExecutor
    prefill_executor = ChunkedPrefillExecutor(model_runner, sampler, kv_cache_manager)
    decode_executor = DecodeExecutor(model_runner, sampler, kv_cache_manager)

    request_queue = RequestQueue()
    # max_prefill_tokens_per_step=24：每步 prefill 总预算 24 tokens
    # 配合 chunk_size=12，两个 prefill 请求可以同时推进
    scheduler = Scheduler(
        request_queue=request_queue,
        max_batch_size=4,
        decode_first=True,
        max_prefill_tokens_per_step=24,
    )

    engine = InferenceEngine(
        tokenizer_adapter=tokenizer_adapter,
        prefill_executor=prefill_executor,
        decode_executor=decode_executor,
        request_queue=request_queue,
        scheduler=scheduler,
        kv_cache_manager=kv_cache_manager,
    )
    return engine, tokenizer_adapter, block_allocator


def build_requests(tokenizer_adapter, prompts, max_new_tokens=32, chunk_size=12):
    """构建请求列表

    Args:
        tokenizer_adapter: tokenizer 适配器
        prompts: prompt 文本列表
        max_new_tokens: 每个请求最大生成 token 数
        chunk_size: 每次 prefill chunk 的 token 数

    Returns:
        Request 对象列表
    """
    tokenizer = tokenizer_adapter.tokenizer

    # 自动获取 eos_token_id 作为 stop token
    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)

    requests = []
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
            chunk_size=chunk_size,
        )
        requests.append(req)

    return requests


def main():
    """主流程：构建引擎 -> 添加请求 -> 逐步执行 -> 输出结果"""
    engine, tokenizer_adapter, block_allocator = build_engine()

    prompts = [
        "请详细解释什么是 chunked prefill，以及它为什么能够提升长 prompt 与短交互请求混部场景下的系统体验。",
        "请解释 paged kv cache 的核心设计思想，并说明为什么 block table 是 serving 系统中的重要抽象。",
        "请分析在 LLM serving 中 decode-first 调度策略的优点、代价和适用场景。",
    ]

    requests = build_requests(
        tokenizer_adapter,
        prompts,
        max_new_tokens=1024,
        chunk_size=12,
    )

    # 文本缓冲区，用于拼接流式输出
    request_text_buffer = {}

    # 将请求加入引擎
    for req in requests:
        engine.add_request(req)
        request_text_buffer[req.request_id] = ""

    print("\n===== STEP EXECUTION =====")
    step_id = 0
    while engine.has_pending():
        step_id += 1
        results = engine.step()

        # 每步打印队列状态和 block 使用情况
        print(
            f"\n[STEP {step_id}] "
            f"waiting={engine.request_queue.num_waiting()} "
            f"prefilling={engine.request_queue.num_prefilling()} "
            f"decoding={engine.request_queue.num_decoding()} "
            f"finished={engine.request_queue.num_finished()} "
            f"free_blocks={block_allocator.num_free_blocks()} "
            f"used_blocks={block_allocator.num_used_blocks()}"
        )

        for r in results:
            # 拼接文本增量
            if r.text_delta:
                request_text_buffer[r.request_id] += r.text_delta

            print(
                {
                    "request_id": r.request_id,
                    "event_type": r.event_type,
                    "next_token_id": r.next_token_id,
                    "finished": r.finished,
                    "text_delta": r.text_delta,
                    "metadata": r.metadata,
                }
            )

    finished_requests = engine.request_queue.all_finished_requests()

    print("\n===== STREAMED OUTPUTS =====")
    for req in finished_requests:
        print(f"[{req.request_id}]")
        print(request_text_buffer[req.request_id])
        print("-" * 60)

    print("\n===== FINAL OUTPUTS (DECODED FROM TOKENS) =====")
    for req in finished_requests:
        print(f"[{req.request_id}]")
        print(tokenizer_adapter.decode(req.generated_token_ids))
        print("-" * 60)

    print("\n===== METRICS =====")
    for m in summarize_requests(finished_requests):
        print(m)


if __name__ == "__main__":
    main()
