"""第二阶段批量调度 Demo（第三阶段兼容版）

演示多请求进入队列后，engine 通过 Scheduler 每轮选择一批请求，
按 decode-first 策略逐 step 推进直到全部完成。

第三阶段更新：使用 ChunkedPrefillExecutor + BlockAllocator。

运行：python scripts/run_batch_demo.py
"""

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
    """自动选择推理设备，优先级：CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str):
    """根据设备选择推理精度"""
    if device == "cpu":
        return "float32"
    return "float16"


def build_engine():
    """构建第三阶段 InferenceEngine

    组装组件：
    HFLoader -> TokenizerAdapter -> HFModelRunner -> BlockAllocator ->
    KVCacheManager -> ChunkedPrefillExecutor / DecodeExecutor ->
    RequestQueue -> Scheduler -> InferenceEngine

    Returns:
        (engine, tokenizer_adapter) 元组
    """
    device = pick_device()
    dtype = pick_dtype(device)

    model_cfg = ModelConfig(
        device=device,
        dtype=dtype,
    )

    print(f"[INFO] device={model_cfg.device}, dtype={model_cfg.dtype}, model={model_cfg.model_name}")

    # 加载 tokenizer 和模型
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

    # 组装推理基础组件
    tokenizer_adapter = TokenizerAdapter(tokenizer)
    model_runner = HFModelRunner(model, model_cfg.device)

    # BlockAllocator + KVCacheManager
    block_allocator = BlockAllocator(num_blocks=1024, block_size=16)
    kv_cache_manager = KVCacheManager(block_allocator=block_allocator)

    sampler = Sampler()

    # 使用 ChunkedPrefillExecutor
    prefill_executor = ChunkedPrefillExecutor(model_runner, sampler, kv_cache_manager)
    decode_executor = DecodeExecutor(model_runner, sampler, kv_cache_manager)

    # 调度组件：请求队列 + decode-first scheduler
    request_queue = RequestQueue()
    scheduler = Scheduler(
        request_queue=request_queue,
        max_batch_size=4,
        decode_first=True,
    )

    engine = InferenceEngine(
        tokenizer_adapter=tokenizer_adapter,
        prefill_executor=prefill_executor,
        decode_executor=decode_executor,
        request_queue=request_queue,
        scheduler=scheduler,
        kv_cache_manager=kv_cache_manager,
    )
    return engine, tokenizer_adapter


def main():
    """运行批量请求 Demo"""
    engine, tokenizer_adapter = build_engine()
    tokenizer = tokenizer_adapter.tokenizer

    prompts = [
        "请用三句话解释什么是 KV Cache。",
        "介绍一下 FlashAttention 的核心思想。",
        "为什么 decode 阶段通常比 prefill 阶段更 memory-bound？",
    ]

    # 构建请求
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
                max_new_tokens=10240,
                temperature=0.7,
                top_k=20,
                top_p=0.9,
                stop_token_ids=stop_token_ids,
            ),
        )
        requests.append(req)

    # 添加请求到引擎
    for req in requests:
        engine.add_request(req)

    # 逐 step 推进直到全部完成
    finished_requests = engine.run_until_complete()

    print("\n===== OUTPUTS =====")
    for req in finished_requests:
        print(f"[{req.request_id}]")
        print(tokenizer_adapter.decode(req.generated_token_ids))
        print("-" * 60)

    print("\n===== METRICS =====")
    for m in summarize_requests(finished_requests):
        print(m)


if __name__ == "__main__":
    main()
