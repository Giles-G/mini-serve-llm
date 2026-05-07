"""第二阶段批量调度 Demo

演示多请求进入队列后，engine 通过 Scheduler 每轮选择一批请求，
按 decode-first 策略逐 step 推进直到全部完成。

运行：python scripts/run_batch_demo.py
"""

import sys
from pathlib import Path

import torch

# 将项目根目录加入 Python 搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from miniservellm.config import ModelConfig
from miniservellm.model_adapter.hf_loader import HFLoader
from miniservellm.model_adapter.tokenizer_adapter import TokenizerAdapter
from miniservellm.model_adapter.hf_model_runner import HFModelRunner

from miniservellm.runtime.sampler import Sampler
from miniservellm.runtime.prefill import PrefillExecutor
from miniservellm.runtime.decode import DecodeExecutor
from miniservellm.runtime.inference_engine import InferenceEngine

from miniservellm.cache.kv_cache_manager import KVCacheManager
from miniservellm.scheduler.request_queue import RequestQueue
from miniservellm.scheduler.scheduler import Scheduler

from miniservellm.benchmark.bench_batching import run_batch_requests


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
    """构建第二阶段 InferenceEngine

    组装组件：
    HFLoader -> TokenizerAdapter -> HFModelRunner -> KVCacheManager ->
    PrefillExecutor / DecodeExecutor -> RequestQueue -> Scheduler -> InferenceEngine

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
    kv_cache_manager = KVCacheManager()
    sampler = Sampler()

    # 执行器接入 KVCacheManager
    prefill_executor = PrefillExecutor(model_runner, sampler, kv_cache_manager)
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

    prompts = [
        "请用三句话解释什么是 KV Cache。",
        "介绍一下 FlashAttention 的核心思想。",
        "为什么 decode 阶段通常比 prefill 阶段更 memory-bound？",
    ]

    outputs, metrics = run_batch_requests(
        engine,
        tokenizer_adapter,
        prompts=prompts,
        max_new_tokens=32,
    )

    print("\n===== OUTPUTS =====")
    for item in outputs:
        print(f"[{item['request_id']}]")
        print(item["text"])
        print("-" * 60)

    print("\n===== METRICS =====")
    for metric in metrics:
        print(metric)


if __name__ == "__main__":
    main()
