"""CLI 对话脚本

第三阶段更新：使用 ChunkedPrefillExecutor + BlockAllocator。
由于第三阶段 Request 构造函数变更（sampling_params 必填），
此脚本也做了相应适配。

运行：python scripts/run_cli_chat.py
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
from miniservellm.scheduler.request import Request, SamplingParams
from miniservellm.scheduler.request_queue import RequestQueue
from miniservellm.scheduler.scheduler import Scheduler
from miniservellm.cache.block_allocator import BlockAllocator
from miniservellm.cache.kv_cache_manager import KVCacheManager
from miniservellm.benchmark.metrics import RequestMetrics


def pick_device():
    """自动选择推理设备

    优先级：CUDA > MPS (Apple Silicon) > CPU

    Returns:
        设备名称字符串
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str):
    """根据设备选择合适的精度

    CPU 使用 float32，GPU/MPS 使用 float16 以节省显存并加速推理。
    """
    if device == "cpu":
        return "float32"
    return "float16"


def build_engine():
    """构建第三阶段推理引擎

    使用 ChunkedPrefillExecutor + BlockAllocator + KVCacheManager。

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

    # 组装 engine 依赖组件
    tokenizer_adapter = TokenizerAdapter(tokenizer)
    model_runner = HFModelRunner(model, model_cfg.device)

    # BlockAllocator + KVCacheManager
    block_allocator = BlockAllocator(num_blocks=1024, block_size=16)
    kv_cache_manager = KVCacheManager(block_allocator=block_allocator)

    sampler = Sampler()
    # 使用 ChunkedPrefillExecutor
    prefill_executor = ChunkedPrefillExecutor(model_runner, sampler, kv_cache_manager)
    decode_executor = DecodeExecutor(model_runner, sampler, kv_cache_manager)

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
    engine, tokenizer_adapter = build_engine()
    tokenizer = tokenizer_adapter.tokenizer

    # 获取用户输入并构建 prompt
    user_input = input("User> ").strip()
    prompt_text = tokenizer_adapter.build_prompt(user_input)
    prompt_token_ids = tokenizer_adapter.encode(prompt_text)

    # EOS 作为停止 token
    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)

    # 构造单请求，使用 sampling_params 必填参数
    request = Request(
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
        # chunk_size 设大一些，单请求时一次性 prefill 整个 prompt
        chunk_size=len(prompt_token_ids),
    )

    # 兼容第一阶段的单请求 generate API
    output_text = engine.generate(request)

    print("\nAssistant>")
    print(output_text)

    metrics = RequestMetrics(request)
    print("\nMetrics>")
    print(metrics.summary())


if __name__ == "__main__":
    main()
