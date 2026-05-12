"""Batched Engine Demo

第四阶段核心演示脚本，展示：
- Fresh Prefill：没有历史 cache 的请求 batched forward
- Incremental Prefill：有历史 cache 的请求逐请求 forward
- Decode：逐请求 forward
- Decode-first 调度
- Step 级事件输出

运行方式：
    python scripts/run_batched_engine_demo.py
"""

from __future__ import annotations

import sys
import uuid
import time
from pathlib import Path

import torch

# 将项目根目录加入 Python 搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from miniservellm.config import ModelConfig
from miniservellm.model_adapter.hf_loader import HFLoader
from miniservellm.model_adapter.tokenizer_adapter import TokenizerAdapter
from miniservellm.model_adapter.hf_model_runner import HFModelRunner

from miniservellm.runtime.sampler import Sampler
from miniservellm.runtime.attention_metadata import BatchedPrefillTensorBuilder
from miniservellm.runtime.batched_fresh_prefill import BatchedFreshPrefillExecutor
from miniservellm.runtime.incremental_prefill import IncrementalPrefillExecutor
from miniservellm.runtime.decode import DecodeExecutor
from miniservellm.runtime.inference_engine import HybridInferenceEngine
from miniservellm.runtime.outputs import StepEventType

from miniservellm.scheduler.request import Request, SamplingParams
from miniservellm.scheduler.request_queue import RequestQueue
from miniservellm.scheduler.scheduler import Scheduler

from miniservellm.benchmark.metrics import summarize_requests


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str):
    if device == "cpu":
        return "float32"
    return "float16"


def build_engine():
    device = pick_device()
    dtype = pick_dtype(device)

    model_cfg = ModelConfig(device=device, dtype=dtype)
    print(f"[INFO] device={model_cfg.device}, dtype={model_cfg.dtype}, model={model_cfg.model_name}")

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
    sampler = Sampler()

    # 请求队列
    queue = RequestQueue()

    # 调度器
    scheduler = Scheduler(
        max_decode_batch_size=8,
        max_prefill_batch_size=8,
        prefill_token_budget=32,
        max_prefill_chunk_size=16,
    )

    # request_index 共享引用
    request_index: dict = {}

    # BatchedPrefillTensorBuilder
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    tensor_builder = BatchedPrefillTensorBuilder(
        device=torch.device(model_cfg.device),
        pad_token_id=pad_token_id,
    )

    # 三个执行器
    fresh_prefill_executor = BatchedFreshPrefillExecutor(
        model_runner=model_runner,
        sampler=sampler,
        tensor_builder=tensor_builder,
        request_index=request_index,
    )
    incremental_prefill_executor = IncrementalPrefillExecutor(
        model_runner=model_runner,
        sampler=sampler,
    )
    decode_executor = DecodeExecutor(
        model_runner=model_runner,
        sampler=sampler,
    )

    engine = HybridInferenceEngine(
        tokenizer_adapter=tokenizer_adapter,
        model_runner=model_runner,
        sampler=sampler,
        queue=queue,
        scheduler=scheduler,
        fresh_prefill_executor=fresh_prefill_executor,
        incremental_prefill_executor=incremental_prefill_executor,
        decode_executor=decode_executor,
    )

    # 共享 request_index
    engine.request_index = request_index
    fresh_prefill_executor.request_index = request_index

    return engine, tokenizer_adapter, queue


def build_requests(tokenizer_adapter, prompts, max_new_tokens=32, chunk_size=12):
    tokenizer = tokenizer_adapter.tokenizer
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
        # 记录到达时间
        req.arrival_time = time.time()
        requests.append(req)

    return requests


def main():
    engine, tokenizer_adapter, queue = build_engine()

    prompts = [
        "请解释 chunked prefill 的作用，以及为什么它可以降低长 prompt 对交互式请求的影响。",
        "请解释 continuous batching 和 static batching 的差别。",
        "请解释为什么真实 serving 系统里 request lifecycle 和 cache lifecycle 都很重要。",
        "请解释 batched prefill 和 batched decode 在执行层面的主要差异。",
    ]

    requests = build_requests(
        tokenizer_adapter=tokenizer_adapter,
        prompts=prompts,
        max_new_tokens=1024,
        chunk_size=12,
    )

    # 文本缓冲区
    request_text_buffer = {}
    for req in requests:
        engine.add_request(req)
        request_text_buffer[req.request_id] = ""

    print("\n===== STEP EXECUTION =====")
    step_id = 0

    while engine.has_pending():
        step_id += 1
        events = engine.step()

        print(
            f"\n[STEP {step_id}] "
            f"waiting={queue.num_waiting()} "
            f"prefilling={queue.num_prefilling()} "
            f"decoding={queue.num_decoding()} "
            f"finished={queue.num_finished()}"
        )

        for event in events:
            # 拼接 decode token 文本
            if event.event_type == StepEventType.DECODE_TOKEN:
                text_delta = tokenizer_adapter.decode([event.payload.token_id])
                request_text_buffer[event.request_id] += text_delta
            elif event.event_type == StepEventType.PREFILL_TO_DECODE:
                text_delta = tokenizer_adapter.decode([event.payload.token_id])
                request_text_buffer[event.request_id] += text_delta

            print(event)

    finished_requests = [req for req in engine.request_index.values() if req.status == "finished"]

    print("\n===== STREAMED OUTPUTS =====")
    for req in finished_requests:
        print(f"[{req.request_id}]")
        print(request_text_buffer.get(req.request_id, ""))
        print("-" * 60)

    print("\n===== FINAL OUTPUTS =====")
    for req in finished_requests:
        print(f"[{req.request_id}]")
        print(tokenizer_adapter.decode(req.generated_token_ids))
        print("-" * 60)

    print("\n===== METRICS =====")
    for m in summarize_requests(finished_requests):
        print(m)


if __name__ == "__main__":
    main()
