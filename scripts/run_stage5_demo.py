"""第五阶段演示脚本

展示 Paged KV Cache + 自研模型前向 + Adapter 架构。

运行方式：
    python scripts/run_stage5_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# 将项目根目录加入 Python 搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from miniservellm.config import EngineConfig
from miniservellm.model_adapter.adapters.qwen2_adapter import Qwen2Adapter
from miniservellm.model_adapter.hf_loader import load_model_bundle
from miniservellm.cache.kv_cache import KVCacheManager
from miniservellm.runtime.model_runner import TransformerModelRunner
from miniservellm.runtime.inference_engine import Stage5Engine
from miniservellm.scheduler.request import SamplingParams


def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    engine_config = EngineConfig.create(
        # 自动选择运行设备：优先 GPU/MPS，无法使用时回退 CPU
        device="auto",
        # 自动选择权重和计算 dtype：通常 GPU 上使用 fp16/bf16，CPU 上使用 fp32
        dtype="auto",
        # Paged KV Cache 的 block 大小：每个物理 block 能存 16 个 token 的 KV
        block_size=16,
        # GPU 上预分配的 KV Cache block 总数，总 token 容量 = num_gpu_blocks * block_size
        num_gpu_blocks=32,
        # 单轮 step 中最多参与计算的请求数（decode 请求 + prefill 请求总数）
        max_batch_size=8,
        # 单轮 step 最多处理的 token 总数：decode 每个请求算 1 个 token，prefill 按 chunk token 数计算
        max_tokens_per_step=64,
        # 单轮 step 中 prefill 最多消耗的 token 数，防止长 prompt prefill 挤占 decode
        max_prefill_tokens_per_step=48,
        # 单轮 step 中最多调度多少个 decode 请求；decode 阶段每个请求每轮生成 1 个 token
        # 所以这里等价于每轮最多生成 8 个 token（来自 8 个不同请求）
        max_decode_requests_per_step=8,
        # 单个请求每轮最多 prefill 的 prompt token 数，长 prompt 会被拆成多个 chunk 多轮处理
        prefill_chunk_size=32,
        # 默认采样温度：越高随机性越强，0 表示贪心选择最大概率 token
        default_temperature=0.8,
        # 默认 top-k 采样：只在概率最高的前 20 个 token 中采样
        default_top_k=20,
        # 默认 top-p 采样：只在累计概率达到 0.95 的候选 token 集合中采样
        default_top_p=0.95,
    )

    print("device:", engine_config.device)
    print("dtype:", engine_config.dtype)

    adapter = Qwen2Adapter()
    tokenizer, hf_config, model_config, hf_model, weights_cpu = load_model_bundle(
        adapter=adapter,
        model_name_or_path=model_name,
        trust_remote_code=False,
        load_model_device="cpu",
        load_dtype=None,
    )

    if tokenizer.eos_token_id is not None:
        engine_config.eos_token_id = int(tokenizer.eos_token_id)

    weights = adapter.move_weights_to_device(
        weights_cpu,
        device=engine_config.device,
        dtype=engine_config.dtype,
    )

    kv_cache_manager = KVCacheManager(engine_config, model_config)
    model_runner = TransformerModelRunner(
        engine_config=engine_config,
        model_config=model_config,
        weights=weights,
        kv_cache_manager=kv_cache_manager,
    )

    engine = Stage5Engine(
        engine_config=engine_config,
        model_config=model_config,
        model_runner=model_runner,
        tokenizer=tokenizer,
    )

    rid1 = engine.add_request(
        text="你好",
        sampling_params=SamplingParams(temperature=0.7, top_k=20, top_p=0.9, repetition_penalty=1.2),
        max_new_tokens=64,
    )
    rid2 = engine.add_request(
        text="今天的星期几？",
        sampling_params=SamplingParams(temperature=0.7, top_k=20, top_p=0.9, repetition_penalty=1.2),
        max_new_tokens=64,
    )

    step_results = engine.run_until_all_finished(max_steps=2000)

    for s in step_results:
        print(f"\n=== step {s.step_id} ===")
        print("fresh_prefill:", [r.request_id for r in s.plan.fresh_prefill_requests])
        print("incremental_prefill:", [r.request_id for r in s.plan.incremental_prefill_requests])
        print("decode:", [r.request_id for r in s.plan.decode_requests])
        for e in s.events:
            print(f"- {e.kind}: {e.request_id} {e.info}")
        print("kv_state:", s.kv_state)

    print("\n=== final outputs ===")
    print("\nRID1 full:")
    print(engine.get_full_text(rid1))

    print("\nRID2 full:")
    print(engine.get_full_text(rid2))


if __name__ == "__main__":
    main()
