"""第五阶段演示脚本

展示 Paged KV Cache + 自研模型前向 + Adapter 架构。

运行方式：
    python scripts/run_stage5_demo.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 将项目根目录加入 Python 搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from miniservellm.config import EngineConfig
from miniservellm.model_adapter.adapter_factory import create_model_adapter
from miniservellm.model_adapter.hf_loader import load_model_bundle
from miniservellm.cache.kv_cache import KVCacheManager
from miniservellm.runtime.model_runner import TransformerModelRunner
from miniservellm.runtime.inference_engine import Stage5Engine
from miniservellm.scheduler.request import SamplingParams


def run_gemma4_eager(
    *,
    model_name: str,
    device: str,
    dtype: str,
    prompt: str,
    max_new_tokens: int,
) -> None:
    """Gemma4 generation through the Stage5Engine service path (stage E)."""
    import time

    import torch
    from transformers import AutoTokenizer

    from miniservellm.cache.gemma4_paged_kv_cache import Gemma4PagedKVCacheManager
    from miniservellm.model_adapter.adapters.gemma4_hf_model import (
        load_gemma4_config,
        load_gemma4_text_weights,
    )
    from miniservellm.model_adapter.gemma4_config import convert_gemma4_config
    from miniservellm.runtime.gemma4_engine_runner import Gemma4EngineModelRunner
    from miniservellm.runtime.inference_engine import Stage5Engine
    from miniservellm.scheduler.request import SamplingParams

    model_path = Path(model_name).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Gemma4 checkpoint not found: {model_path}")

    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    # bf16 matches the checkpoint natively; fp32 doubles memory for parity runs.
    torch_dtype = torch.bfloat16 if dtype in {"auto", "fp16", "bf16"} else torch.float32

    started = time.perf_counter()
    model_config = convert_gemma4_config(load_gemma4_config(model_path))
    weights = load_gemma4_text_weights(model_path, device=device, dtype=torch_dtype)

    engine_config = EngineConfig.create(
        device=device,
        dtype="bf16" if torch_dtype == torch.bfloat16 else "fp32",
        block_size=128,
        # Physical paged blocks: 128 blocks x 128 tokens = 16k token capacity,
        # ~300MB bf16 across the D=256/D=512 cache groups.
        num_gpu_blocks=128,
        max_batch_size=8,
        max_tokens_per_step=256,
        max_prefill_tokens_per_step=128,
        max_decode_requests_per_step=8,
        prefill_chunk_size=128,
        default_temperature=0.0,
        default_top_k=20,
        default_top_p=0.95,
        model_param_count=0,
        num_hidden_layers=model_config.num_hidden_layers,
        num_kv_heads=model_config.num_key_value_heads,
        head_dim=512,
    )
    engine_config.eos_token_id = (1, 106, 50)

    kv_cache_manager = Gemma4PagedKVCacheManager(engine_config, model_config)
    model_runner = Gemma4EngineModelRunner(engine_config, model_config, weights, kv_cache_manager)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    engine = Stage5Engine(
        engine_config=engine_config,
        model_config=model_config,
        model_runner=model_runner,
        tokenizer=tokenizer,
    )
    print(
        f"[gemma4-engine] model={model_path} device={device} dtype={torch_dtype} "
        f"load_s={time.perf_counter() - started:.1f}"
    )

    prompts = [text.strip() for text in prompt.split("|") if text.strip()]
    request_ids = []
    gen_started = time.perf_counter()
    for text in prompts:
        request_ids.append(
            engine.add_request(
                text=text,
                sampling_params=SamplingParams(temperature=0.0, top_k=20, top_p=0.95),
                max_new_tokens=max_new_tokens,
            )
        )
    engine.run_until_all_finished(max_steps=100000, collect_results=False)
    total_s = time.perf_counter() - gen_started

    total_generated = 0
    for idx, request_id in enumerate(request_ids, start=1):
        info = engine.debug_request(request_id)
        total_generated += info["generated"]
        print(f"\nRID{idx} finish={info['finish_reason']} generated={info['generated']}")
        print(engine.get_text(request_id))
    print(
        f"\n[gemma4-engine] requests={len(request_ids)} generated_tokens={total_generated} "
        f"total_s={total_s:.2f} tok/s={total_generated / total_s:.2f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--prompt", default="你好，请简要介绍一下你自己。")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face to download config/weights when --model is a repo id.",
    )
    args = parser.parse_args()
    model_name = args.model

    from transformers import AutoConfig
    model_path = Path(model_name).expanduser()
    is_local_path = model_path.exists()
    if model_path.is_absolute() and not is_local_path:
        raise FileNotFoundError(
            f"Local model path does not exist: {model_path}\n"
            "Download the checkpoint first or pass a valid Hugging Face repo id."
        )
    local_files_only = not args.allow_download
    hf_config = AutoConfig.from_pretrained(
        str(model_path) if is_local_path else model_name,
        trust_remote_code=False,
        local_files_only=local_files_only,
    )
    adapter = create_model_adapter(hf_config)

    model_type = str(getattr(hf_config, "model_type", "")).lower()
    if model_type in {"gemma4", "gemma4_text"}:
        run_gemma4_eager(
            model_name=model_name,
            device=args.device,
            dtype=args.dtype,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
        )
        return

    tokenizer, hf_config, model_config, hf_model, weights_cpu = load_model_bundle(
        adapter=adapter,
        model_name_or_path=model_name,
        trust_remote_code=False,
        load_model_device="cpu",
        load_dtype=None,
    )
    layer_specs = getattr(model_config, "layer_specs", [])

    engine_config = EngineConfig.create(
        # 自动选择运行设备：优先 GPU/MPS，无法使用时回退 CPU
        device=args.device,
        # 自动选择权重和计算 dtype：通常 GPU 上使用 fp16/bf16，CPU 上使用 fp32
        dtype=args.dtype,
        # Paged KV Cache 的 block 大小：每个物理 block 能存 16 个 token 的 KV
        block_size=128,
        # GPU 上预分配的 KV Cache block 总数，总 token 容量 = num_gpu_blocks * block_size
        num_gpu_blocks=0,
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
        model_param_count=sum(p.numel() for p in hf_model.parameters()),
        num_hidden_layers=model_config.num_hidden_layers,
        num_kv_heads=(
            layer_specs[0].num_key_value_heads
            if layer_specs
            else model_config.num_key_value_heads
        ),
        head_dim=max(
            (spec.head_dim for spec in layer_specs),
            default=model_config.head_dim,
        ),
    )

    print("device:", engine_config.device)
    print("dtype:", engine_config.dtype)

    if tokenizer.eos_token_id is not None:
        eos_id = tokenizer.eos_token_id
        engine_config.eos_token_id = int(eos_id[0] if isinstance(eos_id, list) else eos_id)

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

    request_texts = [
        "用300字给我介绍一些elasticsearch的原理和用途吧。",
        "用300字给我介绍一下transformer中主流的kv驱逐策略吧，以及它们的优劣各是什么。",
        "用300字解释一下大模型推理中的prefill和decode阶段分别在做什么。",
        "用300字介绍一下PagedAttention的核心思想和它解决了什么问题。",
        "用300字说明一下KV Cache为什么能加速大模型自回归生成。",
        "用300字介绍一下RMSNorm和LayerNorm的区别。",
        "用300字解释一下RoPE位置编码的基本原理。",
        "用300字介绍一下GQA相比MHA的优势和代价。",
        "用300字说明一下LLM服务中的continuous batching是什么。",
        "用300字介绍一下top-k、top-p和temperature采样参数的作用。",
    ]
    sampling_params = SamplingParams(temperature=0.7, top_k=20, top_p=0.9, repetition_penalty=1.2)
    request_ids = []
    for text in request_texts:
        request_ids.append(
            engine.add_request(
                text=text,
                sampling_params=sampling_params,
                max_new_tokens=512,
            )
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
    for idx, request_id in enumerate(request_ids, start=1):
        print(f"\nRID{idx} full:")
        print(engine.get_full_text(request_id))
if __name__ == "__main__":
    main()
