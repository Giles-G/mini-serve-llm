"""第六阶段 Benchmark 脚本（入口）

benchmark 核心逻辑已抽离到 miniservellm.benchmark：
- core.py: trial 执行与中位数统计
- timer.py: phase 计时与 monkey-patch
- report.py: 汇总展示
"""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
from pathlib import Path

import torch

# 将项目根目录加入 Python 搜索路径，使脚本可以直接从源码目录 import miniservellm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from miniservellm.benchmark import (
    PhaseTimer,
    median_run,
    patch_engine_with_timer,
    print_benchmark_summary,
    run_one_trial,
)
from miniservellm.config import EngineConfig
from miniservellm.model_adapter.adapters.qwen2_adapter import Qwen2Adapter
from miniservellm.model_adapter.hf_loader import load_model_bundle
from miniservellm.cache.kv_cache import KVCacheManager
from miniservellm.runtime.model_runner import TransformerModelRunner
from miniservellm.runtime.inference_engine import Stage5Engine
from miniservellm.scheduler.request import SamplingParams


def build_engine(ec: EngineConfig, mc, tok, w_cpu, adapter, quant_bits: int = 0, quant_group_size: int = 64):
    w = adapter.move_weights_to_device(w_cpu, device=ec.device, dtype=ec.dtype)
    kvm = KVCacheManager(ec, mc)
    mr = TransformerModelRunner(engine_config=ec, model_config=mc, weights=w, kv_cache_manager=kvm)
    if quant_bits > 0:
        mr.quantize_weights(bits=quant_bits, group_size=quant_group_size)
    return Stage5Engine(engine_config=ec, model_config=mc, model_runner=mr, tokenizer=tok)


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def build_fixed_context(tokenizer, context_len: int) -> list[int]:
    """构造长度精确且 token id 合法的固定上下文。"""
    if context_len <= 0:
        raise ValueError("--context-len must be greater than 0")
    seed_ids = tokenizer.encode("请简要介绍机器学习。", add_special_tokens=False)
    if not seed_ids:
        raise RuntimeError("Tokenizer returned an empty seed sequence.")
    repeats = (context_len + len(seed_ids) - 1) // len(seed_ids)
    return (seed_ids * repeats)[:context_len]


def release_engine(engine) -> None:
    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run_decode_only_trial(engine, prompt_token_ids, max_new: int, unrolled_steps: int = 0):
    """完成 prefill 后，仅计量首 token 之后的 decode。"""
    sp = SamplingParams(temperature=0.0, top_k=0, top_p=1.0, repetition_penalty=1.0)
    rid = engine.add_request(
        prompt_token_ids=prompt_token_ids,
        sampling_params=sp,
        max_new_tokens=max_new,
    )

    prefill_start = time.perf_counter()
    while engine.get_request(rid).total_generated_tokens() == 0:
        engine.step()
    sync_device(engine.model_runner.device)
    prefill_elapsed = time.perf_counter() - prefill_start

    request = engine.get_request(rid)
    decode_start_count = request.total_generated_tokens()
    sync_device(engine.model_runner.device)
    decode_start = time.perf_counter()
    if unrolled_steps:
        _run_unrolled_decode_only(engine, request, unrolled_steps)
    else:
        engine.run_until_all_finished(collect_results=False)
    sync_device(engine.model_runner.device)
    decode_elapsed = time.perf_counter() - decode_start

    decode_tokens = request.total_generated_tokens() - decode_start_count
    return {
        "prefill_elapsed": prefill_elapsed,
        "decode_elapsed": decode_elapsed,
        "decode_tokens": decode_tokens,
        "decode_tok_s": decode_tokens / decode_elapsed if decode_elapsed > 0 else 0.0,
    }


def _run_unrolled_decode_only(engine, request, unrolled_steps: int) -> None:
    """仅供 decode-only benchmark 使用的隔离 unrolled greedy decode。"""
    while request.can_decode_more():
        num_steps = min(unrolled_steps, request.max_new_tokens - request.total_generated_tokens())
        write_slots = engine.kv_cache_manager.ensure_slots_for_request(request, num_steps)
        input_token_id = request.pending_prefill_sample_token_id
        if input_token_id is None:
            input_token_id = request.last_token_id_for_decode_input()
        request.pending_prefill_sample_token_id = None

        # 只在 K 步都提交到 MPS 后才同步并取回 token。
        token_ids = engine.model_runner.forward_decode_greedy_unrolled(
            request=request,
            input_token_id=input_token_id,
            write_slots=write_slots,
        )
        sync_device(engine.model_runner.device)
        for token_id in token_ids.tolist():
            request.append_generated_token(int(token_id))

    request.mark_finished_max_new_tokens()
    engine._finish_request(request, [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--batch-list", default="1,4,8")
    parser.add_argument("--max-new", type=int, default=512)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--prompt", default="请用至少500字来介绍机器学习")
    parser.add_argument("--greedy", action="store_true", help="使用贪心解码（temperature=0）")
    parser.add_argument("--decode-only", action="store_true", help="batch=1，仅统计首 token 之后的 decode 吞吐")
    parser.add_argument("--context-len", type=int, default=None, help="decode-only 模式的精确 prompt token 数；不指定时用 --prompt 的真实 token 长度")
    parser.add_argument("--ollama-reference", type=float, default=90.0, help="用于计算相对性能的 Ollama FP16 eval rate")
    parser.add_argument("--kv-decode-block-reserve", type=int, default=2, help="为 decode 预留的最少空闲 KV block")
    parser.add_argument("--kv-reserve-relax-after-no-progress-steps", type=int, default=8, help="连续无进展多少步后放宽一次 KV 水位")
    parser.add_argument("--enable-compile", action="store_true", help="启用 torch.compile")
    parser.add_argument("--compile-mode", default="default", choices=["default", "reduce-overhead", "max-autotune"], help="torch.compile mode")
    parser.add_argument("--compile-fullgraph", action="store_true", help="torch.compile fullgraph 模式")
    parser.add_argument("--context-bucket-multiple", type=int, default=None, help="context长度分桶粒度；MPS默认32，显式传0关闭")
    parser.add_argument("--decode-batch-bucket-multiple", type=int, default=0, help="decode batch分桶粒度，0表示关闭")
    parser.add_argument(
        "--unrolled-decode-steps",
        type=int,
        default=0,
        help="实验：decode-only MPS greedy 每次连续提交 K 步；0 表示关闭",
    )
    # Stage 8 M5：kernel A/B 对比开关
    parser.add_argument("--no-custom-kernels", action="store_true",
                        help="禁用 mini_llm_kernels 自定义 CUDA kernel，强制走 PyTorch fallback（Stage 8 A/B 对比用）")
    parser.add_argument("--quant-bits", type=int, default=0, choices=[0, 4, 8],
                        help="权重量化位宽（0=FP16, 4=INT4, 8=INT8）")
    parser.add_argument("--quant-group-size", type=int, default=64,
                        help="量化 group size（默认 64）")
    args = parser.parse_args()

    # 必须在 import miniservellm 之前设置，否则 nn_ops.py 已经完成初始化
    if args.no_custom_kernels:
        import os
        os.environ["MINI_LLM_NO_CUSTOM_KERNELS"] = "1"

    ec = EngineConfig.create(
        device="auto",
        dtype="auto",
        block_size=16,
        num_gpu_blocks=1024,
        max_batch_size=16,
        max_tokens_per_step=512,
        max_prefill_tokens_per_step=512,
        max_decode_requests_per_step=16,
        prefill_chunk_size=128,
        default_temperature=0.7,
        default_top_k=20,
        default_top_p=0.9,
        kv_decode_block_reserve=args.kv_decode_block_reserve,
        kv_reserve_relax_after_no_progress_steps=args.kv_reserve_relax_after_no_progress_steps,
        enable_torch_compile=args.enable_compile,
        torch_compile_mode=args.compile_mode,
        torch_compile_fullgraph=args.compile_fullgraph,
        context_bucket_multiple=args.context_bucket_multiple,
        decode_batch_bucket_multiple=args.decode_batch_bucket_multiple,
    )

    from miniservellm.runtime.nn_ops import _HAS_CUSTOM_KERNELS
    print(
        f"[bench] device={ec.device} dtype={ec.dtype} "
        f"kv_reserve={ec.kv_decode_block_reserve} "
        f"kv_relax_after={ec.kv_reserve_relax_after_no_progress_steps} "
        f"compile={ec.enable_torch_compile} "
        f"compile_mode={ec.torch_compile_mode} "
        f"fullgraph={ec.torch_compile_fullgraph} "
        f"ctx_bucket={ec.context_bucket_multiple} "
        f"decode_batch_bucket={ec.decode_batch_bucket_multiple} "
        f"quant_bits={args.quant_bits} "
        f"custom_kernels={_HAS_CUSTOM_KERNELS}"
    )

    adapter = Qwen2Adapter()
    tok, _hfcfg, mc, _hf, w_cpu = load_model_bundle(
        adapter=adapter,
        model_name_or_path=args.model,
        trust_remote_code=False,
        load_model_device="cpu",
        load_dtype=None,
    )

    if tok.eos_token_id is not None:
        ec.eos_token_id = int(tok.eos_token_id)

    if args.greedy:
        sp = SamplingParams(temperature=0.0, top_k=0, top_p=1.0, repetition_penalty=1.0)
    else:
        sp = SamplingParams(temperature=0.7, top_k=20, top_p=0.9, repetition_penalty=1.1)

    def make_engine():
        return build_engine(ec, mc, tok, w_cpu, adapter,
                          quant_bits=args.quant_bits,
                          quant_group_size=args.quant_group_size)

    if args.decode_only:
        if args.max_new < 2:
            raise ValueError("--max-new must be at least 2 in --decode-only mode")
        if args.unrolled_decode_steps < 0:
            raise ValueError("--unrolled-decode-steps must be non-negative")
        if args.unrolled_decode_steps == 1:
            raise ValueError("--unrolled-decode-steps must be 0 or >= 2")
        if args.unrolled_decode_steps and ec.device.type != "mps":
            raise ValueError("unrolled decode experiment currently supports MPS only")

        # 构建 prompt token ids：指定 context_len 时用固定长度填充，否则用 --prompt 的真实 token
        if args.context_len is not None:
            prompt_token_ids = build_fixed_context(tok, args.context_len)
        else:
            chat_text = tok.apply_chat_template(
                [{"role": "user", "content": args.prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            prompt_token_ids = tok.encode(chat_text, add_special_tokens=False)

        context_len = len(prompt_token_ids)
        total_tokens = context_len + args.max_new
        kv_capacity = ec.block_size * ec.num_gpu_blocks
        if total_tokens > mc.max_position_embeddings:
            raise ValueError(
                f"context_len + max_new ({total_tokens}) exceeds model context "
                f"length ({mc.max_position_embeddings})"
            )
        if total_tokens > kv_capacity:
            raise ValueError(
                f"context_len + max_new ({total_tokens}) exceeds KV cache "
                f"capacity ({kv_capacity})"
            )
        ec.eos_token_id = None

        print(
            f"\n[decode-only] model={args.model} device={ec.device} dtype={ec.dtype} "
            f"batch=1 context_len={context_len} max_new={args.max_new} greedy=True "
            f"unrolled_steps={args.unrolled_decode_steps}"
        )
        engine = make_engine()
        print("[decode-only] warmup...")
        run_decode_only_trial(
            engine, prompt_token_ids, min(args.max_new, 16), args.unrolled_decode_steps
        )

        results = []
        for run_idx in range(args.runs):
            result = run_decode_only_trial(
                engine, prompt_token_ids, args.max_new, args.unrolled_decode_steps
            )
            results.append(result)
            print(
                f"run={run_idx + 1} prefill_s={result['prefill_elapsed']:.3f} "
                f"decode_tokens={result['decode_tokens']} "
                f"decode_s={result['decode_elapsed']:.3f} "
                f"decode_tok/s={result['decode_tok_s']:.2f}"
            )

        rates = [result["decode_tok_s"] for result in results]
        median_rate = statistics.median(rates)
        print("\n[decode-only summary]")
        print(f"median_decode_tok/s={median_rate:.2f}")
        print(f"min_decode_tok/s={min(rates):.2f}")
        print(f"max_decode_tok/s={max(rates):.2f}")
        if args.ollama_reference > 0:
            relative = median_rate / args.ollama_reference * 100.0
            gap = args.ollama_reference - median_rate
            print(f"ollama_fp16_reference={args.ollama_reference:.2f}")
            print(f"relative_to_ollama={relative:.1f}%")
            print(f"absolute_gap={gap:.2f} tok/s")
        release_engine(engine)
        return

    batches = [int(x) for x in args.batch_list.split(",") if x.strip()]

    print(f"\n[bench] prompt={args.prompt!r}  max_new={args.max_new}  runs={args.runs}  greedy={args.greedy}")
    print(f"{'N':>3} | {'elapsed(s)':>10} | {'gen_tokens':>10} | {'tok/s':>10}")
    print("-" * 50)

    summaries = {}
    for n in batches:
        prompts = [args.prompt] * n

        eng = make_engine()
        timer = PhaseTimer()
        patch_engine_with_timer(eng, timer)
        run_one_trial(eng, prompts, sp, args.max_new, timer)

        elapsed, tokens, outputs, report, elapsed_list, tok_list = median_run(
            make_engine,
            prompts,
            sp,
            args.max_new,
            runs=args.runs,
        )
        thr = tokens / elapsed if elapsed > 0 else 0.0
        print(f"{n:>3} | {elapsed:>10.2f} | {tokens:>10} | {thr:>10.2f}")
        summaries[n] = (
            elapsed,
            tokens,
            thr,
            outputs,
            report,
            list(zip(elapsed_list, tok_list)),
        )

    print_benchmark_summary(summaries)


if __name__ == "__main__":
    main()
