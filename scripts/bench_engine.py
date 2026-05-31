"""第六阶段 Benchmark 脚本（入口）

benchmark 核心逻辑已抽离到 miniservellm.benchmark：
- core.py: trial 执行与中位数统计
- timer.py: phase 计时与 monkey-patch
- report.py: 汇总展示
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def build_engine(ec: EngineConfig, mc, tok, w_cpu, adapter):
    w = adapter.move_weights_to_device(w_cpu, device=ec.device, dtype=ec.dtype)
    kvm = KVCacheManager(ec, mc)
    mr = TransformerModelRunner(engine_config=ec, model_config=mc, weights=w, kv_cache_manager=kvm)
    return Stage5Engine(engine_config=ec, model_config=mc, model_runner=mr, tokenizer=tok)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--batch-list", default="1,4,8")
    parser.add_argument("--max-new", type=int, default=512)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--prompt", default="用400字左右介绍一下机器学习")
    parser.add_argument("--greedy", action="store_true", help="使用贪心解码（temperature=0）")
    parser.add_argument("--kv-decode-block-reserve", type=int, default=2, help="为 decode 预留的最少空闲 KV block")
    parser.add_argument("--kv-reserve-relax-after-no-progress-steps", type=int, default=8, help="连续无进展多少步后放宽一次 KV 水位")
    parser.add_argument("--enable-compile", action="store_true", help="启用 torch.compile")
    parser.add_argument("--compile-mode", default="default", choices=["default", "reduce-overhead", "max-autotune"], help="torch.compile mode")
    parser.add_argument("--compile-fullgraph", action="store_true", help="torch.compile fullgraph 模式")
    parser.add_argument("--context-bucket-multiple", type=int, default=0, help="context长度分桶粒度，0表示关闭")
    parser.add_argument("--decode-batch-bucket-multiple", type=int, default=0, help="decode batch分桶粒度，0表示关闭")
    # Stage 8 M5：kernel A/B 对比开关
    parser.add_argument("--no-custom-kernels", action="store_true",
                        help="禁用 mini_llm_kernels 自定义 CUDA kernel，强制走 PyTorch fallback（Stage 8 A/B 对比用）")
    args = parser.parse_args()

    # 必须在 import miniservellm 之前设置，否则 nn_ops.py 已经完成初始化
    if args.no_custom_kernels:
        import os
        os.environ["MINI_LLM_NO_CUSTOM_KERNELS"] = "1"

    ec = EngineConfig.create(
        device="auto",
        dtype="auto",
        block_size=16,
        num_gpu_blocks=128,
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
        return build_engine(ec, mc, tok, w_cpu, adapter)

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
