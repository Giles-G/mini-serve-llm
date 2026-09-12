#!/usr/bin/env python3
"""Benchmark the independent MLX Qwen2 batch=1 FP16 greedy decode backend.

针对 MLX 后端（Apple Silicon）的 batch=1 贪心解码性能基准测试脚本。

测试内容：
- 仅统计 decode 阶段吞吐（prefill 已单独计时，不参与聚合速度计算）；
- 通过重复固定 prompt 构造指定长度的上下文，排除 tokenizer 差异带来的波动；
- 支持 eager / mx.compile 编译解码 / 自定义 Metal attention kernel 三种模式；
- 支持 FP16 与 INT4/INT8 量化权重对比；
- 输出多次运行的中位数吞吐，并可与 Ollama 参考速度对比相对百分比。

用法示例：
    python scripts/bench_mlx_decode.py --context-len 128 --max-new 256 --runs 5
    python scripts/bench_mlx_decode.py --compiled-decode --quant-bits 4
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

# 把仓库根目录加入 sys.path，保证脚本从任意工作目录运行都能 import miniservellm
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from miniservellm.config import MODEL_NAME
from miniservellm.model_adapter.adapters.qwen2_adapter import Qwen2Adapter
from miniservellm.mlx.runner import MLXQwen2Runner

import mlx.core as mx


def build_fixed_context(tokenizer, context_len: int) -> list[int]:
    """构造长度精确等于 context_len 的固定上下文 token 序列。

    以一句中文文本的 token 序列为种子，循环拼接后截断到目标长度。
    使用固定内容（而非随机 token）保证多次运行与多次实验之间可比。
    """
    if context_len <= 0:
        raise ValueError("--context-len must be greater than 0")
    seed = tokenizer.encode("请用至少1000字来介绍机器学习", add_special_tokens=False)
    if not seed:
        raise RuntimeError("Tokenizer returned an empty sequence")
    # 向上取整重复种子序列，再截断到 context_len
    return (seed * ((context_len + len(seed) - 1) // len(seed)))[:context_len]


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX Qwen2 batch=1 decode benchmark")
    parser.add_argument("--model", default=MODEL_NAME, help="模型名称或本地路径（默认取配置中的 MODEL_NAME）")
    parser.add_argument("--context-len", type=int, default=128, help="prefill 上下文长度（token 数）")
    parser.add_argument("--max-new", type=int, default=256, help="每次运行生成的最大新 token 数")
    parser.add_argument("--runs", type=int, default=5, help="重复测量次数，汇总取中位数")
    parser.add_argument("--batch-size", type=int, default=1, help="同长度 MLX greedy batch size")
    parser.add_argument("--compiled-decode", action="store_true", help="Use mx.compile Tensor-only batch=1 greedy Decode")
    parser.add_argument("--custom-attn", action="store_true", help="Use custom Metal kernel for decode attention")
    parser.add_argument("--quant-bits", type=int, default=0, choices=[0, 4, 8], help="Weight quantization bits (0=fp16, 4=INT4, 8=INT8)")
    parser.add_argument("--quant-group-size", type=int, default=64, help="Quantization group size")
    parser.add_argument("--ollama-reference", type=float, default=90.0, help="Ollama 参考解码速度（tok/s），用于计算相对百分比；<=0 时关闭")
    args = parser.parse_args()

    # ---------- 1. 加载 tokenizer 与模型配置 ----------
    adapter = Qwen2Adapter()
    tokenizer = adapter.load_tokenizer(args.model, trust_remote_code=False)
    hf_config = adapter.load_hf_config(args.model, trust_remote_code=False)
    model_config = adapter.convert_hf_config(hf_config)

    # 本地路径优先；否则从 HF 缓存中解析（仅用本地已下载的快照，不触发联网）
    model_path = Path(args.model)
    if not model_path.exists():
        from huggingface_hub import snapshot_download
        model_path = Path(snapshot_download(args.model, local_files_only=True))
    safetensors_path = model_path / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"Missing model.safetensors: {safetensors_path}")

    print(f"[mlx] loading {args.model} from {safetensors_path}")
    runner = MLXQwen2Runner(model_config, safetensors_path)

    # ---------- 2. 可选：量化权重 ----------
    # 量化后立刻 eval 一层权重，强制 Metal 图执行，确保量化真正落地、
    # 避免把量化开销延迟计入首条 warmup/测量
    if args.quant_bits > 0:
        print(f"[mlx] quantizing weights: bits={args.quant_bits} group_size={args.quant_group_size}")
        runner.model.quantize_weights(bits=args.quant_bits, group_size=args.quant_group_size)
        mx.eval(runner.model.weights.layers[0].qkv_proj)
        print(f"[mlx] quantization done")

    # ---------- 3. 参数合法性校验 ----------
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")
    # 编译解码与自定义 kernel 路径当前只实现了 batch=1 的张量布局
    if (args.compiled_decode or args.custom_attn) and args.batch_size != 1:
        raise ValueError("--compiled-decode/--custom-attn currently requires --batch-size 1")
    # 两种加速路径互斥：一条走 mx.compile，一条走自定义 Metal kernel
    if args.compiled_decode and args.custom_attn:
        raise ValueError("--compiled-decode and --custom-attn are mutually exclusive")

    # ---------- 4. 构造 prompt（batch 内共享同一份固定上下文） ----------
    prompt = build_fixed_context(tokenizer, args.context_len)
    prompts = [prompt] * args.batch_size
    print(f"[mlx] prompt = {prompt}")

    # ---------- 5. Warmup ----------
    # 模式标记仅用于日志展示
    if args.custom_attn:
        mode = "custom_attn"
    elif args.compiled_decode:
        mode = "compiled"
    else:
        mode = "eager"
    print(f"[mlx] warmup mode={mode}...")
    # eager 模式只 warmup 16 个 token 即可让 Metal buffer / KV cache 就绪；
    # compiled / custom_attn 路径的首次编译与 kernel 分配开销很大，
    # 需要完整跑满 max_new，才能避免把一次性开销带进第一次测量
    warmup_tokens = args.max_new if (args.compiled_decode or args.custom_attn) else min(args.max_new, 16)
    runner.generate_greedy_batch(
        prompts,
        warmup_tokens,
        disable_eos=True,  # 固定生成步数，保证解码吞吐不被提前 EOS 截断
        compiled_decode=args.compiled_decode,
        use_custom_attn=args.custom_attn,
    )

    # ---------- 6. 正式测量 ----------
    rates = []
    for index in range(args.runs):
        results = runner.generate_greedy_batch(
            prompts,
            args.max_new,
            disable_eos=True,
            compiled_decode=args.compiled_decode,
            use_custom_attn=args.custom_attn,
        )
        result = results[0]
        # 聚合吞吐 = 单请求速度 × batch 大小（batch 内是同长度并行解码）
        aggregate_rate = result.decode_tok_s * args.batch_size
        rates.append(aggregate_rate)
        print(
            f"run={index + 1} prefill_s={result.prefill_seconds:.3f} "
            f"decode_tokens/request={len(result.generated_token_ids) - 1} "  # 减 1：prefill 产出的首个 token 不算 decode 产出
            f"decode_s={result.decode_seconds:.3f} aggregate_tok/s={aggregate_rate:.2f} "
            f"per_request_tok/s={result.decode_tok_s:.2f}"
        )

    # ---------- 7. 汇总输出 ----------
    median_rate = statistics.median(rates)
    print("\n[mlx decode-only summary]")
    print(f"backend=mlx dtype={'int'+str(args.quant_bits) if args.quant_bits else 'fp16'} batch={args.batch_size} greedy=True compiled_decode={args.compiled_decode} custom_attn={args.custom_attn}")
    # KV cache 需容纳 prefill 上下文 + 生成的新 token
    print(f"context_len={args.context_len} max_new={args.max_new} kv_capacity={args.context_len + args.max_new}")
    print(f"median_decode_tok/s={median_rate:.2f} min={min(rates):.2f} max={max(rates):.2f}")
    # batch=1 时与 Ollama 参考值对比；batch>1 时为聚合吞吐，与单请求 eval 速度不可直接比较
    if args.ollama_reference > 0 and args.batch_size == 1:
        print(f"relative_to_ollama={median_rate / args.ollama_reference * 100.0:.1f}%")
    elif args.batch_size > 1:
        print("note=aggregate throughput; do not compare directly with single-request Ollama eval rate")


if __name__ == "__main__":
    main()
