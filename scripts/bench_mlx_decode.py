#!/usr/bin/env python3
"""Benchmark the independent MLX Qwen2 batch=1 FP16 greedy decode backend."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from miniservellm.config import MODEL_NAME
from miniservellm.model_adapter.adapters.qwen2_adapter import Qwen2Adapter
from miniservellm.mlx.runner import MLXQwen2Runner

import mlx.core as mx


def build_fixed_context(tokenizer, context_len: int) -> list[int]:
    if context_len <= 0:
        raise ValueError("--context-len must be greater than 0")
    seed = tokenizer.encode("请用至少1000字来介绍机器学习", add_special_tokens=False)
    if not seed:
        raise RuntimeError("Tokenizer returned an empty sequence")
    return (seed * ((context_len + len(seed) - 1) // len(seed)))[:context_len]


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX Qwen2 batch=1 decode benchmark")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1, help="同长度 MLX greedy batch size")
    parser.add_argument("--compiled-decode", action="store_true", help="Use mx.compile Tensor-only batch=1 greedy Decode")
    parser.add_argument("--custom-attn", action="store_true", help="Use custom Metal kernel for decode attention")
    parser.add_argument("--quant-bits", type=int, default=0, choices=[0, 4, 8], help="Weight quantization bits (0=fp16, 4=INT4, 8=INT8)")
    parser.add_argument("--quant-group-size", type=int, default=64, help="Quantization group size")
    parser.add_argument("--ollama-reference", type=float, default=90.0)
    args = parser.parse_args()

    adapter = Qwen2Adapter()
    tokenizer = adapter.load_tokenizer(args.model, trust_remote_code=False)
    hf_config = adapter.load_hf_config(args.model, trust_remote_code=False)
    model_config = adapter.convert_hf_config(hf_config)
    model_path = Path(args.model)
    if not model_path.exists():
        from huggingface_hub import snapshot_download
        model_path = Path(snapshot_download(args.model, local_files_only=True))
    safetensors_path = model_path / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"Missing model.safetensors: {safetensors_path}")

    print(f"[mlx] loading {args.model} from {safetensors_path}")
    runner = MLXQwen2Runner(model_config, safetensors_path)
    if args.quant_bits > 0:
        print(f"[mlx] quantizing weights: bits={args.quant_bits} group_size={args.quant_group_size}")
        runner.model.quantize_weights(bits=args.quant_bits, group_size=args.quant_group_size)
        mx.eval(runner.model.weights.layers[0].qkv_proj)
        print(f"[mlx] quantization done")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")
    if (args.compiled_decode or args.custom_attn) and args.batch_size != 1:
        raise ValueError("--compiled-decode/--custom-attn currently requires --batch-size 1")
    if args.compiled_decode and args.custom_attn:
        raise ValueError("--compiled-decode and --custom-attn are mutually exclusive")
    prompt = build_fixed_context(tokenizer, args.context_len)
    prompts = [prompt] * args.batch_size
    print(f"[mlx] prompt = {prompt}")

    if args.custom_attn:
        mode = "custom_attn"
    elif args.compiled_decode:
        mode = "compiled"
    else:
        mode = "eager"
    print(f"[mlx] warmup mode={mode}...")
    warmup_tokens = args.max_new if (args.compiled_decode or args.custom_attn) else min(args.max_new, 16)
    runner.generate_greedy_batch(
        prompts,
        warmup_tokens,
        disable_eos=True,
        compiled_decode=args.compiled_decode,
        use_custom_attn=args.custom_attn,
    )

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
        aggregate_rate = result.decode_tok_s * args.batch_size
        rates.append(aggregate_rate)
        print(
            f"run={index + 1} prefill_s={result.prefill_seconds:.3f} "
            f"decode_tokens/request={len(result.generated_token_ids) - 1} "
            f"decode_s={result.decode_seconds:.3f} aggregate_tok/s={aggregate_rate:.2f} "
            f"per_request_tok/s={result.decode_tok_s:.2f}"
        )

    median_rate = statistics.median(rates)
    print("\n[mlx decode-only summary]")
    print(f"backend=mlx dtype={'int'+str(args.quant_bits) if args.quant_bits else 'fp16'} batch={args.batch_size} greedy=True compiled_decode={args.compiled_decode} custom_attn={args.custom_attn}")
    print(f"context_len={args.context_len} max_new={args.max_new} kv_capacity={args.context_len + args.max_new}")
    print(f"median_decode_tok/s={median_rate:.2f} min={min(rates):.2f} max={max(rates):.2f}")
    if args.ollama_reference > 0 and args.batch_size == 1:
        print(f"relative_to_ollama={median_rate / args.ollama_reference * 100.0:.1f}%")
    elif args.batch_size > 1:
        print("note=aggregate throughput; do not compare directly with single-request Ollama eval rate")


if __name__ == "__main__":
    main()
