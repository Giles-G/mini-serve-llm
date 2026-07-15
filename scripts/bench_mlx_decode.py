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


def build_fixed_context(tokenizer, context_len: int) -> list[int]:
    if context_len <= 0:
        raise ValueError("--context-len must be greater than 0")
    seed = tokenizer.encode("请用至少500字来介绍机器学习", add_special_tokens=False)
    if not seed:
        raise RuntimeError("Tokenizer returned an empty sequence")
    return (seed * ((context_len + len(seed) - 1) // len(seed)))[:context_len]


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX Qwen2 batch=1 decode benchmark")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--runs", type=int, default=5)
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
    prompt = build_fixed_context(tokenizer, args.context_len)
    print(f"[mlx] prompt = {prompt}")

    print("[mlx] warmup...")
    runner.generate_greedy(prompt, min(args.max_new, 16), disable_eos=True)

    rates = []
    for index in range(args.runs):
        result = runner.generate_greedy(prompt, args.max_new, disable_eos=True)
        rates.append(result.decode_tok_s)
        print(
            f"run={index + 1} prefill_s={result.prefill_seconds:.3f} "
            f"decode_tokens={len(result.generated_token_ids) - 1} "
            f"decode_s={result.decode_seconds:.3f} decode_tok/s={result.decode_tok_s:.2f}"
        )

    median_rate = statistics.median(rates)
    print("\n[mlx decode-only summary]")
    print("backend=mlx dtype=fp16 batch=1 greedy=True")
    print(f"context_len={args.context_len} max_new={args.max_new}")
    print(f"median_decode_tok/s={median_rate:.2f} min={min(rates):.2f} max={max(rates):.2f}")
    if args.ollama_reference > 0:
        print(f"relative_to_ollama={median_rate / args.ollama_reference * 100.0:.1f}%")


if __name__ == "__main__":
    main()
