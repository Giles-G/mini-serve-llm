#!/usr/bin/env python3
"""Small batch=1 Gemma4 MLX eager/quantized decode benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from miniservellm.model_adapter.adapters.gemma4_hf_model import load_gemma4_config
from miniservellm.model_adapter.gemma4_config import convert_gemma4_config
from miniservellm.mlx.gemma4 import (
    Gemma4MLXCache,
    build_gemma4_mlx,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-new", type=int, default=16)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--fused-greedy", action="store_true")
    args = parser.parse_args()

    path = Path(args.model).expanduser()
    started = time.perf_counter()
    config = convert_gemma4_config(load_gemma4_config(path))
    model = build_gemma4_mlx(path, config)
    if args.quantize:
        model.quantize_weights(bits=4, group_size=64)
    load_s = time.perf_counter() - started

    # Token IDs are deliberately fixed here: this is a kernel benchmark, not
    # a tokenizer/chat-template benchmark.
    prompt = mx.array([[1, 2]], dtype=mx.uint32)
    cache = Gemma4MLXCache(
        config,
        max_context=max(64, int(prompt.shape[1]) + args.max_new),
    )
    started = time.perf_counter()
    logits = model.forward(prompt, cache)
    mx.eval(logits)
    prefill_s = time.perf_counter() - started

    token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)
    mx.eval(token)
    compiled_decode = None
    if args.fused_greedy:
        compiled_decode = model.get_compiled_decode_greedy()
    elif args.compiled:
        compiled_decode = model.get_compiled_decode()
    started = time.perf_counter()
    for _ in range(args.max_new - 1):
        if compiled_decode is None:
            logits = model.decode(token, cache)
        else:
            result = compiled_decode(
                token, mx.array(cache.offset, dtype=mx.uint32),
                cache.k_layers, cache.v_layers,
            )
            if args.fused_greedy:
                token, cache.k_layers, cache.v_layers = result
                cache.offset += 1
                mx.eval(token, cache.k_layers, cache.v_layers)
                continue
            logits, cache.k_layers, cache.v_layers = result
            cache.offset += 1
        mx.eval(logits)
        token = mx.argmax(logits, axis=-1).astype(mx.uint32)
        mx.eval(token)
    decode_s = time.perf_counter() - started
    print(
        json.dumps(
            {
                "model": str(path),
                "quantized_linear": args.quantize,
                "compiled": args.compiled,
                "fused_greedy": args.fused_greedy,
                "load_s": round(load_s, 3),
                "prefill_s": round(prefill_s, 3),
                "decode_tokens": max(0, args.max_new - 1),
                "decode_s": round(decode_s, 3),
                "decode_tok_s": round(max(0, args.max_new - 1) / decode_s, 3)
                if decode_s
                else 0.0,
            }
        )
    )


if __name__ == "__main__":
    main()
