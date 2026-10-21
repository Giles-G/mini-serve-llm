#!/usr/bin/env python3
"""Benchmark the Gemma4 safe paged attention kernel against the dense reference.

Parity is checked on the same inputs as the timing. This is decode attention
(one query per request), not prefill attention.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from mini_llm_kernels import _C
from mini_llm_kernels.kernels.gemma4_attention import (
    _HAS_GEMMA4_ATTN,
    _cuda_gemma4_attention,
    _gemma4_decode_attention_pytorch,
)


def make_case(dtype, D, H_q, H_kv, block_size, ctx, window, N):
    max_blocks = (ctx + block_size - 1) // block_size + 1
    num_blocks = N * max_blocks + 2
    q = torch.randn(N, H_q, D, device="cuda", dtype=dtype)
    k_cache = torch.randn(num_blocks, block_size, H_kv, D, device="cuda", dtype=dtype)
    v_cache = torch.randn(num_blocks, block_size, H_kv, D, device="cuda", dtype=dtype)
    table = torch.arange(num_blocks, device="cuda", dtype=torch.int32).view(1, -1).repeat(N, 1)
    for r in range(N):
        table[r] = torch.roll(table[r], r * max_blocks)
    table = table[:, :max_blocks].contiguous()
    lens = torch.full((N,), ctx, device="cuda", dtype=torch.int32)
    return q, k_cache, v_cache, table, lens


def bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2], samples[0], samples[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path,
                        default=Path("reports/gemma4_e_attention.json"))
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not _HAS_GEMMA4_ATTN:
        raise SystemExit("Gemma4 attention CUDA extension is unavailable")
    torch.manual_seed(1234)
    rows = []
    for size in (128, 512, 1024, 2048, 4096):
        if size >= 1024:
            # Keep the 4k reference run bounded while still timing the kernel.
            iters = max(5, args.iters // 2)
        else:
            iters = args.iters
        for D in (256, 512):
            for window in (0, 512):
                for N in (1, 2):
                    q, k_cache, v_cache, table, lens = make_case(
                        torch.float16, D, 8, 1, 128, size, window, N
                    )
                    safe = lambda: _cuda_gemma4_attention(
                        q, k_cache, v_cache, table, lens, window
                    )
                    warp_fn = getattr(
                        _C, "gemma4_decode_attention_warpsplit", None
                    )
                    warp = (
                        (lambda: warp_fn(q, k_cache, v_cache, table, lens, window))
                        if warp_fn is not None else None
                    )
                    split_fn = getattr(
                        _C, "gemma4_decode_attention_splitk", None
                    )
                    split = None
                    split_out = None
                    if split_fn is not None:
                        visible = size
                        chunks = max(1, (visible + 511) // 512)
                        po = torch.empty(
                            (N, 8, chunks, D), device="cuda", dtype=torch.float32
                        )
                        pm = torch.empty(
                            (N, 8, chunks), device="cuda", dtype=torch.float32
                        )
                        pl = torch.empty_like(pm)
                        split = lambda: split_fn(
                            q, k_cache, v_cache, table, lens, window,
                            po, pm, pl, 512,
                        )
                        split_out = split()
                    ref = lambda: _gemma4_decode_attention_pytorch(
                        q, k_cache, v_cache, table, lens, window
                    )
                    safe_out = safe()
                    warp_out = warp() if warp is not None else None
                    ref_out = ref()
                    max_abs = float((safe_out.float() - ref_out.float()).abs().max())
                    split_max_abs = (
                        float((split_out.float() - ref_out.float()).abs().max())
                        if split_out is not None else None
                    )
                    warp_max_abs = (
                        float((warp_out.float() - ref_out.float()).abs().max())
                        if warp_out is not None else None
                    )
                    assert torch.isfinite(safe_out).all()
                    # Parity is covered by the thresholded pytest matrix. This
                    # benchmark reports the observed long-context difference
                    # instead of silently relaxing an acceptance threshold.
                    if max_abs > 0.05:
                        print(json.dumps({
                            "warning": "long-context fp16 diff above 0.05",
                            "context": size, "head_dim": D,
                            "window": window, "batch": N,
                            "max_abs": max_abs,
                        }), flush=True)
                    s_med, s_min, s_max = bench(safe, args.warmup, iters)
                    r_med, r_min, r_max = bench(ref, args.warmup, iters)
                    w_row = None
                    if warp is not None:
                        w_med, w_min, w_max = bench(warp, args.warmup, iters)
                        w_row = {
                            "warp_ms_median": round(w_med, 4),
                            "warp_ms_range": [round(w_min, 4), round(w_max, 4)],
                            "warp_speedup_vs_safe": (
                                round(s_med / w_med, 3) if w_med else None
                            ),
                            "warp_speedup_vs_ref": (
                                round(r_med / w_med, 3) if w_med else None
                            ),
                            "warp_max_abs": warp_max_abs,
                        }
                    if split is not None:
                        k_med, k_min, k_max = bench(split, args.warmup, iters)
                        w_row = w_row or {}
                        w_row.update({
                            "splitk_ms_median": round(k_med, 4),
                            "splitk_ms_range": [round(k_min, 4), round(k_max, 4)],
                            "splitk_speedup_vs_warp": (
                                round(w_med / k_med, 3)
                                if warp is not None and k_med else None
                            ),
                            "splitk_max_abs": split_max_abs,
                        })
                    row = {
                        "context": size, "head_dim": D, "window": window,
                        "batch": N, "iters": iters, "max_abs": max_abs,
                        "safe_ms_median": round(s_med, 4),
                        "safe_ms_range": [round(s_min, 4), round(s_max, 4)],
                        "ref_ms_median": round(r_med, 4),
                        "ref_ms_range": [round(r_min, 4), round(r_max, 4)],
                        "speedup": round(r_med / s_med, 3) if s_med else None,
                        "safe_us_per_step": round(s_med * 1000, 2),
                    }
                    if w_row is not None:
                        row.update(w_row)
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    del q, k_cache, v_cache, table, lens, safe_out, ref_out, warp_out
                    torch.cuda.empty_cache()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "note": "decode attention, one query per request; FP16 inputs",
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
