#!/usr/bin/env python3
"""Prefill TTFT / throughput / peak VRAM benchmark.

Compares the fused INT4 GEMV prefill (per-token weight reread) with the
dequantize-once + FP16 GEMM path used when the prompt chunk has at least
``MINI_LLM_GEMMA4_PREFILL_GEMM_MIN_ROWS`` rows.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

import scripts.bench_gemma4_speed as B
from miniservellm.runtime.inference_engine import Stage5Engine
from miniservellm.scheduler.request import SamplingParams

MODEL = ROOT / "models" / "gemma-4-E2B-it"
EOS = (1, 106, 50)
BASE = "请详细介绍KV Cache的作用和实现原理。"


def make_prompt(tokenizer, target):
    text = " ".join([BASE] * (target // 8 + 16))
    templated = B.render_chat_prompt(MODEL, text) or text
    ids = tokenizer.encode(templated, add_special_tokens=False)
    if len(ids) < target:
        raw = tokenizer.encode(text, add_special_tokens=False)
        while len(ids) < target:
            ids.extend(raw)
    return ids[:target]


def measure(weights, config, tokenizer, ids, min_rows):
    os.environ["MINI_LLM_GEMMA4_PREFILL_GEMM_MIN_ROWS"] = str(min_rows)
    engine_config, cache, runner = B.build_runner(
        model_path=MODEL, model_config=config, weights=weights,
        device="cuda", dtype=torch.float16, max_tokens=len(ids),
        history=1, eos_token_ids=EOS,
    )
    engine = Stage5Engine(
        engine_config=engine_config, model_config=config,
        model_runner=runner, tokenizer=tokenizer,
    )
    # Warm lazy CUDA modules and allocator paths on a short request, then
    # measure a fresh request on the same engine.
    warm = engine.add_request(
        prompt_token_ids=ids[: min(8, len(ids))],
        sampling_params=SamplingParams(temperature=0.0, top_k=20, top_p=0.95),
        max_new_tokens=1,
    )
    while engine.has_pending_work():
        engine.step()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()
    rid = engine.add_request(
        prompt_token_ids=ids,
        sampling_params=SamplingParams(temperature=0.0, top_k=20, top_p=0.95),
        max_new_tokens=1,
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    prefill_s = None
    while engine.has_pending_work():
        result = engine.step()
        torch.cuda.synchronize()
        if prefill_s is None and any(
            event.kind == "prefill_sampled_first_token" for event in result.events
        ):
            prefill_s = time.perf_counter() - started
    info = engine.debug_request(rid)
    peak = torch.cuda.max_memory_allocated()
    free, _ = torch.cuda.mem_get_info()
    del engine, cache, runner
    torch.cuda.empty_cache()
    return {
        "prompt_tokens": len(ids),
        "prefill_s": round(prefill_s or (time.perf_counter() - started), 3),
        "prefill_tok_s": round(len(ids) / (prefill_s or 1e-9), 1),
        "peak_alloc_gib": round(peak / 2**30, 2),
        "incremental_peak_gib": round((peak - base_alloc) / 2**30, 2),
        "free_after_gib": round(free / 2**30, 2),
        "generated_tokens": info["generated"],
        "finish_reason": info.get("finish_reason"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[16, 128, 512, 1024, 2048, 4096])
    parser.add_argument("--gemm-min-rows", type=int, default=64)
    parser.add_argument("--gemv-min-rows", type=int, default=100000)
    parser.add_argument("--output-json", type=Path,
                        default=ROOT / "reports" / "gemma4_e_prefill.json")
    args = parser.parse_args()
    weights, config, load_s = B.load_weights(
        MODEL, "cuda", torch.float16, True, "auto"
    )
    tokenizer = B.JsonTokenizer(MODEL)
    rows = []
    for size in args.sizes:
        ids = make_prompt(tokenizer, size)
        gemm = measure(weights, config, tokenizer, ids, args.gemm_min_rows)
        gemv = measure(weights, config, tokenizer, ids, args.gemv_min_rows)
        row = {
            "prompt_tokens": len(ids),
            "gemm": gemm,
            "gemv": gemv,
            "prefill_speedup": round(gemv["prefill_s"] / gemm["prefill_s"], 2),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    out = {"load_s": round(load_s, 2), "device": torch.cuda.get_device_name(0),
           "rows": rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved {args.output_json}")


if __name__ == "__main__":
    main()
