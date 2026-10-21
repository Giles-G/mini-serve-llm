#!/usr/bin/env python3
"""4k KV/transient-memory boundary for batch>1 requests.

Large prefills are serialized by the scheduler's token budget; this checks
that a 4k prompt can coexist with another request without OOM and that two 4k
requests complete sequentially.
"""
from __future__ import annotations

import gc
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


def build(weights, config, tokenizer, max_tokens):
    previous = os.environ.get("MINI_LLM_GEMMA4_GRAPH")
    os.environ["MINI_LLM_GEMMA4_GRAPH"] = "0"
    try:
        ec, cache, runner = B.build_runner(
        model_path=MODEL, model_config=config, weights=weights,
        device="cuda", dtype=torch.float16, max_tokens=max_tokens,
            history=4, eos_token_ids=EOS, max_batch_size=2,
        )
    finally:
        if previous is None:
            os.environ.pop("MINI_LLM_GEMMA4_GRAPH", None)
        else:
            os.environ["MINI_LLM_GEMMA4_GRAPH"] = previous
    return Stage5Engine(
        engine_config=ec, model_config=config,
        model_runner=runner, tokenizer=tokenizer,
    ), cache, runner


def run(weights, config, tokenizer, prompts, max_news, max_tokens):
    engine, cache, runner = build(weights, config, tokenizer, max_tokens)
    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()
    rids = [
        engine.add_request(
            prompt_token_ids=ids,
            sampling_params=SamplingParams(temperature=0.0, top_k=20, top_p=0.95),
            max_new_tokens=n,
        )
        for ids, n in zip(prompts, max_news)
    ]
    dev = torch.device("cuda")
    B.sync(dev)
    started = time.perf_counter()
    while engine.has_pending_work():
        engine.step()
        B.sync(dev)
    elapsed = time.perf_counter() - started
    infos = [engine.debug_request(rid) for rid in rids]
    peak = torch.cuda.max_memory_allocated()
    free, _ = torch.cuda.mem_get_info()
    result = {
        "prompt_tokens": [len(p) for p in prompts],
        "elapsed_s": round(elapsed, 2),
        "peak_alloc_gib": round(peak / 2**30, 2),
        "incremental_gib": round((peak - base_alloc) / 2**30, 2),
        "free_gib": round(free / 2**30, 2),
        "graph_mode": bool(getattr(runner, "_graph_mode", False)),
        "paged_attention": bool(getattr(runner, "_paged_attention", False)),
        "requests": [
            {
                "generated_tokens": info["generated"],
                "finish_reason": info.get("finish_reason"),
                "ids": list(info["generated_token_ids"]),
            }
            for info in infos
        ],
    }
    del engine, cache, runner
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    weights, config, _ = B.load_weights(MODEL, "cuda", torch.float16, True, "auto")
    tokenizer = B.JsonTokenizer(MODEL)
    long_ids = tokenizer.encode(
        B.render_chat_prompt(MODEL, " ".join([BASE] * 600)),
        add_special_tokens=False,
    )
    raw = tokenizer.encode(BASE, add_special_tokens=False)
    while len(long_ids) < 4096:
        long_ids.extend(raw)
    long_ids = long_ids[:4096]
    short_ids = tokenizer.encode(
        B.render_chat_prompt(MODEL, "请用一句话说明KV Cache。"),
        add_special_tokens=False,
    )
    results = [
        {
            "name": "4k_plus_short",
            **run(weights, config, tokenizer, [long_ids, short_ids], [2, 8], len(long_ids)),
        },
        {
            "name": "two_4k",
            **run(weights, config, tokenizer, [long_ids, long_ids], [8, 8], len(long_ids)),
        },
    ]
    payload = {
        "device": torch.cuda.get_device_name(0),
        "note": "batch=2; large prefills are serialized by the scheduler token budget",
        "results": results,
    }
    out = ROOT / "reports" / "gemma4_concurrency.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in results:
        print(json.dumps({k: v for k, v in row.items() if k != "requests"}), flush=True)
        print(json.dumps({"name": row["name"], "requests": row["requests"]}), flush=True)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
