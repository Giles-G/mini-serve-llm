#!/usr/bin/env python3
"""E-stage boundaries: 1k/2k/4k prompt + decode, engine vs eager.

Uses the production graph/paged engine and the same batched prefill path as
the benchmark. Attention kernel parity at 2k/4k is covered separately by the
kernel test matrix; this checks end-to-end token IDs and peak VRAM.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

import scripts.bench_gemma4_speed as B
from miniservellm.runtime.gemma4_runner import Gemma4EagerTextRunner
from miniservellm.runtime.inference_engine import Stage5Engine
from miniservellm.scheduler.request import SamplingParams

MODEL = ROOT / "models" / "gemma-4-E2B-it"
EOS = (1, 106, 50)
BASE = "请详细介绍KV Cache的作用和实现原理。"


def make_ids(tokenizer, target):
    text = " ".join([BASE] * (target // 8 + 32))
    templated = B.render_chat_prompt(MODEL, text) or text
    ids = tokenizer.encode(templated, add_special_tokens=False)
    if len(ids) < target:
        raw = tokenizer.encode(text, add_special_tokens=False)
        while len(ids) < target:
            ids.extend(raw)
    return ids[:target]


def eager_run(weights, config, ids, max_new):
    runner = Gemma4EagerTextRunner(config, weights)
    out = []
    with torch.inference_mode():
        logits = runner.prefill(torch.tensor(ids, device="cuda", dtype=torch.long))
        nxt = int(logits.argmax())
        out.append(nxt)
        while len(out) < max_new and nxt not in EOS:
            logits = runner.decode_step(nxt)
            nxt = int(logits.argmax())
            out.append(nxt)
    del runner
    gc.collect()
    torch.cuda.empty_cache()
    return out


def engine_run(weights, config, tokenizer, ids, max_new):
    engine_config, cache, runner = B.build_runner(
        model_path=MODEL, model_config=config, weights=weights,
        device="cuda", dtype=torch.float16, max_tokens=len(ids),
        history=max_new, eos_token_ids=EOS,
    )
    engine = Stage5Engine(
        engine_config=engine_config, model_config=config,
        model_runner=runner, tokenizer=tokenizer,
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    rid = engine.add_request(
        prompt_token_ids=ids,
        sampling_params=SamplingParams(temperature=0.0, top_k=20, top_p=0.95),
        max_new_tokens=max_new,
    )
    dev = torch.device("cuda")
    B.sync(dev)
    started = time.perf_counter()
    prefill_s = None
    while engine.has_pending_work():
        result = engine.step()
        B.sync(dev)
        if prefill_s is None and any(
            event.kind == "prefill_sampled_first_token" for event in result.events
        ):
            prefill_s = time.perf_counter() - started
    elapsed = time.perf_counter() - started
    info = engine.debug_request(rid)
    peak = torch.cuda.max_memory_allocated()
    free, _ = torch.cuda.mem_get_info()
    row = {
        "prompt_tokens": len(ids),
        "max_new_tokens": max_new,
        "generated_tokens": info["generated"],
        "finish_reason": info.get("finish_reason"),
        "ids": list(info["generated_token_ids"]),
        "prefill_s": round(prefill_s or elapsed, 3),
        "elapsed_s": round(elapsed, 3),
        "graph_mode": bool(getattr(runner, "_graph_mode", False)),
        "peak_alloc_gib": round(peak / 2**30, 2),
        "vram_free_after_gib": round(free / 2**30, 2),
    }
    del engine, cache, runner
    gc.collect()
    torch.cuda.empty_cache()
    return row


def main():
    torch.manual_seed(0)
    weights, config, load_s = B.load_weights(
        MODEL, "cuda", torch.float16, True, "auto"
    )
    tokenizer = B.JsonTokenizer(MODEL)
    cases = [(1024, 16), (2048, 16), (4096, 8)]
    rows = []
    for target, max_new in cases:
        ids = make_ids(tokenizer, target)
        eager = eager_run(weights, config, ids, max_new)
        engine = engine_run(weights, config, tokenizer, ids, max_new)
        first = next(
            (i for i in range(min(len(engine["ids"]), len(eager)))
             if engine["ids"][i] != eager[i]), None
        )
        row = {
            "target_tokens": target,
            "prompt_tokens": len(ids),
            "eager_tokens": len(eager),
            "engine": {k: v for k, v in engine.items() if k != "ids"},
            "ids_match_eager": engine["ids"] == eager[:max_new],
            "first_divergence": first,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    result = {
        "load_s": round(load_s, 2),
        "device": torch.cuda.get_device_name(0),
        "cases": rows,
        "all_passed": all(r["ids_match_eager"] for r in rows),
    }
    out = ROOT / "reports" / "gemma4_e_boundaries.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out}")
    print("E BOUNDARIES PASS" if result["all_passed"] else "E BOUNDARIES FAIL")
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
