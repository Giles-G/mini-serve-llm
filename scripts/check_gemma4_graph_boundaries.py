#!/usr/bin/env python3
"""D-stage boundary checks: long graph context, graph reuse, batch fallback.

* 520-token prompt + 32 decode with CUDA Graph vs eager reference
* several sequential requests on one captured graph
* max_batch_size=2 -> non-graph paged fallback
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from dataclasses import replace
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


def build(weights, config, prompt_len, max_new, max_batch=1, paged=None):
    previous_graph = os.environ.get("MINI_LLM_GEMMA4_GRAPH")
    previous_paged = os.environ.get("MINI_LLM_GEMMA4_PAGED")
    if max_batch > 1 or paged is False:
        os.environ["MINI_LLM_GEMMA4_GRAPH"] = "0"
    if paged is not None:
        os.environ["MINI_LLM_GEMMA4_PAGED"] = "1" if paged else "0"
    try:
        engine_config, cache, runner = B.build_runner(
            model_path=MODEL, model_config=config, weights=weights,
            device="cuda", dtype=torch.float16, max_tokens=prompt_len,
            history=max_new, eos_token_ids=EOS,
        )
    finally:
        if previous_graph is None:
            os.environ.pop("MINI_LLM_GEMMA4_GRAPH", None)
        else:
            os.environ["MINI_LLM_GEMMA4_GRAPH"] = previous_graph
        if previous_paged is None:
            os.environ.pop("MINI_LLM_GEMMA4_PAGED", None)
        else:
            os.environ["MINI_LLM_GEMMA4_PAGED"] = previous_paged
    if max_batch > 1:
        engine_config = replace(
            engine_config,
            max_batch_size=max_batch,
            max_decode_requests_per_step=max_batch,
            max_tokens_per_step=max(engine_config.max_tokens_per_step, max_batch),
        )
    engine = Stage5Engine(engine_config=engine_config, model_config=config,
                          model_runner=runner, tokenizer=B.JsonTokenizer(MODEL))
    return engine, cache, runner


def run(engine, ids, max_new):
    rid = engine.add_request(
        prompt_token_ids=ids,
        sampling_params=SamplingParams(temperature=0.0, top_k=20, top_p=0.95),
        max_new_tokens=max_new,
    )
    dev = torch.device("cuda")
    B.sync(dev)
    started = time.perf_counter()
    while engine.has_pending_work():
        engine.step()
        B.sync(dev)
    elapsed = time.perf_counter() - started
    info = engine.debug_request(rid)
    return {
        "prompt_tokens": len(ids),
        "generated_tokens": info["generated"],
        "finish_reason": info.get("finish_reason"),
        "elapsed_s": round(elapsed, 3),
        "ids": list(info["generated_token_ids"]),
    }


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


def long_ids(tokenizer, target=520):
    base = tokenizer.encode(BASE, add_special_tokens=False)
    repeats = target // max(1, len(base)) + 2
    ids = tokenizer.encode(" ".join([BASE] * repeats), add_special_tokens=False)
    return ids[:target]


def main():
    weights, config, load_s = B.load_weights(MODEL, "cuda", torch.float16, True, "auto")
    tokenizer = B.JsonTokenizer(MODEL)
    ids_long = long_ids(tokenizer)
    results = {"load_s": round(load_s, 2)}

    # 1. Long context graph vs eager.
    free_before, _ = torch.cuda.mem_get_info()
    eager_ids = eager_run(weights, config, ids_long, 32)
    engine, cache, runner = build(weights, config, len(ids_long), 32)
    run(engine, ids_long, 8)  # capture on the same prompt length
    graph = run(engine, ids_long, 32)
    results["long_context"] = {
        "prompt_tokens": len(ids_long),
        "ids_match_eager": graph["ids"] == eager_ids[:32],
        "graph": {k: v for k, v in graph.items() if k != "ids"},
        "eager_tokens": len(eager_ids),
        "first_divergence": next(
            (i for i in range(min(len(graph["ids"]), len(eager_ids)))
             if graph["ids"][i] != eager_ids[i]), None
        ),
    }
    del engine, cache, runner
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Sequential requests on one captured graph.
    prompts = [
        BASE,
        "Explain the purpose and implementation of a KV cache.",
        "What is 27 + 45? Answer with the number only.",
        "请用一句话说明KV Cache。",
        "Write one short sentence about transformers.",
    ]
    engine, cache, runner = build(weights, config, 64, 16)
    run(engine, tokenizer.encode(B.render_chat_prompt(MODEL, BASE), add_special_tokens=False), 8)
    free_after_capture, _ = torch.cuda.mem_get_info()
    reuse = []
    for text in prompts:
        chat = B.render_chat_prompt(MODEL, text)
        ids = tokenizer.encode(chat if chat is not None else text, add_special_tokens=False)
        row = run(engine, ids, 16)
        reuse.append({
            "prompt": text,
            "prompt_tokens": row["prompt_tokens"],
            "generated_tokens": row["generated_tokens"],
            "finish_reason": row["finish_reason"],
            "elapsed_s": row["elapsed_s"],
            "nonempty": bool(row["ids"]),
        })
    free_after_reuse, _ = torch.cuda.mem_get_info()
    results["graph_reuse"] = {
        "requests": reuse,
        "vram_free_after_capture_gib": round(free_after_capture / 2**30, 2),
        "vram_free_after_reuse_gib": round(free_after_reuse / 2**30, 2),
        "all_finished": all(r["generated_tokens"] > 0 for r in reuse),
    }
    del engine, cache, runner
    gc.collect()
    torch.cuda.empty_cache()

    # 3. max_batch_size=2 -> non-graph paged fallback.
    engine, cache, runner = build(weights, config, 64, 16, max_batch=2)
    prompts2 = [
        B.render_chat_prompt(MODEL, "What is 12 + 30? Answer with the number only."),
        B.render_chat_prompt(MODEL, "请用一句话解释KV Cache。"),
    ]
    ids2 = [tokenizer.encode(p, add_special_tokens=False) for p in prompts2]
    rids = [
        engine.add_request(prompt_token_ids=p, sampling_params=SamplingParams(
            temperature=0.0, top_k=20, top_p=0.95), max_new_tokens=8)
        for p in ids2
    ]
    dev = torch.device("cuda")
    B.sync(dev)
    while engine.has_pending_work():
        engine.step()
        B.sync(dev)
    infos = [engine.debug_request(rid) for rid in rids]
    results["batch2"] = {
        "max_batch_size": engine.engine_config.max_batch_size,
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
        "all_finished": all(info["generated"] > 0 for info in infos),
    }
    del engine, cache, runner
    gc.collect()
    torch.cuda.empty_cache()

    # 4. Explicit dense fallback: force paged off and compare a decode run
    # that crosses the 128-token dense-mirror growth boundary.
    dense_prompt = tokenizer.encode(
        B.render_chat_prompt(MODEL, BASE), add_special_tokens=False
    )
    dense_new = 160
    dense_eager = eager_run(weights, config, dense_prompt, dense_new)
    engine, cache, runner = build(
        weights, config, len(dense_prompt), dense_new, paged=False
    )
    dense = run(engine, dense_prompt, dense_new)
    results["dense_fallback"] = {
        "prompt_tokens": len(dense_prompt),
        "max_new_tokens": dense_new,
        "ids_match_eager": dense["ids"] == dense_eager[:dense_new],
        "first_divergence": next(
            (i for i in range(min(len(dense["ids"]), len(dense_eager)))
             if dense["ids"][i] != dense_eager[i]), None
        ),
        "graph_mode": bool(getattr(runner, "_graph_mode", False)),
        "paged_attention": bool(getattr(runner, "_paged_attention", False)),
        "generated_tokens": dense["generated_tokens"],
        "finish_reason": dense["finish_reason"],
        "elapsed_s": dense["elapsed_s"],
    }
    del engine, cache, runner
    gc.collect()
    torch.cuda.empty_cache()
    results["all_passed"] = (
        results["long_context"]["ids_match_eager"]
        and results["graph_reuse"]["all_finished"]
        and results["batch2"]["all_finished"]
        and not results["batch2"]["graph_mode"]
        and results["batch2"]["paged_attention"]
        and results["dense_fallback"]["ids_match_eager"]
        and not results["dense_fallback"]["graph_mode"]
        and not results["dense_fallback"]["paged_attention"]
    )
    out = ROOT / "reports" / "gemma4_d_boundaries.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k not in ("graph_reuse", "batch2")}, ensure_ascii=False))
    print(json.dumps({"graph_reuse": results["graph_reuse"], "batch2": results["batch2"]}, ensure_ascii=False))
    print(f"saved {out}")
    print("D BOUNDARIES PASS" if results["all_passed"] else "D BOUNDARIES FAIL")
    if not results["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
