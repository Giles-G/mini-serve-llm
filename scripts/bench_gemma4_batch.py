#!/usr/bin/env python3
"""Batch decode throughput: per-request vs true batched vs batched CUDA Graph."""
from __future__ import annotations

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
PROMPT = "请详细介绍KV Cache的作用和实现原理。"


def run(engine, ids, max_new):
    rids = [
        engine.add_request(
            prompt_token_ids=row,
            sampling_params=SamplingParams(temperature=0.0, top_k=20, top_p=0.95),
            max_new_tokens=max_new,
        )
        for row in ids
    ]
    dev = torch.device("cuda")
    B.sync(dev)
    started = time.perf_counter()
    last_prefill = started
    while engine.has_pending_work():
        result = engine.step()
        B.sync(dev)
        for event in result.events:
            if event.kind == "prefill_sampled_first_token":
                last_prefill = time.perf_counter()
    finished = time.perf_counter()
    infos = [engine.debug_request(rid) for rid in rids]
    decode_s = max(0.0, finished - last_prefill)
    wall_s = max(1e-9, finished - started)
    decode_tokens = sum(max(0, info["generated"] - 1) for info in infos)
    output_tokens = sum(info["generated"] for info in infos)
    return {
        "wall_s": round(wall_s, 3),
        "last_prefill_s": round(last_prefill - started, 3),
        "aggregate_e2e_tok_s": round(output_tokens / wall_s, 1),
        "decode_s": round(decode_s, 3),
        "decode_tokens": decode_tokens,
        "aggregate_tok_s": round(decode_tokens / decode_s, 1) if decode_s else 0.0,
        "per_request_tok_s": round(
            (decode_tokens / len(infos)) / decode_s, 1
        ) if decode_s else 0.0,
        "generated": [info["generated"] for info in infos],
        "ids": [info["generated_token_ids"] for info in infos],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--output-json", type=Path,
                        default=ROOT / "reports" / "gemma4_batch_benchmark.json")
    args = parser.parse_args()
    max_new = args.max_new_tokens
    weights, config, load_s = B.load_weights(MODEL, "cuda", torch.float16, True, "auto")
    tokenizer = B.JsonTokenizer(MODEL)
    chat = B.render_chat_prompt(MODEL, PROMPT)
    one = tokenizer.encode(chat, add_special_tokens=False)
    ids = [one] * max(args.batch_sizes)
    all_modes = [
        ("per_request", "0", "0"),
        ("batched_no_graph", "0", "1"),
        ("batched_graph", "1", "1"),
    ]
    rows = []
    for n in args.batch_sizes:
        modes = all_modes if n <= 4 else all_modes[1:]
        for name, graph, batch_decode in modes:
            os.environ["MINI_LLM_GEMMA4_GRAPH"] = graph
            os.environ["MINI_LLM_GEMMA4_BATCH_DECODE"] = batch_decode
            ec, cache, runner = B.build_runner(
                model_path=MODEL, model_config=config, weights=weights,
                device="cuda", dtype=torch.float16,
                max_tokens=len(one), history=16, eos_token_ids=EOS,
                max_batch_size=n,
            )
            engine = Stage5Engine(
                engine_config=ec, model_config=config, model_runner=runner,
                tokenizer=tokenizer,
            )
            run(engine, ids[:n], max_new)   # capture / warm
            measured = run(engine, ids[:n], max_new)
            row = {
                "batch": n,
                "mode": name,
                "batch_graphs": sorted(getattr(runner, "_batch_graphs", {})),
                **measured,
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            del engine, cache, runner
            torch.cuda.empty_cache()
    payload = {
        "load_s": round(load_s, 2),
        "prompt_tokens": len(one),
        "max_new_tokens": max_new,
        "note": "decode tokens exclude the first token sampled by prefill",
        "rows": rows,
    }
    out = args.output_json
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
