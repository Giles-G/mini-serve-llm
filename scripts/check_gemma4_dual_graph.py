#!/usr/bin/env python3
"""Short-context graph after a long split-K graph: no short-context regression."""
from __future__ import annotations

import json
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


def main():
    weights, config, _ = B.load_weights(MODEL, "cuda", torch.float16, True, "auto")
    tokenizer = B.JsonTokenizer(MODEL)
    long_ids = tokenizer.encode(
        B.render_chat_prompt(MODEL, " ".join([BASE] * 300)),
        add_special_tokens=False,
    )
    raw = tokenizer.encode(BASE, add_special_tokens=False)
    while len(long_ids) < 2048:
        long_ids.extend(raw)
    long_ids = long_ids[:2048]
    short_ids = tokenizer.encode(
        B.render_chat_prompt(MODEL, "请用一句话说明KV Cache。"),
        add_special_tokens=False,
    )
    ec, cache, runner = B.build_runner(
        model_path=MODEL, model_config=config, weights=weights,
        device="cuda", dtype=torch.float16, max_tokens=2048,
        history=16, eos_token_ids=EOS,
    )
    engine = Stage5Engine(
        engine_config=ec, model_config=config, model_runner=runner,
        tokenizer=tokenizer,
    )

    def run(ids, max_new):
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
        return {
            "generated_tokens": info["generated"],
            "prefill_s": round(prefill_s or elapsed, 3),
            "elapsed_s": round(elapsed, 3),
            "decode_ms": round(
                (elapsed - (prefill_s or elapsed)) * 1000
                / max(1, info["generated"] - 1), 2
            ),
            "ids": list(info["generated_token_ids"]),
        }

    long_row = run(long_ids, 2)
    short_first = run(short_ids, 16)
    short_second = run(short_ids, 16)
    payload = {
        "graphs": [str(k) for k in runner._graphs],
        "long": long_row,
        "short_first_after_long": short_first,
        "short_second": short_second,
        "vram_free_gib": round(torch.cuda.mem_get_info()[0] / 2**30, 2),
        "short_stable": short_first["ids"] == short_second["ids"],
    }
    out = ROOT / "reports" / "gemma4_dual_graph.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
