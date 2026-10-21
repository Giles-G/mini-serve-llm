#!/usr/bin/env python3
"""Batch N decode parity vs eager reference for the Gemma4 engine."""
from __future__ import annotations

import json
import sys
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
PROMPTS = [
    "请详细介绍KV Cache的作用和实现原理。",
    "请用一句话说明注意力机制。",
    "Explain the purpose of a KV cache in transformer inference.",
    "What is 27 + 45? Answer with the number only.",
]


def main():
    weights, config, _ = B.load_weights(MODEL, "cuda", torch.float16, True, "auto")
    tokenizer = B.JsonTokenizer(MODEL)
    ids = [
        tokenizer.encode(B.render_chat_prompt(MODEL, p), add_special_tokens=False)
        for p in PROMPTS
    ]
    eager = Gemma4EagerTextRunner(config, weights)
    refs = []
    with torch.inference_mode():
        for row in ids:
            eager.reset_cache()
            logits = eager.prefill(torch.tensor(row, device="cuda", dtype=torch.long))
            out = [int(logits.argmax())]
            while len(out) < 8 and out[-1] not in EOS:
                logits = eager.decode_step(out[-1])
                out.append(int(logits.argmax()))
            refs.append(out)
    eager.reset_cache()
    del eager
    torch.cuda.empty_cache()

    results = []
    for n in (2, 4, 8, 16):
        rows = [ids[i % len(ids)] for i in range(n)]
        ec, cache, runner = B.build_runner(
            model_path=MODEL, model_config=config, weights=weights,
            device="cuda", dtype=torch.float16,
            max_tokens=max(len(x) for x in rows), history=16,
            eos_token_ids=EOS, max_batch_size=n,
        )
        engine = Stage5Engine(
            engine_config=ec, model_config=config, model_runner=runner,
            tokenizer=tokenizer,
        )
        rids = [
            engine.add_request(
                prompt_token_ids=row,
                sampling_params=SamplingParams(temperature=0.0, top_k=20, top_p=0.95),
                max_new_tokens=8,
            )
            for row in rows
        ]
        while engine.has_pending_work():
            engine.step()
        out = [engine.debug_request(r)["generated_token_ids"] for r in rids]
        expected = [refs[i % len(refs)] for i in range(n)]
        match = [a == b[: len(a)] for a, b in zip(out, expected)]
        results.append({
            "batch": n,
            "match": match,
            "batch_ids": out,
            "eager_ids": expected,
            "batch_graphs": sorted(runner._batch_graphs),
        })
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
        del engine, cache, runner
        torch.cuda.empty_cache()

    out = ROOT / "reports" / "gemma4_batch_parity.json"
    out.write_text(json.dumps({
        "note": "greedy decode, max_new_tokens=8; batch graph enabled",
        "results": results,
        "all_passed": all(all(r["match"]) for r in results),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
