#!/usr/bin/env python3
"""Formal Gemma4 RTX 3060 acceptance run (no profiler).

Loads once, captures the CUDA graph with a warm-up request on the same engine,
then runs Chinese/English/arithmetic prompts for three measured rounds each.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
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
CASES = [
    ("zh", "请详细介绍KV Cache的作用和实现原理。", 128, 3),
    ("en", "Explain the purpose and implementation of a KV cache in transformer inference.", 128, 3),
    ("math", "What is 27 + 45? Answer with the number only.", 32, 3),
    ("zh_long", "请详细介绍KV Cache的作用和实现原理。", 256, 1),
]


def gpu_state():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm,temperature.gpu,power.draw,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return out
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc}"


def git_state(path=ROOT):
    try:
        head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return {"head": head, "dirty": bool(dirty), "dirty_files": dirty.splitlines()}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def build_engine(weights, config, max_prompt, max_new, model_path=MODEL):
    engine_config, cache, runner = B.build_runner(
        model_path=model_path, model_config=config, weights=weights,
        device="cuda", dtype=torch.float16, max_tokens=max_prompt,
        history=max_new, eos_token_ids=EOS,
    )
    engine = Stage5Engine(engine_config=engine_config, model_config=config,
                          model_runner=runner, tokenizer=B.JsonTokenizer(model_path))
    return engine, cache


def run_request(engine, ids, max_new):
    rid = engine.add_request(
        prompt_token_ids=ids,
        sampling_params=SamplingParams(temperature=0.0, top_k=20, top_p=0.95),
        max_new_tokens=max_new,
    )
    dev = torch.device("cuda")
    B.sync(dev)
    started = time.perf_counter()
    prefill_done = None
    while engine.has_pending_work():
        result = engine.step()
        B.sync(dev)
        now = time.perf_counter()
        if prefill_done is None and any(
            event.kind == "prefill_sampled_first_token" for event in result.events
        ):
            prefill_done = now
    finished = time.perf_counter()
    info = engine.debug_request(rid)
    prefill_s = (prefill_done or finished) - started
    decode_s = max(0.0, finished - (prefill_done or started))
    tokens = max(0, info["generated"] - 1)
    return {
        "prefill_s": round(prefill_s, 3),
        "decode_s": round(decode_s, 3),
        "decode_tokens": tokens,
        "decode_ms_per_token": round(decode_s * 1000 / tokens, 2) if tokens else None,
        "decode_tok_s": round(tokens / decode_s, 3) if decode_s else None,
        "end_to_end_tok_s": round(info["generated"] / (finished - started), 3),
        "generated_tokens": info["generated"],
        "finish_reason": info.get("finish_reason"),
        "generated_token_ids": list(info["generated_token_ids"]),
        "generated_text": engine.tokenizer.decode(
            info["generated_token_ids"], skip_special_tokens=True
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--rounds", type=int, default=None,
                        help="override measured rounds for every case")
    parser.add_argument("--warmup", type=int, default=8,
                        help="warm-up tokens used to capture the CUDA graph")
    parser.add_argument(
        "--output-json", type=Path,
        default=ROOT / "reports" / "gemma4_acceptance.json",
    )
    args = parser.parse_args()
    if args.rounds is not None and args.rounds < 1:
        parser.error("--rounds must be >= 1")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    model_path = args.model.resolve()
    torch.manual_seed(0)
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    print("loading weights...", flush=True)
    weights, config, load_s = B.load_weights(
        model_path, "cuda", torch.float16, True, "auto"
    )
    tokenizer = B.JsonTokenizer(model_path)
    prompts = []
    for name, text, max_new, rounds in CASES:
        if args.rounds is not None:
            rounds = args.rounds
        chat = B.render_chat_prompt(model_path, text)
        ids = tokenizer.encode(
            chat if chat is not None else text, add_special_tokens=False
        )
        prompts.append((name, text, max_new, rounds, ids))
    max_prompt = max(len(ids) for *_, ids in prompts)
    engine, cache = build_engine(
        weights, config, max_prompt, 256, model_path
    )

    # Capture the graph on the same engine, outside the measured interval.
    if args.warmup:
        run_request(engine, prompts[0][4], args.warmup)
    state_before = gpu_state()

    results = []
    for name, text, max_new, rounds, ids in prompts:
        rounds_out = []
        for _ in range(rounds):
            rounds_out.append(run_request(engine, ids, max_new))
        ms = sorted(r["decode_ms_per_token"] for r in rounds_out if r["decode_ms_per_token"])
        tps = sorted(r["decode_tok_s"] for r in rounds_out if r["decode_tok_s"])
        results.append({
            "name": name,
            "prompt": text,
            "prompt_tokens": len(ids),
            "max_new_tokens": max_new,
            "rounds": rounds_out,
            "decode_ms_median": ms[len(ms) // 2] if ms else None,
            "decode_ms_range": [ms[0], ms[-1]] if ms else None,
            "decode_tok_s_median": tps[len(tps) // 2] if tps else None,
            "decode_tok_s_range": [tps[0], tps[-1]] if tps else None,
            "first_text": rounds_out[0]["generated_text"],
            "first_finish": rounds_out[0]["finish_reason"],
            "first_ids": rounds_out[0]["generated_token_ids"],
        })
        print(json.dumps({k: results[-1][k] for k in (
            "name", "prompt_tokens", "decode_ms_median", "decode_ms_range",
            "decode_tok_s_median", "decode_tok_s_range", "first_finish"
        )}), flush=True)

    free_after, _ = torch.cuda.mem_get_info()
    payload = {
        "environment": {
            "device": props.name,
            "capability": f"{props.major}.{props.minor}",
            "vram_free_start_gib": round(free / 2**30, 2),
            "vram_total_gib": round(total / 2**30, 2),
            "vram_free_after_gib": round(free_after / 2**30, 2),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_state_before": state_before,
            "gpu_state_after": gpu_state(),
            "int4": True,
            "int4_impl": os.environ.get("MINI_LLM_INT4_IMPL", "pair_vec(default)"),
            "group_size": 64,
            "dtype": "fp16",
            "graph_default": True,
            "load_s": round(load_s, 2),
            "model": str(model_path),
            "git": git_state(),
            "kernels_git": git_state("/home/giles/work/mini-llm-kernels"),
            "kernels_extension": getattr(__import__("mini_llm_kernels"), "__file__", None),
        },
        "results": results,
    }
    out = args.output_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out}", flush=True)
    print("ACCEPTANCE RUN DONE", flush=True)


if __name__ == "__main__":
    main()
