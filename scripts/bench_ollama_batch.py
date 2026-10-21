#!/usr/bin/env python3
"""External Ollama multi-request baseline for Gemma4 E2B.

Measures N concurrent streaming requests with the same prompts and output
budget as scripts/bench_gemma4_batch.py. This is a system-level reference:
both sides use the same Gemma4 E2B checkpoint, but Ollama runs Q4_K_M on
llama.cpp while the project runs INT4 group-64 on its own CUDA kernels.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import requests

MODEL = ROOT / "models" / "gemma-4-E2B-it"
EOS = (1, 106, 50)
PROMPT = "请详细介绍KV Cache的作用和实现原理。"


def _cmd(args):
    try:
        return subprocess.run(
            args, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc}"


def _chat_prompt():
    # Use Ollama's chat API so its own template is applied; think=False keeps
    # the comparison against the project's non-thinking greedy output.
    return PROMPT


def _one_request(url, model, prompt, max_new, num_ctx, start_barrier, result):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "think": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0.0,
            "top_k": 20,
            "top_p": 0.95,
            "num_predict": max_new,
            "num_ctx": num_ctx,
        },
    }
    try:
        start_barrier.wait()
        started = time.perf_counter()
        ttft = None
        final = None
        with requests.post(
            f"{url}/api/chat", json=payload, stream=True, timeout=600
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                event = json.loads(line)
                message = event.get("message", {})
                if message.get("content") and ttft is None:
                    ttft = time.perf_counter() - started
                if event.get("done"):
                    final = event
                    break
        finished = time.perf_counter()
        result.update({
            "ok": final is not None,
            "ttft_s": round(ttft or 0.0, 4),
            "total_s": round(finished - started, 4),
            "eval_count": int(final.get("eval_count", 0)) if final else 0,
            "prompt_eval_count": int(final.get("prompt_eval_count", 0)) if final else 0,
            "eval_duration_s": (
                float(final.get("eval_duration", 0)) / 1e9 if final else 0.0
            ),
            "prompt_eval_duration_s": (
                float(final.get("prompt_eval_duration", 0)) / 1e9 if final else 0.0
            ),
            "response": (
                final.get("message", {}).get("content", "") if final else ""
            ),
        })
    except Exception as exc:  # noqa: BLE001
        result.update({"ok": False, "error": str(exc)})


def run_concurrent(url, model, prompt, max_new, num_ctx, n):
    results = [{} for _ in range(n)]
    barrier = threading.Barrier(n + 1)
    threads = [
        threading.Thread(
            target=_one_request,
            args=(url, model, prompt, max_new, num_ctx, barrier, results[i]),
            daemon=True,
        )
        for i in range(n)
    ]
    for thread in threads:
        thread.start()
    time.sleep(0.05)
    started = time.perf_counter()
    barrier.wait()
    for thread in threads:
        thread.join()
    finished = time.perf_counter()
    ok = [row for row in results if row.get("ok")]
    ttfts = [row["ttft_s"] for row in ok]
    last_ttft = max(ttfts) if ttfts else 0.0
    decode_window = max(1e-6, (finished - started) - last_ttft)
    output_tokens = sum(row["eval_count"] for row in ok)
    decode_tokens = sum(max(0, row["eval_count"] - 1) for row in ok)
    return {
        "concurrency": n,
        "ok_requests": len(ok),
        "wall_s": round(finished - started, 4),
        "ttft_median_s": round(sorted(ttfts)[len(ttfts) // 2], 4) if ttfts else None,
        "ttft_max_s": round(max(ttfts), 4) if ttfts else None,
        "output_tokens": output_tokens,
        "decode_tokens": decode_tokens,
        "aggregate_e2e_tok_s": round(output_tokens / (finished - started), 2)
        if output_tokens else 0.0,
        "aggregate_decode_tok_s": round(decode_tokens / decode_window, 2)
        if decode_tokens else 0.0,
        "per_request_median_tok_s": (
            round(sorted(
                [(row["eval_count"] - 1) / max(row["eval_duration_s"], 1e-9)
                 for row in ok if row["eval_count"] > 1]
            )[sum(1 for row in ok if row["eval_count"] > 1) // 2], 2)
            if any(row["eval_count"] > 1 for row in ok) else 0.0
        ),
        "prompt_eval_count_median": (
            sorted(row["prompt_eval_count"] for row in ok)[len(ok) // 2]
            if ok else None
        ),
        "requests": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--num-ctx", type=int, default=2048)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--output-json", type=Path,
        default=ROOT / "reports" / "gemma4_ollama_batch.json",
    )
    args = parser.parse_args()
    prompt = _chat_prompt()
    version = requests.get(f"{args.base_url}/api/version", timeout=10).json()
    show = requests.post(
        f"{args.base_url}/api/show", json={"model": args.model}, timeout=30
    ).json()
    details = show.get("details", {})
    model_info = show.get("model_info", {})

    # Load and warm the model once; exclude loading from the measurement.
    run_concurrent(
        args.base_url, args.model, prompt, 1, args.num_ctx, 1
    )
    rows = []
    for n in args.concurrency:
        rounds = [
            run_concurrent(
                args.base_url, args.model, prompt,
                args.max_new_tokens, args.num_ctx, n,
            )
            for _ in range(args.rounds)
        ]
        med = lambda key: sorted(r[key] for r in rounds)[len(rounds) // 2]
        row = {
            "concurrency": n,
            "rounds": rounds,
            "median_aggregate_e2e_tok_s": med("aggregate_e2e_tok_s"),
            "median_aggregate_decode_tok_s": med("aggregate_decode_tok_s"),
            "median_ttft_s": med("ttft_median_s"),
            "max_ttft_s": max(r["ttft_max_s"] for r in rounds),
            "median_per_request_tok_s": med("per_request_median_tok_s"),
        }
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "rounds"},
                         ensure_ascii=False), flush=True)

    payload = {
        "environment": {
            "model": args.model,
            "ollama_version": version,
            "quantization": details.get("quantization_level"),
            "parameter_size": details.get("parameter_size"),
            "context_length": model_info.get("gemma4.context_length"),
            "prompt_tokens": (
                rows[0]["rounds"][0]["requests"][0].get("prompt_eval_count")
                if rows and rows[0]["rounds"] else None
            ),
            "num_ctx": args.num_ctx,
            "max_new_tokens": args.max_new_tokens,
            "gpu_state": _cmd([
                "nvidia-smi", "--query-gpu=name,driver_version,clocks.sm,"
                "temperature.gpu,power.draw,memory.used", "--format=csv,noheader",
            ]),
            "ollama_ps": _cmd(["ollama", "ps"]),
            "ollama_num_parallel": "not exposed by the running service; "
            "OLLAMA_NUM_PARALLEL was not set in the unit file",
            "note": (
                "Ollama Q4_K_M system-level reference; not the same quantization "
                "or kernels as the project INT4 path"
            ),
        },
        "prompt": prompt,
        "api": "/api/chat with think=false",
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved {args.output_json}")


if __name__ == "__main__":
    main()
