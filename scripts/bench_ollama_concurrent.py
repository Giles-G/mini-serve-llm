#!/usr/bin/env python3
"""Benchmark concurrent local Ollama generation requests.

Each run sends ``--requests`` independent requests to Ollama while limiting the
number in flight to ``--concurrency``. The script reports both end-to-end wall
clock metrics and Ollama's server-side token counters.

Examples:
    # Generate a fixed synthetic prompt with approximately 128 words.
    python scripts/bench_ollama_concurrent.py \
        --model qwen2.5:0.5b-instruct-fp16 \
        --concurrency 4 --requests 16 --context-len 128 --max-new 256

    # Reproduce a concrete user prompt across concurrent requests.
    python scripts/bench_ollama_concurrent.py \
        --model qwen2.5:0.5b \
        --prompt "请用至少1000字来介绍新疆地区，字数必须要够" \
        --concurrency 1 --requests 1 --runs 1

The benchmark uses ``stream=false`` so one HTTP response represents one logical
request. This makes concurrency and wall-clock accounting deterministic. It is
an end-to-end throughput benchmark, not a first-token-latency benchmark.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from miniservellm.model_adapter.adapters.qwen2_adapter import Qwen2Adapter


@dataclass(frozen=True)
class RequestResult:
    """Metrics collected for one completed Ollama request."""

    request_index: int
    wall_seconds: float
    prompt_eval_count: int
    eval_count: int
    prompt_eval_seconds: float
    eval_seconds: float
    total_seconds: float
    response: str

    @property
    def decode_tok_s(self) -> float:
        """Server-side decode throughput reported by Ollama."""
        return self.eval_count / self.eval_seconds if self.eval_seconds > 0 else 0.0


def build_fixed_prompt(context_len: int) -> str:
    """Build a deterministic prompt with approximately ``context_len`` word tokens.

    Ollama tokenizes internally, so this is only an approximate context-length
    control. The response's ``prompt_eval_count`` is the authoritative actual
    value and is printed in the summary.
    """
    if context_len <= 0:
        raise ValueError("--context-len must be greater than 0")
    seed_words = (
        "Explain machine learning clearly with practical examples and precise "
        "technical detail for an engineering audience. "
    ).split()
    return " ".join(seed_words[index % len(seed_words)] for index in range(context_len))


def post_generate(
    base_url: str,
    model: str,
    prompt: str,
    max_new: int,
    timeout: float,
    request_index: int,
    raw: bool,
) -> RequestResult:
    """Send one non-streaming Ollama /api/generate request and collect metrics."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        # Native mode lets Ollama apply the selected model's own template.
        # Raw mode is retained for callers that already rendered a template.
        "raw": raw,
        # Keep sampling deterministic and avoid EOS shortening the measured decode.
        "options": {"num_predict": max_new, "temperature": 0, "seed": 0},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body: dict[str, Any] = json.loads(response.read())
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Ollama request {request_index} failed. Is Ollama reachable at {base_url}?"
        ) from exc
    wall_seconds = time.perf_counter() - started

    # Ollama returns durations in nanoseconds. Missing fields are treated as zero
    # so the script remains compatible with older local Ollama versions.
    return RequestResult(
        request_index=request_index,
        wall_seconds=wall_seconds,
        prompt_eval_count=int(body.get("prompt_eval_count", 0)),
        eval_count=int(body.get("eval_count", 0)),
        prompt_eval_seconds=float(body.get("prompt_eval_duration", 0)) / 1e9,
        eval_seconds=float(body.get("eval_duration", 0)) / 1e9,
        total_seconds=float(body.get("total_duration", 0)) / 1e9,
        response=str(body.get("response", "")),
    )


def run_round(
    *,
    base_url: str,
    model: str,
    prompts: list[str],
    max_new_list: list[int],
    concurrency: int,
    timeout: float,
    raw: bool,
) -> tuple[list[RequestResult], float]:
    """Run a fixed request count with at most ``concurrency`` in flight."""
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                post_generate,
                base_url,
                model,
                prompts[request_index],
                max_new_list[request_index],
                timeout,
                request_index,
                raw,
            )
            for request_index in range(len(prompts))
        ]
        results = [future.result() for future in futures]
    return results, time.perf_counter() - started


def percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile for a non-empty value list."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark aggregate throughput of concurrent local Ollama requests."
    )
    parser.add_argument("--model", default="qwen2.5:0.5b-instruct-fp16", help="Ollama model tag")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434", help="Ollama server URL")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Exact prompt text. When omitted, a synthetic prompt is built from --context-len.",
    )
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help="Render --prompt with the HF Qwen tokenizer template before sending it as raw Ollama input.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Send raw input instead of applying Ollama's native model template.",
    )
    parser.add_argument(
        "--template-model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HF tokenizer used by --chat-template; must match the Ollama model family.",
    )
    parser.add_argument("--context-len", type=int, default=128, help="Approximate synthetic prompt word count")
    parser.add_argument("--max-new", type=int, default=256, help="Requested generated tokens per request")
    parser.add_argument("--concurrency", type=int, default=1, help="Maximum requests in flight")
    parser.add_argument("--requests", type=int, default=8, help="Requests in each measured run")
    parser.add_argument("--runs", type=int, default=3, help="Measured rounds; median aggregate throughput is reported")
    parser.add_argument("--warmup-requests", type=int, default=1, help="Requests sent before measurement (0 disables warmup)")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-request HTTP timeout in seconds")
    parser.add_argument(
        "--heterogeneous-prompts",
        action="store_true",
        help="Vary prompt lengths within the batch (synthetic prompts only).",
    )
    parser.add_argument(
        "--heterogeneous-max-new",
        action="store_true",
        help="Vary max_new_tokens per request to simulate heterogeneous generation.",
    )
    args = parser.parse_args()

    if args.max_new <= 0:
        raise ValueError("--max-new must be greater than 0")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be greater than 0")
    if args.requests <= 0:
        raise ValueError("--requests must be greater than 0")
    if args.runs <= 0:
        raise ValueError("--runs must be greater than 0")
    if args.warmup_requests < 0:
        raise ValueError("--warmup-requests must not be negative")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than 0")

    if args.chat_template and args.prompt is None:
        raise ValueError("--chat-template requires --prompt")
    use_raw = args.raw or args.chat_template
    if args.heterogeneous_prompts and args.prompt is not None:
        raise ValueError("--heterogeneous-prompts requires synthetic prompts (omit --prompt)")
    if args.prompt is not None:
        if not args.prompt.strip():
            raise ValueError("--prompt must not be empty")
        if args.chat_template:
            tokenizer = Qwen2Adapter().load_tokenizer(args.template_model, trust_remote_code=False)
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": args.prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_source = f"qwen_chat_template:{args.template_model}"
        else:
            prompt = args.prompt
            prompt_source = "explicit_raw" if use_raw else "ollama_native_template"
    else:
        prompt = build_fixed_prompt(args.context_len)
        source_mode = "raw" if use_raw else "ollama_native_template"
        prompt_source = f"synthetic_context_len={args.context_len}:{source_mode}"

    # Build per-request prompt and max_new lists for heterogeneous mode.
    def build_request_params(count: int) -> tuple[list[str], list[int]]:
        if args.heterogeneous_prompts:
            prompts = [
                build_fixed_prompt(max(1, args.context_len - (index % min(4, args.context_len))))
                for index in range(count)
            ]
        else:
            prompts = [prompt] * count
        if args.heterogeneous_max_new:
            max_new_list = [
                max(1, args.max_new - (index % min(4, args.max_new)))
                for index in range(count)
            ]
        else:
            max_new_list = [args.max_new] * count
        return prompts, max_new_list

    print(
        f"[ollama] model={args.model} base_url={args.base_url} "
        f"concurrency={args.concurrency} requests/run={args.requests} "
        f"max_new={args.max_new} prompt_source={prompt_source} "
        f"heterogeneous_prompts={args.heterogeneous_prompts} "
        f"heterogeneous_max_new={args.heterogeneous_max_new}"
    )

    if args.warmup_requests:
        print(f"[ollama] warmup requests={args.warmup_requests}...")
        warmup_prompts, warmup_max_new = build_request_params(args.warmup_requests)
        run_round(
            base_url=args.base_url,
            model=args.model,
            prompts=warmup_prompts,
            max_new_list=warmup_max_new,
            concurrency=min(args.concurrency, args.warmup_requests),
            timeout=args.timeout,
            raw=use_raw,
        )

    aggregate_rates: list[float] = []
    request_p50_latencies: list[float] = []
    request_p95_latencies: list[float] = []
    active_decode_rate_p50s: list[float] = []
    prompt_token_counts: list[int] = []
    output_token_counts: list[int] = []
    final_request_outputs: list[str] = []

    for run_index in range(args.runs):
        run_prompts, run_max_new = build_request_params(args.requests)
        results, round_seconds = run_round(
            base_url=args.base_url,
            model=args.model,
            prompts=run_prompts,
            max_new_list=run_max_new,
            concurrency=args.concurrency,
            timeout=args.timeout,
            raw=use_raw,
        )
        total_output_tokens = sum(result.eval_count for result in results)
        aggregate_rate = total_output_tokens / round_seconds if round_seconds > 0 else 0.0
        latencies = [result.wall_seconds for result in results]
        per_request_server_rates = [result.decode_tok_s for result in results if result.eval_seconds > 0]

        aggregate_rates.append(aggregate_rate)
        request_p50_latencies.append(percentile(latencies, 0.50))
        request_p95_latencies.append(percentile(latencies, 0.95))
        active_decode_rate_p50s.append(statistics.median(per_request_server_rates))
        prompt_token_counts.extend(result.prompt_eval_count for result in results)
        output_token_counts.extend(result.eval_count for result in results)
        final_request_outputs.append(max(results, key=lambda result: result.request_index).response)

        print(
            f"run={run_index + 1} wall_s={round_seconds:.3f} output_tokens={total_output_tokens} "
            f"aggregate_tok/s={aggregate_rate:.2f} "
            f"active_decode_tok/s_p50={statistics.median(per_request_server_rates):.2f} "
            f"request_latency_p50_s={percentile(latencies, 0.50):.3f} "
            f"request_latency_p95_s={percentile(latencies, 0.95):.3f}"
        )

    print("\n[ollama concurrent summary]")
    print(
        f"model={args.model} concurrency={args.concurrency} requests/run={args.requests} "
        f"runs={args.runs} stream=false raw={str(use_raw).lower()} "
        f"heterogeneous_prompts={args.heterogeneous_prompts} "
        f"heterogeneous_max_new={args.heterogeneous_max_new}"
    )
    print(
        f"actual_prompt_tokens_median={statistics.median(prompt_token_counts):.0f} "
        f"output_tokens_median={statistics.median(output_token_counts):.0f}"
    )
    print(
        f"aggregate_decode_tok/s_median={statistics.median(aggregate_rates):.2f} "
        f"min={min(aggregate_rates):.2f} max={max(aggregate_rates):.2f}"
    )
    print(
        f"request_latency_p50_s_median={statistics.median(request_p50_latencies):.3f} "
        f"request_latency_p95_s_median={statistics.median(request_p95_latencies):.3f}"
    )
    if active_decode_rate_p50s:
        print(f"active_decode_tok/s_p50_median={statistics.median(active_decode_rate_p50s):.2f}")
    print(f"final_request_output_run_{args.runs}={final_request_outputs[-1]!r}")


if __name__ == "__main__":
    main()
