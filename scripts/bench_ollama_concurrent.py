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
) -> RequestResult:
    """Send one non-streaming Ollama /api/generate request and collect metrics."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
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
    )


def run_round(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_new: int,
    requests: int,
    concurrency: int,
    timeout: float,
) -> tuple[list[RequestResult], float]:
    """Run a fixed request count with at most ``concurrency`` in flight."""
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                post_generate,
                base_url,
                model,
                prompt,
                max_new,
                timeout,
                request_index,
            )
            for request_index in range(requests)
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
    parser.add_argument("--context-len", type=int, default=128, help="Approximate synthetic prompt word count")
    parser.add_argument("--max-new", type=int, default=256, help="Requested generated tokens per request")
    parser.add_argument("--concurrency", type=int, default=1, help="Maximum requests in flight")
    parser.add_argument("--requests", type=int, default=8, help="Requests in each measured run")
    parser.add_argument("--runs", type=int, default=3, help="Measured rounds; median aggregate throughput is reported")
    parser.add_argument("--warmup-requests", type=int, default=1, help="Requests sent before measurement (0 disables warmup)")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-request HTTP timeout in seconds")
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

    if args.prompt is not None:
        if not args.prompt.strip():
            raise ValueError("--prompt must not be empty")
        prompt = args.prompt
        prompt_source = "explicit"
    else:
        prompt = build_fixed_prompt(args.context_len)
        prompt_source = f"synthetic_context_len={args.context_len}"

    print(
        f"[ollama] model={args.model} base_url={args.base_url} "
        f"concurrency={args.concurrency} requests/run={args.requests} "
        f"max_new={args.max_new} prompt_source={prompt_source}"
    )

    if args.warmup_requests:
        print(f"[ollama] warmup requests={args.warmup_requests}...")
        run_round(
            base_url=args.base_url,
            model=args.model,
            prompt=prompt,
            max_new=args.max_new,
            requests=args.warmup_requests,
            concurrency=min(args.concurrency, args.warmup_requests),
            timeout=args.timeout,
        )

    aggregate_rates: list[float] = []
    request_p50_latencies: list[float] = []
    request_p95_latencies: list[float] = []
    server_decode_rates: list[float] = []
    prompt_token_counts: list[int] = []
    output_token_counts: list[int] = []

    for run_index in range(args.runs):
        results, round_seconds = run_round(
            base_url=args.base_url,
            model=args.model,
            prompt=prompt,
            max_new=args.max_new,
            requests=args.requests,
            concurrency=args.concurrency,
            timeout=args.timeout,
        )
        total_output_tokens = sum(result.eval_count for result in results)
        aggregate_rate = total_output_tokens / round_seconds if round_seconds > 0 else 0.0
        latencies = [result.wall_seconds for result in results]
        per_request_server_rates = [result.decode_tok_s for result in results if result.eval_seconds > 0]

        aggregate_rates.append(aggregate_rate)
        request_p50_latencies.append(percentile(latencies, 0.50))
        request_p95_latencies.append(percentile(latencies, 0.95))
        server_decode_rates.extend(per_request_server_rates)
        prompt_token_counts.extend(result.prompt_eval_count for result in results)
        output_token_counts.extend(result.eval_count for result in results)

        print(
            f"run={run_index + 1} wall_s={round_seconds:.3f} output_tokens={total_output_tokens} "
            f"aggregate_tok/s={aggregate_rate:.2f} "
            f"request_latency_p50_s={percentile(latencies, 0.50):.3f} "
            f"request_latency_p95_s={percentile(latencies, 0.95):.3f}"
        )

    print("\n[ollama concurrent summary]")
    print(
        f"model={args.model} concurrency={args.concurrency} requests/run={args.requests} "
        f"runs={args.runs} stream=false"
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
    if server_decode_rates:
        print(f"server_decode_tok/s_per_request_median={statistics.median(server_decode_rates):.2f}")


if __name__ == "__main__":
    main()
