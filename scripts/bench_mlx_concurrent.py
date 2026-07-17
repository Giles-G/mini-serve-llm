#!/usr/bin/env python3
"""Benchmark the MLX length-bucketed serving engine.

Requests in the same prompt-length bucket are admitted into one fixed-shape
batch. Each row retains an independent KV offset during decode, so unequal
prompt lengths and unequal generation lengths are safe. Completed rows are
refilled from compatible waiting requests without draining the full batch.

Example:
    python scripts/bench_mlx_concurrent.py \
        --batch-size 4 --requests 16 --context-len 128 --max-new 256 \
        --quant-bits 4 --runs 5
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from miniservellm.config import MODEL_NAME
from miniservellm.mlx.engine import MLXInferenceEngine
from miniservellm.mlx.runner import MLXQwen2Runner
from miniservellm.model_adapter.adapters.qwen2_adapter import Qwen2Adapter
from miniservellm.scheduler.request import SamplingParams


def build_fixed_context(tokenizer, context_len: int) -> list[int]:
    """Build one deterministic prompt whose token length is exactly context_len."""
    if context_len <= 0:
        raise ValueError("--context-len must be greater than 0")
    seed = tokenizer.encode("请用至少1000字来介绍机器学习", add_special_tokens=False)
    if not seed:
        raise RuntimeError("Tokenizer returned an empty sequence")
    return (seed * ((context_len + len(seed) - 1) // len(seed)))[:context_len]


def percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sample."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MLX length-bucketed serving batches")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument(
        "--prompt",
        default=None,
        help="Exact prompt text. When omitted, a synthetic prompt is built from --context-len.",
    )
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help="Render --prompt as a Qwen user turn with the tokenizer chat template.",
    )
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4, help="Maximum requests decoded together")
    parser.add_argument(
        "--batch-mode",
        choices=["static", "continuous", "auto"],
        default="auto",
        help="static uses exact homogeneous admission; continuous enables slot refill; auto uses static when possible.",
    )
    parser.add_argument("--prompt-bucket-multiple", type=int, default=64, help="Prompt length bucket width in tokens")
    parser.add_argument("--heterogeneous-prompts", action="store_true", help="Vary prompt lengths within the first bucket")
    parser.add_argument("--heterogeneous-max-new", action="store_true", help="Vary max_new_tokens per request")
    parser.add_argument("--requests", type=int, default=16, help="Total requests per measured run")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--quant-bits", type=int, default=0, choices=[0, 4, 8])
    parser.add_argument("--quant-group-size", type=int, default=64)
    parser.add_argument(
        "--compiled-decode",
        action="store_true",
        help="Use the cached fixed-shape mx.compile decode graph.",
    )
    args = parser.parse_args()

    if args.max_new <= 0 or args.batch_size <= 0 or args.requests <= 0 or args.runs <= 0:
        raise ValueError("--max-new, --batch-size, --requests, and --runs must be greater than 0")
    if args.prompt_bucket_multiple <= 0:
        raise ValueError("--prompt-bucket-multiple must be greater than 0")
    if args.warmup_requests < 0:
        raise ValueError("--warmup-requests must not be negative")
    if args.batch_mode == "static" and (
        args.heterogeneous_prompts or args.heterogeneous_max_new
    ):
        raise ValueError("--batch-mode static does not accept heterogeneous request options")

    adapter = Qwen2Adapter()
    tokenizer = adapter.load_tokenizer(args.model, trust_remote_code=False)
    hf_config = adapter.load_hf_config(args.model, trust_remote_code=False)
    model_config = adapter.convert_hf_config(hf_config)
    model_path = Path(args.model)
    if not model_path.exists():
        from huggingface_hub import snapshot_download
        model_path = Path(snapshot_download(args.model, local_files_only=True))
    safetensors_path = model_path / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"Missing model.safetensors: {safetensors_path}")

    runner = MLXQwen2Runner(model_config, safetensors_path)
    if args.quant_bits:
        print(f"[mlx] quantizing weights: bits={args.quant_bits} group_size={args.quant_group_size}")
        runner.model.quantize_weights(args.quant_bits, args.quant_group_size)
        mx.eval(runner.model.weights.layers[0].qkv_proj)

    if args.chat_template and args.prompt is None:
        raise ValueError("--chat-template requires --prompt")
    if args.prompt is not None:
        if not args.prompt.strip():
            raise ValueError("--prompt must not be empty")
        if args.chat_template:
            rendered_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": args.prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            explicit_prompt = tokenizer.encode(rendered_prompt, add_special_tokens=False)
            prompt_source = "qwen_chat_template"
        else:
            explicit_prompt = tokenizer.encode(args.prompt, add_special_tokens=False)
            prompt_source = "explicit_raw"
        if not explicit_prompt:
            raise ValueError("--prompt tokenized to an empty sequence")
    else:
        explicit_prompt = None
        prompt_source = f"synthetic_context_len={args.context_len}"

    benchmark_prompt_tokens = len(explicit_prompt) if explicit_prompt is not None else args.context_len
    params = SamplingParams(temperature=0.0, top_k=0, top_p=1.0)
    print(
        f"[mlx] bucketed_batch batch_size={args.batch_size} requests/run={args.requests} "
        f"prompt_tokens={benchmark_prompt_tokens} max_new={args.max_new} "
        f"bucket_multiple={args.prompt_bucket_multiple} "
        f"heterogeneous_prompts={args.heterogeneous_prompts} "
        f"heterogeneous_max_new={args.heterogeneous_max_new} "
        f"batch_mode={args.batch_mode} prompt_source={prompt_source} "
        f"quant_bits={args.quant_bits}"
    )

    def run_once(
        request_count: int,
    ) -> tuple[int, float, int, int, int, int, list[float], list[float], list[float], list[int]]:
        engine = MLXInferenceEngine(
            runner=runner,
            tokenizer=tokenizer,
            model_config=model_config,
            eos_token_id=None,
            max_batch_size=args.batch_size,
            prompt_bucket_multiple=args.prompt_bucket_multiple,
            compiled_decode=args.compiled_decode,
            batch_mode=args.batch_mode,
        )
        request_ids: list[str] = []
        for index in range(request_count):
            if explicit_prompt is not None:
                prompt_ids = explicit_prompt
            else:
                prompt_len = args.context_len
                if args.heterogeneous_prompts:
                    # Keep all variants in the same lower bucket when possible.
                    prompt_len = max(1, args.context_len - (index % min(4, args.context_len)))
                prompt_ids = build_fixed_context(tokenizer, prompt_len)
            max_new = args.max_new
            if args.heterogeneous_max_new:
                max_new = max(1, args.max_new - (index % min(4, args.max_new)))
            request_ids.append(engine.add_request(
                prompt_token_ids=prompt_ids,
                sampling_params=params,
                max_new_tokens=max_new,
            ))
        started = time.perf_counter()
        ticks = 0
        finished_at: dict[str, float] = {}
        decode_started_at: dict[str, float] = {}
        slot_refills = 0
        while engine.has_pending_work():
            events = engine.step()
            now = time.perf_counter()
            for event in events:
                if event.kind == "prefill_sampled_first_token":
                    # This follows prefill and starts the active decode window,
                    # matching Ollama's eval_duration scope.
                    decode_started_at[event.request_id] = now - started
                elif event.kind == "request_finished":
                    finished_at[event.request_id] = now - started
                elif event.kind == "slot_refilled":
                    slot_refills += 1
            ticks += 1
        elapsed = time.perf_counter() - started
        generated = sum(req.total_generated_tokens() for req in engine.requests_by_id.values())
        request_latencies = [finished_at[request_id] for request_id in request_ids]
        request_rates = [
            engine.get_request(request_id).total_generated_tokens() / finished_at[request_id]
            for request_id in request_ids
        ]
        active_decode_rates = [
            engine.get_request(request_id).total_generated_tokens()
            / max(finished_at[request_id] - decode_started_at[request_id], 1e-9)
            for request_id in request_ids
        ]
        final_request_tokens = engine.get_request(request_ids[-1]).generated_token_ids
        static_fast_path_ticks, dynamic_fallback_ticks = engine.decode_path_stats
        return (
            generated,
            elapsed,
            ticks,
            slot_refills,
            static_fast_path_ticks,
            dynamic_fallback_ticks,
            request_latencies,
            request_rates,
            active_decode_rates,
            final_request_tokens,
        )

    if args.warmup_requests:
        print(f"[mlx] warmup requests={args.warmup_requests}...")
        run_once(args.warmup_requests)

    aggregate_rates: list[float] = []
    normalized_request_rates: list[float] = []
    request_rate_medians: list[float] = []
    active_decode_rate_medians: list[float] = []
    request_latency_p50s: list[float] = []
    request_latency_p95s: list[float] = []
    average_output_tokens: list[float] = []
    slot_refill_counts: list[int] = []
    static_fast_path_counts: list[int] = []
    dynamic_fallback_counts: list[int] = []
    final_request_outputs: list[str] = []
    for index in range(args.runs):
        (
            generated,
            elapsed,
            ticks,
            slot_refills,
            static_fast_path_ticks,
            dynamic_fallback_ticks,
            request_latencies,
            request_rates,
            active_decode_rates,
            final_request_tokens,
        ) = run_once(args.requests)
        aggregate_rate = generated / elapsed if elapsed else 0.0
        normalized_request_rate = aggregate_rate / args.requests
        aggregate_rates.append(aggregate_rate)
        normalized_request_rates.append(normalized_request_rate)
        request_rate_medians.append(statistics.median(request_rates))
        active_decode_rate_medians.append(statistics.median(active_decode_rates))
        request_latency_p50s.append(percentile(request_latencies, 0.50))
        request_latency_p95s.append(percentile(request_latencies, 0.95))
        average_output_tokens.append(generated / args.requests)
        slot_refill_counts.append(slot_refills)
        static_fast_path_counts.append(static_fast_path_ticks)
        dynamic_fallback_counts.append(dynamic_fallback_ticks)
        final_request_outputs.append(tokenizer.decode(final_request_tokens, skip_special_tokens=False))
        run_summary = " ".join([
            f"run={index + 1}",
            f"wall_s={elapsed:.3f}",
            f"output_tokens={generated}",
            f"avg_output_tokens/request={generated / args.requests:.2f}",
            f"ticks={ticks}",
            f"slot_refills={slot_refills}",
            f"static_fast_path_ticks={static_fast_path_ticks}",
            f"dynamic_fallback_ticks={dynamic_fallback_ticks}",
            f"aggregate_tok/s={aggregate_rate:.2f}",
            f"normalized_tok/s/request={normalized_request_rate:.2f}",
            f"active_decode_tok/s_p50={statistics.median(active_decode_rates):.2f}",
            f"request_latency_p50_s={percentile(request_latencies, 0.50):.3f}",
            f"request_latency_p95_s={percentile(request_latencies, 0.95):.3f}",
        ])
        print(run_summary)

    batches_per_run = (args.requests + args.batch_size - 1) // args.batch_size
    print("\n[mlx bucketed-batch summary]")
    print(
        f"backend=mlx dtype={'int' + str(args.quant_bits) if args.quant_bits else 'fp16'} "
        f"max_batch_size={args.batch_size} requests/run={args.requests} batches/run<={batches_per_run}"
    )
    print(
        f"mode={args.batch_mode} per_slot_offsets=true "
        f"slot_refill={args.batch_mode != 'static'} compiled_decode={args.compiled_decode}"
    )
    print(
        f"aggregate_decode_tok/s_median={statistics.median(aggregate_rates):.2f} "
        f"min={min(aggregate_rates):.2f} max={max(aggregate_rates):.2f}"
    )
    print(
        f"normalized_tok/s/request_median={statistics.median(normalized_request_rates):.2f} "
        f"average_output_tokens/request={statistics.median(average_output_tokens):.2f}"
    )
    print(
        f"active_decode_tok/s_p50_median={statistics.median(active_decode_rate_medians):.2f} "
        f"request_latency_p50_s_median={statistics.median(request_latency_p50s):.3f} "
        f"request_latency_p95_s_median={statistics.median(request_latency_p95s):.3f} "
        f"slot_refills_median={statistics.median(slot_refill_counts):.0f} "
        f"static_fast_path_ticks_median={statistics.median(static_fast_path_counts):.0f} "
        f"dynamic_fallback_ticks_median={statistics.median(dynamic_fallback_counts):.0f}"
    )
    static_hits, static_misses = runner.model.compiled_static_decode_cache_stats
    dynamic_hits, dynamic_misses = runner.model.compiled_dynamic_decode_cache_stats
    print(
        f"compiled_static_graph_cache_hits={static_hits} "
        f"compiled_static_graph_cache_misses={static_misses} "
        f"compiled_dynamic_graph_cache_hits={dynamic_hits} "
        f"compiled_dynamic_graph_cache_misses={dynamic_misses}"
    )
    print(f"final_request_output_run_{args.runs}={final_request_outputs[-1]!r}")


if __name__ == "__main__":
    main()
