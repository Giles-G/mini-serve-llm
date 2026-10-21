#!/usr/bin/env python3
"""Real-activation and logits parity for the Gemma4 INT4 legacy/warp kernels.

Stage-C gate:

* captures the actual activation entering every INT4 projection for several
  real prompts (prefill + optional decode steps);
* compares legacy and warp against the chunked rounded-weight reference, and
  for selected labels against both the exact-math and rounded-weight FP32
  contracts (including CPU FP64 where small enough);
* compares full-model logits, top-1/top-2 and margins for prefill plus several
  consecutive decode steps;
* compares KV caches, shared-layer storage, positions, a synthetic sliding
  window, and EOS/stop behaviour.

The reference is chunked along output rows so lm_head never materializes the
whole [262144, 1536] matrix at once.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bench_gemma4_cuda import _check_parity, _tf32_disabled  # noqa: E402
from scripts.bench_gemma4_speed import (  # noqa: E402
    JsonTokenizer,
    _profile_int4_labels,
    load_weights,
    render_chat_prompt,
)
from miniservellm.runtime.gemma4_quant import Int4Weight  # noqa: E402
from miniservellm.runtime.gemma4_runner import Gemma4EagerTextRunner  # noqa: E402

try:
    from mini_llm_kernels.kernels.int4_matmul import (  # noqa: E402
        _HAS_INT4_CUDA,
        _cuda_int4_matmul,
        _cuda_int4_matmul_legacy,
        _cuda_int4_matmul_pair,
        _int4_dequant_matmul_pytorch,
    )
except ImportError as exc:  # pragma: no cover - depends on the build
    _HAS_INT4_CUDA = False
    _cuda_int4_matmul = None
    _cuda_int4_matmul_legacy = None
    _cuda_int4_matmul_pair = None
    _int4_dequant_matmul_pytorch = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

EOS_IDS = (1, 106, 50)
DEFAULT_PROMPTS = [
    "请详细介绍KV Cache的作用和实现原理。",
    "Explain the purpose and implementation of a KV cache in transformer inference.",
    "What is 27 + 45? Answer with the number only.",
]


def _reference_chunked(x, packed, scales, chunk_rows):
    n = packed.shape[0] * 2
    parts = []
    with _tf32_disabled():
        for start in range(0, n, chunk_rows):
            stop = min(start + chunk_rows, n)
            parts.append(
                _int4_dequant_matmul_pytorch(
                    x, packed[start // 2: stop // 2], scales[start:stop]
                )
            )
    return torch.cat(parts, dim=-1)


def _error_stats(reference, actual):
    diff = (actual.float() - reference.float()).abs()
    finite = torch.isfinite(actual).all() and torch.isfinite(reference).all()
    return {
        "finite": bool(finite),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "p99_abs": float(torch.quantile(diff.flatten(), 0.99)) if diff.numel() else 0.0,
    }


def capture_activations(runner, labels, prompt_ids, steps):
    """Record (label, step, weight, x) for the real forward, without re-running."""
    records = []
    state = {"step": 0}
    original = Int4Weight.matmul

    def recording_matmul(self, x):
        if state["step"] < steps:
            records.append(
                (labels.get(id(self), "other"), state["step"], self,
                 x.detach().clone())
            )
        return original(self, x)

    Int4Weight.matmul = recording_matmul
    try:
        with torch.inference_mode():
            logits = runner.prefill(prompt_ids)
            next_id = int(logits.argmax())
            for step in range(1, steps):
                state["step"] = step
                logits = runner.decode_step(next_id)
                next_id = int(logits.argmax())
    finally:
        Int4Weight.matmul = original
    return records


def compare_activations(records, chunk_rows):
    summary = {}
    for label, step, weight, x in records:
        x = x.contiguous()
        reference = _reference_chunked(x, weight.packed, weight.scales, chunk_rows)
        legacy = _cuda_int4_matmul_legacy(
            x, weight.packed, weight.scales, weight.group_size
        )
        warp = _cuda_int4_matmul(
            x, weight.packed, weight.scales, weight.group_size
        )
        pair = _cuda_int4_matmul_pair(
            x, weight.packed, weight.scales, weight.group_size
        )
        legacy_stats = _error_stats(reference, legacy)
        warp_stats = _error_stats(reference, warp)
        pair_error = _error_stats(reference, pair)
        legacy_warp = _error_stats(legacy, warp)
        legacy_pair = _error_stats(legacy, pair)
        warp_pair = _error_stats(warp, pair)
        entry = summary.setdefault(
            label,
            {
                "shape": list(x.shape),
                "steps": [],
                "legacy_max_abs": 0.0,
                "warp_max_abs": 0.0,
                "pair_max_abs": 0.0,
                "legacy_vs_warp_max_abs": 0.0,
                "legacy_vs_pair_max_abs": 0.0,
                "warp_vs_pair_max_abs": 0.0,
                "max_impl_diff_max_abs": 0.0,
            },
        )
        entry["steps"].append(
            {
                "step": step,
                "legacy": legacy_stats,
                "warp": warp_stats,
                "pair": pair_error,
                "legacy_vs_warp": legacy_warp,
                "legacy_vs_pair": legacy_pair,
                "warp_vs_pair": warp_pair,
            }
        )
        entry["legacy_max_abs"] = max(entry["legacy_max_abs"], legacy_stats["max_abs"])
        entry["warp_max_abs"] = max(entry["warp_max_abs"], warp_stats["max_abs"])
        entry["pair_max_abs"] = max(entry["pair_max_abs"], pair_error["max_abs"])
        entry["legacy_vs_warp_max_abs"] = max(
            entry["legacy_vs_warp_max_abs"], legacy_warp["max_abs"]
        )
        entry["legacy_vs_pair_max_abs"] = max(
            entry["legacy_vs_pair_max_abs"], legacy_pair["max_abs"]
        )
        entry["warp_vs_pair_max_abs"] = max(
            entry["warp_vs_pair_max_abs"], warp_pair["max_abs"]
        )
        entry["max_impl_diff_max_abs"] = max(
            entry["legacy_vs_warp_max_abs"],
            entry["legacy_vs_pair_max_abs"],
            entry["warp_vs_pair_max_abs"],
        )
        del reference, legacy, warp, pair
    return summary


def compare_contracts(records, labels, chunk_rows):
    """Check selected labels against exact-math and rounded-weight references."""
    by_label = {}
    for record in records:
        by_label.setdefault(record[0], record)
    results = {}
    for label in labels:
        record = by_label.get(label)
        if record is None:
            results[label] = {"skipped": "label not used by this prompt"}
            continue
        _, _, weight, x = record
        x = x.contiguous()
        reference = _reference_chunked(x, weight.packed, weight.scales, chunk_rows)
        legacy = _cuda_int4_matmul_legacy(
            x, weight.packed, weight.scales, weight.group_size
        )
        warp = _cuda_int4_matmul(
            x, weight.packed, weight.scales, weight.group_size
        )
        pair = _cuda_int4_matmul_pair(
            x, weight.packed, weight.scales, weight.group_size
        )
        legacy_report = _check_parity(
            x, weight.packed, weight.scales, reference, legacy, chunk_rows=chunk_rows
        )
        warp_report = _check_parity(
            x, weight.packed, weight.scales, reference, warp, chunk_rows=chunk_rows
        )
        pair_report = _check_parity(
            x, weight.packed, weight.scales, reference, pair, chunk_rows=chunk_rows
        )
        results[label] = {
            "legacy": legacy_report,
            "warp": warp_report,
            "pair": pair_report,
            "passed": (
                legacy_report["passed"]
                and warp_report["passed"]
                and pair_report["passed"]
            ),
        }
        del reference, legacy, warp, pair
    return results


def run_logits(runner, prompt_ids, steps, impl):
    os.environ["MINI_LLM_INT4_IMPL"] = impl
    runner.reset_cache()
    rows = []
    stopped_at = None
    with torch.inference_mode():
        logits = runner.prefill(prompt_ids)
        for step in range(steps):
            top = torch.topk(logits, 2)
            rows.append(
                {
                    "step": step,
                    "logits": logits.detach().clone(),
                    "top1": int(top.indices[0]),
                    "top2": int(top.indices[1]),
                    "margin": float(top.values[0] - top.values[1]),
                }
            )
            next_id = int(top.indices[0])
            if next_id in EOS_IDS:
                stopped_at = step
                break
            if step + 1 < steps:
                logits = runner.decode_step(next_id)
    cache = {
        layer: (k.detach().clone(), v.detach().clone())
        for layer, (k, v) in runner._cache.items()
    }
    return {
        "rows": rows,
        "cache": cache,
        "seq_len": runner._seq_len,
        "stopped_at": stopped_at,
    }


def compare_logits(legacy, warp):
    n = min(len(legacy["rows"]), len(warp["rows"]))
    steps = []
    flips = 0
    for i in range(n):
        a = legacy["rows"][i]
        b = warp["rows"][i]
        diff = (a["logits"] - b["logits"]).abs()
        row = {
            "step": i,
            "legacy_top1": a["top1"],
            "warp_top1": b["top1"],
            "top1_same": a["top1"] == b["top1"],
            "top2_same": a["top2"] == b["top2"],
            "legacy_margin": a["margin"],
            "warp_margin": b["margin"],
            "max_abs_logit_diff": float(diff.max()),
            "mean_abs_logit_diff": float(diff.mean()),
        }
        if not row["top1_same"]:
            flips += 1
        steps.append(row)
    return {
        "steps": steps,
        "compared_steps": n,
        "top1_flips": flips,
        "top1_sequence_match": flips == 0,
        "max_abs_logit_diff": max((s["max_abs_logit_diff"] for s in steps), default=0.0),
        "legacy_stopped_at": legacy["stopped_at"],
        "warp_stopped_at": warp["stopped_at"],
        "eos_stop_match": legacy["stopped_at"] == warp["stopped_at"],
    }


def compare_cache(legacy, warp):
    a = legacy["cache"]
    b = warp["cache"]
    layers = sorted(set(a) | set(b))
    rows = []
    for layer in layers:
        if layer not in a or layer not in b:
            rows.append({"layer": layer, "missing": True})
            continue
        ka, va = a[layer]
        kb, vb = b[layer]
        rows.append(
            {
                "layer": layer,
                "k_shape": list(ka.shape),
                "v_shape": list(va.shape),
                "k_max_abs": float((ka.float() - kb.float()).abs().max()),
                "v_max_abs": float((va.float() - vb.float()).abs().max()),
            }
        )
    return {
        "layers": rows,
        "seq_len_legacy": legacy["seq_len"],
        "seq_len_warp": warp["seq_len"],
        "seq_len_match": legacy["seq_len"] == warp["seq_len"],
        "max_k_diff": max((r.get("k_max_abs", 0.0) for r in rows), default=0.0),
        "max_v_diff": max((r.get("v_max_abs", 0.0) for r in rows), default=0.0),
    }


def check_sliding(runner, prompt_ids, steps):
    original = runner.window
    synthetic = min(original or 8, 8)
    runner.window = synthetic
    try:
        legacy = run_logits(runner, prompt_ids, steps, "legacy")
        warp = run_logits(runner, prompt_ids, steps, "warp")
        pair = run_logits(runner, prompt_ids, steps, "pair")
    finally:
        runner.window = original
    result = compare_logits(legacy, warp)
    result["pair_vs_legacy"] = compare_logits(legacy, pair)
    result["synthetic_window"] = synthetic
    return result


def check_eos(runner, raw_ids, steps):
    tensor = torch.tensor(
        raw_ids, device=runner.w.embed_tokens.device, dtype=torch.long
    )
    legacy = run_logits(runner, tensor, steps, "legacy")
    warp = run_logits(runner, tensor, steps, "warp")
    pair = run_logits(runner, tensor, steps, "pair")
    result = compare_logits(legacy, warp)
    result["pair_vs_legacy"] = compare_logits(legacy, pair)
    result["prompt_tokens"] = len(raw_ids)
    result["triggered"] = legacy["stopped_at"] is not None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--prompts", nargs="+", default=None)
    parser.add_argument("--capture-steps", type=int, default=2,
                        help="prefill + N-1 decode steps of real activations")
    parser.add_argument("--logits-steps", type=int, default=8,
                        help="prefill + N-1 decode logits positions")
    parser.add_argument("--reference-chunk-rows", type=int, default=4096)
    parser.add_argument("--contract-chunk-rows", type=int, default=4096)
    parser.add_argument(
        "--contract-labels",
        default="lm_head,layer0.q_proj,layer0.down_proj,per_layer_model_projection",
        help="Comma-separated labels checked against exact and rounded FP32",
    )
    parser.add_argument("--max-logit-diff", type=float, default=0.25)
    parser.add_argument("--max-activation-diff", type=float, default=0.1)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--skip-sliding", action="store_true")
    parser.add_argument(
        "--official-sliding",
        action="store_true",
        help="also run a >model-window (512) prompt to exercise the real window",
    )
    parser.add_argument("--official-sliding-tokens", type=int, default=520)
    parser.add_argument(
        "--official-sliding-steps", type=int, default=32,
        help="logits positions for the official-window generation check",
    )
    parser.add_argument("--skip-eos", action="store_true")
    parser.add_argument("--eos-prompt", default="Hello",
                        help="raw (un-templated) prompt used for the EOS/stop check")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if args.capture_steps < 1 or args.logits_steps < 1:
        parser.error("capture-steps and logits-steps must be >= 1")
    if args.official_sliding_tokens < 1:
        parser.error("official-sliding-tokens must be >= 1")
    if args.official_sliding_steps < 2:
        parser.error("official-sliding-steps must be >= 2")
    for value in (args.reference_chunk_rows, args.contract_chunk_rows):
        if value < 2 or value % 2:
            parser.error("chunk rows must be even and >= 2")
    if not torch.cuda.is_available() or args.device != "cuda":
        raise RuntimeError("this parity check requires CUDA")
    if _IMPORT_ERROR is not None or not _HAS_INT4_CUDA:
        raise RuntimeError("mini_llm_kernels INT4 CUDA extension is unavailable")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    model_path = Path(args.model).expanduser()
    weights, config, load_s = load_weights(
        model_path, args.device, dtype, int4=True, backend="auto"
    )
    labels = _profile_int4_labels(weights)[1]
    tokenizer = JsonTokenizer(model_path)
    runner = Gemma4EagerTextRunner(config, weights)
    contract_labels = [item.strip() for item in args.contract_labels.split(",") if item.strip()]
    prompts = args.prompts or DEFAULT_PROMPTS

    prompt_results = []
    all_passed = True
    for prompt in prompts:
        chat = None if args.no_chat_template else render_chat_prompt(model_path, prompt)
        prompt_ids = tokenizer.encode(
            chat if chat is not None else prompt, add_special_tokens=False
        )
        tensor = torch.tensor(prompt_ids, device=weights.embed_tokens.device,
                              dtype=torch.long)
        records = capture_activations(runner, labels, tensor, args.capture_steps)
        activation = compare_activations(records, args.reference_chunk_rows)
        contracts = compare_contracts(records, contract_labels, args.contract_chunk_rows)
        legacy = run_logits(runner, tensor, args.logits_steps, "legacy")
        warp = run_logits(runner, tensor, args.logits_steps, "warp")
        pair = run_logits(runner, tensor, args.logits_steps, "pair")
        logits = compare_logits(legacy, warp)
        logits_pair = compare_logits(legacy, pair)
        cache = compare_cache(legacy, warp)
        cache_pair = compare_cache(legacy, pair)
        cache_layers = {row["layer"] for row in cache["layers"] if not row.get("missing")}
        shared_layers = set(runner.kv_source)
        storing_layers = set(runner.storing_layers)
        shared_stored = sorted(shared_layers & cache_layers)
        shared_ok = not shared_stored and storing_layers <= cache_layers

        worst_label, worst = max(
            activation.items(),
            key=lambda item: item[1]["max_impl_diff_max_abs"],
        )
        prompt_pass = (
            logits["top1_flips"] == 0
            and logits_pair["top1_flips"] == 0
            and logits["max_abs_logit_diff"] <= args.max_logit_diff
            and logits_pair["max_abs_logit_diff"] <= args.max_logit_diff
            and worst["max_impl_diff_max_abs"] <= args.max_activation_diff
            and all(
                row.get("passed", True) for row in contracts.values()
            )
            and shared_ok
        )
        all_passed = all_passed and prompt_pass
        result = {
            "prompt": prompt,
            "prompt_tokens": len(prompt_ids),
            "chat_template": chat is not None,
            "activation_labels": len(activation),
            "activation_records": len(records),
            "worst_activation_label": worst_label,
            "worst_impl_diff": worst["max_impl_diff_max_abs"],
            "activations": activation,
            "logits": logits,
            "logits_pair_vs_legacy": logits_pair,
            "contracts": contracts,
            "cache": cache,
            "cache_pair_vs_legacy": cache_pair,
            "kv_sharing": {
                "kv_source": {str(k): v for k, v in sorted(runner.kv_source.items())},
                "storing_layers": sorted(storing_layers),
                "stored_layers": sorted(cache_layers),
                "shared_layers_stored": shared_stored,
                "shared_layers_not_stored": shared_ok,
            },
            "passed": prompt_pass,
        }
        prompt_results.append(result)
        print(
            f"prompt_tokens={len(prompt_ids)} labels={len(activation)} "
            f"warp_flips={logits['top1_flips']} "
            f"pair_flips={logits_pair['top1_flips']} "
            f"warp_maxdiff={logits['max_abs_logit_diff']:.6f} "
            f"pair_maxdiff={logits_pair['max_abs_logit_diff']:.6f} "
            f"worst_act={worst_label}:{worst['max_impl_diff_max_abs']:.6f} "
            f"passed={prompt_pass}"
        )

    sliding = None
    if not args.skip_sliding:
        sample = torch.tensor(
            tokenizer.encode(
                prompts[0] if args.no_chat_template
                else (render_chat_prompt(model_path, prompts[0]) or prompts[0]),
                add_special_tokens=False,
            ),
            device=weights.embed_tokens.device, dtype=torch.long,
        )
        sliding = check_sliding(runner, sample, args.logits_steps)
        sliding_pass = (
            sliding["top1_flips"] == 0
            and sliding["pair_vs_legacy"]["top1_flips"] == 0
            and sliding["max_abs_logit_diff"] <= args.max_logit_diff
            and sliding["pair_vs_legacy"]["max_abs_logit_diff"] <= args.max_logit_diff
        )
        all_passed = all_passed and sliding_pass
        print(
            f"sliding window synthetic={sliding['synthetic_window']} "
            f"warp_flips={sliding['top1_flips']} "
            f"pair_flips={sliding['pair_vs_legacy']['top1_flips']} "
            f"warp_maxdiff={sliding['max_abs_logit_diff']:.6f} "
            f"pair_maxdiff={sliding['pair_vs_legacy']['max_abs_logit_diff']:.6f} "
            f"passed={sliding_pass}"
        )

    official_sliding = None
    if args.official_sliding:
        base = prompts[0]
        base_ids = tokenizer.encode(base, add_special_tokens=False)
        repeats = args.official_sliding_tokens // max(1, len(base_ids)) + 2
        long_text = " ".join([base] * repeats)
        long_ids = tokenizer.encode(long_text, add_special_tokens=False)
        if len(long_ids) < args.official_sliding_tokens:
            raise RuntimeError("failed to build a long enough sliding prompt")
        long_ids = long_ids[:args.official_sliding_tokens]
        tensor = torch.tensor(long_ids, device=weights.embed_tokens.device,
                              dtype=torch.long)
        legacy = run_logits(runner, tensor, args.official_sliding_steps, "legacy")
        warp = run_logits(runner, tensor, args.official_sliding_steps, "warp")
        pair = run_logits(runner, tensor, args.official_sliding_steps, "pair")
        official_sliding = compare_logits(legacy, warp)
        official_sliding["pair_vs_legacy"] = compare_logits(legacy, pair)
        official_sliding["cache"] = compare_cache(legacy, warp)
        official_sliding["cache_pair_vs_legacy"] = compare_cache(legacy, pair)
        official_sliding["prompt_tokens"] = len(long_ids)
        official_sliding["window"] = runner.window
        official_sliding["steps"] = args.official_sliding_steps
        official_pass = (
            official_sliding["top1_flips"] == 0
            and official_sliding["pair_vs_legacy"]["top1_flips"] == 0
            and official_sliding["max_abs_logit_diff"] <= args.max_logit_diff
            and official_sliding["pair_vs_legacy"]["max_abs_logit_diff"]
            <= args.max_logit_diff
        )
        all_passed = all_passed and official_pass
        print(
            f"official sliding window={runner.window} "
            f"prompt_tokens={len(long_ids)} steps={args.official_sliding_steps} "
            f"warp_flips={official_sliding['top1_flips']} "
            f"pair_flips={official_sliding['pair_vs_legacy']['top1_flips']} "
            f"warp_maxdiff={official_sliding['max_abs_logit_diff']:.6f} "
            f"pair_maxdiff={official_sliding['pair_vs_legacy']['max_abs_logit_diff']:.6f} "
            f"cache_k={official_sliding['cache']['max_k_diff']:.6f} "
            f"passed={official_pass}"
        )

    eos = None
    if not args.skip_eos:
        raw_ids = tokenizer.encode(args.eos_prompt, add_special_tokens=False)
        eos = check_eos(runner, raw_ids, 4)
        eos_pass = (
            eos["eos_stop_match"]
            and eos["top1_flips"] == 0
            and eos["pair_vs_legacy"]["top1_flips"] == 0
        )
        all_passed = all_passed and eos_pass
        print(
            f"eos raw_prompt={args.eos_prompt!r} triggered={eos['triggered']} "
            f"legacy_stop={eos['legacy_stopped_at']} "
            f"warp_stop={eos['warp_stopped_at']} "
            f"warp_flips={eos['top1_flips']} "
            f"pair_flips={eos['pair_vs_legacy']['top1_flips']} passed={eos_pass}"
        )

    payload = {
        "meta": {
            "load_s": round(load_s, 2),
            "dtype": str(dtype),
            "capture_steps": args.capture_steps,
            "logits_steps": args.logits_steps,
            "max_logit_diff": args.max_logit_diff,
            "max_activation_diff": args.max_activation_diff,
            "prompts": len(prompts),
            "impls": ["legacy", "warp", "pair"],
            "official_sliding_steps": args.official_sliding_steps,
        },
        "prompts": prompt_results,
        "sliding": sliding,
        "official_sliding": official_sliding,
        "eos": eos,
        "passed": all_passed,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"saved {args.output_json}")
    print("C PARITY PASS" if all_passed else "C PARITY FAIL")
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
