"""Gemma4 CUDA INT4 kernel parity and microbenchmark.

This script deliberately tests the lowest-level CUDA path before wiring a
complete Gemma4 CUDA runner:

* verifies that ``mini_llm_kernels`` loaded its compiled CUDA extension;
* compares the fused INT4 matmul with the reference unpack/dequant path;
* measures both paths for projection shapes used by Gemma4;
* optionally loads the real Gemma4 INT4 weights and benchmarks selected
  projections from the checkpoint.

The script is safe to run on the local M4: it reports that CUDA is missing
and exits successfully. The intended full run is on the RTX 3060 Remote-SSH
workspace after installing ``mini-llm-kernels`` there.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
from itertools import chain
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from mini_llm_kernels.kernels.int4_matmul import (
        _HAS_INT4_CUDA,
        _int4_dequant_matmul_pytorch,
        int4_dequant_matmul,
    )
except ImportError as exc:
    _HAS_INT4_CUDA = False
    _int4_dequant_matmul_pytorch = None
    int4_dequant_matmul = None
    _KERNEL_IMPORT_ERROR = exc
else:
    _KERNEL_IMPORT_ERROR = None


def _pack_int4(w_q: torch.Tensor) -> torch.Tensor:
    """Pack [N, K] values in [-8, 7] as [N//2, K] uint8."""
    if w_q.ndim != 2 or w_q.shape[0] % 2:
        raise ValueError("INT4 packing requires a 2D tensor with even N")
    if w_q.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("INT4 packing requires an integer tensor")
    if not bool(((w_q >= -8) & (w_q <= 7)).all()):
        raise ValueError("INT4 values must be in [-8, 7]")
    unsigned = (w_q.to(torch.int16) + 8).to(torch.uint8)
    return unsigned[0::2] | (unsigned[1::2] << 4)


def _make_case(
    name: str,
    n: int,
    k: int,
    tokens: int,
    group_size: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 20261001,
) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
    if n % 2 or k % group_size:
        raise ValueError(f"{name}: requires even N and K divisible by group size")
    generator = torch.Generator(device="cpu").manual_seed(seed + n + k)
    # Generate a deterministic quantized weight and scales. This mirrors the
    # checkpoint layout without allocating a dense bf16 weight.
    w_q = torch.randint(-8, 8, (n, k), generator=generator, dtype=torch.int8)
    packed = _pack_int4(w_q).to(device)
    scales = (0.002 + 0.03 * torch.rand(
        (n, k // group_size), generator=generator, dtype=torch.float32
    )).to(device=device, dtype=torch.float16)
    # The generator is CPU-seeded for reproducibility, so sample on CPU and
    # move afterwards; torch.randn rejects a CPU generator on a CUDA device.
    x = torch.randn(
        (tokens, k), generator=generator, dtype=dtype
    ).to(device)
    return name, x, packed, scales


def _cuda_time(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    started = torch.cuda.Event(enable_timing=True)
    ended = torch.cuda.Event(enable_timing=True)
    started.record()
    for _ in range(iters):
        fn()
    ended.record()
    ended.synchronize()
    return started.elapsed_time(ended) / iters


def _median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _logical_weight_bytes(n: int, k: int) -> int:
    """Packed INT4 weight bytes + FP16 group scales (gs=64)."""
    return n * k // 2 + 2 * n * k // 64


def _gpu_state() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,driver_version,clocks.sm,temperature.gpu,power.draw,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc}"


def _format_error(reference: torch.Tensor, actual: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - reference.float()).abs()
    return float(diff.max().item()), float(diff.mean().item())


def _validate_inputs(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor) -> int:
    if x.ndim < 2 or packed.ndim != 2 or scales.ndim != 2:
        raise ValueError("x must be [..., K], packed/scales must be 2D")
    if packed.shape[0] * 2 != scales.shape[0] or packed.shape[1] != x.shape[-1]:
        raise ValueError("packed, scales, and x have incompatible shapes")
    if scales.shape[1] < 1 or x.shape[-1] % scales.shape[1]:
        raise ValueError("K must be divisible by the number of scale groups")
    if packed.dtype != torch.uint8:
        raise TypeError("packed weights must have dtype torch.uint8")
    if not torch.isfinite(x).all() or not torch.isfinite(scales).all():
        raise ValueError("x and scales must contain only finite values")
    return x.shape[-1] // scales.shape[1]


@contextmanager
def _tf32_disabled():
    """Temporarily make FP32 references independent of TF32 settings."""
    cuda_matmul = torch.backends.cuda.matmul
    cudnn = torch.backends.cudnn
    old_matmul = cuda_matmul.allow_tf32
    old_cudnn = cudnn.allow_tf32
    cuda_matmul.allow_tf32 = False
    cudnn.allow_tf32 = False
    try:
        yield
    finally:
        cuda_matmul.allow_tf32 = old_matmul
        cudnn.allow_tf32 = old_cudnn


def _cpu_fp64_oracle(x, packed, scales):
    """Compute the dequantized matmul in CPU FP64 for small-matrix checking."""
    group_size = _validate_inputs(x, packed, scales)
    x64 = x.detach().cpu().reshape(-1, x.shape[-1]).double()
    packed_cpu = packed.detach().cpu()
    scales64 = scales.detach().cpu().double()
    pair = packed_cpu
    q = torch.stack((pair & 15, pair >> 4), dim=1).reshape(-1, pair.shape[1])
    w = (q.double() - 8.0) * scales64.repeat_interleave(group_size, dim=1)
    return x64 @ w.T


def _validate_fp32_reference_with_cpu64(x, packed, scales, fp32_reference):
    """Check a small FP32 reference against an independent CPU FP64 result."""
    cpu64 = _cpu_fp64_oracle(x, packed, scales)
    pair = packed.detach().cpu()
    q = torch.stack((pair & 15, pair >> 4), dim=1).reshape(-1, pair.shape[1]).double() - 8
    group_size = x.shape[-1] // scales.shape[1]
    w = q * scales.detach().cpu().double().repeat_interleave(group_size, dim=1)
    magnitudes = x.detach().cpu().reshape(-1, x.shape[-1]).double().abs() @ w.abs().T
    # The CPU FP32 matmul is itself rounded, so use a standard dot-product
    # bound plus one FP32 output rounding term. This checks the reference
    # implementation without calling it mathematically exact.
    eps32 = torch.finfo(torch.float32).eps
    k = x.shape[-1]
    gamma = (k * eps32) / (1 - k * eps32)
    allowance = gamma * magnitudes + eps32 * cpu64.abs() + torch.finfo(torch.float32).tiny
    return _error_report(cpu64, fp32_reference, allowance)


def _high_precision_references(x, packed, scales, *, chunk_rows=256):
    """Return bounded FP32 reference results for the two arithmetic contracts.

    The first result uses the mathematical dequantized weight represented in
    FP32, and the second rounds that weight to the activation dtype first.
    Neither result is an exact real-number computation.
    """
    group_size = _validate_inputs(x, packed, scales)
    if not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool) or chunk_rows < 1:
        raise ValueError("chunk_rows must be a positive integer")
    if chunk_rows % 2:
        raise ValueError("chunk_rows must be even because packed rows are pairs")
    x32 = x.detach().reshape(-1, x.shape[-1]).float()
    exact, rounded, magnitudes = [], [], []
    with _tf32_disabled():
        for start in range(0, packed.shape[0] * 2, chunk_rows):
            stop = min(start + chunk_rows, packed.shape[0] * 2)
            pair = packed[start // 2:stop // 2]
            q = torch.stack((pair & 15, pair >> 4), dim=1).reshape(-1, pair.shape[1])
            q = q.float() - 8
            scale = scales[start:stop].float()
            w = q * scale.repeat_interleave(group_size, dim=1)
            exact.append(x32 @ w.T)
            rounded.append(x32 @ w.to(x.dtype).float().T)
            magnitudes.append(x32.abs() @ w.abs().T)
    return tuple(torch.cat(parts, dim=1) for parts in (exact, rounded, magnitudes))


def _error_report(expected, actual, allowance):
    expected = expected.detach().cpu().double()
    allowance = allowance.detach().cpu().double()
    actual = actual.detach().reshape_as(expected).cpu().double()
    diff = (actual - expected).abs()
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    failed = (~finite) | (diff > allowance)
    safe_diff = torch.where(finite, diff, torch.full_like(diff, float("inf")))
    ratio = torch.where(
        finite & (allowance > 0),
        diff / allowance.clamp_min(torch.finfo(torch.float64).tiny),
        torch.full_like(diff, float("inf")),
    )
    index = int(safe_diff.argmax())
    over_index = int(ratio.argmax())
    row, col = divmod(index, expected.shape[1])
    over_row, over_col = divmod(over_index, expected.shape[1])
    return {
        "finite": bool(finite.all()),
        "nan_or_inf_elements": int((~finite).sum()),
        "max_abs": float(safe_diff.max()),
        "mean_abs": float(safe_diff.mean()),
        "p99_abs": float(torch.quantile(safe_diff.flatten(), 0.99)),
        # Stabilized relative error; near-zero outputs are judged by allowance.
        "max_relative": float((safe_diff / expected.abs().clamp_min(1e-12)).max()),
        "failed_elements": int(failed.sum()),
        "elements": failed.numel(),
        "failure_ratio": float(failed.sum()) / max(1, failed.numel()),
        "worst": {"row": row, "column": col, "expected": float(expected[row, col]),
                  "actual": float(actual[row, col]), "allowance": float(allowance[row, col])},
        "worst_over_budget": {
            "row": over_row,
            "column": over_col,
            "ratio": float(ratio[over_row, over_col]),
            "expected": float(expected[over_row, over_col]),
            "actual": float(actual[over_row, over_col]),
            "allowance": float(allowance[over_row, over_col]),
        } if bool(failed.any()) else None,
    }


def _check_parity(x, packed, scales, reference, fused, *, chunk_rows=256):
    exact, rounded, magnitudes = _high_precision_references(
        x, packed, scales, chunk_rows=chunk_rows
    )
    # Conservative FP32 dot/group accumulation budget plus final output rounding.
    # No empirical absolute threshold fitted to one observed failure.
    operations = x.shape[-1] + 3
    eps32 = torch.finfo(torch.float32).eps
    gamma = operations * eps32 / (1 - operations * eps32)
    output_eps = torch.finfo(x.dtype).eps
    tiny = torch.finfo(x.dtype).tiny
    allowance = gamma * magnitudes + output_eps * exact.abs() + tiny
    fused_report = _error_report(exact, fused, allowance)
    fallback_allowance = gamma * magnitudes + output_eps * rounded.abs() + tiny
    fallback_report = _error_report(rounded, reference, fallback_allowance)
    cpu_report = None
    if packed.numel() * 2 <= 65_536:
        cpu_report = _validate_fp32_reference_with_cpu64(x, packed, scales, exact)
    return {
        "fused_vs_exact": fused_report,
        "fallback_vs_rounded": fallback_report,
        "cpu_fp64_reference": cpu_report,
        "reference_contracts": {
            "exact": "dequantized weight, FP32 matmul, TF32 disabled",
            "rounded": "weight rounded to activation dtype, FP32 matmul, TF32 disabled",
        },
        "tf32_disabled": True,
        "reference_operations": operations,
        "reference_gamma": gamma,
        "weight_rounding_max_abs": float((exact - rounded).abs().max()),
        "fallback_failure_policy": (
            "fallback is independently checked; a checked fallback failure fails "
            "the case and is reported separately"
        ),
        "passed": (
            fused_report["failed_elements"] == 0
            and fallback_report["failed_elements"] == 0
            and (cpu_report is None or cpu_report["failed_elements"] == 0)
        ),
        "criterion": "FP32 accumulation bound plus activation-dtype output rounding",
    }


def _tflops(tokens: int, n: int, k: int, milliseconds: float) -> float:
    if milliseconds <= 0:
        return 0.0
    return 2.0 * tokens * n * k / (milliseconds * 1e9)


def _synthetic_cases(args, device: torch.device) -> Iterable[tuple]:
    # Gemma4 E2B projection dimensions. The exact attention K/V dimensions
    # differ between sliding and full layers, so both representative cases
    # are included.
    shapes = [
        ("q_proj", 4096, 1536),
        ("k_proj_sliding", 2048, 1536),
        ("v_proj_sliding", 2048, 1536),
        ("k_proj_full", 4096, 1536),
        ("o_proj", 1536, 4096),
        ("gate_proj", 4096, 1536),
        ("up_proj", 4096, 1536),
        ("down_proj", 1536, 4096),
        ("per_layer_model_projection", 8960, 1536),
        ("lm_head", 262144, 1536),
    ]
    for seed in args.seeds:
        for name, n, k in shapes:
            if name == "lm_head" and args.skip_lm_head:
                continue
            yield _make_case(
                f"{name}.seed{seed}", n, k, args.tokens, args.group_size,
                device, args.dtype, seed,
            )


def _edge_cases(args, device: torch.device) -> Iterable[tuple]:
    """Deterministic edge inputs that the shape-only suite does not cover."""
    for seed in args.seeds:
        generator = torch.Generator(device="cpu").manual_seed(seed + 7)

        def emit(label, n, k, tokens, x, q, scales):
            packed = _pack_int4(q).to(device)
            yield (
                f"{label}.seed{seed}",
                x.to(device=device, dtype=args.dtype),
                packed,
                scales.to(device=device, dtype=torch.float16),
            )

        n, k = 6, 64
        q = torch.randint(-8, 8, (n, k), generator=generator, dtype=torch.int8)
        yield from emit(
            "edge.zero_input", n, k, 1,
            torch.zeros(1, k),
            q,
            torch.full((n, k // 64), 0.05),
        )

        n, k = 130, 128
        q = torch.full((n, k), 7, dtype=torch.int8)
        x = torch.empty(1, k)
        x[:, 0::2] = 1.0
        x[:, 1::2] = -1.0
        yield from emit(
            "edge.cancellation", n, k, 1, x, q,
            torch.full((n, k // 64), 0.125),
        )

        n, k = 6, 256
        q = torch.randint(-8, 8, (n, k), generator=generator, dtype=torch.int8)
        yield from emit(
            "edge.large_input", n, k, 1,
            torch.randn(1, k, generator=generator) * 100.0,
            q,
            torch.full((n, k // 64), 0.05),
        )

        n, k = 6, 256
        q = torch.randint(-8, 8, (n, k), generator=generator, dtype=torch.int8)
        yield from emit(
            "edge.small_input", n, k, 19,
            torch.randn(19, k, generator=generator) * 1e-4,
            q,
            torch.full((n, k // 64), 1e-3),
        )

        n, k = 130, 1536
        q = torch.randint(-8, 8, (n, k), generator=generator, dtype=torch.int8)
        mixed = torch.empty(n, k // 64)
        mixed[:, 0::2] = 1e-4
        mixed[:, 1::2] = 0.1
        yield from emit(
            "edge.mixed_scales", n, k, 2,
            torch.randn(2, k, generator=generator),
            q,
            mixed,
        )

        n, k = 2, 64
        q = torch.randint(-8, 8, (n, k), generator=generator, dtype=torch.int8)
        yield from emit(
            "edge.n_tail_2", n, k, 1,
            torch.randn(1, k, generator=generator),
            q,
            torch.full((n, k // 64), 0.07),
        )

        n, k = 6, 128
        q = torch.randint(-8, 8, (n, k), generator=generator, dtype=torch.int8)
        yield from emit(
            "edge.k_128", n, k, 19,
            torch.randn(19, k, generator=generator),
            q,
            torch.full((n, k // 64), 0.03),
        )

        n, k = 258, 1536
        q = torch.randint(-8, 8, (n, k), generator=generator, dtype=torch.int8)
        yield from emit(
            "edge.n_tail_258", n, k, 1,
            torch.randn(1, k, generator=generator),
            q,
            torch.full((n, k // 64), 0.02),
        )


def _real_cases(args, device: torch.device) -> Iterable[tuple]:
    from miniservellm.model_adapter.adapters.gemma4_hf_model import load_gemma4_config
    from miniservellm.model_adapter.gemma4_config import convert_gemma4_config
    from miniservellm.runtime.gemma4_quant import (
        quantize_checkpoint_tensor,
        quantize_gemma4_weights,
    )
    from miniservellm.safetensors_io import StreamingSafetensors

    model_dir = Path(args.model).expanduser()
    model_config = convert_gemma4_config(load_gemma4_config(model_dir))
    started = time.perf_counter()
    source = StreamingSafetensors(model_dir / "model.safetensors")
    prequantized = {}
    for name in (
        "model.language_model.embed_tokens.weight",
        "model.language_model.embed_tokens_per_layer.weight",
    ):
        prequantized[name] = quantize_checkpoint_tensor(
            source,
            name,
            device=device,
            group_size=args.group_size,
            compute_dtype=args.dtype,
        )
    weights = quantize_gemma4_weights(
        model_dir,
        model_config,
        device=device,
        group_size=args.group_size,
        compute_dtype=args.dtype,
        prequantized=prequantized,
    )
    print(f"real INT4 weights loaded in {time.perf_counter() - started:.1f}s")

    for seed in args.seeds:
        # --limit is per seed so a multi-seed run covers the same projection
        # set for every activation seed instead of spending the budget on the
        # first seed only.
        seen = 0
        generator = torch.Generator(device="cpu").manual_seed(seed)
        if not args.skip_lm_head and (not args.limit or seen < args.limit):
            weight = weights.lm_head
            yield (
                f"lm_head.seed{seed}",
                torch.randn(
                    (args.tokens, weight.packed.shape[1]),
                    generator=generator,
                    dtype=args.dtype,
                ).to(device),
                weight.packed, weight.scales,
            )
            seen += 1
        if not args.limit or seen < args.limit:
            ple = weights.per_layer_model_projection
            yield (
                f"per_layer_model_projection.seed{seed}",
                torch.randn(
                    (args.tokens, ple.packed.shape[1]),
                    generator=generator,
                    dtype=args.dtype,
                ).to(device),
                ple.packed,
                ple.scales,
            )
            seen += 1
        for layer_idx, layer in enumerate(weights.layers):
            if args.limit and seen >= args.limit:
                break
            for name in (
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "per_layer_input_gate",
                "per_layer_projection",
            ):
                if args.limit and seen >= args.limit:
                    break
                weight = getattr(layer, name)
                yield (
                    f"layer{layer_idx}.{name}.seed{seed}",
                    torch.randn(
                        (args.tokens, weight.packed.shape[1]),
                        generator=generator,
                        dtype=args.dtype,
                    ).to(device),
                    weight.packed,
                    weight.scales,
                )
                seen += 1
    del weights
    gc.collect()


def _reference_chunked(x, packed, scales, chunk_rows):
    """Rounded-weight reference without materializing a huge dequantized tile."""
    if chunk_rows < 2 or chunk_rows % 2:
        raise ValueError("reference chunk_rows must be even and >= 2")
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


def _run_case(args, case) -> bool:
    name, x, packed, scales = case
    reference = _reference_chunked(
        x, packed, scales, args.reference_chunk_rows
    )
    fused = int4_dequant_matmul(x, packed, scales)
    max_abs, mean_abs = _format_error(reference, fused)

    fused_samples = [
        _cuda_time(
            lambda: int4_dequant_matmul(x, packed, scales),
            args.warmup,
            args.iters,
        )
        for _ in range(args.rounds)
    ]
    fused_ms = _median(fused_samples)
    fallback_ms = None
    fallback_samples = None
    if not args.skip_fallback:
        fallback_samples = [
            _cuda_time(
                lambda: _reference_chunked(
                    x, packed, scales, args.reference_chunk_rows
                ),
                args.warmup,
                max(1, min(args.iters, 20)),
            )
            for _ in range(min(args.rounds, 3))
        ]
        fallback_ms = _median(fallback_samples)

    n = packed.shape[0] * 2
    tokens = x.shape[0]
    logical_bytes = _logical_weight_bytes(n, x.shape[-1])
    logical_gbps = (
        logical_bytes / (fused_ms * 1e6) if fused_ms > 0 else 0.0
    )
    suffix = (
        f" fallback_ms={fallback_ms:.3f}"
        f" speedup={fallback_ms / fused_ms:.2f}x"
        if fallback_ms is not None
        else ""
    )
    print(
        f"{name:32s} impl={args.impl} shape=[{tokens},{x.shape[-1]}]x[{n},{x.shape[-1]}]"
        f" fused_ms={fused_ms:.3f} range={min(fused_samples):.3f}-{max(fused_samples):.3f}"
        f" bw={logical_gbps:.1f}GB/s"
        f" tflops={_tflops(tokens, n, x.shape[-1], fused_ms):.2f}"
        f"{suffix} max_abs={max_abs:.5f} mean_abs={mean_abs:.6f}"
    )
    diagnostics = _check_parity(
        x, packed, scales, reference, fused, chunk_rows=args.chunk_rows
    )
    print(json.dumps({
        "case": name,
        "impl": args.impl,
        "fused_ms_samples": [round(v, 4) for v in fused_samples],
        "fused_ms_median": round(fused_ms, 4),
        "fused_ms_range": [round(min(fused_samples), 4), round(max(fused_samples), 4)],
        "fallback_ms_median": round(fallback_ms, 4) if fallback_ms is not None else None,
        "fallback_ms_samples": (
            [round(v, 4) for v in fallback_samples]
            if fallback_samples is not None else None
        ),
        "logical_weight_bytes": logical_bytes,
        "logical_weight_gbps": round(logical_gbps, 2),
        "tflops": round(_tflops(tokens, n, x.shape[-1], fused_ms), 4),
        "parity": diagnostics,
    }, allow_nan=True), flush=True)
    return diagnostics["passed"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Optional real Gemma4 checkpoint directory")
    parser.add_argument("--limit", type=int, default=0, help="Limit real projection cases")
    parser.add_argument("--tokens", type=int, default=1, help="Rows in the activation matrix")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=256,
        help="Even logical output rows per bounded FP32 reference chunk",
    )
    parser.add_argument(
        "--reference-chunk-rows",
        type=int,
        default=4096,
        help="Even logical output rows per chunked dequantize+linear reference",
    )
    parser.add_argument(
        "--int4-impl",
        choices=("legacy", "warp", "pair", "pair_vec", "pair_vec8"),
        default="pair_vec8",
        help="CUDA INT4 GEMV implementation under test",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Repeated timing rounds per case (median and range reported)",
    )
    parser.add_argument(
        "--edge-cases",
        action="store_true",
        help="Add zero/cancellation/extreme/scale/N-tail edge cases",
    )
    parser.add_argument("--skip-fallback", action="store_true")
    parser.add_argument("--skip-lm-head", action="store_true",
                        help="Skip the large vocabulary projection (included by default)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261001],
                        help="Synthetic input/weight seeds")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (
        args.tokens < 1
        or args.group_size < 1
        or args.warmup < 0
        or args.iters < 1
        or args.limit < 0
        or args.chunk_rows < 1
        or args.chunk_rows % 2
        or args.reference_chunk_rows < 2
        or args.reference_chunk_rows % 2
        or args.rounds < 1
    ):
        parser.error(
            "tokens/group-size/iters/chunk-rows/rounds must be positive; "
            "chunk-rows and reference-chunk-rows must be even; "
            "warmup/limit nonnegative"
        )
    if args.group_size != 64:
        parser.error(
            "the CUDA INT4 kernel requires group-size=64; unsupported "
            "groups are covered by the dispatch tests, not this benchmark"
        )
    if not args.seeds:
        parser.error("at least one seed is required")
    args.dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    args.impl = args.int4_impl
    os.environ["MINI_LLM_INT4_IMPL"] = args.int4_impl

    if not torch.cuda.is_available() or args.device != "cuda":
        print("CUDA unavailable; run this benchmark in the RTX 3060 Remote-SSH workspace.")
        print(f"torch.cuda.is_available()={torch.cuda.is_available()} device={args.device}")
        return
    if _KERNEL_IMPORT_ERROR is not None:
        raise RuntimeError(
            "mini_llm_kernels import failed; install it with `pip install -e .` "
            "from the mini-llm-kernels checkout"
        ) from _KERNEL_IMPORT_ERROR
    if not _HAS_INT4_CUDA:
        raise RuntimeError(
            "mini_llm_kernels INT4 CUDA extension is unavailable. "
            "Rebuild mini-llm-kernels with CUDA and sm_86 support."
        )

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    print(
        f"device={props.name} capability={props.major}.{props.minor} "
        f"vram_free={free_bytes / 2**30:.2f}GiB/{total_bytes / 2**30:.2f}GiB "
        f"dtype={args.dtype} tokens={args.tokens} impl={args.impl}"
    )
    print("INT4 CUDA extension: loaded")
    kernel_attributes = None
    if args.impl in ("pair", "pair_vec"):
        try:
            from mini_llm_kernels.kernels.int4_matmul import (
                int4_pair_kernel_attributes,
            )
            kernel_attributes = {
                impl: int4_pair_kernel_attributes(
                    pairs_per_block=4,
                    dtype=args.dtype,
                    vectorized=(impl == "pair_vec"),
                )
                for impl in ("pair", "pair_vec")
            }
            print(json.dumps({"kernel_attributes": kernel_attributes}))
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"kernel_attributes_error": str(exc)}))
    gpu_state_before = _gpu_state()
    print(f"gpu_state_before: {gpu_state_before}")

    if args.model:
        cases = _real_cases(args, device)
    else:
        cases = _synthetic_cases(args, device)
        if args.edge_cases:
            cases = chain(cases, _edge_cases(args, device))
    failures = []
    count = 0
    for case in cases:
        count += 1
        if not _run_case(args, case):
            failures.append(case[0])
    gpu_state_after = _gpu_state()
    print(json.dumps({
        "cases": count,
        "failed_cases": failures,
        "impl": args.impl,
        "kernel_attributes": kernel_attributes,
        "dtype": str(args.dtype),
        "rounds": args.rounds,
        "iters": args.iters,
        "gpu_state_before": gpu_state_before,
        "gpu_state_after": gpu_state_after,
    }), flush=True)
    if failures:
        raise RuntimeError(f"INT4 parity failed for {len(failures)}/{count} cases: {failures}")
    print("GEMMA4 CUDA INT4 PARITY OK")


if __name__ == "__main__":
    main()
