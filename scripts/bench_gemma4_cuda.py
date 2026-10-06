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
import gc
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
    unsigned = (w_q.to(torch.int16) + 8).clamp(0, 15).to(torch.uint8)
    return unsigned[0::2] | (unsigned[1::2] << 4)


def _make_case(
    name: str,
    n: int,
    k: int,
    tokens: int,
    group_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
    if n % 2 or k % group_size:
        raise ValueError(f"{name}: requires even N and K divisible by group size")
    generator = torch.Generator(device="cpu").manual_seed(20261001 + n + k)
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


def _format_error(reference: torch.Tensor, actual: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - reference.float()).abs()
    return float(diff.max().item()), float(diff.mean().item())


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
        ("per_layer_projection", 8960, 1536),
    ]
    for name, n, k in shapes:
        yield _make_case(
            name,
            n,
            k,
            args.tokens,
            args.group_size,
            device,
            args.dtype,
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

    seen = 0
    for layer_idx, layer in enumerate(weights.layers):
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
            weight = getattr(layer, name)
            yield (
                f"layer{layer_idx}.{name}",
                torch.randn(
                    (args.tokens, weight.packed.shape[1]),
                    device=device,
                    dtype=args.dtype,
                ),
                weight.packed,
                weight.scales,
            )
            seen += 1
            if args.limit and seen >= args.limit:
                del weights
                gc.collect()
                return
    del weights
    gc.collect()


def _run_case(args, case) -> None:
    name, x, packed, scales = case
    reference = _int4_dequant_matmul_pytorch(x, packed, scales)
    fused = int4_dequant_matmul(x, packed, scales)
    max_abs, mean_abs = _format_error(reference, fused)

    fused_ms = _cuda_time(
        lambda: int4_dequant_matmul(x, packed, scales),
        args.warmup,
        args.iters,
    )
    fallback_ms = None
    if not args.skip_fallback:
        fallback_ms = _cuda_time(
            lambda: _int4_dequant_matmul_pytorch(x, packed, scales),
            args.warmup,
            max(1, min(args.iters, 20)),
        )

    n = packed.shape[0] * 2
    suffix = (
        f" fallback_ms={fallback_ms:.3f}"
        f" speedup={fallback_ms / fused_ms:.2f}x"
        if fallback_ms is not None
        else ""
    )
    print(
        f"{name:32s} shape=[{args.tokens},{x.shape[-1]}]x[{n},{x.shape[-1]}]"
        f" fused_ms={fused_ms:.3f} tflops={_tflops(args.tokens, n, x.shape[-1], fused_ms):.2f}"
        f"{suffix} max_abs={max_abs:.5f} mean_abs={mean_abs:.6f}"
    )
    tolerance = 3e-3 if x.dtype == torch.float16 else 1e-2
    if max_abs > tolerance:
        raise RuntimeError(
            f"{name}: fused INT4 result differs from reference: "
            f"max_abs={max_abs:.6f} > tolerance={tolerance}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Optional real Gemma4 checkpoint directory")
    parser.add_argument("--limit", type=int, default=0, help="Limit real projection cases")
    parser.add_argument("--tokens", type=int, default=1, help="Rows in the activation matrix")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--skip-fallback", action="store_true")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

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
        f"dtype={args.dtype} tokens={args.tokens}"
    )
    print("INT4 CUDA extension: loaded")

    cases = _real_cases(args, device) if args.model else _synthetic_cases(args, device)
    for case in cases:
        _run_case(args, case)
    print("GEMMA4 CUDA INT4 PARITY OK")


if __name__ == "__main__":
    main()
