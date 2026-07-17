#!/usr/bin/env python3
"""Check greedy token parity between Hugging Face Qwen2 and MLX eager inference.

This tool is the correctness gate for MLX performance benchmarks. It feeds the
same raw prompt token IDs into both implementations, compares one greedy token
at a time, and reports the first divergence plus logit error. It deliberately
does not use Ollama: Ollama's HTTP API returns generated text but not the raw
per-step logits/token IDs required for a strict parity check.

Example:
    python scripts/check_mlx_hf_parity.py \
        --prompt "请用至少1000字来介绍新疆地区，字数必须要够" \
        --max-new 32

HF runs in FP32 by default as the numerical reference. Use ``--hf-dtype fp16``
to compare the two FP16 execution paths while still keeping MLX eager decode.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from miniservellm.mlx.qwen2 import MLXKVCache
from miniservellm.mlx.runner import MLXQwen2Runner
from miniservellm.model_adapter.adapters.qwen2_adapter import Qwen2Adapter


def resolve_model_path(model: str) -> Path:
    """Resolve a model path without downloading a model during a parity check."""
    path = Path(model)
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, local_files_only=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare HF and MLX Qwen2 greedy token parity")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", required=True, help="Exact raw prompt text; no chat template is applied")
    parser.add_argument("--max-new", type=int, default=32, help="Number of greedy steps to compare")
    parser.add_argument("--hf-dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--stop-on-divergence", action="store_true", help="Stop immediately at first mismatched token")
    args = parser.parse_args()

    if not args.prompt.strip():
        raise ValueError("--prompt must not be empty")
    if args.max_new <= 0:
        raise ValueError("--max-new must be greater than 0")

    model_path = resolve_model_path(args.model)
    adapter = Qwen2Adapter()
    tokenizer = adapter.load_tokenizer(str(model_path), trust_remote_code=False)
    hf_config = adapter.load_hf_config(str(model_path), trust_remote_code=False)
    model_config = adapter.convert_hf_config(hf_config)
    safetensors_path = model_path / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"Missing model.safetensors: {safetensors_path}")

    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("--prompt tokenized to an empty sequence")

    hf_dtype = torch.float32 if args.hf_dtype == "fp32" else torch.float16
    hf_model = adapter.load_hf_model(str(model_path), device="cpu", torch_dtype=hf_dtype)
    mlx_runner = MLXQwen2Runner(model_config, safetensors_path)
    mlx_cache = MLXKVCache(model_config, len(prompt_ids) + args.max_new)

    hf_ids = torch.tensor([prompt_ids], dtype=torch.long)
    mlx_input = mx.array([prompt_ids], dtype=mx.uint32)
    divergences = 0
    first_divergence: int | None = None

    print(
        f"[parity] model={args.model} prompt_tokens={len(prompt_ids)} "
        f"max_new={args.max_new} hf_dtype={args.hf_dtype} mlx_dtype=fp16"
    )

    with torch.inference_mode():
        hf_outputs = hf_model(input_ids=hf_ids, use_cache=True)
        hf_logits = hf_outputs.logits[:, -1, :]
        hf_past = hf_outputs.past_key_values

        mlx_logits = mlx_runner.model.forward(mlx_input, mlx_cache)[:, -1, :]
        mx.eval(mlx_logits)

        for step in range(args.max_new):
            hf_token = int(torch.argmax(hf_logits, dim=-1).item())
            mlx_token = int(mx.argmax(mlx_logits, axis=-1).item())
            mlx_logits_cpu = torch.tensor(mlx_logits.tolist(), dtype=torch.float32)
            max_abs_logit_error = float(torch.max(torch.abs(hf_logits.float() - mlx_logits_cpu)))
            matched = hf_token == mlx_token
            if not matched:
                divergences += 1
                if first_divergence is None:
                    first_divergence = step

            print(
                f"step={step:03d} match={matched} hf_id={hf_token} mlx_id={mlx_token} "
                f"hf_text={tokenizer.decode([hf_token])!r} mlx_text={tokenizer.decode([mlx_token])!r} "
                f"max_abs_logit_error={max_abs_logit_error:.6f}"
            )
            if not matched and args.stop_on_divergence:
                break

            hf_outputs = hf_model(
                input_ids=torch.tensor([[hf_token]], dtype=torch.long),
                past_key_values=hf_past,
                use_cache=True,
            )
            hf_logits = hf_outputs.logits[:, -1, :]
            hf_past = hf_outputs.past_key_values

            mlx_logits = mlx_runner.model.forward(
                mx.array([[mlx_token]], dtype=mx.uint32), mlx_cache
            )[:, -1, :]
            mx.eval(mlx_logits)

    checked = step + 1
    print("\n[parity summary]")
    print(f"checked_steps={checked} divergent_steps={divergences} first_divergence_step={first_divergence}")
    if first_divergence is None:
        print("status=PASS greedy_token_parity=true")
    else:
        print("status=FAIL greedy_token_parity=false")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
