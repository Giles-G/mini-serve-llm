#!/usr/bin/env python3
"""Compare unquantized MLX Gemma4 logits against the PyTorch reference."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from miniservellm.model_adapter.adapters.gemma4_hf_model import (
    load_gemma4_config,
    load_gemma4_text_weights,
)
from miniservellm.model_adapter.gemma4_config import convert_gemma4_config
from miniservellm.mlx.gemma4 import Gemma4MLXCache, build_gemma4_mlx
from miniservellm.runtime.gemma4_runner import Gemma4EagerTextRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    path = Path(args.model).expanduser()
    config = convert_gemma4_config(load_gemma4_config(path))
    ids = [1, 2]

    torch_weights = load_gemma4_text_weights(
        path,
        device=args.device,
        dtype=torch.float16,
    )
    reference = Gemma4EagerTextRunner(config, torch_weights)
    with torch.inference_mode():
        torch_logits = reference.forward(torch.tensor(ids, device=args.device))[-1]
    torch_top = int(torch.argmax(torch_logits).item())
    torch_values = torch_logits.detach().float().cpu().numpy()
    del reference, torch_weights, torch_logits
    gc.collect()
    if args.device == "mps":
        torch.mps.empty_cache()

    mlx_model = build_gemma4_mlx(path, config)
    cache = Gemma4MLXCache(config, max_context=16)
    import mlx.core as mx

    mlx_logits = mlx_model.forward(mx.array([ids], dtype=mx.uint32), cache)[0, -1]
    mx.eval(mlx_logits)
    mlx_values = np.asarray(mlx_logits, dtype=np.float32)
    mlx_top = int(np.argmax(mlx_values))
    print(
        {
            "torch_top": torch_top,
            "mlx_top": mlx_top,
            "top_match": torch_top == mlx_top,
            "max_abs_diff": float(np.max(np.abs(torch_values - mlx_values))),
            "mean_abs_diff": float(np.mean(np.abs(torch_values - mlx_values))),
        }
    )


if __name__ == "__main__":
    main()
