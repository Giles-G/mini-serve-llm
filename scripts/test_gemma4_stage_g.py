"""Stage G validation: INT4 quantization quality, memory, and speed.

Sequence (weights resident one at a time to fit 16GB unified memory):
1. bf16 reference runner -> greedy tokens (quality baseline) + tok/s.
2. Free bf16; stream-quantize the checkpoint to INT4 (packed, gs=64).
3. Greedy generate with the same prompt; record token overlap vs bf16,
   decode tok/s, and packed memory footprint.

The quantized path is intentionally the SAME runner code: Int4Weight
duck-types tensor matmul/embedding call sites. On CUDA machines the matmul
dispatches to the fused mini_llm_kernels INT4 kernel (same packed format as
the Qwen runtime), making this the 3060 6GB configuration.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from miniservellm.model_adapter.adapters.gemma4_hf_model import load_gemma4_config
from miniservellm.model_adapter.gemma4_config import convert_gemma4_config
from miniservellm.runtime.gemma4_quant import quantize_gemma4_weights
from miniservellm.runtime.gemma4_runner import Gemma4EagerTextRunner


def _tokenizer(model_dir: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)


def _prompt_ids(tokenizer, prompt: str) -> list[int]:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    return tokenizer.encode(text, add_special_tokens=False)


def _eos(model_dir: Path) -> tuple[int, ...]:
    raw = [1, 106, 50]
    gen_config = model_dir / "generation_config.json"
    if gen_config.exists():
        with open(gen_config, "r", encoding="utf-8") as handle:
            raw = json.load(handle).get("eos_token_id", raw)
    return tuple(raw) if isinstance(raw, list) else (int(raw),)


def _int4_bytes(weights) -> int:
    total = 0
    for obj in [weights] + list(weights.layers):
        for name, value in vars(obj).items():
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
            elif hasattr(value, "nbytes"):
                total += value.nbytes()
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models" / "gemma-4-E2B-it"))
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--prompt", default="你好，请简要介绍一下KV Cache的作用。")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--skip-reference", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model).expanduser()
    model_config = convert_gemma4_config(load_gemma4_config(model_dir))
    tokenizer = _tokenizer(model_dir)
    eos_ids = _eos(model_dir)
    prompt_ids = _prompt_ids(tokenizer, args.prompt)
    print(f"device={args.device} prompt_tokens={len(prompt_ids)} max_new={args.max_new_tokens}")

    ref_ids: list[int] = []
    if not args.skip_reference:
        started = time.perf_counter()
        weights = __import__(
            "miniservellm.model_adapter.adapters.gemma4_hf_model", fromlist=["load_gemma4_text_weights"]
        ).load_gemma4_text_weights(model_dir, device=args.device, dtype=torch.bfloat16)
        runner = Gemma4EagerTextRunner(model_config, weights)
        print(f"bf16 loaded in {time.perf_counter() - started:.1f}s")
        ref_ids, ref_stats = runner.generate_cached(
            prompt_ids, max_new_tokens=args.max_new_tokens, eos_token_ids=eos_ids
        )
        print(f"[bf16] decode_tok_s={ref_stats['decode_tok_s']:.2f}")
        print(f"[bf16] text: {tokenizer.decode(ref_ids, skip_special_tokens=True)!r}")
        del runner, weights
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()

    # ---- INT4 ---------------------------------------------------------------
    started = time.perf_counter()
    q_weights = quantize_gemma4_weights(
        model_dir, model_config, device=args.device, group_size=args.group_size
    )
    print(f"int4 quantized+loaded in {time.perf_counter() - started:.1f}s")
    print(f"int4 packed size = {_int4_bytes(q_weights) / 2**30:.2f} GiB (bf16 reference 9.5 GiB)")

    q_runner = Gemma4EagerTextRunner(model_config, q_weights)
    int4_ids, int4_stats = q_runner.generate_cached(
        prompt_ids, max_new_tokens=args.max_new_tokens, eos_token_ids=eos_ids
    )
    print(
        f"[int4] decode_tok_s={int4_stats['decode_tok_s']:.2f} "
        f"prefill_s={int4_stats['prefill_seconds']:.2f} total_s={int4_stats['total_seconds']:.2f}"
    )
    print(f"[int4] text: {tokenizer.decode(int4_ids, skip_special_tokens=True)!r}")

    if ref_ids:
        overlap = sum(1 for a, b in zip(ref_ids, int4_ids) if a == b)
        first_diff = next(
            (i for i, (a, b) in enumerate(zip(ref_ids, int4_ids)) if a != b),
            min(len(ref_ids), len(int4_ids)),
        )
        print(
            f"[quality] token_overlap={overlap}/{len(ref_ids)} "
            f"first_divergence_index={first_diff}"
        )
    print("STAGE G OK")


if __name__ == "__main__":
    main()
