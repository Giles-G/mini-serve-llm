"""Stage D validation: cached vs uncached parity and decode throughput.

Runs the same greedy generation twice on the Gemma4 eager runner:
1. full-recompute (no cache) — the stage C reference;
2. incremental decode with the layer-aware KV cache (stage D).

Exit code 0 requires both paths to produce identical token IDs. Also prints
decode tok/s for the cached path (the stage D performance metric).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from miniservellm.model_adapter.adapters.gemma4_hf_model import (
    load_gemma4_config,
    load_gemma4_text_weights,
)
from miniservellm.model_adapter.gemma4_config import convert_gemma4_config
from miniservellm.runtime.gemma4_runner import Gemma4EagerTextRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models" / "gemma-4-E2B-it"))
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--prompt", default="你好，请简要介绍一下KV Cache的作用。")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()

    model_dir = Path(args.model).expanduser()
    model_config = convert_gemma4_config(load_gemma4_config(model_dir))
    started = time.perf_counter()
    weights = load_gemma4_text_weights(model_dir, device=args.device, dtype=torch.bfloat16)
    runner = Gemma4EagerTextRunner(model_config, weights)
    print(f"device={args.device} load_s={time.perf_counter() - started:.1f}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    eos_ids = (1, 106, 50)
    gen_config = model_dir / "generation_config.json"
    if gen_config.exists():
        with open(gen_config, "r", encoding="utf-8") as handle:
            raw_eos = json.load(handle).get("eos_token_id", [1, 106, 50])
        eos_ids = tuple(raw_eos) if isinstance(raw_eos, list) else (int(raw_eos),)

    print(f"prompt_tokens={len(prompt_ids)} max_new={args.max_new_tokens}")

    ref_ids, ref_stats = runner.generate(
        prompt_ids, max_new_tokens=args.max_new_tokens, eos_token_ids=eos_ids
    )
    print(
        f"[uncached] tokens={ref_stats['generated_tokens']} "
        f"decode_tok_s={ref_stats['decode_tok_s']:.3f}"
    )

    cached_ids, cached_stats = runner.generate_cached(
        prompt_ids, max_new_tokens=args.max_new_tokens, eos_token_ids=eos_ids
    )
    print(
        f"[cached]   tokens={cached_stats['generated_tokens']} "
        f"prefill_s={cached_stats['prefill_seconds']:.2f} "
        f"decode_tok_s={cached_stats['decode_tok_s']:.2f} "
        f"total_s={cached_stats['total_seconds']:.2f}"
    )

    if ref_ids == cached_ids:
        print(f"PARITY OK ({len(ref_ids)} tokens identical)")
        print("text:", tokenizer.decode(cached_ids, skip_special_tokens=True))
    else:
        first_diff = next(
            (i for i, (a, b) in enumerate(zip(ref_ids, cached_ids)) if a != b),
            min(len(ref_ids), len(cached_ids)),
        )
        print(f"PARITY FAILED at token {first_diff}")
        print(f"  uncached: {ref_ids}")
        print(f"  cached:   {cached_ids}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
