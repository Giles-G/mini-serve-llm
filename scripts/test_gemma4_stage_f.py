"""Stage F validation: paged-cache engine parity and throughput.

Sequence (weights loaded one at a time to fit 16GB unified memory):
1. Direct eager runner (stage D reference): greedy 24 tokens -> reference ids.
2. Free weights, then run the same request through Stage5Engine with the
   Gemma4 paged KV cache (stage F) and compare token ids.

Exit 0 requires exact parity; also prints engine tok/s.
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

from miniservellm.model_adapter.adapters.gemma4_hf_model import (
    load_gemma4_config,
    load_gemma4_text_weights,
)
from miniservellm.model_adapter.gemma4_config import convert_gemma4_config


def _load_tokenizer(model_dir: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)


def _eos_ids(model_dir: Path) -> tuple[int, ...]:
    raw = [1, 106, 50]
    gen_config = model_dir / "generation_config.json"
    if gen_config.exists():
        with open(gen_config, "r", encoding="utf-8") as handle:
            raw = json.load(handle).get("eos_token_id", raw)
    return tuple(raw) if isinstance(raw, list) else (int(raw),)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models" / "gemma-4-E2B-it"))
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--prompt", default="你好，请简要介绍一下KV Cache的作用。")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()

    model_dir = Path(args.model).expanduser()
    model_config = convert_gemma4_config(load_gemma4_config(model_dir))
    tokenizer = _load_tokenizer(model_dir)
    eos_ids = _eos_ids(model_dir)
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    print(f"device={args.device} prompt_tokens={len(prompt_ids)} max_new={args.max_new_tokens}")

    # ---- reference: direct eager runner (stage D) --------------------------
    from miniservellm.runtime.gemma4_runner import Gemma4EagerTextRunner

    started = time.perf_counter()
    weights = load_gemma4_text_weights(model_dir, device=args.device, dtype=torch.bfloat16)
    runner = Gemma4EagerTextRunner(model_config, weights)
    print(f"reference loaded in {time.perf_counter() - started:.1f}s")
    ref_ids, ref_stats = runner.generate_cached(
        prompt_ids, max_new_tokens=args.max_new_tokens, eos_token_ids=eos_ids
    )
    print(f"[reference] decode_tok_s={ref_stats['decode_tok_s']:.2f} tokens={ref_ids}")

    del runner, weights
    gc.collect()
    if args.device == "mps":
        torch.mps.empty_cache()

    # ---- stage F: paged-cache engine ---------------------------------------
    from miniservellm.cache.gemma4_paged_kv_cache import Gemma4PagedKVCacheManager
    from miniservellm.config import EngineConfig
    from miniservellm.runtime.gemma4_engine_runner import Gemma4EngineModelRunner
    from miniservellm.runtime.inference_engine import Stage5Engine
    from miniservellm.scheduler.request import SamplingParams

    started = time.perf_counter()
    weights = load_gemma4_text_weights(model_dir, device=args.device, dtype=torch.bfloat16)
    engine_config = EngineConfig.create(
        device=args.device,
        dtype="bf16",
        block_size=128,
        num_gpu_blocks=128,
        max_batch_size=8,
        max_tokens_per_step=256,
        max_prefill_tokens_per_step=128,
        max_decode_requests_per_step=8,
        prefill_chunk_size=128,
        default_temperature=0.0,
        model_param_count=0,
        num_hidden_layers=model_config.num_hidden_layers,
        num_kv_heads=model_config.num_key_value_heads,
        head_dim=512,
    )
    engine_config.eos_token_id = eos_ids
    kv_cache_manager = Gemma4PagedKVCacheManager(engine_config, model_config)
    model_runner = Gemma4EngineModelRunner(engine_config, model_config, weights, kv_cache_manager)
    engine = Stage5Engine(
        engine_config=engine_config,
        model_config=model_config,
        model_runner=model_runner,
        tokenizer=tokenizer,
    )
    print(f"paged engine loaded in {time.perf_counter() - started:.1f}s")
    print("kv groups:", kv_cache_manager.debug_global_state()["cache_groups"])

    rid = engine.add_request(
        prompt_token_ids=prompt_ids,
        sampling_params=SamplingParams(temperature=0.0),
        max_new_tokens=args.max_new_tokens,
    )
    gen_started = time.perf_counter()
    engine.run_until_all_finished(max_steps=100000, collect_results=False)
    total_s = time.perf_counter() - gen_started
    info = engine.debug_request(rid)
    engine_ids = list(info["generated_token_ids"])
    print(f"[paged-engine] generated={info['generated']} total_s={total_s:.2f} tok/s={info['generated'] / total_s:.2f}")
    print("[paged-engine] text:", engine.get_text(rid))

    if ref_ids == engine_ids:
        print(f"STAGE F PARITY OK ({len(ref_ids)} tokens identical)")
    else:
        first_diff = next(
            (i for i, (a, b) in enumerate(zip(ref_ids, engine_ids)) if a != b),
            min(len(ref_ids), len(engine_ids)),
        )
        print(f"STAGE F PARITY FAILED at token {first_diff}")
        print(f"  reference: {ref_ids}")
        print(f"  paged:     {engine_ids}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
