#!/usr/bin/env python3
"""Benchmark the project-native Gemma4 engine on a single MPS/CUDA device.

This benchmark intentionally separates:

* checkpoint loading;
* prompt prefill / first-token latency;
* incremental decode throughput.

``--ignore-eos`` is useful for fixed-length kernel measurements.  It changes
only benchmark stopping behavior; normal service generation still honors EOS.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from miniservellm.cache.gemma4_paged_kv_cache import Gemma4PagedKVCacheManager
from miniservellm.config import EngineConfig
from miniservellm.model_adapter.adapters.gemma4_hf_model import (
    load_gemma4_config,
    load_gemma4_text_weights,
)
from miniservellm.model_adapter.gemma4_config import convert_gemma4_config
from miniservellm.runtime.gemma4_engine_runner import Gemma4EngineModelRunner
from miniservellm.runtime.inference_engine import Stage5Engine
from miniservellm.scheduler.request import SamplingParams


class JsonTokenizer:
    """Small local tokenizer adapter for Gemma4 tokenizer.json files."""

    def __init__(self, path: Path):
        from tokenizers import Tokenizer

        self.backend = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.eos_token_id = self.backend.token_to_id("<eos>")

    def encode(self, text: str, add_special_tokens: bool = False):
        return self.backend.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids, **kwargs):
        return self.backend.decode(
            list(ids),
            skip_special_tokens=kwargs.get("skip_special_tokens", True),
        )


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def build_engine(model_path: Path, device: str, dtype: str):
    started = time.perf_counter()
    torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[dtype]
    model_config = convert_gemma4_config(load_gemma4_config(model_path))
    weights = load_gemma4_text_weights(
        model_path,
        device=device,
        dtype=torch_dtype,
    )
    engine_config = EngineConfig.create(
        device=device,
        dtype=dtype,
        block_size=128,
        num_gpu_blocks=128,
        max_batch_size=1,
        max_tokens_per_step=128,
        max_prefill_tokens_per_step=128,
        max_decode_requests_per_step=1,
        prefill_chunk_size=128,
        default_temperature=0.0,
        default_top_k=20,
        default_top_p=0.95,
        model_param_count=0,
        num_hidden_layers=model_config.num_hidden_layers,
        num_kv_heads=model_config.num_key_value_heads,
        head_dim=512,
    )
    engine_config.eos_token_id = (1, 106, 50)
    kv_cache_manager = Gemma4PagedKVCacheManager(engine_config, model_config)
    model_runner = Gemma4EngineModelRunner(
        engine_config,
        model_config,
        weights,
        kv_cache_manager,
    )
    tokenizer = JsonTokenizer(model_path)
    engine = Stage5Engine(
        engine_config=engine_config,
        model_config=model_config,
        model_runner=model_runner,
        tokenizer=tokenizer,
    )
    return engine, model_runner, tokenizer, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--prompt", default="请简要介绍新疆地区。")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--ignore-eos", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser()
    if args.device == "auto":
        args.device = "mps" if torch.backends.mps.is_available() else "cpu"
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)

    engine, model_runner, tokenizer, load_s = build_engine(
        model_path,
        args.device,
        args.dtype,
    )
    if args.ignore_eos:
        engine.eos_token_ids = set()

    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    request_id = engine.add_request(
        prompt_token_ids=prompt_ids,
        sampling_params=SamplingParams(temperature=0.0, top_k=20, top_p=0.95),
        max_new_tokens=args.max_new_tokens,
    )

    device = model_runner.device
    sync_device(device)
    generation_started = time.perf_counter()
    prefill_finished_at = None
    finished_at = None
    while engine.has_pending_work():
        result = engine.step()
        sync_device(device)
        now = time.perf_counter()
        if prefill_finished_at is None and any(
            event.kind == "prefill_sampled_first_token" for event in result.events
        ):
            prefill_finished_at = now
        if not engine.has_pending_work():
            finished_at = now

    if finished_at is None:
        finished_at = time.perf_counter()
    info = engine.debug_request(request_id)
    total_s = finished_at - generation_started
    prefill_s = (
        prefill_finished_at - generation_started
        if prefill_finished_at is not None
        else total_s
    )
    decode_s = max(0.0, finished_at - (prefill_finished_at or generation_started))
    decode_tokens = max(0, info["generated"] - 1)

    print(
        json.dumps(
            {
                "model": str(model_path),
                "device": str(device),
                "dtype": str(model_runner.dtype),
                "load_s": round(load_s, 3),
                "prompt_tokens": len(prompt_ids),
                "generated_tokens": info["generated"],
                "finish_reason": info["finish_reason"],
                "prefill_s_to_first_token": round(prefill_s, 3),
                "decode_s_after_first_token": round(decode_s, 3),
                "decode_tokens_after_first_token": decode_tokens,
                "decode_tok_s": round(decode_tokens / decode_s, 3) if decode_s else 0.0,
                "end_to_end_tok_s": round(info["generated"] / total_s, 3) if total_s else 0.0,
                "ignore_eos": args.ignore_eos,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
