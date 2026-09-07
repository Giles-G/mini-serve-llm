"""Qwen regression smoke: verify the shared engine still works after the
Gemma4 stage E changes (DecodeRunner CUDA-graph guard, multi-EOS support).

Runs one short greedy request through Stage5Engine with the Qwen2.5-0.5B
custom runner and asserts a clean finish with non-empty text.

Usage: python scripts/regression_qwen_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from miniservellm.config import EngineConfig
from miniservellm.model_adapter.adapters.qwen2_adapter import Qwen2Adapter
from miniservellm.model_adapter.hf_loader import load_model_bundle
from miniservellm.cache.kv_cache import KVCacheManager
from miniservellm.runtime.model_runner import TransformerModelRunner
from miniservellm.runtime.inference_engine import Stage5Engine
from miniservellm.scheduler.request import SamplingParams


def main() -> None:
    adapter = Qwen2Adapter()
    tokenizer, hf_config, model_config, hf_model, weights_cpu = load_model_bundle(
        adapter=adapter,
        model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        trust_remote_code=False,
        load_model_device="cpu",
        load_dtype=None,
    )
    engine_config = EngineConfig.create(
        device="auto",
        dtype="auto",
        block_size=128,
        num_gpu_blocks=256,
        max_batch_size=4,
        max_tokens_per_step=64,
        max_prefill_tokens_per_step=64,
        max_decode_requests_per_step=4,
        prefill_chunk_size=32,
        default_temperature=0.0,
        model_param_count=sum(p.numel() for p in hf_model.parameters()),
        num_hidden_layers=model_config.num_hidden_layers,
        num_kv_heads=model_config.num_key_value_heads,
        head_dim=model_config.head_dim,
    )
    if tokenizer.eos_token_id is not None:
        raw = tokenizer.eos_token_id
        engine_config.eos_token_id = int(raw[0] if isinstance(raw, list) else raw)

    weights = adapter.move_weights_to_device(
        weights_cpu,
        device=engine_config.device,
        dtype=engine_config.dtype,
    )
    kv_cache_manager = KVCacheManager(engine_config, model_config)
    model_runner = TransformerModelRunner(
        engine_config=engine_config,
        model_config=model_config,
        weights=weights,
        kv_cache_manager=kv_cache_manager,
    )
    engine = Stage5Engine(
        engine_config=engine_config,
        model_config=model_config,
        model_runner=model_runner,
        tokenizer=tokenizer,
    )
    rid = engine.add_request(
        text="用一句话介绍KV Cache。",
        sampling_params=SamplingParams(temperature=0.0),
        max_new_tokens=16,
    )
    engine.run_until_all_finished(max_steps=2000, collect_results=False)
    info = engine.debug_request(rid)
    text = engine.get_text(rid)
    print(f"finish={info['finish_reason']} generated={info['generated']}")
    print("text:", text)
    assert info["generated"] > 0, "no tokens generated"
    assert info["finish_reason"] in {"EOS", "MAX_NEW_TOKENS"}
    # block_table must be released after finish
    assert info["block_table"] == [], "KV cache blocks were not released"
    print("QWEN REGRESSION OK")


if __name__ == "__main__":
    main()
