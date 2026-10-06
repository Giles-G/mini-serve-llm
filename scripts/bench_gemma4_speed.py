#!/usr/bin/env python3
"""End-to-end Gemma4 speed benchmark for the project-native runner on CUDA.

Measures the two things that matter for the "was it slow, is it fast now"
question, on one device, in one process:

* ``--mode direct``  : ``Gemma4EagerTextRunner.generate_cached`` (prefill +
  incremental decode, no engine, no paged cache);
* ``--mode engine``  : ``Stage5Engine`` + ``Gemma4EngineModelRunner``
  (paged KV, scheduler, sampler — the serving path).

KV sharing, sliding window, PLE and logit soft-capping are identical on both
paths, so the two numbers are directly comparable.

``--int4`` quantizes the checkpoint to INT4 (group_size 64) instead of loading
bf16/fp16 weights. On a 6 GB laptop GPU this is not an optimization but a
requirement: the E2B text backbone is ~10.2 GB in bf16.

Note on the INT4 matmul backend: ``Int4Weight.matmul`` prefers the fused
``mini_llm_kernels`` CUDA kernel. ``--int4-backend auto`` uses that kernel;
``--int4-backend reference`` forces the PyTorch unpack path, which is correct
but slower. Use ``reference`` whenever the fused kernel is suspect, and
``auto`` to measure what the kernel actually delivers.
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
from miniservellm.runtime.gemma4_runner import Gemma4EagerTextRunner
from miniservellm.runtime.inference_engine import Stage5Engine
from miniservellm.scheduler.request import SamplingParams


class JsonTokenizer:
    """Local tokenizer adapter for Gemma4 ``tokenizer.json`` files."""

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


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def gpu_memory_gib() -> tuple[float, float]:
    free, total = torch.cuda.mem_get_info()
    return free / 2**30, total / 2**30


def load_weights(model_path: Path, device: str, dtype: torch.dtype, int4: bool, backend: str):
    """Load dense or INT4 text weights for the Gemma4 backbone."""
    started = time.perf_counter()
    model_config = convert_gemma4_config(load_gemma4_config(model_path))
    if int4:
        from miniservellm.runtime.gemma4_quant import (
            StreamingSafetensors,
            quantize_checkpoint_tensor,
            quantize_gemma4_weights,
        )

        # The PLE table [262144, 8960] is 4.4 GiB in bf16 and its own row
        # chunk is larger than this host's free RAM, so quantize it by
        # streaming instead of letting the loader materialize it whole.
        source = StreamingSafetensors(model_path / "model.safetensors")
        prequantized = {}
        for name in (
            "model.language_model.embed_tokens_per_layer.weight",
            "model.language_model.embed_tokens.weight",
        ):
            prequantized[name] = quantize_checkpoint_tensor(
                source, name, device=device, group_size=64, compute_dtype=dtype
            )
        weights = quantize_gemma4_weights(
            model_path,
            model_config,
            device=device,
            group_size=64,
            compute_dtype=dtype,
            prequantized=prequantized,
        )
    else:
        weights = load_gemma4_text_weights(model_path, device=device, dtype=dtype)
    return weights, model_config, time.perf_counter() - started


def weight_report(weights) -> dict:
    """Parameter count and on-device byte footprint of the loaded weights."""
    if hasattr(weights, "layers") and hasattr(weights.layers[0], "q_proj"):
        q = weights.layers[0].q_proj
        if hasattr(q, "nbytes"):  # INT4
            total_bytes = weights.embed_tokens.nbytes()
            total_bytes += getattr(weights.embed_tokens_per_layer, "nbytes", lambda: 0)()
            total_bytes += weights.per_layer_model_projection.nbytes()
            params = total_bytes * (8 / 4.5)  # packed 4-bit + fp16 scales, approximate
            for layer in weights.layers:
                for name in (
                    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                    "up_proj", "down_proj", "per_layer_input_gate", "per_layer_projection",
                ):
                    w = getattr(layer, name)
                    total_bytes += w.nbytes()
                    params += w.packed.numel() * 2
            return {
                "quantized": "int4",
                "packed_gib": round(total_bytes / 2**30, 3),
                "approx_params_b": round(params / 1e9, 3),
            }
    total_bytes = sum(t.numel() * t.element_size() for t in _iter_tensors(weights))
    return {
        "quantized": "dense",
        "dense_gib": round(total_bytes / 2**30, 3),
        "approx_params_b": round(sum(t.numel() for t in _iter_tensors(weights)) / 1e9, 3),
    }


def _iter_tensors(weights):
    for name in ("embed_tokens", "final_norm", "embed_tokens_per_layer",
                 "per_layer_model_projection", "per_layer_projection_norm"):
        value = getattr(weights, name, None)
        if isinstance(value, torch.Tensor):
            yield value
    for layer in weights.layers:
        for value in vars(layer).values():
            if isinstance(value, torch.Tensor):
                yield value


def build_runner(
    *,
    model_path: Path,
    model_config,
    weights,
    device: str,
    dtype: torch.dtype,
    max_tokens: int,
    history: int,
):
    """Build the Stage5Engine serving path around the given weights."""
    head_dim = max(spec.head_dim for spec in model_config.layer_specs)
    engine_config = EngineConfig.create(
        device=device,
        dtype="fp16" if dtype == torch.float16 else "bf16",
        block_size=128,
        num_gpu_blocks=max(8, (max_tokens + history + 127) // 128 + 2),
        max_batch_size=1,
        max_tokens_per_step=1,
        max_prefill_tokens_per_step=max_tokens,
        max_decode_requests_per_step=1,
        prefill_chunk_size=max_tokens,
        default_temperature=0.0,
        default_top_k=20,
        default_top_p=0.95,
        model_param_count=0,
        num_hidden_layers=model_config.num_hidden_layers,
        num_kv_heads=model_config.num_key_value_heads,
        head_dim=head_dim,
    )
    engine_config.eos_token_id = (1, 106, 50)
    cache_manager = Gemma4PagedKVCacheManager(engine_config, model_config)
    runner = Gemma4EngineModelRunner(engine_config, model_config, weights, cache_manager)
    return engine_config, cache_manager, runner


def bench_direct(runner: Gemma4EagerTextRunner, prompt_ids, max_new_tokens: int) -> dict:
    device = runner.w.embed_tokens.device
    runner.reset_cache()
    sync(device)
    started = time.perf_counter()
    with torch.inference_mode():
        logits = runner.prefill(torch.tensor(prompt_ids, device=device, dtype=torch.long))
        next_id = int(logits.argmax())
        sync(device)
        prefill_done = time.perf_counter()
        generated = [next_id] if max_new_tokens > 0 else []
        while len(generated) < max_new_tokens:
            logits = runner.decode_step(next_id)
            next_id = int(logits.argmax())
            generated.append(next_id)
        sync(device)
    finished = time.perf_counter()
    decode_s = finished - prefill_done
    decode_tokens = max(0, len(generated) - 1)
    return {
        "path": "direct_eager_runner",
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(generated),
        "prefill_s": round(prefill_done - started, 3),
        "decode_s": round(decode_s, 3),
        "decode_tokens": decode_tokens,
        "decode_tok_s": round(decode_tokens / decode_s, 3) if decode_s else 0.0,
        "sample": generated[:12],
    }


def bench_engine(
    *,
    model_config,
    weights,
    model_path: Path,
    device: str,
    dtype: torch.dtype,
    prompt_ids,
    max_new_tokens: int,
) -> dict:
    engine_config, cache_manager, runner = build_runner(
        model_path=model_path,
        model_config=model_config,
        weights=weights,
        device=device,
        dtype=dtype,
        max_tokens=max(len(prompt_ids), 1),
        history=max_new_tokens,
    )
    engine = Stage5Engine(
        engine_config=engine_config,
        model_config=model_config,
        model_runner=runner,
        tokenizer=JsonTokenizer(model_path),
    )
    request_id = engine.add_request(
        prompt_token_ids=prompt_ids,
        sampling_params=SamplingParams(temperature=0.0, top_k=20, top_p=0.95),
        max_new_tokens=max_new_tokens,
    )
    dev = torch.device(device)
    sync(dev)
    started = time.perf_counter()
    prefill_done = None
    while engine.has_pending_work():
        result = engine.step()
        sync(dev)
        now = time.perf_counter()
        if prefill_done is None and any(
            event.kind == "prefill_sampled_first_token" for event in result.events
        ):
            prefill_done = now
    finished = time.perf_counter()
    info = engine.debug_request(request_id)

    prefill_s = (prefill_done or finished) - started
    decode_s = max(0.0, finished - (prefill_done or started))
    decode_tokens = max(0, info["generated"] - 1)
    return {
        "path": "stage5_engine",
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": info["generated"],
        "finish_reason": info.get("finish_reason"),
        "prefill_s": round(prefill_s, 3),
        "decode_s": round(decode_s, 3),
        "decode_tokens": decode_tokens,
        "decode_tok_s": round(decode_tokens / decode_s, 3) if decode_s else 0.0,
        "end_to_end_tok_s": round(
            info["generated"] / (finished - started), 3
        ) if finished > started else 0.0,
        "kv_cache": cache_manager.debug_global_state()["cache_groups"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Gemma4 checkpoint directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--mode", default="both", choices=["direct", "engine", "both"])
    parser.add_argument("--int4", action="store_true", help="quantize weights to INT4")
    parser.add_argument(
        "--int4-backend",
        default="auto",
        choices=["auto", "reference"],
        help="auto uses the fused kernel when importable; reference forces the PyTorch path",
    )
    parser.add_argument("--prompt", default="请简要介绍新疆地区。")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=0, help="unmeasured decode steps")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark targets CUDA; run it in the 3060 workspace")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    from miniservellm.runtime.nn_ops import gemma4_decode_attention_available

    props = torch.cuda.get_device_properties(0)
    free, total = gpu_memory_gib()
    print(
        json.dumps(
            {
                "device": props.name,
                "capability": f"{props.major}.{props.minor}",
                "vram_free_gib": round(free, 2),
                "vram_total_gib": round(total, 2),
                "dtype": str(dtype),
                "int4": args.int4,
                "int4_backend": args.int4_backend,
                "gemma4_attention_kernel": gemma4_decode_attention_available(),
            }
        )
    )

    if args.int4 and args.int4_backend == "reference":
        import mini_llm_kernels.kernels.int4_matmul as int4_mod

        int4_mod._HAS_INT4_CUDA = False  # force the correct PyTorch unpack path
        print('{"int4_matmul_backend": "pytorch_reference"}')
    elif args.int4:
        # Instrument the entry point so the report can state which backend the
        # model actually executed, not which one was expected.
        import mini_llm_kernels.kernels.int4_matmul as int4_mod

        stats = {"fused": 0, "reference": 0}
        fused_fn = int4_mod.int4_dequant_matmul

        def counting_fused(*a, **kw):
            if int4_mod._HAS_INT4_CUDA and a[0].is_cuda:
                stats["fused"] += 1
            else:
                stats["reference"] += 1
            return fused_fn(*a, **kw)

        int4_mod.int4_dequant_matmul = counting_fused
        globals()["_INT4_STATS"] = stats
        print('{"int4_matmul_backend": "fused_kernel_when_cuda"}')

    weights, model_config, load_s = load_weights(
        model_path, args.device, dtype, args.int4, args.int4_backend
    )
    free_after, _ = gpu_memory_gib()
    print(
        json.dumps(
            {
                "load_s": round(load_s, 2),
                "vram_free_after_load_gib": round(free_after, 2),
                "weights": weight_report(weights),
            }
        )
    )

    tokenizer = JsonTokenizer(model_path)
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    print(json.dumps({"prompt_tokens": len(prompt_ids), "prompt": args.prompt}))

    results = []
    if args.mode in ("direct", "both"):
        direct = Gemma4EagerTextRunner(model_config, weights)
        if args.warmup:
            direct.generate_cached(prompt_ids, args.warmup, eos_token_ids=())
            direct.reset_cache()
        results.append(bench_direct(direct, prompt_ids, args.max_new_tokens))

    if args.mode in ("engine", "both"):
        if args.warmup:
            bench_engine(
                model_config=model_config, weights=weights, model_path=model_path,
                device=args.device, dtype=dtype, prompt_ids=prompt_ids,
                max_new_tokens=args.warmup,
            )
        results.append(
            bench_engine(
                model_config=model_config,
                weights=weights,
                model_path=model_path,
                device=args.device,
                dtype=dtype,
                prompt_ids=prompt_ids,
                max_new_tokens=args.max_new_tokens,
            )
        )

    for result in results:
        print(json.dumps(result, ensure_ascii=False))

    stats = globals().get("_INT4_STATS")
    if stats:
        print(json.dumps({"int4_matmul_calls": stats}))


if __name__ == "__main__":
    main()
