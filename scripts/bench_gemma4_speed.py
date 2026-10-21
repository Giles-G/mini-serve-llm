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
import os
import subprocess
import sys
import time
from contextlib import nullcontext
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


def render_chat_prompt(model_path: Path, prompt: str) -> str | None:
    """Render one user turn with the checkpoint's chat template.

    Gemma4 checkpoints ship the template as a standalone
    ``chat_template.jinja`` (``tokenizer_config.json`` carries no
    ``chat_template`` key), so the template text has to be handed to
    ``apply_chat_template`` explicitly. An instruct checkpoint that is fed a
    raw, un-templated prompt answers ``<end_of_turn>`` immediately, which would
    make this benchmark report a single-token decode.

    Returns ``None`` when no template can be rendered, so the caller can fall
    back to the raw prompt instead of failing.
    """
    template = None
    config_path = model_path / "tokenizer_config.json"
    if config_path.is_file():
        template = json.loads(config_path.read_text(encoding="utf-8")).get("chat_template")
    if not template:
        jinja_path = model_path / "chat_template.jinja"
        if jinja_path.is_file():
            template = jinja_path.read_text(encoding="utf-8")
    if not template:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    hf_tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    return hf_tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        chat_template=template,
    )


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def gpu_memory_gib() -> tuple[float, float]:
    free, total = torch.cuda.mem_get_info()
    return free / 2**30, total / 2**30


def _cmd(args):
    try:
        return subprocess.run(
            args, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc}"


def runtime_metadata(device: str, dtype: torch.dtype, eos_token_ids) -> dict:
    """Small, reproducible environment snapshot for the JSON report."""
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    free, total = gpu_memory_gib() if torch.cuda.is_available() else (0.0, 0.0)
    kernels = {}
    try:
        import mini_llm_kernels
        from mini_llm_kernels.kernels.gemma4_attention import cuda_attention_status

        kernels = {
            "path": getattr(mini_llm_kernels, "__file__", None),
            "cuda_ops": bool(getattr(mini_llm_kernels, "_HAS_CUDA_OPS", False)),
            "attention": cuda_attention_status(),
        }
    except Exception as exc:  # noqa: BLE001
        kernels = {"error": str(exc)}
    return {
        "device": props.name if props else None,
        "capability": f"{props.major}.{props.minor}" if props else None,
        "vram_free_gib": round(free, 2),
        "vram_total_gib": round(total, 2),
        "dtype": str(dtype),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "driver": _cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "gpu_state": _cmd([
            "nvidia-smi",
            "--query-gpu=clocks.sm,temperature.gpu,power.draw,memory.used",
            "--format=csv,noheader",
        ]),
        "git_head": _cmd(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"]),
        "git_dirty": bool(_cmd([
            "git", "-C", str(Path(__file__).resolve().parents[1]), "status", "--porcelain"
        ])),
        "eos_token_ids": list(eos_token_ids),
        "int4_backend_env": os.environ.get("MINI_LLM_INT4_IMPL", "pair_vec(default)"),
        "kernels": kernels,
    }


def _event_self_cuda_us(event) -> float:
    """Self CUDA time across PyTorch profiler attribute versions."""
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        value = getattr(event, attr, None)
        if value:
            return float(value)
    return 0.0


def profile_categories(key_averages, actual_steps: int) -> list[dict]:
    """Group self times by category; parent labels are reported separately.

    Only ``self_*`` times are summed. ``engine::``/``int4::`` record_function
    labels are not added to the raw-kernel categories, because their CUDA time
    can duplicate the child kernels.
    """
    groups = {}

    def add(category, event, include):
        row = groups.setdefault(
            category, {"category": category, "self_cpu_us": 0.0,
                       "self_cuda_us": 0.0, "calls": 0, "include_in_kernel_sum": include}
        )
        row["self_cpu_us"] += float(getattr(event, "self_cpu_time_total", 0.0))
        row["self_cuda_us"] += _event_self_cuda_us(event)
        row["calls"] += int(getattr(event, "count", 0))

    for event in key_averages:
        name = event.key
        if name.startswith("int4::"):
            category, include = "int4_label", False
        elif "int4_dequant_matmul" in name:
            category, include = "int4_kernel", True
        elif name.startswith(("engine::", "direct::")):
            category, include = "engine_label", False
        elif "gemma4_attention" in name:
            category, include = "attention_kernel", True
        elif any(t in name for t in ("rms_norm", "fused_add_rms_norm")):
            category, include = "norm", True
        elif any(t in name for t in ("bmm", "softmax", "matmul")):
            category, include = "attention_math", True
        elif any(t in name for t in ("copy_", "Memcpy", "index_select", "cat", "to_copy")):
            category, include = "copy", True
        elif any(t in name for t in ("argmax", "topk", "multinomial")):
            category, include = "sampler", True
        elif any(t in name for t in ("mul", "add", "gelu", "silu", "addcmul",
                                     "neg", "pow", "mean", "rsqrt", "elementwise")):
            category, include = "elementwise", True
        else:
            category, include = "other", True
        add(category, event, include)
    steps = max(1, actual_steps)
    rows = []
    for row in groups.values():
        row["self_cpu_ms_per_step"] = round(row["self_cpu_us"] / 1000 / steps, 3)
        row["self_cuda_ms_per_step"] = round(row["self_cuda_us"] / 1000 / steps, 3)
        rows.append(row)
    return sorted(rows, key=lambda r: r["self_cuda_us"], reverse=True)


def profile_hotspots(key_averages, actual_steps: int, limit: int = 20) -> list[dict]:
    steps = max(1, actual_steps)
    rows = []
    for event in key_averages:
        name = event.key
        calls = int(getattr(event, "count", 0))
        cuda = _event_self_cuda_us(event)
        cpu = float(getattr(event, "self_cpu_time_total", 0.0) or 0.0)
        rows.append({
            "name": name,
            "kind": "label" if name.startswith(("int4::", "engine::", "direct::")) else "kernel",
            "calls": calls,
            "self_cuda_ms": round(cuda / 1000, 3),
            "self_cpu_ms": round(cpu / 1000, 3),
            "cuda_us_per_call": round(cuda / calls, 3) if calls else 0.0,
            "cuda_ms_per_step": round(cuda / 1000 / steps, 3),
        })
    return sorted(rows, key=lambda r: r["self_cuda_ms"], reverse=True)[:limit]


def profile_event_shapes(events_or_profiler, limit: int = 40) -> list[dict]:
    """Per-event input shapes for the compact profile summary.

    Key averages drop ``input_shapes``, so this reads the raw profiler event
    stream and aggregates identical (name, shape) pairs. It is only populated
    when the profiler ran with ``record_shapes=True``.
    """
    if hasattr(events_or_profiler, "events"):
        events = events_or_profiler.events()
    else:
        events = events_or_profiler
    grouped = {}
    for event in events:
        shapes = getattr(event, "input_shapes", None)
        if not shapes:
            continue
        shape_key = tuple(tuple(shape) for shape in shapes)
        key = (event.key, shape_key)
        row = grouped.setdefault(key, {
            "name": event.key,
            "calls": 0,
            "input_shapes": [list(shape) for shape in shapes],
            "self_cuda_us": 0.0,
            "self_cpu_us": 0.0,
        })
        row["calls"] += 1
        row["self_cuda_us"] += _event_self_cuda_us(event)
        row["self_cpu_us"] += float(
            getattr(event, "self_cpu_time_total", 0.0) or 0.0
        )
    rows = [
        {
            "name": row["name"],
            "calls": row["calls"],
            "input_shapes": row["input_shapes"],
            "self_cuda_ms": round(row["self_cuda_us"] / 1000, 3),
            "self_cpu_ms": round(row["self_cpu_us"] / 1000, 3),
        }
        for row in grouped.values()
    ]
    return sorted(rows, key=lambda row: row["self_cuda_ms"], reverse=True)[:limit]


def compare_profiles(engine: dict, direct: dict) -> dict:
    """Side-by-side per-step attribution for engine vs direct eager.

    The two profiles come from separate runs, each started before its second
    decode step. Values are self times per profiled step; they are intended
    for attribution comparisons, not as wall-clock claims.
    """
    def by_category(profile):
        return {
            row["category"]: row for row in profile.get("categories", [])
            if row.get("include_in_kernel_sum", True)
        }

    engine_categories = by_category(engine)
    direct_categories = by_category(direct)
    categories = []
    for name in sorted(set(engine_categories) | set(direct_categories)):
        e = engine_categories.get(name, {})
        d = direct_categories.get(name, {})
        e_ms = float(e.get("self_cuda_ms_per_step", 0.0))
        d_ms = float(d.get("self_cuda_ms_per_step", 0.0))
        categories.append({
            "category": name,
            "engine_cuda_ms_per_step": e_ms,
            "direct_cuda_ms_per_step": d_ms,
            "delta_engine_minus_direct_ms": round(e_ms - d_ms, 3),
            "engine_over_direct": round(e_ms / d_ms, 3) if d_ms else None,
        })

    def by_name(profile):
        return {
            row["name"]: row for row in profile.get("hotspots", [])
            if row.get("kind") != "label"
        }

    engine_hotspots = by_name(engine)
    direct_hotspots = by_name(direct)
    hotspots = []
    for name in sorted(set(engine_hotspots) | set(direct_hotspots)):
        e = engine_hotspots.get(name, {})
        d = direct_hotspots.get(name, {})
        e_ms = float(e.get("cuda_ms_per_step", 0.0))
        d_ms = float(d.get("cuda_ms_per_step", 0.0))
        hotspots.append({
            "name": name,
            "engine_cuda_ms_per_step": e_ms,
            "direct_cuda_ms_per_step": d_ms,
            "delta_engine_minus_direct_ms": round(e_ms - d_ms, 3),
        })
    hotspots.sort(key=lambda row: max(
        row["engine_cuda_ms_per_step"], row["direct_cuda_ms_per_step"]
    ), reverse=True)
    return {
        "note": (
            "engine and direct profiled in separate runs; per-step self times "
            "are for attribution, not wall-clock comparison"
        ),
        "engine_steps": engine.get("actual_decode_steps"),
        "direct_steps": direct.get("actual_decode_steps"),
        "categories": categories,
        "hotspots": hotspots[:25],
    }


def _profile_int4_labels(weights):
    from miniservellm.runtime.gemma4_quant import Int4Weight

    labels = {}
    for name in ("embed_tokens", "lm_head", "per_layer_model_projection"):
        value = getattr(weights, name, None)
        if isinstance(value, Int4Weight):
            labels[id(value)] = name
    for layer_idx, layer in enumerate(weights.layers):
        for name, value in vars(layer).items():
            if isinstance(value, Int4Weight):
                labels[id(value)] = f"layer{layer_idx}.{name}"
    return Int4Weight, labels


def projection_shapes(weights):
    """Actual checkpoint dimensions, not synthetic benchmark assumptions."""
    shapes = {}
    for name in ("lm_head", "per_layer_model_projection"):
        value = getattr(weights, name, None)
        if value is not None:
            shapes[name] = (
                [value.packed.shape[0] * 2, value.packed.shape[1]]
                if hasattr(value, "packed") else list(value.shape)
            )
    for index, layer in enumerate(weights.layers):
        for name, value in vars(layer).items():
            if hasattr(value, "packed"):
                shapes[f"layer{index}.{name}"] = [
                    value.packed.shape[0] * 2, value.packed.shape[1]
                ]
    return shapes


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
    eos_token_ids: tuple[int, ...] = (1, 106, 50),
    max_batch_size: int = 1,
):
    """Build the Stage5Engine serving path around the given weights."""
    head_dim = max(spec.head_dim for spec in model_config.layer_specs)
    engine_config = EngineConfig.create(
        device=device,
        dtype="fp16" if dtype == torch.float16 else "bf16",
        block_size=128,
        num_gpu_blocks=max(
            8,
            ((max_tokens + history + 127) // 128 + 2) * max_batch_size,
        ),
        max_batch_size=max_batch_size,
        # Admit one full chunk for every resident batch slot. A budget of only
        # one prompt lets the scheduler prefill a single request per step, so
        # the other batch slots arrive later and the first slots keep decoding
        # in between. With a batch-sized budget all fresh prefills can enter
        # the same step and use the cross-request prefill path.
        max_tokens_per_step=max(max_tokens, max_tokens * max_batch_size),
        max_prefill_tokens_per_step=max(max_tokens, max_tokens * max_batch_size),
        max_decode_requests_per_step=max_batch_size,
        prefill_chunk_size=max_tokens,
        default_temperature=0.0,
        default_top_k=20,
        default_top_p=0.95,
        model_param_count=0,
        num_hidden_layers=model_config.num_hidden_layers,
        num_kv_heads=model_config.num_key_value_heads,
        head_dim=head_dim,
    )
    engine_config.eos_token_id = eos_token_ids
    cache_manager = Gemma4PagedKVCacheManager(engine_config, model_config)
    runner = Gemma4EngineModelRunner(engine_config, model_config, weights, cache_manager)
    return engine_config, cache_manager, runner


def bench_direct(
    runner: Gemma4EagerTextRunner,
    prompt_ids,
    max_new_tokens: int,
    tokenizer,
    eos_token_ids: tuple[int, ...] = (1, 106, 50),
    profile_steps: bool = False,
    profiler=None,
) -> dict:
    device = runner.w.embed_tokens.device
    runner.reset_cache()
    sync(device)
    started = time.perf_counter()
    profile_start_context_len = None
    profile_started = False
    decode_steps = 0
    try:
        with torch.inference_mode():
            logits = runner.prefill(torch.tensor(prompt_ids, device=device, dtype=torch.long))
            next_id = int(logits.argmax())
            sync(device)
            prefill_done = time.perf_counter()
            generated = [next_id] if max_new_tokens > 0 else []
            while len(generated) < max_new_tokens and next_id not in eos_token_ids:
                # Mirror bench_engine: profile starts before the second
                # decode step, excluding prefill and the first handoff.
                if (
                    profiler is not None
                    and decode_steps >= 1
                    and not profile_started
                ):
                    sync(device)
                    profile_start_context_len = len(prompt_ids) + decode_steps - 1
                    profiler.start()
                    profile_started = True
                scope = (
                    torch.profiler.record_function("direct::decode")
                    if profile_steps
                    else nullcontext()
                )
                with scope:
                    logits = runner.decode_step(next_id)
                    next_id = int(logits.argmax())
                    sync(device)
                if profile_started:
                    profiler.step()
                decode_steps += 1
                generated.append(next_id)
            sync(device)
    finally:
        if profile_started:
            profiler.stop()
    if profiler is not None and not profile_started:
        raise RuntimeError(
            "Direct request ended before decode-only profiling could start; "
            "increase token budget or check EOS"
        )
    finished = time.perf_counter()
    decode_s = finished - prefill_done
    decode_tokens = max(0, len(generated) - 1)
    decode_started = time.perf_counter()
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    generated_raw = tokenizer.decode(generated, skip_special_tokens=False)
    text_decode_s = time.perf_counter() - decode_started
    post_started = time.perf_counter()
    visible_empty = not generated_text.strip()
    text_postprocess_s = time.perf_counter() - post_started
    finish_reason = (
        "EOS" if generated and generated[-1] in eos_token_ids
        else "MAX_NEW_TOKENS"
    )
    return {
        "path": "direct_eager_runner",
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(generated),
        "finish_reason": finish_reason,
        "prefill_s": round(prefill_done - started, 3),
        "decode_s": round(decode_s, 3),
        "decode_tokens": decode_tokens,
        "decode_tok_s": round(decode_tokens / decode_s, 3) if decode_s else 0.0,
        "end_to_end_tok_s": round(
            len(generated) / (finished - started), 3
        ) if finished > started else 0.0,
        "sample": generated[:12],
        "generated_token_ids": generated,
        "generated_text": generated_text,
        "generated_text_raw": generated_raw,
        "visible_text_empty": visible_empty,
        "generation_status": "empty_text" if visible_empty else "ok",
        "text_decode_s": round(text_decode_s, 6),
        "text_postprocess_s": round(text_postprocess_s, 6),
        "profile_start_context_len": profile_start_context_len,
        "profile_expected_steps": (
            max(0, max_new_tokens - 2) if profiler is not None else 0
        ),
        "profile_actual_steps": (
            max(0, decode_steps - 1) if profiler is not None else 0
        ),
        "profile_steps_match": (
            max(0, max_new_tokens - 2) == max(0, decode_steps - 1)
            if profiler is not None else None
        ),
        "profile_decode_steps": (
            max(0, decode_steps - 1) if profiler is not None else 0
        ),
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
    warmup_tokens: int = 0,
    profile_steps: bool = False,
    profiler=None,
    eos_token_ids: tuple[int, ...] = (1, 106, 50),
) -> dict:
    engine_config, cache_manager, runner = build_runner(
        model_path=model_path,
        model_config=model_config,
        weights=weights,
        device=device,
        dtype=dtype,
        max_tokens=max(len(prompt_ids), 1),
        history=max_new_tokens,
        eos_token_ids=eos_token_ids,
    )
    engine = Stage5Engine(
        engine_config=engine_config,
        model_config=model_config,
        model_runner=runner,
        tokenizer=JsonTokenizer(model_path),
    )
    dev = torch.device(device)
    if warmup_tokens:
        # Run the warm-up on the same engine as the measured request. For the
        # CUDA-graph path this is what captures the graph before timing; a
        # fresh engine would pay capture latency inside the measured decode.
        engine.add_request(
            prompt_token_ids=prompt_ids,
            sampling_params=SamplingParams(temperature=0.0, top_k=20, top_p=0.95),
            max_new_tokens=warmup_tokens,
        )
        while engine.has_pending_work():
            engine.step()
            sync(dev)
    request_id = engine.add_request(
        prompt_token_ids=prompt_ids,
        sampling_params=SamplingParams(temperature=0.0, top_k=20, top_p=0.95),
        max_new_tokens=max_new_tokens,
    )
    sync(dev)
    started = time.perf_counter()
    prefill_done = None
    profile_started = False
    profile_start_context_len = None
    decode_steps = 0
    try:
        while engine.has_pending_work():
            # Start immediately before the second decode step. This excludes
            # both prefill and the first decode handoff from the profile.
            if (
                profiler is not None
                and prefill_done is not None
                and decode_steps >= 1
                and not profile_started
            ):
                sync(dev)
                start_info = engine.debug_request(request_id)
                # At this point the prompt and the first generated token have
                # been written; ``generated`` includes the token sampled by the
                # first decode but not yet processed.
                profile_start_context_len = (
                    int(start_info.get("prompt_total", len(prompt_ids)))
                    + max(0, int(start_info.get("generated", 1)) - 1)
                )
                profiler.start()
                profile_started = True
            scope = (
                torch.profiler.record_function(
                    "engine::prefill" if prefill_done is None else "engine::decode"
                )
                if profile_steps
                else nullcontext()
            )
            with scope:
                result = engine.step()
                sync(dev)
            if profile_started:
                profiler.step()
            if prefill_done is not None:
                decode_steps += 1
            now = time.perf_counter()
            if prefill_done is None and any(
                event.kind == "prefill_sampled_first_token" for event in result.events
            ):
                prefill_done = now
        finished = time.perf_counter()
    finally:
        if profile_started:
            profiler.stop()
    if profiler is not None and not profile_started:
        raise RuntimeError(
            "Request ended before decode-only profiling could start; "
            "increase token budget or check EOS"
        )
    info = engine.debug_request(request_id)
    generated_token_ids = list(info["generated_token_ids"])
    decode_started = time.perf_counter()
    generated_text = engine.tokenizer.decode(
        generated_token_ids, skip_special_tokens=True
    )
    generated_raw = engine.tokenizer.decode(
        generated_token_ids, skip_special_tokens=False
    )
    text_decode_s = time.perf_counter() - decode_started
    post_started = time.perf_counter()
    visible_empty = not generated_text.strip()
    text_postprocess_s = time.perf_counter() - post_started

    prefill_s = (prefill_done or finished) - started
    decode_s = max(0.0, finished - (prefill_done or started))
    decode_tokens = max(0, info["generated"] - 1)
    actual_profile_steps = max(0, decode_steps - 1) if profiler is not None else 0
    expected_profile_steps = max(0, max_new_tokens - 2) if profiler is not None else 0
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
        "generated_token_ids": generated_token_ids,
        "generated_text": generated_text,
        "generated_text_raw": generated_raw,
        "visible_text_empty": visible_empty,
        "generation_status": "empty_text" if visible_empty else "ok",
        "text_decode_s": round(text_decode_s, 6),
        "text_postprocess_s": round(text_postprocess_s, 6),
        "runner_backend": {
            "graph_mode": bool(getattr(runner, "_graph_mode", False)),
            "graph_captured": getattr(runner, "_decode_graph", None) is not None,
            "paged_attention": bool(getattr(runner, "_paged_attention", False)),
        },
        "profile_start_context_len": profile_start_context_len,
        "profile_expected_steps": expected_profile_steps,
        "profile_actual_steps": actual_profile_steps,
        "profile_steps_match": (
            expected_profile_steps == actual_profile_steps
            if profiler is not None else None
        ),
        "profile_decode_steps": actual_profile_steps,
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
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="feed --prompt raw instead of rendering the checkpoint chat template",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=0, help="unmeasured decode steps")
    parser.add_argument(
        "--eos-token-ids",
        type=int,
        nargs="+",
        default=[1, 106, 50],
        help="token IDs that finish generation (default: 1 106 50)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        metavar="RESULTS_JSON",
        help="save the actual benchmark results as JSON",
    )
    parser.add_argument(
        "--profile", type=Path, metavar="SUMMARY_JSON",
        help="run a separate short engine profile after normal benchmark",
    )
    parser.add_argument(
        "--profile-trace", type=Path, metavar="TRACE_JSON",
        help="also export a detailed Chrome trace; omitted by default",
    )
    parser.add_argument(
        "--profile-summary", type=Path, metavar="SUMMARY_JSON",
        help="write the compact profiler summary as JSON",
    )
    parser.add_argument(
        "--profile-mode", default="auto",
        choices=["auto", "both", "direct", "engine"],
        help="which backend to profile (auto: both when --mode both)",
    )
    parser.add_argument(
        "--profile-tokens", type=int, default=16,
        help="generated tokens in the separate profiling run",
    )
    parser.add_argument(
        "--profile-shapes", action="store_true",
        help="record tensor shapes in the profiler (larger, slower summaries)",
    )
    parser.add_argument(
        "--allow-empty-generation", action="store_true",
        help="do not exit non-zero when generated tokens decode to empty text",
    )
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.warmup < 0 or args.profile_tokens < 3:
        parser.error("max-new-tokens >= 1, warmup >= 0, profile-tokens >= 3 required")
    if (args.profile or args.profile_trace or args.profile_summary) and args.mode == "direct":
        parser.error("profiling requires engine mode")
    # Backward-compatible: the old --profile path now receives the compact
    # summary.  A large Chrome trace requires the explicit --profile-trace flag.
    if args.profile and not args.profile_trace and not args.profile_summary:
        args.profile_summary = args.profile
        args.profile = None

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
    weights_report = weight_report(weights)
    shapes = projection_shapes(weights)
    runtime = runtime_metadata(args.device, dtype, tuple(args.eos_token_ids))
    runtime.update({
        "int4": args.int4,
        "int4_backend": args.int4_backend,
        "load_s": round(load_s, 2),
        "weights": weights_report,
        "projection_shapes": shapes,
        "attention_kernel_available": gemma4_decode_attention_available(),
    })
    free_after, _ = gpu_memory_gib()
    print(
        json.dumps(
            {
                "load_s": round(load_s, 2),
                "vram_free_after_load_gib": round(free_after, 2),
                "weights": weights_report,
                "projection_shapes": shapes,
            }
        )
    )

    tokenizer = JsonTokenizer(model_path)
    # Instruct checkpoints must be prompted through their chat template: the
    # raw prompt carries no turn markers, so the model emits <end_of_turn> as
    # its first token and the decode numbers below measure nothing.
    chat_text = None if args.no_chat_template else render_chat_prompt(model_path, args.prompt)
    if chat_text is None and not args.no_chat_template:
        print(json.dumps({"chat_template": "unavailable, falling back to raw prompt"}))
    tokenize_started = time.perf_counter()
    prompt_ids = tokenizer.encode(
        chat_text if chat_text is not None else args.prompt,
        add_special_tokens=False,
    )
    text_tokenize_s = time.perf_counter() - tokenize_started
    print(
        json.dumps(
            {
                "prompt_tokens": len(prompt_ids),
                "chat_template": chat_text is not None,
                "prompt": args.prompt,
            },
            ensure_ascii=False,
        )
    )

    eos_token_ids = tuple(args.eos_token_ids)
    results = []
    if args.mode in ("direct", "both"):
        direct = Gemma4EagerTextRunner(model_config, weights)
        try:
            if args.warmup:
                direct.generate_cached(
                    prompt_ids, args.warmup, eos_token_ids=eos_token_ids
                )
                direct.reset_cache()
            results.append(
                bench_direct(
                    direct,
                    prompt_ids,
                    args.max_new_tokens,
                    tokenizer,
                    eos_token_ids=eos_token_ids,
                )
            )
        finally:
            direct.reset_cache()
            del direct
            if args.device == "cuda":
                torch.cuda.empty_cache()

    if args.mode in ("engine", "both"):
        results.append(
            bench_engine(
                model_config=model_config,
                weights=weights,
                model_path=model_path,
                device=args.device,
                dtype=dtype,
                prompt_ids=prompt_ids,
                max_new_tokens=args.max_new_tokens,
                warmup_tokens=args.warmup,
                eos_token_ids=eos_token_ids,
            )
        )

    empty_results = []
    for result in results:
        printable = dict(result)
        generated_text = printable.pop("generated_text", "")
        print(json.dumps(printable, ensure_ascii=False))
        print_started = time.perf_counter()
        print(f"=== GEMMA4 GENERATION BEGIN ({result['path']}) ===")
        print(f"prompt: {args.prompt}")
        print(f"finish_reason: {result.get('finish_reason', 'N/A')}")
        print(f"generated_tokens: {result['generated_tokens']}")
        if generated_text:
            print(generated_text)
        else:
            empty_results.append(result["path"])
            print("[WARNING] decoded generation is empty")
            print(f"generated_token_ids: {result.get('generated_token_ids', [])}")
            print(
                "raw decoded generation: "
                f"{result.get('generated_text_raw', '')!r}"
            )
        print(f"=== GEMMA4 GENERATION END ({result['path']}) ===")
        result["text_print_s"] = round(time.perf_counter() - print_started, 6)

    if empty_results and not args.allow_empty_generation:
        print(
            json.dumps({
                "generation_failure": "visible text empty",
                "paths": empty_results,
            }),
            file=sys.stderr,
        )
        sys.exit(1)

    stats = globals().get("_INT4_STATS")
    if stats:
        print(json.dumps({"int4_matmul_calls": stats}))

    profile_info = None
    if args.profile or args.profile_trace or args.profile_summary:
        profile_int4_weight, labels = _profile_int4_labels(weights)
        original_matmul = profile_int4_weight.matmul

        def profiled_matmul(weight, x):
            label = labels.get(id(weight), "other")
            with torch.profiler.record_function(f"int4::{label}"):
                return original_matmul(weight, x)

        if args.profile_mode == "auto":
            profile_targets = (
                ["engine", "direct"] if args.mode == "both" else [args.mode]
            )
        elif args.profile_mode == "both":
            profile_targets = ["engine", "direct"]
        else:
            profile_targets = [args.profile_mode]
        if args.profile_trace and len(profile_targets) > 1:
            parser.error(
                "--profile-trace supports one backend; use --profile-mode "
                "direct|engine or omit the trace"
            )

        profile_int4_weight.matmul = profiled_matmul
        collected = {}
        try:
            for target in profile_targets:
                profiler = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=args.profile_shapes,
                    profile_memory=False,
                    with_stack=False,
                )
                profile_result = None
                if target == "engine":
                    profile_result = bench_engine(
                        model_config=model_config,
                        weights=weights,
                        model_path=model_path,
                        device=args.device,
                        dtype=dtype,
                        prompt_ids=prompt_ids,
                        max_new_tokens=args.profile_tokens,
                        profile_steps=True,
                        profiler=profiler,
                        eos_token_ids=eos_token_ids,
                    )
                else:
                    direct = Gemma4EagerTextRunner(model_config, weights)
                    try:
                        profile_result = bench_direct(
                            direct,
                            prompt_ids,
                            args.profile_tokens,
                            tokenizer,
                            eos_token_ids=eos_token_ids,
                            profile_steps=True,
                            profiler=profiler,
                        )
                    finally:
                        direct.reset_cache()
                        del direct
                        if args.device == "cuda":
                            torch.cuda.empty_cache()

                summary = profiler.key_averages()
                print(f"=== profile backend: {target} ===")
                print(summary.table(sort_by="self_cuda_time_total", row_limit=30))
                print(summary.table(sort_by="self_cpu_time_total", row_limit=20))
                actual_steps = profile_result["profile_decode_steps"]
                info = {
                    "backend": target,
                    "profile_tokens": args.profile_tokens,
                    "start_context_len": profile_result["profile_start_context_len"],
                    "expected_decode_steps": profile_result["profile_expected_steps"],
                    "actual_decode_steps": actual_steps,
                    "steps_match": profile_result["profile_steps_match"],
                    "record_shapes": bool(args.profile_shapes),
                    "categories": profile_categories(summary, actual_steps),
                    "hotspots": profile_hotspots(summary, actual_steps, limit=25),
                    "parent_label_note": (
                        "categories sum only self_* times; int4::/engine::/"
                        "direct:: labels are reported separately and not added "
                        "to raw kernel sums"
                    ),
                    "cuda_summary": summary.table(
                        sort_by="self_cuda_time_total", row_limit=30
                    ),
                    "cpu_summary": summary.table(
                        sort_by="self_cpu_time_total", row_limit=20
                    ),
                }
                if args.profile_shapes:
                    info["event_shapes"] = profile_event_shapes(
                        profiler, limit=40
                    )
                    info["event_shapes_note"] = (
                        "record_shapes stores input shapes on CPU aten op "
                        "events; CUDA kernel events do not carry shapes. Use "
                        "metadata.projection_shapes for the real INT4 shapes."
                    )
                if not profile_result["profile_steps_match"]:
                    info["warning"] = (
                        "profile collected fewer/more steps than requested "
                        "(early EOS or token budget); decode-only attribution "
                        "is based on the actual count"
                    )
                    print("[WARNING]", info["warning"])
                if args.profile_trace:
                    args.profile_trace.parent.mkdir(parents=True, exist_ok=True)
                    profiler.export_chrome_trace(str(args.profile_trace))
                    info["profile_trace"] = str(args.profile_trace.resolve())
                collected[target] = info

            primary = collected.get("engine") or collected["direct"]
            profile_info = dict(primary)
            profile_info["backends"] = list(collected)
            if "direct" in collected and "engine" in collected:
                profile_info["direct"] = collected["direct"]
                profile_info["comparison"] = compare_profiles(
                    collected["engine"], collected["direct"]
                )
            if args.profile_summary:
                args.profile_summary.parent.mkdir(parents=True, exist_ok=True)
                args.profile_summary.write_text(
                    json.dumps(profile_info, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                profile_info["profile_summary"] = str(args.profile_summary.resolve())
            print(json.dumps(profile_info, ensure_ascii=False))
        finally:
            profile_int4_weight.matmul = original_matmul

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(runtime)
        metadata.update({
            "prompt": args.prompt,
            "prompt_tokens": len(prompt_ids),
            "text_tokenize_s": round(text_tokenize_s, 6),
            "chat_template": chat_text is not None,
            "eos_token_ids": list(eos_token_ids),
            "int4_matmul_calls": stats,
            "results_path": str(args.output_json.resolve()),
        })
        payload = {"metadata": metadata, "results": results}
        if profile_info is not None:
            payload["profile"] = profile_info
        args.output_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
