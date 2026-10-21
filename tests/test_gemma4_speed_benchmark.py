import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench_gemma4_speed.py"
SPEC = importlib.util.spec_from_file_location("bench_gemma4_speed", SCRIPT)
BENCH = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BENCH)


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        if skip_special_tokens:
            return "".join(str(token) for token in ids if token != 1)
        return "<" + ",".join(str(token) for token in ids) + ">"


class FakeDirectRunner:
    class Weights:
        embed_tokens = torch.empty(0)

    def __init__(self, logits):
        self.w = self.Weights()
        self.logits = iter(logits)
        self.decode_calls = 0

    def reset_cache(self):
        pass

    def prefill(self, prompt):
        return next(self.logits)

    def decode_step(self, token_id):
        self.decode_calls += 1
        return next(self.logits)


@dataclass
class FakeEvent:
    kind: str


@dataclass
class FakeStep:
    events: list


class FakeProfiler:
    def __init__(self, fail_step=False):
        self.started = 0
        self.steps = 0
        self.stopped = 0
        self.fail_step = fail_step

    def start(self):
        self.started += 1

    def step(self):
        self.steps += 1
        if self.fail_step:
            raise RuntimeError("fake profiler failure")

    def stop(self):
        self.stopped += 1


class FakeCache:
    def debug_global_state(self):
        return {"cache_groups": []}


class FakeEngine:
    def __init__(self, steps, tokenizer):
        self.steps = iter(steps)
        self.tokenizer = tokenizer

    def add_request(self, **kwargs):
        return "request-1"

    def has_pending_work(self):
        return True

    def step(self):
        return next(self.steps)

    def debug_request(self, request_id):
        return {
            "generated": 4,
            "finish_reason": "MAX_NEW_TOKENS",
            "generated_token_ids": [10, 11, 12, 13],
        }


def test_bench_direct_returns_all_tokens_and_honors_eos():
    runner = FakeDirectRunner(
        [
            torch.tensor([0.0, 0.0, 4.0]),
            torch.tensor([0.0, 0.0, 5.0]),
            torch.tensor([0.0, 1.0, 0.0]),
        ]
    )

    result = BENCH.bench_direct(
        runner, [7], max_new_tokens=3, tokenizer=FakeTokenizer(), eos_token_ids=(1,)
    )

    assert result["generated_token_ids"] == [2, 2, 1]
    assert result["generated_tokens"] == 3
    assert result["generated_text"] == "22"
    assert result["generated_text_raw"] == "<2,2,1>"
    assert result["finish_reason"] == "EOS"
    assert result["visible_text_empty"] is False
    assert result["generation_status"] == "ok"
    assert result["text_decode_s"] >= 0.0
    assert runner.decode_calls == 2


class EmptyTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return ""


def test_bench_direct_flags_empty_visible_text():
    runner = FakeDirectRunner([torch.tensor([0.0, 0.0, 4.0])])
    result = BENCH.bench_direct(
        runner, [7], max_new_tokens=1, tokenizer=EmptyTokenizer(), eos_token_ids=(1,)
    )
    assert result["generated_tokens"] == 1
    assert result["visible_text_empty"] is True
    assert result["generation_status"] == "empty_text"


def test_bench_direct_profiles_decode_only_with_accurate_count():
    runner = FakeDirectRunner(
        [
            torch.tensor([0.0, 0.0, 4.0]),
            torch.tensor([0.0, 0.0, 5.0]),
            torch.tensor([0.0, 0.0, 6.0]),
            torch.tensor([0.0, 0.0, 7.0]),
        ]
    )
    profiler = FakeProfiler()

    result = BENCH.bench_direct(
        runner, [7], max_new_tokens=4, tokenizer=FakeTokenizer(),
        eos_token_ids=(1,), profile_steps=True, profiler=profiler,
    )

    assert profiler.started == 1
    assert profiler.steps == 2
    assert profiler.stopped == 1
    assert result["profile_decode_steps"] == 2
    assert result["profile_expected_steps"] == 2
    assert result["profile_steps_match"] is True
    assert result["profile_start_context_len"] == 1
    assert result["text_postprocess_s"] >= 0.0


def test_profile_event_shapes_and_backend_comparison():
    events = [
        SimpleNamespace(
            key="void int4_dequant_matmul_pair_vec_kernel",
            count=2,
            self_cpu_time_total=10.0,
            self_device_time_total=800.0,
            input_shapes=[[1, 1536], [768, 1536], [262144, 1]],
        ),
    ]
    shapes = BENCH.profile_event_shapes(SimpleNamespace(events=lambda: events))
    assert shapes[0]["input_shapes"][0] == [1, 1536]
    assert shapes[0]["self_cuda_ms"] == 0.8
    assert shapes[0]["calls"] == 1

    engine = {
        "actual_decode_steps": 2,
        "categories": [{
            "category": "int4_kernel", "self_cuda_ms_per_step": 1.0,
            "include_in_kernel_sum": True,
        }],
        "hotspots": [{
            "name": "int4", "cuda_ms_per_step": 1.0, "kind": "kernel",
        }],
    }
    direct = {
        "actual_decode_steps": 2,
        "categories": [{
            "category": "int4_kernel", "self_cuda_ms_per_step": 1.5,
            "include_in_kernel_sum": True,
        }],
        "hotspots": [{
            "name": "int4", "cuda_ms_per_step": 1.5, "kind": "kernel",
        }],
    }
    comparison = BENCH.compare_profiles(engine, direct)
    assert comparison["categories"][0]["delta_engine_minus_direct_ms"] == -0.5
    assert comparison["hotspots"][0]["engine_cuda_ms_per_step"] == 1.0


def test_profile_categories_keep_labels_separate():
    events = [
        SimpleNamespace(
            key="void int4_dequant_matmul_pair_kernel",
            count=10,
            self_cpu_time_total=100.0,
            self_device_time_total=2000.0,
        ),
        SimpleNamespace(
            key="int4::lm_head",
            count=2,
            self_cpu_time_total=10.0,
            self_device_time_total=500.0,
        ),
        SimpleNamespace(
            key="engine::decode",
            count=2,
            self_cpu_time_total=300.0,
            self_device_time_total=1000.0,
        ),
        SimpleNamespace(
            key="void gemma4_attention_safe_kernel",
            count=4,
            self_cpu_time_total=50.0,
            self_device_time_total=400.0,
        ),
    ]
    rows = {row["category"]: row for row in BENCH.profile_categories(events, 2)}
    assert rows["int4_kernel"]["include_in_kernel_sum"] is True
    assert rows["int4_kernel"]["self_cuda_ms_per_step"] == 1.0
    assert rows["int4_label"]["include_in_kernel_sum"] is False
    assert rows["engine_label"]["include_in_kernel_sum"] is False
    assert rows["attention_kernel"]["self_cuda_ms_per_step"] == 0.2

    hotspots = BENCH.profile_hotspots(events, 2)
    assert hotspots[0]["name"].startswith("void int4_dequant_matmul_pair_kernel")
    assert hotspots[0]["cuda_us_per_call"] == 200.0
    assert hotspots[0]["cuda_ms_per_step"] == 1.0


def _patch_engine(monkeypatch, steps, tokenizer):
    monkeypatch.setattr(
        BENCH,
        "build_runner",
        lambda **kwargs: (object(), FakeCache(), object()),
    )
    monkeypatch.setattr(
        BENCH,
        "Stage5Engine",
        lambda **kwargs: FakeEngine(steps, tokenizer),
    )
    monkeypatch.setattr(BENCH, "JsonTokenizer", lambda path: tokenizer)
    monkeypatch.setattr(BENCH, "sync", lambda device: None)


def test_bench_engine_profiles_decode_only_with_accurate_count(monkeypatch):
    tokenizer = FakeTokenizer()
    steps = [
        FakeStep([FakeEvent("prefill_sampled_first_token")]),
        FakeStep([FakeEvent("decode_token")]),
        FakeStep([FakeEvent("decode_token")]),
        FakeStep([FakeEvent("decode_token")]),
    ]
    _patch_engine(monkeypatch, steps, tokenizer)
    profiler = FakeProfiler()
    pending = iter([True, True, True, True, False])
    monkeypatch.setattr(FakeEngine, "has_pending_work", lambda self: next(pending))

    result = BENCH.bench_engine(
        model_config=object(),
        weights=object(),
        model_path=Path("."),
        device="cpu",
        dtype=torch.float32,
        prompt_ids=[1, 2],
        max_new_tokens=4,
        profile_steps=True,
        profiler=profiler,
    )

    assert profiler.started == 1
    assert profiler.steps == 2
    assert profiler.stopped == 1
    assert result["profile_decode_steps"] == 2
    assert result["profile_steps_match"] is True
    assert result["profile_expected_steps"] == 2
    assert result["profile_start_context_len"] == 5
    assert result["generated_token_ids"] == [10, 11, 12, 13]
    assert result["generated_text"] == "10111213"


def test_bench_engine_stops_profiler_when_profile_step_raises(monkeypatch):
    tokenizer = FakeTokenizer()
    steps = [
        FakeStep([FakeEvent("prefill_sampled_first_token")]),
        FakeStep([FakeEvent("decode_token")]),
        FakeStep([FakeEvent("decode_token")]),
    ]
    _patch_engine(monkeypatch, steps, tokenizer)
    profiler = FakeProfiler(fail_step=True)
    pending = iter([True, True, True])
    monkeypatch.setattr(FakeEngine, "has_pending_work", lambda self: next(pending))

    with pytest.raises(RuntimeError, match="fake profiler failure"):
        BENCH.bench_engine(
            model_config=object(),
            weights=object(),
            model_path=Path("."),
            device="cpu",
            dtype=torch.float32,
            prompt_ids=[1],
            max_new_tokens=3,
            profiler=profiler,
        )

    assert profiler.started == 1
    assert profiler.stopped == 1
