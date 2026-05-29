from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict

import torch


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


class PhaseTimer:
    def __init__(self) -> None:
        self.totals: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    @contextmanager
    def record(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.totals[name] = self.totals.get(name, 0.0) + dt
            self.counts[name] = self.counts.get(name, 0) + 1

    def reset(self) -> None:
        self.totals.clear()
        self.counts.clear()

    def report(self, total_step_time: float) -> str:
        if not self.totals:
            return "(no phases recorded)"
        lines = []
        for name, total in sorted(self.totals.items(), key=lambda x: -x[1]):
            pct = total / total_step_time * 100 if total_step_time > 0 else 0.0
            cnt = self.counts[name]
            lines.append(
                f"  {name:<24} total={total*1000:>7.1f}ms  "
                f"avg={total/cnt*1000:>6.2f}ms  calls={cnt:>4}  ({pct:>5.1f}%)"
            )
        return "\n".join(lines)


def patch_engine_with_timer(engine, timer: PhaseTimer) -> None:
    mr = engine.model_runner
    sampler = engine.sampler

    orig_fp = mr.forward_fresh_prefill
    orig_ip = mr.forward_incremental_prefill
    orig_dec = mr.forward_decode
    orig_sample = sampler.sample_batch

    def wrap_fp(*a, **k):
        with timer.record("forward_fresh_prefill"):
            out = orig_fp(*a, **k)
            maybe_sync(mr.device)
        return out

    def wrap_ip(*a, **k):
        with timer.record("forward_incremental_prefill"):
            out = orig_ip(*a, **k)
            maybe_sync(mr.device)
        return out

    def wrap_dec(*a, **k):
        with timer.record("forward_decode"):
            out = orig_dec(*a, **k)
            maybe_sync(mr.device)
        return out

    def wrap_sample(*a, **k):
        with timer.record("sample_batch"):
            out = orig_sample(*a, **k)
        return out

    mr.forward_fresh_prefill = wrap_fp
    mr.forward_incremental_prefill = wrap_ip
    mr.forward_decode = wrap_dec
    sampler.sample_batch = wrap_sample
