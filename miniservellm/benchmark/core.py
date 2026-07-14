from __future__ import annotations

import time
from typing import Callable, List, Tuple

from miniservellm.benchmark.timer import PhaseTimer, maybe_sync, patch_engine_with_timer


def run_one_trial(engine, prompts: List[str], sp, max_new: int, timer: PhaseTimer):
    rids = [engine.add_request(text=p, sampling_params=sp, max_new_tokens=max_new) for p in prompts]

    timer.reset()
    maybe_sync(engine.model_runner.device)
    t0 = time.perf_counter()
    engine.run_until_all_finished(max_steps=1000000, collect_results=False)
    maybe_sync(engine.model_runner.device)
    elapsed = time.perf_counter() - t0

    total_new = sum(len(engine.get_request(r).generated_token_ids) for r in rids)
    outputs = [engine.get_text(r) for r in rids]
    return elapsed, total_new, outputs


def median_run(
    make_engine: Callable[[], object],
    prompts: List[str],
    sp,
    max_new: int,
    runs: int = 3,
) -> Tuple[float, int, List[str], str, List[float], List[int]]:
    elapsed_list = []
    tok_list = []
    out_list = []
    timer_reports = []

    for _ in range(runs):
        engine = make_engine()
        timer = PhaseTimer()
        patch_engine_with_timer(engine, timer)
        elapsed, total_new, outputs = run_one_trial(engine, prompts, sp, max_new, timer)
        elapsed_list.append(elapsed)
        tok_list.append(total_new)
        out_list.append(outputs)
        timer_reports.append((elapsed, timer.report(elapsed)))

    idx = sorted(range(runs), key=lambda i: elapsed_list[i])[runs // 2]
    return elapsed_list[idx], tok_list[idx], out_list[idx], timer_reports[idx][1], elapsed_list, tok_list
