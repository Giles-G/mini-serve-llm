"""第六阶段 Benchmark 脚本

用途：
  1. 拿到固定 prompt / max_new / N 组合下的吞吐 baseline
  2. 给出每步前向各个子阶段的时间分解（embed / decode block / sample / prefill）

使用：
    python scripts/bench_engine.py
    python scripts/bench_engine.py --batch 8 --max-new 64

设计原则：
  - 多次取中位数（warmup + 多次正式跑）
  - 用 time.perf_counter（MPS 上的 torch.cuda.Event 不可用）
  - 在 KV Cache / Sampler 等关键路径上加轻量 timer，不引入侵入式重构

整体流程：
  1. 解析命令行参数，确定模型、batch 列表、max_new、采样方式等
  2. 加载 tokenizer / HF config / 模型权重，并通过 Adapter 转成自研权重结构
  3. 每次 trial 重新构建 Stage5Engine，避免上一次请求/KV Cache 状态污染下一次测试
  4. 对每个 batch size 先 warmup 一次，再正式运行多次
  5. 取 elapsed 的中位数作为该 batch size 的结果，并打印 phase breakdown
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List

# 将项目根目录加入 Python 搜索路径，使脚本可以直接从源码目录 import miniservellm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from miniservellm.config import EngineConfig
from miniservellm.model_adapter.adapters.qwen2_adapter import Qwen2Adapter
from miniservellm.model_adapter.hf_loader import load_model_bundle
from miniservellm.cache.kv_cache import KVCacheManager
from miniservellm.runtime.model_runner import TransformerModelRunner
from miniservellm.runtime.inference_engine import Stage5Engine
from miniservellm.scheduler.request import SamplingParams


# ---------------------------------------------------------------------------
# 轻量计时器：通过 monkey-patch 关键方法收集时间分解
# ---------------------------------------------------------------------------

class PhaseTimer:
    """按 phase 名称累计耗时和调用次数的轻量计时器

    这个类不关心具体业务逻辑，只负责统计：
      - 每个 phase 总共耗时多少秒
      - 每个 phase 被调用了多少次

    使用方式：
        with timer.record("forward_decode"):
            model.forward_decode(...)

    Attributes:
        totals: phase 名称 -> 累计耗时（秒）
        counts: phase 名称 -> 调用次数
    """

    def __init__(self) -> None:
        self.totals: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    @contextmanager
    def record(self, name: str):
        """记录一段代码的执行耗时

        Args:
            name: phase 名称，例如 "forward_decode"、"sample_batch"

        Notes:
            这里用 time.perf_counter() 统计墙钟时间。
            如果被计时代码包含 GPU/MPS 异步计算，调用方需要在合适位置同步设备，
            否则计时可能只统计到 kernel launch 时间，而不是实际执行时间。
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.totals[name] = self.totals.get(name, 0.0) + dt
            self.counts[name] = self.counts.get(name, 0) + 1

    def reset(self) -> None:
        """清空所有累计数据，用于每次 trial 开始前重置计时器"""
        self.totals.clear()
        self.counts.clear()

    def report(self, total_step_time: float) -> str:
        """生成 phase breakdown 文本报告

        Args:
            total_step_time: 本次 trial 的总耗时（秒），用于计算 phase 占比

        Returns:
            多行字符串，每行包含 phase 总耗时、平均耗时、调用次数和占比
        """
        if not self.totals:
            return "(no phases recorded)"

        lines = []
        # 按总耗时从大到小排序，方便快速看到热点
        for name, total in sorted(self.totals.items(), key=lambda x: -x[1]):
            pct = total / total_step_time * 100 if total_step_time > 0 else 0.0
            cnt = self.counts[name]
            lines.append(
                f"  {name:<24} total={total*1000:>7.1f}ms  "
                f"avg={total/cnt*1000:>6.2f}ms  calls={cnt:>4}  ({pct:>5.1f}%)"
            )
        return "\n".join(lines)


def patch_engine_with_timer(engine: Stage5Engine, timer: PhaseTimer) -> None:
    """给 engine 的关键方法加上 timer 包装

    这里使用 monkey-patch 的方式替换对象方法，而不是改业务代码。
    好处是 benchmark 逻辑和推理实现解耦；坏处是只适合调试/实验脚本，不适合生产代码。

    被统计的热点：
      - model_runner.forward_fresh_prefill: 首次 prefill 前向
      - model_runner.forward_incremental_prefill: 增量 prefill 前向
      - model_runner.forward_decode: decode 前向
      - sampler.sample_batch: 批量采样

    Args:
        engine: 需要打补丁的 Stage5Engine 实例
        timer: PhaseTimer 实例，所有 wrapper 都写入这个 timer
    """
    mr = engine.model_runner
    sampler = engine.sampler

    # 保存原始方法，wrapper 内部仍然调用原方法
    orig_fp = mr.forward_fresh_prefill
    orig_ip = mr.forward_incremental_prefill
    orig_dec = mr.forward_decode
    orig_sample = sampler.sample_batch

    def wrap_fp(*a, **k):
        # fresh prefill 可能包含较长 prompt chunk，需要同步设备后再结束计时
        with timer.record("forward_fresh_prefill"):
            out = orig_fp(*a, **k)
            _maybe_sync(mr.device)
        return out

    def wrap_ip(*a, **k):
        # incremental prefill 是长 prompt 的后续 chunk
        with timer.record("forward_incremental_prefill"):
            out = orig_ip(*a, **k)
            _maybe_sync(mr.device)
        return out

    def wrap_dec(*a, **k):
        # decode 阶段每个请求每步生成 1 个 token，是生成阶段的主要热点
        with timer.record("forward_decode"):
            out = orig_dec(*a, **k)
            _maybe_sync(mr.device)
        return out

    def wrap_sample(*a, **k):
        # sample_batch 在 CPU/PyTorch 层做采样，通常不需要额外同步
        with timer.record("sample_batch"):
            out = orig_sample(*a, **k)
        return out

    # monkey-patch：把实例方法替换成计时 wrapper
    mr.forward_fresh_prefill = wrap_fp
    mr.forward_incremental_prefill = wrap_ip
    mr.forward_decode = wrap_dec
    sampler.sample_batch = wrap_sample


def _maybe_sync(device: torch.device) -> None:
    """对当前设备做同步，确保 timer 数字反映真实设备执行时间

    CUDA/MPS kernel 通常是异步提交的，如果不同步，perf_counter 可能只统计提交时间，
    不统计真正的 GPU/MPS 执行时间。

    Args:
        device: 当前模型运行设备
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def build_engine(weights_cache, ec: EngineConfig, mc, tok, _hf, w_cpu, adapter):
    """构建一个全新的 Stage5Engine

    每次 trial 都重新构建 engine，原因是：
      - KVCacheManager 内部有 block 分配状态
      - RequestQueue 内部有请求状态
      - Stage5Engine 内部有 request_id 计数和 requests_by_id

    如果复用同一个 engine，会导致前一次 benchmark 的状态污染后一次结果。

    Args:
        weights_cache: 预留参数，目前未使用
        ec: EngineConfig
        mc: ModelConfig
        tok: tokenizer
        _hf: HF model，预留参数，目前未使用
        w_cpu: CPU 上的自研权重结构
        adapter: 模型 adapter，用于把权重搬到目标设备/dtype

    Returns:
        新构建的 Stage5Engine
    """
    # 每次新 engine 都将 CPU 权重搬到目标设备，保证 runner 使用 device/dtype 正确的权重
    w = adapter.move_weights_to_device(w_cpu, device=ec.device, dtype=ec.dtype)
    # KV Cache 每次重新初始化，避免 block 使用状态跨 trial 残留
    kvm = KVCacheManager(ec, mc)
    mr = TransformerModelRunner(engine_config=ec, model_config=mc, weights=w, kv_cache_manager=kvm)
    engine = Stage5Engine(engine_config=ec, model_config=mc, model_runner=mr, tokenizer=tok)
    return engine


def run_one_trial(
    engine: Stage5Engine,
    prompts: List[str],
    sp: SamplingParams,
    max_new: int,
    timer: PhaseTimer,
):
    """执行一次 benchmark trial

    流程：
      1. 将 prompts 全部加入 engine
      2. 清空 timer，避免 add_request 阶段被统计
      3. 同步设备并开始计时
      4. run_until_all_finished() 跑到所有请求结束
      5. 再次同步设备并统计 elapsed
      6. 统计总生成 token 数
      7. 收集每个请求的推理结果文本，便于在 benchmark 结尾展示样例输出

    Args:
        engine: 已构建好的 Stage5Engine
        prompts: 请求文本列表，长度就是本次 batch size
        sp: 所有请求共用的采样参数
        max_new: 每个请求最大生成 token 数
        timer: 计时器

    Returns:
        (elapsed, total_new, outputs):
            - elapsed: 总耗时（秒）
            - total_new: 总生成 token 数
            - outputs: 每个请求的推理结果文本（仅生成部分）
    """
    rids = [engine.add_request(text=p, sampling_params=sp, max_new_tokens=max_new) for p in prompts]

    # 只统计正式推理阶段，不统计 add_request/tokenize 阶段
    timer.reset()
    _maybe_sync(engine.model_runner.device)
    t0 = time.perf_counter()
    engine.run_until_all_finished(max_steps=10000)
    _maybe_sync(engine.model_runner.device)
    elapsed = time.perf_counter() - t0

    # 总生成 token 数 = 所有请求 generated_token_ids 的长度之和
    total_new = sum(len(engine.get_request(r).generated_token_ids) for r in rids)
    outputs = [engine.get_text(r) for r in rids]
    return elapsed, total_new, outputs


def median_run(make_engine, prompts, sp, max_new, runs=3):
    """执行多次 trial，并返回耗时中位数对应的结果

    为什么取中位数：
      - 单次运行容易受系统调度、首次 kernel 编译、缓存状态影响
      - 中位数比平均值更不容易被异常慢/异常快的一次污染

    Args:
        make_engine: 无参函数，每次调用返回一个全新的 engine
        prompts: 请求文本列表
        sp: 采样参数
        max_new: 每个请求最大生成 token 数
        runs: 正式 trial 次数

    Returns:
        elapsed: 中位数 trial 的耗时
        tokens: 中位数 trial 的生成 token 数
        outputs: 中位数 trial 的推理结果文本列表
        report: 中位数 trial 的 phase breakdown
        elapsed_list: 所有 trial 的耗时列表
        tok_list: 所有 trial 的生成 token 数列表
    """
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

    # 找到耗时排序后的中位数对应的原始索引
    idx = sorted(range(runs), key=lambda i: elapsed_list[i])[runs // 2]
    return elapsed_list[idx], tok_list[idx], out_list[idx], timer_reports[idx][1], elapsed_list, tok_list


def main():
    """命令行入口：加载模型、构建 benchmark 配置并执行所有 batch size 测试"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--batch-list", default="1,4,8")
    parser.add_argument("--max-new", type=int, default=512)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--prompt", default="用400字左右介绍一下机器学习")
    parser.add_argument("--greedy", action="store_true", help="使用贪心解码（temperature=0）")
    parser.add_argument("--kv-decode-block-reserve", type=int, default=2, help="为 decode 预留的最少空闲 KV block")
    parser.add_argument("--kv-reserve-relax-after-no-progress-steps", type=int, default=8, help="连续无进展多少步后放宽一次 KV 水位")
    args = parser.parse_args()

    # 引擎参数（容量按最大 batch 设置一次，所有 N 共用）
    # 注意：这些配置会影响 benchmark 结果，尤其是 max_decode_requests_per_step、prefill_chunk_size、KV block 容量。
    ec = EngineConfig.create(
        # 自动选择 device：CUDA / MPS / CPU
        device="auto",
        # 自动选择 dtype：GPU/MPS 通常用半精度，CPU 通常用 fp32
        dtype="auto",
        # Paged KV Cache 每个 block 能存多少个 token 的 KV
        block_size=16,
        # 预分配多少个 GPU KV block，总 KV token 容量 = block_size * num_gpu_blocks
        num_gpu_blocks=128,
        # 单个 step 最多调度多少个请求（decode + prefill 总和）
        max_batch_size=16,
        # 单个 step 最多处理多少 token（decode 每请求 1 token，prefill 按 chunk 大小计）
        max_tokens_per_step=512,
        # 单个 step 中 prefill 最多处理多少 token，避免长 prompt 抢占 decode
        max_prefill_tokens_per_step=512,
        # 单个 step 最多处理多少个 decode 请求
        max_decode_requests_per_step=16,
        # 单个请求每轮 prefill chunk 最大 token 数
        prefill_chunk_size=128,
        # 默认采样参数；如果 --greedy 开启，会被下面的 SamplingParams 覆盖
        default_temperature=0.7,
        default_top_k=20,
        default_top_p=0.9,
        # 调度器 KV 水位参数：用于避免 prefill 抢光 KV 导致 decode 卡死
        kv_decode_block_reserve=args.kv_decode_block_reserve,
        kv_reserve_relax_after_no_progress_steps=args.kv_reserve_relax_after_no_progress_steps,
    )
    print(
        f"[bench] device={ec.device} dtype={ec.dtype} "
        f"kv_reserve={ec.kv_decode_block_reserve} "
        f"kv_relax_after={ec.kv_reserve_relax_after_no_progress_steps}"
    )

    adapter = Qwen2Adapter()
    # 加载 tokenizer、HF config、model_config、HF model 和 CPU 权重
    # 当前 benchmark 只用 CPU 权重 w_cpu，后续每个 trial 会 move_weights_to_device 到目标设备。
    tok, _hfcfg, mc, _hf, w_cpu = load_model_bundle(
        adapter=adapter,
        model_name_or_path=args.model,
        trust_remote_code=False,
        load_model_device="cpu",
        load_dtype=None,
    )

    # 将 tokenizer 的 EOS token 写入 engine_config，方便生成时正确停止
    if tok.eos_token_id is not None:
        ec.eos_token_id = int(tok.eos_token_id)

    # 采样参数：greedy 模式用于更稳定地观察纯前向性能；非 greedy 更接近真实采样场景
    if args.greedy:
        sp = SamplingParams(temperature=0.0, top_k=0, top_p=1.0, repetition_penalty=1.0)
    else:
        sp = SamplingParams(temperature=0.7, top_k=20, top_p=0.9, repetition_penalty=1.1)

    def make_engine():
        """为每次 trial 创建全新 engine，避免 KV Cache / 请求状态污染"""
        return build_engine(None, ec, mc, tok, _hf, w_cpu, adapter)

    # batch-list 形如 "1,2,4,8"，解析成 [1, 2, 4, 8]
    batches = [int(x) for x in args.batch_list.split(",") if x.strip()]

    print(f"\n[bench] prompt={args.prompt!r}  max_new={args.max_new}  runs={args.runs}  greedy={args.greedy}")
    print(f"{'N':>3} | {'elapsed(s)':>10} | {'gen_tokens':>10} | {'tok/s':>10}")
    print("-" * 50)

    summaries = {}
    for n in batches:
        # 当前 batch size 下构造 n 个相同 prompt，用于观察不同并发数下的吞吐变化
        prompts = [args.prompt] * n

        # warmup once：预热一次但不纳入统计，减少首次运行开销对正式结果的影响
        eng = make_engine()
        timer = PhaseTimer()
        patch_engine_with_timer(eng, timer)
        run_one_trial(eng, prompts, sp, args.max_new, timer)

        # formal runs：正式运行多次并取中位数
        elapsed, tokens, outputs, report, elapsed_list, tok_list = median_run(
            make_engine,
            prompts,
            sp,
            args.max_new,
            runs=args.runs,
        )
        thr = tokens / elapsed if elapsed > 0 else 0.0
        print(f"{n:>3} | {elapsed:>10.2f} | {tokens:>10} | {thr:>10.2f}")
        summaries[n] = (
            elapsed,
            tokens,
            thr,
            outputs,
            report,
            list(zip(elapsed_list, tok_list)),
        )

    # 打印每个 batch size 的中位数 trial 的阶段耗时分解
    print("\n[bench] phase breakdown (median run):")
    for n, (elapsed, tokens, thr, outputs, report, raw) in summaries.items():
        print(f"\n--- N={n}  elapsed={elapsed:.2f}s  {tokens} tokens  {thr:.2f} tok/s ---")
        print(f"raw runs (elapsed, tokens): {raw}")
        print(report)
        print("[sample outputs]")
        for i, out in enumerate(outputs[: min(2, len(outputs))]):
            preview = out.strip().replace("\n", " ")
            # if len(preview) > 160:
            #     preview = preview[:160] + "..."
            print(f"  req#{i}: {preview}")


if __name__ == "__main__":
    main()
