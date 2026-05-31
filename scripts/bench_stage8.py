"""bench_stage8.py — Stage 8 系统性 Benchmark 脚本

在不同 batch_size × context_len 组合下对比 Stage 8 (custom kernel) 与 Stage 7
(PyTorch fallback) 的吞吐，并打印汇总表格。

用法示例：
  # CUDA 环境：kernel on（默认）
  python scripts/bench_stage8.py --model Qwen/Qwen2.5-0.5B-Instruct --greedy

  # CUDA 环境：A/B 对比（kernel on vs off）
  python scripts/bench_stage8.py --model Qwen/Qwen2.5-0.5B-Instruct --greedy --ab

  # M1/CPU：只跑 fallback
  python scripts/bench_stage8.py --model Qwen/Qwen2.5-0.5B-Instruct --greedy
"""

from __future__ import annotations

import argparse
import os
import subprocess
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
TABLE_RE = re.compile(r"^\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*(\d+)\s*\|\s*([0-9.]+)\s*$")


def run_bench_once(
    batch_list: str,
    max_new: int,
    runs: int,
    prompt: str,
    greedy: bool,
    no_custom_kernels: bool = False,
    extra_args: Optional[List[str]] = None,
) -> Dict[int, float]:
    """调用 bench_engine.py，返回 {batch_size: tok_s}"""
    cmd = [
        sys.executable,
        "scripts/bench_engine.py",
        "--batch-list", batch_list,
        "--max-new", str(max_new),
        "--runs", str(runs),
        "--prompt", prompt,
        "--decode-batch-bucket-multiple", "0",  # 关闭分桶，减少干扰
    ]
    if greedy:
        cmd.append("--greedy")
    if no_custom_kernels:
        cmd.append("--no-custom-kernels")
    if extra_args:
        cmd.extend(extra_args)

    try:
        out = subprocess.check_output(cmd, cwd=str(ROOT), text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f"[bench] subprocess failed:\n{e.output[:2000]}")
        return {}

    toks: Dict[int, float] = {}
    for line in out.splitlines():
        m = TABLE_RE.match(line)
        if m:
            toks[int(m.group(1))] = float(m.group(4))
    return toks


def print_grid(
    results: Dict[str, Dict[int, float]],
    batches: List[int],
    label_a: str,
    label_b: Optional[str] = None,
) -> None:
    """打印 batch × config 网格对比表"""
    configs = list(results.keys())
    # 表头
    col_w = 12
    header = f"{'batch':>6}"
    for cfg in configs:
        header += f"  {cfg[:col_w]:>{col_w}}"
    if len(configs) == 2 and label_b:
        header += f"  {'delta%':>8}"
    print(header)
    print("-" * len(header))

    for n in batches:
        row = f"{n:>6}"
        vals = []
        for cfg in configs:
            v = results[cfg].get(n)
            vals.append(v)
            row += f"  {str(round(v, 1)) if v is not None else 'N/A':>{col_w}}"
        if len(configs) == 2 and label_b and vals[0] and vals[1]:
            pct = (vals[1] - vals[0]) / vals[0] * 100
            row += f"  {pct:>+7.1f}%"
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--batch-list", default="1,4,8,16",
                        help="测试的 batch size 列表")
    parser.add_argument("--max-new", type=int, default=128,
                        help="每请求生成 token 数（近似 decode 长度）")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--prompt", default="请介绍一下机器学习的基本概念")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--ab", action="store_true",
                        help="Stage 8 A/B 对比：kernel off vs kernel on")
    args = parser.parse_args()

    batches = [int(x) for x in args.batch_list.split(",") if x.strip()]
    kw = dict(
        batch_list=args.batch_list,
        max_new=args.max_new,
        runs=args.runs,
        prompt=args.prompt,
        greedy=args.greedy,
    )

    if args.ab:
        print(f"\n[Stage 8 A/B Benchmark]  model={args.model}  max_new={args.max_new}  greedy={args.greedy}")
        print("── 运行 baseline (kernel=OFF)...")
        fallback = run_bench_once(no_custom_kernels=True, **kw)
        print("── 运行 candidate (kernel=ON)...")
        kernel   = run_bench_once(no_custom_kernels=False, **kw)

        results = {"PyTorch fallback": fallback, "Custom kernel": kernel}
        print(f"\n{'tok/s':>6}", end="")
        print_grid(results, batches, "fallback", "kernel")
    else:
        print(f"\n[Stage 8 Benchmark]  model={args.model}  max_new={args.max_new}  greedy={args.greedy}")
        print("── 运行...")
        res = run_bench_once(**kw)
        from miniservellm.runtime.nn_ops import _HAS_CUSTOM_KERNELS
        label = "Custom kernel" if _HAS_CUSTOM_KERNELS else "PyTorch fallback"
        results = {label: res}
        print(f"\ntok/s ({label})")
        print_grid(results, batches, label)

    print("\n[Stage 8 调优建议 (CUDA 环境)]")
    print("  - M4 block_size: 试验 16/32/64，大 context 时 64 更优")
    print("  - M4 向量化访存: float4 128-bit load（尤其 D=64 时减少 load 指令）")
    print("  - M4 shared memory padding: 避免 bank conflict（+1 列）")
    print("  - M3 warp 数: blockDim.x=256 vs 512，视 hidden_size 调整")
    print("  - M2 GEMM: cuBLAS batch GEMM，大 batch 时考虑 Triton/cutlass")
    print("  工具: nsys profile + ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active")


if __name__ == "__main__":
    main()
