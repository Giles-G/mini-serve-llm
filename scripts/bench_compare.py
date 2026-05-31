"""A/B 对比 benchmark 脚本。

用途：在同一组测试口径下，快速比较两套参数的 tok/s 差异。
Stage 8 新增：支持 --kernel-ab 模式，对比 custom kernel on vs off 的吞吐。

示例：
  # 原有：kv reserve 参数对比
  python scripts/bench_compare.py \\
    --batch-list 1,4,8 --max-new 48 --runs 2 --greedy \\
    --base-kv-reserve 0 --cand-kv-reserve 2

  # Stage 8 新增：kernel on vs off
  python scripts/bench_compare.py \\
    --batch-list 1,4,8 --max-new 64 --runs 2 --greedy --kernel-ab
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple


TABLE_RE = re.compile(r"^\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*(\d+)\s*\|\s*([0-9.]+)\s*$")


def run_bench(
    python_bin: str,
    repo_root: Path,
    batch_list: str,
    max_new: int,
    runs: int,
    prompt: str,
    greedy: bool,
    kv_reserve: int,
    kv_relax_after: int,
    no_custom_kernels: bool = False,
) -> Tuple[str, Dict[int, float]]:
    cmd = [
        python_bin,
        "scripts/bench_engine.py",
        "--batch-list", batch_list,
        "--max-new", str(max_new),
        "--runs", str(runs),
        "--prompt", prompt,
        "--kv-decode-block-reserve", str(kv_reserve),
        "--kv-reserve-relax-after-no-progress-steps", str(kv_relax_after),
    ]
    if greedy:
        cmd.append("--greedy")
    if no_custom_kernels:
        cmd.append("--no-custom-kernels")

    out = subprocess.check_output(cmd, cwd=str(repo_root), text=True, stderr=subprocess.STDOUT)
    toks_by_n: Dict[int, float] = {}
    for line in out.splitlines():
        m = TABLE_RE.match(line)
        if m:
            n = int(m.group(1))
            tok_s = float(m.group(4))
            toks_by_n[n] = tok_s
    return out, toks_by_n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-bin", default=sys.executable, help="运行 bench_engine.py 使用的 Python")
    parser.add_argument("--batch-list", default="1,4,8")
    parser.add_argument("--max-new", type=int, default=48)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--prompt", default="请简单介绍一下机器学习")
    parser.add_argument("--greedy", action="store_true")

    parser.add_argument("--base-kv-reserve", type=int, default=0)
    parser.add_argument("--base-kv-relax-after", type=int, default=8)
    parser.add_argument("--cand-kv-reserve", type=int, default=2)
    parser.add_argument("--cand-kv-relax-after", type=int, default=8)
    parser.add_argument("--print-raw", action="store_true", help="打印两组完整原始输出")
    # Stage 8 M5：kernel A/B 对比模式
    parser.add_argument("--kernel-ab", action="store_true",
                        help="Stage 8 A/B: baseline=custom kernel OFF，candidate=custom kernel ON")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    # kernel-ab 模式：baseline=no kernel，candidate=with kernel
    base_no_kernel = args.kernel_ab
    cand_no_kernel = False
    base_label = "kernel=OFF (PyTorch fallback)" if args.kernel_ab else f"kv_reserve={args.base_kv_reserve}"
    cand_label = "kernel=ON  (custom CUDA)"      if args.kernel_ab else f"kv_reserve={args.cand_kv_reserve}"

    print(f"[compare] 运行 baseline ({base_label})...")
    base_raw, base = run_bench(
        python_bin=args.python_bin,
        repo_root=repo_root,
        batch_list=args.batch_list,
        max_new=args.max_new,
        runs=args.runs,
        prompt=args.prompt,
        greedy=args.greedy,
        kv_reserve=args.base_kv_reserve,
        kv_relax_after=args.base_kv_relax_after,
        no_custom_kernels=base_no_kernel,
    )

    print(f"[compare] 运行 candidate ({cand_label})...")
    cand_raw, cand = run_bench(
        python_bin=args.python_bin,
        repo_root=repo_root,
        batch_list=args.batch_list,
        max_new=args.max_new,
        runs=args.runs,
        prompt=args.prompt,
        greedy=args.greedy,
        kv_reserve=args.cand_kv_reserve,
        kv_relax_after=args.cand_kv_relax_after,
        no_custom_kernels=cand_no_kernel,
    )

    if args.print_raw:
        print("\n===== BASELINE RAW =====")
        print(base_raw)
        print("\n===== CANDIDATE RAW =====")
        print(cand_raw)

    ns = sorted(set(base.keys()) | set(cand.keys()))
    print(f"\n[compare] tok/s 对比  baseline={base_label}  candidate={cand_label}")
    print(f"{'N':>3} | {'base':>10} | {'cand':>10} | {'delta':>10} | {'delta%':>8}")
    print("-" * 56)
    for n in ns:
        b = base.get(n)
        c = cand.get(n)
        if b is None or c is None:
            print(f"{n:>3} | {str(b):>10} | {str(c):>10} | {'-':>10} | {'-':>8}")
            continue
        d = c - b
        pct = (d / b * 100.0) if b > 0 else 0.0
        print(f"{n:>3} | {b:>10.2f} | {c:>10.2f} | {d:>10.2f} | {pct:>7.2f}%")


if __name__ == "__main__":
    main()
