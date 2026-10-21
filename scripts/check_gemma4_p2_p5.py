#!/usr/bin/env python3
"""P2-P5 evidence collector for the Gemma4 engine.

Runs the existing acceptance checks and consolidates their artifacts into one
JSON/Markdown evidence bundle:

  P2 cache      batch/paged/KV-sharing parity, prefix-cache hit TTFT
  P3 low memory INT4 footprint, graph VRAM, prefill peak
  P4 CUDA       server + kernel test suites
  P5 graph      batch graph keys and graph-vs-eager parity

Use --quick to reuse the current artifacts without re-running benchmarks.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT.parent / "mini-llm-kernels"
REPORTS = ROOT / "reports"


def _run(cmd, cwd, timeout=1800):
    started = time.time()
    proc = subprocess.run(
        cmd, cwd=str(cwd), text=True, capture_output=True, timeout=timeout
    )
    return {
        "cmd": cmd,
        "cwd": str(cwd),
        "returncode": proc.returncode,
        "seconds": round(time.time() - started, 1),
        "tail": (proc.stdout + proc.stderr)[-3000:],
    }


def _passed_count(text):
    m = re.findall(r"(\d+) passed", text)
    return int(m[-1]) if m else None


def _load(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="only summarize existing artifacts")
    parser.add_argument("--skip-prefill", action="store_true")
    parser.add_argument("--skip-prefix-bench", action="store_true")
    parser.add_argument("--output-json", type=Path,
                        default=REPORTS / "gemma4_p2_p5_evidence.json")
    parser.add_argument("--output-md", type=Path,
                        default=REPORTS / "gemma4_p2_p5_evidence.md")
    args = parser.parse_args()

    runs = []
    if not args.quick:
        runs.append(_run(
            [sys.executable, "-m", "pytest", "tests", "-q",
             "--ignore=tests/test_gemma4_mlx.py"], ROOT, 900))
        runs.append(_run(
            [sys.executable, "-m", "pytest", "tests", "-q"], KERNELS, 900))

    parity = _load(REPORTS / "gemma4_batch_parity.json")
    if not args.quick:
        runs.append(_run(
            [sys.executable, "scripts/check_gemma4_batch_decode.py"],
            ROOT, 1800))
        parity = _load(REPORTS / "gemma4_batch_parity.json")

    prefill = _load(REPORTS / "gemma4_e_prefill.json")
    if not args.quick and not args.skip_prefill:
        runs.append(_run(
            [sys.executable, "scripts/bench_gemma4_prefill.py",
             "--sizes", "2048", "4096",
             "--gemm-min-rows", "8", "--gemv-min-rows", "100000"],
            ROOT, 1800))
        prefill = _load(REPORTS / "gemma4_e_prefill.json")

    prefix = None
    if not args.quick and not args.skip_prefix_bench:
        out = REPORTS / "gemma4_prefix_cache_bench.json"
        runs.append(_run(
            [sys.executable, "scripts/bench_gemma4_batch.py",
             "--batch-sizes", "4", "--max-new-tokens", "64",
             "--output-json", str(out)], ROOT, 1800))
        prefix = _load(out)

    verification = _load(REPORTS / "gemma4_batch_final_verification.json") or {}

    server_passed = _passed_count(runs[0]["tail"]) if len(runs) > 0 else None
    kernel_passed = _passed_count(runs[1]["tail"]) if len(runs) > 1 else None
    all_parity = bool(parity and parity.get("all_passed"))
    graphs = []
    if parity:
        for row in parity.get("results", []):
            graphs.append({"batch": row["batch"], "graph_keys": row.get("batch_graphs")})

    p4 = {}
    if prefill:
        for row in prefill.get("rows", []):
            if row["prompt_tokens"] in (2048, 4096):
                p4[str(row["prompt_tokens"])] = {
                    "prefill_s": row["gemm"]["prefill_s"],
                    "prefill_tok_s": row["gemm"]["prefill_tok_s"],
                    "peak_alloc_gib": row["gemm"]["peak_alloc_gib"],
                    "free_after_gib": row["gemm"]["free_after_gib"],
                }

    hit = None
    if prefix:
        for row in prefix.get("rows", []):
            if row.get("mode") == "batched_graph" and row.get("batch") == 4:
                hit = {
                    "batch": 4,
                    "output": 64,
                    "last_prefill_s": row.get("last_prefill_s"),
                    "wall_s": row.get("wall_s"),
                    "e2e_tok_s": row.get("aggregate_e2e_tok_s"),
                    "per_request_tok_s": row.get("per_request_tok_s"),
                }

    evidence = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quick": args.quick,
        "P2_cache": {
            "batch_parity_all_passed": all_parity,
            "batches": [r["batch"] for r in parity.get("results", [])] if parity else [],
            "batch_graph_keys": graphs,
            "prefix_cache_hit_n4_64out": hit,
        },
        "P3_memory": {
            "vram": verification.get("vram"),
            "prefill": p4,
            "kv_dtype": "fp16",
            "weight_format": "INT4 group64",
        },
        "P4_cuda": {
            "server_tests_passed": server_passed,
            "kernel_tests_passed": kernel_passed,
        },
        "P5_graph": {
            "graph_replay_parity": all_parity,
            "short_and_long_graphs": bool(graphs and any(
                len(g["graph_keys"] or []) > 1 for g in graphs)),
        },
        "runs": runs,
        "artifacts": [
            str(REPORTS / "gemma4_batch_parity.json"),
            str(REPORTS / "gemma4_e_prefill.json"),
            str(REPORTS / "gemma4_batch_final_verification.json"),
        ],
    }
    args.output_json.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Gemma4 P2-P5 evidence",
        "",
        f"Generated: {evidence['generated_at']}",
        "",
        "## P2 cache",
        f"- batch/paged/KV-sharing token parity: **{'PASS' if all_parity else 'FAIL'}**",
        f"- batches: {evidence['P2_cache']['batches']}",
        f"- graph keys: {graphs}",
    ]
    if hit:
        lines += [
            f"- prefix-cache hit (N=4, 64 out): TTFT **{hit['last_prefill_s']} s**, "
            f"e2e **{hit['e2e_tok_s']} tok/s**, per-req **{hit['per_request_tok_s']}**",
        ]
    lines += [
        "",
        "## P3 low memory",
        f"- format: INT4 group64, KV fp16",
        f"- N=16 graph VRAM: {evidence['P3_memory']['vram']}",
    ]
    for size, row in p4.items():
        lines.append(
            f"- prefill {size}: {row['prefill_s']} s, "
            f"{row['prefill_tok_s']} tok/s, peak {row['peak_alloc_gib']} GiB"
        )
    lines += [
        "",
        "## P4 CUDA",
        f"- server tests: **{server_passed} passed**",
        f"- kernel tests: **{kernel_passed} passed**",
        "",
        "## P5 graph",
        f"- graph-vs-eager parity: **{'PASS' if all_parity else 'FAIL'}**",
        f"- short+long graphs present: {evidence['P5_graph']['short_and_long_graphs']}",
        "",
        "## Commands",
    ]
    for run in runs:
        lines.append(f"- `{' '.join(run['cmd'])}` → rc={run['returncode']}, {run['seconds']} s")
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "P2_pass": all_parity,
        "P4_server": server_passed,
        "P4_kernel": kernel_passed,
        "P5_pass": all_parity,
        "json": str(args.output_json),
        "md": str(args.output_md),
    }, ensure_ascii=False))
    return 0 if all_parity and (server_passed and kernel_passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
