from __future__ import annotations


def print_benchmark_summary(summaries: dict) -> None:
    print("\n[bench] phase breakdown (median run):")
    for n, (elapsed, tokens, thr, outputs, report, raw) in summaries.items():
        print(f"\n--- N={n}  elapsed={elapsed:.2f}s  {tokens} tokens  {thr:.2f} tok/s ---")
        print(f"raw runs (elapsed, tokens): {raw}")
        print(report)
        print("[sample outputs]")
        for i, out in enumerate(outputs[: min(2, len(outputs))]):
            preview = out.strip().replace("\n", " ")
            print(f"  req#{i}: {preview}")
