# Gemma4 P2-P5 evidence

Generated: 2026-10-21 16:24:27

## P2 cache
- batch/paged/KV-sharing token parity: **PASS**
- batches: [2, 4, 8, 16]
- graph keys: [{'batch': 2, 'graph_keys': [[2, False]]}, {'batch': 4, 'graph_keys': [[2, False], [2, True], [3, False], [4, False], [4, True]]}, {'batch': 8, 'graph_keys': [[2, False], [2, True], [3, False], [4, False], [4, True], [5, False], [6, False], [7, False], [8, False], [8, True]]}, {'batch': 16, 'graph_keys': [[2, False], [2, True], [3, False], [4, False], [4, True], [5, False], [6, False], [7, False], [8, False], [8, True], [9, False], [10, False], [11, False], [12, False], [13, False], [14, False], [15, False], [16, False], [16, True]]}]
- prefix-cache hit (N=4, 64 out): TTFT **0.009 s**, e2e **376.1 tok/s**, per-req **93.8**

## P3 low memory
- format: INT4 group64, KV fp16
- N=16 graph VRAM: {'n': 16, 'free_gib_after_startup': 0.646, 'batch_graphs': 19, 'frag_prepack_mb': 996}
- prefill 2048: 0.808 s, 2535.9 tok/s, peak 2.7 GiB
- prefill 4096: 1.646 s, 2488.7 tok/s, peak 3.02 GiB

## P4 CUDA
- server tests: **75 passed**
- kernel tests: **688 passed**

## P5 graph
- graph-vs-eager parity: **PASS**
- short+long graphs present: True

## Commands
- `/home/giles/work/mini-serve-llm/venv/bin/python3 -m pytest tests -q --ignore=tests/test_gemma4_mlx.py` → rc=0, 5.7 s
- `/home/giles/work/mini-serve-llm/venv/bin/python3 -m pytest tests -q` → rc=0, 8.4 s
- `/home/giles/work/mini-serve-llm/venv/bin/python3 scripts/check_gemma4_batch_decode.py` → rc=0, 60.3 s
- `/home/giles/work/mini-serve-llm/venv/bin/python3 scripts/bench_gemma4_prefill.py --sizes 2048 4096 --gemm-min-rows 8 --gemv-min-rows 100000` → rc=0, 81.2 s
- `/home/giles/work/mini-serve-llm/venv/bin/python3 scripts/bench_gemma4_batch.py --batch-sizes 4 --max-new-tokens 64 --output-json /home/giles/work/mini-serve-llm/reports/gemma4_prefix_cache_bench.json` → rc=0, 71.9 s
