# Gemma4 D/E 缺口与边界覆盖报告

日期：2026-10-20  
设备：RTX 3060 Laptop 6GB，sm_86；torch 2.13.0+cu130；pair_vec8 + CUDA Graph  
状态：**D 的小算子/权重/KV/fallback 已补；E 的 batched prefill、快速反量化、warp-split + 全局 split-K、1k/2k/4k 边界已完成；P1 的 pair_vec8 与 graph 预 warm 已落地**

测试：server **75 passed**，kernels **644 passed**。

## D. 小算子、CPU 调度与 KV

### 已实现

- residual add、layer scale、per-layer gate 乘法原地化，prefill/dense/graph 三条路径一致。
- 融合 qkv/gate-up 权重后把原 projection 指向融合张量的视图，释放重复副本；graph 路径空闲显存 **2.01 → 2.40 GiB**。
- block table 的 int32 CUDA tensor 按 request 缓存，长度不变时不再每步重建。
- 新增 `MINI_LLM_GEMMA4_PAGED=0` 显式 dense fallback gate；修复 prefill dense mirror 语义后，19+160 token 跨容量增长点与 eager **160/160 一致**。
- `reports/gemma4_d_boundaries.json` 覆盖 520 graph、graph 复用、batch=2、forced dense。

### 边界结果

| 场景 | 结果 |
| --- | --- |
| 520-token prompt + 32 decode，graph vs eager | **逐 token 一致**；520 prefill+32 decode 约 **0.56 s** |
| 同一 graph 连续 5 个请求 | 全部完成，显存稳定在 **2.40 GiB free** |
| max_batch_size=2 | 非 graph paged，两个请求都完成 |
| forced dense，19+160 token | **160/160 一致**，跨 128 mirror 增长点；约 5.76 s |
| 4k + short，batch=2 | 全部完成；peak 3.29 GiB，free 1.36 GiB |
| 两个 4k，batch=2，各 8 token | 两个都完成；大 prefill 由 token budget 串行化，peak 3.36 GiB，free 1.20 GiB |
| 2k 长请求后接短请求 | 长 graph 用 split-K；短请求首次 capture 后，稳态 decode 约 **7.75 ms/token**，token ID 稳定 |

### 仍保留

- sampler 的最终 `.item()` 是顺序 decode 的依赖；单步 sampler 约 0.02 ms。
- graph 首次请求 capture、batch>1 graph、dense fallback 速度仍未优化。
- 融合权重仍保留一份融合副本；没有做加载期直接融合布局。

## E. Attention 与 prefill

### batched prefill

调度器的 `max_tokens_per_step` 原先在 benchmark 中设为 1，prompt chunk 被压成 1 token。修正后 19-token prompt prefill 从约 0.70 s 降到 **0.09 s**；继续使用快速反量化后降到 **0.059–0.069 s**。

### 快速反量化 + prefill GEMM

- 新增 INT4→FP16/BF16 dense dequant CUDA kernel，替代逐 projection 的多个 PyTorch 逐元素 kernel。
- `prompt rows >= 8`：反量化一次 + FP16 GEMM；更小 prompt 仍用 GEMV。
- 覆盖 qkv、o_proj、gate/up、down、per-layer gate/projection、PLE projection；lm_head 只处理最后一行，继续 GEMV。

`scripts/bench_gemma4_prefill.py`（warmup 后计时，FP16）：

| prompt tokens | GEMV | 快速反量化 + GEMM | 加速 |
| ---: | ---: | ---: | ---: |
| 16–19 | 0.09 s | **0.059–0.069 s** | 约 1.3–1.5× |
| 512 | 2.251 s | **0.294 s**（1743 tok/s） | 7.66× |
| 1024 | 4.595 s | **0.547 s**（1873 tok/s） | 8.40× |
| 2048 | 9.323 s | **1.042 s**（1966 tok/s） | 8.95× |
| 4096 | 19.327 s | **2.372 s**（1727 tok/s） | 8.15× |

峰值显存（GEMM 路径）：512/1024/2048/4096 token 约 **2.50 / 2.59 / 2.79 / 3.21 GiB**。query-chunked prefill attention 把 4k 峰值从约 4.18 GiB 降到 **3.21 GiB**，4k 运行后空闲显存从约 0.20 提高到约 **2.1 GiB**（prefill-only 进程实测约 1.5 GiB）。

### warp-split 与全局 split-K decode attention

短上下文仍使用 warp-split：

- 一个 block 负责一个 (request, q-head)，8 个 warp 分段扫描 KV；
- 每个 warp 独立维护 online-softmax 的 m/l 和 FP32 输出累加；
- 段结果写 shared，最后一次 barrier 合并；
- 短上下文没有额外 split-K 开销。

长上下文（`context > 1024`）对 full-attention 层使用 **全局 split-K**：

- grid = `(N, Hq, chunks)`，每个 block 只处理一段 KV；
- partial `o/m/l` 写 global scratch；
- 一个 merge kernel 合并所有 chunk；
- scratch 由 engine 预分配，CUDA Graph 中复用；短上下文 graph 不创建/不使用该路径。

`reports/gemma4_e_attention.json`（batch=1，FP16，与 FP32 参考对比）：

| context | D | safe ms | warp ms | 全局 split-K ms | split-K/warp | split vs ref 最大差 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 512 | 0.220 | **0.111** | 0.152 | 0.75×（不用） | 0.00049 |
| 512 | 512 | 0.580 | **0.167** | 0.207 | 0.81×（不用） | 0.00049 |
| 1024 | 512 | 0.841 | 0.270 | **0.164** | 1.38× | 0.00049 |
| 2048 | 512 | 2.127 | 0.502 | **0.190** | 2.75× | 0.00049 |
| 4096 | 512 | 4.180 | 0.915 | **0.270** | 3.29× | 0.00098 |
| 4096 window=512 | 512 | 0.417 | 0.161 | 0.127 | 1.23×（sliding 不用） | 0.00049 |

dispatch 规则：`context > 1024` 且 full-attention 才走全局 split-K；短上下文和 sliding 层保持 warp-split，因此短上下文速度不变。

### 端到端边界

`scripts/check_gemma4_e_boundaries.py`：

| prompt | prefill | elapsed | decode 约 | peak alloc | eager 对齐 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1024 | 0.473 s | 0.615 s | 9.47 ms/token | 2.59 GiB | 是 |
| 2048 | 1.011 s | 1.156 s | 9.67 ms/token | 2.79 GiB | 是 |
| 4096 | 2.427 s | 2.499 s | 10.29 ms/token | 3.21 GiB | 是 |

此前 2k/4k 的 18/29 ms 包含了首次 graph capture；P1 启动预 warm 后，表中是稳态 decode。4k peak 4.18 → **3.21 GiB**。

### 仍保留

- 反量化仍是**每次 prefill 临时**执行，不是持久化权重 cache；4096 prompt 峰值 3.21 GiB，仍不适合与大请求并发。
- 19-token TTFT 约 **60–74 ms**，没有达到 35–50 ms 的理想区间；权重读取/35 层固定成本仍占主导。
- 4k 稳态 decode 约 **10.29 ms/token**；短上下文 128/256 未使用 split-K，仍是最快路径。
- 全局 split-K scratch 与 merge kernel 已实现；不是无 merge 的单 kernel flash-decoding。
- 没有 graph 首请求预捕获、batch>1 graph、Nsight 硬件计数器。

## 关键文件

- `miniservellm/runtime/gemma4_engine_runner.py`：in-place、权重去重、prefill GEMM、chunked prefill attention、warp-split dispatch。
- `miniservellm/cache/gemma4_paged_kv_cache.py`：block table tensor 缓存。
- `miniservellm/runtime/gemma4_runner.py`：`_linear` hook。
- kernels：`csrc/gemma4_attention.cu`（warp-split）、`csrc/int4_matmul.cu`（pair_vec、快速反量化）。
- `scripts/bench_gemma4_prefill.py`、`scripts/bench_gemma4_attention.py`。
- `scripts/check_gemma4_graph_boundaries.py`、`scripts/check_gemma4_e_boundaries.py`、`scripts/check_gemma4_concurrency.py`、`scripts/check_gemma4_dual_graph.py`。
- `reports/gemma4_p1_report.md`：P1-3/P1-4 工作包。
- `tests/test_gemma4_attention_safe.py`、`tests/test_gemma4_prefill_linear.py`。
- `reports/gemma4_d_boundaries.json`、`reports/gemma4_e_*.json`、`reports/gemma4_concurrency.json`、`reports/gemma4_dual_graph.json`。
