# Tensor-core batch linear 原型结果

日期：2026-10-20  
设备：RTX 3060 Laptop 6GB，sm_86，FP16 激活，INT4 group64  
目的：验证 P0 的 tensor-core tile-dequant batch linear 能否把 N=4 decode step 从约 17 ms 拉到 7–9 ms。

## 实现

在 `mini-llm-kernels/csrc/int4_matmul.cu` 新增两条路径：

1. `int4_dequant_matmul_tc`：WMMA 16x16x16，K tile 64，tile 级 dequant 到 shared memory，FP16×FP16 + FP32 accumulate，支持 B=2..16。
2. `int4_dequant_matmul_pair_batch_parallel`：把 batch 行映射到 warp，block 内共享 4 个 packed pair，用于验证“并行 B”能否替代串行 `pair_batch`。

两条路径都保持：

- INT4 权重只读一遍；
- FP16 激活/权重；
- FP32 累加；
- 与 `pair_batch` 的 token 输出逐元素对比。

## Parity

随机权重/激活，`pair_batch` 作为参考：

| shape | B | max abs diff |
| --- | ---: | ---: |
| N=64, K=64 | 2/4 | 6e-5 / 1.2e-4 |
| N=1536, K=1536 | 2/4 | 4.9e-4 |
| N=1536, K=6144 | 2/4 | 9.8e-4 |
| N=12288, K=1536 | 2/4 | 4.9e-4 |

`pair_batch_parallel` 与 `pair_batch` 逐元素完全一致（0 diff）。

## Microbench

单位 µs，单次 kernel；B=1 参考为 `pair_vec8`。

### N=12288, K=1536

| B | pair_vec8 / pair_batch | tensor-core |
| ---: | ---: | ---: |
| 1 | 42.8 | — |
| 2 | 95.0 | 165.9 |
| 4 | 154.4 | 166.7 |
| 8 | — | 168.5 |
| 16 | — | 172.1 |

### N=1536, K=6144

| B | pair_vec8 / pair_batch | tensor-core |
| ---: | ---: | ---: |
| 1 | 28.3 | — |
| 2 | 44.4 | 142.7 |
| 4 | 66.4 | 138.3 |
| 8 | — | 138.8 |
| 16 | — | 165.7 |

### N=2048, K=1536

| B | pair_vec8 / pair_batch | tensor-core |
| ---: | ---: | ---: |
| 1 | 12.5 | — |
| 2 | 16.6 | 40.7 |
| 4 | 22.6 | 41.2 |
| 8 | — | 40.8 |
| 16 | — | 42.8 |

### lm_head，N=262144, K=1536

| B | pair_vec8 / pair_batch | tensor-core |
| ---: | ---: | ---: |
| 1 | 730.3 | — |
| 4 | 2780.5 | 2554.9 |
| 8 | — | 2626.5 |
| 16 | — | 2727.8 |

## 结论

1. **TC 的数值路径没问题**：与 `pair_batch` 的误差在 1e-3 以内，FP16 权重 + FP32 累加保持不变。
2. **N=4 没有达到目标**：
   - `N=12288, K=1536`：TC B=4 166.7 µs vs pair_batch 154.4 µs，反而略慢；
   - `N=1536, K=6144`：TC 138.3 µs vs pair_batch 66.4 µs，明显更慢；
   - `N=2048, K=1536`：TC 41.2 µs vs pair_batch 22.6 µs，明显更慢。
   只有 `lm_head` 这种 N 很大的 shape，TC B=4 才略快（2555 vs 2781 µs）。
3. **原因是 WMMA 的 M 粒度**：WMMA 最小 M=16；B=4 时 12 个 M 行是 padding，有效利用率只有 25%。单请求 GEMV 已经把 3060 的 CUDA core/指令吞吐压满，tensor-core 的 4× 算力被 padding 和 shared dequant 开销抵消。
4. **N=8/16 才是 tensor-core 的甜点**：
   - N=12288 B=16 172 µs，B=4 167 µs：16 行几乎和 4 行同价；
   - N=6144 B=16 78.5 µs，B=4 96.6 µs：16 行反而更快；
   - 因此 batch=8/16 可以把 aggregate 推到约 2–4×，但 N=4 不行。
5. **“并行 B” GEMV 也救不了 N=4**：`pair_batch_parallel` 把 B 行映射到 warp，但每个 warp 的工作量和总计算量不变；B=4 约 168 µs，和 `pair_batch` 的 154–177 µs 同一水平。单行 kernel 已经占满 GPU，单纯并行 batch 行不会产生额外吞吐。

## 建议的修订方案

- **不要把 N=4 的 3–4× 作为硬目标**：在 M=16 tensor core 上，N=4 最多只能做到接近单行时间（4 行 / 4× 单行时间 ≈ 1× aggregate），不可能接近 4×。
- **把 N=8/16 作为 batch linear 的主目标**：
  - B=8/16 用 tensor-core path；
  - B=2/4 继续用 `pair_batch`（或只在 lm_head 用 TC）；
  - batch graph 扩到 8/16，scheduler `max_batch_size` 支持 16；
  - 实测目标改为：N=16 aggregate 接近 3–4×，N=8 接近 2–3×。
- **N=4 想再提升**，可选：
  - WMMA M=8/N=32：把有效利用率从 25% 提到 50%，理论上最多约 2×；但仍受 shared dequant/occupancy 限制；
  - 或者接受 N=4 维持现状，把 4 个请求与别的请求/预填 token 合批，填满 M=16。
- **cross-request prefill 和 batch split-K 仍然独立且值得做**，不依赖 N=4 的 tensor-core 结果；建议在 batch 8/16 落地后再做。

## 已确认并落地

- 采纳“N=8/16 为主、N=4 保持 pair_batch”的修订目标。
- B=8/16 已接入 tensor-core path；batch graph 已扩到 8/16。
- cross-request batched prefill 和 batch 4k split-K 已实现。
- 最新端到端数据见 `reports/gemma4_batch_optimized.md`：
  - N=16、16 输出：e2e 395.5 tok/s，decode 1445.8 tok/s；
  - N=16、64 输出：e2e 889.2 tok/s，decode 1430.3 tok/s；
  - N=4 仍走 pair_batch，不做 M=8/N=32。

## 后续 kernel 优化（同日）

- dequant 改为 `half2` 计算 + 每行两个 `uint4` 写 shared；
- B=16 增加 split2（4 个 K 分区 block + FP32 reduce）、splitK（4 warp K split）和 pipelined 大 N 路径；
- batch graph 的 16 次逐字段 device 写改成每个字段一次 `copy_`；
- 结果（后续第二轮 kernel 优化后）：N=16、64 输出 per-request decode 从 32.9 提升到 **89.4 tok/s**，decode 从 526 提升到 **1430.3 tok/s**；仍未达到单请求约 147 tok/s。
