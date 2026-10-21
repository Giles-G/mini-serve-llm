# Gemma4 E2B 多请求 decode 衰减分析

日期：2026-10-20  

> 后续落地（同日）：已按修订方案实现 tensor-core B=8/16、batch graph 8/16、cross-request batched prefill、batch 4k split-K。最新数据见 `reports/gemma4_batch_optimized.md`；本文保留的是 N=4 `pair_batch` 的历史 profile 与归因，N=4 仍未达到 3–4×。
设备：RTX 3060 Laptop 6GB；prompt 19 token；短上下文 decode  
目的：解释为什么项目 N=4 的 per-request decode 从约 140 tok/s 掉到约 56 tok/s，而 Ollama 表面上接近 4×。

## 结论

1. **Ollama 的 “4× decode” 不是有效的 4-row 真并发数字。** 它的 per-request decode 在 N=1/2/4 都保持在约 100–106 tok/s，没有变快；`aggregate_decode_tok_s` 用 `wall - max(TTFT)` 作分母，N>1 时各请求 TTFT 明显错开，早完成的请求也被算进分子，数字被高估。公平看 e2e：16 输出 N=4 只有单请求的 **2.07×**，64 输出只有 **1.38×**，不是 4×。
2. **项目的衰减是真实存在的，而且主要不在 scheduler/graph，而在 batch linear kernel。** N=4 decode 的 GPU 时间里 **约 88% 是 `int4_dequant_matmul_pair_batch_kernel`**；attention 只占约 1%，norm/elementwise 约 12%。batch graph 已经生效：无 graph 的 N=4 decode 约 127 tok/s，有 graph 约 225 tok/s；graph 去掉了 launch/CPU 间隙，但不能减少 kernel 内部算术。
3. **`pair_batch` 把 batch 维串行在一个 warp 里。** 一个 warp 负责一对输出行，然后 `for b in 0..B` 依次算 B 个请求；grid 和 B=1 时几乎一样，batch 维没有并行化。它确实复用了权重字节，但每个额外请求仍然增加几乎一份 dequant + FMA 工作。所以 4 行只换来约 1.4–1.6× 总吞吐，per-request 掉到约 1/2.5。
4. **单请求 kernel 的优化不能直接迁移到多请求。** `pair_vec8` 是单行专用 GEMV，已经把单请求做到接近权重带宽/指令开销的边界；多请求需要的是 **tile 级 dequant + tensor-core GEMM（MMQ 类）**，让 4/8/16 行共享权重并并行做 MMA，而不是在 CUDA core 上把同一套 GEMV 循环 4 次。
5. **短输出 e2e 还叠加了 prefill 串行化。** 项目 N=4 的 last prefill 约 0.424 s，N=1 约 0.065 s；16 输出时 prefill 占总 wall 的一半以上。当前没有 cross-request batched prefill，4 个 prefill 基本是串行/低重叠的。64 输出时 decode 占比更大，主瓶颈回到 batch linear。
6. 因此，“之前主要优化单请求”这个判断 **基本正确，但不够精确**：多请求已经做了 batch graph、`pair_batch`、batch sampler，问题是核心线性层仍然是 GEMV 思路，只做了权重复用，没有做计算并行/tensor-core 复用。

## 1. 先纠正 Ollama 的 4×

Ollama 原始数据（16 输出，单轮）：

| 请求 | TTFT | eval 时长 | per-request decode |
| ---: | ---: | ---: | ---: |
| 0 | 0.860 s | 0.147 s | 101.8 tok/s |
| 1 | 0.494 s | 0.150 s | 100.2 tok/s |
| 2 | 1.043 s | 0.149 s | 100.7 tok/s |
| 3 | 0.677 s | 0.150 s | 100.0 tok/s |

四个请求的 per-request decode 都是约 100 tok/s，和 N=1 的 102.9 几乎一样。TTFT 从 0.49 s 错开到 1.04 s；最后一个请求开始时，前面的请求有的已经结束。`aggregate_decode_tok_s = 60 / (wall - max_ttft)` 会把 60 个 token 都放进最后一个请求开始后的窗口，得到 403 tok/s，但这不是四个请求同时以 4× 速度生成。

公平的 e2e 扩展：

| 输出 | N=1 | N=2 | N=4 | N=4 / N=1 |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 25.7 | 39.5 | 53.2 | **2.07×** |
| 64 | 61.0 | 73.4 | 84.3 | **1.38×** |

所以 Ollama 也有明显衰减，只是它的 per-request decode 不变、e2e 仍有 1.4–2.1× 提升；这不是“4 个请求等于 4 倍单请求”。

## 2. 项目 N=1 vs N=4 的 profile

`torch.profiler`，每个配置 profile 3 个 decode step，所有 prefill 已完成，19-token 上下文。绝对时间受 profiler 影响略高，但占比和比例与 bench 一致。

| 指标 | N=1 | N=4 | N=4 / N=1 |
| --- | ---: | ---: | ---: |
| 总 GPU 时间 / 3 step | 18.25 ms | 51.15 ms | 2.80× |
| 每 step GPU 时间 | 6.08 ms | 17.05 ms | 2.80× |
| linear kernel / 3 step | 13.40 ms | 44.92 ms | 3.35× |
| linear 占 GPU 时间 | 73.4% | 87.8% | — |
| attention / 3 step | 0.55 ms | 0.65 ms | 1.18× |
| norm / 3 step | 1.66 ms | 1.65 ms | 1.00× |
| elementwise / 3 step | 2.59 ms | 3.76 ms | 1.45× |

关键点：

- N=1 的 linear 是 `int4_dequant_matmul_pair_vec8_kernel`，每 step 约 4.47 ms。
- N=4 的 linear 是 `int4_dequant_matmul_pair_batch_kernel`，每 step 约 14.97 ms。
- N=4 的 pair_batch 每行成本 = 14.97 / 4 ≈ **3.74 ms/row**，相比 N=1 的 4.47 ms/row 只好了约 **16%**。
- 也就是说，权重复用确实发生了一点，但 4 行没有变成“几乎免费”；step 延迟从 6.08 ms 涨到 17.05 ms，聚合吞吐只有 4 / 2.80 ≈ **1.43×**。
- attention 在这个短上下文几乎不变；所以不是 attention、不是 graph、也不是 sampler。

## 3. 根因：`pair_batch` 把 batch 维串行化

`mini-llm-kernels/csrc/int4_matmul.cu` 里的 `int4_dequant_matmul_pair_batch_kernel`：

- 一个 warp 仍然只负责一对输出行 `pair_idx`；
- 每个 lane 读一次 packed weight；
- 然后在一个运行时循环 `for (int b = 0; b < B; ++b)` 里，对 B 个请求各做一遍 8 个 nibble 的解包、x 读取、16 次 FMA；
- grid 大小只由 `n_pairs` 决定，和 B 基本无关。

这意味着：

- **权重带宽被摊薄了**：同一份 weight word 被 B 行复用；
- **但算术没有并行**：B 行在同一个 warp 里串行执行，warp 数没有增加；
- 单行 kernel 本来就已经有很高的 byte unpack/scale/FMA 指令密度，所以额外 3 行几乎就是额外 3 份指令；
- 结果就是“权重读一次，但 CUDA core 算 4 次”，延迟接近线性增长。

微基准也支持这个判断（`N=12288, K=1536`）：

| B | 总时间 | 每行时间 |
| ---: | ---: | ---: |
| 1（pair_vec8） | 42.5 µs | 42.5 µs |
| 2 | 94.9 µs | 47.4 µs |
| 3 | 123.6 µs | 41.2 µs |
| 4 | 154.1 µs | 38.5 µs |

B=4 的每行只比 B=1 好约 9%，总时间却是 3.6×。`lm_head` 更明显：B=1 约 745 µs，B=4 约 2647 µs，每行只好了约 11%。

这不是说 `pair_batch` 写错了；它是在“权重复用”和“batch 并行”之间选择了前者。对 6GB / 单卡 decode，这个选择在 B=2 还有点收益，但 B=4 已经把 CUDA core 算术推到瓶颈。

## 4. 为什么 Ollama 看起来更好

Ollama 的 per-request decode 在 N=1/2/4 都约 100 tok/s，说明它并没有让单个请求变快；它更像是多个请求排队/部分重叠执行，e2e 靠请求重叠提升，而不是把 4 行做成一个 4× 的 GEMV。

即使不讨论它的调度，llama.cpp 的 batch>1 路径通常会走 quantized matmul / MMQ 类的批量 GEMM，用 tensor cores 处理 M 维；单请求则走 GEMV。这样 4 行可以共享权重并并行做 MMA，额外行的代价远小于“再跑一遍 GEMV”。这与项目 `pair_batch` 的 CUDA-core 串行 B 循环是两种计算模型。

## 5. 另一个独立瓶颈：prefill 没有跨请求 batch

项目 N=4、16 输出的时间构成：

| 指标 | N=1 | N=4 |
| --- | ---: | ---: |
| wall | 0.168 s | 0.690 s |
| last prefill | 0.065 s | 0.424 s |
| decode | 0.102 s | 0.266 s |
| e2e | 95.5 tok/s | 92.8 tok/s |

4 个请求的 prefill 不是一次 batched prefill，而是近似串行/低重叠，last prefill 从 0.065 s 涨到 0.424 s。16 输出时这部分比 decode 还大，所以 e2e 几乎不随并发提升；64 输出时 decode 占比更大，e2e 也只从 127.5 提升到 149.0（1.17×）。

Ollama 的 TTFT 也错开（0.49–1.04 s），所以这不是项目独有的问题；但项目单请求 prefill 太快（约 0.06 s），4 个串行后变得显眼。要做多请求 e2e，cross-request batched prefill 或 prefill/decode overlap 必须一起做。

## 6. 优化优先级

### P0：把 batch linear 换成 tensor-core 路径

目标：N=4 的 step 时间从 17 ms 降到接近 N=1 的 7–9 ms，聚合吞吐接近 3–4×。

建议：

- 按 tile 加载 INT4 weight + scales 到 shared memory；
- 在 shared memory 里 dequant 成 FP16；
- 用 `mma.sync`（m16n8k16）做 FP16×FP16 + FP32 accumulate；
- M 维用 batch 行填充；N=4 时补到 16 会有 75% 浪费，最好同时支持 N=8/16；
- 复用现有 prefill 的 INT4→FP16 数值路径和 parity 测试，保持 FP16 激活、FP32 累加，不放宽阈值；
- 不要每 token 全量 dequant 权重；按 tile 流式 dequant，权重只读一遍。

这是唯一有机会接近“4 行几乎免费”的路线。只把 `pair_batch` 的 B 循环模板化/unroll，预计只有 10–20%，不够。

### P0/P1：batch graph 扩到 8/16

当前 batch graph 只捕获 2/3/4；N>4 回退到非 graph。tensor-core kernel 的 M=16 tile 需要更大 batch 才不浪费；建议 graph 覆盖 2/4/8/16，并保留 N=4 的 padding 路径。

### P1：cross-request batched prefill

把多个请求的 prefill token 拼成一个 batch，或至少让 prefill 和 decode 重叠；否则 16 输出的 e2e 仍会被 4 次串行 prefill 拖住。

### P1：长上下文 batch attention

batch graph 目前用 warp-split attention，没有走 global split-K；context>1024 时需要给 batch 也接 split-K，否则长上下文会叠加新的衰减。

### P2：短期微优化

- `pair_batch` 按 B=2/3/4 做模板特化 + `#pragma unroll`；
- x 读取用 `half2`/更宽的向量 load；
- 减少 scale/byte unpack 的重复计算。

这些可以作为过渡，但不要把目标定成 4×；它们解决不了 CUDA-core 算术瓶颈。

## 附：口径说明

- 项目 batch bench 的 decode 窗口从“最后一个 prefill 完成”开始，四个请求都在生成，因此项目的 `aggregate_tok_s` 是同一批 4 行的真实稳态吞吐。
- Ollama 的 `aggregate_decode_tok_s` 是 `wall - max(TTFT)`，早完成请求的 token 仍计入分子；N>1 时会高估，不能与项目的 batch decode 直接对比。
- 公平对比应看 e2e，或做一个所有请求严格同步、同一时刻开始 decode 的客户端；Ollama 当前 API 不能保证这一点。
