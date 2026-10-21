# Gemma4 E2B 多请求 decode：第二轮 kernel 优化

日期：2026-10-21  
设备：RTX 3060 Laptop 6GB，sm_86，INT4 group64，FP16 激活 / FP32 累加  
模型：`models/gemma-4-E2B-it`，19-token prompt，greedy decode

## 结果摘要

| N=16 | 优化前 | 现在 | 变化 |
| --- | ---: | ---: | ---: |
| 16 输出 e2e | 367.2 tok/s | **395.5** | +7.7% |
| 16 输出 per-request | 76.7 tok/s | **90.4** | +17.9% |
| 64 输出 e2e | 784.0 tok/s | **889.2** | +13.4% |
| 64 输出 per-request | 78.3 tok/s | **89.4** | +14.2% |
| N=16 step | 12.8 ms | **约 11.1 ms** | -13% |

单请求仍约 **147 tok/s**；N=2–16 的 per-request 从 75–82 提升到 **80–92 tok/s**。

> 后续又尝试了 cp.async、tile-major prepack、qkv/o 2D tile 和 fragment-major direct-B；只有 down/pl_proj 的 direct-B 落地，N=16 进一步到 **96.6/94.6**（16/64 输出）。详见 `reports/gemma4_85_90_attempt.md`。

## 本轮改动

### 1. nibble → half：`prmt` + half2

旧路径每个 nibble 做整数取位、`I2F`、FP32 乘 scale、`F2F` pack。新路径利用

```text
half_bits(1 + n/1024) = 0x3C00 | n
```

用一条 `prmt` 从 4 个 nibble 生成两个这样的 half，再用一条 half2 FMA 得到精确的 `n-8`，最后乘 half scale：

- 每 2 个权重从约 5 条标量指令降到约 1 条 `prmt` + 1 条 `HFMA2`/`HMUL2`；
- 对 100 万随机值（含负 scale）与原实现 **bitwise 相同**；
- 所有 TC kernel 共用，gateup/lm_head 约 **-10~15%**。

### 2. pipe 直接写 FP16

原 pipe kernel 把 16×N 的 FP32 结果写到全局 `tmp`，再跑一个 `tc_f16_convert_kernel`。现在：

- 最后一个 K tile 后 sB 已无用，复用它作为每 warp 1 KiB 的 FP32 accumulator staging；
- 直接 `__floats2half2_rn` 写 FP16 输出；
- 去掉 37 次/步的 convert kernel 和 16×N 的 FP32 往返。

gateup：**77 → 63 µs**；lm_head：**1270 → 约 1083 µs**。

### 3. down_proj：每 warp 两个 N tile

down 的 A 是 16×6144，每个输出 tile 都要读一遍，A 流量约为权重的 4 倍。新 kernel 让每个 warp 同时处理两个相邻的 16-column tile：

- 每个 `kk` 只 load 一次 A fragment，两个 tile 复用它；
- 两个 tile 的 packed B 各自预取、各自 dequant；
- 仍保留 4 个 K 分区 block + FP32 reduce，reduce 顺序不变，token parity 保持。

down：**62 → 51 µs**（每层），35 层约省 **0.38 ms/step**。

### 4. 融合 RoPE

`mul + addcmul + addcmul` 三个 kernel 合成一个 `gemma4_rope_fused`：

- 每个线程处理一对 `(i, i+D/2)`，按 eager 顺序做两次 FP16 rounding；
- 每步减少约 **100 次 kernel launch**；
- 对 16×8×256、16×1×256、16×8×512 均与 eager **bitwise 相同**。

### 5. 融合 GELU×up

`gelu(gate) * up` 合成一个 kernel：

- 用 libdevice `__nv_tanhf`，避免 `--use_fast_math` 的 `MUFU.TANH` 造成约 0.6% 的 1-ulp 差异；
- GELU 先 round 到 FP16，再 half2 乘 up，匹配 eager 的两次 rounding；
- 对连续和 strided 的 up 均 **bitwise 相同**；
- 每层少一次 launch，MLP 激活读写少一次。

## 最终吞吐

### 16 输出

| N | e2e tok/s | decode tok/s | per-request |
| ---: | ---: | ---: | ---: |
| 2 | 117.9 | 159.6 | 79.8 |
| 3 | 175.2 | 275.1 | 91.7 |
| 4 | 216.2 | 369.9 | 92.5 |
| 8 | 295.7 | 706.4 | 88.3 |
| 16 | **395.5** | **1445.8** | **90.4** |

### 64 输出

| N | e2e tok/s | decode tok/s | per-request |
| ---: | ---: | ---: | ---: |
| 2 | 146.7 | 159.6 | 79.8 |
| 3 | 240.3 | 272.2 | 90.7 |
| 4 | 310.3 | 364.6 | 91.2 |
| 8 | 533.9 | 704.2 | 88.0 |
| 16 | **889.2** | **1430.3** | **89.4** |

数据：`reports/gemma4_batch_benchmark_round2_16.json`、`..._64.json`。

## 验证

- token parity：N=2/4/8/16 全部 `true`（`reports/gemma4_batch_parity.json`）；
- kernel 测试：**688 passed**（含新增 fused RoPE/GELU 4 项）；
- server 测试：**75 passed**（排除未安装的 MLX）；
- 未 commit、未 push。

## 剩余瓶颈

N=16 每步约 11.1 ms：

| 部分 | 约 |
| --- | ---: |
| gateup + lm_head pipe | 3.3 ms |
| down（two-tile split2） | 1.8 ms |
| qkv/o/小矩阵 | 1.8 ms |
| norm/attention/elementwise/host | 4.0 ms |

要到单请求的 140 tok/s，需要 step ≈ 7 ms。下一步收益最高的是：

1. qkv/o 的 A 复用或 split-K 变体，降低固定延迟；
2. gateup/lm_head 的 `cp.async` producer-consumer，进一步隐藏 shared staging；
3. norm+RoPE 融合、residual add+scalar 融合，继续压缩 kernel 数与 host 间隙。
