# Gemma4 E2B INT4 N=16 decode roofline

日期：2026-10-21  
设备：RTX 3060 Laptop 6GB  
模型：Gemma4 E2B text 4.647B，INT4 group64，FP16 激活，FP32 累加

## 实测硬件上限

| 项目 | 实测 |
| --- | ---: |
| FP16 tensor-core（8192³ GEMM） | **26.0 TFLOPS** |
| FP16 tensor-core（4096³ GEMM） | 20.1 TFLOPS |
| device-to-device copy 有效带宽 | **约 295 GB/s** |

## N=16 的计算下限

每 token 约 `2 × 4.647B = 9.29 GFLOP`；N=16 时：

```text
148.7 GFLOP / 26.0 TFLOPS = 5.72 ms
```

这是 **linear 部分的纯 tensor-core 计算下限**，不含 dequant、A staging、norm、attention、sampling。

当前 N=16 的 linear 约 **6.45 ms**，已经达到峰值的 **89%**。linear 最多还能省 **0.7 ms**。

## 非 linear 部分

当前约 **2.5–2.8 ms**：

| 部分 | 约 |
| --- | ---: |
| norms | 0.60 ms |
| attention | 0.31 ms |
| elementwise / RoPE / copy / index | 0.7–0.9 ms |
| graph gap / launch | 1.2–1.6 ms |

即使把非 linear 压到 2.0 ms（非常激进），N=16 的 step 下限约：

```text
5.72 + 2.0 = 7.72 ms
per-request = 1 / 7.72ms ≈ 129.5 tok/s
```

相对单请求 147.6，约 **88%**。这是**所有东西都接近完美**的理论值；实际还要加上 dequant、A 读写、kernel 无法完全重叠的部分。

如果非 linear 保持 2.3–2.5 ms：

```text
5.72 + 2.4 = 8.12 ms
per-request ≈ 123 tok/s ≈ 83.5%
```

## 对照目标

| 目标 | 需要的 step | 相对计算下限 |
| --- | ---: | --- |
| 85% | 8.0–8.3 ms | 线性必须 5.7–5.9 ms，非 linear 2.2–2.4 ms |
| 90% | 7.56–7.8 ms | 线性 5.72 ms（100% 峰值），非 linear 1.8–2.1 ms |

结论：

- **85% 是当前硬件 roofline 的边缘**：要求 linear 接近 100% tensor-core 峰值，同时非 linear 压到 2.2–2.4 ms。
- **90% 基本低于 roofline**：7.56 ms 只剩 1.84 ms 给所有非 linear，而当前光 graph gap + norms + attention 就超过 2 ms。
- 当前 68–69%（约 100 tok/s）离 85% 还差约 2 ms；其中 linear 最多只能再给 0.7 ms，剩余必须从 graph gap/非 linear 中拿 1.3 ms 以上，难度极高。

## 当前实现位置

- linear：6.45 ms，**约为 26 TFLOPS 峰值的 89%**
- 非 linear + gap：约 2.5–2.8 ms
- step：约 9.9–10.1 ms
- per-request：约 99–101 tok/s，单请求约 147.6，比例 **67–69%**

当前 kernel 已经不是在“浪费 tensor core”；它离计算峰值只差约 11%。继续用同样的 WMMA/dequant 结构，最乐观也只能到 **80–83%**，而且需要非 linear 同时下降 0.5 ms 以上。

## Swizzled ldmatrix 实测结果

已经实现并验证了 8x8 packed A + `ldmatrix.x4`，结果与 frag 路径 bitwise 相同，但交错 A/B 没有收益（down 40.6 vs 40.8 µs，wide down 68.6 vs 68.3 µs，gateup/lm 反而略慢）。说明 A 的 shared load 不是瓶颈，剩余成本在 B dequant/MMA、非 linear 和 graph gap。

## 建议

- 如果验收目标是 **85–90%**：在当前 GPU/FP16-FP32accum/INT4 dequant 路线下，这个目标位于 roofline 边缘或之外，不建议继续投入。
- 更现实的目标是 **75–80%**：需要 swizzled `ldmatrix`/更好的 A 复用 + 非 linear 再压 0.5 ms。
- 或者接受当前 **68–69%** 作为这条实现路线的结果。
