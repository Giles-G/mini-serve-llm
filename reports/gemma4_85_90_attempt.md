# 冲 85–90% 的第一轮尝试：结果与瓶颈

日期：2026-10-21  
设备：RTX 3060 Laptop 6GB，Gemma4 E2B INT4 group64  
目标：N=16 per-request 从 89–90 提升到 125–132 tok/s（step 11.1 → 7.6–8.0 ms）

## 实际结果

| N=16 | 开始 | 现在 | 变化 |
| --- | ---: | ---: | ---: |
| 16 输出 per-req | 90.4 | **100.9** | +11.6% |
| 64 输出 per-req | 89.4 | **101.1** | +13.1% |
| 64 输出 e2e | 889.2 | **941.5** | +5.9% |
| 64 输出 step | 11.07 ms | **约 9.89 ms** | -10.7% |
| 相对单请求 | 62% | **约 68.5%** | +6.5pt |

单请求仍 **147.6 tok/s**（64 输出复测），没有回退。

**结论：85–90% 没有达到。** 这轮拿到约 6% 的净收益；剩余差距仍然很大。

## 按计划做了哪些尝试

### 1. pipe cp.async

实现了 2/3/4 stage 的 `cp.async` shared ring buffer：

| shape | 原 pipe | cp.async |
| --- | ---: | ---: |
| gateup | 67.8 µs | 85.5 µs |
| lm_head | 1103.7 µs | 1160.9 µs |

**更慢。** 原因：

- 原 pipe 已经用寄存器预取 2 个 K tile；
- cp.async 把数据搬进 shared 后还要多读一次 512 B；
- 每个 K tile 的 `wait_group` + `__syncwarp` + `__syncthreads` 开销超过了延迟收益；
- 4 warps/block、28 warps/SM 的 occupancy 已经够，问题不是没有异步拷贝。

没有 dispatch。

### 2. tile-major 权重 prepack

把 `[N/2, K]` 重排成 `[n_tile, k_tile, 8, 64]`，让一个 warp 的 32 lane 读连续 512 B：

| shape | 原 pipe | tiled pipe |
| --- | ---: | ---: |
| gateup | 67.1 µs | 63.3 µs |
| lm_head | 1101.3 µs | 1068.1 µs |
| down | 53.3 µs | 116.7 µs |
| qkv | 29.7 µs | 35.1 µs |

只对 gateup/lm_head 有小幅收益（3–6%），但 prepack 需要额外 531 MB 显存；单独上不划算，没有 dispatch。

### 3. qkv/o 2D tile

把两个 N tile 放进同一个 block/warp，复用 A：

| shape | 原 splitk | 2-tile direct |
| --- | ---: | ---: |
| qkv | 27.4 µs | 27.0 µs |
| o | 25.8 µs | 28.7 µs |

qkv 持平，o 更慢。A 复用的收益不足以抵消 block 数/occupancy 下降。没有 dispatch。

### 4. down cp.async / 更多 K split

没有继续做 cp.async；改为 fragment 直接 B + K split：

| down | 时间 |
| --- | ---: |
| 原 two-tile split2 (parts=4) | 50.3 µs |
| frag parts=4 | 48.4 µs |
| **frag parts=8** | **40.8 µs** |
| frag parts=12 | 43.2 µs |
| frag parts=16 | 42.7 µs |

**-19%**，这是本轮唯一的大头，已 dispatch（仅 B=16）。token parity 通过。

### 5. fragment-major prepack + direct B

按 MMA B-fragment 布局离线 prepack，每个 lane 的 16 B 正好包含一个 K tile 的 B 操作数；dequant 后直接进 `mma.sync` 的 B 寄存器，不再写 shared B。

| shape | 原 TC | frag direct B | 结果 |
| --- | ---: | ---: | --- |
| gateup | 68.1 µs | 66.1 µs | -3% |
| lm_head | 1178 µs | 1113 µs | -5.5% |
| down | 50.3 µs | 40.8 µs (parts=8) | -19% |
| qkv | 27.4 µs | 27.0–33 µs | 持平/更慢 |
| o | 25.8 µs | 26–31 µs | 持平/更慢 |
| pl_proj | 18.1 µs | 12.5 µs | -31% |

已 dispatch：

- down：`[768, 6144]`，parts=8；
- per_layer_projection：`[768, 256]`，parts=1；
- 仅 `rows == 16`，B<16 仍走原路径。

显存代价：down 165 MB + pl_proj 7 MB prepack，可接受。

### 6. add + scalar 融合

`(a + b) * layer_scalar` 合成一个 kernel，bitwise 与 eager 一致。省 35 次 launch，但对 step 的实测影响在噪声内（<0.05 ms）。

## 追加的落地项（本轮后半段）

0. **gateup / lm_head 也接入 frag direct-B**  
   normal/wide gateup + lm_head 全部 prepack；gateup -3~4%，lm_head -5~6%。代价是额外约 731 MB prepack 显存。

1. **double-wide down 也走 direct-B**  
   15 个 `[768,6144]` + 20 个 `[768,12288]` 都使用 frag parts=8。  
   wide down：88.6 → **68.8 µs（-22%）**；N=16 step 从 10.9 降到约 10.6 ms。

2. **pl_proj 改用 global-A + direct-B**  
   `[768,256]`：19.2 → **12.6 µs（-34%）**。

3. **q/k/v RMSNorm + RoPE 合并（默认关闭）**  
   `gemma4_qkv_norm_rope` 每个 layer 把 5 个 kernel 合成 1 个；bitwise 对齐 eager（D=512 的 v_norm 有 1 个 1-ulp 差异，token parity 通过）。  
   但 A/B 实测 step 反而略慢（per-req 97.1 → 94.6），因此默认关闭，保留为可选实验路径。

4. **L2::128B load hint**  
   给 packed B load 加 `ld.global.L2::128B`，对 lm_head 有小幅正收益，保留。

5. **fragment dequant 的 scale-FMA 尝试失败**  
   把 `(n-8)*scale` 合成一条 HFMA2 会产生灾难性抵消，误差最大 0.7；已回退为“先精确得到 n-8，再乘 scale”。

当前 N=16 实测：

- 16 输出：**417.8 e2e / 1613.9 decode / 100.9 per-req**
- 64 输出：**941.5 e2e / 1616.9 decode / 101.1 per-req**
- 单请求：约 147 tok/s
- 相对单请求：**约 68–69%**
- 全部 frag prepack 后 free VRAM 约 **0.646 GiB**（偏紧但可运行）
- frag 8-warp 版本在 lm_head / wide gateup 上比 4-warp 快 3–7%，已 dispatch

## 为什么还是 62%

当前 kernel 的属性：

| kernel | regs | shared | blocks/SM | warps/SM |
| --- | ---: | ---: | ---: | ---: |
| pipe | 72 | 11.5 KB | 7 | 28 |
| splitk | 56 | 13.3 KB | 7 | 28 |
| split2 | 72 | 13.3 KB | 7 | 28 |
| mt2 | 95 | 22.5 KB | 4 | 16 |
| frag | 62 | 2.3 KB | 8 | 32 |

occupancy 不是瓶颈。frag 已经 32 warps/SM，仍然只比原 pipe 快 3–5%。

当前有效权重带宽：

| shape | 有效带宽 |
| --- | ---: |
| gateup | 140–150 GB/s |
| lm_head | 180–188 GB/s |
| down | 92–117 GB/s |
| qkv/o | 60–75 GB/s |
| 单请求 pair_vec8 | 约 184 GB/s |

单请求 GEMV 也只有约 184 GB/s，并不是 330 GB/s 峰值。所以 batch TC 并不是“离峰值很远”，而是和单请求同处一个 **instruction/latency 受限** 的区间；tensor-core 的 dequant + shared staging + A 读放大抵消了 M=16 的复用。

非 linear 部分约 3.3 ms（norm/attention/elementwise/graph gap），单请求约 1.4 ms；即使 linear 完全追平单请求，step 也只能到约 8.2 ms（约 85%），但需要 linear 再快 30% 以上。

## 下一步判断

这轮验证了一个重要事实：

> 85–90% 不是靠 cp.async、warp 数、tile 大小、prepack 布局这些“常规参数优化”能拿到的；当前 kernel 已经在 28–32 warps/SM、180 GB/s 量级，继续微调只会得到个位数百分比。

要真正冲 80–90%，需要换实现路线，优先级：

1. **swizzled shared + `ldmatrix` 自定义 MMA**  
   消除 shared B/A 的 bank conflict 和 WMMA load 开销；预期 linear -15~25%。

2. **weight-stationary SIMT 16-row kernel**  
   每个权重只读一次，用 16 个寄存器累加 16 行；理论 memory-bound 2.5–3 ms，但 instruction 压力大，需要实测。

3. **把 non-linear 压到 2 ms 以内**  
   q/k/v norm + RoPE 合并成一个 kernel、residual add+scalar 已做、减少 graph 节点；预期 -0.4~0.7 ms。

4. **重新评估 85% 目标**  
   以当前 GPU 和 INT4 dequant 路线，合理工程上限可能是 **72–78%**；80% 需要第 1 或第 2 项成功；85–90% 需要多项同时成立。

## 当前状态

- N=16 16 输出：**411.0 e2e / 1472.4 decode / 92.0 per-req**
- N=16 64 输出：**904.5 e2e / 1462.3 decode / 91.4 per-req**
- N=4/8/16 64 输出：**312.5 / 548.9 / 904.5**
- 单请求：**147.6 tok/s**，未回退
- token parity：N=2/4/8/16 全部 true
- kernel tests：**688 passed**
- server tests：**75 passed**
- 未 commit、未 push
