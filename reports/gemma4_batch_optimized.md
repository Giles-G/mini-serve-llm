# Gemma4 E2B 多请求优化落地结果

日期：2026-10-21  
设备：RTX 3060 Laptop 6GB，sm_86，INT4 group64，FP16 激活  
prompt：19 token；decode batch 2/4/8/16；短上下文

## 实现

### 1. Tensor-core batch linear

新增 `int4_dequant_matmul_tc`：

- WMMA `16x16x16`，K tile 64；
- 每个 K tile 只把 16x64 的 INT4 weight + scales dequant 到 shared memory；
- FP16×FP16，FP32 accumulate；
- M 维 padding 到 16，支持 B=2..16；
- shared 行加 8 个 half 的 padding，避免 WMMA load 的 bank conflict；
- **dequant 使用 `half2` 计算 + `uint4` 写 shared**，把每线程 16 次标量 half 写降到 4 次 128-bit 写；
- **nibble→half 用 `prmt` 一次生成两个 half**：利用 `0x3C00 | n = 1 + n/1024`，一条 FMA 得到 `n-8`，结果与旧 dequant bitwise 相同；
- **pipe 路径直接写 FP16**：复用 sB 作为 FP32 epilogue staging，去掉 fp32 tmp 和单独的 convert kernel；
- **down_proj 每个 warp 处理 2 个 N tile**：A fragment 在两个 tile 间复用，A 读流量减半，仍保持 4 个 K 分区 reduce；
- **融合 RoPE 与 GELU×up**：分别把 3 个 elementwise kernel 合成 1 个，输出与 eager 路径 bitwise 一致；
- B=16 分三条路径：窄 N/大 K 用 split2（4 个 K 分区 block + FP32 reduce），窄 N 用 4-warp K split，较宽 N 用 pipelined B16（预取下两个 K tile）；
- B=2 仍走 `pair_batch`；B=3–15 通过 zero-padding 到 16 行复用 B16 的 split/split2/pipe 路径，避免 B8 的旧 slow path；
- B=3–4 也走 padded TC，因 B16 kernel 的 step 比 `pair_batch` 更短。

### 2. Batch CUDA Graph 2..16 + host 更新向量化

- `_batch_graph_tensors` 把 16 次逐字段/逐 block-table 的 device 写改成每个字段一次 `copy_`，逐步消除约 80–96 个小 CUDA op；
- batch logits 不再 clone，sampler 直接消费 graph 静态输出。

- batch graph key 从 `N` 改为 `(N, long_context)`；
- 短上下文和长上下文各一份 graph；
- 短上下文预 warm 2..16；长上下文预 warm 2/4/8/16，其他 N 按需捕获；
- 长上下文 graph 用 batch split-K；短上下文用 warp-split；
- N>16 仍回退非 graph。

### 3. Cross-request batched prefill

新增 `_prefill_batch`：

- 多个请求的 token 拼成一次 forward；
- RMSNorm/QKV/O/MLP/PLE 在线性层按 token 行 batch；
- attention 和 KV 写入按请求分开，请求之间不会互相注意；
- 支持 fresh 和 incremental/chunked prefill；
- scheduler 的 prefill token 预算改为 `prompt_chunk * max_batch_size`，让同一批 fresh prefill 进入同一个 step。

### 4. Batch 长上下文 split-K

- `_decode_batch_graphable` 增加 `splitk_o/m/l`；
- full-attention 层在长上下文走 `gemma4_decode_attention_splitk`；sliding 层仍走 warp-split；
- split-K scratch 按 `(N, Hq, chunks, D)` 分配，可被 CUDA Graph 捕获。

## Parity

- kernel 测试：`mini-llm-kernels` **688 passed**（含新增 TC/parallel 40 项与 fused RoPE/GELU 4 项）。
- engine batch 2/4/8/16 vs eager：8-token token ID **全部一致**（`reports/gemma4_batch_parity.json`）。
- TC vs `pair_batch`：max abs diff ≤ 1e-3；FP16 权重/FP32 累加不变。
- 4k N=2：split-K 与 warp-split token ID 一致。
- 300-token prompt + chunk=64，N=2 incremental prefill 与 N=1 输出一致。
- server 回归：**75 passed**（排除未安装的 MLX 测试）。

## 性能

### 输出 16 token

| N | e2e tok/s | decode tok/s | per-request decode |
| ---: | ---: | ---: | ---: |
| 1 | 95.5 | 146.9 | 146.9 |
| 2 | 117.9 | 159.6 | 79.8 |
| 3 | 175.2 | 275.1 | 91.7 |
| 4 | **216.2** | **369.9** | 92.5 |
| 8 | **295.7** | **706.4** | 88.3 |
| 16 | **417.8** | **1613.9** | 100.9 |

### 输出 64 token

| N | e2e tok/s | decode tok/s | per-request decode |
| ---: | ---: | ---: | ---: |
| 1 | 127.5 | 142.5 | 142.5 |
| 2 | 146.7 | 159.6 | 79.8 |
| 3 | 240.3 | 272.2 | 90.7 |
| 4 | **310.3** | **364.6** | 91.2 |
| 8 | **533.9** | **704.2** | 88.0 |
| 16 | **941.5** | **1616.9** | 101.1 |

### 与优化前对比

| 场景 | 优化前 | 现在 |
| --- | ---: | ---: |
| N=2, 64 输出 e2e | 130.5 | **146.7** |
| N=4, 16 输出 e2e | 92.8 | **216.2** |
| N=4, 64 输出 e2e | 149.0 | **310.3** |
| N=8, 16 输出 e2e | 93.2 | **295.7** |
| N=8, 64 输出 e2e | 194.3 | **533.9** |
| N=16, 16 输出 e2e | 84.2 | **417.8** |
| N=16, 64 输出 e2e | 211.7 | **941.5** |
| N=16 last prefill | 2.73 s | **0.481 s** |
| N=8 last prefill | 1.05 s | **0.263 s** |

N=16 相对 N=1：e2e **4.38× / 7.38×**；decode **10.99× / 11.35×**。
per-request decode：N=16 **100.9 / 101.1**；N=8 **88.3 / 89.0**；N=4 **92.5 / 92.1**；N=2/3 约 **80–92**。
所有 N=2–16 现在都在 **80–92 tok/s**，不再只有 N=16 可用；其中 N=16 已接入 down（含 double-wide）和 pl_proj 的 fragment-major direct-B 路径。

### per-request 仍低于单请求的原因

- 单请求 `pair_vec8` 约 7 ms/step，已经贴近 2.3 GB INT4 权重的 HBM 带宽上限；
- N=16 现在约 11.1 ms/step，per-request 90 tok/s；要到约 140 tok/s 需要 7 ms/step，即所有 16 行几乎免费；
- 当前 TC 线约 6.9 ms：gateup/lm_head ~3.3 ms、down（two-tile split2）~1.8 ms、qkv/o/小矩阵 ~1.8 ms；另有 norm/attention/host 约 4 ms；
- 剩余瓶颈是 tensor-core 线的 shared staging/dequant 和 qkv/o 的固定延迟，不是 graph/attention；进一步需要 direct-fragment dequant、更大 tile 或完整 cp.async overlap。

### 长上下文

4k prompt，N=2，3 个输出：

| attention | decode tok/s | token ID |
| --- | ---: | --- |
| warp-split | 95.7 | 基准 |
| batch split-K | **117.2** | 一致 |

1050 context 时 split-K 与 warp-split 持平（约 170 tok/s）；4k 时 split-K 约 **+22%**。

### VRAM / 启动

- N=16 短+长 graph 预 warm（19 份 graph）后 free VRAM 约 **1.49 GiB**；新增 fused kernel 不增加持久显存；
- 启动约 3 s 量级（含 16 个 batch graph 捕获）；
- N=16 decode batch graph + TC 正常工作，无 OOM。

## 口径与限制

- N=4 现在通过 padding 复用 B16 kernel，per-request 约 79 tok/s；仍受 WMMA M=16 padding 限制，未到单请求 147。
- cross-request prefill 目前只 batch 线性/MLP/PLE，attention 仍按请求循环；不是 block-diagonal 单 kernel。
- TC 路径目前仅 FP16；BF16 会回退 `pair_batch`/`pair_vec`。
- 长上下文 batch graph 预 warm 2/4/8/16，其他 N 首步会捕获。
- 所有数据仍是同机、同 prompt、短上下文；不是 Ollama 的等价对比。
- 所有改动未 commit、未推送。
- 第二轮 kernel 细节（`prmt` dequant、pipe 直接写 FP16、down two-tile、fused RoPE/GELU）见 `reports/gemma4_kernel_round2.md`。
- 冲 85–90% 的第一轮尝试（cp.async、prepack、2D tile、direct-B）见 `reports/gemma4_85_90_attempt.md`；当前 N=16 per-req 约 91–92，约为单请求的 62%。

## 开关

- `MINI_LLM_INT4_TC=0`：关闭 tensor-core batch path，B=5–16 回退 `pair_vec`。
- `MINI_LLM_GEMMA4_BATCH_DECODE=0`：回到逐请求 decode。
- `MINI_LLM_GEMMA4_GRAPH=0`：关闭 graph，保留 batch forward。
- `MINI_LLM_GEMMA4_PREWARM_BATCH_GRAPH=0`：跳过 batch graph 预 warm。
- `MINI_LLM_GEMMA4_SPLITK_CTX`：短/长上下文分界，默认 1024。

## 相关文件

- `miniservellm/runtime/gemma4_engine_runner.py`
- `mini-llm-kernels/csrc/int4_matmul.cu`
- `mini-llm-kernels/mini_llm_kernels/kernels/int4_matmul.py`
- `mini-llm-kernels/tests/test_int4_matmul_tc.py`
- `scripts/bench_gemma4_batch.py`
- `scripts/check_gemma4_batch_decode.py`
- `reports/gemma4_batch_benchmark_2_4_bp16.json`
- `reports/gemma4_batch_benchmark_2_4_bp64.json`
- `reports/gemma4_batch_benchmark_batched_prefill.json`
- `reports/gemma4_batch_benchmark_batched_prefill_64.json`
