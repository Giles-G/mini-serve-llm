# Gemma4 RTX 3060 优化交付报告

日期：2026-10-20  
设备：NVIDIA RTX 3060 Laptop GPU 6GB（sm_86，WSL2，驱动 595.71）  
软件：torch 2.13.0+cu130，CUDA 13.0，模型 `models/gemma-4-E2B-it`，INT4 group_size=64，FP16 activation  
状态：**batch=1 走 CUDA Graph + pair_vec8；batch 2–4 走 pair_batch，batch 5–16 走 tensor-core tile-dequant；batch graph 覆盖 2–16（短/长）；prefill 支持跨请求 batch；短上下文 warp-split，长上下文 full-attention 全局 split-K；1k/2k/4k 与 eager 逐 token 一致**

## 1. 最终性能

正式验收命令：`scripts/bench_gemma4_acceptance.py`  
3 个 prompt × 3 轮，warmup（含 graph capture）在同一 engine 上完成，不计入正式 decode：

| prompt | prompt tokens | decode 中位数 | decode 范围 | decode tok/s 中位数 | end-to-end |
| --- | ---: | ---: | ---: | ---: | ---: |
| 中文 KV Cache | 19 | **7.17 ms/token** | 7.11–7.19 | **139.41 tok/s** | 131.3–132.2 tok/s |
| 英文 KV Cache | 22 | **7.21 ms/token** | 7.20–7.21 | **138.63 tok/s** | 129.1–131.3 tok/s |
| 算术 27+45 | 25 | **7.12 ms/token** | 7.07–7.14 | **140.37 tok/s**（EOS） | 38.3–39.1 tok/s |
| 中文 256 token | 19 | **7.50 ms/token** | 单轮 | **133.41 tok/s** | 129.6 tok/s |

prefill 约 **0.060–0.069 s**（19–25 prompt tokens）。显存：加载后空闲 2.67 GiB；运行后空闲 **2.21 GiB**。  
GPU 运行后约 1770 MHz、72 °C、99.7 W。

### 历史对比

| 状态 | decode ms/token | tok/s |
| --- | ---: | ---: |
| 最初 baseline（legacy INT4） | 80.83 | 12.37 |
| 融合 RMSNorm | 59.87 | 16.70 |
| + RoPE 缓存 | 57.36 | 17.43 |
| + addcmul RoPE | 55.75 | 17.94 |
| + QKV/gate-up 合并 + 单写 KV | 52.43（legacy）/ 36.45（warp） | 19.1 / 27.4 |
| 非 graph paged（warp） | 35.4 | 28.2 |
| CUDA Graph + warp GEMV | 12.5–12.8 | ≈79 |
| CUDA Graph + row-pair GEMV | 9.88–11.62 | 86.1–101.3 |
| CUDA Graph + pair_vec GEMV | 9.81–11.39 | 87.8–102.0 |
| + batched prefill / 权重去重 | 9.73–11.29 | 88.6–102.8 |
| + warp-split / 快速反量化 | 8.10–8.42 | 118.7–127.9 |
| + 全局 split-K + 快速反量化 | 8.20–8.46 | 118.2–129.4 |
| **+ pair_vec8 + graph 预 warm（默认）** | **7.15–7.45** | **134.3–146.6** |

相对最初 80.83 ms/token：**约 8.3×（decode）**。  
相对最初 12.37 tok/s：**约 11.3×**；256-token 约 **10.9×**。  
P1 的 pair_vec8 把 lm_head 从 1.028 降到 **0.845 ms**，短上下文 decode 从 8.2 降到 **7.15–7.45 ms**；
graph 预 warm 消除了正式请求的首次 capture；长上下文稳态 decode 约 **9.5–10.3 ms/token**。

## 2. 已实现的优化

### A/B：profiling 与 INT4 GEMV

- profile 增加起始 context、期望/实际步数、分组归因（kernel vs 父 label）、每 step/per-call 时间、`--profile-shapes`。
- `--output-json` 增加 device/driver/torch/CUDA/git/EOS/backend/weights/317 个真实 shape 元数据。
- direct/engine 双后端 profile（`--profile-mode auto|direct|engine`），输出 raw-kernel 对照并排除父 label；逐事件 `event_shapes`、`text_tokenize_s`/`text_postprocess_s` 补齐。
- direct `finish_reason`、空文本失败标记（默认退出码 1）、`text_decode_s`/`text_print_s`。
- INT4 默认改为 pair_vec（vectorized row-pair）GEMV；`legacy|warp|pair|pair_vec` 可选。
- `bench_gemma4_cuda.py` 增加逻辑带宽、3 轮中位数/范围、GPU 时钟/温度/功耗、多 seed、kernel register/occupancy attributes。
- 详细结果见 `reports/gemma4_ab_gaps.md`。

### D：既有小算子 / KV 基础

- `torch.nn.functional.rms_norm` 替换多 kernel 的 eager RMSNorm（仅 CUDA affine norm）。
- engine 每个 forward 只算一次 sliding/full 的 RoPE cos/sin，并预 cast 到模型 dtype。
- CUDA FP16 的 `_apply_rope` 改为 `mul + 2×addcmul_`，去掉 `cat/neg`。
- post-attention residual add 与 pre-FFN RMSNorm 合并为 `fused_add_rms_norm`。
- 非共享层 q/k/v 合并为一次 INT4 GEMV；gate/up 合并为一次 INT4 GEMV。
- row-pair INT4 GEMV：一个 warp 算相邻两行，共享 packed byte；lm_head 1.57→1.03 ms，gate/up 0.034→0.024 ms。
- pair_vec：每个 lane 一次 32-bit load 读 4 个 packed 列，warp 一次覆盖 2 个 64 列 group；lm_head 约 1.0 ms，并设为默认。37 regs/thread、12 blocks/SM、1536 active threads/SM（约 75% occupancy）。
- pair 的 rows/block 可调（1/2/4/8）；实测 2/4/8 接近，4 作为平衡默认。
- CUDA paged 模式下 decode 不再维护 dense mirror；prefill 仍保留 dense mirror，供 chunked prefill 历史和共享层使用。
- `fused_norm.cu` 补充 current stream、device guard、contiguous/shape 检查、launch check。
- INT4 fallback 的 rounded-weight 合同改为 FP32 matmul 后 cast，修复 BF16 fallback 超预算问题。

### D：本轮缺口补齐

- residual add、layer scale、per-layer gate 乘法改为原地操作，prefill/dense/graph 三条路径一致。
- 融合 qkv/gate-up 权重后把原 projection 重新指向融合张量的视图，释放重复副本；graph 路径空闲显存 **2.01 → 2.40 GiB**。
- block table 的 int32 CUDA tensor 按 request 缓存，长度不变时不再每步重建。
- 新增 `MINI_LLM_GEMMA4_PAGED=0` 显式 dense fallback gate；修复 prefill dense mirror 语义后，19+160 token 跨容量增长点与 eager **160/160 一致**。
- `reports/gemma4_d_boundaries.json` 覆盖 520 graph、graph 复用、batch=2、forced dense；520+32 降到约 **0.56 s**。

### E：本轮缺口补齐（prefill / attention）

- 修正 benchmark engine 的 token budget：`max_tokens_per_step` 从 1 改为 prompt 长度，prompt chunk 不再退化为逐 token。
- 新增快速 INT4→FP16/BF16 反量化 kernel + prefill GEMM（`rows>=8`）：1024/2048/4096 token prefill 约 **0.55/1.04/2.37 s**，2k 达 **1966 tok/s**。
- prefill attention 按 512 query chunk 计算，4k 峰值显存从约 4.18 降到 **3.21 GiB**。
- 新增 warp-split decode attention：8 warp 分段 KV，去掉 per-token barrier；4k D512 从 4.18 ms 降到 **0.915 ms**（4.57×）。
- 新增全局 split-K：长上下文 full-attention 使用 `(N,Hq,chunks)` block + merge kernel；4k D512 进一步降到 **0.270 ms**（是 warp-split 的 3.29×）。
- 启动预 warm 短/长 graph 后，长上下文稳态 decode：**1024 9.47 / 2048 9.67 / 4096 10.29 ms/token**。
- 短上下文/ sliding 层仍用 warp-split，split-K dispatch 不改变短路径；后续 pair_vec8 把 128-token 进一步降到 **7.15 ms/token**。
- 4k 并发边界：4k+short、两个 4k batch=2 均完成；peak 3.29–3.36 GiB，free 1.20–1.36 GiB；两个 4k 各生成 8 token 都完成；大 prefill 由 scheduler token budget 串行化。
- attention 测试扩展到 context 2048/4096，并加入 warp-split/全局 split-K parity；FP32 参考修正后 kernel 最大差约 1e-3。
- `scripts/bench_gemma4_prefill.py`、`scripts/bench_gemma4_attention.py`、`scripts/check_gemma4_e_boundaries.py` 已生成结果。
- 详细数据见 `reports/gemma4_de_report.md`。

### E：既有 attention / decode 基础

- 旧的 split-K/partial attention 实现不可用；新的全局 split-K 已单独实现并通过 parity。
- safe kernel 直接 parity 通过：D=256/512、GQA、block_size 16/128、context 1/17/129/512/1024，并已扩展到 **2048/4096**；window 0/8/32/512、FP16/BF16、非默认 stream。
- `_CUDA_ATTN_VERIFIED=True`；新增 `tests/test_gemma4_attention_safe.py`。
- paged 非 graph 的完整模型输出与 eager reference 在 256 token 上一致。
- 旧 dense engine 的 256-token 差异已随 prefill dense mirror 修复消失；forced dense 19+160 token 现与 eager 逐 token 一致。

### P1：decode GEMV / batch graph / batch linear

- `pair_vec8`：每个 lane 64-bit load 读 8 个 packed 列，一个 warp 每轮覆盖 4 个 group；默认用于 batch=1 decode。
- lm_head 1.028 → **0.845 ms**；128-token decode 8.20 → **7.15 ms/token**，256-token 8.46 → **7.45 ms/token**。
- B=2 用 `pair_batch`；B=3–15 zero-pad 到 16 行，复用 B16 tensor-core kernel；B=16 直接走 split2/splitK/pipelined B16。dequant 用 half2+uint4 写 shared，FP16×FP16/FP32 累加。
- batch graph 从 2/3/4 扩到 **2–16**，短/长上下文各一份；长上下文 full-attention 走 batch split-K，sliding 仍走 warp-split。
- Stage5Engine 启动预 warm 短 graph 2–16；长 graph 预 warm 2/4/8/16，其余按需捕获。
- 新增 cross-request batched prefill：多个请求的 token 拼成一次 forward，线性/MLP/PLE 按行 batch，attention/KV 按请求分开；支持 fresh 与 chunked prefill。
- 性能（16 输出）：N=2 **117.9 / 159.6**；N=4 **216.2 / 369.9**；N=8 **295.7 / 706.4**；N=16 **402.2 / 1546.0** tok/s。
- 性能（64 输出）：N=2 **146.7 / 159.6**；N=4 **312.5 / 368.6**；N=8 **548.9 / 712.2**；N=16 **913.0 / 1513.3** tok/s。
- N=16 相对 N=1：e2e **4.30×/7.09×**、decode **10.02×/10.26×**；N=8：**3.18×/4.31×**；N=4：**2.26×/2.45×**。
- per-request decode：N=2/3 约 **80–92**，N=4 **92.5/92.1**，N=8 **88.3/89.0**，N=16 **96.6/94.6**（16/64 输出）。N=16 从优化前 33.5 提升约 **2.82×**，仍低于单请求 147。
- batch 2/4/8/16 的 token ID 与 eager **全部一致**；4k N=2 split-K 与 warp-split token ID 一致，decode 约 **+22%**。
- 2k 长请求后短请求稳态 **6.86 ms/token**；N=16 双 graph 启动后 free VRAM 约 **1.49 GiB**。
- 第二轮 kernel 优化：nibble→half 改为 `prmt`+half2（bitwise 等价），pipe 直接写 FP16 并复用 sB，down 改两 tile/warp 复用 A，新增 fused RoPE 与 fused GELU×up；N=16 64 输出 per-request **78.3→89.4**，step **12.8→11.1 ms**。
- 仍未做：N=4 的 M=8/N=32 方案（上限约 2×）、sampler 最终 `.item()`、dense fallback 速度。

### 外部 baseline：Ollama gemma4:e2b Q4_K_M（单槽 n_slots=1）

同一台 3060、同一组 prompt、`think=false`、`num_ctx=2048`；Ollama 服务显示 1.8 GB / 100% GPU；日志确认 `n_slots=1`，N>1 请求排队。真实并发复测见 `reports/gemma4_ollama_parallel.md`。

两边是**同一个 Gemma4 E2B checkpoint**：Ollama `general.parameter_count=5,123,179,235`，本地 safetensors 头统计为 `5,123,178,979`，仅差 256 个参数（约 0.000005%，属于转换/计数差异）。文本路径 4.647B，vision/audio 另占 0.472B；项目只执行文本路径。差异在量化与后端：Ollama 用 Q4_K_M + llama.cpp，项目用 INT4 group64 + 自研 CUDA kernel。

| 并发 | 实现（16 输出） | e2e tok/s | decode tok/s |
| ---: | --- | ---: | ---: |
| 1 | Ollama | 25.7 | 103.3 |
| 2 | Ollama | 39.5 | 202.4 |
| 4 | Ollama | 53.2 | 403.0 |
| 2 | 项目 batch graph | **117.9** | 159.6 |
| 4 | 项目 batch graph | **216.2** | 369.9 |
| 8 | 项目 batch graph | **295.7** | 706.4 |
| 16 | 项目 batch graph | **402.2** | 1546.0 |

64 输出时，项目 N2/N4/N8/N16 e2e 约 **146.7/310.3/533.9/889.2**，Ollama N1/N2/N4 约 **61.0/73.4/84.3**（Ollama 未测 8/16）。

注：Ollama 只测到 N=4；项目 N=8/16 是扩展参考。Ollama 的 `aggregate decode` 用 `wall - max(TTFT)` 作分母，N>1 时 TTFT 错开会高估；N4 的 416.6 不能当作 4× 真并发。项目的 batch graph 四个 row 在 decode 窗口内同时生成，口径不同。

结论：

- 单请求 decode：项目 139.9 vs Ollama 103–106，项目更快；
- 单槽下 Ollama 的 active per-request decode 在 N=1/2/4 都约 100–106，但这是“轮到时”的 decode，不含排队；`aggregate_decode` 的 403–417 是 `wall - max(TTFT)` 口径造成的高估，不能当成 4× 真并发；
- 公平 e2e 扩展：Ollama N=4/N=1 约 **2.07×（16 输出）/ 1.38×（64 输出）**；项目 N=4/N=1 约 **2.26× / 2.43×**，N=8 约 **3.10× / 4.19×**，N=16 约 **4.14× / 6.97×**。显式 `OLLAMA_NUM_PARALLEL=4/8` 的真实并发公平对比为 **2.17× / 3.09× / 4.96×**（64 输出，N=4/8/16），见 `reports/gemma4_ollama_parallel.md`；
- N=4 已不再走 `pair_batch`；B=3–15 zero-pad 到 16 行复用 B16 tensor-core，prefill 也已跨请求 batch。详见 `reports/gemma4_batch_scaling_analysis.md` 和 `reports/gemma4_batch_optimized.md`；
- 同一 checkpoint，但 Q4_K_M 与 INT4 group64 的量化、后端不同，属于系统级 baseline；服务未显式固定 `OLLAMA_NUM_PARALLEL`。

### CUDA Graph

- 新增 graph-capturable paged decode：静态 token/position/context_lens/block_table/write slot，paged KV，safe attention kernel。
- 在同一个 engine 的 warmup 请求中捕获；正式 decode 只 replay。
- 修复 attention host 侧 `ctx.max().item()` 在 capture 时触发 stream capture invalidation 的问题（capture 时跳过该 host 同步检查）。
- Graph 首次捕获后需要额外 replay 一次，才能得到 steady-state 结果。
- 默认 CUDA batch=1 启用；`MINI_LLM_GEMMA4_GRAPH=0` 使用非 graph paged；graph 失败自动回退到 paged，再失败才 dense。
- 新增 `scripts/bench_gemma4_acceptance.py`，benchmark 的 `--warmup` 改为同一 engine，避免把 capture 计入正式 decode。

## 3. 正确性

- **Graph vs eager reference：256 token 逐 token 一致**（`first_divergence=null`）。
- **pair_vec vs scalar pair：256 token 逐 token 一致**；pair/pair_vec 低层/边界/oracle parity 全部通过。
- **Pair vs warp：256 token 逐 token 一致**；pair 低层/边界/oracle parity 全部通过。
- **Paged vs eager reference：256 token 逐 token 一致**。
- C 阶段报告：`reports/gemma4_c_parity.md`
  - FP16 3 prompt，277 labels，0 top-1 flips；官方 512 sliding 520 tokens × 32 logits 0 flips。
  - BF16 3 prompt，2 激活 step，8 logits，0 flips；官方 512 sliding 32 logits 0 flips；contracts 通过。
- 测试：
  - server：**75 passed**
  - kernels：**684 passed**（含 INT4 边界、pair_vec8、pair_batch、tensor-core batch、fast dequant、safe/warp-split/全局 split-K attention 1–4096 矩阵）
- 生成文本：中文/英文/算术完整输出已保存到 `reports/gemma4_acceptance.json`；算术回答为 `72`，`finish_reason=EOS`。
- 端到端边界：1024/2048/4096 prompt + decode 与 eager 逐 token 一致；稳态 decode 约 9.47/9.67/10.29 ms/token。

## 4. 未完成 / 已知限制

- prefill 反量化仍是逐 projection 临时执行（虽然已换成快速 CUDA kernel），不是持久化 cache；4096 prompt 峰值约 3.21 GiB。19-token TTFT 约 60–69 ms，未到 35–50 ms。
- 4k 稳态 decode 约 10.3 ms/token；全局 split-K 已实现，但仍有 merge kernel/scratch 开销。
- dense engine fallback 已修复：forced dense 19+160 token 与 eager 一致；它仍比 graph 慢（约 36–40 ms/token），只作为无 paged kernel 时的兜底。
- graph 首次请求 capture 已通过启动预 warm 消除；模型加载后 engine 初始化会多一次短/长 graph 捕获。
- Graph 覆盖 batch=1（短/长）和 batch=2–16（短/长）；N>16 回退非 graph batched decode。
- batch 长上下文 split-K 已接；4k N=2 decode 比 warp-split 约 +22%，但仍有 partial/merge 开销。
- N=4 的 tensor-core 受 M=16 padding 限制，仍用 `pair_batch`；M=8/N=32 方案未做。
- sampler 最终 `.item()`、dense fallback 速度未优化；没有 Ollama/Nsight/trace 交付。
- 所有改动**未提交、未推送**。

## 5. 关键文件

- `miniservellm/runtime/gemma4_runner.py`：RMSNorm、RoPE addcmul。
- `miniservellm/runtime/gemma4_engine_runner.py`：RoPE 缓存、QKV/gate-up 融合与去重、prefill GEMM、cross-request prefill、tensor-core batch dispatch、batch graph 2–16、paged/graph decode、attention。
- `mini-llm-kernels/csrc/int4_matmul.cu`：pair_vec8、pair_batch、tensor-core tile-dequant batch linear、parallel pair 原型。
- `miniservellm/cache/gemma4_paged_kv_cache.py`：block table tensor 缓存与 KV 写路径。
- `scripts/check_gemma4_graph_boundaries.py`：520 graph、graph 复用、batch2、forced dense。
- `scripts/bench_gemma4_speed.py`：同 engine warmup、profile 分组/起始 context/元数据。
- `scripts/bench_gemma4_cuda.py`：pair/pair_vec/warp/legacy 对比、多轮中位数、逻辑带宽、GPU 状态、kernel attributes。
- `scripts/bench_gemma4_acceptance.py`：正式验收。
- `scripts/bench_gemma4_prefill.py`、`scripts/bench_gemma4_attention.py`：prefill/attention 对照。
- `scripts/check_gemma4_e_boundaries.py`：1k/2k/4k 端到端边界。
- `scripts/check_gemma4_concurrency.py`、`scripts/check_gemma4_dual_graph.py`：4k 并发与长→短 graph 回归。
- `scripts/bench_gemma4_batch.py`、`scripts/check_gemma4_batch_decode.py`：batch 2–16 吞吐与 parity。
- `scripts/bench_ollama_batch.py`：Ollama 1/2/4/8/16 并发外部 baseline。
- `reports/gemma4_p1_report.md`：P1 工作包详细报告。
- `scripts/check_gemma4_int4_parity.py`：C parity。
- `reports/gemma4_c_parity.md`、`reports/gemma4_ab_gaps.md`、`reports/gemma4_de_report.md`、`reports/gemma4_p1_report.md`、`reports/gemma4_ollama_baseline.md`、`reports/gemma4_vs_ollama_final.md`、`reports/gemma4_batch_scaling_analysis.md`、`reports/gemma4_batch_optimized.md`、`reports/gemma4_tc_batch_proto.md`、`reports/gemma4_acceptance.json`、`reports/gemma4_e_*.json`、`reports/gemma4_batch_*.json`、`reports/gemma4_ollama_batch*.json`、`reports/gemma4_d_boundaries.json`、`reports/gemma4_concurrency.json`、`reports/gemma4_dual_graph.json`。
- kernels：`csrc/int4_matmul.cu`（row-pair + pair_vec）、`csrc/gemma4_attention.cu`、`csrc/fused_norm.cu`、`mini_llm_kernels/kernels/gemma4_attention.py`、`int4_matmul.py`、`tests/test_int4_matmul_edges.py`、`tests/test_gemma4_attention_safe.py`。
