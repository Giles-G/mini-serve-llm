# Gemma4 P1 工作报告

日期：2026-10-20  
设备：RTX 3060 Laptop 6GB，sm_86；pair_vec8 + CUDA Graph  
状态：**P1-3（INT4/lm_head）完成；P1-4 完成 batch 2–4 pair_batch、batch 5–16 tensor-core（half2/uint4 dequant + split2/splitK/pipe）、batch graph 2–16（短/长）、cross-request batched prefill、batch 4k split-K；per-request N=16 从 33.5 提升到约 77，N=2–16 都在 77–81，仍未到单请求 147**

测试：server **75 passed**，kernels **684 passed**。

## P1-3：pair_vec8 / lm_head

### 实现

- 新增 `pair_vec8` decode GEMV：
  - 每个 lane 一次对齐 **64-bit load** 读取 8 个 packed 列；
  - 8 lanes 覆盖一个 64 列 group，一个 warp 每轮覆盖 4 个 group；
  - 相比 pair_vec 的 32-bit load，loop 次数和 load 指令减半；
  - 只用于 batch=1；batch>1 自动回退 pair_vec。
- 默认 `MINI_LLM_INT4_IMPL=pair_vec8`，保留 legacy/warp/pair/pair_vec。
- 覆盖 lm_head、qkv、gate/up、down、o_proj、PLE projection。

### 真实 shape（batch=1，FP16，3 轮中位数）

| shape | pair_vec | pair_vec8 | 提升 |
| --- | ---: | ---: | ---: |
| lm_head [262144,1536] | 1.028 ms | **0.845 ms** | **约 18%** |
| PLE [8960,1536] | 0.034 | **0.029** | 15% |
| gate_proj [6144,1536] | 0.024 | **0.022** | 8% |
| up_proj [6144,1536] | 0.027 | **0.022** | 19% |
| down_proj [1536,6144] | 0.027 | **0.022** | 19% |
| o_proj [1536,2048] | 0.020 | **0.018** | 10% |

小 shape（k/v 等）仍受 launch 限制，收益在噪声内。

### 端到端收益

- 128-token decode：**8.20 → 7.15 ms/token（约 12.8%）**。
- 256-token decode：**8.46 → 7.45 ms/token（约 12%）**。
- e2e：128-token 约 **131–132 tok/s**，256-token 约 **130 tok/s**。
- 与 pair_vec 的 256-token token ID 完全一致（pair_vec8 与 pair_vec 的低层 parity 与边界测试通过）。

## P1-4：Graph / runtime

### 已完成

- **启动时预 warm 双 graph**：
  - 短 graph 用 context=1 捕获；
  - 长 graph 用 context=`splitk_threshold+1` 捕获（仅当 KV capacity 足够）；
  - 正式请求不再支付第一次 graph capture；
  - 短/长 graph 分开保存，短请求不会复用长 split-K graph。
- **短上下文回归**：
  - 2k 长请求后接短请求，短请求首次不再 capture；
  - 短请求稳态约 **6.86 ms/token**，token ID 稳定；
  - 双 graph 同时存在时 free VRAM 约 **1.85 GiB**。
- **KV 预算**：
  - `build_runner` 支持 `max_batch_size`，KV blocks 按 `per-request blocks × max_batch` 分配；
  - batch>1 自动非 graph paged；
  - 4k+short、两个 4k（各 8 token）均完成；peak 3.29–3.36 GiB，free 1.20–1.36 GiB。

### True batched decode + batch graph + tensor-core batch linear（新增）

- `forward_decode` 对 N>1 走一次 batched forward：
  - 静态 `token[N]`、`position[N]`、`context_lens[N]`、`block_table[N,max]`、`write_slot[N]`；
  - attention 一次处理 N 行；
  - KV 写入按 N 个 slot 向量化；
  - sampler 批量处理 N 个 logits。
- batch 2–4 用 `pair_batch`；batch 5–16 用 tensor-core tile-dequant（WMMA 16x16x16，K tile 64，shared 反量化）；B=4 因 M=16 padding 仍保留 pair_batch。
- batch graph 2–16，短/长上下文各一份；长上下文 full-attention 用 batch split-K。
- cross-request batched prefill：多个请求的 token 拼成一次 forward，线性/MLP/PLE 按行 batch，attention/KV 按请求分开。

`reports/gemma4_batch_benchmark_2_4_bp16.json`、`..._batched_prefill.json`（16 输出）：

| batch | e2e tok/s | decode tok/s | per-request decode |
| ---: | ---: | ---: | ---: |
| 1 | 95.5 | 146.9 | 146.9 |
| 2 | 117.9 | 159.6 | 79.8 |
| 4 | 216.2 | 369.9 | 92.5 |
| 8 | **295.7** | **706.4** | 88.3 |
| 16 | **395.5** | **1445.8** | 90.4 |

`..._bp64.json`、`..._batched_prefill_64.json`（64 输出）：

| batch | e2e tok/s | decode tok/s | per-request decode |
| ---: | ---: | ---: | ---: |
| 1 | 127.5 | 142.5 | 142.5 |
| 2 | 146.7 | 159.6 | 79.8 |
| 4 | 310.3 | 364.6 | 91.2 |
| 8 | **533.9** | **704.2** | 88.0 |
| 16 | **889.2** | **1430.3** | 89.4 |

- batch 2/4/8/16 的 token ID 与 eager 逐 token 一致（`reports/gemma4_batch_parity.json`）。
- N=16 last prefill 从 2.73 s 降到 **0.481 s**；N=8 从 1.05 s 降到 **0.263 s**。
- 4k N=2 batch split-K decode 约 **117 tok/s** vs warp-split 95.7，token ID 一致。
- N=16 双 graph 启动后 free VRAM 约 **1.49 GiB**。

### 外部 baseline：Ollama `gemma4:e2b` Q4_K_M（单槽 n_slots=1）

新增 `scripts/bench_ollama_batch.py`，同一台 3060、同一组 prompt、`think=false`、`num_ctx=2048`、输出 16/64 token；Ollama 服务显示 1.8 GB / 100% GPU；日志确认 `n_slots=1`，N>1 请求排队。真实并发复测见 `reports/gemma4_ollama_parallel.md`。

两边是**同一个 Gemma4 E2B checkpoint**：Ollama 的 `general.parameter_count=5,123,179,235`，本地 safetensors 头统计为 `5,123,178,979`，仅差 256 个参数（约 0.000005%，属于转换/计数差异）。其中文本路径 4.647B，vision/audio 另占 0.472B；项目只执行文本路径。差异在量化与后端：Ollama 是 Q4_K_M 混合量化 + llama.cpp，项目是 INT4 group64 + 自研 CUDA kernel。

| 并发 | 实现（16 输出） | e2e tok/s | decode tok/s |
| ---: | --- | ---: | ---: |
| 1 | Ollama | 25.7 | 103.3 |
| 2 | Ollama | 39.5 | 202.4 |
| 4 | Ollama | 53.2 | 403.0 |
| 2 | 项目 batch graph | **117.9** | 159.6 |
| 4 | 项目 batch graph | **216.2** | 369.9 |
| 8 | 项目 batch graph | **295.7** | 706.4 |
| 16 | 项目 batch graph | **395.5** | 1445.8 |

| 并发 | 实现（64 输出） | e2e tok/s | decode tok/s |
| ---: | --- | ---: | ---: |
| 1 | Ollama | 61.0 | 105.6 |
| 2 | Ollama | 73.4 | 210.2 |
| 4 | Ollama | 84.3 | 416.6 |
| 2 | 项目 batch graph | **146.7** | 159.6 |
| 4 | 项目 batch graph | **310.3** | 364.6 |
| 8 | 项目 batch graph | **533.9** | 704.2 |
| 16 | 项目 batch graph | **889.2** | 1430.3 |

注：Ollama 只测到 N=4；项目 N=8/16 是扩展参考。Ollama 的 `aggregate decode` 用 `wall - max(TTFT)` 作分母，N>1 时 TTFT 错开会高估；项目的 batch graph 四个 row 在 decode 窗口内同时生成，口径不同。

结论：

- 单请求 decode：项目 **139.9** vs Ollama **103–106**，项目更快；
- 单槽下 Ollama 的 active per-request decode 在 N=1/2/4 都约 **100–106**，但这是“轮到时”的 decode，不含排队；表里的 403–417 是 `aggregate_decode` 用 `wall - max(TTFT)` 作分母造成的高估，不能当成 4× 真并发；
- 公平 e2e 扩展：Ollama N=4/N=1 约 **2.07×（16 输出）/ 1.38×（64 输出）**；项目 N=4/N=1 约 **2.26× / 2.43×**，N=8 约 **3.10× / 4.19×**，N=16 约 **4.14× / 6.97×**。显式 `OLLAMA_NUM_PARALLEL=4/8` 的真实并发公平对比为 **2.17× / 3.09× / 4.96×**（64 输出，N=4/8/16），见 `reports/gemma4_ollama_parallel.md`；
- N=4 衰减主因仍是 `pair_batch` 把 B 行串行在一个 warp 里；N=5–16 已换 tensor-core，prefill 已跨请求 batch。详见 `reports/gemma4_batch_scaling_analysis.md`、`reports/gemma4_batch_optimized.md`；
- 同一 checkpoint，但 Q4_K_M 与 INT4 group64 的量化、后端不同，属于系统级 baseline，不是 kernel 等价对比。

详细结果：`reports/gemma4_ollama_baseline.md`、`reports/gemma4_ollama_batch*.json`。

### 仍未做

- N=4 的 M=8/N=32 tensor-core 方案（理论上限约 2×，未做；N=4 继续 pair_batch）。
- sampler 最终 `.item()` 的进一步消除（顺序 decode 依赖，当前约 0.02 ms/step）。
- dense fallback 的速度优化（当前约 36–40 ms/token，仅无 paged kernel 时使用）。
- 8k/16k context 与多请求长上下文并发；4k batch split-K 已验 N=2，N=16 长上下文未系统压测。
- 加载期直接融合权重布局（当前已有运行期 compaction + 视图）。

## 最新正式验收

| 用例 | decode | tok/s | e2e |
| --- | ---: | ---: | ---: |
| 中文 128 token | **7.15 ms** | **139.91** | 132.1–132.3 |
| 英文 128 token | **7.19 ms** | **139.14** | 131.0–131.4 |
| 算术 27+45 | **6.82 ms** | **146.58** | 38.4–39.7 |
| 中文 256 token | **7.45 ms** | **134.29** | 129.9–130.9 |

prefill：**0.060–0.069 s**；VRAM free **2.21 GiB**。  
相对最初 80.83 ms/token：约 **11.3×**；256-token 约 **10.9×**。

## 长上下文（P0 split-K + P1）

| prompt | prefill | steady decode |
| ---: | ---: | ---: |
| 1024 | 0.473 s | **9.47 ms/token** |
| 2048 | 1.011 s | **9.67 ms/token** |
| 4096 | 2.427 s | **10.29 ms/token** |

此前报告的 2k/4k 约 18/29 ms 包含了首次 graph capture；P1 预 warm 后，表中是稳态 decode。

## 关键文件

- `csrc/int4_matmul.cu`：pair_vec8、pair_batch、tensor-core tile-dequant batch linear、parallel pair 原型。
- `mini_llm_kernels/kernels/int4_matmul.py`：pair_vec8 / pair_batch / tensor-core dispatch。
- `miniservellm/runtime/gemma4_engine_runner.py`：graph prewarm、双 graph、true batched decode、batch graph 2–16、cross-request prefill、split-K dispatch。
- `miniservellm/runtime/inference_engine.py`：Stage5Engine 启动时 prewarm。
- `scripts/bench_gemma4_cuda.py`、`scripts/bench_gemma4_batch.py`、`scripts/bench_ollama_batch.py`。
- `scripts/check_gemma4_batch_decode.py`、`scripts/check_gemma4_dual_graph.py`、`scripts/check_gemma4_concurrency.py`。
- `reports/gemma4_acceptance.json`、`reports/gemma4_e_boundaries.json`、`reports/gemma4_batch_benchmark*.json`、`reports/gemma4_batch_parity.json`、`reports/gemma4_ollama_baseline.md`、`reports/gemma4_ollama_batch*.json`、`reports/gemma4_dual_graph.json`、`reports/gemma4_concurrency.json`。
