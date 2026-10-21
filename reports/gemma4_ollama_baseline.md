# Ollama gemma4:e2b 多请求外部 baseline（单槽 n_slots=1）

> **注：本文件是系统默认 Ollama 的单槽基线。** 日志确认 `n_slots = 1`，N>1 请求排队；表中的 per-request 是 active decode（不含排队），不代表并发能力。真实并发复测（`OLLAMA_NUM_PARALLEL=1/4/8`）见 `reports/gemma4_ollama_parallel.md`。

日期：2026-10-20  
设备：RTX 3060 Laptop 6GB，Ollama 0.32.1  
模型：`gemma4:e2b`，Q4_K_M；Ollama 报告 5.123B 总参数，与本地 checkpoint 的参数规模一致（5,123,179,235 vs 5,123,178,979，仅差 256 个；文本路径 4.647B，另含 vision/audio）；服务显示 **1.8 GB / 100% GPU / num_ctx=2048**

## 口径

- 同一张 GPU、同一组中文 prompt、同一台机器；
- **同一个模型/checkpoint**：Ollama `general.parameter_count=5,123,179,235`，本地 `model.safetensors` 头统计为 `5,123,178,979`，仅差 256 个参数（约 0.000005%，属于转换/计数差异，不是另一个模型）；config 的 35 层、hidden 1536、8 heads、1 KV head、20 shared-KV 层、sliding window 512 也一致。项目只执行文本路径（4.647B），Ollama 权重还包含 vision（0.167B）和 audio（0.305B）；
- 因此差异不是“不同模型”，而是 **Q4_K_M（混合 4/6-bit GGUF，llama.cpp） vs 项目 INT4 group64（统一 4-bit，自研 CUDA kernel）**；
- Ollama 使用 `/api/chat`，`think=false`，temperature=0，`num_predict=16/64`，`num_ctx=2048`；
- 项目使用现有 Engine，prompt 19 token，同样输出 16/64；
- 两边都 warmup 后测；Ollama TTFT 为流式首 content 时间；
- 项目 TTFT 为 prefill 到首 token；
- aggregate e2e = 总输出 token / 所有请求 wall time；
- aggregate decode 只统计生成阶段；
- 当前 Ollama service 没有在 unit file 中设置 `OLLAMA_NUM_PARALLEL`；`journalctl` 中始终只有 `slot id 0`，行为上单槽排队，N>1 不产生真正的 N 行 batch。

## 输出 16 token

| 并发 | 实现 | aggregate e2e tok/s | aggregate decode tok/s† | per-request decode tok/s | TTFT |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Ollama | 25.7 | 103.3 | 102.9 | 0.485 s |
| 2 | Ollama | 39.5 | 202.4 | 101.8 | 0.661 s |
| 4 | Ollama | 53.2 | 403.0 | 100.7 | 0.863 s |
| 1 | 项目 single graph | 95.5 | 146.9 | 146.9 | ~0.065 s |
| 2 | 项目 batched graph | **117.9** | 159.6 | 79.8 | ~0.083 s（最后 prefill） |
| 4 | 项目 batched graph | **216.2** | 369.9 | 92.5 | ~0.134 s（最后 prefill） |
| 8 | 项目 batched graph | **295.7** | 706.4 | 88.3 | ~0.263 s（最后 prefill） |
| 16 | 项目 batched graph | **395.5** | 1445.8 | 90.4 | ~0.481 s（最后 prefill） |

† Ollama 的 `aggregate decode` 以 `wall - max(TTFT)` 为分母；N>1 时各请求 TTFT 错开，早完成的请求仍计入分子，因此会高估。它不是四个请求同时 decode 的 4× 吞吐。项目的 batch graph 四个 row 在 decode 窗口内同时生成，口径不同。

## 输出 64 token

| 并发 | 实现 | aggregate e2e tok/s | aggregate decode tok/s† | per-request decode tok/s |
| ---: | --- | ---: | ---: | ---: |
| 1 | Ollama | 61.0 | 105.6 | 105.6 |
| 2 | Ollama | 73.4 | 210.2 | 105.7 |
| 4 | Ollama | 84.3 | 416.6 | 104.4 |
| 1 | 项目 single graph | 127.5 | 142.5 | 142.5 |
| 2 | 项目 batched graph | **146.7** | 159.6 | 79.8 |
| 4 | 项目 batched graph | **310.3** | 364.6 | 91.2 |
| 8 | 项目 batched graph | **533.9** | 704.2 | 88.0 |
| 16 | 项目 batched graph | **889.2** | 1430.3 | 89.4 |

## N=8/16 补测

当前服务 `n_slots=1`（`-np 1`），N>1 严格排队，不是真正的 N 行 batch。补测的 active decode 基本不变，但 wall-effective per-request 快速下降：

| 输出 | N=8 e2e | N=16 e2e | N=8 per-req decode | N=16 per-req decode | N=8 TTFT | N=16 TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 60.2 | 64.2 | 101.3 | 101.1 | 1.407 s | 2.533 s |
| 64 | 89.2 | 90.3 | 104.8 | 104.1 | 3.232 s | 6.252 s |

active per-request decode 基本不变，e2e 在 N=8 后饱和；按 wall 算 N=8/16 的 per-request 只有约 **7.5/4.0**（16 输出）和 **11.2/5.7**（64 输出）。`aggregate_decode` 的 790/1539（16 输出）和 841/1660（64 输出）是排队窗口假象，不采用。真实并发见 `reports/gemma4_ollama_parallel.md`。

## 结论

- **单请求 decode**：项目约 **140–147 tok/s**，Ollama 约 **103–106 tok/s**；项目更快。
- **单槽下 active decode 不衰减，是因为请求在排队**：N=1/2/4 都约 **100–106 tok/s**；但 wall-effective per-request 在 N=16/64 输出只有约 **5.7 tok/s**。真实并发会明显衰减，见 `reports/gemma4_ollama_parallel.md`。
- **公平看 e2e 扩展**：
  - Ollama N=4/N=1：16 输出 **2.07×**（25.7→53.2），64 输出 **1.38×**（61.0→84.3）；
  - 项目 N=4/N=1：16 输出 **2.26×**（95.5→216.2），64 输出 **2.43×**（127.5→310.3）；
  - 项目 N=8/N=1：**3.10× / 4.19×**；N=16/N=1：**4.14× / 6.97×**。
  - 结论：Ollama 的 N=4 扩展约 2.07×/1.38×；项目 N=4 已接近/超过，N=8/16 明显超过。
- **项目 per-request decode 在 N=2–16 基本平坦**：约 **80–92 tok/s**，N=16 为 **90.4/89.4**；已不再是 N=4 的 50、N=16 的 33。
- **prefill 串行化已解决**：cross-request batched prefill 让 N=16 last prefill 从 2.73 s 降到 **0.481 s**，N=4 从 0.42 s 降到 **0.134 s**。详见 `reports/gemma4_batch_optimized.md`。
- 现在的结论：
  - 单请求：项目 decode 更快；
  - 单槽顺序 e2e：项目 N=16 约 **6.17×/9.84×**；真实并发公平对比为 **3.74×/4.96×**（见并行报告）；
  - active per-request：单槽 Ollama 约 100–105，项目约 89–92；但 Ollama 的 active 数字不含排队，wall-effective 远低于项目；
  - 项目 per-request 仍低于单请求，剩余瓶颈是 tensor-core 线的 HBM/shared staging，而不是 prefill 或 graph。

## 注意

- 两边是**同一个 Gemma4 E2B checkpoint**：Ollama 的 5.123B 是含 vision/audio 的总参数，项目只跑 4.647B 文本路径，不是两个不同模型；
- 但 Ollama 是 Q4_K_M 混合量化 + llama.cpp，项目是 INT4 group64 + 自研 CUDA kernel，数值格式和后端都不同，输出 token 不要求一致；
- Ollama 服务未显式固定 `OLLAMA_NUM_PARALLEL`，不同部署可能不同；
- 这是系统级 baseline，不是 kernel 级等价对照。
