# Ollama 真实并发 vs 单槽：OLLAMA_NUM_PARALLEL 复测

日期：2026-10-21  
设备：RTX 3060 Laptop 6GB  
模型：`gemma4:e2b` Q4_K_M，Ollama 0.32.1  
项目：同一 checkpoint，INT4 group64 + 自研 CUDA kernel

## 为什么要重测

之前的 Ollama 对比里，N=16 的 per-request 仍在 **104 tok/s** 左右，看起来“不衰减”。复查服务日志后确认：

```text
srv load_model: initializing, n_slots = 1, n_ctx_slot = 2048
```

原服务只有 **1 个 slot**，N>1 的请求是排队的。Ollama 返回的 `eval_duration` 只统计该请求真正被 decode 的时间，不含排队时间，所以 per-request 看起来没衰减；但用户感知的完成速度已经大幅下降。

例如 64 输出、N=16、单槽：

```text
e2e = 90.64 tok/s
wall = 16 × 64 / 90.64 = 11.30 s
wall-effective per-request = 64 / 11.30 = 5.66 tok/s
```

也就是说单槽 N=16 时，每个请求平均只有 **5.66 tok/s**，不是 104。

真实并发必须显式设置 `OLLAMA_NUM_PARALLEL`，让服务创建多个 slot。

## 测试方法

为了不改动系统服务，单独在 `127.0.0.1:11435` 启动一个 Ollama：

```bash
OLLAMA_HOST=127.0.0.1:11435 \
OLLAMA_MODELS=/usr/share/ollama/.ollama/models \
OLLAMA_NUM_PARALLEL=4 \
OLLAMA_MAX_LOADED_MODELS=1 \
ollama serve
```

分别测 `OLLAMA_NUM_PARALLEL=1/4/8`。其他条件保持一致：

- `/api/chat`，`think=false`；
- `temperature=0`，`num_ctx=2048`，`num_predict=16/64`；
- N 个 Python 线程用 barrier 同时释放；
- 每个并发跑 2 轮取中位数；
- 预热一次，排除模型加载；
- 日志确认 `n_slots = 1/4/8`。

另外用 [ollama_bench](https://github.com/dkruyt/ollama_bench) 在 `OLLAMA_NUM_PARALLEL=4`、16 并发下交叉验证：

```text
16/16 ok, throughput=2.30 rps
latency p50=4841.9 ms, TTFT p50=3664.8 ms
output tokens=1024, wall≈6.94 s
```

`64 / (4.842 - 3.665) ≈ 54 tok/s`，与我们的 52 tok/s active decode 基本一致，说明不是采样脚本的问题。

## 结果：64 输出

### e2e（总输出 token / wall）

| N | p1 单槽 | p4 | p8 | 最佳 Ollama |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 59.5 | - | 62.5 | 62.5 |
| 4 | 84.9 | **143.1** | 140.0 | **143.1** |
| 8 | - | - | **173.0** | **173.0** |
| 16 | 90.6 | 163.5 | **179.4** | **179.4** |

### active decode（Ollama eval_duration，不含排队）

| N | p1 单槽 | p4 | p8 |
| ---: | ---: | ---: | ---: |
| 1 | 105.5 | - | 105.5 |
| 4 | 105.3 | 52.3 | 52.7 |
| 8 | - | - | **29.4** |
| 16 | 104.8 | 52.0 | **29.1** |

### wall-effective per-request（`e2e / N`，包含排队+prefill）

| N | p1 单槽 | p4 | p8 |
| ---: | ---: | ---: | ---: |
| 4 | 21.2 | **35.8** | 35.0 |
| 8 | - | - | **21.6** |
| 16 | **5.7** | 10.2 | **11.2** |

这就是关键：**真实并发时 per-request 一定衰减**。

- 单槽不衰减，是因为根本没有并发，只是排队；
- 4 并发：active decode 从 105 降到 **52（约 50%）**；
- 8 并发：active decode 降到 **29（约 28%）**；
- e2e 则从单槽 N=16 的 90.6 提升到 p8 的 179.4，用 per-request 换总吞吐。

## 结果：16 输出（p8）

| N | Ollama e2e | active per-req | wall-effective | 项目 e2e | 项目 active per-req |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 70.6 | 52.6 | 17.6 | **216.2** | **92.5** |
| 8 | 91.4 | 27.8 | 11.4 | **295.7** | **88.3** |
| 16 | 105.9 | 27.6 | 6.6 | **395.5** | **90.4** |

## 项目 vs Ollama 最佳并发配置

Ollama 取每个 N 下 e2e 最好的配置（N=4 用 p4，N=8/16 用 p8）。

### 64 输出

| N | 项目 e2e | Ollama e2e | e2e 倍数 | 项目 active | Ollama active | active 倍数 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | **310.3** | 143.1 | **2.17×** | **91.2** | 52.3 | **1.74×** |
| 8 | **533.9** | 173.0 | **3.09×** | **88.0** | 29.4 | **2.99×** |
| 16 | **889.2** | 179.4 | **4.96×** | **89.4** | 29.1 | **3.07×** |

### 16 输出

| N | 项目 e2e | Ollama e2e | e2e 倍数 | 项目 active | Ollama active | active 倍数 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | **216.2** | 70.6 | **3.06×** | **92.5** | 52.6 | **1.76×** |
| 8 | **295.7** | 91.4 | **3.23×** | **88.3** | 27.8 | **3.18×** |
| 16 | **395.5** | 105.9 | **3.74×** | **90.4** | 27.6 | **3.28×** |

## 正确结论

1. **之前的“Ollama per-request 不衰减”是单槽排队造成的口径错觉**。`eval_duration` 不含排队时间，不能当作并发时的 per-request 吞吐。
2. **真实并发下 Ollama 衰减很明显**：4 并发约 **50%**，8 并发约 **28%**。
3. **项目的扩展性明显更好**：
   - 64 输出 N=4/8/16，active per-request 约 **91/88/89**，相对单请求 142.5 约 **62–64%**；
   - Ollama 对应只有 **52/29/29**，约为单请求的 **49%/28%/28%**；
   - e2e 项目领先 **2.17× / 3.09× / 4.96×**。
4. 项目也没有达到“90% 不衰减”，但比 Ollama 的单请求衰减小得多；剩余瓶颈仍是 tensor-core 权重 staging 和 qkv/o 固定开销。
5. 之前的 9.84× 是“项目 16 行 batch vs Ollama 单槽排队”，不是公平并发对比；公平对比应使用 **Ollama 最佳并行配置**，即本页的 **4.96×（64 输出）/ 3.74×（16 输出）**。

## 数据

- `reports/gemma4_ollama_parallel.json`
- `reports/gemma4_ollama_batch*.json`（原单槽数据）
- `scripts/bench_ollama_batch.py`
- `/tmp/ollama_par/bench_p4_n16.json`（ollama_bench 交叉验证）
