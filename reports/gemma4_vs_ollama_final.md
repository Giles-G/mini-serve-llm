# Gemma4 E2B 多请求：项目 vs Ollama（已修正并发口径）

日期：2026-10-21  
设备：RTX 3060 Laptop 6GB；同一台机器、同一组中文 prompt  
项目：INT4 group64 + 自研 CUDA kernel  
Ollama：`gemma4:e2b` Q4_K_M + llama.cpp，`num_ctx=2048`，`think=false`，0.32.1

> **重要修正**
>
> 之前的对比使用系统里默认运行的 Ollama。日志确认它只有 **1 个 slot**：
>
> ```text
> srv load_model: initializing, n_slots = 1, n_ctx_slot = 2048
> ```
>
> 因此 N>1 的请求是排队的。Ollama 的 `eval_duration` 只统计请求真正被 decode 的时间，不含排队，所以旧表里 N=16 的 per-request 仍有 104 tok/s；但按 wall 算，每个请求平均只有 **5.7 tok/s**。
>
> 真实并发必须显式设置 `OLLAMA_NUM_PARALLEL`。本页现在把两种口径分开，并以 **Ollama 最佳并行配置** 做公平对比。详细复测见 `reports/gemma4_ollama_parallel.md`。

## 公平对比：项目 vs Ollama 最佳并行配置

Ollama 取每个 N 下 e2e 最好的配置：N=4 用 `OLLAMA_NUM_PARALLEL=4`，N=8/16 用 `OLLAMA_NUM_PARALLEL=8`。

### 64 输出

| N | 项目 e2e | Ollama e2e | e2e 倍数 | 项目 active per-req | Ollama active per-req | active 倍数 | 项目 wall-effective | Ollama wall-effective |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | **310.3** | 143.1 | **2.17×** | **91.2** | 52.3 | **1.74×** | **77.6** | 35.8 |
| 8 | **533.9** | 173.0 | **3.09×** | **88.0** | 29.4 | **2.99×** | **66.7** | 21.6 |
| 16 | **889.2** | 179.4 | **4.96×** | **89.4** | 29.1 | **3.07×** | **55.6** | 11.2 |

### 16 输出

| N | 项目 e2e | Ollama e2e | e2e 倍数 | 项目 active per-req | Ollama active per-req | active 倍数 | 项目 wall-effective | Ollama wall-effective |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | **216.2** | 70.6 | **3.06×** | **92.5** | 52.6 | **1.76×** | **54.1** | 17.6 |
| 8 | **295.7** | 91.4 | **3.23×** | **88.3** | 27.8 | **3.18×** | **37.0** | 11.4 |
| 16 | **395.5** | 105.9 | **3.74×** | **90.4** | 27.6 | **3.28×** | **24.7** | 6.6 |

`active per-req` 是 Ollama 的 `eval_duration` 口径，不含排队；`wall-effective = e2e / N`，包含排队和 prefill，更接近用户感知。项目 16 行同时 decode，两者差异主要来自 prefill（N=16 约 0.45 s）。

### 单请求

- 项目 decode **147 / 142.5 tok/s**（16/64 输出）；
- Ollama **102.5 / 105.5 tok/s**；
- 项目快 **1.43× / 1.35×**。

## Ollama 并行配置实测（64 输出）

### e2e tok/s

| N | p1 单槽 | p4 | p8 | 最佳 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 59.5 | - | 62.5 | 62.5 |
| 4 | 84.9 | **143.1** | 140.0 | **143.1** |
| 8 | - | - | **173.0** | **173.0** |
| 16 | 90.6 | 163.5 | **179.4** | **179.4** |

### active per-request decode（不含排队）

| N | p1 单槽 | p4 | p8 |
| ---: | ---: | ---: | ---: |
| 1 | 105.5 | - | 105.5 |
| 4 | 105.3 | 52.3 | 52.7 |
| 8 | - | - | **29.4** |
| 16 | 104.8 | 52.0 | **29.1** |

### 真实并发衰减

| 配置 | N=4 | N=8 | N=16 |
| --- | ---: | ---: | ---: |
| p1 active | 105.3 | - | 104.8 |
| p4 active | **52.3（50%）** | - | 52.0（50%） |
| p8 active | 52.7（50%） | **29.4（28%）** | 29.1（28%） |

结论很明确：**Ollama 真实并发时 per-request 衰减很大**。4 并发已经掉到约 50%，8 并发约 28%。之前“不衰减”确实是因为单槽排队，`eval_duration` 把排队时间排除掉了。

## 单槽服务数据（旧口径，仅作历史）

原服务 `n_slots=1`，N>1 请求排队。

### 16 输出

| N | e2e | active per-req | wall-effective |
| ---: | ---: | ---: | ---: |
| 1 | 25.7 | 102.9 | 25.7 |
| 2 | 39.5 | 101.8 | 19.8 |
| 4 | 53.2 | 100.7 | 13.3 |
| 8 | 60.2 | 101.3 | 7.5 |
| 16 | 64.2 | 101.1 | 4.0 |

### 64 输出

| N | e2e | active per-req | wall-effective |
| ---: | ---: | ---: | ---: |
| 1 | 61.0 | 105.6 | 61.0 |
| 2 | 73.4 | 105.7 | 36.7 |
| 4 | 84.3 | 104.4 | 21.1 |
| 8 | 89.2 | 104.8 | 11.2 |
| 16 | 90.3 | 104.1 | 5.7 |

旧表里的 104 是“轮到这个请求时 decode 多快”，不是“并发时每个请求多快”。真正用户感知的 per-request 要按 wall 算。

## 测试方法

### 1. 独立 Ollama 实例

不改系统服务，单独启动：

```bash
OLLAMA_HOST=127.0.0.1:11435 \
OLLAMA_MODELS=/usr/share/ollama/.ollama/models \
OLLAMA_NUM_PARALLEL=4 \
OLLAMA_MAX_LOADED_MODELS=1 \
ollama serve
```

分别测 `OLLAMA_NUM_PARALLEL=1/4/8`，日志确认 `n_slots = 1/4/8`。

### 2. 请求释放

- N 个 Python 线程；
- `threading.Barrier(N+1)` 让所有请求准备好；
- 主线程一次放行；
- 每个线程流式读 `/api/chat`，`think=false`；
- `temperature=0`，`num_ctx=2048`，`num_predict=16/64`；
- 每档 2 轮取中位数，先 warmup。

### 3. 指标

```text
e2e              = 总输出 token / wall
active per-req   = median((eval_count - 1) / eval_duration)   # 不含排队
wall-effective   = e2e / N                                    # 含排队+prefill
```

### 4. ollama_bench 交叉验证

用 [ollama_bench](https://github.com/dkruyt/ollama_bench) 在 `OLLAMA_NUM_PARALLEL=4`、16 并发、64 输出下复测：

```text
16/16 ok, throughput=2.30 rps
latency p50=4841.9 ms, TTFT p50=3664.8 ms
output tokens=1024, wall≈6.94 s
64 / (4.842 - 3.665) ≈ 54 tok/s active
```

与我们的 52 tok/s 基本一致，说明结论不是采样脚本造成的。

## 正确结论

1. 旧对比的 per-request 104 是单槽 active decode，不是并发吞吐；真实并发 per-request 会衰减。
2. Ollama 4 并发约 **50%**，8 并发约 **28%**；并没有接近 90%。
3. 项目在 N=4/8/16 的 64 输出 active per-request 约 **91/88/89**，相对单请求 142.5 约 **62–64%**，明显好于 Ollama 的 49%/28%。
4. 公平 e2e：项目领先 **2.17× / 3.09× / 4.96×**（64 输出，N=4/8/16），16 输出为 **3.06× / 3.23× / 3.74×**。
5. 之前的 9.84× 是“项目 16 行 batch vs Ollama 单槽排队”，不应作为并发能力结论；公平对比以本页为准。

## 注意

- 两边是同一个 Gemma4 E2B checkpoint，但 Ollama 是 Q4_K_M + llama.cpp，项目是 INT4 group64 + 自研 kernel，不是 kernel 等价对比。
- Ollama 的 `OLLAMA_FLASH_ATTENTION` 默认为 false；开启后并发数字可能不同，本页只测默认配置。
- 6GB 显存下 p8 已使用约 3.45 GiB；p16 未测，16 并发在 p8 下分两批。
