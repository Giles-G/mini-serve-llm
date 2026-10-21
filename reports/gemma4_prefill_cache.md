# 4k/2k prefill 优化 + 持久 prefix cache（3/4/5）

日期：2026-10-21  
设备：RTX 3060 Laptop 6GB，Gemma4 E2B INT4 group64

## 状态

| 项 | 状态 | 结果 |
|---|---|---|
| 4. 4k prefill 2000 ms | 完成 | 4096: **2.308 → 1.647 s（-28.6%）**；2048: 0.967 → **0.817 s（-15.5%）** |
| 3. 单 kernel flash | 完成（prefill） | Triton flash prefill，支持 d=256/512、GQA、causal/sliding；2k/4k token parity 通过 |
| 5. 持久 prefix cache | 完成 | 19-token 重复 prefix：N=16 TTFT **0.466 → 0.037 s（-92%）**，e2e +65% |

## 4. 4k prefill：滑窗切片 + Triton flash

### 4.1 滑窗层按 window 切片

之前 sliding 层虽然 mask 只保留 512 window，但 K/V 仍然按完整 4096 计算，softmax 也是 4096 宽；这是纯浪费。

改成每个 q chunk：

```text
kv_start = max(0, start + q_start - window + 1)
kv_end   = total
K/V slice [kv_start:kv_end]
mask slice 同一区间
```

这样就只算真正可能被窗口选中的 KV。

结果：

| prompt | 优化前 | 优化后 |
| ---: | ---: | ---: |
| 2048 | 0.967 s | 0.961 s |
| 4096 | 2.308 s | **2.188 s（-5.2%）** |
| 4096 prefill tok/s | 1775 | **1872** |

2k/4k 的 prefill+8 decode token ID 与关闭切片时 **完全一致**。


### 4.2 Triton flash prefill（最终落地）

新增 `miniservellm/runtime/flash_prefill.py`：

- 一个 query-block 一个 head，FP32 online softmax；
- 支持 `d=256/512`、GQA（Hq/Hkv）、causal、sliding window、chunked prefill 的 `start/total`；
- 只用 FP16 输入/输出，softmax/累加 FP32；
- 默认只对 `total >= 256` 的 prefill 启用，避免 19-token/单请求回退；
- 启动时用 256-token dummy 输入预编译 D=256/512 两个 kernel，避免第一次长 prefill 吃 JIT 时间；
- 可关：`MINI_LLM_GEMMA4_PREFILL_FLASH=0`；阈值：`MINI_LLM_GEMMA4_PREFILL_FLASH_MIN_TOKENS`。

最终 prefill：

| prompt | 原路径 | window slice | **flash** |
| ---: | ---: | ---: | ---: |
| 2048 | 0.967 s | 0.961 s | **0.817 s（-15.5%）** |
| 4096 | 2.308 s | 2.188 s | **1.647 s（-28.6%）** |

4k prefill tok/s：**2487**；2k：**2507**。

验证：

- 2k/4k prefill + 8 decode token ID：flash on/off **完全一致**；
- 300-token prompt + chunk=64：flash on/off **完全一致**；
- N=2/4/8/16 batch parity：全部 true；
- short 19-token prefill 仍走原路径，TTFT 约 0.06 s，无回退。

flash 接管 attention 后，4k 已经低于 2.0 s 目标。

## 5. 持久 prefix cache

### 实现

在 `Gemma4EngineModelRunner` 里加了一个 **exact-prefix KV/logits cache**：

- key = `tuple(prompt_token_ids)`；
- 只缓存长度 ≤ `MINI_LLM_GEMMA4_PREFIX_CACHE_MAX_TOKENS`（默认 64）的完整 prompt；
- 命中时把缓存的每层 K/V 写回该请求的 paged KV blocks，并直接返回缓存的 last-token logits；
- 未命中时走原来的 prefill，并在结束时把 source layer 的 K/V 和 logits 存入 cache；
- LRU，默认最多 8 条 / 64 MB，可用 env 调整：
  - `MINI_LLM_GEMMA4_PREFIX_CACHE=0`
  - `MINI_LLM_GEMMA4_PREFIX_CACHE_ENTRIES`
  - `MINI_LLM_GEMMA4_PREFIX_CACHE_MAX_TOKENS`
  - `MINI_LLM_GEMMA4_PREFIX_CACHE_MAX_BYTES`

只对 `paged_attention` 生效；dense fallback 不启用。

### 结果（同一 prompt 重复请求）

| 配置 | cold TTFT | cache-hit TTFT | cold e2e | hit e2e |
| --- | ---: | ---: | ---: | ---: |
| N=16, 16 输出 | 0.466 s | **0.037 s** | 395.5 | **1426.6** |
| N=16, 64 输出 | ~0.45 s | **0.037 s** | ~941.5 | **1551.8** |
| N=4, 64 输出 | 0.134 s | **0.009 s** | 310.3 | **376.5** |
| N=8, 64 输出 | 0.263 s | **0.018 s** | 533.9 | **712.3** |
| N=1, 16 输出 | 0.065 s | **0.003 s** | 95.5 | **158.1** |

N=16 16 输出 cache 命中与 cold 的生成 token IDs **完全一致**。

### 注意

- 这个收益依赖 **prefix 命中**；不同 prompt 仍是 cold prefill。
- 只缓存 ≤64 token 的短 prefix，避免 4k KV 占显存。
- 缓存 K/V 是从第一次请求的 dense mirror clone 出来的，约每条约 1–2 MB（19 token，15 个 source layer）。
- 当前 free VRAM 约 0.646 GiB，cache 默认上限远小于这个值。

## 3. 单 kernel flash：已落地（prefill）

实现的是 **prefill 单 kernel flash**（`flash_prefill.py`），不是 decode flash：

- 一个 Triton kernel 完成 QK、mask、online softmax、PV，无中间 `[T,T]` score；
- d=256/512；GQA；causal/sliding；支持 chunked prefill 的 start/total；
- 只对 `total >= 256` 启用，短 prefill/单请求走原路径，避免回退。

效果已经包含在上面 2k/4k prefill 数字里；2k -15.5%，4k -28.6%。

## 验证

- N=2/4/8/16 batch token parity：全部 true
- 2k/4k prefill+decode token ID：window slice on/off 一致
- server tests：75 passed
- kernel tests：688 passed
- 未 commit、未 push
