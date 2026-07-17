当前完成的是 **阶段 1：固定等长 batch serving**。后续建议按下面顺序推进。

## 阶段 2：异构 Prompt 的 length bucketing

目标：请求不必 token 长度完全一样。

- 等待队列按 prompt token 长度分 bucket，例如 `64 / 128 / 192 / 256`
- 同 bucket 内短 prompt 右侧 padding
- Prefill 使用 attention mask 屏蔽 padding
- 每个请求记录实际 `prompt_len`
- Decode 仍是固定 batch、共享 offset，不做中途补位

效果：
- 服务可以处理真实业务的不同长度 prompt
- padding 开销可控制
- 仍保持实现简单和较稳定的 MLX batch shape

验证：
- 17、64、65、127、128 token 的请求分 bucket 正确
- padding 不影响 token 输出
- 与单请求输出 token parity

## 阶段 3：Per-slot KV offset

目标：让同一个 batch 的不同请求拥有不同生成进度。

当前 KV cache：

```text
offset: int  // 全 batch 共用
```

改为：

```text
slot_offsets: mx.array[int32]  // shape [B]
slot_active: mx.array[bool]    // shape [B]
```

每行仅看自己的有效 KV：

```text
attention_mask[b, :, :, pos] = pos <= slot_offsets[b]
```

难点：KV 更新不能继续使用统一位置的 `slice_update`，需要支持：

```text
slot 0 -> 写入 position 438
slot 1 -> 写入 position 91
slot 2 -> 写入 position 772
```

第一版可以逐 slot 写入，先保证正确；后续再做 batched scatter/update 或 Metal kernel。

效果：
- 不同 `max_new_tokens`、不同完成进度可以处于同一 active batch
- 是 continuous batching 的数据基础

## 阶段 4：动态 slot refill，真正 Continuous Batching

目标：请求完成后立刻让新请求占用空 slot。

每个 decode tick：

```text
1. 对 active slots 做 batched decode
2. 逐行采样，检查 EOS / max_new / abort
3. 已完成请求释放 slot
4. 从兼容 bucket 的等待队列挑选新请求
5. 新请求 prefill 后写入空 slot
6. 下一个 tick 继续运行固定 shape [max_batch_size, 1]
```

效果：
- 接近 Ollama 的服务行为
- 长请求不阻塞短请求
- `concurrency=8, requests=32` 时才能持续维持高 GPU 利用率
- 请求 p50/p95 延迟和 aggregate throughput 都更有意义

## 阶段 5：批量 KV scatter/update 优化

目标：解决阶段 3/4 的性能核心风险。

如果每层、每 slot 分别更新 K/V：

```text
B × 24 × 2 次 KV update / decode tick
```

并发一大，Python/Metal dispatch 开销会抵消 batch 收益。

最终目标：

```text
每层：K 一次 batched update + V 一次 batched update
```

实现候选：

- 先确认 MLX 是否有可用 indexed/scatter update 原语
- 没有则写一个 Metal KV scatter kernel
- 输入：`slot_ids[B]`、`slot_offsets[B]`、`k_new[B,Hkv,1,D]`
- 输出：更新后的 `[B,Hkv,T,D]` KV cache

效果：
- 是 batch>4 后还能继续扩展的关键
- 避免动态 offset 写 KV 成为新瓶颈

## 阶段 6：按 bucket 缓存 `mx.compile` 批量 decode 图

目标：让 multi-batch 也吃到 compiled decode 的收益。

缓存 key：

```text
(quant_bits, batch_capacity, context_bucket, dtype)
```

例子：

```text
INT4 + B=8 + context=1024 -> 一份 compile 图
INT4 + B=8 + context=2048 -> 另一份图
```

active 请求不足 B 时，用 inactive slot mask 占位，保持 shape 不变：

```text
token_ids: [8, 1]
slot_active: [True, True, True, False, ...]
```

效果：
- 减少 batch decode 的 Python 构图和 kernel dispatch
- 对高并发、小模型 INT4，预期是阶段 4 后最值得做的优化之一
- 需要避免 bucket 太细导致 compile 图爆炸和 warmup 抖动

## 阶段 7：服务指标、backpressure 与保护机制

目标：让它从 benchmark engine 变成可以稳定压测的服务组件。

增加：

- `max_waiting_requests`
- `max_active_slots`
- `max_context_tokens`
- 超时与 cancel
- 排队时间、TTFT、decode latency、active batch size
- bucket 命中率、padding ratio、slot refill 次数
- 显存/统一内存水位与 admission 拒绝

压测对齐：

```text
MLX:
--max-batch-size 8
--requests 32
--context-len 128
--max-new 1024
--quant-bits 4

Ollama:
--concurrency 8
--requests 32
--context-len 128
--max-new 1024
```

核心指标：

```text
aggregate decode tok/s
TTFT p50 / p95
request latency p50 / p95
active batch size average
slot refill count
KV memory usage
```

建议下一步直接做 **阶段 2 + 阶段 3**。阶段 2 解决请求长度异构，阶段 3 解决生成进度异构；随后阶段 4 才能安全实现真正的 slot refill。



ReportID: fc86abb5-f6dc-4428-94a8-8abaad27ebcb
ConversationID: 8830732a-de29-4eb9-97ea-64fb27bda53c