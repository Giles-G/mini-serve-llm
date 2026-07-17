# NVIDIA RTX 3060 第二轮优化方案

## 平台回顾

- RTX 3060 6GB, 28 SMs, 360 GB/s, Ampere sm_86
- INT8 Tensor Core: 256 TOPS, FP16 Tensor Core: 128 TFLOPS
- L2 Cache: 3MB

## 第一轮已实现（P0-P6）

✅ 跨 warp 归约 bug 修复
✅ Shared memory 超限检查
✅ fused_norm 消除重复 HBM 读
✅ KV 加载向量化 (float4)
✅ KV sequence 分区 (batch=1 SM 利用率 50% → 100%)
✅ 动态 VRAM 适配
✅ INT4 量化 (runtime group quantization, fallback dequantize+matmul)

---

## 第二轮优化项

### C1: CUDA Graphs — 消除 kernel launch 开销

**问题**: batch=1 decode 每步约 100+ kernel launch（24 层 × 4 matmul + norm + attention + rope + sampling），每个 launch ~5μs，总计 ~500μs/步 的 launch 开销。在 2.78ms/步的总时间里占 18%。

**方案**: 用 `torch.cuda.CUDAGraph` 捕获整个 decode step，后续 replay 零 launch 开销。

**实现**:

```python
# model_runner.py 新增
class CudaGraphRunner:
    def __init__(self, model_runner, batch_size, seq_len, num_decode_steps=1):
        self.graph = torch.cuda.CUDAGraph()
        self.static_tokens = torch.zeros(batch_size, 1, dtype=torch.long, device='cuda')
        self.static_logits = torch.zeros(batch_size, vocab_size, dtype=torch.float16, device='cuda')
        
        # 预热
        for _ in range(3):
            model_runner.decode_step(self.static_tokens)
        
        # 捕获
        with torch.cuda.graph(self.graph):
            self.static_logits = model_runner.decode_step(self.static_tokens)
    
    def replay(self, tokens):
        self.static_tokens.copy_(tokens)
        self.graph.replay()
        return self.static_logits.clone()
```

**难点**:
- KV cache 每步增长，shape 变化 → 需要预分配固定大小或用 static graph + 动态 offset
- 采样逻辑在 Python 侧 → 需要移入 graph 或在 graph 外做
- 只适用于固定 shape 的 decode（batch=1 或固定 batch_size）

**预期收益**: batch=1 decode 10-30% 加速

**涉及文件**: `model_runner.py`, `inference_engine.py`

---

### D1: INT4 Fused Dequantize+Matmul Kernel

**问题**: 当前 P6 的 INT4 路径 fallback 到 `dequantize(w_q, scales) → F.linear(x, w_fp16)`，先反量化出完整 FP16 权重再 matmul，没有减少 HBM 流量。权重仍以 INT8 存储（1 byte/weight），但反量化后临时 FP16 矩阵仍占 2 bytes/weight。

**方案**: 写自定义 CUDA kernel，直接从 INT4 packed 权重读取，on-the-fly 反量化并做 matmul，避免中间 FP16 权重物化。

**kernel 设计**:

```cpp
// 每个 thread block 处理 [out_tile, in_tile] 的 GEMM tile
// 权重以 int4 packed 存储 (2 weights per byte)
// 输入 x: [batch, in_features] fp16
// 权重 w_packed: [out_features // 2, in_features] uint8 (每字节 2 个 int4)
// scales: [out_features, in_features // group_size] fp16
//
// kernel 内部:
//   1. 从 w_packed 读取 1 byte → 解包 2 个 int4
//   2. 查 scales 表反量化: w_fp16 = int4_val * scale[group_idx]
//   3. 累加 x * w_fp16 到输出
//
// HBM 流量: uint8 (1 byte/weight) vs FP16 (2 bytes/weight) → 2x 减少
// 如果用真 INT4 packed (4 bit): 4x 减少
```

**替代方案**: 使用 CUTLASS 的 INT4 GEMM，或 PyTorch 2.4+ 的 `torch._weight_int4pack_mm`（如果格式匹配）。

**预期收益**: 量化场景下权重带宽 4x 减少，decode 吞吐 ~2x 提升

**涉及文件**: `mini-llm-kernels/csrc/int4_matmul.cu` (新增), `nn_ops.py`

---

### B1: Paged Prefill Attention Kernel

**问题**: 当前 prefill 路径 `batched_causal_attention_prefill` 先用 advanced indexing 把 paged KV gather 到 padded tensor `[N, max_kv_len, H_kv, D]`，再做标准 attention。gather 操作：
1. 额外 HBM 分配（max_kv_len 可能远大于实际 context）
2. 额外 KV 拷贝（N × max_kv_len × H_kv × D × 2 bytes）
3. padding 浪费计算

**方案**: 写 paged prefill attention kernel，类似 decode kernel 但支持 Q length > 1 + causal mask，直接按 block_table 跳读 KV。

**kernel 设计**:

```cpp
// grid: [N, H_q, ceil(T_q / TILE_Q)]
// 每个 block 处理 [TILE_Q, D] 的 Q tile
// 遍历 KV blocks（按 block_table），加载 K_tile/V_tile 到 shared memory
// 计算 [TILE_Q, block_size] 的 score 矩阵
// 应用 causal mask: score[i, j] = -inf if j > history_len + i
// Online softmax over (TILE_Q, total_KV) → 输出 [TILE_Q, D]
```

**预期收益**:
- 消除 KV gather 拷贝（省 N × max_kv_len × H_kv × D × 4 bytes HBM 流量）
- 消除 padding 浪费（实际 context vs max_kv_len 的差异）
- prefill 阶段 30-50% 加速

**涉及文件**: `mini-llm-kernels/csrc/prefill_attention.cu` (新增), `nn_ops.py`

---

### A1: Speculative Decoding（投机解码）

**问题**: Greedy decode 每步只生成 1 个 token，GPU 利用率低（0.5B 模型 matmul 很快，大量时间在 launch 开销和 Python 调度）。

**方案**: 用同一模型做 draft（连续预测 K 个 token），然后一次 forward 验证 K 个 token。

**流程**:

```
1. Draft: 用当前模型连续跑 K 步 greedy（不采样），得到 t1, t2, ..., tK
2. Verify: 一次 forward 计算 [t1, t2, ..., tK] 的 logits
3. Accept: 对比 draft 和 verify 的 argmax
   - 如果 t_i 匹配: 接受
   - 如果 t_i 不匹配: 拒绝，用 verify 的 logits 重新采样 t_i
4. 接受 n 个 token (0 ≤ n ≤ K)，回到 step 1
```

**实现要点**:
- Draft 阶段用已有的 unrolled decode（K 步连续提交）
- Verify 阶段用 prefill 路径（Q length = K）
- 需要 causal mask 保证 t_i 只能看到 t_0..t_{i-1}

**预期收益**: greedy decode 2-3x 加速（平均接受率 60-80%）

**涉及文件**: `inference_engine.py`, `model_runner.py`

---

### B2: QKV + RoPE 融合 Kernel

**问题**: 当前 QKV 投影后：
1. `qkv = linear(x, qkv_proj)` — 1 个 matmul kernel
2. `q, k, v = qkv.split(...)` — view 操作（零开销）
3. `q = q.reshape(...).transpose(...)` — 2 个 reshape/transpose kernel
4. `q = apply_rope(q, ...)` — 1 个 elementwise kernel
5. 同理 k 也要 reshape + rope

总计每层 8+ 个 kernel launch 用于 QKV 后处理。

**方案**: 写融合 kernel，一次完成 matmul + split + reshape + rope：

```cpp
// 输入: x [seq_len, hidden_size] fp16
// 权重: qkv_proj [qkv_dim, hidden_size] fp16
// 输出: q_rope [seq_len, H_q, D], k_rope [seq_len, H_kv, D], v [seq_len, H_kv, D]
//
// kernel 内部:
//   1. 计算 qkv = x @ qkv_proj^T (标准 GEMM)
//   2. 直接写入 reshape 后的位置 (避免额外 transpose)
//   3. 对 q/k 应用 RoPE (cos/sin 查表)
//   4. v 直接写入 (不需要 RoPE)
```

**预期收益**: 24 层 × 8 kernel → 24 层 × 1 kernel，减少 168 个 kernel launch/步，约 5-10% 加速

**涉及文件**: `mini-llm-kernels/csrc/qkv_rope_fused.cu` (新增), `nn_ops.py`

---

## 优先级与实施顺序

```
C1 (CUDA Graphs)        ★★★  10-30%     中等   先做，独立性强
D1 (INT4 GEMM kernel)   ★★★  量化2x     高     量化场景核心
B1 (Paged Prefill)      ★★   prefill 30%+ 中等   prefill 瓶颈时做
A1 (Speculative)        ★★   greedy 2-3x 高     算法层，独立
B2 (QKV+RoPE 融合)      ★    5-10%      中等   锦上添花
```

建议顺序: **C1 → D1 → B1 → A1 → B2**

## 涉及文件总览

| 文件 | 改动项 |
|------|--------|
| `miniservellm/runtime/model_runner.py` | C1: CudaGraphRunner, A1: speculative decode |
| `miniservellm/runtime/inference_engine.py` | C1: graph replay 调度, A1: draft/verify 循环 |
| `miniservellm/runtime/nn_ops.py` | D1: int4_matmul 调用, B1: paged prefill 调用, B2: qkv_rope_fused 调用 |
| `mini-llm-kernels/csrc/int4_matmul.cu` | D1: 新增 |
| `mini-llm-kernels/csrc/prefill_attention.cu` | B1: 新增 |
| `mini-llm-kernels/csrc/qkv_rope_fused.cu` | B2: 新增 |
| `mini-llm-kernels/setup.py` | 注册新 kernel |
| `mini-llm-kernels/mini_llm_kernels/kernels/*.py` | Python 绑定 |
