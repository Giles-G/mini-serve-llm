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
✅ INT4 量化 (runtime group quantization, fallback dequantize+matmul, 未含 AWQ)

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

### D1: INT4 Fused Dequantize+Matmul Kernel（含 AWQ）

**问题**: 当前 P6 的 INT4 路径 fallback 到 `dequantize(w_q, scales) → F.linear(x, w_fp16)`，没有减少 HBM 流量。且量化算法是简单的 max-abs group quantization，未利用激活统计量，量化精度不如 AWQ。

**方案**: 分两步实现：

**D1a: AWQ 校准（Python 侧）**

```python
# 1. 跑一小批校准数据（16-32 个 prompt），收集每层输入激活
# 2. 计算 per-channel activation magnitude
# 3. 找 salient channels (top-k by activation magnitude)
# 4. 计算 per-channel AWQ scale: s_c = max(|w_c|) * (activation_magnitude_c) ^ alpha
#    alpha=0.5 是经验最优值
# 5. 应用 scale: w_scaled = w / s_c, x_input = x * s_c（推理时通过 pre-scale 激活实现）
```

**AWQ 核心公式**：

```
给定 layer input activation 统计量 a ∈ R^in_features (校准集平均 L2 norm):

salient_mask[c] = a[c] > percentile(a, 95%)              # 前 5% 通道为 salient
s[c] = mean(|w[:,c]|)^α × (a[c] / mean(a))^β              # AWQ scale, α=1, β=0.5
w'[:,c] = w[:,c] / s[c]                                    # 放大重要通道权重
x'[:,c] = x[:,c] * s[c]                                     # 对应缩小激活（推理时）

量化: scale_group = max(|w'_group|) / 7                   # 在 scaled 权重上做 group quantization
量化为: (w_q, group_scales, awq_scales)                    # awq_scales 用于推理时 pre-scale 激活
```

**D1b: Fused INT4 Matmul Kernel（CUDA 侧）**

warp-level dot product + on-the-fly dequantize，支持 batch=1-4 decode 场景。

```cpp
// grid:  [B, ceil(N_out / 32)]
// block: 128 threads (4 warps)
//
// 每个 block 处理 N_TILE=32 个输出维度的内积
//
// 内层循环:
//   for k_tile in range(0, K, K_TILE):
//       1. float4 加载 INT4 packed 权重 [32, K_TILE/8] 到 shared memory
//       2. __syncthreads
//       3. 解包 int4_val = (byte >> (bit_offset)) & 0xF → 偏移到 [-8,7]
//       4. 反量化 w_fp16 = int4_val * group_scale[group_idx]
//       5. 加载激活 x[k_tile:k_tile+K_TILE] 到寄存器
//       6. dot product (32 threads × 32 outputs, 每个 thread 负责 1 个输出)
//       7. warp_reduce_sum 归约，累加到 fp32 accumulator
//   写出: y[batch, out_start:out_start+32] = cast_fp16(accumulator)
```

**线程布局**:

```
Block: 128 threads = 4 warps (32 × 4)
  Warp 0: 处理 output[0:32] vs 所有 K（每个 lane 1 个输出）
  Warp 1-3: 同上，各处理 32 个输出（每个 block 共处理 128 个输出）
  K 维度按 K_TILE=64 分块迭代

Shared memory:
  [0..2047]: 权重 tile [128, 64/8] uint8 = 8KB（解包后变为 16KB fp16，不物化）
  total: 8KB < 48KB OK
```

**AWQ 推理时的额外操作**:

```python
# AWQ 量化后的 linear 调用:
def _awq_int4_linear(x, weight_packed, group_scales, awq_scales, bias):
    # Step 1: pre-scale 激活（AWQ 特有，开销 < 0.01ms）
    x_scaled = x * awq_scales  # [B, in_features]
    # Step 2: fused INT4 matmul（D1 kernel）
    y = int4_dequant_matmul(x_scaled, weight_packed, group_scales)
    return y + bias if bias is not None else y
```

**预期收益**: 量化场景下权重带宽 4x 减少（真 INT4 packed → 4 bit/weight），AWQ 精度损失从 ~12% 降到 ~3-5%，decode 吞吐 ~2x 提升。结合 P6 已有量化基础设施，D1 是 kernel 层替换。

**涉及文件**: `mini-llm-kernels/csrc/int4_matmul.cu` (新增), `miniservellm/quantization/awq.py` (新增), `nn_ops.py`

---

### B1: Paged Prefill Attention Kernel（已接入 block-aware 路径）

**问题**: 当前 prefill 路径 `batched_causal_attention_prefill` 先用 advanced indexing 把 paged KV gather 到 padded tensor `[N, max_kv_len, H_kv, D]`，再做标准 attention。gather 操作：
1. 额外 HBM 分配（max_kv_len 可能远大于实际 context）
2. 额外 KV 拷贝（N × max_kv_len × H_kv × D × 2 bytes）
3. padding 浪费计算

**方案**: 已在 `/Users/gengzhiqiang/User_Program/mini-llm-kernels` 实现 `paged_prefill_attention` CUDA kernel，支持 Q length > 1 + causal mask，按 `block_table` 跳读 KV，并用 online softmax 避免完整 padded KV 物化。主仓库通过同名 Python binding 调用，未编译 CUDA 扩展时使用等价 PyTorch fallback。

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

**涉及文件**: `miniservellm/runtime/nn_ops.py`, `miniservellm/runtime/model_runner.py`, `miniservellm/cache/kv_cache.py`（复用 block table 接口）；`/Users/gengzhiqiang/User_Program/mini-llm-kernels/csrc/prefill_attention.cu`, `csrc/bindings.cpp`, `setup.py`, `mini_llm_kernels/kernels/prefill_attention.py`

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
C1 (CUDA Graphs)        ★★★  10-30%     中等   已完成
D1 (INT4 GEMM + AWQ)    ★★★  量化 2x    高     量化场景核心（分 D1a AWQ 校准 + D1b fused kernel）
B1 (Paged Prefill)      ★★   prefill 30%+ 中等   已接入 block-aware eager 路径，CUDA binding 可替换
A1 (Speculative)        ★★   greedy 2-3x 高     算法层，独立
B2 (QKV+RoPE 融合)      ★    5-10%      中等   锦上添花
```

建议顺序: **C1（已完成）→ D1 → B1 → A1 → B2**

## 涉及文件总览

| 文件 | 改动项 |
|------|--------|
| `miniservellm/runtime/model_runner.py` | C1: CudaGraphRunner, A1: speculative decode |
| `miniservellm/runtime/cuda_graph_runner.py` | C1: 新增 |
| `miniservellm/runtime/inference_engine.py` | C1: graph replay 调度, A1: draft/verify 循环 |
| `miniservellm/runtime/nn_ops.py` | D1: int4_matmul + awq 调用, B1: block-aware paged prefill + CUDA binding hook, B2: qkv_rope_fused |
| `miniservellm/quantization/awq.py` | D1a: AWQ 校准模块（新增） |
| `mini-llm-kernels/csrc/int4_matmul.cu` | D1b: fused INT4 dequant+matmul kernel（新增） |
| `mini-llm-kernels/csrc/prefill_attention.cu` | B1: 新增 |
| `mini-llm-kernels/csrc/qkv_rope_fused.cu` | B2: 新增 |
| `mini-llm-kernels/setup.py` | 注册新 kernel |
| `mini-llm-kernels/mini_llm_kernels/kernels/*.py` | Python 绑定 |
