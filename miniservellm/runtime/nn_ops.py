"""神经网络基础算子

第五阶段新增：手写模型前向所需的底层算子，
包括 RMSNorm、SiLU+Mul、RoPE、Causal Attention 等。

第八阶段新增：尝试加载 mini_llm_kernels 自定义 CUDA kernel，
失败时自动退回原有 PyTorch 实现，MPS/CPU 环境透明降级。
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Stage 8: 自定义 CUDA kernel（可选）
# 在有编译好的 mini_llm_kernels 扩展时启用，否则退回纯 PyTorch 实现。
# 设置环境变量 MINI_LLM_NO_CUSTOM_KERNELS=1 可在运行时强制关闭，
# 用于 A/B 性能对比或 fallback 路径验证。
# --------------------------------------------------------------------------- #
try:
    import mini_llm_kernels as _mkl
    _HAS_CUSTOM_KERNELS: bool = (
        _mkl._HAS_CUDA_OPS
        and os.environ.get("MINI_LLM_NO_CUSTOM_KERNELS", "0") != "1"
    )
except ImportError:
    _mkl = None  # type: ignore[assignment]
    _HAS_CUSTOM_KERNELS: bool = False


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """融合 RMSNorm + Residual Add（M3 优化）

    一次 kernel 完成：
        residual_out = x + residual
        x_normed     = RMSNorm(residual_out, gamma, eps)

    有 CUDA kernel 时调用 mini_llm_kernels.fused_add_rms_norm，
    否则退回等价的 PyTorch 实现（M1/CPU 透明降级）。

    Args:
        x:        [N, H]  attention/mlp 输出（delta）
        residual: [N, H]  上一子层的残差流
        gamma:    [H]     RMSNorm 可学习缩放参数
        eps:      float   防除零小量

    Returns:
        (x_normed, residual_out)
        x_normed:     [N, H]  归一化结果，供下一算子（QKV/MLP）使用
        residual_out: [N, H]  x + residual，作为下一子层的残差输入
    """
    if _HAS_CUSTOM_KERNELS and _mkl is not None and x.is_cuda and residual.is_cuda and gamma.is_cuda:
        return _mkl.fused_add_rms_norm(x, residual, gamma, eps)
    # PyTorch fallback
    orig_dtype = x.dtype
    residual_out = x + residual
    x_fp32 = residual_out.float()
    rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
    x_normed = (x_fp32 * rms).to(orig_dtype) * gamma
    return x_normed, residual_out


def decode_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_ctx: int = 0,
) -> torch.Tensor:
    """Block-aware Paged Decode Attention（M4 优化）

    直接按 block_table 跳着读 KV，配合 online softmax，
    消除 gather 拷贝 + 全量 score 矩阵写回。

    有 CUDA kernel 时调用 mini_llm_kernels.decode_paged_attention，
    否则退回等价 PyTorch 实现（gather + batched GQA matmul）。

    CUDA graph capture 时跳过 custom kernel（内部 fallback 用 .item() 不兼容），
    此时必须通过 max_ctx 预传入 context 长度。

    Args:
        q:            [N, H_q, D]
        k_cache:      [num_blocks, block_size, H_kv, D]
        v_cache:      [num_blocks, block_size, H_kv, D]
        block_table:  [N, max_blocks]  int32/int64
        context_lens: [N]              int32/int64
        max_ctx:      context 总长度（CUDA graph capture 时必须提供）

    Returns:
        out: [N, H_q, D]
    """
    if torch.cuda.is_current_stream_capturing():
        return _decode_paged_attention_fallback(
            q, k_cache, v_cache, block_table, context_lens, max_ctx
        )
    if _HAS_CUSTOM_KERNELS and _mkl is not None and q.is_cuda:
        return _mkl.decode_paged_attention(
            q, k_cache, v_cache,
            block_table.to(torch.int32),
            context_lens.to(torch.int32),
        )
    # PyTorch fallback（等价于原 gather + gathered_paged_kv_decode_attention）
    return _decode_paged_attention_fallback(q, k_cache, v_cache, block_table, context_lens, max_ctx)


def _decode_paged_attention_fallback(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_ctx: int = 0,
) -> torch.Tensor:
    """PyTorch fallback：gather + batched GQA attention

    max_ctx > 0 时使用该值作为 context 长度（用于 CUDA graph capture 兼容），
    否则从 context_lens 中 .item() 获取。
    """
    N, H_q, D = q.shape
    block_size = k_cache.size(1)
    H_kv = k_cache.size(2)
    if max_ctx <= 0:
        max_ctx = int(context_lens.max().item())
    max_blocks = block_table.size(1)

    pos = torch.arange(max_ctx, device=q.device, dtype=torch.long)
    blk_idx = pos // block_size
    blk_off  = pos % block_size

    bt = block_table[:, :max_blocks].long()
    bt_expanded = bt[:, blk_idx]                         # [N, max_ctx]

    k_padded = k_cache[bt_expanded, blk_off]             # [N, max_ctx, H_kv, D]
    v_padded = v_cache[bt_expanded, blk_off]

    group = H_q // H_kv
    q_grouped = q.view(N, H_kv, group, D)
    k_p = k_padded.permute(0, 2, 1, 3)
    v_p = v_padded.permute(0, 2, 1, 3)

    scale = D ** -0.5
    scores = torch.matmul(q_grouped, k_p.transpose(-1, -2)) * scale

    pos_idx = torch.arange(max_ctx, device=q.device).view(1, 1, 1, max_ctx)
    valid = pos_idx < context_lens.view(N, 1, 1, 1)
    scores = scores.masked_fill(~valid, float("-inf"))

    weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    out = torch.matmul(weights, v_p)
    return out.reshape(N, H_q, D)


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """线性变换（全连接层），自动检测量化权重。

    公式：
        y = x @ W.T + b

    当 weight 是 tuple (w_q, scales) 时使用 INT4 group quantization：
        y = x @ dequantize(w_q, scales).T + b

    Args:
        x: 输入张量 [..., in_features]
        weight: 权重矩阵 [out_features, in_features] 或 (w_q, scales) 量化 tuple
        bias: 偏置向量 [out_features]，可选

    Returns:
        输出张量 [..., out_features]
    """
    if isinstance(weight, tuple):
        return _int4_linear(x, weight, bias)
    return F.linear(x, weight, bias)


def _int4_linear(
    x: torch.Tensor,
    weight: tuple,  # (w_q_int8, scales_fp16) or (w_packed_uint8, group_scales, awq_scales)
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """INT4 quantized matmul with AWQ support.

    Supports two weight formats:

    1. Basic (P6): weight = (w_q_int8, scales_fp16)
       - w_q: [N, K] int8, values in [-8, 7]
       - scales: [N, K//group_size] fp16

    2. AWQ (D1a): weight = (w_packed, group_scales, awq_scales)
       - w_packed: [N//2, K] uint8, 2×INT4 per byte
       - group_scales: [N, K//group_size] fp16
       - awq_scales: [K] fp16 per-channel pre-scale for activations
    """
    if len(weight) == 3:
        # AWQ format: pre-scale activations, then INT4 matmul
        w_packed, group_scales, awq_scales = weight
        x_scaled = x.to(w_packed.dtype) * awq_scales  # pre-scale by AWQ channel
        # Prefer fused CUDA kernel; fallback to PyTorch unpack
        try:
            from mini_llm_kernels.kernels.int4_matmul import int4_dequant_matmul
            y = int4_dequant_matmul(x_scaled, w_packed, group_scales)
        except (ImportError, RuntimeError):
            from mini_llm_kernels.kernels.int4_matmul import _int4_dequant_matmul_pytorch
            y = _int4_dequant_matmul_pytorch(x_scaled, w_packed, group_scales)
    else:
        # Basic format: (w_packed_uint8, scales) is the native runtime format.
        # Keep accepting the legacy (w_q_int8, scales) format for compatibility.
        w_data, scales = weight
        group_size = w_data.shape[1] // scales.shape[1]

        if w_data.dtype == torch.uint8:
            w_packed = w_data
        else:
            if w_data.shape[0] % 2 != 0:
                raise ValueError(
                    f"INT4 CUDA packing requires an even output size, got {w_data.shape[0]}"
                )
            w_unsigned = (w_data.to(torch.int16) + 8).clamp(0, 15).to(torch.uint8)
            w_packed = w_unsigned[0::2] | (w_unsigned[1::2] << 4)

        # Prefer the fused CUDA kernel; fallback keeps the same packed layout.
        try:
            from mini_llm_kernels.kernels.int4_matmul import int4_dequant_matmul as _im
            y = _im(x, w_packed.contiguous(), scales)
        except (ImportError, RuntimeError):
            from mini_llm_kernels.kernels.int4_matmul import _int4_dequant_matmul_pytorch
            y = _int4_dequant_matmul_pytorch(x, w_packed, scales)

    return y if bias is None else y + bias


def quantize_weight_group(
    weight: torch.Tensor,
    bits: int = 4,
    group_size: int = 64,
) -> tuple:
    """对称 group quantization：将 FP16 权重压缩为 INT4。

    公式：
        scale = max(|w_group|) / (2^{bits-1} - 1)
        w_q   = round(w_group / scale).clamp(-(2^{bits-1}), 2^{bits-1}-1)

    Args:
        weight: [out_features, in_features] fp16
        bits: 量化位宽（4 或 8）
        group_size: 每组共享 scale 的输入维度大小

    Returns:
        (w_q, scales): 量化权重和 scale
    """
    if weight.dim() != 2:
        raise ValueError("quantize_weight_group requires 2D weight [out, in]")
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(f"in_features ({in_features}) must be divisible by group_size ({group_size})")

    num_groups = in_features // group_size
    w_reshaped = weight.view(out_features, num_groups, group_size)

    max_val = 2 ** (bits - 1) - 1  # 7 for INT4
    w_abs = w_reshaped.abs().amax(dim=-1)  # [out, num_groups]
    scales = (w_abs / max_val).clamp(min=1e-6).to(weight.dtype)  # fp16

    w_q = torch.round(w_reshaped / scales.unsqueeze(-1)).clamp(-max_val, max_val).to(torch.int8)
    # Squeeze group dim back: [out, num_groups, group_size] → [out, in_features]
    return w_q.view(out_features, in_features), scales


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm（Root Mean Square Layer Normalization）

    与 LayerNorm 的区别：RMSNorm 不减均值，只除以 RMS，计算量更小。

    公式：
        RMS(x) = √(mean(x²))
        x_norm = x / RMS(x)
        output = x_norm * γ

    其中 γ (weight) 是可学习的缩放参数，eps 用于数值稳定性防止除零。

    实现中先转 fp32 计算再转回原精度，避免半精度下的数值溢出。

    Args:
        x: 输入张量 [..., hidden_size]
        weight: 可学习的缩放参数 γ [hidden_size]
        eps: 防止除零的小常数（如 1e-6）

    Returns:
        归一化后的张量 [..., hidden_size]
    """
    orig_dtype = x.dtype
    x_fp32 = x.float()  # 转 fp32 避免半精度下 pow(2) 溢出
    # mean(x²): 计算 RMS 的平方，即方差
    var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    # x / √(var + eps): rsqrt = 1/√x，等价于 x * (1/√(var+eps))
    x_norm = x_fp32 * torch.rsqrt(var + eps)
    # 转回原精度，再乘以可学习参数 γ
    return x_norm.to(orig_dtype) * weight


def silu_and_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SiLU 激活函数与门控相乘（SwiGLU 的核心操作）

    公式：
        output = SiLU(gate) * up
        其中 SiLU(x) = x · σ(x)，σ 为 sigmoid 函数

    SiLU 也叫 Swish 激活函数。门控机制让网络可以选择性地通过信息：
    sigmoid 输出 0~1 之间的值，起到"门"的作用，控制 up 信号的通过程度。

    在 SwiGLU MLP 中：
        gate = x @ W_gate.T
        up   = x @ W_up.T
        act  = SiLU(gate) * up      ← 本函数
        out  = act @ W_down.T

    Args:
        gate: Gate 投影结果 [..., intermediate_size]
        up: Up 投影结果 [..., intermediate_size]

    Returns:
        门控激活结果 [..., intermediate_size]
    """
    return F.silu(gate) * up


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """RoPE 的旋转操作：将向量后半部分取反后与前半部分交换拼接

    这是 RoPE 旋转矩阵乘法的等价实现（避免显式构造稀疏旋转矩阵）。

    将 x 沿最后一个维度分为两半：
        x = [x₁, x₂]   其中 x₁ = x[..., :d/2], x₂ = x[..., d/2:]

    旋转操作：
        rotate_half(x) = [-x₂, x₁]

    这个操作等价于对每对 (x_{2i}, x_{2i+1}) 执行 2D 旋转的矩阵乘法：
        [cos θ  -sin θ] [x_{2i}  ]   [x_{2i} cos θ - x_{2i+1} sin θ]
        [sin θ   cos θ] [x_{2i+1}] = [x_{2i} sin θ + x_{2i+1} cos θ]

    但使用前半/后半分割而非奇偶分割，与 Qwen2/LLaMA 的 HF 实现保持一致。

    Args:
        x: 输入张量 [..., head_dim]，head_dim 必须为偶数

    Returns:
        旋转后的张量 [..., head_dim]
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]    # 前半部分 x₁
    x2 = x[..., half:]    # 后半部分 x₂
    return torch.cat((-x2, x1), dim=-1)  # [-x₂, x₁]


def build_rope_cache(
    max_seq_len: int,
    head_dim: int,
    theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """预计算 RoPE 的 cos/sin 缓存

    RoPE (Rotary Position Embedding) 通过旋转矩阵将位置信息编码到 Q/K 中，
    使得内积自然包含相对位置信息： <RoPE(q_m), RoPE(k_n)> = f(m-n)

    公式：
        频率：θ_i = 1 / (base^(2i/d))    i = 0, 1, ..., d/2-1
        位置频率：freqs[m, i] = m · θ_i    m = 0, 1, ..., max_seq_len-1
        cos_cache[m, :] = cos(freqs[m, :])  重复拼接以匹配 head_dim
        sin_cache[m, :] = sin(freqs[m, :])

    其中 base 即 theta 参数（Qwen2.5 为 10000.0，LLaMA 为 10000.0）。

    预计算后，apply_rope 只需查表即可，避免每次前向重复计算三角函数。

    Args:
        max_seq_len: 最大序列长度
        head_dim: 每个注意力头的维度（必须为偶数）
        theta: RoPE 的 base 值（频率基数）
        device: 计算设备
        dtype: 输出数据类型

    Returns:
        (cos_cache, sin_cache): 各为 [max_seq_len, head_dim]
    """
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE head_dim must be even, got {head_dim}")
    # θ_i = 1 / (base^(2i/d))，i = 0, 2, 4, ..., d-2
    # shape: [head_dim / 2]
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    # 位置索引 [0, 1, ..., max_seq_len-1]
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    # 外积：freqs[m, i] = m * θ_i，shape: [max_seq_len, head_dim/2]
    freqs = torch.outer(t, inv_freq)
    # 重复 freqs 以匹配 head_dim（与 HF 实现一致）
    # [max_seq_len, head_dim/2] → [max_seq_len, head_dim]
    # 这样可以直接与 [head_dim] 维度的向量做逐元素乘法
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
):
    """对 Q/K 应用 RoPE 位置编码

    RoPE 的核心思想：对每对 (x_{2i}, x_{2i+1}) 施加旋转，旋转角度与位置成正比。
    这样两个位置的内积只依赖相对位置差。

    公式（使用前半/后半分割实现）：
        RoPE(x, m) = x ⊙ cos(m·θ) + rotate_half(x) ⊙ sin(m·θ)

    展开即为旋转矩阵乘法：
        x'_{2i}   = x_{2i} cos(m·θ_i) - x_{2i+1} sin(m·θ_i)
        x'_{2i+1} = x_{2i} sin(m·θ_i) + x_{2i+1} cos(m·θ_i)

    其中 m 是位置索引，θ_i 是第 i 组的频率。

    Args:
        q: Query 张量 [seq_len, num_heads, head_dim]
        k: Key 张量 [seq_len, num_heads, head_dim]
        positions: 位置索引 [seq_len]，每个 token 的绝对位置
        cos_cache: 预计算的 cos 缓存 [max_seq_len, head_dim]
        sin_cache: 预计算的 sin 缓存 [max_seq_len, head_dim]

    Returns:
        (q_out, k_out): 应用 RoPE 后的 Q/K，shape 不变
    """
    # 按位置索引查表，unsqueeze(1) 广播到 head 维度
    # cos: [seq_len, 1, head_dim]
    cos = cos_cache.index_select(0, positions).unsqueeze(1)
    sin = sin_cache.index_select(0, positions).unsqueeze(1)
    # RoPE 核心公式：x' = x * cos + rotate_half(x) * sin
    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    return q_out, k_out


def repeat_kv(kv: torch.Tensor, num_query_heads: int) -> torch.Tensor:
    """GQA 中将 KV head 重复扩展到与 Q head 相同数量

    GQA (Grouped Query Attention) 中 num_kv_heads < num_q_heads，
    每个 KV head 被同一组内的多个 Q head 共享。

    例如 Qwen2.5-0.5B：num_q_heads=14, num_kv_heads=2
    → 每 7 个 Q head 共享 1 个 KV head
    → 每个 KV head 重复 7 次

    公式：
        repeat_ratio = num_q_heads / num_kv_heads
        kv_expanded[kv_head * r + j, ...] = kv[kv_head, ...]   j = 0..r-1

    Args:
        kv: KV 张量 [seq_len, num_kv_heads, head_dim]
        num_query_heads: Query head 数量

    Returns:
        扩展后的 KV 张量 [seq_len, num_q_heads, head_dim]
    """
    num_kv_heads = kv.shape[1]
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    r = num_query_heads // num_kv_heads  # 重复比例
    if r == 1:
        return kv  # MHA（Multi-Head Attention）无需重复
    # 沿 head 维度重复：每个 KV head 连续重复 r 次
    return kv.repeat_interleave(r, dim=1)


def causal_attention_single_query(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """单 query token 的因果注意力（Decode 阶段使用）

    Decode 阶段只有 1 个 query token，无需 causal mask
    （因为它关注的是之前所有 token，天然满足因果性）。

    公式：
        scores = Q @ K.T / √d            # [H, 1, S]，S 为上下文长度
        weights = softmax(scores)         # [H, 1, S]
        output = weights @ V              # [H, 1, D]

    Args:
        q: Query [num_heads, head_dim]
        k: Key [seq_len, num_heads, head_dim]
        v: Value [seq_len, num_heads, head_dim]

    Returns:
        注意力输出 [num_heads, head_dim]
    """
    H, D = q.shape
    S = k.shape[0]
    q_3d = q.unsqueeze(1)        # [H, 1, D]
    k_t = k.permute(1, 2, 0)     # [H, D, S]
    v_t = v.permute(1, 0, 2)     # [H, S, D]

    # 缩放因子 1/√d，防止点积过大导致 softmax 饱和
    scale = D ** -0.5
    scores = torch.bmm(q_3d, k_t) * scale  # [H, 1, S]
    # softmax 在 float32 下计算，避免 fp16 数值溢出
    weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)  # [H, 1, S]
    out = torch.bmm(weights, v_t)  # [H, 1, D]
    return out.squeeze(1)  # [H, D]


def gathered_paged_kv_decode_attention(
    q: torch.Tensor,
    k_padded: torch.Tensor,
    v_padded: torch.Tensor,
    context_lens: torch.Tensor,
    num_q_heads: int,
) -> torch.Tensor:
    """N 个请求批量 decode attention（基于 paged KV gather 后的 padded 实现）

    ⚠️ 命名澄清：这 *不是* vLLM 那种「真 paged attention」kernel。

    真 paged attention 的关键是：在 attention kernel 内部直接按 block_table
    跳着读取离散物理 block 的 KV，配合 shared memory + online softmax 完成
    QK/softmax/PV 全流程，KV 不会被物化为连续张量。

    本实现做的事情：
      1. 上层（KVCacheManager.gather_kv_decode_batch）已经用 advanced indexing
         把每请求的 paged KV 物化成 padded 稠密张量
         k_padded/v_padded: [N, max_ctx, num_kv_heads, head_dim]
      2. 本函数只是在这块 padded 稠密张量上做一次普通 batched attention，
         用 context_lens mask 屏蔽 padding 位

    所以「paged」体现在 *存储侧*（KV 物理 block 离散 + block_table 映射），
    *计算侧* 仍是常规的 gather → batched matmul，不是 paged kernel。
    真正的 paged kernel 留给后续 CUDA 阶段（mini-llm-kernels）实现。

    GQA 不物化 repeat_kv：把 Q 的 head 维拆成 (kv_heads, group)，
    让 KV head 维直接广播参与计算，避免 group 倍显存拷贝。

    形状变换：
        q                          [N, num_q_heads, head_dim]
        → reshape                  [N, num_kv_heads, group, head_dim]
        k_padded                   [N, max_ctx, num_kv_heads, head_dim]
        → permute                  [N, num_kv_heads, max_ctx, head_dim]

        scores = q_grouped @ k_p.T  [N, num_kv_heads, group, max_ctx]
        + 上下文长度 mask
        weights = softmax(scores)
        out = weights @ v_p         [N, num_kv_heads, group, head_dim]
        → reshape                  [N, num_q_heads, head_dim]

    Args:
        q: Query [N, num_q_heads, head_dim]
        k_padded: Key, padding 后 [N, max_ctx, num_kv_heads, head_dim]
        v_padded: Value, padding 后 [N, max_ctx, num_kv_heads, head_dim]
        context_lens: 每请求的真实上下文长度 [N]，long
        num_q_heads: Q 的 head 数（用于 GQA 拆分）

    Returns:
        Attention 输出 [N, num_q_heads, head_dim]
    """
    N, H_q, D = q.shape
    _, max_ctx, H_kv, _ = k_padded.shape
    assert H_q % H_kv == 0, "num_q_heads must be divisible by num_kv_heads"

    if q.device.type == "mps":
        q_sdpa = q.unsqueeze(2)  # [N, H_q, 1, D]
        k_sdpa = k_padded.permute(0, 2, 1, 3)  # [N, H_kv, S, D]
        v_sdpa = v_padded.permute(0, 2, 1, 3)
        positions = torch.arange(max_ctx, device=q.device).view(1, 1, 1, max_ctx)
        valid = positions < context_lens.view(N, 1, 1, 1)
        return F.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            attn_mask=valid,
            dropout_p=0.0,
            enable_gqa=H_q != H_kv,
        ).squeeze(2)

    group = H_q // H_kv

    # GQA 视图：[N, H_kv, group, D]
    q_grouped = q.view(N, H_kv, group, D)
    # KV 转 [N, H_kv, max_ctx, D]
    k_p = k_padded.permute(0, 2, 1, 3)
    v_p = v_padded.permute(0, 2, 1, 3)

    scale = D ** -0.5
    # [N, H_kv, group, D] @ [N, H_kv, D, max_ctx] = [N, H_kv, group, max_ctx]
    scores = torch.matmul(q_grouped, k_p.transpose(-1, -2)) * scale

    # 长度 mask：超出真实 context 的位置置 -inf
    pos = torch.arange(max_ctx, device=q.device).view(1, 1, 1, max_ctx)
    valid = pos < context_lens.view(N, 1, 1, 1)
    scores = scores.masked_fill(~valid, float("-inf"))

    # softmax 在 fp32 下计算，转回原 dtype
    weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    # [N, H_kv, group, max_ctx] @ [N, H_kv, max_ctx, D] = [N, H_kv, group, D]
    out = torch.matmul(weights, v_p)
    # 还原回 [N, num_q_heads, D]
    return out.reshape(N, H_q, D)


def batched_causal_attention_prefill(
    q_padded: torch.Tensor,
    k_padded: torch.Tensor,
    v_padded: torch.Tensor,
    chunk_lens: torch.Tensor,
    history_lens: torch.Tensor,
    num_q_heads: int,
) -> torch.Tensor:
    """N 个请求的批量 prefill attention（block-diagonal causal mask + GQA 不物化）

    ⚠️ 同 gathered_paged_kv_decode_attention：本函数也不是 paged kernel。
    KV 由上层从 paged 物理 block 通过 advanced indexing 物化成 padded 稠密张量
    后才传入；本函数只在 padded 张量上做常规的 batched matmul + mask。

    每个请求 i 的 chunk 长度 T_i、历史长度 history_i 各异，但已 pad 到统一形状。
    每个 query（chunk 中位置 p）只能关注本请求的 [0, history_i + p]，
    跨请求互不可见，且超出 chunk 真实长度的 padding query 不计入。

    形状：
        q_padded:        [N, max_chunk_len, num_q_heads, head_dim]
        k_padded:        [N, max_kv_len,    num_kv_heads, head_dim]
        v_padded:        [N, max_kv_len,    num_kv_heads, head_dim]
        chunk_lens:      [N]  long, 每请求 chunk 的真实长度
        history_lens:    [N]  long, 每请求 chunk 之前的历史长度
        max_kv_len = max(history_i + chunk_i) for i

    GQA 不物化：把 Q 的 head 维拆成 (kv_heads, group)，让 KV head 维直接广播。

    返回：
        attention 输出 [N, max_chunk_len, num_q_heads, head_dim]
        （调用方按 chunk_lens 切回真实 token 顺序）
    """
    N, T, H_q, D = q_padded.shape
    _, S, H_kv, _ = k_padded.shape
    assert H_q % H_kv == 0

    if q_padded.device.type == "mps":
        q_sdpa = q_padded.permute(0, 2, 1, 3)  # [N, H_q, T, D]
        k_sdpa = k_padded.permute(0, 2, 1, 3)  # [N, H_kv, S, D]
        v_sdpa = v_padded.permute(0, 2, 1, 3)
        q_pos = torch.arange(T, device=q_padded.device).view(1, T, 1)
        k_pos = torch.arange(S, device=q_padded.device).view(1, 1, S)
        valid_q = q_pos < chunk_lens.view(N, 1, 1)
        causal_bound = history_lens.view(N, 1, 1) + q_pos + 1
        valid = (valid_q & (k_pos < causal_bound)).view(N, 1, T, S)
        out = F.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            attn_mask=valid,
            dropout_p=0.0,
            enable_gqa=H_q != H_kv,
        )
        return out.permute(0, 2, 1, 3)

    group = H_q // H_kv

    # Q reshape 为 [N, T, H_kv, group, D] → [N, H_kv, group, T, D]
    q_g = q_padded.view(N, T, H_kv, group, D).permute(0, 2, 3, 1, 4)
    # K/V → [N, H_kv, S, D]
    k_p = k_padded.permute(0, 2, 1, 3)
    v_p = v_padded.permute(0, 2, 1, 3)

    scale = D ** -0.5
    # scores: [N, H_kv, group, T, D] @ [N, H_kv, 1, D, S] = [N, H_kv, group, T, S]
    scores = torch.matmul(q_g, k_p.transpose(-1, -2).unsqueeze(2)) * scale

    # Mask 构造：[N, T, S]，True = 保留
    q_pos = torch.arange(T, device=q_padded.device).view(1, T, 1)
    k_pos = torch.arange(S, device=q_padded.device).view(1, 1, S)
    valid_q = q_pos < chunk_lens.view(N, 1, 1)                           # [N, T, 1]
    causal_bound = history_lens.view(N, 1, 1) + q_pos + 1                # [N, T, 1]
    valid_k = k_pos < causal_bound                                       # [N, T, S]
    mask = valid_q & valid_k                                             # [N, T, S]
    # 广播到 [N, 1, 1, T, S]
    scores = scores.masked_fill(~mask.view(N, 1, 1, T, S), float("-inf"))

    weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
    # out: [N, H_kv, group, T, S] @ [N, H_kv, 1, S, D] = [N, H_kv, group, T, D]
    out = torch.matmul(weights, v_p.unsqueeze(2))
    # → [N, T, H_q, D]
    return out.permute(0, 3, 1, 2, 4).reshape(N, T, H_q, D)


def causal_attention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Prefill 阶段的因果注意力

    Prefill 阶段有多个 query token（即 chunk 中的所有 token），
    需要应用 causal mask 保证每个 token 只能关注自身及之前的 token。

    公式：
        scores = Q @ K.T / √d                        # [H, T, S]
        scores[i,j] = -∞  if j > history_len + i     # Causal mask
        weights = softmax(scores)                     # [H, T, S]
        output = weights @ V                          # [H, T, D]

    Causal mask 解释：
        K 的前 history_len 个是历史 token（chunk 之前的），
        后 T 个是当前 chunk 的 token。
        当前 chunk 中位置 i 的 query 可以关注：
        - 全部 history_len 个历史 token
        - 当前 chunk 中位置 0~i 的 token
        即 j ≤ history_len + i

    Args:
        q: Query [t_q, num_heads, head_dim]
        k: Key [s_k, num_heads, head_dim]，s_k = history_len + t_q
        v: Value [s_k, num_heads, head_dim]

    Returns:
        注意力输出 [t_q, num_heads, head_dim]
    """
    t_q = q.shape[0]   # 当前 chunk 的 token 数
    s_k = k.shape[0]   # 总上下文长度（历史 + 当前 chunk）
    history_len = s_k - t_q  # 历史 KV Cache 的长度
    if history_len < 0:
        raise ValueError("Invalid attention shapes.")

    D = q.shape[-1]
    H = q.shape[1]
    scale = D ** -0.5

    q_t = q.permute(1, 0, 2)   # [H, T, D]
    k_t = k.permute(1, 2, 0)   # [H, D, S]
    v_t = v.permute(1, 0, 2)   # [H, S, D]

    # Q @ K.T / √d
    scores = torch.bmm(q_t, k_t) * scale  # [H, T, S]

    # 构造 causal mask：
    # q_pos: chunk 内的相对位置 [T, 1]
    # k_pos: 全局 K 的位置 [1, S]
    # 条件 j > history_len + i 为 True 时，mask 掉（设为 -inf）
    q_pos = torch.arange(t_q, device=q.device).unsqueeze(1)
    k_pos = torch.arange(s_k, device=q.device).unsqueeze(0)
    mask = k_pos > (history_len + q_pos)  # True = 需要屏蔽
    scores.masked_fill_(mask.unsqueeze(0), float("-inf"))

    # softmax 在 float32 下计算，避免 fp16 数值溢出
    weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)  # [H, T, S]
    out = torch.bmm(weights, v_t)  # [H, T, D]
    return out.permute(1, 0, 2)  # [T, H, D]
