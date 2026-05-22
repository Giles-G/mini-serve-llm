"""神经网络基础算子

第五阶段新增：手写模型前向所需的底层算子，
包括 RMSNorm、SiLU+Mul、RoPE、Causal Attention 等。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """线性变换（全连接层）

    公式：
        y = x @ W.T + b

    封装 F.linear，其中 weight 的 shape 为 [out_features, in_features]，
    F.linear 内部自动执行 x @ weight.T 的计算。

    Args:
        x: 输入张量 [..., in_features]
        weight: 权重矩阵 [out_features, in_features]
        bias: 偏置向量 [out_features]，可选

    Returns:
        输出张量 [..., out_features]
    """
    return F.linear(x, weight, bias)


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
