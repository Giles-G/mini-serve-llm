"""通用 Transformer 模型运行器

第五阶段核心：使用自研前向替代 HF model.forward()。
从 HF 模型提取权重，手写每一层的 forward 逻辑，
KV 直接写入 Paged KV Cache，不再依赖 HF 的 past_key_values。

通用化设计：通过 ModelConfig 参数化，支持 Qwen2/LLaMA/Mistral/Gemma 等
相同架构的模型（RMSNorm + GQA + RoPE + SwiGLU MLP）。

单层 Block 的前向流程（Pre-Norm 残差结构）：
    x_out = x + Attention(RMSNorm(x))
    x_out = x_out + MLP(RMSNorm(x_out))
    最终输出 = lm_head(RMSNorm(x_out))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from miniservellm.config import EngineConfig, ModelConfig
from miniservellm.cache.kv_cache import KVCacheManager
from miniservellm.runtime.metadata import PrefillRequestMetadata, DecodeRequestMetadata, SlotRef
from miniservellm.runtime.model_interface import PrefillModelOutput, DecodeModelOutput
from miniservellm.runtime.nn_ops import (
    apply_rope,
    batched_causal_attention_prefill,
    build_rope_cache,
    causal_attention_prefill,
    causal_attention_single_query,
    decode_paged_attention,
    fused_add_rms_norm,
    gathered_paged_kv_decode_attention,
    linear,
    quantize_weight_group,
    rms_norm,
    rotate_half,
    silu_and_mul,
)
from miniservellm.scheduler.request import Request


# ---------------------------------------------------------------------------
# 通用权重数据结构
# ---------------------------------------------------------------------------

@dataclass
class DecodeBatchGatherCtx:
    """N 个 decode 请求一次 step 内的批量索引/位置/槽位上下文

    跨 24 层共享（同一 batch 的 block_ids/offsets/context_lens/positions/write_slots
    在所有层都一样），避免每层重复构造同样的 LongTensor。
    """
    positions: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    write_slots: List[SlotRef] = field(default_factory=list)
    write_block_ids: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    write_block_offsets: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    block_ids: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    block_offsets: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    context_lens_tensor: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    valid_batch_size: int = 0
    block_table_tensor: Optional[torch.Tensor] = None


@dataclass
class PrefillBatchCtx:
    """N 个 prefill 请求一次 step 内的批量上下文（跨层共享）

    Attributes:
        positions: 拼接后所有 chunk token 的全局位置 [sum_T]（RoPE 用）
        write_slots: 拼接后所有 token 的写入 slot 列表，长度 sum_T
        write_block_ids: [sum_T]  写入用 block_id（slot_refs 预计算）
        write_block_offsets: [sum_T]  写入用块内偏移
        chunk_lens: [N]  每请求 chunk 真实长度
        history_lens: [N]  每请求 chunk 之前的历史长度
        chunk_offsets: [N+1]  每请求 chunk 在 [sum_T] 中的起止偏移
        max_chunk_len: 本批次最大 chunk 长度
        max_kv_len: 本批次最大 (history + chunk) 长度
        block_ids: [N, max_kv_len]  KV gather 用
        block_offsets: [N, max_kv_len]
        last_token_indices: [N]  每请求 chunk 最后一个 token 在 [sum_T] 中的索引（取 logits 用）
        pad_row: [sum_T]  q [sum_T,...] → q_padded [N, T_max,...] 时每个 flat token 的请求 idx
        pad_col: [sum_T]  对应的 chunk 内位置
    """
    positions: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    write_slots: List[SlotRef] = field(default_factory=list)
    write_block_ids: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    write_block_offsets: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    chunk_lens: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    history_lens: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    chunk_offsets: List[int] = field(default_factory=list)
    max_chunk_len: int = 0
    max_kv_len: int = 0
    block_ids: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    block_offsets: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    last_token_indices: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    pad_row: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    pad_col: torch.Tensor = field(default_factory=lambda: torch.empty(0))


@dataclass
class TransformerLayerWeights:
    """单层 Transformer Block 权重

    适用于 Qwen2/LLaMA/Mistral/Gemma 等相同架构的模型。
    bias 字段为 Optional，不同模型可能没有。

    Attention 部分（GQA）：
        qkv_proj: QKV 合并投影权重 [(num_q_heads + 2*num_kv_heads) * head_dim, hidden_size]
                  由 q_proj/k_proj/v_proj 在 dim=0 concat 而来（M2 优化三：减少 kernel launch）
        o_proj: Output 投影权重 [hidden_size, num_q_heads * head_dim]

    MLP 部分（SwiGLU）：
        gate_up_proj: Gate/Up 合并投影权重 [2 * intermediate_size, hidden_size]
        down_proj: Down 投影权重 [hidden_size, intermediate_size]

    Norm 部分：
        input_layernorm: Attention 前的 RMSNorm 权重 [hidden_size]
        post_attention_layernorm: MLP 前的 RMSNorm 权重 [hidden_size]
    """
    qkv_proj: torch.Tensor
    o_proj: torch.Tensor
    gate_up_proj: torch.Tensor
    down_proj: torch.Tensor
    input_layernorm: torch.Tensor
    post_attention_layernorm: torch.Tensor
    qkv_proj_bias: Optional[torch.Tensor] = None
    o_proj_bias: Optional[torch.Tensor] = None


@dataclass
class TransformerWeights:
    """完整 Transformer 模型权重

    Attributes:
        embed_tokens: Token Embedding 矩阵 [vocab_size, hidden_size]
        layers: 每层的权重列表，长度 = num_hidden_layers
        final_norm: 最终 RMSNorm 权重 [hidden_size]
        lm_head: 语言模型头（logits 投影）[vocab_size, hidden_size]
    """
    embed_tokens: torch.Tensor
    layers: List[TransformerLayerWeights]
    final_norm: torch.Tensor
    lm_head: torch.Tensor


# ---------------------------------------------------------------------------
# 单层 Block 执行器
# ---------------------------------------------------------------------------

class TransformerBlockRunner:
    """单层 Transformer Block 执行器

    通用结构（Pre-Norm 残差）：

        # Attention 子层
        residual = x
        x = RMSNorm(x, γ_attn)
        x = residual + Attention(x)          # x = x + Attention(RMSNorm(x))

        # MLP 子层
        residual = x
        x = RMSNorm(x, γ_mlp)
        x = residual + SwiGLU_MLP(x)         # x = x + MLP(RMSNorm(x))
    """

    def __init__(
        self,
        layer_idx: int,
        model_config: ModelConfig,
        layer_weights: TransformerLayerWeights,
        kv_cache_manager: KVCacheManager,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> None:
        """初始化单层 Block 执行器

        Args:
            layer_idx: 层编号（0-based），用于索引 KV Cache 中的对应层
            model_config: 模型配置（头数、hidden_size 等）
            layer_weights: 本层的权重
            kv_cache_manager: Paged KV Cache 管理器
            rope_cos: RoPE cos 缓存 [max_seq_len, head_dim]
            rope_sin: RoPE sin 缓存 [max_seq_len, head_dim]
        """
        self.layer_idx = layer_idx
        self.model_config = model_config
        self.weights = layer_weights
        self.kv_cache_manager = kv_cache_manager
        self.rope_cos = rope_cos
        self.rope_sin = rope_sin

        self.hidden_size = model_config.hidden_size
        self.num_q_heads = model_config.num_attention_heads
        self.num_kv_heads = model_config.num_key_value_heads
        self.head_dim = model_config.head_dim

    def _project_qkv(self, x: torch.Tensor):
        """QKV 线性投影（M2 优化：单次大 GEMM + split）

        将 q/k/v 三个独立 GEMM 合并为一次：
            qkv = x @ W_qkv.T + b_qkv
        再按维度 split 回 Q/K/V，split 几乎零开销（共享存储）。

        W_qkv 在权重加载时由 cat([W_q, W_k, W_v], dim=0) 构造，
        bias 同理（若存在）。

        Args:
            x: 输入隐藏状态 [seq_len, hidden_size]

        Returns:
            (q, k, v) 三元组，reshape 为多头格式
        """
        q_dim = self.num_q_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        qkv = linear(x, self.weights.qkv_proj, self.weights.qkv_proj_bias)
        q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
        q = q.view(x.shape[0], self.num_q_heads, self.head_dim)
        k = k.view(x.shape[0], self.num_kv_heads, self.head_dim)
        v = v.view(x.shape[0], self.num_kv_heads, self.head_dim)
        return q, k, v

    def _attention_prefill(self, hidden_states: torch.Tensor, meta: PrefillRequestMetadata, req: Request):
        """Prefill 阶段的注意力计算

        流程：
        1. QKV 投影 + RoPE 位置编码
        2. 从 Paged KV Cache 读取历史 K、V
        3. 将新 K、V 写入 Paged KV Cache
        4. 拼接历史 + 新的 K、V，GQA 扩展
        5. 计算因果注意力 + Output 投影

        公式：
            Q, K_new, V_new = W_qkv(x)                     # QKV 投影
            Q, K_new = RoPE(Q, K_new, positions)            # 位置编码
            K_full = concat(K_hist, K_new)                  # 拼接历史 + 新 KV
            V_full = concat(V_hist, V_new)
            K_full, V_full = repeat_kv(K_full, V_full)      # GQA: 扩展 KV head 到与 Q 相同
            Attn = softmax(Q @ K_full.T / √d) @ V_full      # 因果注意力
            Output = Attn @ W_o.T                            # Output 投影

        Args:
            hidden_states: 本 chunk 的隐藏状态 [chunk_len, hidden_size]
            meta: prefill 元数据（positions、write_slots、context_len 等）
            req: 请求对象（包含 block_table 用于 KV Cache 寻址）

        Returns:
            注意力输出 [chunk_len, hidden_size]
        """
        t = hidden_states.shape[0]  # chunk_len: 本步处理的 token 数
        q, k_new, v_new = self._project_qkv(hidden_states)

        # RoPE 位置编码：为每个 token 注入绝对位置信息
        positions = torch.tensor(meta.positions, device=hidden_states.device, dtype=torch.long)
        q, k_new = apply_rope(q, k_new, positions, self.rope_cos, self.rope_sin)

        # 从 Paged KV Cache 读取本 chunk 之前已有的历史 K、V
        # upto_logical_length=history_len 表示只读到 chunk 开始前的位置
        history_len = meta.context_len_before_chunk
        hist_k, hist_v = self.kv_cache_manager.gather_kv_for_request(
            layer_idx=self.layer_idx,
            req=req,
            upto_logical_length=history_len,
        )

        # 将新计算的 K、V 写入 Paged KV Cache 的预分配 slot
        self.kv_cache_manager.write_kv_for_tokens(
            layer_idx=self.layer_idx,
            slot_refs=meta.write_slots,
            k_values=k_new,
            v_values=v_new,
        )

        # 拼接历史 KV 和新 KV，构成完整的上下文
        # full_k: [history_len + chunk_len, num_kv_heads, head_dim]
        full_k = torch.cat([hist_k, k_new], dim=0)
        full_v = torch.cat([hist_v, v_new], dim=0)

        # GQA 扩展：将 KV head 重复到与 Q head 相同数量
        # 例如 num_q_heads=14, num_kv_heads=2 → 每个 KV head 重复 7 次
        # repeat_kv 后: [seq_len, num_q_heads, head_dim]
        full_k_q = repeat_kv(full_k, self.num_q_heads)
        full_v_q = repeat_kv(full_v, self.num_q_heads)

        # 因果注意力计算（含 causal mask，每个 token 只能关注自身及之前的 token）
        attn_out = causal_attention_prefill(q, full_k_q, full_v_q)
        # reshape: [t, num_q_heads, head_dim] → [t, hidden_size]
        attn_out = attn_out.reshape(t, self.hidden_size)
        # Output 投影：将多头注意力的拼接结果做线性变换，让不同 head 之间交互信息
        # 虽然输入输出维度都是 hidden_size，但 W_o 让每个位置的特征变为所有 head 的加权组合，
        # 而非各 head 互不相干的简单拼接
        return linear(attn_out, self.weights.o_proj, self.weights.o_proj_bias)

    def _attention_decode(self, hidden_state: torch.Tensor, meta: DecodeRequestMetadata, req: Request):
        """Decode 阶段的注意力计算（单 query token）

        流程与 prefill 类似，但只有 1 个 query token，无需 causal mask。

        公式：
            Q, K_new, V_new = W_qkv(x)                     # 只有 1 个 token
            Q, K_new = RoPE(Q, K_new, pos)                  # pos = 当前生成位置
            写入 K_new, V_new 到 Paged KV Cache
            K_full, V_full = 读取全部历史 + 新 KV
            K_full, V_full = repeat_kv(...)                   # GQA 扩展
            Attn = softmax(Q @ K_full.T / √d) @ V_full      # 单 query 注意力（无需 causal mask）
            Output = Attn @ W_o.T

        Args:
            hidden_state: 当前 token 的隐藏状态 [1, hidden_size]
            meta: decode 元数据（query_position、write_slot、context_len 等）
            req: 请求对象

        Returns:
            注意力输出 [1, hidden_size]
        """
        q, k_new, v_new = self._project_qkv(hidden_state)

        # Decode 只有一个 token，position 是当前生成位置
        pos = torch.tensor([meta.query_position], device=hidden_state.device, dtype=torch.long)
        q, k_new = apply_rope(q, k_new, pos, self.rope_cos, self.rope_sin)

        # 先写后读：将新 K、V 写入 Paged KV Cache
        self.kv_cache_manager.write_kv_for_tokens(
            layer_idx=self.layer_idx,
            slot_refs=[meta.write_slot],
            k_values=k_new,
            v_values=v_new,
        )

        # 读取全部 KV（包括刚写入的），context_len 已包含新 token
        full_k, full_v = self.kv_cache_manager.gather_kv_for_request(
            layer_idx=self.layer_idx,
            req=req,
            upto_logical_length=meta.context_len,
        )

        # GQA 扩展
        full_k_q = repeat_kv(full_k, self.num_q_heads)
        full_v_q = repeat_kv(full_v, self.num_q_heads)

        # 单 query 注意力：无需 causal mask（因为只有一个 query，且它可以关注所有已有 token）
        q_one = q[0]  # [num_q_heads, head_dim]
        attn_out = causal_attention_single_query(q_one, full_k_q, full_v_q)
        attn_out = attn_out.reshape(1, self.hidden_size)
        # Output 投影：将多头注意力的拼接结果做线性变换，让不同 head 之间交互信息
        return linear(attn_out, self.weights.o_proj, self.weights.o_proj_bias)

    def _mlp(self, x: torch.Tensor):
        """SwiGLU MLP 前向

        SwiGLU 是 GLU (Gated Linear Unit) 变体，用 SiLU 激活函数作为门控。

        公式：
            gate = x @ W_gate.T          # Gate 投影 [seq_len, intermediate_size]
            up   = x @ W_up.T            # Up 投影   [seq_len, intermediate_size]
            act  = SiLU(gate) * up        # 门控激活: SiLU(g) * u
            out  = act @ W_down.T         # Down 投影 [seq_len, hidden_size]

        其中 SiLU(x) = x * σ(x)（σ 为 sigmoid 函数），也称 swish 激活函数。
        门控机制让网络可以选择性地传递信息。

        Args:
            x: 输入隐藏状态 [seq_len, hidden_size]

        Returns:
            MLP 输出 [seq_len, hidden_size]
        """
        gate_up = linear(x, self.weights.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)
        act = silu_and_mul(gate, up)
        return linear(act, self.weights.down_proj)

    def forward_prefill_batch(
        self,
        hidden_states: torch.Tensor,
        ctx: "PrefillBatchCtx",
    ) -> torch.Tensor:
        """N 个 prefill 请求的批量前向（一次过一层）

        所有 token 算子（rms_norm/QKV/RoPE/KV 写/O_proj/MLP）在 [sum_T, hidden] 上一次做完。
        Attention 走 batched_causal_attention_prefill：
        - Q 按请求 pad 到 [N, max_chunk_len, H_q, D]
        - K/V 一次 advanced indexing 拿到 padded [N, max_kv_len, H_kv, D]
        - block-diagonal causal mask 同时屏蔽跨请求注意和未来 token

        Args:
            hidden_states: 拼接后的隐藏状态 [sum_T, hidden_size]
            ctx: 跨层共享的批量上下文

        Returns:
            本层输出 [sum_T, hidden_size]
        """
        # ---- Attention 子层 ----
        residual = hidden_states
        x = rms_norm(hidden_states, self.weights.input_layernorm, self.model_config.rms_norm_eps)

        # 批量 QKV：[sum_T, hidden] → q [sum_T, H_q, D], k/v [sum_T, H_kv, D]
        q, k_new, v_new = self._project_qkv(x)
        # 批量 RoPE
        q, k_new = apply_rope(q, k_new, ctx.positions, self.rope_cos, self.rope_sin)

        # 批量 KV 写入（先写后读：写完 chunk KV 再 gather full = history + chunk）
        self.kv_cache_manager.write_kv_for_tokens_indexed(
            layer_idx=self.layer_idx,
            block_ids=ctx.write_block_ids,
            block_offsets=ctx.write_block_offsets,
            k_values=k_new,
            v_values=v_new,
        )

        # 批量 KV gather：[N, max_kv_len, H_kv, D]
        k_padded = self.kv_cache_manager.k_cache[
            self.layer_idx, ctx.block_ids, ctx.block_offsets
        ]
        v_padded = self.kv_cache_manager.v_cache[
            self.layer_idx, ctx.block_ids, ctx.block_offsets
        ]

        # 把 q [sum_T, H_q, D] pack 成 [N, max_chunk_len, H_q, D]
        # 用预计算的 (pad_row, pad_col) 索引一次散列写入，避免 Python for 循环
        N = ctx.chunk_lens.shape[0]
        T_max = ctx.max_chunk_len
        H_q, D = q.shape[1], q.shape[2]
        q_padded = torch.zeros(
            (N, T_max, H_q, D), device=q.device, dtype=q.dtype,
        )
        q_padded[ctx.pad_row, ctx.pad_col] = q

        # 批量 attention
        attn_padded = batched_causal_attention_prefill(
            q_padded=q_padded,
            k_padded=k_padded,
            v_padded=v_padded,
            chunk_lens=ctx.chunk_lens,
            history_lens=ctx.history_lens,
            num_q_heads=self.num_q_heads,
        )  # [N, max_chunk_len, H_q, D]

        # Unpack 回 [sum_T, H_q, D]：advanced indexing 一次拿到
        attn_flat = attn_padded[ctx.pad_row, ctx.pad_col]

        # O 投影 + 融合残差 add + MLP norm（M3 优化）
        attn_out = attn_flat.reshape(-1, self.hidden_size)
        attn_out = linear(attn_out, self.weights.o_proj, self.weights.o_proj_bias)
        x, hidden_states = fused_add_rms_norm(
            attn_out, residual,
            self.weights.post_attention_layernorm, self.model_config.rms_norm_eps,
        )

        # ---- MLP 子层 ----
        hidden_states = hidden_states + self._mlp(x)
        return hidden_states

    def forward_prefill_one(self, hidden_states: torch.Tensor, meta: PrefillRequestMetadata, req: Request):
        """单个请求的 prefill 前向（处理一个 chunk）

        完整的单层前向流程（Pre-Norm 残差结构）：

            # Attention 子层
            residual = x
            x = RMSNorm(x, γ_attn, ε)                     # x = x / RMS(x) * γ_attn
            x = residual + Attention(x)                    # 残差连接

            # MLP 子层
            residual = x
            x = RMSNorm(x, γ_mlp, ε)                      # x = x / RMS(x) * γ_mlp
            x = residual + SwiGLU_MLP(x)                   # 残差连接

        Args:
            hidden_states: 本 chunk 的隐藏状态 [chunk_len, hidden_size]
            meta: prefill 元数据
            req: 请求对象

        Returns:
            本层输出 [chunk_len, hidden_size]
        """
        # Attention 子层
        residual = hidden_states
        x = rms_norm(hidden_states, self.weights.input_layernorm, self.model_config.rms_norm_eps)
        attn_out = self._attention_prefill(x, meta, req)

        # 融合残差 add + MLP norm（M3 优化）
        x, hidden_states = fused_add_rms_norm(
            attn_out, residual,
            self.weights.post_attention_layernorm, self.model_config.rms_norm_eps,
        )

        # MLP 子层
        hidden_states = hidden_states + self._mlp(x)
        return hidden_states

    def forward_decode_batch(
        self,
        hidden_states: torch.Tensor,
        metas: List[DecodeRequestMetadata],
        reqs: List[Request],
        gather_ctx: Optional["DecodeBatchGatherCtx"] = None,
    ) -> torch.Tensor:
        """N 个 decode 请求的批量前向（每个请求 1 个 query token）

        所有按 token 的算子（rms_norm/QKV/RoPE/KV 写入/O_proj/MLP/残差）都在
        [N, hidden] 上一次完成。Attention 走 gathered_paged_kv_decode_attention：
        - 一次性 gather（advanced indexing）把 paged 物理 block 物化成
          [N, max_ctx, kv_heads, D] 的 padded 稠密 KV
        - 在 padded 张量上做常规 batched matmul + mask（不是 vLLM 那种
          paged kernel；那是 stage 8 用 CUDA 写的事）
        - 用 GQA 分组而非物化 repeat_kv
        - 用 context_lens mask 屏蔽 padding 位

        gather_ctx 可由调用方预先构造（同一 batch 的 N 个请求 N 层共享），避免
        每层重复构造索引张量。
        """
        n = hidden_states.shape[0]

        # ---- Attention 子层 ----
        residual = hidden_states
        x = rms_norm(hidden_states, self.weights.input_layernorm, self.model_config.rms_norm_eps)

        # 批量 QKV：把 [N, hidden] 视作 seq_len=N 的输入
        q, k_new, v_new = self._project_qkv(x)

        # 批量 RoPE（每请求 query_position 不同）
        positions = (
            gather_ctx.positions if gather_ctx is not None
            else torch.tensor(
                [m.query_position for m in metas], device=x.device, dtype=torch.long
            )
        )
        q, k_new = apply_rope(q, k_new, positions, self.rope_cos, self.rope_sin)

        # 批量 KV 写入：N 个 token → 各自的 write_slot
        if gather_ctx is not None:
            # batch 分桶时仅前 valid_batch_size 行对应真实请求，padding 行不写 KV。
            valid_n = gather_ctx.valid_batch_size if gather_ctx.valid_batch_size > 0 else n
            self.kv_cache_manager.write_kv_for_tokens_indexed(
                layer_idx=self.layer_idx,
                block_ids=gather_ctx.write_block_ids[:valid_n],
                block_offsets=gather_ctx.write_block_offsets[:valid_n],
                k_values=k_new[:valid_n],
                v_values=v_new[:valid_n],
            )
        else:
            self.kv_cache_manager.write_kv_for_tokens_batch(
                layer_idx=self.layer_idx,
                slot_refs=[m.write_slot for m in metas],
                k_values=k_new,
                v_values=v_new,
            )

        # 批量 paged-attention：M4 优先走 decode_paged_attention（block-aware kernel/fallback）
        # 否则退回 gather + batched matmul
        if gather_ctx is not None and gather_ctx.block_table_tensor is not None:
            # M4 路径：直接传 block_table + context_lens 给 kernel/fallback
            attn_out = decode_paged_attention(
                q=q,
                k_cache=self.kv_cache_manager.k_cache[self.layer_idx],
                v_cache=self.kv_cache_manager.v_cache[self.layer_idx],
                block_table=gather_ctx.block_table_tensor,
                context_lens=gather_ctx.context_lens_tensor.to(torch.int32),
            )  # [N, num_q_heads, head_dim]
        elif gather_ctx is not None:
            # 原 gather 路径（fallback for M1/no-kernel）
            k_padded = self.kv_cache_manager.k_cache[
                self.layer_idx, gather_ctx.block_ids, gather_ctx.block_offsets
            ]
            v_padded = self.kv_cache_manager.v_cache[
                self.layer_idx, gather_ctx.block_ids, gather_ctx.block_offsets
            ]
            ctx_lens_t = gather_ctx.context_lens_tensor
            attn_out = gathered_paged_kv_decode_attention(
                q=q,
                k_padded=k_padded,
                v_padded=v_padded,
                context_lens=ctx_lens_t,
                num_q_heads=self.num_q_heads,
            )  # [N, num_q_heads, head_dim]
        else:
            k_padded, v_padded, ctx_lens_t = self.kv_cache_manager.gather_kv_decode_batch(
                layer_idx=self.layer_idx,
                reqs=reqs,
                context_lens=[m.context_len for m in metas],
            )
            attn_out = gathered_paged_kv_decode_attention(
                q=q,
                k_padded=k_padded,
                v_padded=v_padded,
                context_lens=ctx_lens_t,
                num_q_heads=self.num_q_heads,
            )  # [N, num_q_heads, head_dim]
        attn_out = attn_out.reshape(n, self.hidden_size)
        attn_out = linear(attn_out, self.weights.o_proj, self.weights.o_proj_bias)
        # 融合残差 add + MLP norm（M3 优化）
        x, hidden_states = fused_add_rms_norm(
            attn_out, residual,
            self.weights.post_attention_layernorm, self.model_config.rms_norm_eps,
        )

        # ---- MLP 子层 ----
        hidden_states = hidden_states + self._mlp(x)
        return hidden_states

    def forward_decode_one(self, hidden_state: torch.Tensor, meta: DecodeRequestMetadata, req: Request):
        """单个请求的 decode 前向（处理 1 个 token）

        结构与 forward_prefill_one 相同，只是调用 _attention_decode 而非 _attention_prefill。
        Decode 阶段每步只有 1 个 query token，KV Cache 中有全部历史上下文。

        Args:
            hidden_state: 当前 token 的隐藏状态 [1, hidden_size]
            meta: decode 元数据
            req: 请求对象

        Returns:
            本层输出 [1, hidden_size]
        """
        residual = hidden_state
        x = rms_norm(hidden_state, self.weights.input_layernorm, self.model_config.rms_norm_eps)
        attn_out = self._attention_decode(x, meta, req)

        # 融合残差 add + MLP norm（M3 优化）
        x, hidden_state = fused_add_rms_norm(
            attn_out, residual,
            self.weights.post_attention_layernorm, self.model_config.rms_norm_eps,
        )
        hidden_state = hidden_state + self._mlp(x)
        return hidden_state


# ---------------------------------------------------------------------------
# 模型运行器
# ---------------------------------------------------------------------------

class TransformerModelRunner:
    """通用 Transformer 模型运行器

    使用自研前向替代 HF model.forward()，
    KV 直接写入 Paged KV Cache。

    完整推理流程：
        hidden = Embed(token_ids)                     # Token Embedding
        for block in blocks:
            hidden = block.forward(hidden)             # 逐层前向
        logits = lm_head(RMSNorm(hidden))              # 最终 Norm + 投影到词表

    Prefill 取最后一个 token 的 logits（用于预测下一个 token），
    Decode 取唯一 token 的 logits。
    """

    def __init__(
        self,
        engine_config: EngineConfig,
        model_config: ModelConfig,
        weights: TransformerWeights,
        kv_cache_manager: KVCacheManager,
    ) -> None:
        """初始化模型运行器

        Args:
            engine_config: 引擎配置（device、dtype 等）
            model_config: 模型配置（层数、头数等）
            weights: 完整模型权重
            kv_cache_manager: Paged KV Cache 管理器
        """
        self._device = engine_config.device
        self._dtype = engine_config.dtype
        self.engine_config = engine_config
        self.model_config = model_config
        self.weights = weights
        self.kv_cache_manager = kv_cache_manager
        self._cuda_graph: Optional["CudaGraphBatch1Runner"] = None  # lazy init

        # 预计算 RoPE 的 cos/sin 缓存，避免每次前向重复计算
        # rope_cos/rope_sin: [max_seq_len, head_dim]
        self.rope_cos, self.rope_sin = build_rope_cache(
            max_seq_len=model_config.max_position_embeddings,
            head_dim=model_config.head_dim,
            theta=model_config.rope_theta,
            device=self._device,
            dtype=self._dtype,
        )

        # 为每一层创建 BlockRunner，每层有独立的权重和共享的 KV Cache 管理器
        self.blocks: List[TransformerBlockRunner] = []
        for i in range(model_config.num_hidden_layers):
            self.blocks.append(
                TransformerBlockRunner(
                    layer_idx=i,
                    model_config=model_config,
                    layer_weights=weights.layers[i],
                    kv_cache_manager=kv_cache_manager,
                    rope_cos=self.rope_cos,
                    rope_sin=self.rope_sin,
                )
            )

        # 可替换的执行入口：默认 eager，开启 compile 后会被替换为编译版本
        self._prefill_layers_fn: Callable[[torch.Tensor, PrefillBatchCtx], torch.Tensor] = (
            self._run_prefill_layers_eager
        )
        self._decode_layers_fn: Callable[
            [torch.Tensor, List[DecodeRequestMetadata], List[Request], DecodeBatchGatherCtx],
            torch.Tensor,
        ] = self._run_decode_layers_eager

        if engine_config.enable_torch_compile:
            self._maybe_enable_torch_compile()

        # ---- 预分配 metadata 索引 buffer（避免每 step 重复创建小 tensor） ----
        max_bs = engine_config.max_batch_size
        self._decode_input_ids_buf = torch.empty(max_bs, device=self._device, dtype=torch.long)
        self._decode_positions_buf = torch.empty(max_bs, device=self._device, dtype=torch.long)
        self._decode_ctx_lens_buf = torch.empty(max_bs, device=self._device, dtype=torch.long)
        self._decode_wbids_buf = torch.empty(max_bs, device=self._device, dtype=torch.long)
        self._decode_wboff_buf = torch.empty(max_bs, device=self._device, dtype=torch.long)
        self._max_batch_size = max_bs
        self._arange_cache: Dict[int, torch.Tensor] = {}

    def quantize_weights(self, bits: int = 4, group_size: int = 64) -> None:
        """将所有权重矩阵运行时量化为 INT4 group quantization。

        量化后 ``nn_ops.linear()` 自动检测 tuple 格式并走量化路径。
        Embedding、lm_head、layernorm 保持 fp16 不变。
        """
        for block in self.blocks:
            lw = block.weights
            lw.qkv_proj = quantize_weight_group(lw.qkv_proj, bits=bits, group_size=group_size)
            lw.o_proj = quantize_weight_group(lw.o_proj, bits=bits, group_size=group_size)
            lw.gate_up_proj = quantize_weight_group(lw.gate_up_proj, bits=bits, group_size=group_size)
            lw.down_proj = quantize_weight_group(lw.down_proj, bits=bits, group_size=group_size)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def _cached_arange(self, length: int) -> torch.Tensor:
        """缓存常用 torch.arange。"""
        if length not in self._arange_cache:
            self._arange_cache[length] = torch.arange(length, device=self._device, dtype=torch.long)
        return self._arange_cache[length]

    def _copy_to_buf(self, buf: torch.Tensor, src: list, n: int) -> torch.Tensor:
        """把 Python list 拷贝到预分配 buffer 并返回 [n] slice。"""
        if n > buf.shape[0]:
            buf = torch.empty(max(n, buf.shape[0] * 2), device=self._device, dtype=buf.dtype)
        if n > 0:
            buf[:n] = torch.tensor(src[:n], device=self._device, dtype=buf.dtype)
        return buf[:n]

    # 保持 _decode_*_buf 可变，_copy_to_buf 可能扩容
    _decode_input_ids_buf: torch.Tensor
    _decode_positions_buf: torch.Tensor
    _decode_ctx_lens_buf: torch.Tensor
    _decode_wbids_buf: torch.Tensor
    _decode_wboff_buf: torch.Tensor

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token Embedding 查表

        公式：hidden = Embedding[token_ids]

        使用 index_select 而非 embedding 层，因为权重是预提取的 Tensor。

        Args:
            input_ids: token ID 张量 [seq_len]

        Returns:
            对应的 embedding 向量 [seq_len, hidden_size]
        """
        return self.weights.embed_tokens.index_select(0, input_ids)

    def _project_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """最终 logits 投影

        公式：
            logits = lm_head(RMSNorm(hidden, γ_final, ε))

        先做最终 RMSNorm，再线性投影到词表大小。

        Args:
            hidden_states: 最后一层的输出隐藏状态

        Returns:
            logits: [seq_len, vocab_size]，每个位置对应词表中每个词的未归一化得分
        """
        x = rms_norm(hidden_states, self.weights.final_norm, self.model_config.rms_norm_eps)
        return linear(x, self.weights.lm_head)

    def _run_prefill_layers_eager(
        self,
        hidden_states: torch.Tensor,
        ctx: "PrefillBatchCtx",
    ) -> torch.Tensor:
        for block in self.blocks:
            hidden_states = block.forward_prefill_batch(hidden_states, ctx)
        return hidden_states

    def _run_decode_layers_eager(
        self,
        hidden_states: torch.Tensor,
        metas: List[DecodeRequestMetadata],
        requests: List[Request],
        gather_ctx: "DecodeBatchGatherCtx",
    ) -> torch.Tensor:
        for block in self.blocks:
            hidden_states = block.forward_decode_batch(hidden_states, metas, requests, gather_ctx)
        return hidden_states

    def _maybe_enable_torch_compile(self) -> None:
        """按配置启用 torch.compile；失败时自动回退到 eager。"""
        if not hasattr(torch, "compile"):
            print("[compile] torch.compile not available, fallback to eager")
            return

        # MPS 上当前图中会触发 float64 不支持问题，先显式回退 eager。
        if self._device.type == "mps":
            print("[compile] disabled on mps, fallback to eager")
            return

        try:
            self._prefill_layers_fn = torch.compile(
                self._run_prefill_layers_eager,
                mode=self.engine_config.torch_compile_mode,
                fullgraph=self.engine_config.torch_compile_fullgraph,
                dynamic=True,
            )
            self._decode_layers_fn = torch.compile(
                self._run_decode_layers_eager,
                mode=self.engine_config.torch_compile_mode,
                fullgraph=self.engine_config.torch_compile_fullgraph,
                dynamic=True,
            )
            print(
                f"[compile] enabled mode={self.engine_config.torch_compile_mode} "
                f"fullgraph={self.engine_config.torch_compile_fullgraph}"
            )
        except Exception as e:
            print(f"[compile] failed, fallback to eager: {e}")
            self._prefill_layers_fn = self._run_prefill_layers_eager
            self._decode_layers_fn = self._run_decode_layers_eager

    def _bucket_len(self, x: int) -> int:
        """按配置把长度向上取整到桶边界；0 表示不分桶。"""
        multiple = int(self.engine_config.context_bucket_multiple)
        if multiple <= 0:
            return x
        return ((x + multiple - 1) // multiple) * multiple

    def _bucket_batch(self, n: int) -> int:
        """按配置把 decode batch 向上取整到桶边界；0 表示不分桶。"""
        multiple = int(self.engine_config.decode_batch_bucket_multiple)
        if multiple <= 0:
            return n
        return ((n + multiple - 1) // multiple) * multiple

    def _build_prefill_batch_ctx(
        self,
        requests: List[Request],
        metas: List[PrefillRequestMetadata],
    ) -> "PrefillBatchCtx":
        """为一批 prefill 请求构造跨层共享的批量上下文。

        约定：调用前 metas 中的 chunk_token_ids 都已通过 ensure_slots_for_request
        分配好 write_slots（runner 路径已经做这件事），所以 cache 里历史 KV 完整、
        新 chunk 的 slot 也已经预留。
        """
        device = self._device
        chunk_lens_list = [len(m.chunk_token_ids) for m in metas]
        history_lens_list = [m.context_len_before_chunk for m in metas]
        # KV 总长 = history + chunk
        total_kv_lens = [h + t for h, t in zip(history_lens_list, chunk_lens_list)]

        # 拼接所有 token 的位置和 write_slots
        positions_flat: List[int] = []
        write_slots_flat: List[SlotRef] = []
        chunk_offsets: List[int] = [0]
        pad_row_list: List[int] = []
        pad_col_list: List[int] = []
        for i, m in enumerate(metas):
            positions_flat.extend(m.positions)
            write_slots_flat.extend(m.write_slots)
            t = len(m.chunk_token_ids)
            chunk_offsets.append(chunk_offsets[-1] + t)
            pad_row_list.extend([i] * t)
            pad_col_list.extend(range(t))

        positions = torch.tensor(positions_flat, device=device, dtype=torch.long)
        chunk_lens = torch.tensor(chunk_lens_list, device=device, dtype=torch.long)
        history_lens = torch.tensor(history_lens_list, device=device, dtype=torch.long)
        max_chunk_len = max(chunk_lens_list) if chunk_lens_list else 0
        max_kv_len = max(total_kv_lens) if total_kv_lens else 0
        max_kv_len_bucketed = self._bucket_len(max_kv_len)

        # 复用 KV gather 索引构造（按 total_kv_lens 当 context_len）
        # 为提升 compile 命中率，可把 max_ctx 向上 pad 到分桶边界。
        block_ids, block_offsets = self.kv_cache_manager.build_decode_batch_indices(
            requests, total_kv_lens, padded_max_ctx=max_kv_len_bucketed,
        )

        # 写入 slot 的 (block_ids, block_offsets) 一次性算出，跨层复用
        write_block_ids, write_block_offsets = self.kv_cache_manager.slot_refs_to_indices(
            write_slots_flat,
        )

        # 每请求 chunk 最后一个 token 在拼接序列中的索引
        last_idx_list = [chunk_offsets[i + 1] - 1 for i in range(len(metas))]
        last_token_indices = torch.tensor(last_idx_list, device=device, dtype=torch.long)

        # pad_row / pad_col：把 q [sum_T] 散列写入 q_padded [N, T_max] 用
        pad_row = torch.tensor(pad_row_list, device=device, dtype=torch.long)
        pad_col = torch.tensor(pad_col_list, device=device, dtype=torch.long)

        return PrefillBatchCtx(
            positions=positions,
            write_slots=write_slots_flat,
            write_block_ids=write_block_ids,
            write_block_offsets=write_block_offsets,
            chunk_lens=chunk_lens,
            history_lens=history_lens,
            chunk_offsets=chunk_offsets,
            max_chunk_len=max_chunk_len,
            max_kv_len=max_kv_len_bucketed,
            block_ids=block_ids,
            block_offsets=block_offsets,
            last_token_indices=last_token_indices,
            pad_row=pad_row,
            pad_col=pad_col,
        )

    def _forward_prefill_impl(
        self,
        requests: List[Request],
        metas: List[PrefillRequestMetadata],
    ) -> PrefillModelOutput:
        """Prefill 前向（批量）：N 个请求的 chunk 拼成一条长序列一次跑完

        - 所有 chunk token 拼接为 [sum_T, hidden]，一次 embed
        - 每层走 forward_prefill_batch（block-diagonal causal mask 处理跨请求隔离）
        - 最后只取每请求 chunk 末尾 token 的 logits（用于采样下一个 token）
        """
        # 过滤掉 chunk 为空的请求（理论上 scheduler 不会发空 chunk，但保险起见）
        nonempty = [(req, m) for req, m in zip(requests, metas) if m.chunk_token_ids]
        if not nonempty:
            return PrefillModelOutput(logits_by_request={})
        active_reqs = [x[0] for x in nonempty]
        active_metas = [x[1] for x in nonempty]

        # 拼接 token ids → 一次 embed
        flat_token_ids: List[int] = []
        for m in active_metas:
            flat_token_ids.extend(m.chunk_token_ids)
        input_ids = torch.tensor(flat_token_ids, device=self._device, dtype=torch.long)
        hidden_states = self._embed(input_ids)  # [sum_T, hidden]

        # 构造跨层共享上下文
        ctx = self._build_prefill_batch_ctx(active_reqs, active_metas)

        # 逐层批量前向（可切换 eager / compiled）
        hidden_states = self._prefill_layers_fn(hidden_states, ctx)

        # 只有本 chunk 结束后完成全部 prompt 的请求才需要首 token logits。
        # 先选末 hidden 再做 LM Head，避免构造 [sum_T, vocab] 的大临时张量。
        final_request_indices = [
            i
            for i, (req, meta) in enumerate(zip(active_reqs, active_metas))
            if meta.chunk_end == req.total_prompt_tokens()
        ]
        if not final_request_indices:
            return PrefillModelOutput(logits_by_request={})

        final_rows = ctx.last_token_indices.index_select(
            0,
            torch.tensor(final_request_indices, device=self._device, dtype=torch.long),
        )
        final_hidden = hidden_states.index_select(0, final_rows)
        final_logits = self._project_logits(final_hidden)

        return PrefillModelOutput(
            logits_by_request={
                active_reqs[request_idx].request_id: final_logits[output_idx]
                for output_idx, request_idx in enumerate(final_request_indices)
            }
        )

    def forward_fresh_prefill(
        self,
        requests: List[Request],
        metas: List[PrefillRequestMetadata],
    ) -> PrefillModelOutput:
        """Fresh prefill 前向（首次处理请求的 prompt chunk）

        与 incremental prefill 使用相同的实现，区别在于元数据不同：
        - fresh: context_len_before_chunk=0, 没有历史 KV Cache
        - incremental: context_len_before_chunk>0, 有之前 chunk 的 KV Cache

        Args:
            requests: 请求列表
            metas: prefill 元数据列表

        Returns:
            PrefillModelOutput
        """
        return self._forward_prefill_impl(requests, metas)

    def forward_incremental_prefill(
        self,
        requests: List[Request],
        metas: List[PrefillRequestMetadata],
    ) -> PrefillModelOutput:
        """Incremental prefill 前向（继续处理请求的剩余 prompt chunk）

        实现与 fresh prefill 相同，通过 meta.context_len_before_chunk 区分。

        Args:
            requests: 请求列表
            metas: prefill 元数据列表

        Returns:
            PrefillModelOutput
        """
        return self._forward_prefill_impl(requests, metas)

    def enable_cuda_graph(
        self, first_token: int, context_len: int, block_table: torch.Tensor
    ) -> None:
        """Create and capture CUDA graph runner for batch=1 greedy decode."""
        if self._device.type != "cuda":
            return
        from miniservellm.runtime.cuda_graph_runner import CudaGraphBatch1Runner
        self._cuda_graph = CudaGraphBatch1Runner(self, max_context=context_len + 512)
        self._cuda_graph.capture(first_token, context_len, block_table)

    def cuda_graph_step(
        self, token_id: int, context_len: int, block_table: torch.Tensor
    ) -> torch.Tensor:
        """Replay CUDA graph for one decode step. Returns [1, vocab] logits."""
        assert self._cuda_graph is not None
        return self._cuda_graph.step(token_id, context_len, block_table)

    @property
    def has_cuda_graph(self) -> bool:
        return self._cuda_graph is not None and self._cuda_graph.is_captured

    def forward_decode(
        self,
        requests: List[Request],
        metas: List[DecodeRequestMetadata],
    ) -> DecodeModelOutput:
        """Decode 前向（批量：N 个请求并行生成 1 个 token）

        将 N 个请求的 input_token_id 堆叠成 [N] 一次 embed 得到 [N, hidden]，
        再依次过每一层的 forward_decode_batch，最后批量 logits 投影。
        相比逐请求循环：
        - Embed/Norm/QKV/MLP/O_proj/lm_head 都从 N 次 kernel 启动降为 1 次
        - KV 写入从 N 次 advanced indexing 合为 1 次
        - 仅 attention 仍按请求循环（context_len 不同，先不做 paged-attention）

        Args:
            requests: 请求列表
            metas: decode 元数据列表（输入 token、query_position 等）

        Returns:
            DecodeModelOutput: request_id → logits[i] 的映射
        """
        if not requests:
            return DecodeModelOutput(logits_by_request={})

        valid_n = len(requests)
        bucket_n = self._bucket_batch(valid_n)

        # 批量 embed：[N] → [N, hidden_size]；batch 分桶时做尾部 pad
        input_id_list = [m.input_token_id for m in metas]
        if bucket_n > valid_n and valid_n > 0:
            input_id_list.extend([input_id_list[-1]] * (bucket_n - valid_n))
        input_ids = torch.tensor(input_id_list, device=self._device, dtype=torch.long)
        hidden_states = self._embed(input_ids)

        # 一次性构造跨层共享的批量上下文（block_ids / offsets / positions / context_lens）
        pos_list = [m.query_position for m in metas]
        if bucket_n > valid_n and valid_n > 0:
            pos_list.extend([pos_list[-1]] * (bucket_n - valid_n))
        positions = torch.tensor(pos_list, device=self._device, dtype=torch.long)

        ctx_lens_list = [m.context_len for m in metas]
        if bucket_n > valid_n and valid_n > 0:
            ctx_lens_list.extend([ctx_lens_list[-1]] * (bucket_n - valid_n))
        max_ctx = max(ctx_lens_list) if ctx_lens_list else 0
        max_ctx_bucketed = self._bucket_len(max_ctx)

        reqs_for_index = requests
        if bucket_n > valid_n and valid_n > 0:
            reqs_for_index = requests + [requests[-1]] * (bucket_n - valid_n)
        block_ids, block_offsets = self.kv_cache_manager.build_decode_batch_indices(
            reqs_for_index, ctx_lens_list, padded_max_ctx=max_ctx_bucketed,
        )
        context_lens_tensor = torch.tensor(ctx_lens_list, device=self._device, dtype=torch.long)

        # M4: 为 CUDA kernel 构造紧凑 block_table [N, max_blocks]
        from miniservellm.runtime.nn_ops import _HAS_CUSTOM_KERNELS
        block_table_tensor = None
        if _HAS_CUSTOM_KERNELS and max_ctx > 0:
            block_table_tensor = self.kv_cache_manager.build_decode_block_table(
                reqs_for_index, max_ctx_bucketed if max_ctx_bucketed > 0 else max_ctx,
            )

        write_slots = [m.write_slot for m in metas]
        if bucket_n > valid_n and valid_n > 0:
            write_slots.extend([metas[-1].write_slot] * (bucket_n - valid_n))
        write_block_ids, write_block_offsets = self.kv_cache_manager.slot_refs_to_indices(
            write_slots,
        )
        gather_ctx = DecodeBatchGatherCtx(
            positions=positions,
            write_slots=write_slots,
            write_block_ids=write_block_ids,
            write_block_offsets=write_block_offsets,
            block_ids=block_ids,
            block_offsets=block_offsets,
            context_lens_tensor=context_lens_tensor,
            valid_batch_size=valid_n,
            block_table_tensor=block_table_tensor,
        )

        # 逐层批量前向（可切换 eager / compiled，且复用 gather_ctx）
        hidden_states = self._decode_layers_fn(hidden_states, metas, requests, gather_ctx)

        # 仅对真实请求返回 logits；padding 行不参与输出
        logits = self._project_logits(hidden_states[:valid_n])

        return DecodeModelOutput(
            logits_by_request={req.request_id: logits[i] for i, req in enumerate(requests)}
        )

    def _forward_decode_tensor_step(
        self,
        request: Request,
        input_token: torch.Tensor,
        write_slot: SlotRef,
    ) -> torch.Tensor:
        """单请求 Tensor-only Decode step，返回设备端 greedy token。"""
        meta = DecodeRequestMetadata(
            request_id=request.request_id,
            input_token_id=0,  # input_token 保持在设备端；该字段不参与本方法的前向。
            query_position=write_slot.logical_pos,
            context_len=write_slot.logical_pos + 1,
            write_slot=write_slot,
        )
        context_len = meta.context_len
        max_ctx = self._bucket_len(context_len)
        block_ids, block_offsets = self.kv_cache_manager.build_decode_batch_indices(
            [request], [context_len], padded_max_ctx=max_ctx,
        )
        write_block_ids, write_block_offsets = self.kv_cache_manager.slot_refs_to_indices([write_slot])
        gather_ctx = DecodeBatchGatherCtx(
            positions=torch.tensor([meta.query_position], device=self._device, dtype=torch.long),
            write_slots=[write_slot],
            write_block_ids=write_block_ids,
            write_block_offsets=write_block_offsets,
            block_ids=block_ids,
            block_offsets=block_offsets,
            context_lens_tensor=torch.tensor([context_len], device=self._device, dtype=torch.long),
            valid_batch_size=1,
            block_table_tensor=None,
        )
        hidden_states = self._embed(input_token)
        hidden_states = self._decode_layers_fn(hidden_states, [meta], [request], gather_ctx)
        return torch.argmax(self._project_logits(hidden_states), dim=-1)

    @torch.inference_mode()
    def forward_decode_greedy_unrolled(
        self,
        request: Request,
        input_token_id: int,
        write_slots: List[SlotRef],
    ) -> torch.Tensor:
        """实验性 batch=1 greedy Decode：连续提交 K 步，末尾才同步 token。"""
        if self._device.type != "mps":
            raise RuntimeError("Unrolled greedy decode currently supports MPS only.")
        if not write_slots:
            return torch.empty(0, device=self._device, dtype=torch.long)

        input_token = torch.tensor([input_token_id], device=self._device, dtype=torch.long)
        output_tokens: List[torch.Tensor] = []
        for slot in write_slots:
            input_token = self._forward_decode_tensor_step(request, input_token, slot)
            output_tokens.append(input_token)
        return torch.cat(output_tokens, dim=0)
