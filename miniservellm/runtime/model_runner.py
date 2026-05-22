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

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from miniservellm.config import EngineConfig, ModelConfig
from miniservellm.cache.kv_cache import KVCacheManager
from miniservellm.runtime.metadata import PrefillRequestMetadata, DecodeRequestMetadata
from miniservellm.runtime.model_interface import PrefillModelOutput, DecodeModelOutput
from miniservellm.runtime.nn_ops import (
    rms_norm,
    silu_and_mul,
    build_rope_cache,
    apply_rope,
    repeat_kv,
    causal_attention_prefill,
    causal_attention_single_query,
    linear,
)
from miniservellm.scheduler.request import Request


# ---------------------------------------------------------------------------
# 通用权重数据结构
# ---------------------------------------------------------------------------

@dataclass
class TransformerLayerWeights:
    """单层 Transformer Block 权重

    适用于 Qwen2/LLaMA/Mistral/Gemma 等相同架构的模型。
    bias 字段为 Optional，不同模型可能没有。

    Attention 部分（GQA）：
        q_proj: Query 投影权重  [num_q_heads * head_dim, hidden_size]
        k_proj: Key 投影权重    [num_kv_heads * head_dim, hidden_size]
        v_proj: Value 投影权重  [num_kv_heads * head_dim, hidden_size]
        o_proj: Output 投影权重 [hidden_size, num_q_heads * head_dim]

    MLP 部分（SwiGLU）：
        gate_proj: Gate 投影权重  [intermediate_size, hidden_size]
        up_proj:   Up 投影权重    [intermediate_size, hidden_size]
        down_proj: Down 投影权重  [hidden_size, intermediate_size]

    Norm 部分：
        input_layernorm: Attention 前的 RMSNorm 权重 [hidden_size]
        post_attention_layernorm: MLP 前的 RMSNorm 权重 [hidden_size]
    """
    q_proj: torch.Tensor
    k_proj: torch.Tensor
    v_proj: torch.Tensor
    o_proj: torch.Tensor
    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor
    input_layernorm: torch.Tensor
    post_attention_layernorm: torch.Tensor
    q_proj_bias: Optional[torch.Tensor] = None
    k_proj_bias: Optional[torch.Tensor] = None
    v_proj_bias: Optional[torch.Tensor] = None
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
        """QKV 线性投影

        将输入 x 分别投影为 Query、Key、Value，并 reshape 为多头格式。

        公式：
            Q = x @ W_q.T + b_q    →  reshape to [seq_len, num_q_heads, head_dim]
            K = x @ W_k.T + b_k    →  reshape to [seq_len, num_kv_heads, head_dim]
            V = x @ W_v.T + b_v    →  reshape to [seq_len, num_kv_heads, head_dim]

        GQA 模型中 num_q_heads > num_kv_heads，所以 K/V 的 head 数少于 Q。
        例如 Qwen2.5-0.5B: num_q_heads=14, num_kv_heads=2

        Args:
            x: 输入隐藏状态 [seq_len, hidden_size]

        Returns:
            (q, k, v) 三元组，分别 reshape 为多头格式
        """
        q = linear(x, self.weights.q_proj, self.weights.q_proj_bias)
        k = linear(x, self.weights.k_proj, self.weights.k_proj_bias)
        v = linear(x, self.weights.v_proj, self.weights.v_proj_bias)
        # 将 [seq_len, num_heads * head_dim] reshape 为 [seq_len, num_heads, head_dim]
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
        gate = linear(x, self.weights.gate_proj)
        up = linear(x, self.weights.up_proj)
        act = silu_and_mul(gate, up)
        return linear(act, self.weights.down_proj)

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
        hidden_states = residual + self._attention_prefill(x, meta, req)

        # MLP 子层
        residual = hidden_states
        x = rms_norm(hidden_states, self.weights.post_attention_layernorm, self.model_config.rms_norm_eps)
        hidden_states = residual + self._mlp(x)
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
        hidden_state = residual + self._attention_decode(x, meta, req)

        residual = hidden_state
        x = rms_norm(hidden_state, self.weights.post_attention_layernorm, self.model_config.rms_norm_eps)
        hidden_state = residual + self._mlp(x)
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

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

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

    def _forward_prefill_impl(
        self,
        requests: List[Request],
        metas: List[PrefillRequestMetadata],
    ) -> PrefillModelOutput:
        """Prefill 前向的统一实现（fresh 和 incremental 共用）

        逐请求处理（非 batched），每个请求独立计算前向。
        取每个请求最后一个 token 的 logits 用于采样下一个 token。

        Args:
            requests: 请求列表
            metas: 对应的 prefill 元数据列表

        Returns:
            PrefillModelOutput: request_id → logits[-1] 的映射
                logits[-1] 是 chunk 最后一个 token 的预测，用于采样下一个生成 token
        """
        logits_by_request: Dict[str, torch.Tensor] = {}
        for req, meta in zip(requests, metas):
            if not meta.chunk_token_ids:
                continue
            # Token Embedding
            input_ids = torch.tensor(meta.chunk_token_ids, device=self._device, dtype=torch.long)
            hidden_states = self._embed(input_ids)
            # 逐层前向：每一层都会将 K/V 写入 Paged KV Cache
            for block in self.blocks:
                hidden_states = block.forward_prefill_one(hidden_states, meta, req)
            # 最终投影到词表
            logits = self._project_logits(hidden_states)
            # 只取最后一个 token 的 logits：因为自回归模型中，位置 t 的输出预测 t+1 的 token
            logits_by_request[req.request_id] = logits[-1]
        return PrefillModelOutput(logits_by_request=logits_by_request)

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

    def forward_decode(
        self,
        requests: List[Request],
        metas: List[DecodeRequestMetadata],
    ) -> DecodeModelOutput:
        """Decode 前向（每个请求生成 1 个 token）

        逐请求处理。每个请求输入上一步采样的 token ID，
        经过所有层后投影为 logits，用于采样下一个 token。

        Args:
            requests: 请求列表
            metas: decode 元数据列表（包含 input_token_id、query_position 等）

        Returns:
            DecodeModelOutput: request_id → logits[0] 的映射
        """
        logits_by_request: Dict[str, torch.Tensor] = {}
        for req, meta in zip(requests, metas):
            # 输入为单个 token（上一步采样的结果）
            input_ids = torch.tensor([meta.input_token_id], device=self._device, dtype=torch.long)
            hidden_state = self._embed(input_ids)
            # 逐层前向：每层将新 K/V 写入 Paged KV Cache，并读取全部历史 KV
            for block in self.blocks:
                hidden_state = block.forward_decode_one(hidden_state, meta, req)
            # 最终投影到词表
            logits = self._project_logits(hidden_state)
            # decode 只有 1 个 token，取 logits[0]
            logits_by_request[req.request_id] = logits[0]
        return DecodeModelOutput(logits_by_request=logits_by_request)
