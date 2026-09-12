"""Load Qwen2 HF safetensors directly into MLX arrays.

把 HuggingFace 格式的 Qwen2 safetensors 权重加载为 MLX 数组。

关键转换（与 qwen2.py 的前向实现一一对应）：
- QKV 融合：q/k/v 三个独立投影按输出维 cat 成 ``qkv_proj``，切分顺序 q→k→v；
- Gate/Up 融合：gate_proj/up_proj cat 成 ``gate_up_proj``，前向时对半切开；
- LM head 与 embedding 共享权重（tied embeddings），不单独加载；
- 所有权重统一转为 FP16 存储（源模型为 BF16）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import mlx.core as mx
import torch
from safetensors import safe_open

from miniservellm.config import ModelConfig
from miniservellm.mlx.qwen2 import MLXLayerWeights, MLXWeights


def _to_mx(tensor: torch.Tensor) -> mx.array:
    """CPU torch 张量 → MLX FP16 数组（统一经 float32 中转，避免直接 BF16→FP16 的精度坑）。"""
    return mx.array(tensor.float().numpy()).astype(mx.float16)


def load_qwen2_weights(model_path: str | Path, config: ModelConfig) -> MLXWeights:
    """从本地 HF safetensors 文件加载 Qwen2 权重。

    源模型是 BF16；MLX 后端刻意存成 FP16，与既有 MPS 推理基准保持同一精度口径。

    权重布局转换在加载期一次完成：
    - ``q/k/v_proj`` → ``qkv_proj``（输出维拼接，顺序 q→k→v，决定前向切分点）；
    - ``gate/up_proj`` → ``gate_up_proj``；
    - ``lm_head`` 不从文件读取，直接复用 embedding（tied embeddings）。
    """
    model_path = str(model_path)
    # 一次性把全部张量读入内存 dict（按名字索引，逐层取用）
    tensors: Dict[str, torch.Tensor] = {}
    with safe_open(model_path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensors[name] = handle.get_tensor(name)

    def get(name: str) -> torch.Tensor:
        # 缺张量时给出可读的报错名，而不是裸 KeyError
        try:
            return tensors[name]
        except KeyError as exc:
            raise KeyError(f"Missing Qwen2 tensor: {name}") from exc

    layers = []
    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}"
        # 注意力：Qwen2 原始格式是 q/k/v 三个独立投影（含 QKV 偏置）
        q = get(f"{prefix}.self_attn.q_proj.weight")
        k = get(f"{prefix}.self_attn.k_proj.weight")
        v = get(f"{prefix}.self_attn.v_proj.weight")
        q_bias = get(f"{prefix}.self_attn.q_proj.bias")
        k_bias = get(f"{prefix}.self_attn.k_proj.bias")
        v_bias = get(f"{prefix}.self_attn.v_proj.bias")
        # FFN：SwiGLU 的 gate/up 两个独立投影
        gate = get(f"{prefix}.mlp.gate_proj.weight")
        up = get(f"{prefix}.mlp.up_proj.weight")
        layers.append(
            MLXLayerWeights(
                # 沿输出维(dim=0)拼接：cat 后形状 [q_dim + 2*kv_dim, hidden]，
                # 一次 matmul 等价于三个独立投影，qwen2.py 按同样顺序切开
                qkv_proj=_to_mx(torch.cat([q, k, v], dim=0)),
                qkv_bias=_to_mx(torch.cat([q_bias, k_bias, v_bias], dim=0)),
                o_proj=_to_mx(get(f"{prefix}.self_attn.o_proj.weight")),
                # 同理融合为 [2*inter, hidden]，前向时 split 成对等的 gate/up
                gate_up_proj=_to_mx(torch.cat([gate, up], dim=0)),
                down_proj=_to_mx(get(f"{prefix}.mlp.down_proj.weight")),
                input_layernorm=_to_mx(get(f"{prefix}.input_layernorm.weight")),
                post_attention_layernorm=_to_mx(get(f"{prefix}.post_attention_layernorm.weight")),
            )
        )

    embed = _to_mx(get("model.embed_tokens.weight"))
    return MLXWeights(
        embed_tokens=embed,
        layers=layers,
        final_norm=_to_mx(get("model.norm.weight")),
        # tied embeddings：Qwen2 小模型不存独立 lm_head，直接与 embedding 共享；
        # _linear 里 weight.T 后正好得到 [hidden, vocab] 的输出投影
        lm_head=embed,
    )
