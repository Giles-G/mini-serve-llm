"""全局配置模块

第五阶段重构：
- 新增 EngineConfig：包含 Paged KV Cache 和调度相关参数
- 新增 ModelConfig：模型结构参数（从 HF config 提取）
- 保留 MODEL_NAME 常量
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

# Hugging Face 模型标识符，切换模型时只改这一处
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def detect_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_dtype_for_device(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


@dataclass
class EngineConfig:
    """推理引擎配置

    Attributes:
        device: 推理设备
        dtype: 权重精度
        block_size: Paged KV Cache 的 block 大小
        num_gpu_blocks: GPU 上可用的 block 数量
        max_batch_size: 最大 batch 大小
        max_tokens_per_step: 每步最大 token 数
        max_prefill_tokens_per_step: 每步 prefill 最大 token 数
        max_decode_requests_per_step: 每步最大 decode 请求数
        prefill_chunk_size: 单个请求每步 prefill chunk 大小
        default_temperature: 默认采样温度
        default_top_k: 默认 top-k
        default_top_p: 默认 top-p
        kv_decode_block_reserve: 调度时为 decode 预留的最少空闲 block 数
        kv_reserve_relax_after_no_progress_steps: 连续无进展多少步后放宽一次水位
        enable_torch_compile: 是否启用 torch.compile 图优化
        torch_compile_mode: torch.compile mode（default/reduce-overhead/max-autotune）
        torch_compile_fullgraph: 是否要求 fullgraph 编译
        context_bucket_multiple: context length 分桶粒度（0 表示关闭）
        decode_batch_bucket_multiple: decode batch 分桶粒度（0 表示关闭）
        eos_token_id: EOS token id
    """

    device: torch.device
    dtype: torch.dtype

    block_size: int = 16
    num_gpu_blocks: int = 1024

    max_batch_size: int = 16
    max_tokens_per_step: int = 128
    max_prefill_tokens_per_step: int = 96
    max_decode_requests_per_step: int = 16
    prefill_chunk_size: int = 32

    default_temperature: float = 0.8
    default_top_k: int = 50
    default_top_p: float = 0.95
    kv_decode_block_reserve: int = 2
    kv_reserve_relax_after_no_progress_steps: int = 8
    enable_torch_compile: bool = False
    torch_compile_mode: str = "default"
    torch_compile_fullgraph: bool = False
    context_bucket_multiple: int = 0
    decode_batch_bucket_multiple: int = 0
    eos_token_id: Optional[int] = None

    @staticmethod
    def create(
        device: str = "auto",
        dtype: str = "auto",
        block_size: int = 16,
        num_gpu_blocks: int = 1024,
        max_batch_size: int = 16,
        max_tokens_per_step: int = 128,
        max_prefill_tokens_per_step: int = 96,
        max_decode_requests_per_step: int = 16,
        prefill_chunk_size: int = 32,
        default_temperature: float = 0.8,
        default_top_k: int = 50,
        default_top_p: float = 0.95,
        kv_decode_block_reserve: int = 2,
        kv_reserve_relax_after_no_progress_steps: int = 8,
        enable_torch_compile: bool = False,
        torch_compile_mode: str = "default",
        torch_compile_fullgraph: bool = False,
        context_bucket_multiple: Optional[int] = None,
        decode_batch_bucket_multiple: int = 0,
        eos_token_id: Optional[int] = None,
    ) -> "EngineConfig":
        if device == "auto":
            dev = detect_best_device()
        else:
            dev = torch.device(device)

        if dtype == "auto":
            dt = default_dtype_for_device(dev)
        elif dtype == "fp16":
            dt = torch.float16
        elif dtype == "bf16":
            dt = torch.bfloat16
        elif dtype == "fp32":
            dt = torch.float32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        # MPS 对逐 token 变化的动态 context shape 很敏感；32 桶在 M4 实测中
        # 显著优于关闭分桶。调用方显式传 0 时仍可关闭用于 A/B。
        context_bucket = 32 if context_bucket_multiple is None and dev.type == "mps" else int(context_bucket_multiple or 0)

        return EngineConfig(
            device=dev,
            dtype=dt,
            block_size=block_size,
            num_gpu_blocks=num_gpu_blocks,
            max_batch_size=max_batch_size,
            max_tokens_per_step=max_tokens_per_step,
            max_prefill_tokens_per_step=max_prefill_tokens_per_step,
            max_decode_requests_per_step=max_decode_requests_per_step,
            prefill_chunk_size=prefill_chunk_size,
            default_temperature=default_temperature,
            default_top_k=default_top_k,
            default_top_p=default_top_p,
            kv_decode_block_reserve=kv_decode_block_reserve,
            kv_reserve_relax_after_no_progress_steps=kv_reserve_relax_after_no_progress_steps,
            enable_torch_compile=enable_torch_compile,
            torch_compile_mode=torch_compile_mode,
            torch_compile_fullgraph=torch_compile_fullgraph,
            context_bucket_multiple=context_bucket,
            decode_batch_bucket_multiple=decode_batch_bucket_multiple,
            eos_token_id=eos_token_id,
        )


@dataclass
class ModelConfig:
    """模型结构参数

    从 HF config 中提取的模型结构信息，用于构建 Paged KV Cache 和模型前向。

    Attributes:
        model_type: 模型类型（如 "qwen2"）
        vocab_size: 词表大小
        hidden_size: 隐藏层维度
        intermediate_size: FFN 中间层维度
        num_hidden_layers: Transformer 层数
        num_attention_heads: 注意力头数
        num_key_value_heads: KV 头数（GQA）
        head_dim: 每个头的维度
        max_position_embeddings: 最大位置编码长度
        rope_theta: RoPE 的 theta 参数
        rms_norm_eps: RMSNorm 的 epsilon
    """

    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
