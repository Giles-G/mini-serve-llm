"""通用类型定义模块

定义模型 forward 输出等跨模块共享的数据结构。
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ModelForwardOutput:
    """模型单次 forward 的输出

    这是连接模型层和推理引擎的核心数据结构。
    每次 forward 调用都会返回 logits 和 KV Cache，
    它们分别服务于两个不同的阶段：

    - logits → 送给 Sampler 采样下一个 token
    - past_key_values → 保存到 Request 中，供后续 Decode 步骤复用

    在 Prefill 阶段，past_key_values 为 None（首次推理无缓存），
    模型会为所有 prompt token 计算并返回完整的 KV Cache。
    在 Decode 阶段，每次只输入 1 个新 token，模型利用已有 KV Cache
    只计算新 token 的注意力，避免重复计算，这就是 KV Cache 的核心价值。

    Attributes:
        logits: 模型输出的未归一化概率分布，形状 [batch, seq_len, vocab_size]。
                只取最后一个位置的 logits（logits[:, -1, :]）用于采样。
        past_key_values: KV Cache（Key-Value Cache），是一个多层嵌套的 tuple，
                结构为 ((key_layer0, value_layer0), (key_layer1, value_layer1), ...)
                每层的 key/value 形状为 [batch, num_heads, seq_len, head_dim]。
                随着解码推进，seq_len 会逐步增长。
    """
    logits: Any
    past_key_values: Optional[Any]
