"""HuggingFace 模型运行器

封装模型的前向推理调用，分为 prefill 和 decode 两种模式，
均使用 KV Cache（use_cache=True）以支持自回归生成。
"""

import torch
from miniservellm.types import ModelForwardOutput


class HFModelRunner:
    """HuggingFace 模型运行器

    封装模型 forward 调用，处理设备迁移和输出打包。
    prefill 和 decode 逻辑上分开，当前实现相同（后续可独立优化）。

    Attributes:
        model: HuggingFace 模型实例
        device: 推理设备
    """

    def __init__(self, model, device: str):
        self.model = model
        self.device = device

    @torch.no_grad()
    def forward_prefill(self, input_ids, attention_mask=None, past_key_values=None):
        """Prefill 阶段前向推理

        将完整 prompt 一次性送入模型，产出第一个 token 的 logits 和 KV Cache。

        Args:
            input_ids: 输入 token ids，形状 [1, seq_len]
            attention_mask: 注意力掩码
            past_key_values: 此阶段为 None（首次推理无缓存）

        Returns:
            ModelForwardOutput: 包含 logits 和 past_key_values
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,   # 启用 KV Cache
        )

        return ModelForwardOutput(
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
        )

    @torch.no_grad()
    def forward_decode(self, input_ids, attention_mask=None, past_key_values=None):
        """Decode 阶段前向推理

        每次只输入上一个生成的 token，配合 KV Cache 逐步解码。
        attention_mask 长度必须覆盖整个序列（prompt + 已生成 token）。

        Args:
            input_ids: 上一个生成的 token id，形状 [1, 1]
            attention_mask: 注意力掩码，长度 = prompt_len + generated_len
            past_key_values: 之前步骤积累的 KV Cache

        Returns:
            ModelForwardOutput: 包含 logits 和更新后的 past_key_values
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

        return ModelForwardOutput(
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
        )
