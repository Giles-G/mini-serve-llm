"""Batched Prefill Tensor Builder

第四阶段新增组件：将多个 fresh prefill 请求的 chunk
组装成 batched tensor，供 BatchedFreshPrefillExecutor 使用。

只处理 past_key_values is None 的请求（fresh prefill），
这些请求没有历史 cache，可以自然地 padding 后 batched forward。
"""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class BatchedPrefillTensors:
    """Batched Fresh Prefill 的输入 tensor 集合

    例如三个请求 chunk 长度分别是 4、2、3，则打包成：
    input_ids = [[11,12,13,14], [21,22,0,0], [31,32,33,0]]
    attention_mask = [[1,1,1,1], [1,1,0,0], [1,1,1,0]]

    Attributes:
        input_ids: [B, max_chunk_len]，padding 后的 chunk token ids
        attention_mask: [B, max_chunk_len]，1 表示有效 token，0 表示 padding
        valid_lengths: 每个请求的 chunk 实际长度（不含 padding）
        request_order: 本 batch 中请求的 request_id 顺序
        last_valid_indices: 每个请求最后一个有效 token 在行内的列索引
    """
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    valid_lengths: list[int]
    request_order: list[str]
    last_valid_indices: list[int]


class BatchedPrefillTensorBuilder:
    """Fresh Prefill 的 Batched Tensor 构建器

    将多个 fresh prefill 请求的 chunk padding 成统一的 [B, max_chunk_len] tensor。
    只处理没有 past_key_values 的请求，这些请求可以自然 batch。

    Attributes:
        device: 目标设备
        pad_token_id: 用于填充短 chunk 的 token id
    """

    def __init__(self, device: torch.device, pad_token_id: int):
        self.device = device
        self.pad_token_id = pad_token_id

    def build(
        self,
        requests,
        chunk_sizes: dict[str, int],
    ) -> BatchedPrefillTensors:
        """将多个 fresh prefill 请求的 chunk 组装成 batched tensor

        Args:
            requests: 本轮需要 fresh prefill 的请求列表
            chunk_sizes: request_id -> 本轮分配的 chunk token 数

        Returns:
            BatchedPrefillTensors
        """
        assert len(requests) > 0

        chunks: list[list[int]] = []
        request_order: list[str] = []
        valid_lengths: list[int] = []
        last_valid_indices: list[int] = []

        for req in requests:
            chunk_size = chunk_sizes[req.request_id]
            start = req.prefill_offset
            end = start + chunk_size
            chunk = req.prompt_token_ids[start:end]

            chunks.append(chunk)
            request_order.append(req.request_id)
            valid_lengths.append(len(chunk))
            last_valid_indices.append(len(chunk) - 1)

        max_len = max(valid_lengths)

        padded_input_ids = []
        padded_attention_mask = []

        for chunk in chunks:
            pad_len = max_len - len(chunk)
            padded = chunk + [self.pad_token_id] * pad_len
            mask = [1] * len(chunk) + [0] * pad_len
            padded_input_ids.append(padded)
            padded_attention_mask.append(mask)

        input_ids = torch.tensor(
            padded_input_ids,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.tensor(
            padded_attention_mask,
            dtype=torch.long,
            device=self.device,
        )

        return BatchedPrefillTensors(
            input_ids=input_ids,
            attention_mask=attention_mask,
            valid_lengths=valid_lengths,
            request_order=request_order,
            last_valid_indices=last_valid_indices,
        )
