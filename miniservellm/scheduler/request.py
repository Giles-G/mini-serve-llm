"""请求与采样参数定义

第四阶段在第三阶段基础上重构：
- 使用 prefill_offset 替代 prefill_cursor（语义更清晰）
- 新增 is_prefill_done() / prompt_len() / remaining_prompt_tokens() 等便捷方法
- status 改为字符串，与执行器直接设置保持一致
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SamplingParams:
    """采样参数

    Attributes:
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度，0.0 为贪心解码
        top_k: top-k 候选数，0 表示不限制
        top_p: nucleus sampling 累积概率阈值
        stop_token_ids: 遇到这些 token 时停止生成（如 eos_token_id）
    """
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    stop_token_ids: list[int] = field(default_factory=list)


@dataclass
class Request:
    """推理请求

    一个 Request 代表一次用户请求，包含输入、采样参数、运行状态和性能时间戳。

    Attributes:
        request_id: 请求唯一标识
        prompt: 完整 prompt 文本
        prompt_token_ids: prompt 编码后的 token id 列表
        sampling_params: 采样参数
        chunk_size: 每次 prefill chunk 的最大 token 数
        status: 请求状态，waiting / prefilling / decoding / finished
        generated_token_ids: 已生成的 token id 列表
        last_token_id: 上一个生成的 token id
        prefill_offset: 当前 prefill 已处理到的位置
        arrival_time: 请求加入系统的时间
        first_token_time: 首 token 生成时间
        finish_time: 请求完成时间
        past_key_values: HF 模型返回的 KV Cache
    """
    request_id: str
    prompt: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams

    # chunked prefill 参数
    chunk_size: int = 16

    # 生命周期状态
    status: str = "waiting"  # waiting / prefilling / decoding / finished

    # 推理状态
    generated_token_ids: list[int] = field(default_factory=list)
    last_token_id: int | None = None

    # prefill 进度
    prefill_offset: int = 0

    # 性能指标时间戳
    arrival_time: float = field(default_factory=time.time)
    first_token_time: float | None = None
    finish_time: float | None = None

    # KV Cache
    past_key_values: Any = None

    def prompt_len(self) -> int:
        """返回 prompt 的 token 数"""
        return len(self.prompt_token_ids)

    def num_generated_tokens(self) -> int:
        """返回已生成的 token 数"""
        return len(self.generated_token_ids)

    def remaining_prompt_tokens(self) -> int:
        """返回 prefill 阶段还未处理的 token 数"""
        return self.prompt_len() - self.prefill_offset

    def is_prefill_done(self) -> bool:
        """是否已完成全部 prompt 的 prefill"""
        return self.prefill_offset >= self.prompt_len()

    def reached_max_new_tokens(self) -> bool:
        """是否达到最大生成 token 数"""
        return self.num_generated_tokens() >= self.sampling_params.max_new_tokens

    def total_sequence_length(self) -> int:
        """返回当前序列总长度（已 prefill 部分 + 已生成 token）"""
        return self.prefill_offset + len(self.generated_token_ids)
