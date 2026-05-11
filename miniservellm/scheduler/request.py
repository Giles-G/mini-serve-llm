"""请求与采样参数定义

第三阶段在第二阶段基础上增加 chunked prefill 支持：
- SamplingParams 增加 stop_token_ids
- Request 增加 prefill_cursor / chunk_size 字段，支持分段 prefill
- 新增 remaining_prefill_tokens() / advance_prefill_cursor() / total_sequence_length() 方法
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


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
    第三阶段引入 chunked prefill 状态机：
    waiting -> prefilling（可多步） -> decoding -> finished。

    Attributes:
        request_id: 请求唯一标识
        prompt: 完整 prompt 文本
        prompt_token_ids: prompt 编码后的 token id 列表
        sampling_params: 采样参数
        chunk_size: 每次 prefill chunk 的最大 token 数
        status: 请求状态，waiting / prefilling / decoding / finished
        generated_token_ids: 已生成的 token id 列表
        last_token_id: 上一个生成的 token id
        prefill_cursor: 当前 prefill 已处理到的位置
        prefill_done: 是否已完成 prefill 阶段
        arrival_time: 请求加入系统的时间
        first_token_time: 首 token 生成时间，用于计算 TTFT
        finish_time: 请求完成时间，用于计算端到端延迟
        past_key_values: 兼容 HF runner 的 cache 存储
    """
    request_id: str
    prompt: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams

    # chunked prefill 参数
    chunk_size: int = 16

    # 生命周期状态字段
    status: str = "waiting"  # waiting / prefilling / decoding / finished

    # 推理状态字段
    generated_token_ids: list[int] = field(default_factory=list)
    last_token_id: int | None = None

    # prefill 进度追踪
    prefill_cursor: int = 0       # 当前 prefill 已处理到的 token 位置
    prefill_done: bool = False    # 是否已完成全部 prompt 的 prefill

    # 性能指标时间戳
    arrival_time: float = field(default_factory=time.time)
    first_token_time: float | None = None
    finish_time: float | None = None

    # 兼容 HF runner 的 cache 存储
    past_key_values: object | None = None

    def remaining_prefill_tokens(self) -> int:
        """返回 prefill 阶段还未处理的 token 数"""
        return max(0, len(self.prompt_token_ids) - self.prefill_cursor)

    def advance_prefill_cursor(self, n: int) -> None:
        """推进 prefill 游标 n 个 token

        Args:
            n: 推进的 token 数，必须非负

        当游标到达 prompt 末尾时，自动标记 prefill_done。
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        self.prefill_cursor = min(len(self.prompt_token_ids), self.prefill_cursor + n)
        self.prefill_done = self.prefill_cursor >= len(self.prompt_token_ids)

    def total_sequence_length(self) -> int:
        """返回当前序列总长度（prompt 已处理部分 + 已生成 token）"""
        return self.prefill_cursor + len(self.generated_token_ids)

    def mark_prefilling(self) -> None:
        """标记请求进入 prefilling 状态"""
        self.status = "prefilling"

    def mark_decoding(self) -> None:
        """标记请求进入 decoding 状态"""
        self.status = "decoding"

    def mark_finished(self) -> None:
        """标记请求完成"""
        self.status = "finished"
        if self.finish_time is None:
            self.finish_time = time.time()
