"""请求与采样参数定义

定义推理请求的数据结构，包含输入 prompt、采样参数和推理状态。
"""

from dataclasses import dataclass, field
import time


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
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    stop_token_ids: list[int] = field(default_factory=list)


@dataclass
class Request:
    """推理请求

    包含请求的输入、采样参数、以及推理过程中动态更新的状态。

    Attributes:
        request_id: 请求唯一标识
        prompt: 原始 prompt 文本
        prompt_token_ids: prompt 编码后的 token id 列表
        generated_token_ids: 已生成的 token id 列表（动态更新）
        sampling_params: 采样参数
        past_key_values: KV Cache，推理过程中动态更新
        last_token_id: 上一个生成的 token id
        arrival_time: 请求到达时间
        first_token_time: 首 token 产出时间（用于计算 TTFT）
        finish_time: 请求完成时间（用于计算端到端延迟）
    """
    request_id: str
    prompt: str
    prompt_token_ids: list[int]

    generated_token_ids: list[int] = field(default_factory=list)
    sampling_params: SamplingParams = field(default_factory=SamplingParams)

    # 推理过程中动态更新的状态
    past_key_values: object | None = None
    last_token_id: int | None = None

    # 性能指标时间戳
    arrival_time: float = field(default_factory=time.time)
    first_token_time: float | None = None
    finish_time: float | None = None
