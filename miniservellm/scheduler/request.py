"""请求与采样参数定义

第二阶段在第一阶段 Request 的基础上增加请求状态字段，
用于支持 RequestQueue、Scheduler 和 continuous batching 调度。
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

    一个 Request 代表一次用户请求，包含输入、采样参数、运行状态和性能时间戳。
    第二阶段引入状态机：WAITING -> PREFILLING -> DECODING -> FINISHED。

    Attributes:
        request_id: 请求唯一标识
        prompt: 完整 prompt 文本
        prompt_token_ids: prompt 编码后的 token id 列表
        generated_token_ids: 已生成的 token id 列表
        sampling_params: 采样参数
        status: 请求状态，WAITING / PREFILLING / DECODING / FINISHED
        prefill_done: 是否已完成 prefill 阶段
        finished: 是否已完成整个生成流程
        past_key_values: HF 模型返回的 KV Cache（兼容第一阶段逻辑）
        last_token_id: 上一个生成的 token id
        arrival_time: 请求加入系统的时间
        first_token_time: 首 token 生成时间，用于计算 TTFT
        finish_time: 请求完成时间，用于计算端到端延迟
    """
    request_id: str
    prompt: str
    prompt_token_ids: list[int]

    generated_token_ids: list[int] = field(default_factory=list)
    sampling_params: SamplingParams = field(default_factory=SamplingParams)

    # 生命周期状态字段
    status: str = "WAITING"
    prefill_done: bool = False
    finished: bool = False

    # 推理状态字段
    past_key_values: object | None = None
    last_token_id: int | None = None

    # 性能指标时间戳
    arrival_time: float = field(default_factory=time.time)
    first_token_time: float | None = None
    finish_time: float | None = None

    def total_output_tokens(self) -> int:
        """返回当前已生成 token 数"""
        return len(self.generated_token_ids)
