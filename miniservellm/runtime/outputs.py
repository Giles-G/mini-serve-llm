"""Step 级事件输出

第三阶段新增 StepResult，用于描述每次 engine.step() 中单个请求的结果。
通过 event_type 区分不同阶段事件，便于调试和后续 streaming UI。

同时定义各执行器的结构化返回类型，替代 dict / bare tuple。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChunkedPrefillResult:
    """Chunked Prefill 单步执行结果

    Attributes:
        chunk_processed_tokens: 本轮处理的 token 数
        produced_token: 产出的首 token id（prefill 未完成时为 None）
        prefill_done: 是否完成全部 prefill
        finished: 请求是否已完成（首 token 就满足终止条件时为 True）
    """
    chunk_processed_tokens: int
    produced_token: int | None
    prefill_done: bool
    finished: bool


@dataclass
class DecodeStepResult:
    """Decode 单步执行结果

    Attributes:
        next_token_id: 本步生成的 token id
        finished: 请求是否已完成
    """
    next_token_id: int
    finished: bool


@dataclass
class PrefillResult:
    """Prefill 执行结果

    Attributes:
        next_token_id: 首个生成的 token id
    """
    next_token_id: int


@dataclass
class MetricsSummary:
    """请求性能指标摘要

    Attributes:
        request_id: 请求唯一标识
        status: 请求状态
        prompt_tokens: prompt token 数
        output_tokens: 生成 token 数
        prefill_cursor: prefill 游标位置
        prefill_done: 是否完成 prefill
        chunk_size: chunk 大小
        ttft: 首 token 延迟（秒）
        e2e_latency: 端到端延迟（秒）
    """
    request_id: str
    status: str
    prompt_tokens: int
    output_tokens: int
    prefill_cursor: int
    prefill_done: bool
    chunk_size: int
    ttft: float | None
    e2e_latency: float | None


@dataclass
class StepResult:
    """单步单个请求的执行结果

    Attributes:
        request_id: 请求唯一标识
        next_token_id: 本步生成的 token id，未产出 token 时为 None
        finished: 请求是否已完成
        text_delta: 本步生成的文本增量
        event_type: 事件类型，区分不同阶段：
            - PREFILL_PROGRESS: chunked prefill 进行中，未产出 token
            - PREFILL_TO_DECODE: prefill 完成，产出首 token，进入 decode
            - DECODE_TOKEN: decode 阶段产出一个 token
            - FINISHED: 请求生成完毕
        metadata: 附加元数据，用于调试和监控
    """
    request_id: str
    next_token_id: int | None
    finished: bool
    text_delta: str
    event_type: str = "UNKNOWN"
    metadata: dict[str, Any] = field(default_factory=dict)
