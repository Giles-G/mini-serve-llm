"""运行时输出类型定义

定义推理过程中每一步的输出结构。
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StepResult:
    """单步解码结果

    Attributes:
        request_id: 请求唯一标识
        next_token_id: 本步采样的 token id
        finished: 是否满足终止条件（达到 max_tokens 或遇到 stop token）
        text_delta: 本步生成的文本片段（流式输出时使用）
    """
    request_id: str
    next_token_id: int
    finished: bool
    text_delta: Optional[str] = None
