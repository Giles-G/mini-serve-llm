"""请求与采样参数定义

第五阶段重构：
- 使用 RequestStatus/FinishReason 枚举替代字符串状态
- 新增 block_table / total_slots_reserved 用于 Paged KV Cache
- 新增 pending_prefill_sample_token_id 处理 prefill→decode 首 token 交接
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional


class RequestStatus(Enum):
    WAITING = auto()
    RUNNING_PREFILL = auto()
    RUNNING_DECODE = auto()
    FINISHED = auto()


class FinishReason(Enum):
    NONE = auto()
    EOS = auto()
    MAX_NEW_TOKENS = auto()
    ABORTED = auto()


@dataclass
class SamplingParams:
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.0


@dataclass
class Request:
    request_id: str
    prompt_token_ids: List[int]
    max_new_tokens: int
    sampling_params: SamplingParams

    arrival_step: int = 0
    status: RequestStatus = RequestStatus.WAITING
    finish_reason: FinishReason = FinishReason.NONE

    num_prompt_tokens_processed: int = 0
    generated_token_ids: List[int] = field(default_factory=list)

    block_table: List[int] = field(default_factory=list)
    total_slots_reserved: int = 0

    last_update_step: int = 0

    pending_prefill_sample_token_id: Optional[int] = None

    def total_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    def remaining_prompt_tokens(self) -> int:
        return self.total_prompt_tokens() - self.num_prompt_tokens_processed

    def total_generated_tokens(self) -> int:
        return len(self.generated_token_ids)

    def total_context_len(self) -> int:
        return self.total_prompt_tokens() + self.total_generated_tokens()

    def can_decode_more(self) -> bool:
        return self.total_generated_tokens() < self.max_new_tokens

    def append_generated_token(self, token_id: int) -> None:
        self.generated_token_ids.append(token_id)

    def mark_finished_eos(self) -> None:
        self.status = RequestStatus.FINISHED
        self.finish_reason = FinishReason.EOS

    def mark_finished_max_new_tokens(self) -> None:
        self.status = RequestStatus.FINISHED
        self.finish_reason = FinishReason.MAX_NEW_TOKENS

    def mark_aborted(self) -> None:
        self.status = RequestStatus.FINISHED
        self.finish_reason = FinishReason.ABORTED

    def all_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.generated_token_ids

    def last_token_id_for_decode_input(self) -> int:
        if self.generated_token_ids:
            return self.generated_token_ids[-1]
        return self.prompt_token_ids[-1]
