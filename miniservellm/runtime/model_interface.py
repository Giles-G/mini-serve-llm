"""模型接口定义

第五阶段新增：定义 ModelRunner 的协议接口和输出数据结构。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol

import torch

from miniservellm.scheduler.request import Request
from miniservellm.runtime.metadata import PrefillRequestMetadata, DecodeRequestMetadata


@dataclass
class PrefillModelOutput:
    logits_by_request: Dict[str, torch.Tensor]


@dataclass
class DecodeModelOutput:
    logits_by_request: Dict[str, torch.Tensor]


class ModelRunner(Protocol):
    @property
    def device(self) -> torch.device: ...

    @property
    def dtype(self) -> torch.dtype: ...

    def forward_fresh_prefill(
        self,
        requests: List[Request],
        metas: List[PrefillRequestMetadata],
    ) -> PrefillModelOutput: ...

    def forward_incremental_prefill(
        self,
        requests: List[Request],
        metas: List[PrefillRequestMetadata],
    ) -> PrefillModelOutput: ...

    def forward_decode(
        self,
        requests: List[Request],
        metas: List[DecodeRequestMetadata],
    ) -> DecodeModelOutput: ...
