"""模型适配器接口

第五阶段新增：定义模型适配器的协议接口，
用于将 HF 模型转换为自研前向所需的权重格式。
"""

from __future__ import annotations

from typing import Any, Optional, Protocol

import torch

from miniservellm.config import ModelConfig


class AdapterWeights(Protocol):
    pass


class ModelAdapter(Protocol):
    def load_tokenizer(self, model_name_or_path: str, trust_remote_code: bool = False) -> Any: ...
    def load_hf_config(self, model_name_or_path: str, trust_remote_code: bool = False) -> Any: ...
    def convert_hf_config(self, hf_config: Any) -> ModelConfig: ...
    def load_hf_model(
        self,
        model_name_or_path: str,
        device: str = "cpu",
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ) -> Any: ...
    def extract_weights(self, hf_model: Any) -> AdapterWeights: ...
    def move_weights_to_device(
        self,
        weights: AdapterWeights,
        device: torch.device,
        dtype: torch.dtype,
    ) -> AdapterWeights: ...
