"""HuggingFace 模型加载器

第五阶段重构：使用 Adapter 模式加载模型，
支持从 HF Hub 加载 tokenizer、config、模型，并提取权重。
"""

from __future__ import annotations

from typing import Optional

import torch
from miniservellm.model_adapter.adapter_factory import create_model_adapter


def load_model_bundle(
    adapter=None,
    model_name_or_path: str = "",
    trust_remote_code: bool = False,
    load_model_device: str = "cpu",
    load_dtype: Optional[torch.dtype] = None,
):
    """加载模型全套资源

    Args:
        adapter: 模型适配器（如 Qwen2Adapter）
        model_name_or_path: 模型名称或路径
        trust_remote_code: 是否信任远程代码
        load_model_device: 模型加载设备
        load_dtype: 模型加载精度

    Returns:
        (tokenizer, hf_config, model_config, hf_model, weights)
    """
    if adapter is None:
        config_loader = create_model_adapter
        # The first config read is intentionally done through Transformers so
        # the factory can select the correct model-family adapter.
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        adapter = config_loader(hf_config)
    tokenizer = adapter.load_tokenizer(
        model_name_or_path=model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    hf_config = adapter.load_hf_config(
        model_name_or_path=model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    model_config = adapter.convert_hf_config(hf_config)
    hf_model = adapter.load_hf_model(
        model_name_or_path=model_name_or_path,
        device=load_model_device,
        torch_dtype=load_dtype,
        trust_remote_code=trust_remote_code,
    )
    weights = adapter.extract_weights(hf_model)
    return tokenizer, hf_config, model_config, hf_model, weights
