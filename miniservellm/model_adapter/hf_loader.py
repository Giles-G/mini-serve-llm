"""HuggingFace 模型加载器

负责从 HuggingFace Hub 加载 tokenizer 和模型，
自动根据模型类型选择正确的 Auto 类（CausalLM 或 ImageTextToText）。
"""

import torch
from transformers import AutoTokenizer, AutoConfig


def resolve_torch_dtype(dtype: str):
    """将字符串 dtype 转换为 torch.dtype

    Args:
        dtype: 精度名称，如 "float16", "bfloat16", "float32"

    Returns:
        对应的 torch.dtype

    Raises:
        ValueError: 不支持的 dtype 名称
    """
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


def _get_model_class(model_name: str, trust_remote_code: bool = True):
    """根据模型的 model_type 自动选择正确的 Auto 加载类

    Qwen3.5 等多模态模型需要 AutoModelForImageTextToText，
    纯文本模型（如 Qwen2.5）使用 AutoModelForCausalLM。

    Args:
        model_name: Hugging Face 模型标识符
        trust_remote_code: 是否信任远程代码

    Returns:
        对应的 Auto 模型类
    """
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model_type = getattr(config, "model_type", "")

    # Qwen3.5 等多模态模型需要用 AutoModelForImageTextToText
    if model_type in ("qwen3_5", "qwen2_vl", "qwen2_5_vl"):
        return AutoModelForImageTextToText

    return AutoModelForCausalLM


class HFLoader:
    """HuggingFace 模型加载器

    封装 tokenizer 和模型的加载逻辑，自动处理 dtype 和 device 映射。
    """

    def load_tokenizer(self, model_name: str, trust_remote_code: bool = True):
        """加载 tokenizer

        Args:
            model_name: Hugging Face 模型标识符
            trust_remote_code: 是否信任远程代码

        Returns:
            加载好的 tokenizer 实例
        """
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        return tokenizer

    def load_model(
        self,
        model_name: str,
        device: str = "cpu",
        dtype: str = "float16",
        trust_remote_code: bool = True,
    ):
        """加载模型到指定设备

        自动根据模型类型选择 AutoModelForCausalLM 或 AutoModelForImageTextToText，
        设置精度并移动到目标设备。

        Args:
            model_name: Hugging Face 模型标识符
            device: 目标设备 ("cpu" / "mps" / "cuda")
            dtype: 权重精度 ("float16" / "bfloat16" / "float32")
            trust_remote_code: 是否信任远程代码

        Returns:
            加载好的模型实例（已设为 eval 模式并移至目标设备）
        """
        torch_dtype = resolve_torch_dtype(dtype)
        # 根据模型类型自动选择正确的 Auto 类
        model_cls = _get_model_class(model_name, trust_remote_code)

        model = model_cls.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )

        model.eval()      # 切换到评估模式，关闭 dropout 等
        model.to(device)   # 将模型移至目标设备
        return model
