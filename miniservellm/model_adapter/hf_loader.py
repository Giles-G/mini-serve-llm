import torch
from transformers import AutoTokenizer, AutoConfig


def resolve_torch_dtype(dtype: str):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


def _get_model_class(model_name: str, trust_remote_code: bool = True):
    """根据 config 中的 model_type 自动选择正确的 Auto 类。"""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model_type = getattr(config, "model_type", "")

    # Qwen3.5 等多模态模型需要用 AutoModelForImageTextToText
    if model_type in ("qwen3_5", "qwen2_vl", "qwen2_5_vl"):
        return AutoModelForImageTextToText

    return AutoModelForCausalLM


class HFLoader:
    def load_tokenizer(self, model_name: str, trust_remote_code: bool = True):
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
        torch_dtype = resolve_torch_dtype(dtype)
        model_cls = _get_model_class(model_name, trust_remote_code)

        model = model_cls.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )

        model.eval()
        model.to(device)
        return model
