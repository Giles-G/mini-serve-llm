from dataclasses import dataclass

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class ModelConfig:
    model_name: str = MODEL_NAME
    device: str = "mps"       # "mps" / "cpu" / "cuda"
    dtype: str = "float16"    # "float16" / "bfloat16" / "float32"
    trust_remote_code: bool = True


@dataclass
class EngineConfig:
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_k: int = 20
    top_p: float = 0.9
