"""全局配置模块

定义模型名称、模型配置和推理引擎配置。
切换模型时只需修改 MODEL_NAME 即可。
"""

from dataclasses import dataclass

# Hugging Face 模型标识符，切换模型时只改这一处
# Mac 上推荐使用 Qwen2.5 系列（纯 full attention，兼容性好）
# Windows/Linux + NVIDIA GPU 可使用 Qwen3.5 系列（需要安装 fla + causal-conv1d）
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class ModelConfig:
    """模型加载配置

    Attributes:
        model_name: Hugging Face 模型标识符
        device: 推理设备，"mps"(Apple Silicon) / "cpu" / "cuda"(NVIDIA GPU)
        dtype: 模型权重精度，"float16" / "bfloat16" / "float32"
        trust_remote_code: 是否信任模型仓库中的远程代码
    """
    model_name: str = MODEL_NAME
    device: str = "mps"       # "mps" / "cpu" / "cuda"
    dtype: str = "float16"    # "float16" / "bfloat16" / "float32"
    trust_remote_code: bool = True


@dataclass
class EngineConfig:
    """推理引擎默认配置

    Attributes:
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度，0.0 为贪心解码，越高越随机
        top_k: top-k 采样候选数，0 表示不限制
        top_p: nucleus sampling 累积概率阈值
    """
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_k: int = 20
    top_p: float = 0.9
