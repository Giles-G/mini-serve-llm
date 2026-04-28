"""CLI 对话脚本

加载模型，接收用户输入，执行推理并输出结果和性能指标。
运行：python scripts/run_cli_chat.py
"""

import sys
import uuid
from pathlib import Path

import torch

# 将项目根目录加入 Python 搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from miniservellm.config import ModelConfig
from miniservellm.model_adapter.hf_loader import HFLoader
from miniservellm.model_adapter.tokenizer_adapter import TokenizerAdapter
from miniservellm.model_adapter.hf_model_runner import HFModelRunner
from miniservellm.runtime.sampler import Sampler
from miniservellm.runtime.prefill import PrefillExecutor
from miniservellm.runtime.decode import DecodeExecutor
from miniservellm.runtime.inference_engine import InferenceEngine
from miniservellm.scheduler.request import Request, SamplingParams
from miniservellm.benchmark.metrics import RequestMetrics


def pick_device():
    """自动选择推理设备

    优先级：CUDA > MPS (Apple Silicon) > CPU

    Returns:
        设备名称字符串
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str):
    """根据设备选择合适的精度

    CPU 使用 float32（MPS/CUDA 的 float16 在 CPU 上不支持），
    GPU 使用 float16 以节省显存和加速计算。

    Args:
        device: 设备名称

    Returns:
        精度名称字符串
    """
    if device == "cpu":
        return "float32"
    return "float16"


def main():
    # 自动选择设备和精度
    device = pick_device()
    dtype = pick_dtype(device)

    model_cfg = ModelConfig(
        device=device,
        dtype=dtype,
    )

    print(f"[INFO] device={model_cfg.device}, dtype={model_cfg.dtype}, model={model_cfg.model_name}")

    # 加载 tokenizer 和模型
    loader = HFLoader()
    tokenizer = loader.load_tokenizer(
        model_cfg.model_name,
        trust_remote_code=model_cfg.trust_remote_code,
    )
    model = loader.load_model(
        model_cfg.model_name,
        device=model_cfg.device,
        dtype=model_cfg.dtype,
        trust_remote_code=model_cfg.trust_remote_code,
    )

    # 组装推理管线：TokenizerAdapter → ModelRunner → Sampler → Prefill/Decode → Engine
    tokenizer_adapter = TokenizerAdapter(tokenizer)
    model_runner = HFModelRunner(model, model_cfg.device)
    sampler = Sampler()
    prefill_executor = PrefillExecutor(model_runner, sampler)
    decode_executor = DecodeExecutor(model_runner, sampler)
    engine = InferenceEngine(tokenizer_adapter, prefill_executor, decode_executor)

    # 获取用户输入
    user_input = input("User> ").strip()
    # 使用 chat template 构建 prompt
    prompt_text = tokenizer_adapter.build_prompt(user_input)
    # 编码为 token ids
    prompt_token_ids = tokenizer_adapter.encode(prompt_text)

    # 收集 stop token（遇到 EOS 自动停止）
    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)

    # 构造推理请求
    request = Request(
        request_id=str(uuid.uuid4()),
        prompt=prompt_text,
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(
            max_new_tokens=2048,
            temperature=0.7,
            top_k=20,
            top_p=0.9,
            stop_token_ids=stop_token_ids,
        ),
    )

    # 执行推理
    output_text = engine.generate(request)

    # 输出结果
    print("\nAssistant>")
    print(output_text)

    # 输出性能指标
    metrics = RequestMetrics(request)
    print("\nMetrics>")
    print(metrics.summary())


if __name__ == "__main__":
    main()
