import sys
import uuid
from pathlib import Path

import torch

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
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str):
    if device == "cpu":
        return "float32"
    return "float16"


def main():
    device = pick_device()
    dtype = pick_dtype(device)

    model_cfg = ModelConfig(
        device=device,
        dtype=dtype,
    )

    print(f"[INFO] device={model_cfg.device}, dtype={model_cfg.dtype}, model={model_cfg.model_name}")

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

    tokenizer_adapter = TokenizerAdapter(tokenizer)
    model_runner = HFModelRunner(model, model_cfg.device)
    sampler = Sampler()
    prefill_executor = PrefillExecutor(model_runner, sampler)
    decode_executor = DecodeExecutor(model_runner, sampler)
    engine = InferenceEngine(tokenizer_adapter, prefill_executor, decode_executor)

    user_input = input("User> ").strip()
    prompt_text = tokenizer_adapter.build_prompt(user_input)
    prompt_token_ids = tokenizer_adapter.encode(prompt_text)

    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)

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

    output_text = engine.generate(request)

    print("\nAssistant>")
    print(output_text)

    metrics = RequestMetrics(request)
    print("\nMetrics>")
    print(metrics.summary())


if __name__ == "__main__":
    main()
