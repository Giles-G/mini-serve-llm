from dataclasses import dataclass, field
import time


@dataclass
class SamplingParams:
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    stop_token_ids: list[int] = field(default_factory=list)


@dataclass
class Request:
    request_id: str
    prompt: str
    prompt_token_ids: list[int]

    generated_token_ids: list[int] = field(default_factory=list)
    sampling_params: SamplingParams = field(default_factory=SamplingParams)

    past_key_values: object | None = None
    last_token_id: int | None = None

    arrival_time: float = field(default_factory=time.time)
    first_token_time: float | None = None
    finish_time: float | None = None
