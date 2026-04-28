from dataclasses import dataclass
from typing import Optional


@dataclass
class StepResult:
    request_id: str
    next_token_id: int
    finished: bool
    text_delta: Optional[str] = None
