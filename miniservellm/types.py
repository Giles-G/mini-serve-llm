from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ModelForwardOutput:
    logits: Any
    past_key_values: Optional[Any]
