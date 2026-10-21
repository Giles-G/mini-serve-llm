"""The engine builds RoPE cos/sin once per attention type per forward."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from miniservellm.runtime.gemma4_engine_runner import Gemma4EngineModelRunner


class _Dummy:
    _rope_types = {"sliding_attention", "full_attention"}
    w = SimpleNamespace(embed_tokens=torch.empty(0, dtype=torch.float16))

    def __init__(self):
        self.calls = []

    @property
    def dtype(self):
        return torch.float16

    def _cos_sin_for(self, attention_type, positions):
        self.calls.append(attention_type)
        return torch.ones(1, 4), torch.zeros(1, 4)


def test_rope_for_builds_once_per_type():
    dummy = _Dummy()
    rope = Gemma4EngineModelRunner._rope_for(dummy, torch.zeros(1))

    assert sorted(dummy.calls) == ["full_attention", "sliding_attention"]
    assert set(rope) == {"full_attention", "sliding_attention"}
    for cos, sin in rope.values():
        assert cos.dtype == torch.float16
        assert sin.dtype == torch.float16
