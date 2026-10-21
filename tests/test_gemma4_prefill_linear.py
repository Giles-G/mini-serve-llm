"""Prefill projection dispatch: one-row GEMV vs dequantize-once GEMM."""

from __future__ import annotations

import torch

from miniservellm.runtime.gemma4_engine_runner import Gemma4EngineModelRunner
from miniservellm.runtime.gemma4_quant import Int4Weight


class _CountingWeight:
    def __init__(self, inner: Int4Weight):
        self.inner = inner
        self.packed = inner.packed
        self.scales = inner.scales
        self.dequant_calls = 0

    def dequant_dense(self):
        self.dequant_calls += 1
        return self.inner.dequant_dense()

    def t(self):
        return self.inner.t()


def test_prefill_linear_uses_gemm_only_above_threshold():
    torch.manual_seed(3)
    packed = torch.randint(0, 256, (32, 64), dtype=torch.uint8)
    scales = torch.rand(64, 1, dtype=torch.float16) * 0.1
    weight = _CountingWeight(Int4Weight(packed, scales, 64, torch.float16))

    runner = object.__new__(Gemma4EngineModelRunner)
    runner._prefill_gemm_min_rows = 4

    one = torch.randn(1, 64, dtype=torch.float16)
    out_one = runner._linear(weight, one)
    assert weight.dequant_calls == 0
    torch.testing.assert_close(out_one, one @ weight.t(), atol=0, rtol=0)

    chunk = torch.randn(8, 64, dtype=torch.float16)
    out_chunk = runner._linear(weight, chunk)
    assert weight.dequant_calls == 1
    # Dequantize-once + FP16 GEMM rounds the dequantized weights to FP16;
    # it must stay within the same activation-dtype tolerance as the GEMV.
    torch.testing.assert_close(
        out_chunk.float(), (chunk @ weight.t()).float(), atol=2e-2, rtol=2e-2
    )
