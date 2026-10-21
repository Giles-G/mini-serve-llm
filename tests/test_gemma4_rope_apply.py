"""RoPE apply must stay equivalent after the CUDA addcmul fast path."""

from __future__ import annotations

import pytest
import torch

from miniservellm.runtime.gemma4_runner import Gemma4EagerTextRunner


def _apply(x, cos, sin):
    runner = object.__new__(Gemma4EagerTextRunner)
    return runner._apply_rope(x, cos, sin)


def _reference(x, cos, sin):
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    half = x.shape[-1] // 2
    rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos[:, None, :] + rot * sin[:, None, :]


def _tol(dtype):
    if dtype == torch.bfloat16:
        return {"atol": 2e-2, "rtol": 2e-3}
    return {"atol": 2e-3, "rtol": 2e-3}


@pytest.mark.parametrize("shape", [(1, 8, 256), (1, 8, 512), (19, 8, 256), (19, 8, 512)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_apply_rope_cuda_matches_reference(shape, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(11)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    cos = torch.randn(shape[0], shape[2], device="cuda", dtype=dtype)
    sin = torch.randn(shape[0], shape[2], device="cuda", dtype=dtype)
    out = _apply(x, cos, sin)
    ref = _reference(x, cos, sin)
    assert out.dtype == x.dtype and out.shape == x.shape
    torch.testing.assert_close(out, ref, **_tol(dtype))


@pytest.mark.parametrize("shape", [(1, 8, 256), (19, 8, 512)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_apply_rope_cpu_matches_reference(shape, dtype):
    torch.manual_seed(12)
    x = torch.randn(shape, dtype=dtype)
    cos = torch.randn(shape[0], shape[2], dtype=dtype)
    sin = torch.randn(shape[0], shape[2], dtype=dtype)
    out = _apply(x, cos, sin)
    ref = _reference(x, cos, sin)
    torch.testing.assert_close(out, ref, **_tol(dtype))
