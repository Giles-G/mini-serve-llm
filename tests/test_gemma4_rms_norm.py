"""Tests for the Gemma4 RMSNorm fast path.

The CUDA path uses ``torch.nn.functional.rms_norm`` (one fused launch); the
CPU/MPS path keeps the eager formula. Both must stay numerically equivalent to
the reference formula within the activation dtype's rounding budget.
"""

from __future__ import annotations

import pytest
import torch

from miniservellm.runtime.gemma4_runner import _rms_norm


def _reference(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    x32 = x.float()
    normed = x32 * torch.pow(x32.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    if weight is not None:
        normed = normed * weight.float()
    return normed.to(x.dtype)


def _tol(dtype: torch.dtype) -> dict:
    if dtype == torch.bfloat16:
        return {"atol": 2e-2, "rtol": 2e-3}
    return {"atol": 2e-3, "rtol": 2e-3}


@pytest.mark.parametrize(
    "shape",
    [(1, 1536), (19, 1536), (1, 8, 256), (1, 8, 512), (1, 4, 8960)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rms_norm_matches_reference_cpu(shape, dtype):
    torch.manual_seed(3)
    x = torch.randn(shape, dtype=dtype) * 3
    weight = torch.randn(shape[-1], dtype=dtype)
    out = _rms_norm(x, weight, 1e-6)
    ref = _reference(x, weight, 1e-6)
    assert out.dtype == x.dtype and out.shape == x.shape
    torch.testing.assert_close(out, ref, **_tol(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rms_norm_without_weight(dtype):
    torch.manual_seed(4)
    x = torch.randn(2, 1536, dtype=dtype)
    out = _rms_norm(x, None, 1e-6)
    ref = _reference(x, None, 1e-6)
    torch.testing.assert_close(out, ref, **_tol(dtype))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    "shape",
    [(1, 1536), (19, 1536), (1, 8, 256), (1, 8, 512), (1, 4, 8960)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rms_norm_cuda_matches_reference(shape, dtype):
    torch.manual_seed(5)
    x = (torch.randn(shape, device="cuda", dtype=dtype) * 3)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype)
    out = _rms_norm(x, weight, 1e-6)
    ref = _reference(x, weight, 1e-6)
    assert out.dtype == x.dtype and out.shape == x.shape
    torch.testing.assert_close(out, ref, **_tol(dtype))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_rms_norm_cuda_uses_fused_path(monkeypatch):
    import miniservellm.runtime.gemma4_runner as runner

    calls = {}

    def fake_fused(x, normalized_shape, weight, eps):
        calls["shape"] = tuple(normalized_shape)
        calls["weight"] = weight
        return x

    monkeypatch.setattr(runner, "_FUSED_RMS_NORM", fake_fused)
    x = torch.randn(1, 1536, device="cuda", dtype=torch.float16)
    weight = torch.ones(1536, device="cuda", dtype=torch.float16)
    out = runner._rms_norm(x, weight, 1e-6)
    assert out is x
    assert calls["shape"] == (1536,)
    assert calls["weight"] is weight


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_rms_norm_weightless_keeps_eager_path(monkeypatch):
    import miniservellm.runtime.gemma4_runner as runner

    calls = []
    monkeypatch.setattr(
        runner,
        "_FUSED_RMS_NORM",
        lambda *args: calls.append(args) or args[0],
    )
    x = torch.randn(1, 256, device="cuda", dtype=torch.float16)
    out = runner._rms_norm(x, None, 1e-6)
    assert calls == []
    assert out.dtype == x.dtype and out.shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    "shape", [(1, 1536), (19, 1536), (1, 8, 256), (1, 8, 512)]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_add_rms_norm_matches_add_then_norm(shape, dtype):
    from miniservellm.runtime.nn_ops import fused_add_rms_norm

    torch.manual_seed(13)
    x = torch.randn(shape, device="cuda", dtype=dtype) * 2
    residual = torch.randn(shape, device="cuda", dtype=dtype) * 2
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        normed, summed = fused_add_rms_norm(x, residual, weight, 1e-6)
    stream.synchronize()

    expected_sum = residual + x
    expected_norm = _rms_norm(expected_sum, weight, 1e-6)
    torch.testing.assert_close(summed, expected_sum, atol=0.0, rtol=0.0)
    # The fused kernel normalizes the unrounded FP32 sum while the reference
    # first rounds the add to the activation dtype; allow one extra ULP.
    tol = {"atol": 1e-2, "rtol": 2e-3} if dtype == torch.float16 else {
        "atol": 1e-1, "rtol": 2e-3
    }
    torch.testing.assert_close(normed, expected_norm, **tol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rms_norm_cuda_weightless_matches_reference(dtype):
    torch.manual_seed(6)
    x = torch.randn(1, 8, 256, device="cuda", dtype=dtype)
    out = _rms_norm(x, None, 1e-6)
    ref = _reference(x, None, 1e-6)
    assert out.dtype == x.dtype and out.shape == x.shape
    torch.testing.assert_close(out, ref, **_tol(dtype))


