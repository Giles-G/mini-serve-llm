"""Optional Triton flash-attention for Gemma4 prefill.

Supports head_dim 256/512, GQA, causal masking and the sliding window.  The
kernel keeps the online-softmax state in FP32 and writes FP16 output; it is
only used for prefill, never for single-request decode.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _flash_kernel(
        Q, K, V, O,
        T, TOTAL, START, WINDOW,
        H: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_d = tl.arange(0, D)
        qm = H * D
        qh = D
        kn = HKV * D
        kh = D
        om = H * D
        oh = D
        khd = pid_h // (H // HKV)

        q = tl.load(
            Q + offs_m[:, None] * qm + pid_h * qh + offs_d[None, :],
            mask=offs_m[:, None] < T, other=0.0,
        )
        m_i = tl.full([BM], -float("inf"), tl.float32)
        l_i = tl.zeros([BM], tl.float32)
        acc = tl.zeros([BM, D], tl.float32)

        pos_m = START + offs_m
        lo = tl.maximum(0, START + pid_m * BM - WINDOW + 1)
        hi = tl.minimum(TOTAL, START + (pid_m + 1) * BM)
        for start_n in range(lo, hi, BN):
            offs_n = start_n + tl.arange(0, BN)
            k = tl.load(
                K + offs_n[:, None] * kn + khd * kh + offs_d[None, :],
                mask=offs_n[:, None] < TOTAL, other=0.0,
            )
            qk = tl.dot(q, tl.trans(k))
            keep = (
                (pos_m[:, None] >= offs_n[None, :])
                & (pos_m[:, None] - offs_n[None, :] < WINDOW)
                & (offs_n[None, :] < TOTAL)
            )
            qk = tl.where(keep, qk, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.exp(qk - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(
                V + offs_n[:, None] * kn + khd * kh + offs_d[None, :],
                mask=offs_n[:, None] < TOTAL, other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
            m_i = m_new

        out = acc / l_i[:, None]
        tl.store(
            O + offs_m[:, None] * om + pid_h * oh + offs_d[None, :],
            out.to(tl.float16), mask=offs_m[:, None] < T,
        )


def flash_prefill_attention(q, k, v, start: int, total: int, window: int):
    """q [T,H,D], k/v [total,Hkv,D] -> [T,H,D] (FP16, FP32 online softmax)."""
    if not _HAS_TRITON:
        raise RuntimeError("triton unavailable")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise RuntimeError("flash prefill requires CUDA")
    if q.dtype != torch.float16 or k.dtype != torch.float16 or v.dtype != torch.float16:
        raise RuntimeError("flash prefill requires fp16")
    t, h, d = q.shape
    hkv = k.shape[1]
    if d not in (256, 512) or t == 0 or total <= 0:
        raise RuntimeError("unsupported flash prefill shape")
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()
    out = torch.empty_like(q)
    bm = 16
    bn = 32 if d == 256 else 16
    grid = (triton.cdiv(t, bm), h)
    _flash_kernel[grid](
        q, k, v, out, t, total, start, window, h, hkv, d, bm, bn,
        num_warps=4,
    )
    return out
