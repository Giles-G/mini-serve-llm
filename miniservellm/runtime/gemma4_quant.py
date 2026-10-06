"""INT4 weight-only quantization for the Gemma4 text runner (stage G).

Format: symmetric group quantization (group_size=64, values [-8, 7]),
packed two weights per byte — the same layout used by the Qwen runtime and
the ``mini_llm_kernels`` CUDA kernel, so on CUDA machines
``nn_ops._int4_linear`` automatically selects the fused kernel while MPS
falls back to the PyTorch dequantize matmul.

Memory math (E2B, 5.1B params):
    bf16 weights            ~9.5 GB
    INT4 packed linears      ~1.2 GB  (2.35B params)
    INT4 packed PLE table    ~1.2 GB  (2.35B params)
    INT4 packed embeddings   ~0.2 GB  (0.40B params, tied lm_head)
    scales (fp16, gs=64)    ~0.08 GB
    total                   ~2.7 GB

Quantization streams tensor-by-tensor from the safetensors file so peak
host memory stays near one tensor + the packed result (16 GB machines can
quantize without holding both copies).

Embedding rows are dequantized per lookup (only T rows per step); the tied
lm_head is dequantized fully for the logits matmul.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch

from miniservellm.config import ModelConfig
from miniservellm.model_adapter.adapters.gemma4_adapter import Gemma4LayerWeights
from miniservellm.model_adapter.gemma4_config import build_gemma4_layer_specs
from miniservellm.runtime.nn_ops import quantize_weight_group
from miniservellm.safetensors_io import StreamingSafetensors
from miniservellm.model_adapter.adapters.gemma4_hf_model import (
    TEXT_PREFIX,
    load_gemma4_config,
)

QuantWeight = Tuple[torch.Tensor, torch.Tensor]  # (packed uint8 [N/2, K], scales [N, K/gs])


class _Int4MatmulProxy:
    """Returned by ``Int4Weight.t()`` so ``x @ w.t()`` keeps working."""

    def __init__(self, weight: "Int4Weight"):
        self.weight = weight

    def __rmatmul__(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight.matmul(x)


class Int4Weight:
    """INT4-packed [out, in] weight that duck-types the tensor call sites.

    - ``x @ w.t()`` → dequantize-matmul (CUDA fused kernel when available,
      otherwise the PyTorch unpack path below);
    - ``w[ids]``    → dequantize only the selected rows (embedding/PLE
      lookups touch T rows per step instead of the whole table).
    """

    def __init__(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        compute_dtype: torch.dtype = torch.bfloat16,
    ):
        self.packed = packed
        self.scales = scales
        self.group_size = group_size
        self.compute_dtype = compute_dtype

    # -- tensor-ish surface -------------------------------------------------
    def t(self) -> _Int4MatmulProxy:
        return _Int4MatmulProxy(self)

    def __getitem__(self, ids: torch.Tensor) -> torch.Tensor:
        return self.dequant_rows(ids)

    @property
    def dtype(self) -> torch.dtype:
        # Compute dtype of the dequantized representation.
        return self.compute_dtype

    @property
    def device(self) -> torch.device:
        return self.packed.device

    def nbytes(self) -> int:
        return self.packed.numel() + self.scales.numel() * self.scales.element_size()

    # -- kernels -------------------------------------------------------------
    def _unpack_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """Dequantize the given packed rows to the configured compute dtype."""
        packed = self.packed.index_select(0, rows)
        low = (packed.to(torch.int16) & 0x0F).to(torch.int16) - 8
        high = (packed.to(torch.int16) >> 4).to(torch.int16) - 8
        w_q = torch.empty(
            (packed.shape[0] * 2, packed.shape[1]), dtype=self.compute_dtype, device=packed.device
        )
        w_q[0::2] = low
        w_q[1::2] = high
        n_groups = self.scales.shape[1]
        w = w_q.view(packed.shape[0] * 2, n_groups, self.group_size)
        w = w * self.scales.to(self.compute_dtype).unsqueeze(-1)
        return w.view(packed.shape[0] * 2, -1)

    def dequant_rows(self, ids: torch.Tensor) -> torch.Tensor:
        """Dequantize selected original rows: byte i holds rows 2i (low) and 2i+1 (high)."""
        ids = ids.to(torch.long).reshape(-1)
        rows = ids >> 1
        packed = self.packed.index_select(0, rows)
        low = (packed.to(torch.int16) & 0x0F).to(torch.int16) - 8
        high = (packed.to(torch.int16) >> 4).to(torch.int16) - 8
        is_odd = (ids & 1).to(torch.bool).unsqueeze(-1)
        w_q = torch.where(is_odd, high, low).to(self.compute_dtype)  # [T, K]
        n_groups = self.scales.shape[1]
        s = self.scales.to(self.compute_dtype).index_select(0, ids)  # [T, n_groups]
        w = w_q.view(ids.numel(), n_groups, self.group_size) * s.unsqueeze(-1)
        return w.view(ids.numel(), -1)

    def dequant_dense(self) -> torch.Tensor:
        """Dequantize the full matrix to the configured compute dtype."""
        n2, k = self.packed.shape
        low = (self.packed.to(torch.int16) & 0x0F).to(torch.int16) - 8
        high = (self.packed.to(torch.int16) >> 4).to(torch.int16) - 8
        w_q = torch.empty((n2 * 2, k), dtype=self.compute_dtype, device=self.packed.device)
        w_q[0::2] = low
        w_q[1::2] = high
        n_groups = self.scales.shape[1]
        w = w_q.view(n2 * 2, n_groups, self.group_size) * self.scales.to(self.compute_dtype).unsqueeze(-1)
        return w.view(n2 * 2, k)

    def matmul(self, x: torch.Tensor) -> torch.Tensor:
        """y = x @ W.T via the fused CUDA kernel when it is actually usable.

        Only a missing/CPU-only kernel falls back to dequantize-then-matmul.
        A ``RuntimeError`` raised *by* the kernel is a real failure (bad
        shapes, OOM, a broken kernel) and is re-raised: silently rerouting it
        to the reference path hides correctness bugs behind a 3-5x slower
        fallback, which is exactly how a wrong fused kernel goes unnoticed.
        """
        try:
            from mini_llm_kernels.kernels.int4_matmul import (
                _HAS_INT4_CUDA,
                int4_dequant_matmul,
            )
        except ImportError:
            _HAS_INT4_CUDA = False

        if _HAS_INT4_CUDA and x.is_cuda and self.packed.is_cuda:
            return int4_dequant_matmul(x, self.packed, self.scales)
        return torch.nn.functional.linear(x, self.dequant_dense())


@dataclass
class Gemma4Int4LayerWeights:
    """Layer weights with INT4-packed projections; norms stay in fp dtype."""

    q_proj: Int4Weight
    k_proj: Int4Weight
    v_proj: Int4Weight
    o_proj: Int4Weight
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    v_norm: Optional[torch.Tensor]
    input_layernorm: torch.Tensor
    post_attention_layernorm: torch.Tensor
    pre_feedforward_layernorm: torch.Tensor
    post_feedforward_layernorm: torch.Tensor
    post_per_layer_input_norm: torch.Tensor
    gate_proj: Int4Weight
    up_proj: Int4Weight
    down_proj: Int4Weight
    per_layer_input_gate: Int4Weight
    per_layer_projection: Int4Weight
    layer_spec: object
    layer_scalar: Optional[torch.Tensor] = None


@dataclass
class Gemma4Int4Weights:
    """Model weights with INT4-packed projections and embeddings."""

    embed_tokens: Int4Weight
    lm_head: Int4Weight  # tied: same storage as embed_tokens
    final_norm: torch.Tensor
    embed_tokens_per_layer: Int4Weight
    per_layer_model_projection: Int4Weight
    per_layer_projection_norm: torch.Tensor
    layers: list


def _pack_int4(w_q: torch.Tensor) -> torch.Tensor:
    """Pack int8 [-8, 7] weights into uint8 pairs (low nibble = even row)."""
    if w_q.shape[0] % 2 != 0:
        raise ValueError(f"INT4 packing requires an even output size, got {w_q.shape[0]}")
    w_unsigned = (w_q.to(torch.int16) + 8).clamp(0, 15).to(torch.uint8)
    return w_unsigned[0::2] | (w_unsigned[1::2] << 4)


def _quantize_tensor(
    weight: torch.Tensor,
    group_size: int,
    device: torch.device,
    compute_dtype: torch.dtype = torch.bfloat16,
    chunk_rows: int = 32768,
) -> Int4Weight:
    """Quantize one [out, in] tensor and move the packed result to device.

    Runs in row chunks so the fp staging copy never exceeds
    ``chunk_rows * in_features`` elements. The largest Gemma4 tensor is the
    262144-row embedding table; at fp32 that single tensor is >500 MB, which
    is the difference between fitting and not fitting on a 16 GB-or-less
    host. Chunking is numerically identical: each group lies inside one row.
    """
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features ({in_features}) must be divisible by group_size ({group_size})"
        )
    chunk_rows = max(group_size, chunk_rows)

    packed_chunks = []
    scale_chunks = []
    for start in range(0, out_features, chunk_rows):
        rows = weight[start : start + chunk_rows].float()
        w_q, scales = quantize_weight_group(rows, bits=4, group_size=group_size)
        packed_chunks.append(_pack_int4(w_q))
        scale_chunks.append(scales)
        del rows, w_q

    packed = torch.cat(packed_chunks, dim=0).to(device)
    scales = torch.cat(scale_chunks, dim=0).to(device=device, dtype=torch.float16)
    return Int4Weight(packed, scales, group_size, compute_dtype)


def quantize_checkpoint_tensor(
    source: StreamingSafetensors,
    name: str,
    device: str | torch.device = "cpu",
    group_size: int = 64,
    compute_dtype: torch.dtype = torch.bfloat16,
    chunk_rows: int = 4096,
) -> Int4Weight:
    """Quantize one checkpoint tensor by streaming it row-chunk by row-chunk.

    Needed for the PLE table (``[262144, 8960]``, 4.4 GiB in bf16): its rows
    are read, quantized and released in slices, so neither the file nor the
    tensor is ever materialized whole.
    """
    device = torch.device(device)
    out_features, in_features = source.shape(name)
    if in_features % group_size != 0:
        raise ValueError(
            f"{name}: in_features ({in_features}) must be divisible by group_size ({group_size})"
        )

    packed = torch.empty(
        (out_features // 2, in_features),
        dtype=torch.uint8,
        device=device,
    )
    scale_buffer = torch.empty(
        (out_features, in_features // group_size),
        dtype=torch.float16,
        device=device,
    )
    for start in range(0, out_features, chunk_rows):
        rows = source.read(name, (start, min(start + chunk_rows, out_features)))
        w_q, row_scales = quantize_weight_group(rows.float(), bits=4, group_size=group_size)
        end = start + rows.shape[0]
        packed[start // 2 : end // 2].copy_(_pack_int4(w_q).to(device))
        scale_buffer[start:end].copy_(row_scales.to(device=device, dtype=torch.float16))
        del rows, w_q, row_scales

    return Int4Weight(packed, scale_buffer, group_size, compute_dtype)


def quantize_gemma4_weights(
    model_dir: str | Path,
    model_config: ModelConfig,
    device: str | torch.device = "cpu",
    group_size: int = 64,
    compute_dtype: torch.dtype = torch.bfloat16,
    prequantized: Optional[dict[str, Int4Weight]] = None,
) -> Gemma4Int4Weights:
    """Stream-quantize the checkpoint text backbone to INT4.

    Reads each tensor from the shard individually, quantizes on CPU, moves the
    packed result to ``device`` and frees the fp copy — peak host memory stays
    around one chunk instead of the whole model.

    ``prequantized`` supplies already-quantized tensors (keyed by their
    checkpoint name) so callers can substitute a chunked result for the two
    multi-gigabyte tables that cannot be held in host RAM as one tensor.
    """
    path = Path(model_dir).expanduser()
    device = torch.device(device)
    specs = build_gemma4_layer_specs(load_gemma4_config(path))
    source = StreamingSafetensors(path / "model.safetensors")
    supplied = prequantized or {}

    def read(name: str) -> torch.Tensor:
        key = f"{TEXT_PREFIX}{name}"
        return source.read(key).to(compute_dtype)

    def quant(name: str) -> Int4Weight:
        key = f"{TEXT_PREFIX}{name}"
        if key in supplied:
            return supplied[key]
        shape = source.shape(key)
        if len(shape) == 2 and shape[0] >= 65536:
            return quantize_checkpoint_tensor(
                source,
                key,
                device=device,
                group_size=group_size,
                compute_dtype=compute_dtype,
            )
        tensor = source.read(key)
        result = _quantize_tensor(tensor, group_size, device, compute_dtype)
        del tensor
        return result

    def keep(name: str) -> torch.Tensor:
        return read(name).to(device=device, dtype=compute_dtype)

    embed_tokens = quant("embed_tokens.weight")
    layers: list = []
    for spec in specs:
        idx = spec.layer_idx
        layers.append(
            Gemma4Int4LayerWeights(
                q_proj=quant(f"layers.{idx}.self_attn.q_proj.weight"),
                k_proj=quant(f"layers.{idx}.self_attn.k_proj.weight"),
                v_proj=quant(f"layers.{idx}.self_attn.v_proj.weight"),
                o_proj=quant(f"layers.{idx}.self_attn.o_proj.weight"),
                q_norm=keep(f"layers.{idx}.self_attn.q_norm.weight"),
                k_norm=keep(f"layers.{idx}.self_attn.k_norm.weight"),
                v_norm=None,
                input_layernorm=keep(f"layers.{idx}.input_layernorm.weight"),
                post_attention_layernorm=keep(f"layers.{idx}.post_attention_layernorm.weight"),
                pre_feedforward_layernorm=keep(f"layers.{idx}.pre_feedforward_layernorm.weight"),
                post_feedforward_layernorm=keep(f"layers.{idx}.post_feedforward_layernorm.weight"),
                post_per_layer_input_norm=keep(f"layers.{idx}.post_per_layer_input_norm.weight"),
                gate_proj=quant(f"layers.{idx}.mlp.gate_proj.weight"),
                up_proj=quant(f"layers.{idx}.mlp.up_proj.weight"),
                down_proj=quant(f"layers.{idx}.mlp.down_proj.weight"),
                per_layer_input_gate=quant(f"layers.{idx}.per_layer_input_gate.weight"),
                per_layer_projection=quant(f"layers.{idx}.per_layer_projection.weight"),
                layer_spec=spec,
                layer_scalar=keep(f"layers.{idx}.layer_scalar"),
            )
        )

    return Gemma4Int4Weights(
        embed_tokens=embed_tokens,
        lm_head=embed_tokens,  # tied
        final_norm=keep("norm.weight"),
        embed_tokens_per_layer=quant("embed_tokens_per_layer.weight"),
        per_layer_model_projection=quant("per_layer_model_projection.weight"),
        per_layer_projection_norm=keep("per_layer_projection_norm.weight"),
        layers=layers,
    )
