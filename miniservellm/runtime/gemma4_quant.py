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
from safetensors import safe_open

from miniservellm.config import ModelConfig
from miniservellm.model_adapter.adapters.gemma4_adapter import Gemma4LayerWeights
from miniservellm.model_adapter.gemma4_config import build_gemma4_layer_specs
from miniservellm.runtime.nn_ops import quantize_weight_group
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

    def __init__(self, packed: torch.Tensor, scales: torch.Tensor, group_size: int):
        self.packed = packed
        self.scales = scales
        self.group_size = group_size

    # -- tensor-ish surface -------------------------------------------------
    def t(self) -> _Int4MatmulProxy:
        return _Int4MatmulProxy(self)

    def __getitem__(self, ids: torch.Tensor) -> torch.Tensor:
        return self.dequant_rows(ids)

    @property
    def dtype(self) -> torch.dtype:
        # Compute dtype of the dequantized representation.
        return torch.bfloat16

    @property
    def device(self) -> torch.device:
        return self.packed.device

    def nbytes(self) -> int:
        return self.packed.numel() + self.scales.numel() * self.scales.element_size()

    # -- kernels -------------------------------------------------------------
    def _unpack_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """Dequantize the given packed rows to bf16 [T, K]."""
        packed = self.packed.index_select(0, rows)
        low = (packed.to(torch.int16) & 0x0F).to(torch.int16) - 8
        high = (packed.to(torch.int16) >> 4).to(torch.int16) - 8
        w_q = torch.empty(
            (packed.shape[0] * 2, packed.shape[1]), dtype=torch.bfloat16, device=packed.device
        )
        w_q[0::2] = low
        w_q[1::2] = high
        n_groups = self.scales.shape[1]
        w = w_q.view(packed.shape[0] * 2, n_groups, self.group_size)
        w = w * self.scales.to(torch.bfloat16).unsqueeze(-1)
        return w.view(packed.shape[0] * 2, -1)

    def dequant_rows(self, ids: torch.Tensor) -> torch.Tensor:
        """Dequantize selected original rows: byte i holds rows 2i (low) and 2i+1 (high)."""
        ids = ids.to(torch.long).reshape(-1)
        rows = ids >> 1
        packed = self.packed.index_select(0, rows)
        low = (packed.to(torch.int16) & 0x0F).to(torch.int16) - 8
        high = (packed.to(torch.int16) >> 4).to(torch.int16) - 8
        is_odd = (ids & 1).to(torch.bool).unsqueeze(-1)
        w_q = torch.where(is_odd, high, low).to(torch.bfloat16)  # [T, K]
        n_groups = self.scales.shape[1]
        s = self.scales.to(torch.bfloat16).index_select(0, ids)  # [T, n_groups]
        w = w_q.view(ids.numel(), n_groups, self.group_size) * s.unsqueeze(-1)
        return w.view(ids.numel(), -1)

    def dequant_dense(self) -> torch.Tensor:
        """Dequantize the full matrix to bf16 [N, K]."""
        n2, k = self.packed.shape
        low = (self.packed.to(torch.int16) & 0x0F).to(torch.int16) - 8
        high = (self.packed.to(torch.int16) >> 4).to(torch.int16) - 8
        w_q = torch.empty((n2 * 2, k), dtype=torch.bfloat16, device=self.packed.device)
        w_q[0::2] = low
        w_q[1::2] = high
        n_groups = self.scales.shape[1]
        w = w_q.view(n2 * 2, n_groups, self.group_size) * self.scales.to(torch.bfloat16).unsqueeze(-1)
        return w.view(n2 * 2, k)

    def matmul(self, x: torch.Tensor) -> torch.Tensor:
        """y = x @ W.T via fused CUDA kernel when available, else dequant."""
        try:
            from mini_llm_kernels.kernels.int4_matmul import int4_dequant_matmul

            return int4_dequant_matmul(x, self.packed, self.scales)
        except (ImportError, RuntimeError):
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


def _quantize_tensor(weight: torch.Tensor, group_size: int, device: torch.device) -> Int4Weight:
    """Quantize one [out, in] tensor and move the packed result to device."""
    w_q, scales = quantize_weight_group(weight.float(), bits=4, group_size=group_size)
    packed = _pack_int4(w_q).to(device)
    return Int4Weight(packed, scales.to(device=device, dtype=torch.float16), group_size)


def quantize_gemma4_weights(
    model_dir: str | Path,
    model_config: ModelConfig,
    device: str | torch.device = "cpu",
    group_size: int = 64,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> Gemma4Int4Weights:
    """Stream-quantize the checkpoint text backbone to INT4.

    Reads each tensor from safetensors individually, quantizes on CPU, moves
    the packed result to ``device`` and frees the fp copy — peak memory stays
    around one tensor instead of the whole model.
    """
    path = Path(model_dir).expanduser()
    device = torch.device(device)
    specs = build_gemma4_layer_specs(load_gemma4_config(path))
    weights_file = safe_open(str(path / "model.safetensors"), framework="pt")

    def read(name: str) -> torch.Tensor:
        return weights_file.get_tensor(f"{TEXT_PREFIX}{name}").to(torch.float32)

    def quant(name: str) -> QuantWeight:
        tensor = read(name)
        result = _quantize_tensor(tensor, group_size, device)
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
