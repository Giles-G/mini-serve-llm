"""Stage E engine runner: wires the Gemma4 eager forward into Stage5Engine.

Implements the :class:`ModelRunner` protocol (fresh/incremental prefill +
decode) on top of :class:`Gemma4EagerTextRunner` primitives and
:class:`Gemma4PagedKVCacheManager`. Decode batches 2-16 share one forward;
prefill batches concatenate linear/MLP rows across requests while keeping
attention and KV writes per request. N=1 keeps the specialized GEMV path.

Key semantics preserved from the official implementation:
- KV sharing: layers 15..34 never compute K/V; they gather the source
  layer's entry, which by construction already contains the current
  chunk/token because the source layer runs earlier in the layer loop.
- Sliding window: enforced by slicing the gathered K/V to the last
  ``sliding_window`` positions instead of building masks (equivalent for
  causal decoding and contiguous prefill chunks).
- PLE: recomputed per chunk from the chunk token ids (identity branch) and
  chunk embeddings (context branch), exactly like the official forward.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import os
from typing import Dict, List, Optional

import torch

from miniservellm.cache.gemma4_paged_kv_cache import Gemma4PagedKVCacheManager
from miniservellm.config import EngineConfig, ModelConfig
from miniservellm.model_adapter.adapters.gemma4_adapter import Gemma4TextWeights
from miniservellm.runtime.gemma4_runner import (
    SLIDING,
    Gemma4EagerTextRunner,
    _rms_norm,
)
from miniservellm.runtime.nn_ops import fused_add_rms_norm


def _gelu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """GELU(tanh) * up in one launch when the fused kernel is available."""
    if (
        gate.is_cuda
        and up.is_cuda
        and gate.dtype == torch.float16
        and up.dtype == torch.float16
        and gate.dim() == 2
        and up.dim() == 2
        and gate.shape == up.shape
        and gate.stride(1) == 1
        and up.stride(1) == 1
        and gate.shape[1] % 2 == 0
    ):
        try:
            from mini_llm_kernels._C import gemma4_gelu_mul

            return gemma4_gelu_mul(gate, up)
        except (ImportError, RuntimeError, ValueError):
            pass
    return torch.nn.functional.gelu(gate, approximate="tanh") * up


def _add_mul(a: torch.Tensor, b: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
    """(a + b) * scalar with the eager add then mul FP16 rounding."""
    if (
        a.is_cuda
        and b.is_cuda
        and scalar.is_cuda
        and a.dtype == torch.float16
        and b.dtype == torch.float16
        and scalar.dtype == torch.float16
        and a.shape == b.shape
        and a.is_contiguous()
        and b.is_contiguous()
        and scalar.numel() == 1
        and a.numel() % 2 == 0
    ):
        try:
            from mini_llm_kernels._C import gemma4_add_mul

            return gemma4_add_mul(a, b, scalar)
        except (ImportError, RuntimeError, ValueError):
            pass
    return (a + b).mul_(scalar)


def _qkv_norm_rope(q, k, v, qw, kw, vw, cos, sin, eps):
    """Fused q/k/v RMSNorm + q/k RoPE; eager fallback for unusual shapes."""
    if (
        os.environ.get("MINI_LLM_FUSED_QKV_NORM", "0") != "0"
        and q.is_cuda
        and k.is_cuda
        and v.is_cuda
        and q.dtype == torch.float16
        and k.dtype == torch.float16
        and v.dtype == torch.float16
        and q.dim() == 3
        and k.dim() == 3
        and v.dim() == 3
        and q.is_contiguous()
        and k.is_contiguous()
        and v.is_contiguous()
        and cos.is_contiguous()
        and sin.is_contiguous()
        and q.shape[-1] == k.shape[-1] == v.shape[-1]
        and q.shape[-1] <= 512
        and q.shape[-1] % 2 == 0
    ):
        try:
            from mini_llm_kernels._C import gemma4_qkv_norm_rope

            return gemma4_qkv_norm_rope(q, k, v, qw, kw, vw, cos, sin, eps)
        except (ImportError, RuntimeError, ValueError):
            pass
    def norm(x, w):
        if hasattr(torch.nn.functional, "rms_norm"):
            return torch.nn.functional.rms_norm(x, (x.shape[-1],), w, eps)
        x32 = x.float()
        n = x32 * torch.pow(x32.pow(2).mean(-1, keepdim=True) + eps, -0.5)
        return (n * w.float()).to(x.dtype)

    def rope(x):
        out = x * cos[:, None, :]
        half = x.shape[-1] // 2
        out[..., :half].addcmul_(
            x[..., half:], sin[:, None, :half], value=-1.0
        )
        out[..., half:].addcmul_(
            x[..., :half], sin[:, None, half:], value=1.0
        )
        return out

    return rope(norm(q, qw)), rope(norm(k, kw)), norm(v, vw)


def _has_paged_attention_kernel() -> bool:
    try:
        from mini_llm_kernels.kernels import gemma4_attention as _ga

        return bool(_ga._HAS_GEMMA4_ATTN)
    except Exception:  # noqa: BLE001
        return False
from miniservellm.runtime.metadata import DecodeRequestMetadata, PrefillRequestMetadata
from miniservellm.runtime.model_interface import (
    DecodeModelOutput,
    PrefillModelOutput,
)
from miniservellm.scheduler.request import Request

def _paged_attention_kernel(q, k_cache, v_cache, block_table, context_lens, window):
    """Call the verified Gemma4 paged-attention kernel directly.

    The warp-split kernel assigns one warp per KV segment for a (request,
    query-head) pair and merges the partial softmax states once at the end.
    It removes the safe kernel's per-token block barriers and is faster from
    short contexts through 4k. ``MINI_LLM_GEMMA4_ATTN_IMPL=safe`` keeps the
    old kernel available as a fallback.
    """
    from mini_llm_kernels import _C as _ext

    impl = os.environ.get("MINI_LLM_GEMMA4_ATTN_IMPL", "warpsplit")
    if impl == "warpsplit":
        fn = getattr(_ext, "gemma4_decode_attention_warpsplit", None)
        if fn is not None:
            return fn(q, k_cache, v_cache, block_table, context_lens, int(window))
    return _ext.gemma4_decode_attention(
        q, k_cache, v_cache, block_table, context_lens, int(window)
    )


def _splitk_attention_kernel(
    q, k_cache, v_cache, block_table, context_lens, window,
    scratch_o, scratch_m, scratch_l, chunk_size,
):
    from mini_llm_kernels import _C as _ext

    return _ext.gemma4_decode_attention_splitk(
        q, k_cache, v_cache, block_table, context_lens, int(window),
        scratch_o, scratch_m, scratch_l, int(chunk_size),
    )


@dataclass
class _DenseKVEntry:
    """Amortized-growth contiguous KV mirror for one layer/request.

    ``torch.cat`` on every decode token copies the complete history and makes
    incremental generation O(T^2).  Keep a geometrically growing buffer and
    expose only the populated prefix to attention instead.
    """

    k: torch.Tensor
    v: torch.Tensor
    length: int


class Gemma4EngineModelRunner(Gemma4EagerTextRunner):
    """Gemma4 runner behind the Stage5Engine ModelRunner protocol."""

    supports_cuda_graph = False

    def __init__(
        self,
        engine_config: EngineConfig,
        model_config: ModelConfig,
        weights: Gemma4TextWeights,
        kv_cache_manager: Gemma4PagedKVCacheManager,
    ):
        # Weights must already live on the target device/dtype (see
        # load_gemma4_text_weights); the eager runner forbids moving here.
        super().__init__(model_config, weights)
        self.engine_config = engine_config
        self.kv_cache_manager = kv_cache_manager
        # Dense per-request KV mirror (stage H): attention reads contiguous
        # tensors (same zero-copy pattern as the direct runner) while the
        # paged pages stay the durable/accounting storage. Mirror size is
        # ~18KB/token/request — negligible. Cleared on request finish.
        self._dense: dict[str, dict[int, _DenseKVEntry]] = {}
        engine_config  # pages remain authoritative for capacity accounting
        # Gemma4 has only two RoPE frequency sets (sliding/full). Building them
        # once per forward instead of once per layer removes ~5 launches and a
        # cos/sin pair per layer from the batch=1 decode path.
        self._rope_types = {lw.layer_spec.attention_type for lw in weights.layers}
        # The dedicated paged-attention kernel is verified by parity tests and
        # is now the CUDA path for both graph and non-graph decode. When it is
        # unavailable the engine keeps the dense mirror as the fallback.
        paged_env = os.environ.get("MINI_LLM_GEMMA4_PAGED")
        self._paged_attention = (
            self.device.type == "cuda"
            and _has_paged_attention_kernel()
            and paged_env != "0"
        )
        self._prefill_gemm_min_rows = int(
            os.environ.get("MINI_LLM_GEMMA4_PREFILL_GEMM_MIN_ROWS", "8")
        )
        self._in_prefill = False
        self._tc_batch_enabled = (
            os.environ.get("MINI_LLM_INT4_TC", "1") != "0"
        )
        self._prefill_attn_chunk_size = int(
            os.environ.get("MINI_LLM_GEMMA4_PREFILL_ATTN_CHUNK", "512")
        )
        self._prefix_cache_enabled = (
            os.environ.get("MINI_LLM_GEMMA4_PREFIX_CACHE", "1") != "0"
        )
        self._prefix_cache_entries = int(
            os.environ.get("MINI_LLM_GEMMA4_PREFIX_CACHE_ENTRIES", "8")
        )
        self._prefix_cache_max_tokens = int(
            os.environ.get("MINI_LLM_GEMMA4_PREFIX_CACHE_MAX_TOKENS", "64")
        )
        self._prefix_cache_max_bytes = int(
            os.environ.get("MINI_LLM_GEMMA4_PREFIX_CACHE_MAX_BYTES",
                           str(64 << 20))
        )
        self._prefix_cache: "OrderedDict[tuple, tuple]" = OrderedDict()
        self._prefix_cache_bytes = 0
        self._prefill_window_slice = (
            os.environ.get("MINI_LLM_GEMMA4_PREFILL_WINDOW_SLICE", "1") != "0"
        )
        self._prefill_flash = (
            os.environ.get("MINI_LLM_GEMMA4_PREFILL_FLASH", "1") != "0"
        )
        self._prefill_flash_min_tokens = int(
            os.environ.get("MINI_LLM_GEMMA4_PREFILL_FLASH_MIN_TOKENS", "256")
        )
        # Combined q/k/v INT4 weights for non-shared layers, built once.
        self._qkv_fused: dict[int, tuple] = {}
        # Combined gate/up INT4 weights (same intermediate size), built once.
        self._gate_up_fused: dict[int, tuple] = {}
        # CUDA graph decode is the default for the batch=1 CUDA serving path;
        # set MINI_LLM_GEMMA4_GRAPH=0 to use non-graph paged decode.
        graph_env = os.environ.get("MINI_LLM_GEMMA4_GRAPH")
        if graph_env is None:
            self._graph_mode = (
                self.device.type == "cuda"
                and engine_config.max_batch_size == 1
                and self._paged_attention
            )
        else:
            self._graph_mode = graph_env == "1"
        self._decode_graph = None
        self._graph_static = None
        self._graphs: dict[bool, torch.cuda.CUDAGraph] = {}
        self._graph_states: dict[bool, dict] = {}
        self._batch_decode_enabled = (
            os.environ.get("MINI_LLM_GEMMA4_BATCH_DECODE", "1") != "0"
        )
        self._batch_graph_mode = (
            self._batch_decode_enabled
            and self.device.type == "cuda"
            and engine_config.max_batch_size > 1
            and self._paged_attention
            and graph_env != "0"
        )
        self._batch_graphs: dict[tuple[int, bool], torch.cuda.CUDAGraph] = {}
        self._batch_graph_states: dict[tuple[int, bool], dict] = {}
        self._splitk_threshold = int(
            os.environ.get("MINI_LLM_GEMMA4_SPLITK_CTX", "1024")
        )
        self._splitk_chunk_size = int(
            os.environ.get("MINI_LLM_GEMMA4_SPLITK_CHUNK", "512")
        )
        self._splitk_scratch: dict[tuple, tuple] = {}
        self._compact_fused_weights()
        if self._prefill_flash:
            try:
                from miniservellm.runtime.flash_prefill import (
                    flash_prefill_attention,
                )

                dims = sorted({spec.head_dim for spec in self.config.layer_specs})
                for dim in dims:
                    if dim not in (256, 512):
                        continue
                    qw = torch.zeros((256, 8, dim), device=self.device,
                                     dtype=torch.float16)
                    kw = torch.zeros((256, 1, dim), device=self.device,
                                     dtype=torch.float16)
                    flash_prefill_attention(qw, kw, kw, 0, 256, 512)
            except Exception:
                self._prefill_flash = False

    def _linear(self, weight, x: torch.Tensor) -> torch.Tensor:
        """Dispatch INT4 matmul for prefill or batched decode.

        Prefill dequantizes a weight once and uses an FP16 GEMM. Decode keeps
        the weight in INT4: rows 1 uses pair_vec8, rows 2-4 pair_batch, and
        rows 5-16 the tensor-core tile-dequant kernel. The tensor-core path is
        only used outside prefill so prompt chunks still get the dequant-once
        GEMM.
        """
        rows = x.shape[0]
        in_prefill = getattr(self, "_in_prefill", True)
        if not in_prefill and rows >= 2 and hasattr(weight, "packed"):
            if rows == 2:
                try:
                    from mini_llm_kernels.kernels.int4_matmul import (
                        int4_dequant_matmul_pair_batch,
                    )
                    return int4_dequant_matmul_pair_batch(
                        x, weight.packed, weight.scales
                    )
                except (ImportError, ValueError):
                    pass
            if rows == 16 and getattr(self, "_tc_batch_enabled", False):
                shaped = tuple(weight.packed.shape)
                parts = 8 if shaped in ((768, 6144), (768, 12288)) else (
                    1 if shaped in (
                        (768, 256), (6144, 1536), (12288, 1536),
                        (131072, 1536),
                    ) else 0)
                if parts:
                    try:
                        from mini_llm_kernels._C import (
                            int4_dequant_matmul_tc_frag,
                            int4_dequant_matmul_tc_frag_ga,
                            int4_dequant_matmul_tc_frag_warps,
                            int4_prepack_frag,
                        )

                        frag = getattr(weight, "_frag_packed", None)
                        if frag is None:
                            frag = int4_prepack_frag(weight.packed)
                            weight._frag_packed = frag
                        if shaped == (768, 256):
                            return int4_dequant_matmul_tc_frag_ga(
                                x, frag, weight.scales, parts
                            )
                        if shaped in ((12288, 1536), (131072, 1536)):
                            return int4_dequant_matmul_tc_frag_warps(
                                x, frag, weight.scales, parts, 8
                            )
                        return int4_dequant_matmul_tc_frag(
                            x, frag, weight.scales, parts
                        )
                    except (ImportError, RuntimeError, ValueError):
                        pass
            if rows >= 3:
                if getattr(self, "_tc_batch_enabled", False) and rows <= 16:
                    try:
                        from mini_llm_kernels.kernels.int4_matmul import (
                            int4_dequant_matmul_tc,
                        )
                        return int4_dequant_matmul_tc(
                            x, weight.packed, weight.scales
                        )
                    except (ImportError, ValueError, RuntimeError):
                        pass
                if rows <= 4:
                    try:
                        from mini_llm_kernels.kernels.int4_matmul import (
                            int4_dequant_matmul_pair_batch,
                        )
                        return int4_dequant_matmul_pair_batch(
                            x, weight.packed, weight.scales
                        )
                    except (ImportError, ValueError):
                        pass
                # A batched decode must not fall into the prefill dense path:
                # that would dequantize the full weight matrix every token.
                return x @ weight.t()
        if (
            rows >= self._prefill_gemm_min_rows
            and hasattr(weight, "packed")
        ):
            try:
                from mini_llm_kernels.kernels.int4_matmul import (
                    int4_dequant_dense,
                )
                dense = int4_dequant_dense(
                    weight.packed, weight.scales, weight.compute_dtype
                )
            except (ImportError, AttributeError):
                dense = weight.dequant_dense()
            try:
                return torch.nn.functional.linear(x, dense)
            finally:
                del dense
        return x @ weight.t()

    def _compact_fused_weights(self) -> None:
        """Replace q/k/v and gate/up weights with views into the fused copies.

        QKV and gate/up are concatenated once per layer for the fused INT4
        GEMV. Keeping the original tensors alive in addition to the combined
        copies duplicates about 0.6 GiB of packed weights on E2B. Re-pointing
        the per-projection attributes at contiguous slices of the combined
        tensor keeps every eager/fallback call site working while releasing
        the duplicate storage.
        """
        from miniservellm.runtime.gemma4_quant import Int4Weight

        for lw in self.w.layers:
            gate, up = lw.gate_proj, lw.up_proj
            if hasattr(gate, "packed") and hasattr(up, "packed"):
                fused = self._fused_gate_up_for(lw)
                if fused is not None:
                    combined, n_gate = fused
                    lw.gate_proj = Int4Weight(
                        combined.packed[: n_gate // 2],
                        combined.scales[:n_gate],
                        combined.group_size,
                        combined.compute_dtype,
                    )
                    lw.up_proj = Int4Weight(
                        combined.packed[n_gate // 2 :],
                        combined.scales[n_gate:],
                        combined.group_size,
                        combined.compute_dtype,
                    )

            if self.kv_source.get(lw.layer_spec.layer_idx) is not None:
                continue
            q, k, v = lw.q_proj, lw.k_proj, lw.v_proj
            if not all(hasattr(w, "packed") for w in (q, k, v)):
                continue
            fused = self._fused_qkv_for(lw)
            if fused is None:
                continue
            combined, n_q, n_k, n_v = fused
            lw.q_proj = Int4Weight(
                combined.packed[: n_q // 2],
                combined.scales[:n_q],
                combined.group_size,
                combined.compute_dtype,
            )
            lw.k_proj = Int4Weight(
                combined.packed[n_q // 2 : (n_q + n_k) // 2],
                combined.scales[n_q : n_q + n_k],
                combined.group_size,
                combined.compute_dtype,
            )
            lw.v_proj = Int4Weight(
                combined.packed[(n_q + n_k) // 2 :],
                combined.scales[n_q + n_k :],
                combined.group_size,
                combined.compute_dtype,
            )

    def _fused_gate_up_for(self, lw):
        layer_idx = lw.layer_spec.layer_idx
        if layer_idx in self._gate_up_fused:
            return self._gate_up_fused[layer_idx]
        gate, up = lw.gate_proj, lw.up_proj
        if not (hasattr(gate, "packed") and hasattr(up, "packed")):
            self._gate_up_fused[layer_idx] = None
            return None
        if gate.packed.shape != up.packed.shape or gate.group_size != up.group_size:
            self._gate_up_fused[layer_idx] = None
            return None
        from miniservellm.runtime.gemma4_quant import Int4Weight

        packed = torch.cat([gate.packed, up.packed], dim=0)
        scales = torch.cat([gate.scales, up.scales], dim=0)
        combined = Int4Weight(packed, scales, gate.group_size, gate.compute_dtype)
        result = (combined, gate.packed.shape[0] * 2)
        self._gate_up_fused[layer_idx] = result
        return result

    def _mlp(self, h, layer_idx):
        lw = self.w.layers[layer_idx]
        fused = self._fused_gate_up_for(lw)
        if fused is None:
            return super()._mlp(h, layer_idx)
        combined, n_gate = fused
        projected = self._linear(combined, h)
        return self._linear(
            lw.down_proj,
            _gelu_mul(projected[:, :n_gate], projected[:, n_gate:]),
        )

    def _fused_qkv_for(self, lw):
        """Return (combined_weight, n_q, n_k, n_v) or None when unavailable."""
        layer_idx = lw.layer_spec.layer_idx
        if layer_idx in self._qkv_fused:
            return self._qkv_fused[layer_idx]
        q, k, v = lw.q_proj, lw.k_proj, lw.v_proj
        if not all(hasattr(w, "packed") for w in (q, k, v)):
            self._qkv_fused[layer_idx] = None
            return None
        if not (q.group_size == k.group_size == v.group_size):
            self._qkv_fused[layer_idx] = None
            return None
        from miniservellm.runtime.gemma4_quant import Int4Weight

        packed = torch.cat([q.packed, k.packed, v.packed], dim=0)
        scales = torch.cat([q.scales, k.scales, v.scales], dim=0)
        combined = Int4Weight(packed, scales, q.group_size, q.compute_dtype)
        result = (
            combined,
            q.packed.shape[0] * 2,
            k.packed.shape[0] * 2,
            v.packed.shape[0] * 2,
        )
        self._qkv_fused[layer_idx] = result
        return result

    def _dense_for(self, request_id: str) -> dict:
        return self._dense.setdefault(request_id, {})

    @staticmethod
    def _next_capacity(required: int, current: int = 0) -> int:
        capacity = max(128, current)
        while capacity < required:
            capacity *= 2
        return capacity

    def _dense_append(
        self,
        dense: dict[int, _DenseKVEntry],
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append K/V with geometric growth and return the populated prefix."""
        if k.shape != v.shape:
            raise ValueError(
                f"dense K/V shape mismatch: {tuple(k.shape)} vs {tuple(v.shape)}"
            )
        if k.shape[0] == 0:
            if layer_idx in dense:
                entry = dense[layer_idx]
                return entry.k[: entry.length], entry.v[: entry.length]
            return k, v
        entry = dense.get(layer_idx)
        required = (entry.length if entry is not None else 0) + k.shape[0]
        if entry is None:
            capacity = self._next_capacity(required)
            k_buffer = torch.empty(
                (capacity, *k.shape[1:]), device=k.device, dtype=k.dtype
            )
            v_buffer = torch.empty(
                (capacity, *v.shape[1:]), device=v.device, dtype=v.dtype
            )
            length = 0
        elif required > entry.k.shape[0]:
            capacity = self._next_capacity(required, entry.k.shape[0])
            k_buffer = torch.empty(
                (capacity, *entry.k.shape[1:]),
                device=entry.k.device,
                dtype=entry.k.dtype,
            )
            v_buffer = torch.empty(
                (capacity, *entry.v.shape[1:]),
                device=entry.v.device,
                dtype=entry.v.dtype,
            )
            k_buffer[: entry.length].copy_(entry.k[: entry.length])
            v_buffer[: entry.length].copy_(entry.v[: entry.length])
            length = entry.length
        else:
            k_buffer = entry.k
            v_buffer = entry.v
            length = entry.length

        k_buffer[length:required].copy_(k)
        v_buffer[length:required].copy_(v)
        dense[layer_idx] = _DenseKVEntry(k=k_buffer, v=v_buffer, length=required)
        return k_buffer[:required], v_buffer[:required]

    @staticmethod
    def _dense_get(
        dense: dict[int, _DenseKVEntry], layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        entry = dense[layer_idx]
        return entry.k[: entry.length], entry.v[: entry.length]

    def drop_dense(self, request_id: str) -> None:
        self._dense.pop(request_id, None)

    def _gc_dense(self) -> None:
        """Drop mirrors of requests no longer tracked by the cache manager."""
        live = self.kv_cache_manager.req_block_tables
        for rid in [rid for rid in self._dense if rid not in live]:
            del self._dense[rid]

    # ------------------------------------------------------------------
    # protocol surface
    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self.w.embed_tokens.device

    @property
    def dtype(self) -> torch.dtype:
        return self.w.embed_tokens.dtype

    def forward_fresh_prefill(
        self, requests: List[Request], metas: List[PrefillRequestMetadata]
    ) -> PrefillModelOutput:
        return self._run_prefill(requests, metas)

    def forward_incremental_prefill(
        self, requests: List[Request], metas: List[PrefillRequestMetadata]
    ) -> PrefillModelOutput:
        return self._run_prefill(requests, metas)

    def forward_decode(
        self, requests: List[Request], metas: List[DecodeRequestMetadata]
    ) -> DecodeModelOutput:
        logits_by_request: Dict[str, torch.Tensor] = {}
        if len(requests) == 1:
            logits_by_request[requests[0].request_id] = self._decode_one(
                requests[0], metas[0]
            )
            return DecodeModelOutput(logits_by_request=logits_by_request)

        if not self._batch_decode_enabled:
            for req, meta in zip(requests, metas):
                logits_by_request[req.request_id] = self._decode_one(req, meta)
            return DecodeModelOutput(logits_by_request=logits_by_request)

        if self._batch_graph_mode and len(requests) <= 16:
            try:
                logits = self._decode_batch_graphed(requests, metas)
            except Exception as exc:  # noqa: BLE001
                print(f"[gemma4 graph] batch graph disabled: {exc}")
                self._batch_graph_mode = False
                self._batch_graphs.clear()
                self._batch_graph_states.clear()
                logits = self._decode_batch_dense(requests, metas)
        else:
            logits = self._decode_batch_dense(requests, metas)
        for i, req in enumerate(requests):
            logits_by_request[req.request_id] = logits[i]
        return DecodeModelOutput(
            logits_by_request=logits_by_request, batch_logits=logits
        )

    # ------------------------------------------------------------------
    # prefill
    # ------------------------------------------------------------------
    def _run_prefill(
        self, requests: List[Request], metas: List[PrefillRequestMetadata]
    ) -> PrefillModelOutput:
        self._gc_dense()
        logits_by_request: Dict[str, torch.Tensor] = {}
        previous_prefill = self._in_prefill
        self._in_prefill = True
        try:
            active = [
                (req, meta) for req, meta in zip(requests, metas)
                if meta.chunk_token_ids
            ]
            if len(active) >= 2:
                logits_by_request.update(self._prefill_batch(active))
            else:
                for req, meta in active:
                    logits = self._prefill_chunk(req, meta)
                    if (
                        meta.chunk_end == req.total_prompt_tokens()
                        and logits is not None
                    ):
                        logits_by_request[req.request_id] = logits
        finally:
            self._in_prefill = previous_prefill
        return PrefillModelOutput(logits_by_request=logits_by_request)

    def _flash_attn(self, q, k, v, start, total, sliding):
        if not self._prefill_flash or total < self._prefill_flash_min_tokens:
            raise RuntimeError("flash disabled")
        from miniservellm.runtime.flash_prefill import flash_prefill_attention

        window = self.window if sliding else total + 1
        if not window:
            window = total + 1
        return flash_prefill_attention(q, k, v, start, total, window)

    def _prefix_hit(self, req, meta):
        if (
            not self._paged_attention
            or not self._prefix_cache_enabled
            or meta.context_len_before_chunk != 0
            or len(meta.chunk_token_ids) != len(req.prompt_token_ids)
            or len(req.prompt_token_ids) > self._prefix_cache_max_tokens
        ):
            return None
        entry = self._prefix_cache_get(tuple(req.prompt_token_ids))
        if entry is None:
            return None
        kvs, logits = entry
        for layer_idx, k, v in kvs:
            self.kv_cache_manager.write_kv_for_tokens(
                layer_idx, meta.write_slots, k, v
            )
        return logits.clone()

    @torch.inference_mode()
    def _prefill_batch(self, active) -> Dict[str, torch.Tensor]:
        hits = {}
        misses = []
        for req, meta in active:
            logits = self._prefix_hit(req, meta)
            if logits is None:
                misses.append((req, meta))
            else:
                hits[req.request_id] = logits
        if not misses:
            return hits
        result = self._prefill_batch_inner(misses)
        result.update(hits)
        return result

    @torch.inference_mode()
    def _prefill_batch_inner(self, active) -> Dict[str, torch.Tensor]:
        """Batch independent fresh/incremental prefills through one forward.

        Linear/MLP/PLE work is concatenated across requests; attention and KV
        writes stay per request so requests never attend to each other.
        """
        w = self.w
        reqs = [req for req, _ in active]
        metas = [meta for _, meta in active]
        lengths = [len(meta.chunk_token_ids) for meta in metas]
        starts = [meta.context_len_before_chunk for meta in metas]
        totals = [start + length for start, length in zip(starts, lengths)]
        offsets = []
        cursor = 0
        for length in lengths:
            offsets.append(cursor)
            cursor += length

        input_ids = torch.cat([
            torch.tensor(meta.chunk_token_ids, dtype=torch.long, device=self.device)
            for meta in metas
        ])
        x = w.embed_tokens[input_ids] * float(self.config.hidden_size) ** 0.5
        per_layer_inputs = (
            self._ple(input_ids, x)
            if self.config.hidden_size_per_layer_input else None
        )

        ropes = []
        for start, total in zip(starts, totals):
            positions = torch.arange(
                start, total, device=self.device, dtype=torch.float32
            )
            ropes.append(self._rope_for(positions))

        dense = []
        for req, meta in zip(reqs, metas):
            if meta.context_len_before_chunk == 0:
                self._dense.pop(req.request_id, None)
            dense.append(self._dense_for(req.request_id))

        for lw in w.layers:
            spec = lw.layer_spec
            head_dim = spec.head_dim
            num_q = spec.num_attention_heads
            num_kv = spec.num_key_value_heads
            source = self.kv_source.get(spec.layer_idx)
            fused = self._fused_qkv_for(lw) if source is None else None

            residual = x
            h = _rms_norm(x, lw.input_layernorm, self.eps)
            q_parts = []
            kv_parts = []

            if fused is not None:
                combined, n_q, n_k, n_v = fused
                qkv = self._linear(combined, h)
                for i, (off, length) in enumerate(zip(offsets, lengths)):
                    cos, sin = ropes[i][spec.attention_type]
                    q = qkv[off:off + length, :n_q].reshape(
                        length, num_q, head_dim
                    )
                    k = qkv[off:off + length, n_q:n_q + n_k].reshape(
                        length, num_kv, head_dim
                    )
                    v = qkv[
                        off:off + length, n_q + n_k:n_q + n_k + n_v
                    ].reshape(length, num_kv, head_dim)
                    q = self._apply_rope(
                        _rms_norm(q, lw.q_norm, self.eps), cos, sin
                    )
                    k = self._apply_rope(
                        _rms_norm(k, lw.k_norm, self.eps), cos, sin
                    )
                    v = _rms_norm(v, lw.v_norm, self.eps)
                    if self._paged_attention:
                        self.kv_cache_manager.write_kv_for_tokens(
                            spec.layer_idx, metas[i].write_slots, k, v
                        )
                    k, v = self._dense_append(dense[i], spec.layer_idx, k, v)
                    q_parts.append(q)
                    kv_parts.append((k, v))
            else:
                q_all = self._linear(lw.q_proj, h)
                if source is None:
                    k_all = self._linear(lw.k_proj, h)
                    v_all = self._linear(lw.v_proj, h)
                for i, (off, length) in enumerate(zip(offsets, lengths)):
                    cos, sin = ropes[i][spec.attention_type]
                    q = q_all[off:off + length].reshape(
                        length, num_q, head_dim
                    )
                    q = _rms_norm(q, lw.q_norm, self.eps)
                    q = self._apply_rope(q, cos, sin)
                    if source is not None:
                        k, v = self._dense_get(dense[i], source)
                    else:
                        k = k_all[off:off + length].reshape(
                            length, num_kv, head_dim
                        )
                        k = _rms_norm(k, lw.k_norm, self.eps)
                        k = self._apply_rope(k, cos, sin)
                        v = v_all[off:off + length].reshape(
                            length, num_kv, head_dim
                        )
                        v = _rms_norm(v, lw.v_norm, self.eps)
                        if self._paged_attention:
                            self.kv_cache_manager.write_kv_for_tokens(
                                spec.layer_idx, metas[i].write_slots, k, v
                            )
                        k, v = self._dense_append(
                            dense[i], spec.layer_idx, k, v
                        )
                    q_parts.append(q)
                    kv_parts.append((k, v))

            attn_parts = []
            q_step = max(1, self._prefill_attn_chunk_size)
            for i, (q, (k, v)) in enumerate(zip(q_parts, kv_parts)):
                q_len = lengths[i]
                rows = None
                if self._prefill_flash:
                    try:
                        rows = self._flash_attn(
                            q, k, v, starts[i], totals[i],
                            spec.attention_type == SLIDING,
                        ).reshape(q_len, num_q * head_dim)
                    except Exception:
                        rows = None
                if rows is None:
                    rows = torch.empty(
                        (q_len, num_q * head_dim), device=q.device, dtype=v.dtype
                    )
                    sliding = (
                        spec.attention_type == SLIDING and self.window
                        and self._prefill_window_slice
                    )
                    for q_start in range(0, q_len, q_step):
                        q_end = min(q_start + q_step, q_len)
                        kv_start = 0
                        if sliding:
                            kv_start = max(
                                0, starts[i] + q_start - self.window + 1
                            )
                        kv_end = totals[i]
                        k_use = k[kv_start:kv_end]
                        v_use = v[kv_start:kv_end]
                        k_t = k_use.transpose(0, 1).transpose(-1, -2)
                        v_t = v_use.transpose(0, 1)
                        scores = torch.matmul(
                            q[q_start:q_end].transpose(0, 1), k_t
                        ).float()
                        mask = self._prefill_mask(
                            spec.attention_type,
                            starts[i] + q_start,
                            totals[i],
                            q_end - q_start,
                        )
                        scores.add_(mask[:, kv_start:kv_end])
                        probs = torch.softmax(scores, dim=-1).to(v.dtype)
                        part = torch.matmul(probs, v_t)
                        rows[q_start:q_end] = part.transpose(0, 1).reshape(
                            q_end - q_start, num_q * head_dim
                        )
                attn_parts.append(rows)
            attn_rows = torch.cat(attn_parts, dim=0) if len(attn_parts) > 1 else attn_parts[0]
            h = self._linear(lw.o_proj, attn_rows)

            h = _rms_norm(h, lw.post_attention_layernorm, self.eps)
            h, x = fused_add_rms_norm(
                h, residual, lw.pre_feedforward_layernorm, self.eps
            )
            residual = x
            h = self._mlp(h, spec.layer_idx)
            h = _rms_norm(h, lw.post_feedforward_layernorm, self.eps)
            if per_layer_inputs is not None:
                x = residual.add_(h)
                residual = x
                h = _gelu_mul(
                    self._linear(lw.per_layer_input_gate, x), per_layer_inputs[:, spec.layer_idx, :]
                )
                h = self._linear(lw.per_layer_projection, h)
                h = _rms_norm(h, lw.post_per_layer_input_norm, self.eps)
                x = _add_mul(residual, h, lw.layer_scalar)
            else:
                x = _add_mul(residual, h, lw.layer_scalar)

        x = _rms_norm(x, w.final_norm, self.eps)
        last_hidden = torch.stack([
            x[off + length - 1] for off, length in zip(offsets, lengths)
        ], dim=0)
        logits = last_hidden @ w.lm_head.t()
        if self.softcap is not None:
            logits = self.softcap * torch.tanh(logits.float() / self.softcap)
        logits = logits.float()

        result: Dict[str, torch.Tensor] = {}
        for i, (req, meta) in enumerate(zip(reqs, metas)):
            if meta.chunk_end != req.total_prompt_tokens():
                continue
            result[req.request_id] = logits[i]
            if (
                self._paged_attention
                and self._prefix_cache_enabled
                and starts[i] == 0
                and lengths[i] == len(req.prompt_token_ids)
                and len(req.prompt_token_ids) <= self._prefix_cache_max_tokens
            ):
                dense_i = self._dense.get(req.request_id)
                if dense_i:
                    kvs = [
                        (layer_idx, e.k[:e.length].clone(),
                         e.v[:e.length].clone())
                        for layer_idx, e in dense_i.items()
                    ]
                    if kvs:
                        self._prefix_cache_put(
                            tuple(req.prompt_token_ids), kvs,
                            logits[i].detach().clone(),
                        )
        return result

    def _prefix_cache_get(self, key):
        if not self._prefix_cache_enabled:
            return None
        entry = self._prefix_cache.pop(key, None)
        if entry is None:
            return None
        self._prefix_cache[key] = entry
        return entry

    def _prefix_cache_put(self, key, kvs, logits):
        if not self._prefix_cache_enabled:
            return
        size = sum(k.numel() * k.element_size() + v.numel() * v.element_size()
                   for _, k, v in kvs) + logits.numel() * logits.element_size()
        if size > self._prefix_cache_max_bytes:
            return
        old = self._prefix_cache.pop(key, None)
        if old is not None:
            self._prefix_cache_bytes -= sum(
                k.numel() * k.element_size() + v.numel() * v.element_size()
                for _, k, v in old[0]
            ) + old[1].numel() * old[1].element_size()
        self._prefix_cache[key] = (kvs, logits)
        self._prefix_cache_bytes += size
        while (
            len(self._prefix_cache) > self._prefix_cache_entries
            or self._prefix_cache_bytes > self._prefix_cache_max_bytes
        ):
            _, (old_kvs, old_logits) = self._prefix_cache.popitem(last=False)
            self._prefix_cache_bytes -= sum(
                k.numel() * k.element_size() + v.numel() * v.element_size()
                for _, k, v in old_kvs
            ) + old_logits.numel() * old_logits.element_size()

    def _prefill_chunk(
        self, req: Request, meta: PrefillRequestMetadata
    ) -> Optional[torch.Tensor]:
        w = self.w
        chunk_ids = meta.chunk_token_ids
        start = meta.context_len_before_chunk
        total = start + len(chunk_ids)
        full_prompt = (
            start == 0 and len(chunk_ids) == len(req.prompt_token_ids)
        )
        cache_key = None
        if (
            full_prompt
            and self._paged_attention
            and self._prefix_cache_enabled
            and len(req.prompt_token_ids) <= self._prefix_cache_max_tokens
            and tuple(req.prompt_token_ids) not in self._prefix_cache
        ):
            cache_key = tuple(req.prompt_token_ids)
        elif (
            full_prompt
            and self._paged_attention
            and self._prefix_cache_enabled
            and len(req.prompt_token_ids) <= self._prefix_cache_max_tokens
        ):
            cache_key = tuple(req.prompt_token_ids)
            hit = self._prefix_cache_get(cache_key)
            if hit is not None:
                kvs, logits = hit
                for layer_idx, k, v in kvs:
                    self.kv_cache_manager.write_kv_for_tokens(
                        layer_idx, meta.write_slots, k, v
                    )
                return logits.clone()
        input_ids = torch.tensor(chunk_ids, dtype=torch.long, device=self.device)
        if start == 0:
            # Fresh prefill: reset any stale mirror for this request.
            self._dense.pop(req.request_id, None)
        dense = self._dense_for(req.request_id)

        x = w.embed_tokens[input_ids] * float(self.config.hidden_size) ** 0.5
        per_layer_inputs = self._ple(input_ids, x) if self.config.hidden_size_per_layer_input else None

        # Absolute positions for RoPE: [start, start+T)
        positions = torch.arange(start, total, device=self.device, dtype=torch.float32)
        rope = self._rope_for(positions)

        for lw in w.layers:
            spec = lw.layer_spec
            residual = x
            h = _rms_norm(x, lw.input_layernorm, self.eps)

            num_q = spec.num_attention_heads
            head_dim = spec.head_dim
            num_kv = spec.num_key_value_heads
            source = self.kv_source.get(spec.layer_idx)
            fused = self._fused_qkv_for(lw) if source is None else None
            cos, sin = rope[spec.attention_type]

            if fused is not None:
                combined, n_q, n_k, n_v = fused
                qkv = self._linear(combined, h)
                q = qkv[:, :n_q].view(len(chunk_ids), num_q, head_dim)
                k = qkv[:, n_q:n_q + n_k].view(len(chunk_ids), num_kv, head_dim)
                v = qkv[:, n_q + n_k:].view(len(chunk_ids), num_kv, head_dim)
                q = self._apply_rope(
                    _rms_norm(q, lw.q_norm, self.eps), cos, sin
                )
                k = self._apply_rope(
                    _rms_norm(k, lw.k_norm, self.eps), cos, sin
                )
                v = _rms_norm(v, lw.v_norm, self.eps)
                if self._paged_attention:
                    self.kv_cache_manager.write_kv_for_tokens(
                        spec.layer_idx, meta.write_slots, k, v
                    )
                # Prefill attention reads the dense mirror for every layer,
                # including paged mode: chunked prefill must attend the
                # history accumulated by earlier chunks, and shared layers
                # read their source layer's mirror.
                k, v = self._dense_append(dense, spec.layer_idx, k, v)
            else:
                q = (h @ lw.q_proj.t()).view(len(chunk_ids), num_q, head_dim)
                q = _rms_norm(q, lw.q_norm, self.eps)
                q = self._apply_rope(q, cos, sin)

                if source is not None:
                    # Shared layer: K/V come from the source layer's mirror. The
                    # source layer (< this index) already stored the current chunk.
                    k, v = self._dense_get(dense, source)
                else:
                    k = (h @ lw.k_proj.t()).view(len(chunk_ids), num_kv, head_dim)
                    k = _rms_norm(k, lw.k_norm, self.eps)
                    k = self._apply_rope(k, cos, sin)
                    v = (h @ lw.v_proj.t()).view(len(chunk_ids), num_kv, head_dim)
                    v = _rms_norm(v, lw.v_norm, self.eps)
                    if self._paged_attention:
                        self.kv_cache_manager.write_kv_for_tokens(
                            spec.layer_idx, meta.write_slots, k, v
                        )
                    k, v = self._dense_append(dense, spec.layer_idx, k, v)
            # NOTE: no window slicing here — prefill attention uses the full
            # gathered K/V with the sliding mask built over absolute
            # positions (_prefill_mask). Slicing would misalign K with the
            # mask when the chunk spans multiple windows.

            # Query-chunked prefill attention keeps the [H, T, total]
            # score matrix bounded. A 4k prompt otherwise allocates a
            # multi-hundred-MB FP32 score tensor per layer.
            q_len = len(chunk_ids)
            q_step = max(1, self._prefill_attn_chunk_size)
            attn_rows = None
            if self._prefill_flash:
                try:
                    attn_rows = self._flash_attn(
                        q, k, v, start, total,
                        spec.attention_type == SLIDING,
                    ).reshape(q_len, num_q * head_dim)
                except Exception:
                    attn_rows = None
            if attn_rows is None:
                attn_rows = torch.empty(
                    (q_len, num_q * head_dim), device=q.device, dtype=v.dtype
                )
                sliding = (
                    spec.attention_type == SLIDING and self.window
                    and self._prefill_window_slice
                )
                for q_start in range(0, q_len, q_step):
                    q_end = min(q_start + q_step, q_len)
                    kv_start = 0
                    if sliding:
                        kv_start = max(0, start + q_start - self.window + 1)
                    kv_end = total
                    k_use = k[kv_start:kv_end]
                    v_use = v[kv_start:kv_end]
                    k_t = k_use.transpose(0, 1).transpose(-1, -2)
                    v_t = v_use.transpose(0, 1)
                    scores = torch.matmul(
                        q[q_start:q_end].transpose(0, 1), k_t
                    ).float()
                    mask = self._prefill_mask(
                        spec.attention_type, start + q_start, total,
                        q_end - q_start,
                    )
                    scores.add_(mask[:, kv_start:kv_end])
                    probs = torch.softmax(scores, dim=-1).to(v.dtype)
                    part = torch.matmul(probs, v_t)
                    attn_rows[q_start:q_end] = part.transpose(0, 1).reshape(
                        q_end - q_start, num_q * head_dim
                    )
            h = self._linear(lw.o_proj, attn_rows)

            h = _rms_norm(h, lw.post_attention_layernorm, self.eps)
            # Fuse the residual add with the next pre-FFN RMSNorm: one kernel
            # instead of a separate add + norm launch on every layer.
            h, x = fused_add_rms_norm(
                h, residual, lw.pre_feedforward_layernorm, self.eps
            )
            residual = x
            h = self._mlp(h, spec.layer_idx)
            h = _rms_norm(h, lw.post_feedforward_layernorm, self.eps)
            # residual is no longer needed after this add; mutate it in
            # place to avoid one allocation per layer in eager/dense decode.
            if per_layer_inputs is not None:
                x = residual.add_(h)
                residual = x
                h = _gelu_mul(
                    self._linear(lw.per_layer_input_gate, x), per_layer_inputs[:, spec.layer_idx, :]
                )
                h = self._linear(lw.per_layer_projection, h)
                h = _rms_norm(h, lw.post_per_layer_input_norm, self.eps)
                x = _add_mul(residual, h, lw.layer_scalar)
            else:
                x = _add_mul(residual, h, lw.layer_scalar)

        x = _rms_norm(x, w.final_norm, self.eps)
        last_hidden = x[-1:]
        logits = last_hidden @ w.lm_head.t()
        if self.softcap is not None:
            logits = self.softcap * torch.tanh(logits.float() / self.softcap)
        logits = logits.float()[0]
        if full_prompt and self._paged_attention and cache_key is not None:
            dense = self._dense.get(req.request_id)
            if dense:
                kvs = [
                    (layer_idx, e.k[:e.length].clone(),
                     e.v[:e.length].clone())
                    for layer_idx, e in dense.items()
                ]
                if kvs:
                    self._prefix_cache_put(
                        cache_key, kvs, logits.detach().clone()
                    )
        return logits

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------
    def _decode_one(self, req: Request, meta: DecodeRequestMetadata) -> torch.Tensor:
        if self._graph_mode:
            long_context = meta.context_len > self._splitk_threshold
            try:
                return self._decode_one_graphed(req, meta, long_context)
            except Exception as exc:  # noqa: BLE001
                print(f"[gemma4 graph] disabled after failure: {exc}")
                self._graph_mode = False
                self._decode_graph = None
                self._graph_static = None
                self._graphs.clear()
                self._graph_states.clear()
                # Keep the paged path: it already contains the prompt and all
                # previous decode steps, so falling back to dense here would
                # silently lose the decode history.
                try:
                    return self._decode_one_dense(req, meta)
                except Exception as paged_exc:  # noqa: BLE001
                    print(f"[gemma4 paged] disabled after failure: {paged_exc}")
                    self._paged_attention = False
        return self._decode_one_dense(req, meta)

    def _decode_one_dense(self, req: Request, meta: DecodeRequestMetadata) -> torch.Tensor:
        w = self.w
        dense = self._dense_for(req.request_id)
        input_ids = torch.tensor([meta.input_token_id], dtype=torch.long, device=self.device)
        position = meta.query_position
        ctx_len = meta.context_len

        x = w.embed_tokens[input_ids] * float(self.config.hidden_size) ** 0.5
        per_layer_inputs = None
        if self.config.hidden_size_per_layer_input:
            ple_dim = self.config.hidden_size_per_layer_input
            identity = w.embed_tokens_per_layer[input_ids] * float(ple_dim) ** 0.5
            identity = identity.view(1, self.config.num_hidden_layers, ple_dim)
            ctx = (x @ w.per_layer_model_projection.t()) * float(self.config.hidden_size) ** -0.5
            ctx = ctx.view(1, self.config.num_hidden_layers, ple_dim)
            ctx = _rms_norm(ctx, w.per_layer_projection_norm, self.eps)
            per_layer_inputs = (ctx + identity) * (2.0 ** -0.5)

        positions = torch.tensor([float(position)], device=self.device, dtype=torch.float32)
        rope = self._rope_for(positions)
        paged_block_table = (
            self.kv_cache_manager.block_table_tensor(req)
            if self._paged_attention else None
        )
        paged_context_lens = (
            torch.tensor([ctx_len], device=self.device, dtype=torch.int32)
            if self._paged_attention else None
        )
        for lw in w.layers:
            spec = lw.layer_spec
            head_dim = spec.head_dim
            num_q = spec.num_attention_heads

            residual = x
            h = _rms_norm(x, lw.input_layernorm, self.eps)
            num_kv = spec.num_key_value_heads
            source = self.kv_source.get(spec.layer_idx)
            fused = self._fused_qkv_for(lw) if source is None else None
            cos, sin = rope[spec.attention_type]

            if fused is not None:
                combined, n_q, n_k, n_v = fused
                qkv = self._linear(combined, h)
                q = qkv[:, :n_q].view(1, num_q, head_dim)
                k = qkv[:, n_q:n_q + n_k].view(1, num_kv, head_dim)
                v = qkv[:, n_q + n_k:].view(1, num_kv, head_dim)
                q = self._apply_rope(
                    _rms_norm(q, lw.q_norm, self.eps), cos, sin
                )
                k = self._apply_rope(
                    _rms_norm(k, lw.k_norm, self.eps), cos, sin
                )
                v = _rms_norm(v, lw.v_norm, self.eps)
                if self._paged_attention:
                    self.kv_cache_manager.write_kv_for_tokens(
                        spec.layer_idx, [meta.write_slot], k, v
                    )
                else:
                    k, v = self._dense_append(dense, spec.layer_idx, k, v)
            else:
                q = (h @ lw.q_proj.t()).view(1, num_q, head_dim)
                q = _rms_norm(q, lw.q_norm, self.eps)
                q = self._apply_rope(q, cos, sin)

                if source is not None:
                    if not self._paged_attention:
                        k, v = self._dense_get(dense, source)
                else:
                    k = (h @ lw.k_proj.t()).view(1, num_kv, head_dim)
                    k = _rms_norm(k, lw.k_norm, self.eps)
                    k = self._apply_rope(k, cos, sin)
                    v = (h @ lw.v_proj.t()).view(1, num_kv, head_dim)
                    v = _rms_norm(v, lw.v_norm, self.eps)
                    if self._paged_attention:
                        self.kv_cache_manager.write_kv_for_tokens(
                            spec.layer_idx, [meta.write_slot], k, v
                        )
                    else:
                        k, v = self._dense_append(dense, spec.layer_idx, k, v)
            if self._paged_attention:
                cache_layer = source if source is not None else spec.layer_idx
                k_cache, v_cache = self.kv_cache_manager.layer_cache(cache_layer)
                if (
                    meta.context_len > self._splitk_threshold
                    and spec.attention_type != SLIDING
                ):
                    scratch = self._splitk_scratch_for(1, num_q, head_dim)
                    attn = _splitk_attention_kernel(
                        q, k_cache, v_cache,
                        paged_block_table.view(1, -1), paged_context_lens,
                        self.window if spec.attention_type == SLIDING else 0,
                        *scratch, self._splitk_chunk_size,
                    )
                else:
                    attn = _paged_attention_kernel(
                        q,
                        k_cache,
                        v_cache,
                        paged_block_table.view(1, -1),
                        paged_context_lens,
                        self.window if spec.attention_type == SLIDING else 0,
                    )
            else:
                k, v = self._window_slice(spec, k, v)
                scores = torch.matmul(
                    q.transpose(0, 1), k.transpose(0, 1).transpose(-1, -2)
                )
                probs = torch.softmax(scores.float(), dim=-1).to(v.dtype)
                attn = torch.matmul(probs, v.transpose(0, 1))
            h = attn.transpose(0, 1).reshape(1, num_q * head_dim) @ lw.o_proj.t()

            h = _rms_norm(h, lw.post_attention_layernorm, self.eps)
            # Fuse the residual add with the next pre-FFN RMSNorm: one kernel
            # instead of a separate add + norm launch on every layer.
            h, x = fused_add_rms_norm(
                h, residual, lw.pre_feedforward_layernorm, self.eps
            )
            residual = x
            h = self._mlp(h, spec.layer_idx)
            h = _rms_norm(h, lw.post_feedforward_layernorm, self.eps)
            # residual is no longer needed after this add; mutate it in
            # place to avoid one allocation per layer in eager/dense decode.
            if per_layer_inputs is not None:
                x = residual.add_(h)
                residual = x
                h = _gelu_mul(
                    self._linear(lw.per_layer_input_gate, x), per_layer_inputs[:, spec.layer_idx, :]
                )
                h = self._linear(lw.per_layer_projection, h)
                h = _rms_norm(h, lw.post_per_layer_input_norm, self.eps)
                x = _add_mul(residual, h, lw.layer_scalar)
            else:
                x = _add_mul(residual, h, lw.layer_scalar)

        x = _rms_norm(x, w.final_norm, self.eps)
        logits = x @ w.lm_head.t()
        if self.softcap is not None:
            logits = self.softcap * torch.tanh(logits.float() / self.softcap)
        return logits.float()[0]

    # ------------------------------------------------------------------
    # shared helpers
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # graph-capturable paged decode
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _decode_batch_graphable(
        self,
        token: torch.Tensor,
        position: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        write_block: torch.Tensor,
        write_offset: torch.Tensor,
        splitk_o=None,
        splitk_m=None,
        splitk_l=None,
        skip_softcap: bool = False,
    ) -> torch.Tensor:
        """One decode step for N requests with static paged inputs.

        This is the batch counterpart of ``_decode_one_graphable``. N is fixed
        by the static input shapes; all branches are Python-static, so the
        method is safe to capture in a CUDA Graph.
        """
        if not self._paged_attention:
            raise RuntimeError("batch graph decode requires paged attention")
        w = self.w
        n = token.shape[0]
        x = w.embed_tokens[token] * float(self.config.hidden_size) ** 0.5
        per_layer_inputs = None
        if self.config.hidden_size_per_layer_input:
            ple_dim = self.config.hidden_size_per_layer_input
            identity = w.embed_tokens_per_layer[token] * float(ple_dim) ** 0.5
            identity = identity.view(n, self.config.num_hidden_layers, ple_dim)
            ctx = self._linear(w.per_layer_model_projection, x) * float(self.config.hidden_size) ** -0.5
            ctx = ctx.view(n, self.config.num_hidden_layers, ple_dim)
            ctx = _rms_norm(ctx, w.per_layer_projection_norm, self.eps)
            per_layer_inputs = (ctx + identity) * (2.0 ** -0.5)

        rope = self._rope_for(position)
        for lw in w.layers:
            spec = lw.layer_spec
            head_dim = spec.head_dim
            num_q = spec.num_attention_heads
            num_kv = spec.num_key_value_heads
            source = self.kv_source.get(spec.layer_idx)
            fused = self._fused_qkv_for(lw) if source is None else None
            cos, sin = rope[spec.attention_type]

            residual = x
            h = _rms_norm(x, lw.input_layernorm, self.eps)
            if fused is not None:
                combined, n_q, n_k, n_v = fused
                qkv = self._linear(combined, h)
                q = qkv[:, :n_q].view(n, num_q, head_dim)
                k = qkv[:, n_q:n_q + n_k].view(n, num_kv, head_dim)
                v = qkv[:, n_q + n_k:].view(n, num_kv, head_dim)
                q, k, v = _qkv_norm_rope(
                    q, k, v, lw.q_norm, lw.k_norm, lw.v_norm,
                    cos, sin, self.eps,
                )
                layer_cache, v_cache = self.kv_cache_manager.layer_cache(
                    spec.layer_idx
                )
                layer_cache.index_put_((write_block, write_offset), k)
                v_cache.index_put_((write_block, write_offset), v)
            else:
                q = self._linear(lw.q_proj, h).view(n, num_q, head_dim)
                q = self._apply_rope(_rms_norm(q, lw.q_norm, self.eps), cos, sin)
                if source is None:
                    k = self._linear(lw.k_proj, h).view(n, num_kv, head_dim)
                    k = self._apply_rope(_rms_norm(k, lw.k_norm, self.eps), cos, sin)
                    v = self._linear(lw.v_proj, h).view(n, num_kv, head_dim)
                    v = _rms_norm(v, lw.v_norm, self.eps)
                    layer_cache, v_cache = self.kv_cache_manager.layer_cache(
                        spec.layer_idx
                    )
                    layer_cache.index_put_((write_block, write_offset), k)
                    v_cache.index_put_((write_block, write_offset), v)

            cache_layer = source if source is not None else spec.layer_idx
            k_cache, v_cache = self.kv_cache_manager.layer_cache(cache_layer)
            if splitk_o is not None and spec.attention_type != SLIDING:
                attn = _splitk_attention_kernel(
                    q, k_cache, v_cache, block_table, context_lens, 0,
                    splitk_o, splitk_m, splitk_l, self._splitk_chunk_size,
                )
            else:
                attn = _paged_attention_kernel(
                    q,
                    k_cache,
                    v_cache,
                    block_table,
                    context_lens,
                    self.window if spec.attention_type == SLIDING else 0,
                )
            h = self._linear(lw.o_proj, attn.reshape(n, num_q * head_dim))
            h = _rms_norm(h, lw.post_attention_layernorm, self.eps)
            h, x = fused_add_rms_norm(
                h, residual, lw.pre_feedforward_layernorm, self.eps
            )
            residual = x
            h = self._mlp(h, spec.layer_idx)
            h = _rms_norm(h, lw.post_feedforward_layernorm, self.eps)
            if per_layer_inputs is not None:
                x = residual.add_(h)
                residual = x
                h = _gelu_mul(
                    self._linear(lw.per_layer_input_gate, x), per_layer_inputs[:, spec.layer_idx, :]
                )
                h = self._linear(lw.per_layer_projection, h)
                h = _rms_norm(h, lw.post_per_layer_input_norm, self.eps)
                x = _add_mul(residual, h, lw.layer_scalar)
            else:
                x = _add_mul(residual, h, lw.layer_scalar)

        x = _rms_norm(x, w.final_norm, self.eps)
        logits = self._linear(w.lm_head, x)
        if skip_softcap:
            return logits
        if self.softcap is not None:
            logits = self.softcap * torch.tanh(logits.float() / self.softcap)
        return logits.float()

    def _batch_inputs(self, reqs, metas):
        n = len(reqs)
        head_dim = max(spec.head_dim for spec in self.config.layer_specs)
        n_blocks = int(self.kv_cache_manager.engine_config.num_gpu_blocks)
        token = torch.tensor(
            [m.input_token_id for m in metas], device=self.device, dtype=torch.long
        )
        position = torch.tensor(
            [float(m.query_position) for m in metas],
            device=self.device, dtype=torch.float32,
        )
        context = torch.tensor(
            [m.context_len for m in metas], device=self.device, dtype=torch.int32
        )
        block_table = torch.zeros(
            (n, n_blocks), device=self.device, dtype=torch.int32
        )
        for i, req in enumerate(reqs):
            table = self.kv_cache_manager.block_table_tensor(req)
            block_table[i, : table.numel()].copy_(table)
        write_block = torch.tensor(
            [m.write_slot.block_id for m in metas], device=self.device,
            dtype=torch.long,
        )
        write_offset = torch.tensor(
            [m.write_slot.block_offset for m in metas], device=self.device,
            dtype=torch.long,
        )
        return token, position, context, block_table, write_block, write_offset

    def _batch_splitk_scratch(self, n):
        max_heads = max(spec.num_attention_heads for spec in self.config.layer_specs)
        max_dim = max(spec.head_dim for spec in self.config.layer_specs)
        return self._splitk_scratch_for(n, max_heads, max_dim)

    def _decode_batch_dense(self, reqs, metas):
        args = self._batch_inputs(reqs, metas)
        long_context = max(m.context_len for m in metas) > self._splitk_threshold
        if long_context:
            scratch = self._batch_splitk_scratch(len(reqs))
            return self._decode_batch_graphable(*args, *scratch)
        return self._decode_batch_graphable(*args)

    def _batch_graph_tensors(self, reqs, metas, create=False, long_context=False):
        n = len(reqs)
        key = (n, bool(long_context))
        state = self._batch_graph_states.get(key)
        if create:
            n_blocks = int(self.kv_cache_manager.engine_config.num_gpu_blocks)
            state = {
                "token": torch.zeros(n, dtype=torch.long, device=self.device),
                "position": torch.zeros(n, dtype=torch.float32, device=self.device),
                "context_lens": torch.zeros(n, dtype=torch.int32, device=self.device),
                "block_table": torch.zeros((n, n_blocks), dtype=torch.int32, device=self.device),
                "write_block": torch.zeros(n, dtype=torch.long, device=self.device),
                "write_offset": torch.zeros(n, dtype=torch.long, device=self.device),
                "logits": None,
                "splitk": self._batch_splitk_scratch(n) if long_context else None,
            }
            self._batch_graph_states[key] = state
        tables = [
            self.kv_cache_manager.block_table_tensor(req) for req in reqs
        ]
        lengths = [int(table.numel()) for table in tables]
        max_len = max(lengths) if lengths else 0
        if max_len and all(length == max_len for length in lengths):
            state["block_table"][:, :max_len].copy_(
                torch.stack(tables, dim=0)
            )
        else:
            for i, table in enumerate(tables):
                state["block_table"][i, : table.numel()].copy_(table)
        state["token"].copy_(torch.tensor(
            [int(m.input_token_id) for m in metas],
            device=self.device, dtype=torch.long,
        ))
        state["position"].copy_(torch.tensor(
            [float(m.query_position) for m in metas],
            device=self.device, dtype=torch.float32,
        ))
        state["context_lens"].copy_(torch.tensor(
            [int(m.context_len) for m in metas],
            device=self.device, dtype=torch.int32,
        ))
        state["write_block"].copy_(torch.tensor(
            [int(m.write_slot.block_id) for m in metas],
            device=self.device, dtype=torch.long,
        ))
        state["write_offset"].copy_(torch.tensor(
            [int(m.write_slot.block_offset) for m in metas],
            device=self.device, dtype=torch.long,
        ))
        return state

    def _capture_batch_graph(self, reqs, metas, long_context=False):
        n = len(reqs)
        state = self._batch_graph_tensors(
            reqs, metas, create=True, long_context=long_context
        )
        args = (
            state["token"], state["position"], state["context_lens"],
            state["block_table"], state["write_block"], state["write_offset"],
        )
        if state["splitk"] is not None:
            args = args + tuple(state["splitk"])
        for _ in range(3):
            self._decode_batch_graphable(*args, skip_softcap=True)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits = self._decode_batch_graphable(*args, skip_softcap=True)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        state["logits"] = logits
        self._batch_graphs[(n, bool(long_context))] = graph
        print(
            f"[gemma4 graph] captured batch={n} "
            f"long={bool(long_context)} for batch decode"
        )
        return logits

    def _decode_batch_graphed(self, reqs, metas):
        n = len(reqs)
        long_context = max(m.context_len for m in metas) > self._splitk_threshold
        key = (n, bool(long_context))
        graph = self._batch_graphs.get(key)
        if graph is None:
            logits = self._capture_batch_graph(reqs, metas, long_context)
        else:
            state = self._batch_graph_tensors(
                reqs, metas, long_context=long_context
            )
            graph.replay()
            logits = state["logits"]
        greedy = all(
            r.sampling_params is not None
            and float(r.sampling_params.temperature) <= 0.0
            for r in reqs
        )
        if not greedy:
            if self.softcap is not None:
                logits = self.softcap * torch.tanh(
                    logits.float() / self.softcap
                )
            else:
                logits = logits.float()
        return logits

    @torch.inference_mode()
    def _decode_one_graphable(
        self,
        token: torch.Tensor,
        position: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        write_block: torch.Tensor,
        write_offset: torch.Tensor,
        splitk_o=None,
        splitk_m=None,
        splitk_l=None,
    ) -> torch.Tensor:
        """One decode step with static inputs and paged KV only.

        Every tensor is either a persistent weight or a static input supplied
        by the graph owner; the method has no Python-level value-dependent
        control flow, so it can be captured with ``torch.cuda.CUDAGraph``.
        """
        if not self._paged_attention:
            raise RuntimeError("graph decode requires the paged attention path")
        w = self.w
        x = w.embed_tokens[token] * float(self.config.hidden_size) ** 0.5
        per_layer_inputs = None
        if self.config.hidden_size_per_layer_input:
            ple_dim = self.config.hidden_size_per_layer_input
            identity = w.embed_tokens_per_layer[token] * float(ple_dim) ** 0.5
            identity = identity.view(1, self.config.num_hidden_layers, ple_dim)
            ctx = (x @ w.per_layer_model_projection.t()) * float(self.config.hidden_size) ** -0.5
            ctx = ctx.view(1, self.config.num_hidden_layers, ple_dim)
            ctx = _rms_norm(ctx, w.per_layer_projection_norm, self.eps)
            per_layer_inputs = (ctx + identity) * (2.0 ** -0.5)

        rope = self._rope_for(position)
        for lw in w.layers:
            spec = lw.layer_spec
            head_dim = spec.head_dim
            num_q = spec.num_attention_heads
            num_kv = spec.num_key_value_heads
            source = self.kv_source.get(spec.layer_idx)
            fused = self._fused_qkv_for(lw) if source is None else None
            cos, sin = rope[spec.attention_type]

            residual = x
            h = _rms_norm(x, lw.input_layernorm, self.eps)
            if fused is not None:
                combined, n_q, n_k, n_v = fused
                qkv = self._linear(combined, h)
                q = qkv[:, :n_q].view(1, num_q, head_dim)
                k = qkv[:, n_q:n_q + n_k].view(1, num_kv, head_dim)
                v = qkv[:, n_q + n_k:].view(1, num_kv, head_dim)
                q, k, v = _qkv_norm_rope(
                    q, k, v, lw.q_norm, lw.k_norm, lw.v_norm,
                    cos, sin, self.eps,
                )
                layer_cache, v_cache = self.kv_cache_manager.layer_cache(
                    spec.layer_idx
                )
                layer_cache.index_put_((write_block, write_offset), k[0])
                v_cache.index_put_((write_block, write_offset), v[0])
            else:
                q = (h @ lw.q_proj.t()).view(1, num_q, head_dim)
                q = self._apply_rope(_rms_norm(q, lw.q_norm, self.eps), cos, sin)
                if source is None:
                    k = (h @ lw.k_proj.t()).view(1, num_kv, head_dim)
                    k = self._apply_rope(_rms_norm(k, lw.k_norm, self.eps), cos, sin)
                    v = (h @ lw.v_proj.t()).view(1, num_kv, head_dim)
                    v = _rms_norm(v, lw.v_norm, self.eps)
                    layer_cache, v_cache = self.kv_cache_manager.layer_cache(
                        spec.layer_idx
                    )
                    layer_cache.index_put_((write_block, write_offset), k[0])
                    v_cache.index_put_((write_block, write_offset), v[0])

            cache_layer = source if source is not None else spec.layer_idx
            k_cache, v_cache = self.kv_cache_manager.layer_cache(cache_layer)
            if splitk_o is not None and spec.attention_type != SLIDING:
                attn = _splitk_attention_kernel(
                    q, k_cache, v_cache, block_table, context_lens,
                    self.window if spec.attention_type == SLIDING else 0,
                    splitk_o, splitk_m, splitk_l, self._splitk_chunk_size,
                )
            else:
                attn = _paged_attention_kernel(
                    q,
                    k_cache,
                    v_cache,
                    block_table,
                    context_lens,
                    self.window if spec.attention_type == SLIDING else 0,
                )
            h = attn.transpose(0, 1).reshape(1, num_q * head_dim) @ lw.o_proj.t()
            h = _rms_norm(h, lw.post_attention_layernorm, self.eps)
            h, x = fused_add_rms_norm(
                h, residual, lw.pre_feedforward_layernorm, self.eps
            )
            residual = x
            h = self._mlp(h, spec.layer_idx)
            h = _rms_norm(h, lw.post_feedforward_layernorm, self.eps)
            x = residual.add_(h)
            if per_layer_inputs is not None:
                residual = x
                h = _gelu_mul(
                    x @ lw.per_layer_input_gate.t(), per_layer_inputs[:, spec.layer_idx, :]
                )
                h = self._linear(lw.per_layer_projection, h)
                h = _rms_norm(h, lw.post_per_layer_input_norm, self.eps)
                x = residual.add_(h)
            x.mul_(lw.layer_scalar)

        x = _rms_norm(x, w.final_norm, self.eps)
        logits = x[-1:] @ w.lm_head.t()
        if self.softcap is not None:
            logits = self.softcap * torch.tanh(logits.float() / self.softcap)
        return logits.float()[0]

    def _splitk_scratch_for(self, N: int, Hq: int, D: int):
        capacity = (
            int(self.kv_cache_manager.engine_config.num_gpu_blocks)
            * int(self.kv_cache_manager.engine_config.block_size)
        )
        chunks = max(
            1, (capacity + self._splitk_chunk_size - 1) // self._splitk_chunk_size
        )
        key = (N, Hq, D, chunks)
        scratch = self._splitk_scratch.get(key)
        if scratch is None:
            scratch = (
                torch.empty((N, Hq, chunks, D), device=self.device, dtype=torch.float32),
                torch.empty((N, Hq, chunks), device=self.device, dtype=torch.float32),
                torch.empty((N, Hq, chunks), device=self.device, dtype=torch.float32),
            )
            self._splitk_scratch[key] = scratch
        return scratch

    def _graph_tensors(
        self, req: Request, meta: DecodeRequestMetadata,
        create=False, long_context=False,
    ):
        if create:
            n_blocks = int(self.kv_cache_manager.engine_config.num_gpu_blocks)
            if n_blocks <= 0:
                raise RuntimeError("no GPU KV blocks for graph capture")
            state = {
                "block_table": torch.zeros(
                    (1, n_blocks), dtype=torch.int32, device=self.device
                ),
                "context_lens": torch.zeros(
                    1, dtype=torch.int32, device=self.device
                ),
                "token": torch.zeros(1, dtype=torch.long, device=self.device),
                "position": torch.zeros(1, dtype=torch.float32, device=self.device),
                "write_block": torch.zeros(1, dtype=torch.long, device=self.device),
                "write_offset": torch.zeros(1, dtype=torch.long, device=self.device),
                "logits": None,
                "long_context": bool(long_context),
                "splitk": None,
            }
            if long_context:
                max_heads = max(
                    spec.num_attention_heads for spec in self.config.layer_specs
                )
                max_dim = max(spec.head_dim for spec in self.config.layer_specs)
                state["splitk"] = self._splitk_scratch_for(1, max_heads, max_dim)
            self._graph_states[bool(long_context)] = state
            self._graph_static = state
        state = self._graph_states[bool(long_context)]
        table = self.kv_cache_manager.block_table_tensor(req)
        if table.numel() > state["block_table"].shape[1]:
            raise RuntimeError(
                "request block table exceeds the captured graph width"
            )
        state["block_table"].zero_()
        state["block_table"][0, : table.numel()].copy_(table)
        state["context_lens"].fill_(int(meta.context_len))
        state["token"].fill_(int(meta.input_token_id))
        state["position"].fill_(float(meta.query_position))
        state["write_block"].fill_(int(meta.write_slot.block_id))
        state["write_offset"].fill_(int(meta.write_slot.block_offset))
        return state

    def warmup_batch_graphs(self) -> None:
        """Capture the common batch sizes before serving."""
        if not self._batch_graph_mode:
            return
        if os.environ.get("MINI_LLM_GEMMA4_PREWARM_BATCH_GRAPH", "1") == "0":
            return
        max_batch = int(self.kv_cache_manager.engine_config.max_batch_size)
        capacity = (
            int(self.kv_cache_manager.engine_config.num_gpu_blocks)
            * int(self.kv_cache_manager.engine_config.block_size)
        )
        long_prewarm = {2, 4, 8, 16}
        for n in range(2, min(max_batch, 16) + 1):
            for long_context in ((False, True) if n in long_prewarm else (False,)):
                if long_context and capacity <= self._splitk_threshold:
                    continue
                tag = "long" if long_context else "short"
                reqs = [
                    Request(
                        request_id=f"__batch_graph_warmup_{tag}_{n}_{i}__",
                        prompt_token_ids=[0],
                        max_new_tokens=1,
                        sampling_params=None,
                    )
                    for i in range(n)
                ]
                metas = []
                try:
                    for req in reqs:
                        slots = self.kv_cache_manager.ensure_slots_for_request(req, 1)
                        metas.append(DecodeRequestMetadata(
                            request_id=req.request_id,
                            input_token_id=0,
                            query_position=0,
                            context_len=1,
                            write_slot=slots[0],
                        ))
                    self._capture_batch_graph(
                        reqs, metas, long_context=long_context
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[gemma4 graph] batch prewarm {n} {tag} failed: {exc}"
                    )
                finally:
                    for req in reqs:
                        self.kv_cache_manager.free_request(req)

    def warmup_graphs(self) -> None:
        """Capture short and long graphs before serving the first request."""
        if not self._graph_mode or not self._paged_attention:
            return
        if os.environ.get("MINI_LLM_GEMMA4_PREWARM_GRAPH", "1") == "0":
            return
        capacity = (
            int(self.kv_cache_manager.engine_config.num_gpu_blocks)
            * int(self.kv_cache_manager.engine_config.block_size)
        )
        for long_context in (False, True):
            if long_context and capacity <= self._splitk_threshold:
                continue
            context_len = min(
                capacity,
                self._splitk_threshold + 1 if long_context else 1,
            )
            req = Request(
                request_id=f"__graph_warmup_{'long' if long_context else 'short'}__",
                prompt_token_ids=[0],
                max_new_tokens=1,
                sampling_params=None,
            )
            try:
                slots = self.kv_cache_manager.ensure_slots_for_request(
                    req, context_len
                )
                meta = DecodeRequestMetadata(
                    request_id=req.request_id,
                    input_token_id=0,
                    query_position=context_len - 1,
                    context_len=context_len,
                    write_slot=slots[-1],
                )
                self._capture_graph(req, meta, long_context)
            except Exception as exc:  # noqa: BLE001
                print(f"[gemma4 graph] prewarm {long_context} failed: {exc}")
            finally:
                self.kv_cache_manager.free_request(req)

    def _capture_graph(
        self, req: Request, meta: DecodeRequestMetadata, long_context: bool
    ):
        state = self._graph_tensors(
            req, meta, create=True, long_context=long_context
        )
        scratch = state["splitk"]
        args = (
            state["token"],
            state["position"],
            state["context_lens"],
            state["block_table"],
            state["write_block"],
            state["write_offset"],
            *((scratch[0], scratch[1], scratch[2]) if scratch is not None else (None, None, None)),
        )
        for _ in range(3):
            self._decode_one_graphable(*args)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits = self._decode_one_graphable(*args)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        state["logits"] = logits
        self._graphs[bool(long_context)] = graph
        self._decode_graph = graph
        print(
            f"[gemma4 graph] captured blocks={state['block_table'].shape[1]} "
            f"context_lens={int(meta.context_len)} long={bool(long_context)} "
            f"splitk={scratch is not None}"
        )
        return logits.clone()

    def _decode_one_graphed(
        self, req: Request, meta: DecodeRequestMetadata, long_context: bool
    ):
        graph = self._graphs.get(bool(long_context))
        if graph is None:
            return self._capture_graph(req, meta, long_context)
        state = self._graph_tensors(
            req, meta, create=False, long_context=long_context
        )
        graph.replay()
        return state["logits"].clone()

    def _cos_sin_for(self, attention_type: str, positions: torch.Tensor):
        """cos/sin for arbitrary absolute positions [T, D]."""
        inv_freq = self.inv_freq[attention_type]
        freqs = positions[:, None] * inv_freq[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(torch.float32), emb.sin().to(torch.float32)

    def _rope_for(self, positions: torch.Tensor):
        """RoPE cos/sin for every attention type, built once per forward."""
        dtype = self.dtype
        return {
            attention_type: tuple(
                value.to(dtype)
                for value in self._cos_sin_for(attention_type, positions)
            )
            for attention_type in self._rope_types
        }

    def _window_slice(self, spec, k: torch.Tensor, v: torch.Tensor):
        if spec.attention_type == SLIDING and self.window and k.shape[0] > self.window:
            return k[-self.window:], v[-self.window:]
        return k, v

    def _prefill_mask(
        self, attention_type: str, start: int, total: int, chunk_len: int
    ) -> torch.Tensor:
        """Additive mask [chunk_len, total] over absolute positions.

        Query i sits at absolute position start+i and may attend key j iff
        j <= start+i (causal), restricted to the sliding window when the
        layer is a sliding layer.
        """
        device = self.device
        q_abs = torch.arange(start, start + chunk_len, device=device)
        k_abs = torch.arange(0, total, device=device)
        dist = q_abs[:, None] - k_abs[None, :]
        if attention_type == SLIDING:
            allowed = (dist >= 0) & (dist < self.window)
        else:
            allowed = dist >= 0
        mask = torch.zeros(chunk_len, total, device=device, dtype=torch.float32)
        return mask.masked_fill(~allowed, torch.finfo(torch.float32).min)
