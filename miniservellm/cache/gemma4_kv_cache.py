"""Layer-aware KV cache manager for Gemma4 (stage E reference cache).

Implements the same public surface as :class:`KVCacheManager` so the
scheduler, engine and runners work unchanged, but the physical storage is
per-request dense tensors with per-layer shapes. This is required because
Gemma4 mixes head_dim=256 (sliding) and head_dim=512 (full) layers and
shares KV across layers 15..34 — neither fits the uniform paged layout.

Block accounting stays virtual (identical arithmetic to the paged manager)
so capacity checks and the decode reserve watermark behave the same.

Stage F will replace the dense storage with per-layer-group paged blocks.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from miniservellm.config import EngineConfig, ModelConfig
from miniservellm.runtime.metadata import SlotRef
from miniservellm.scheduler.request import Request


class Gemma4KVCacheManager:
    """Per-request, per-layer KV storage with paged-compatible accounting."""

    def __init__(self, engine_config: EngineConfig, model_config: ModelConfig) -> None:
        self.engine_config = engine_config
        self.model_config = model_config

        # Virtual block pool: same math as the paged manager, no tensors.
        self.free_block_ids: List[int] = list(range(engine_config.num_gpu_blocks))
        self.req_block_tables: Dict[str, List[int]] = {}
        # request_id -> layer_idx -> ordered list of (start_pos, k, v)
        self._kv: Dict[str, Dict[int, List[Tuple[int, torch.Tensor, torch.Tensor]]]] = {}

    def _layer_shape(self, layer_idx: int) -> Tuple[int, int]:
        specs = getattr(self.model_config, "layer_specs", [])
        if specs:
            spec = specs[layer_idx]
            return spec.num_key_value_heads, spec.head_dim
        return self.model_config.num_key_value_heads, self.model_config.head_dim

    # -- capacity accounting (virtual blocks, identical to paged manager) ---
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    def needed_new_blocks(self, req: Request, num_new_tokens: int) -> int:
        if num_new_tokens <= 0:
            return 0
        block_size = self.engine_config.block_size
        table = self.req_block_tables.get(req.request_id, req.block_table)
        needed_total_slots = req.total_slots_reserved + num_new_tokens
        needed_total_blocks = (needed_total_slots + block_size - 1) // block_size
        return max(0, needed_total_blocks - len(table))

    def can_allocate_slots(self, req: Request, num_new_tokens: int) -> bool:
        return self.needed_new_blocks(req, num_new_tokens) <= self.num_free_blocks()

    def _allocate_block(self) -> int:
        if not self.free_block_ids:
            raise RuntimeError("Out of KV cache blocks.")
        return self.free_block_ids.pop()

    def ensure_slots_for_request(self, req: Request, num_new_tokens: int) -> List[SlotRef]:
        """Reserve slots for num_new_tokens tokens and return SlotRefs."""
        table = self.req_block_tables.setdefault(req.request_id, list(req.block_table))
        block_size = self.engine_config.block_size
        needed_total = req.total_slots_reserved + num_new_tokens
        needed_blocks = (needed_total + block_size - 1) // block_size
        while len(table) < needed_blocks:
            table.append(self._allocate_block())
        req.block_table = list(table)

        slots: List[SlotRef] = []
        for logical_pos in range(req.total_slots_reserved, needed_total):
            slots.append(
                SlotRef(
                    block_id=table[logical_pos // block_size],
                    block_offset=logical_pos % block_size,
                    logical_pos=logical_pos,
                )
            )
        req.total_slots_reserved += num_new_tokens
        return slots

    # -- KV storage ---------------------------------------------------------
    def write_kv_chunk(
        self,
        request_id: str,
        layer_idx: int,
        start_pos: int,
        k_values: torch.Tensor,
        v_values: torch.Tensor,
    ) -> None:
        """Store one contiguous chunk of K/V for a request and layer."""
        per_layer = self._kv.setdefault(request_id, {})
        chunks = per_layer.setdefault(layer_idx, [])
        expected_start = chunks[-1][0] + chunks[-1][1].shape[0] if chunks else 0
        if start_pos != expected_start:
            raise RuntimeError(
                f"KV chunk gap for {request_id} layer {layer_idx}: "
                f"expected start {expected_start}, got {start_pos}"
            )
        chunks.append((start_pos, k_values, v_values))

    def gather_kv_for_request(
        self,
        layer_idx: int,
        req: Request,
        upto_logical_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return K/V for logical positions [0, upto_logical_length)."""
        chunks = self._kv.get(req.request_id, {}).get(layer_idx, [])
        kv_heads, head_dim = self._layer_shape(layer_idx)
        device = self.engine_config.device
        dtype = self.engine_config.dtype
        if upto_logical_length <= 0 or not chunks:
            empty = torch.empty((0, kv_heads, head_dim), device=device, dtype=dtype)
            return empty, empty
        ks, vs = [], []
        for start, k, v in chunks:
            end = start + k.shape[0]
            if end <= 0 or start >= upto_logical_length:
                continue
            take = min(end, upto_logical_length) - max(start, 0)
            lo = max(start, 0) - start
            ks.append(k[lo : lo + take])
            vs.append(v[lo : lo + take])
        if not ks:
            empty = torch.empty((0, kv_heads, head_dim), device=device, dtype=dtype)
            return empty, empty
        return torch.cat(ks, dim=0), torch.cat(vs, dim=0)

    def free_request(self, req: Request) -> None:
        table = self.req_block_tables.pop(req.request_id, req.block_table)
        for block_id in table:
            self.free_block_ids.append(block_id)
        req.block_table = []
        req.total_slots_reserved = 0
        self._kv.pop(req.request_id, None)

    def debug_global_state(self):
        return {
            "num_free_blocks": len(self.free_block_ids),
            "num_used_blocks": self.engine_config.num_gpu_blocks - len(self.free_block_ids),
            "active_requests": list(self.req_block_tables.keys()),
        }
