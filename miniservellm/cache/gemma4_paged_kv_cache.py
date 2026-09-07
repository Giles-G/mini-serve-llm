"""Layer-grouped paged KV cache for Gemma4 (stage F).

Gemma4 cannot use the uniform ``[n_layers, n_blocks, block_size, kv_heads,
head_dim]`` paged layout: sliding layers use head_dim=256 with 1 KV head and
full layers use head_dim=512. Layers 15..34 share KV with their source layer
and therefore never allocate storage.

Design: group layers by (num_key_value_heads, head_dim) into cache groups,
each with its own physical block pool layout ``[n_blocks, block_size, kv,
head_dim]`` for K and V. A single per-request ``block_table`` maps logical
token positions to physical blocks across ALL groups (one block allocation
covers every layer, like the Qwen paged manager).

Block accounting is identical to the paged/dense managers so the scheduler
(capacity pre-checks, decode reserve watermark) works unchanged.

Compared to the stage-E dense manager, gather is a single advanced-indexing
call per layer (no chunk concatenation), and the per-request index tensors
are memoized so the 15 non-shared layers reuse one computation per step.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from miniservellm.config import EngineConfig, ModelConfig
from miniservellm.runtime.metadata import SlotRef
from miniservellm.scheduler.request import Request


class Gemma4PagedKVCacheManager:
    """Paged KV storage grouped by per-layer KV shape."""

    def __init__(self, engine_config: EngineConfig, model_config: ModelConfig) -> None:
        self.engine_config = engine_config
        self.model_config = model_config

        specs = getattr(model_config, "layer_specs", None)
        if not specs:
            raise ValueError("Gemma4PagedKVCacheManager requires model_config.layer_specs")

        device = engine_config.device
        dtype = engine_config.dtype
        n_blocks = engine_config.num_gpu_blocks
        block_size = engine_config.block_size

        # Shared layers (per num_kv_shared_layers) never store; collect the
        # rest. The config converter leaves kv_source_layer=None on purpose,
        # so derive sharing from the layer count rule here.
        num_layers = model_config.num_hidden_layers
        first_shared = num_layers - model_config.num_kv_shared_layers
        storing = [s for s in specs if s.layer_idx < first_shared]
        # group key -> list of storing layer indices
        self._groups: Dict[Tuple[int, int], List[int]] = {}
        for spec in storing:
            key = (spec.num_key_value_heads, spec.head_dim)
            self._groups.setdefault(key, []).append(spec.layer_idx)
        self._layer_group: Dict[int, Tuple[Tuple[int, int], int]] = {
            layer: (key, local_idx)
            for key, layers in self._groups.items()
            for local_idx, layer in enumerate(layers)
        }

        self._k_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        self._v_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        for (kv_heads, head_dim), layers in self._groups.items():
            # Physical layout includes the GROUP-LOCAL layer dimension:
            # [n_group_layers, n_blocks, block_size, kv_heads, head_dim].
            # Layers in the same group share block accounting but each has
            # its own storage slice — dropping this dim would make layers
            # overwrite each other's KV.
            shape = (len(layers), n_blocks, block_size, kv_heads, head_dim)
            self._k_cache[(kv_heads, head_dim)] = torch.zeros(shape, device=device, dtype=dtype)
            self._v_cache[(kv_heads, head_dim)] = torch.zeros(shape, device=device, dtype=dtype)

        # Virtual block accounting (same math as the paged manager).
        self.free_block_ids: List[int] = list(range(n_blocks))
        self.req_block_tables: Dict[str, List[int]] = {}
        # Single-entry memo for gather indices: within one forward the 15
        # non-shared layers query the same (rid, upto) sequentially.
        self._index_memo: Optional[Tuple[str, int, torch.Tensor, torch.Tensor]] = None

    # -- capacity accounting -------------------------------------------------
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
        self._index_memo = None
        return slots

    # -- KV storage -----------------------------------------------------------
    def write_kv_for_tokens(
        self,
        layer_idx: int,
        slot_refs: List[SlotRef],
        k_values: torch.Tensor,
        v_values: torch.Tensor,
    ) -> None:
        """Scatter K/V [T, kv, dim] into the layer group's physical blocks."""
        assert len(slot_refs) == k_values.shape[0] == v_values.shape[0]
        if not slot_refs:
            return
        key, local_layer = self._layer_group[layer_idx]
        block_ids = torch.tensor([s.block_id for s in slot_refs], device=self.engine_config.device, dtype=torch.long)
        offsets = torch.tensor([s.block_offset for s in slot_refs], device=self.engine_config.device, dtype=torch.long)
        self._k_cache[key][local_layer, block_ids, offsets] = k_values
        self._v_cache[key][local_layer, block_ids, offsets] = v_values

    def gather_kv_for_request(
        self,
        layer_idx: int,
        req: Request,
        upto_logical_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather K/V for logical positions [0, upto) via advanced indexing."""
        key, local_layer = self._layer_group[layer_idx]
        kv_heads, head_dim = key
        device = self.engine_config.device
        dtype = self.engine_config.dtype
        if upto_logical_length <= 0:
            empty = torch.empty((0, kv_heads, head_dim), device=device, dtype=dtype)
            return empty, empty

        block_ids, offsets = self._gather_indices(req, upto_logical_length)
        k = self._k_cache[key][local_layer, block_ids, offsets]
        v = self._v_cache[key][local_layer, block_ids, offsets]
        return k, v

    def _gather_indices(
        self, req: Request, upto: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(block_ids, block_offsets) for positions [0, upto), memoized.

        Within one forward the 15 non-shared layers query the same
        (rid, upto) sequentially, so the index computation happens once per
        request-step. ensure_slots_for_request invalidates the memo.
        """
        memo = self._index_memo
        if memo is not None and memo[0] == req.request_id and memo[1] == upto:
            return memo[2], memo[3]
        table = self.req_block_tables.get(req.request_id, req.block_table)
        device = self.engine_config.device
        block_size = self.engine_config.block_size
        positions = torch.arange(upto, device=device, dtype=torch.long)
        block_idx = positions // block_size
        offsets = positions % block_size
        table_t = torch.tensor(table, device=device, dtype=torch.long)
        block_ids = table_t[block_idx]
        self._index_memo = (req.request_id, upto, block_ids, offsets)
        return block_ids, offsets

    def free_request(self, req: Request) -> None:
        table = self.req_block_tables.pop(req.request_id, req.block_table)
        for block_id in table:
            self.free_block_ids.append(block_id)
        req.block_table = []
        req.total_slots_reserved = 0
        self._index_memo = None

    def debug_global_state(self):
        return {
            "num_free_blocks": len(self.free_block_ids),
            "num_used_blocks": self.engine_config.num_gpu_blocks - len(self.free_block_ids),
            "active_requests": list(self.req_block_tables.keys()),
            "cache_groups": {
                f"kv{kv}xdim{dim}": len(layers) for (kv, dim), layers in self._groups.items()
            },
        }
