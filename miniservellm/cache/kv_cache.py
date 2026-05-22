"""Paged KV Cache Manager

第五阶段核心组件：将 KV Cache 组织成固定大小的 block，
支持按请求分配、写入、收集和释放。

不同于第四阶段使用 HF 模型自带的 past_key_values，
第五阶段使用自研模型前向，KV 直接写入预分配的物理 block。

核心设计思路（类操作系统虚拟内存）：
- 物理显存预分配为固定大小的 block（类似物理页帧）
- 每个请求通过 block_table 映射逻辑位置到物理 block（类似页表）
- block 按需分配、请求结束时释放（类似内存分配/回收）
- 不同请求的 KV 数据物理上散落在同一个大 tensor 中，逻辑上通过 block_table 保持连续

数据流转：
    1. Runner 调用 ensure_slots_for_request() 为新 token 分配 slot → 返回 SlotRef 列表
    2. Runner 调用 write_kv_for_tokens() 将计算出的 K/V 写入物理 block
    3. Runner 调用 gather_kv_for_request() 从物理 block 读取历史 KV 用于 attention 计算
    4. 请求结束时，引擎调用 free_request() 释放所有 block
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from miniservellm.config import EngineConfig, ModelConfig
from miniservellm.runtime.metadata import SlotRef
from miniservellm.scheduler.request import Request


@dataclass
class BlockAllocation:
    """Block 分配结果

    Attributes:
        block_id: 分配到的物理 block 编号（在 k_cache/v_cache 的第 2 维中的索引）
    """
    block_id: int


class KVCacheManager:
    """Paged KV Cache 管理器

    将 KV Cache 组织为 [n_layers, n_blocks, block_size, n_kv_heads, head_dim] 的物理 tensor。
    每个请求通过 block_table 映射到物理 block，支持按需分配和释放。

    物理存储结构：
        k_cache / v_cache: [n_layers, n_blocks, block_size, n_kv_heads, head_dim]

        维度说明：
        - n_layers:     Transformer 层数（如 24），每层有独立的 KV
        - n_blocks:     物理 block 总数（如 1024），决定 KV Cache 总容量
        - block_size:   每个 block 存储的 token 数（如 16），类似内存页大小
        - n_kv_heads:   KV head 数量（GQA 下小于 Q head 数量，如 2）
        - head_dim:     每个 head 的维度（如 64）

        访问某个 token 的 KV：
            k_cache[layer_idx, block_id, block_offset]  →  [n_kv_heads, head_dim]

    逻辑到物理的映射（block_table）：
        每个请求维护一个 block_table（整数列表），记录该请求占用了哪些物理 block。

        逻辑位置 → 物理 block 的转换：
            block_idx    = logical_pos // block_size   # 逻辑 block 索引
            block_offset = logical_pos % block_size    # 块内偏移
            block_id     = block_table[block_idx]       # 查表得到物理 block 编号

        示例（block_size=16, block_table=[5, 17, 3]）：
            逻辑位置 0~15  → 物理 block 5 的 offset 0~15
            逻辑位置 16~31 → 物理 block 17 的 offset 0~15
            逻辑位置 32~47 → 物理 block 3 的 offset 0~15

    Block 生命周期：
        分配：ensure_slots_for_request() → _allocate_block() → 从 free_block_ids 取一个
        使用：write_kv_for_tokens() 写入 / gather_kv_for_request() 读取
        释放：free_request() → block_id 归还 free_block_ids

    Attributes:
        k_cache: Key Cache 物理存储 [n_layers, n_blocks, block_size, n_kv_heads, head_dim]
        v_cache: Value Cache 物理存储 [n_layers, n_blocks, block_size, n_kv_heads, head_dim]
        free_block_ids: 空闲物理 block 编号列表，分配时从末尾 pop，释放时 append 回来
        req_block_tables: request_id → block_table 的映射，记录每个请求占用的物理 block 列表
    """

    def __init__(self, engine_config: EngineConfig, model_config: ModelConfig) -> None:
        """初始化 KV Cache 管理器

        预分配全部 KV Cache 显存，运行时不再动态分配。
        这确保了运行时零显存分配开销，且启动时即可确定最大服务容量。

        Args:
            engine_config: 引擎配置，包含 num_gpu_blocks、block_size、device、dtype 等
            model_config: 模型配置，包含 num_hidden_layers、num_key_value_heads、head_dim 等
        """
        self.engine_config = engine_config
        self.model_config = model_config

        n_layers = model_config.num_hidden_layers
        n_blocks = engine_config.num_gpu_blocks
        block_size = engine_config.block_size
        n_kv_heads = model_config.num_key_value_heads
        head_dim = model_config.head_dim
        device = engine_config.device
        dtype = engine_config.dtype

        # 预分配全部 KV Cache 显存
        # 每个 block 可存储 block_size 个 token 的 KV，总共 n_blocks * block_size 个 token
        self.k_cache = torch.zeros(
            (n_layers, n_blocks, block_size, n_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self.v_cache = torch.zeros(
            (n_layers, n_blocks, block_size, n_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )

        # 初始时所有 block 都是空闲的
        self.free_block_ids: List[int] = list(range(n_blocks))
        # 请求的 block_table 缓存，避免频繁从 Request 对象拷贝
        self.req_block_tables: Dict[str, List[int]] = {}

    def _allocate_block(self) -> int:
        """从空闲列表分配一个物理 block

        Returns:
            分配到的 block 编号

        Raises:
            RuntimeError: 没有空闲 block 可分配（KV Cache 容量不足）
        """
        if not self.free_block_ids:
            raise RuntimeError("Out of KV cache blocks.")
        return self.free_block_ids.pop()

    def _ensure_block_table(self, req: Request) -> List[int]:
        """确保请求在 req_block_tables 中有 block_table 缓存

        首次访问时从 req.block_table 复制一份到 req_block_tables，
        后续操作都修改 req_block_tables 中的副本，最后同步回 req.block_table。

        Args:
            req: 请求对象

        Returns:
            该请求的 block_table 列表（可修改的引用）
        """
        if req.request_id not in self.req_block_tables:
            self.req_block_tables[req.request_id] = list(req.block_table)
        return self.req_block_tables[req.request_id]

    def ensure_slots_for_request(self, req: Request, num_new_tokens: int) -> List[SlotRef]:
        """为请求的 num_new_tokens 个新 token 确保足够的 slot，必要时分配新 block

        这是 Runner 在 prefill/decode 前调用的核心方法，用于：
        1. 计算当前需要的总 block 数（向上取整除法）
        2. 不足时分配新 block，追加到 block_table
        3. 为每个新 token 构造 SlotRef（包含 block_id、block_offset、logical_pos）

        SlotRef 随后传给 write_kv_for_tokens() 使用，避免重复计算物理地址。

        使用示例：
            # Prefill: 为 chunk 的 32 个 token 分配 slot
            slots = kv_cache_manager.ensure_slots_for_request(req, 32)
            # 写入 KV
            kv_cache_manager.write_kv_for_tokens(layer_idx=0, slot_refs=slots, k_values=k, v_values=v)

            # Decode: 为 1 个新 token 分配 slot
            slots = kv_cache_manager.ensure_slots_for_request(req, 1)

        Args:
            req: 请求对象，其 total_slots_reserved 记录已分配的 slot 数
            num_new_tokens: 需要新增的 token 数

        Returns:
            SlotRef 列表，长度为 num_new_tokens，每个 SlotRef 包含：
                - block_id: 物理 block 编号
                - block_offset: 块内偏移（0 ~ block_size-1）
                - logical_pos: 逻辑位置（全局递增）
        """
        table = self._ensure_block_table(req)
        block_size = self.engine_config.block_size

        # 计算需要的总 block 数（向上取整：needed_total / block_size）
        needed_total = req.total_slots_reserved + num_new_tokens
        needed_blocks = (needed_total + block_size - 1) // block_size

        # 分配不足的 block
        while len(table) < needed_blocks:
            new_block = self._allocate_block()
            table.append(new_block)

        # 同步回请求对象
        req.block_table = list(table)

        # 为每个新 token 构造 SlotRef
        slots: List[SlotRef] = []
        start = req.total_slots_reserved  # 从已分配的位置之后开始
        end = start + num_new_tokens
        for logical_pos in range(start, end):
            block_idx = logical_pos // block_size    # 逻辑 block 索引
            block_offset = logical_pos % block_size   # 块内偏移
            block_id = table[block_idx]                # 查表得到物理 block
            slots.append(
                SlotRef(
                    block_id=block_id,
                    block_offset=block_offset,
                    logical_pos=logical_pos,
                )
            )

        # 更新已分配的 slot 总数
        req.total_slots_reserved += num_new_tokens
        return slots

    def write_kv_for_tokens(
        self,
        layer_idx: int,
        slot_refs: List[SlotRef],
        k_values: torch.Tensor,
        v_values: torch.Tensor,
    ) -> None:
        """将计算出的 K/V 值写入物理 KV Cache

        在 Transformer 每一层的前向中调用，将新 token 的 K/V 写入对应的物理 slot。
        写入位置由 ensure_slots_for_request() 返回的 SlotRef 指定。

        写入过程（对每个 token）：
            k_cache[layer_idx, slot.block_id, slot.block_offset] = k_values[i]
            v_cache[layer_idx, slot.block_id, slot.block_offset] = v_values[i]

        使用示例：
            # 在 TransformerBlockRunner._attention_prefill() 中
            self.kv_cache_manager.write_kv_for_tokens(
                layer_idx=self.layer_idx,
                slot_refs=meta.write_slots,
                k_values=k_new,   # [chunk_len, n_kv_heads, head_dim]
                v_values=v_new,   # [chunk_len, n_kv_heads, head_dim]
            )

        Args:
            layer_idx: Transformer 层编号（0 ~ n_layers-1）
            slot_refs: SlotRef 列表，由 ensure_slots_for_request() 生成
            k_values: Key 值 [T, n_kv_heads, head_dim]，T = len(slot_refs)
            v_values: Value 值 [T, n_kv_heads, head_dim]
        """
        assert len(slot_refs) == k_values.shape[0] == v_values.shape[0]
        for i, slot in enumerate(slot_refs):
            self.k_cache[layer_idx, slot.block_id, slot.block_offset].copy_(k_values[i])
            self.v_cache[layer_idx, slot.block_id, slot.block_offset].copy_(v_values[i])

    def gather_kv_for_request(
        self,
        layer_idx: int,
        req: Request,
        upto_logical_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """收集请求在指定范围内的历史 KV Cache

        在 attention 计算前调用，读取该请求的所有历史 K/V，用于与当前 Q 计算注意力。

        读取过程：
            对 logical_pos = 0, 1, ..., upto_logical_length-1：
                block_idx    = logical_pos // block_size
                block_offset = logical_pos % block_size
                block_id     = block_table[block_idx]
                k_list.append(k_cache[layer_idx, block_id, block_offset])
                v_list.append(v_cache[layer_idx, block_id, block_offset])
            最后 stack 为连续 tensor

        使用示例：
            # Prefill: 读取 chunk 之前的历史 KV
            hist_k, hist_v = self.kv_cache_manager.gather_kv_for_request(
                layer_idx=self.layer_idx,
                req=req,
                upto_logical_length=meta.context_len_before_chunk,  # chunk 之前的长度
            )
            # hist_k: [history_len, n_kv_heads, head_dim]

            # Decode: 读取全部 KV（包括刚写入的新 token）
            full_k, full_v = self.kv_cache_manager.gather_kv_for_request(
                layer_idx=self.layer_idx,
                req=req,
                upto_logical_length=meta.context_len,  # 包含新 token 的总长度
            )

        Args:
            layer_idx: Transformer 层编号
            req: 请求对象（通过 block_table 定位物理 block）
            upto_logical_length: 收集到哪个逻辑位置（不含），即收集 [0, upto_logical_length) 的 KV

        Returns:
            (key, value) 元组，各自形状 [upto_logical_length, n_kv_heads, head_dim]
            当 upto_logical_length=0 时返回空 tensor [0, n_kv_heads, head_dim]
        """
        table = self.req_block_tables.get(req.request_id, req.block_table)
        kv_heads = self.model_config.num_key_value_heads
        head_dim = self.model_config.head_dim
        device = self.engine_config.device
        dtype = self.engine_config.dtype
        block_size = self.engine_config.block_size

        # 边界情况：没有历史 KV，返回空 tensor
        if upto_logical_length == 0:
            empty = torch.empty((0, kv_heads, head_dim), device=device, dtype=dtype)
            return empty, empty

        # 逐 token 读取：逻辑位置 → 查 block_table → 读取物理 block 中的数据
        k_list = []
        v_list = []
        for logical_pos in range(upto_logical_length):
            block_idx = logical_pos // block_size
            block_offset = logical_pos % block_size
            block_id = table[block_idx]
            k_list.append(self.k_cache[layer_idx, block_id, block_offset])
            v_list.append(self.v_cache[layer_idx, block_id, block_offset])

        # stack 成连续 tensor: [upto_logical_length, n_kv_heads, head_dim]
        return torch.stack(k_list, dim=0), torch.stack(v_list, dim=0)

    def free_request(self, req: Request) -> None:
        """释放请求占用的所有 KV Cache block

        请求结束时（EOS 或达到最大生成长度）由引擎调用。
        将该请求的所有物理 block 归还到 free_block_ids，
        并清理 req_block_tables 和请求对象的状态。

        Args:
            req: 已完成的请求对象
        """
        table = self.req_block_tables.pop(req.request_id, req.block_table)
        for block_id in table:
            self.free_block_ids.append(block_id)  # 归还空闲列表
        req.block_table = []
        req.total_slots_reserved = 0

    def debug_global_state(self):
        """获取 KV Cache 全局状态的调试信息

        Returns:
            字典包含：
                - num_free_blocks: 空闲 block 数
                - num_used_blocks: 已使用 block 数
                - active_requests: 活跃请求 ID 列表
        """
        return {
            "num_free_blocks": len(self.free_block_ids),
            "num_used_blocks": self.engine_config.num_gpu_blocks - len(self.free_block_ids),
            "active_requests": list(self.req_block_tables.keys()),
        }
