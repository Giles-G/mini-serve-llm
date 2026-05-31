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

        注意：正常服务路径不应该依赖这里抛异常来做流控。
        Scheduler 会先调用 can_allocate_slots() 做容量预检查；这里的异常只是
        防止调用方绕过调度器导致 KV Cache 状态损坏的最后防线。
        """
        if not self.free_block_ids:
            raise RuntimeError("Out of KV cache blocks.")
        return self.free_block_ids.pop()

    def num_free_blocks(self) -> int:
        """返回当前空闲物理 block 数。"""
        return len(self.free_block_ids)

    def needed_new_blocks(self, req: Request, num_new_tokens: int) -> int:
        """计算为 req 追加 num_new_tokens 个 slot 还需要新分配多少个 block。

        只做纯计算，不修改 block_table / total_slots_reserved / free_block_ids。
        Scheduler 用它在生成执行计划前做容量预检查，避免运行到一半才发现KV block 不够然后直接抛异常退出。
        """
        if num_new_tokens <= 0:
            return 0
        block_size = self.engine_config.block_size
        table = self.req_block_tables.get(req.request_id, req.block_table)
        needed_total_slots = req.total_slots_reserved + num_new_tokens
        needed_total_blocks = (needed_total_slots + block_size - 1) // block_size
        return max(0, needed_total_blocks - len(table))

    def can_allocate_slots(self, req: Request, num_new_tokens: int) -> bool:
        """判断当前空闲 block 是否足够为 req 追加 num_new_tokens 个 slot。"""
        return self.needed_new_blocks(req, num_new_tokens) <= self.num_free_blocks()

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
        """将计算出的 K/V 值写入物理 KV Cache（向量化版本）

        将一批 token 的 K/V 一次性写入指定的物理 slot，避免 Python for-loop 调度开销。

        写入过程（向量化等价于）：
            for i, slot in enumerate(slot_refs):
                k_cache[layer_idx, slot.block_id, slot.block_offset] = k_values[i]
                v_cache[layer_idx, slot.block_id, slot.block_offset] = v_values[i]

        实现：先把 (block_id, block_offset) 提取成两个 LongTensor，
        通过 advanced indexing 一次完成 N 个 token 的散列写入。

        Args:
            layer_idx: Transformer 层编号（0 ~ n_layers-1）
            slot_refs: SlotRef 列表，由 ensure_slots_for_request() 生成
            k_values: Key 值 [T, n_kv_heads, head_dim]，T = len(slot_refs)
            v_values: Value 值 [T, n_kv_heads, head_dim]
        """
        assert len(slot_refs) == k_values.shape[0] == v_values.shape[0]
        if len(slot_refs) == 0:
            return
        device = self.engine_config.device
        block_ids = torch.tensor(
            [s.block_id for s in slot_refs], device=device, dtype=torch.long
        )
        block_offsets = torch.tensor(
            [s.block_offset for s in slot_refs], device=device, dtype=torch.long
        )
        # 一次性 advanced indexing 写入：避免 N 次 kernel launch
        self.k_cache[layer_idx, block_ids, block_offsets] = k_values
        self.v_cache[layer_idx, block_ids, block_offsets] = v_values

    def write_kv_for_tokens_batch(
        self,
        layer_idx: int,
        slot_refs: List[SlotRef],
        k_values: torch.Tensor,
        v_values: torch.Tensor,
    ) -> None:
        """语义同 write_kv_for_tokens，专门用于 batch decode/prefill 的 N 个 token 一起写

        与 write_kv_for_tokens 实现相同（已经是向量化的），保留单独命名以便上层语义清晰。
        """
        self.write_kv_for_tokens(layer_idx, slot_refs, k_values, v_values)

    def slot_refs_to_indices(
        self,
        slot_refs: List[SlotRef],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """将 SlotRef 列表转换为 (block_ids, block_offsets) 两个 LongTensor。

        用于跨层共享：上层在一个 step 中构造一次后传入 write_kv_for_tokens_indexed，
        避免每层重复构造同样的 host→device 张量（24 层 × 2 = 48 次）。
        """
        device = self.engine_config.device
        if not slot_refs:
            empty = torch.zeros(0, device=device, dtype=torch.long)
            return empty, empty
        block_ids = torch.tensor(
            [s.block_id for s in slot_refs], device=device, dtype=torch.long
        )
        block_offsets = torch.tensor(
            [s.block_offset for s in slot_refs], device=device, dtype=torch.long
        )
        return block_ids, block_offsets

    def write_kv_for_tokens_indexed(
        self,
        layer_idx: int,
        block_ids: torch.Tensor,
        block_offsets: torch.Tensor,
        k_values: torch.Tensor,
        v_values: torch.Tensor,
    ) -> None:
        """使用预计算的 (block_ids, block_offsets) 写入 KV，避免每层重建索引张量。"""
        if k_values.shape[0] == 0:
            return
        self.k_cache[layer_idx, block_ids, block_offsets] = k_values
        self.v_cache[layer_idx, block_ids, block_offsets] = v_values

    def build_decode_block_table(
        self,
        reqs: List[Request],
        max_ctx: int,
    ) -> torch.Tensor:
        """构造 decode 批次的 block_table 张量 [N, max_blocks]（供 CUDA kernel 使用）

        M4 优化：decode_paged_attention CUDA kernel 直接按 block_table 跳读 KV，
        不需要先 gather 成连续 tensor。

        Args:
            reqs: 长度 N 的请求列表
            max_ctx: 本批次最大上下文长度（用于计算 max_blocks）

        Returns:
            block_table_tensor: [N, max_blocks]  int32，不足位补 0
        """
        N = len(reqs)
        block_size = self.engine_config.block_size
        device = self.engine_config.device
        max_blocks = (max_ctx + block_size - 1) // block_size if max_ctx > 0 else 1

        bt = torch.zeros((N, max_blocks), device=device, dtype=torch.int32)
        for i, req in enumerate(reqs):
            table = self.req_block_tables.get(req.request_id, req.block_table)
            if table:
                t = torch.tensor(table[:max_blocks], device=device, dtype=torch.int32)
                bt[i, : t.shape[0]] = t
        return bt

    def build_decode_batch_indices(
        self,
        reqs: List[Request],
        context_lens: List[int],
        padded_max_ctx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """为 N 个 decode 请求构造批量 KV gather 用的索引张量

        返回：
          - block_ids_padded: [N, max_ctx]  每个 token 的物理 block id（无效位补 0）
          - block_offsets:    [N, max_ctx]  每个 token 的块内偏移

        无效位（pos >= context_len[i]）填 (0, 0)，对应一个合法但内容无关的位置；
        attention 计算时会通过 context_lens mask 屏蔽这些位置。

        Args:
            reqs: 长度 N 的请求列表
            context_lens: 长度 N 的真实上下文长度列表
            padded_max_ctx: 指定 padding 后的 max_ctx（0 表示按真实 max_ctx）

        Returns:
            (block_ids_padded, block_offsets) 两个 LongTensor
        """
        N = len(reqs)
        max_ctx_raw = max(context_lens) if context_lens else 0
        max_ctx = max(max_ctx_raw, padded_max_ctx)
        device = self.engine_config.device
        block_size = self.engine_config.block_size

        if max_ctx == 0:
            empty = torch.zeros((N, 0), device=device, dtype=torch.long)
            return empty, empty

        # 每行 [0, 1, ..., max_ctx-1]
        pos_grid = torch.arange(max_ctx, device=device, dtype=torch.long)
        pos_grid = pos_grid.unsqueeze(0).expand(N, max_ctx)              # [N, max_ctx]
        block_idx = pos_grid // block_size                                # [N, max_ctx]
        block_offsets = pos_grid % block_size                             # [N, max_ctx]

        # 构造 padding 后的 block_table: [N, max_blocks]，无效位填 0
        max_blocks = (max_ctx + block_size - 1) // block_size
        block_table_padded = torch.zeros((N, max_blocks), device=device, dtype=torch.long)
        for i, req in enumerate(reqs):
            table = self.req_block_tables.get(req.request_id, req.block_table)
            if table:
                t = torch.tensor(table[:max_blocks], device=device, dtype=torch.long)
                block_table_padded[i, : t.shape[0]] = t

        # 用 block_idx 在 block_table_padded 上 gather 得到物理 block_id
        # block_idx 在无效位也会取到 (max_blocks-1) 之前的某个值，但因为 block_table_padded
        # 末尾填 0，且无效位会被 attention mask 屏蔽，所以即便取到 0 号 block 也无害。
        # 但为了安全，clamp 到 max_blocks-1
        block_idx_safe = block_idx.clamp(max=max_blocks - 1)
        block_ids_padded = torch.gather(block_table_padded, 1, block_idx_safe)  # [N, max_ctx]

        return block_ids_padded, block_offsets

    def gather_kv_decode_batch(
        self,
        layer_idx: int,
        reqs: List[Request],
        context_lens: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """N 个 decode 请求一次性批量 gather padded KV。

        把每请求的 paged 物理 block（按 block_table 离散存放）通过一次
        advanced indexing **物化** 成 [N, max_ctx, kv_heads, head_dim] 的
        padded 稠密张量，供上层走常规 batched attention。

        注意：这一步等价于「把 paged 存储 flatten 回连续布局」，并不是
        vLLM 风格的 paged-attention kernel（那种 kernel 不会物化、直接
        在 attention 内沿 block 跳读）。本实现是 stage 6 纯 Python/PyTorch
        路线下的折中：存储侧 paged，计算侧 gather → padded → batched matmul。

        Returns:
            k_padded: [N, max_ctx, kv_heads, head_dim]
            v_padded: [N, max_ctx, kv_heads, head_dim]
            ctx_lens_t: [N] long，真实上下文长度（供 attention mask 用）
        """
        device = self.engine_config.device
        ctx_lens_t = torch.tensor(context_lens, device=device, dtype=torch.long)
        block_ids, block_offsets = self.build_decode_batch_indices(reqs, context_lens)
        # advanced indexing 一次拿到 [N, max_ctx, kv_heads, head_dim]
        k_padded = self.k_cache[layer_idx, block_ids, block_offsets]
        v_padded = self.v_cache[layer_idx, block_ids, block_offsets]
        return k_padded, v_padded, ctx_lens_t

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

        # 向量化读取：一次性算出每个 logical_pos 对应的物理 (block_id, block_offset)，
        # 通过 advanced indexing 一次拿到全部 KV，避免 Python for-loop。
        logical_positions = torch.arange(upto_logical_length, device=device, dtype=torch.long)
        block_idx = logical_positions // block_size           # [L]
        block_offsets = logical_positions % block_size        # [L]
        # block_table 转 tensor 后用 logical block_idx 查表得到物理 block_id
        block_table_tensor = torch.tensor(table, device=device, dtype=torch.long)
        block_ids = block_table_tensor[block_idx]             # [L]

        # 一次 advanced indexing 拿到 [L, n_kv_heads, head_dim]
        k = self.k_cache[layer_idx, block_ids, block_offsets]
        v = self.v_cache[layer_idx, block_ids, block_offsets]
        return k, v

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
