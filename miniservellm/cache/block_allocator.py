"""Block Allocator — Paged KV Cache 的块分配器

模拟 vLLM 中 PagedAttention 的 block 管理逻辑：
- 物理内存被划分为固定大小的 block（block_size 个 token 共享一个 block）
- 每个 request 按需占用 block，随 token 增长动态扩容
- request 完成后释放 block，供其他请求复用

当前版本只管理 block id 元数据，不管理真实 tensor 存储。
后续替换为 paged attention kernel 时，block id 将对应真实 GPU 内存页。
"""

from __future__ import annotations

import math


class BlockAllocator:
    """块分配器

    管理一组固定大小的 block，支持按 token 数量分配、扩容和释放。

    Attributes:
        num_blocks: 总 block 数量
        block_size: 每个 block 可容纳的 token 数
    """

    def __init__(self, num_blocks: int = 1024, block_size: int = 16):
        self.num_blocks = num_blocks
        self.block_size = block_size

        # 空闲 block 列表（用 list 模拟栈结构，pop 分配、extend 回收）
        self._free_blocks: list[int] = list(range(num_blocks))
        # request_id -> 已分配的 block id 列表
        self._request_to_blocks: dict[str, list[int]] = {}

    def _required_blocks(self, token_count: int) -> int:
        """根据 token 数量计算需要的 block 数

        Args:
            token_count: 需要容纳的 token 数

        Returns:
            需要的 block 数（向上取整）
        """
        if token_count <= 0:
            return 0
        return math.ceil(token_count / self.block_size)

    def allocate_for_request(self, request_id: str, token_count: int = 0) -> list[int]:
        """为新请求分配 block

        Args:
            request_id: 请求唯一标识
            token_count: 初始 token 数量，用于计算所需 block 数

        Returns:
            分配的 block id 列表

        Raises:
            RuntimeError: 空闲 block 不足
        """
        need = self._required_blocks(token_count)
        blocks = []

        if need > 0:
            if len(self._free_blocks) < need:
                raise RuntimeError(
                    f"Not enough free blocks: need={need}, free={len(self._free_blocks)}"
                )
            for _ in range(need):
                blocks.append(self._free_blocks.pop())

        self._request_to_blocks[request_id] = blocks
        return blocks

    def ensure_capacity(self, request_id: str, token_count: int) -> list[int]:
        """确保请求有足够的 block 容纳 token_count 个 token

        如果已有 block 不够，会自动扩容分配新 block。
        如果已够，直接返回当前 block 列表。

        Args:
            request_id: 请求唯一标识
            token_count: 需要容纳的 token 数

        Returns:
            请求当前拥有的 block id 列表

        Raises:
            RuntimeError: 空闲 block 不足以扩容
        """
        need = self._required_blocks(token_count)
        cur = self._request_to_blocks.get(request_id, [])
        have = len(cur)

        # 已有 block 足够，直接返回
        if need <= have:
            return cur

        # 需要额外分配
        extra = need - have
        if len(self._free_blocks) < extra:
            raise RuntimeError(
                f"Not enough free blocks for growth: need_extra={extra}, free={len(self._free_blocks)}"
            )

        for _ in range(extra):
            cur.append(self._free_blocks.pop())

        self._request_to_blocks[request_id] = cur
        return cur

    def free_request(self, request_id: str) -> None:
        """释放请求占用的所有 block

        Args:
            request_id: 请求唯一标识
        """
        blocks = self._request_to_blocks.pop(request_id, [])
        # 将 block 归还空闲池
        self._free_blocks.extend(blocks)

    def blocks_of(self, request_id: str) -> list[int]:
        """获取请求当前占用的 block id 列表

        Args:
            request_id: 请求唯一标识

        Returns:
            block id 列表的拷贝
        """
        return list(self._request_to_blocks.get(request_id, []))

    def num_free_blocks(self) -> int:
        """返回当前空闲 block 数量"""
        return len(self._free_blocks)

    def num_used_blocks(self) -> int:
        """返回当前已使用 block 数量"""
        return self.num_blocks - len(self._free_blocks)
