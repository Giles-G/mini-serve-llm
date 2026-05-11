"""KV Cache 管理器

第三阶段在第二阶段基础上引入 BlockAllocator，
将 KV Cache 的管理从简单的 dict 映射升级为 paged 风格的 block 管理：
- PagedKVCacheHandle 记录 request 对应的 block_ids 和 token_count
- BlockAllocator 负责 block 的分配、扩容和回收
- past_key_values 仍然以 HF 格式存储在 handle 中（后续替换为 paged attention）

这套抽象的价值在于：后续替换底层实现时，上层 scheduler / engine 代码无需改动。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PagedKVCacheHandle:
    """单个请求对应的 Paged KV Cache 句柄

    Attributes:
        request_id: 请求唯一标识
        block_ids: 请求占用的物理 block id 列表
        token_count: 当前 cache 覆盖的 token 总数（prompt + generated）
        past_key_values: 真实 cache 数据，当前存放 HF 返回的 past_key_values
    """
    request_id: str
    block_ids: list[int] = field(default_factory=list)
    token_count: int = 0
    past_key_values: object | None = None


class KVCacheManager:
    """Paged KV Cache 管理器

    管理请求级别的 KV Cache 生命周期：allocate / get / update / free。
    底层通过 BlockAllocator 管理 block 资源。

    Attributes:
        block_allocator: block 分配器
        handles: request_id -> PagedKVCacheHandle 映射
    """

    def __init__(self, block_allocator):
        self.block_allocator = block_allocator
        self.handles: dict[str, PagedKVCacheHandle] = {}

    def allocate(self, request_id: str) -> PagedKVCacheHandle:
        """为请求分配一个 cache handle

        初始 token_count 为 0，不分配 block。
        后续 update 时根据 token_count 按需分配 block。

        Args:
            request_id: 请求唯一标识

        Returns:
            新分配的 PagedKVCacheHandle
        """
        if request_id in self.handles:
            return self.handles[request_id]

        block_ids = self.block_allocator.allocate_for_request(request_id, token_count=0)
        handle = PagedKVCacheHandle(
            request_id=request_id,
            block_ids=block_ids,
            token_count=0,
            past_key_values=None,
        )
        self.handles[request_id] = handle
        return handle

    def get(self, request_id: str) -> PagedKVCacheHandle | None:
        """获取请求对应的 cache handle

        Args:
            request_id: 请求唯一标识

        Returns:
            PagedKVCacheHandle 或 None
        """
        return self.handles.get(request_id)

    def update(self, request_id: str, past_key_values, token_count: int) -> PagedKVCacheHandle:
        """更新请求的 KV Cache 内容

        同时通过 BlockAllocator 确保请求有足够的 block 容纳 token_count 个 token。

        Args:
            request_id: 请求唯一标识
            past_key_values: HF 模型返回的 past_key_values
            token_count: 当前 cache 覆盖的 token 总数

        Returns:
            更新后的 PagedKVCacheHandle
        """
        handle = self.handles.get(request_id)
        if handle is None:
            handle = self.allocate(request_id)

        # 确保 block 容量足够
        handle.block_ids = self.block_allocator.ensure_capacity(request_id, token_count)
        handle.token_count = token_count
        handle.past_key_values = past_key_values
        self.handles[request_id] = handle
        return handle

    def free(self, request_id: str) -> None:
        """释放请求对应的 cache handle 和 block

        Args:
            request_id: 请求唯一标识
        """
        if request_id in self.handles:
            self.block_allocator.free_request(request_id)
            del self.handles[request_id]
