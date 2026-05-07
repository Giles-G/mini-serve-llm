"""KV Cache 管理器

当前版本先不自己管理真实 tensor 内存，只维护：
request_id -> KVCacheHandle 的映射。

HF 模型返回的 past_key_values 仍然作为 data 存在 handle 中。
这个抽象层的价值在于：后续可以在不改 scheduler / engine 的情况下，
把 data 替换成 paged KV cache block handle。
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class KVCacheHandle:
    """单个请求对应的 KV Cache 句柄

    Attributes:
        request_id: 请求唯一标识
        backend: 当前 cache 后端名称，本阶段使用 HF 的 past_key_values
        token_count: 当前 cache 覆盖的 token 总数（prompt + generated）
        data: 真实 cache 数据，本阶段存放 HF 返回的 past_key_values
    """
    request_id: str
    backend: str = "hf_past_key_values"
    token_count: int = 0
    data: Optional[Any] = None


class KVCacheManager:
    """KV Cache 管理器

    管理请求级别的 KV Cache 生命周期：allocate / get / update / free。
    """

    def __init__(self):
        # request_id -> KVCacheHandle
        self._handles: dict[str, KVCacheHandle] = {}

    def allocate(self, request_id: str) -> KVCacheHandle:
        """为请求分配一个 cache handle

        Args:
            request_id: 请求 ID

        Returns:
            新建的 KVCacheHandle
        """
        handle = KVCacheHandle(request_id=request_id)
        self._handles[request_id] = handle
        return handle

    def get(self, request_id: str) -> Optional[KVCacheHandle]:
        """获取请求对应的 cache handle

        Args:
            request_id: 请求 ID

        Returns:
            如果存在则返回 KVCacheHandle，否则返回 None
        """
        return self._handles.get(request_id)

    def update(self, request_id: str, past_key_values, token_count: int):
        """更新请求的 KV Cache 内容和覆盖 token 数

        Args:
            request_id: 请求 ID
            past_key_values: HF 模型返回的 KV Cache
            token_count: 当前 cache 覆盖的总 token 数
        """
        if request_id not in self._handles:
            self.allocate(request_id)
        self._handles[request_id].data = past_key_values
        self._handles[request_id].token_count = token_count

    def free(self, request_id: str):
        """释放请求对应的 cache handle

        Args:
            request_id: 请求 ID
        """
        if request_id in self._handles:
            del self._handles[request_id]

    def num_active_handles(self) -> int:
        """返回当前活跃 cache handle 数量"""
        return len(self._handles)
