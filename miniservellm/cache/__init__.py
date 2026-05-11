"""KV Cache 管理模块

第三阶段引入 BlockAllocator 和 PagedKVCacheHandle：
- BlockAllocator 管理 block 资源的分配、扩容和回收
- KVCacheManager 通过 BlockAllocator 管理 request 级别的 KV Cache 生命周期
- PagedKVCacheHandle 记录 request 对应的 block_ids 和 token_count
"""
