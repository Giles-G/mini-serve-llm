"""KV Cache 管理模块

第二阶段开始将 KV Cache 从 Request 中抽象出来，
方便后续替换为 paged KV cache、block allocator 等真实 serving 结构。
"""
