"""执行器

第五阶段新增：FreshPrefillRunner / IncrementalPrefillRunner / DecodeRunner，
负责为每个请求分配 KV slot、构建 metadata、调用 model runner。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from miniservellm.cache.kv_cache import KVCacheManager
from miniservellm.runtime.metadata import AttentionMetadataBuilder, PrefillRequestMetadata, DecodeRequestMetadata
from miniservellm.runtime.model_interface import PrefillModelOutput, DecodeModelOutput
from miniservellm.scheduler.request import Request
from miniservellm.scheduler.scheduler import SchedulePlan


@dataclass
class PrefillRunResult:
    requests: List[Request]
    metas: List[PrefillRequestMetadata]
    output: PrefillModelOutput


@dataclass
class DecodeRunResult:
    requests: List[Request]
    metas: List[DecodeRequestMetadata]
    output: DecodeModelOutput


class FreshPrefillRunner:
    """Fresh Prefill 执行器

    处理新到达的 prefill 请求（无历史 cache），
    为每个请求分配 KV slot 并构建 metadata。
    """

    def __init__(self, model, kv_cache_manager: KVCacheManager, metadata_builder: AttentionMetadataBuilder):
        self.model = model
        self.kv_cache_manager = kv_cache_manager
        self.metadata_builder = metadata_builder

    def run(self, plan: SchedulePlan) -> PrefillRunResult:
        requests = []
        metas = []
        for req in plan.fresh_prefill_requests:
            chunk_size = plan.prefill_chunks[req.request_id]
            kv_cache_write_slots = self.kv_cache_manager.ensure_slots_for_request(req, chunk_size)
            meta = self.metadata_builder.build_prefill_metadata(req, chunk_size, kv_cache_write_slots, is_fresh=True)
            requests.append(req)
            metas.append(meta)
        output = self.model.forward_fresh_prefill(requests, metas)
        return PrefillRunResult(requests=requests, metas=metas, output=output)


class IncrementalPrefillRunner:
    """Incremental Prefill 执行器

    处理继续中的 prefill 请求（有历史 cache），
    为每个请求分配新 KV slot 并构建 metadata。
    """

    def __init__(self, model, kv_cache_manager: KVCacheManager, metadata_builder: AttentionMetadataBuilder):
        self.model = model
        self.kv_cache_manager = kv_cache_manager
        self.metadata_builder = metadata_builder

    def run(self, plan: SchedulePlan) -> PrefillRunResult:
        requests = []
        metas = []
        for req in plan.incremental_prefill_requests:
            chunk_size = plan.prefill_chunks[req.request_id]
            kv_cache_write_slots = self.kv_cache_manager.ensure_slots_for_request(req, chunk_size)
            meta = self.metadata_builder.build_prefill_metadata(req, chunk_size, kv_cache_write_slots, is_fresh=False)
            requests.append(req)
            metas.append(meta)
        output = self.model.forward_incremental_prefill(requests, metas)
        return PrefillRunResult(requests=requests, metas=metas, output=output)


class DecodeRunner:
    """Decode 执行器，支持 CUDA Graph 加速（batch=1 greedy）。

    为每个 decode 请求分配 1 个 KV slot 并构建 metadata。
    当 batch=1 且模型已启用 CUDA Graph 时，通过 graph replay 执行。
    """

    def __init__(self, model, kv_cache_manager: KVCacheManager, metadata_builder: AttentionMetadataBuilder):
        self.model = model
        self.kv_cache_manager = kv_cache_manager
        self.metadata_builder = metadata_builder

    def run(self, plan: SchedulePlan) -> DecodeRunResult:
        requests = []
        metas = []
        for req in plan.decode_requests:
            kv_cache_write_slots = self.kv_cache_manager.ensure_slots_for_request(req, 1)
            meta = self.metadata_builder.build_decode_metadata(req, kv_cache_write_slots[0])
            requests.append(req)
            metas.append(meta)

        # C1: CUDA Graph fast path for batch=1 greedy decode
        if len(requests) == 1:
            from miniservellm.runtime.nn_ops import _HAS_CUSTOM_KERNELS
            if _HAS_CUSTOM_KERNELS:
                req = requests[0]
                meta = metas[0]
                token_id = meta.input_token_id
                ctx_len = meta.context_len
                block_table = self.kv_cache_manager.build_decode_block_table([req], ctx_len)
                # Lazy capture on first call
                if not self.model.has_cuda_graph:
                    self.model.enable_cuda_graph(token_id, ctx_len + 1, block_table)
                logits = self.model.cuda_graph_step(token_id, ctx_len + 1, block_table)
                output = DecodeModelOutput(logits_by_request={req.request_id: logits})
                return DecodeRunResult(requests=requests, metas=metas, output=output)

        output = self.model.forward_decode(requests, metas)
        return DecodeRunResult(requests=requests, metas=metas, output=output)
