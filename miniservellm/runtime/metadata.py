"""Attention Metadata

第五阶段新增：为 prefill/decode 请求构建 metadata，
包含 slot 引用、位置信息等，供 Paged KV Cache 和模型前向使用。
"""

from __future__ import annotations

from dataclasses import dataclass

from miniservellm.scheduler.request import Request


@dataclass
class SlotRef:
    """Paged KV Cache 中的一个 slot 引用

    Attributes:
        block_id: block 编号
        block_offset: block 内偏移
        logical_pos: 逻辑位置（请求视角的 token 序号）
    """

    block_id: int
    block_offset: int
    logical_pos: int


@dataclass
class PrefillRequestMetadata:
    """Prefill 请求的 metadata

    Attributes:
        request_id: 请求 ID
        is_fresh: 是否为 fresh prefill（无历史 cache）
        chunk_start: chunk 在 prompt 中的起始位置
        chunk_end: chunk 在 prompt 中的结束位置
        chunk_token_ids: chunk 的 token ids
        context_len_before_chunk: chunk 之前的上下文长度
        write_slots: 写入 KV Cache 的 slot 引用列表
        positions: 位置编码列表
    """

    request_id: str
    is_fresh: bool
    chunk_start: int
    chunk_end: int
    chunk_token_ids: list[int]
    context_len_before_chunk: int
    write_slots: list[SlotRef]
    positions: list[int]


@dataclass
class DecodeRequestMetadata:
    """Decode 请求的 metadata

    Attributes:
        request_id: 请求 ID
        input_token_id: 输入 token id
        query_position: 查询位置
        context_len: 上下文长度
        write_slot: 写入 KV Cache 的 slot 引用
    """

    request_id: str
    input_token_id: int
    query_position: int
    context_len: int
    write_slot: SlotRef


class AttentionMetadataBuilder:
    """构建 prefill/decode 请求的 metadata"""

    def build_prefill_metadata(
        self,
        req: Request,
        chunk_size: int,
        write_slots: list[SlotRef],
        is_fresh: bool,
    ) -> PrefillRequestMetadata:
        start = req.num_prompt_tokens_processed
        end = start + chunk_size
        token_ids = req.prompt_token_ids[start:end]
        positions = list(range(start, end))
        # context_len_before_chunk 是当前 chunk 之前已分配的 slot 数，
        # 即 total_slots_reserved 减去刚分配的 chunk slot 数。
        # （ensure_slots_for_request 已经将 total_slots_reserved 增加了 chunk_size）
        context_len_before_chunk = req.total_slots_reserved - len(write_slots)
        return PrefillRequestMetadata(
            request_id=req.request_id,
            is_fresh=is_fresh,
            chunk_start=start,
            chunk_end=end,
            chunk_token_ids=token_ids,
            context_len_before_chunk=context_len_before_chunk,
            write_slots=write_slots,
            positions=positions,
        )

    def build_decode_metadata(
        self,
        req: Request,
        write_slot: SlotRef,
    ) -> DecodeRequestMetadata:
        input_token_id = req.pending_prefill_sample_token_id
        if input_token_id is None:
            input_token_id = req.last_token_id_for_decode_input()

        # query_position 是当前 decode token 的逻辑位置
        # 等于已分配的 slot 数 - 1（最后分配的 slot 就是给这个 token 的）
        query_position = req.total_slots_reserved - 1
        # context_len 是包括当前 token 在内的总上下文长度
        context_len = req.total_slots_reserved
        return DecodeRequestMetadata(
            request_id=req.request_id,
            input_token_id=input_token_id,
            query_position=query_position,
            context_len=context_len,
            write_slot=write_slot,
        )
