"""Stage5 引擎

第五阶段核心：Paged KV Cache + 自研模型前向 + Adapter 架构。
不再依赖 HF model.forward()，而是从提取的权重手写前向逻辑。
KV 直接写入预分配的物理 block，通过 block_table 映射。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import torch

from miniservellm.config import EngineConfig, ModelConfig
from miniservellm.cache.kv_cache import KVCacheManager
from miniservellm.runtime.metadata import AttentionMetadataBuilder
from miniservellm.runtime.runner import FreshPrefillRunner, IncrementalPrefillRunner, DecodeRunner
from miniservellm.runtime.sampler import Sampler
from miniservellm.scheduler.request import Request, RequestStatus, SamplingParams
from miniservellm.scheduler.request_queue import RequestQueue
from miniservellm.scheduler.scheduler import Scheduler, SchedulePlan


@dataclass
class StepEvent:
    """单步执行过程中产生的事件

    用于记录和追踪引擎在每个 step 中发生的各种操作，
    例如 prefill 完成一个 chunk、decode 生成一个 token、请求结束等。

    Attributes:
        kind: 事件类型，如 "prefill_chunk_done"、"prefill_sampled_first_token"、
              "decode_token"、"request_finished"
        request_id: 该事件所属的请求 ID
        info: 事件附加信息，不同 kind 有不同字段
    """
    kind: str
    request_id: str
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """引擎执行一步后的完整结果

    每调用一次 engine.step() 就返回一个 StepResult，
    包含调度计划、产生的事件列表以及当前 KV Cache 状态快照。

    Attributes:
        step_id: 步骤编号（递增）
        plan: 本次步骤的调度计划，包含哪些请求参与 decode/prefill
        events: 本次步骤产生的所有事件列表
        kv_state: KV Cache 管理器的调试状态快照（block 使用情况等）
    """
    step_id: int
    plan: SchedulePlan
    events: List[StepEvent] = field(default_factory=list)
    kv_state: Dict[str, Any] = field(default_factory=dict)


class Stage5Engine:
    """第五阶段推理引擎

    核心特性：
    - Paged KV Cache：KV 存入预分配 block，通过 block_table 映射
    - 自研模型前向：从 HF 模型提取权重，手写每层 forward
    - Adapter 架构：通过 ModelAdapter 接入不同模型
    - Decode-first 调度：优先执行 decode，再处理 prefill
    - Prefill→Decode 首 token 交接：避免重复采样
    """

    def __init__(
        self,
        engine_config: EngineConfig,
        model_config: ModelConfig,
        model_runner,
        tokenizer=None,
    ) -> None:
        """初始化引擎

        Args:
            engine_config: 引擎配置（batch 大小、token 预算、采样默认参数等）
            model_config: 模型配置（层数、头数、hidden_size 等结构参数）
            model_runner: 模型执行器，负责实际的前向计算，内部持有权重和 KV Cache 管理器
            tokenizer: 分词器，用于将文本转为 token IDs 或将 token IDs 解码为文本
        """
        self.engine_config = engine_config
        self.model_config = model_config
        self.model_runner = model_runner
        self.tokenizer = tokenizer

        # 请求队列：存储等待调度和正在执行的请求
        self.request_queue = RequestQueue()
        # Paged KV Cache 管理器：管理物理 block 的分配/释放和 block_table 映射
        self.kv_cache_manager = model_runner.kv_cache_manager
        # 调度器：决定每步哪些请求做 prefill、哪些做 decode；同时在调度前做 KV 容量预检查，
        # 容量不足的请求留在队列中等待后续 step 重试，而不是让 _allocate_block 抛异常退出。
        self.scheduler = Scheduler(
            queue=self.request_queue,
            max_batch_size=engine_config.max_batch_size,
            max_tokens_per_step=engine_config.max_tokens_per_step,
            max_prefill_tokens_per_step=engine_config.max_prefill_tokens_per_step,
            max_decode_requests_per_step=engine_config.max_decode_requests_per_step,
            prefill_chunk_size=engine_config.prefill_chunk_size,
            kv_cache_manager=self.kv_cache_manager,
            kv_decode_block_reserve=engine_config.kv_decode_block_reserve,
        )
        # Attention 元数据构建器：为 batched 前向计算构造 attention mask、position ids 等
        self.metadata_builder = AttentionMetadataBuilder()

        # 三种 Runner，分别处理不同的 prefill/decode 阶段：
        # - FreshPrefillRunner：从零开始 prefill（请求首次进入，没有历史 KV Cache）
        # - IncrementalPrefillRunner：增量 prefill（长 prompt 分 chunk 处理，已有部分 KV Cache）
        # - DecodeRunner：逐 token 解码（每步每个请求生成一个 token）
        self.fresh_prefill_runner = FreshPrefillRunner(self.model_runner, self.kv_cache_manager, self.metadata_builder)
        self.incremental_prefill_runner = IncrementalPrefillRunner(self.model_runner, self.kv_cache_manager, self.metadata_builder)
        self.decode_runner = DecodeRunner(self.model_runner, self.kv_cache_manager, self.metadata_builder)
        # 采样器：根据 logits 和采样参数（temperature、top_k、top_p）选择下一个 token
        self.sampler = Sampler()

        # 请求注册表：request_id → Request 的映射，用于快速查找
        self.requests_by_id: Dict[str, Request] = {}
        # 自增的请求编号，用于生成唯一 request_id
        self.next_request_idx = 0
        # 连续无事件步数：用于 deadlock breaker（临时放宽 KV 预留水位）
        self._no_progress_steps = 0
        self._kv_reserve_relax_after_no_progress_steps = max(
            1, int(engine_config.kv_reserve_relax_after_no_progress_steps)
        )

    def _new_request_id(self) -> str:
        """生成唯一的请求 ID，格式为 req_0, req_1, req_2, ..."""
        rid = f"req_{self.next_request_idx}"
        self.next_request_idx += 1
        return rid

    def add_request(
        self,
        text: Optional[str] = None,
        prompt_token_ids: Optional[List[int]] = None,
        sampling_params: Optional[SamplingParams] = None,
        max_new_tokens: int = 32,
    ) -> str:
        """向引擎添加一个推理请求

        可以传入原始文本（text）或已编码的 token IDs（prompt_token_ids），
        二者必须提供其一。如果传入文本，需要引擎已配置 tokenizer。

        对于 instruct 模型，会自动使用 chat template 包装输入，
        这样模型能正确识别对话边界并生成 EOS token。

        Args:
            text: 原始输入文本，会通过 tokenizer 编码为 token IDs
            prompt_token_ids: 已编码的 token ID 列表，跳过 tokenizer 编码
            sampling_params: 采样参数（temperature、top_k、top_p），为 None 时使用引擎默认值
            max_new_tokens: 最大生成 token 数，达到后请求结束

        Returns:
            请求 ID 字符串，后续可通过此 ID 查询请求状态和生成结果

        Raises:
            ValueError: text 和 prompt_token_ids 都为 None，或提供了 text 但没有 tokenizer
        """
        if prompt_token_ids is None:
            if text is None:
                raise ValueError("Either text or prompt_token_ids must be provided.")
            if self.tokenizer is None:
                raise ValueError("Tokenizer is required when adding request with text.")
            # 对 instruct 模型使用 chat template，使模型能正确生成 EOS
            # chat template 会添加 <|im_start|>user\n...<|im_end|><|im_start|>assistant\n 等标记
            if hasattr(self.tokenizer, 'apply_chat_template'):
                chat_text = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True,   # 添加 assistant 的开头标记
                    tokenize=False,               # 先返回文本，再单独 encode
                )
                prompt_token_ids = self.tokenizer.encode(chat_text, add_special_tokens=False)
            else:
                # 不支持 chat template 的 tokenizer，直接编码原文
                prompt_token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        if sampling_params is None:
            # 使用引擎配置中的默认采样参数
            sampling_params = SamplingParams(
                temperature=self.engine_config.default_temperature,
                top_k=self.engine_config.default_top_k,
                top_p=self.engine_config.default_top_p,
            )

        request_id = self._new_request_id()
        req = Request(
            request_id=request_id,
            prompt_token_ids=list(prompt_token_ids),
            max_new_tokens=max_new_tokens,
            sampling_params=sampling_params,
            arrival_step=self.scheduler.step_id,  # 记录请求到达时的 step 编号
        )
        self.requests_by_id[request_id] = req
        self.scheduler.add_request(req)  # 将请求加入调度器的等待队列
        return request_id

    def get_request(self, request_id: str) -> Request:
        """根据请求 ID 获取请求对象

        Args:
            request_id: 请求 ID

        Returns:
            对应的 Request 对象
        """
        return self.requests_by_id[request_id]

    def _finish_request(self, req: Request, events: List[StepEvent]) -> None:
        """结束一个请求的完整清理流程

        1. 释放该请求占用的 KV Cache 物理 block
        2. 从调度器中移除该请求
        3. 记录 request_finished 事件

        Args:
            req: 需要结束的请求
            events: 事件列表，用于追加 request_finished 事件
        """
        self.kv_cache_manager.free_request(req)  # 释放 block_table 中所有物理 block
        self.scheduler.on_request_finished(req)   # 从调度器的活跃队列中移除
        events.append(
            StepEvent(
                kind="request_finished",
                request_id=req.request_id,
                info={"finish_reason": req.finish_reason.name},
            )
        )

    def _handle_prefill_result(self, run_result, events: List[StepEvent]) -> None:
        """处理 prefill 阶段（fresh 或 incremental）的执行结果

        Prefill 有两种情况：
        - 请求的 prompt 还没处理完：本次只是处理了一个 chunk，更新进度，不采样
        - 请求的 prompt 刚好处理完（最后一个 chunk）：此时 prefill 的 logits
          对应最后一个 token 的预测，需要进行采样得到第一个生成 token

        当 prefill 完成时，请求状态会从 RUNNING_PREFILL 转为 RUNNING_DECODE，
        此时通过 sampled 字典获取首 token，并存入 pending_prefill_sample_token_id，
        供 decode 阶段使用（decode 的第一次前向需要知道这个 token）。

        Args:
            run_result: Runner 的执行结果，包含 output（含 logits）、requests、metas
            events: 事件列表，用于追加事件
        """
        # 仅最后一个 prefill chunk 会产生 logits；中间 chunk 不应执行采样或推进 RNG。
        sampled_requests = [
            req for req in run_result.requests
            if req.request_id in run_result.output.logits_by_request
        ]
        sampled = self.sampler.sample_batch(
            run_result.output.logits_by_request,
            sampled_requests,
        )

        for req, meta in zip(run_result.requests, run_result.metas):
            # 更新该请求已处理的 prompt token 数量
            chunk_size = len(meta.chunk_token_ids)
            req.num_prompt_tokens_processed += chunk_size
            req.last_update_step = self.scheduler.step_id

            events.append(
                StepEvent(
                    kind="prefill_chunk_done",
                    request_id=req.request_id,
                    info={
                        "is_fresh": meta.is_fresh,      # 是否为 fresh prefill（首次）
                        "chunk_start": meta.chunk_start, # 本 chunk 在 prompt 中的起始位置
                        "chunk_end": meta.chunk_end,     # 本 chunk 在 prompt 中的结束位置
                        "prompt_done": req.num_prompt_tokens_processed,  # 已处理的 prompt token 数
                        "prompt_total": req.total_prompt_tokens(),        # prompt 总 token 数
                    },
                )
            )

            # 记录 prefill 进度前的状态，用于判断是否刚从 prefill 转为 decode
            was_prefill = req.status == RequestStatus.RUNNING_PREFILL
            # 更新请求状态：如果 prompt 全部处理完，状态会从 RUNNING_PREFILL → RUNNING_DECODE
            self.scheduler.on_prefill_progress(req)

            # 如果请求刚从 prefill 转为 decode（即本次 chunk 是 prompt 的最后一块），
            # 则 prefill 的 logits 对应的采样结果就是第一个生成的 token
            if req.status == RequestStatus.RUNNING_DECODE and was_prefill:
                first_token = sampled[req.request_id]
                req.append_generated_token(first_token)
                # 暂存首 token，decode 阶段第一次前向需要使用此 token 的 embedding 作为输入
                req.pending_prefill_sample_token_id = first_token

                events.append(
                    StepEvent(
                        kind="prefill_sampled_first_token",
                        request_id=req.request_id,
                        info={"token_id": first_token},
                    )
                )

                # 检查首 token 是否为 EOS（虽然罕见，但理论可能）
                eos_id = self.engine_config.eos_token_id
                if eos_id is not None and first_token == eos_id:
                    req.mark_finished_eos()
                    self._finish_request(req, events)
                    continue

                # 检查是否已达到最大生成长度（max_new_tokens=1 的极端情况）
                if not req.can_decode_more():
                    req.mark_finished_max_new_tokens()
                    self._finish_request(req, events)
                    continue

    def _handle_decode_result(self, run_result, events: List[StepEvent]) -> None:
        """处理 decode 阶段的执行结果

        Decode 阶段每个请求每步生成一个 token。对每个请求：
        1. 从采样结果中获取本次生成的 token
        2. 追加到生成 token 列表
        3. 清除 pending_prefill_sample_token_id（decode 阶段不再需要）
        4. 检查是否达到 EOS 或最大生成长度

        Args:
            run_result: DecodeRunner 的执行结果，包含 output（含 logits）和 requests
            events: 事件列表，用于追加事件
        """
        # 批量采样：对 decode 请求的 logits 进行采样
        sampled = self.sampler.sample_batch(run_result.output.logits_by_request, run_result.requests)

        for req in run_result.requests:
            token_id = sampled[req.request_id]
            req.append_generated_token(token_id)
            # decode 阶段开始后，首 token 已通过 prefill 交接获得，清除暂存
            req.pending_prefill_sample_token_id = None
            req.last_update_step = self.scheduler.step_id

            events.append(
                StepEvent(
                    kind="decode_token",
                    request_id=req.request_id,
                    info={"token_id": token_id, "generated_len": req.total_generated_tokens()},
                )
            )

            # 检查是否生成了 EOS token，若是则结束请求
            eos_id = self.engine_config.eos_token_id
            if eos_id is not None and token_id == eos_id:
                req.mark_finished_eos()
                self._finish_request(req, events)
                continue

            # 检查是否已达到最大生成长度
            if not req.can_decode_more():
                req.mark_finished_max_new_tokens()
                self._finish_request(req, events)
                continue

    @torch.inference_mode()
    def step(self) -> StepResult:
        """执行引擎的一步调度和推理

        执行顺序：Decode → Incremental Prefill → Fresh Prefill
        这是 decode-first 策略：优先保证 decode 请求的延迟，
        因为 decode 每步只生成一个 token，延迟对用户体验影响大。

        另外带一个轻量 deadlock breaker：如果连续多步没有任何进展事件，
        就临时放宽一轮 prefill 的 KV 预留水位，尝试打破等待队列长期无进展。

        Returns:
            StepResult 包含本步的调度计划、事件列表和 KV Cache 状态
        """
        # 1. 调度：决定本步哪些请求做 decode、哪些做 fresh/incremental prefill
        relax_reserve = (
            self._no_progress_steps >= self._kv_reserve_relax_after_no_progress_steps
        )
        plan = self.scheduler.schedule_step(relax_kv_reserve=relax_reserve)
        events: List[StepEvent] = []

        # 2. 执行 decode（优先级最高）
        if plan.decode_requests:
            decode_result = self.decode_runner.run(plan)
            self._handle_decode_result(decode_result, events)

        # 3. 执行增量 prefill（已有部分 KV Cache 的请求继续处理）
        if plan.incremental_prefill_requests:
            incr_result = self.incremental_prefill_runner.run(plan)
            self._handle_prefill_result(incr_result, events)

        # 4. 执行 fresh prefill（首次进入的请求，从零开始处理 prompt）
        if plan.fresh_prefill_requests:
            fresh_result = self.fresh_prefill_runner.run(plan)
            self._handle_prefill_result(fresh_result, events)

        # 更新无进展计数器：有事件则归零，无事件则累加
        if events:
            self._no_progress_steps = 0
        else:
            self._no_progress_steps += 1

        return StepResult(
            step_id=plan.step_id,
            plan=plan,
            events=events,
            kv_state=self.kv_cache_manager.debug_global_state(),  # 调试用：KV Cache block 使用情况
        )

    def has_pending_work(self) -> bool:
        """是否还有未完成的请求或等待调度的请求

        Returns:
            True 表示还有工作要做，False 表示所有请求已完成
        """
        return self.scheduler.has_pending_work()

    def run_until_all_finished(self, max_steps: int = 50000, *, collect_results: bool = True) -> Any:
        """循环执行 step() 直到所有请求完成

        Args:
            max_steps: 最大执行步数，防止死循环或异常情况无限运行
            collect_results: 是否收集每一步的 StepResult。
                False 时仅运行到所有请求结束，大幅降低长 decode 的
                Python list 分配和内存。Benchmark 或纯吞吐测试设置 False。

        Returns:
            collect_results=True 时返回 List[StepResult]，否则返回 None

        Raises:
            RuntimeError: 超过 max_steps 仍未完成所有请求
        """
        if not collect_results:
            n = 0
            while self.has_pending_work():
                if n >= max_steps:
                    raise RuntimeError(f"Exceeded max_steps={max_steps}")
                self.step()
                n += 1
            return None

        results = []
        n = 0
        while self.has_pending_work():
            if n >= max_steps:
                raise RuntimeError(f"Exceeded max_steps={max_steps}")
            results.append(self.step())
            n += 1
        return results

    def get_text(self, request_id: str) -> str:
        """获取请求的生成文本（仅 decode 生成的部分，不含 prompt）

        Args:
            request_id: 请求 ID

        Returns:
            解码后的生成文本；若无 tokenizer 则返回 token ID 列表的字符串表示
        """
        req = self.get_request(request_id)
        if self.tokenizer is None:
            return str(req.generated_token_ids)
        return self.tokenizer.decode(req.generated_token_ids, skip_special_tokens=True)

    def get_full_text(self, request_id: str) -> str:
        """获取请求的完整文本（prompt + 生成部分）

        Args:
            request_id: 请求 ID

        Returns:
            解码后的完整文本（包含 prompt 和生成内容）；若无 tokenizer 则返回全部 token ID 列表
        """
        req = self.get_request(request_id)
        if self.tokenizer is None:
            return str(req.all_token_ids())
        return self.tokenizer.decode(req.all_token_ids(), skip_special_tokens=True)

    def debug_request(self, request_id: str):
        """获取请求的调试信息字典

        用于开发和调试时查看请求的完整内部状态。

        Args:
            request_id: 请求 ID

        Returns:
            包含请求状态、进度、block_table、生成 token 等信息的字典
        """
        req = self.get_request(request_id)
        return {
            "request_id": req.request_id,
            "status": req.status.name,
            "finish_reason": req.finish_reason.name,
            "prompt_total": req.total_prompt_tokens(),
            "prompt_done": req.num_prompt_tokens_processed,
            "generated": req.total_generated_tokens(),
            "block_table": list(req.block_table),           # 该请求占用的物理 block 编号列表
            "total_slots_reserved": req.total_slots_reserved, # 已预留的 KV Cache slot 总数
            "generated_token_ids": list(req.generated_token_ids),
            "pending_prefill_sample_token_id": req.pending_prefill_sample_token_id,  # prefill 交接的首 token
        }
