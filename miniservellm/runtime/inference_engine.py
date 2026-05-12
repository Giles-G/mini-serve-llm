"""Hybrid Inference Engine

第四阶段核心：根据请求是否带有 past_key_values，
将 prefill 请求拆分成 fresh 和 incremental 两条路径：
- fresh prefill（无 cache）：batched forward
- incremental prefill（有 cache）：逐请求 forward
- decode：逐请求 forward

step() 流程：
1. 把 waiting 请求转成 PREFILLING
2. Scheduler 输出本轮计划（decode-first）
3. 先执行 decode
4. 再把 prefill 请求拆成 fresh / incremental
5. 执行 batched fresh prefill
6. 执行 per-request incremental prefill
7. 更新状态
8. 输出事件
"""

from __future__ import annotations

from miniservellm.runtime.outputs import StepEvent, StepEventType


class HybridInferenceEngine:
    """混合推理引擎

    根据请求的 cache 状态选择不同的执行路径：
    - fresh prefill → batched forward（真正 batch）
    - incremental prefill → 逐请求 forward
    - decode → 逐请求 forward

    Attributes:
        tokenizer_adapter: tokenizer 适配器
        model_runner: 模型运行器
        sampler: 采样器
        queue: 请求队列
        scheduler: 调度器
        fresh_prefill_executor: batched fresh prefill 执行器
        incremental_prefill_executor: incremental prefill 执行器
        decode_executor: decode 执行器
        request_index: request_id -> Request 映射
    """

    def __init__(
        self,
        tokenizer_adapter,
        model_runner,
        sampler,
        queue,
        scheduler,
        fresh_prefill_executor,
        incremental_prefill_executor,
        decode_executor,
    ):
        self.tokenizer_adapter = tokenizer_adapter
        self.model_runner = model_runner
        self.sampler = sampler
        self.queue = queue
        self.scheduler = scheduler
        self.fresh_prefill_executor = fresh_prefill_executor
        self.incremental_prefill_executor = incremental_prefill_executor
        self.decode_executor = decode_executor
        self.request_index: dict = {}

    def add_request(self, request):
        """添加新请求"""
        self.request_index[request.request_id] = request
        self.queue.add_new_request(request)

    def has_pending(self) -> bool:
        """是否仍有未完成请求"""
        return self.queue.has_pending()

    def step(self) -> list[StepEvent]:
        """推进引擎一步

        流程：
        1. 提升 waiting 请求为 prefill 状态
        2. 获取调度计划
        3. 执行 decode（逐请求）
        4. 将 prefill 请求拆成 fresh / incremental
        5. 执行 batched fresh prefill
        6. 执行 per-request incremental prefill
        7. 清理已完成的请求

        Returns:
            StepEvent 列表
        """
        # 1) 提升 waiting 请求
        self.queue.promote_waiting_to_prefill()

        # 2) 获取调度计划
        plan = self.scheduler.schedule(self.queue)

        all_events: list[StepEvent] = []

        # 3) 执行 decode（逐请求）
        for req in plan.decode_requests:
            events = self.decode_executor.run_one(req)
            all_events.extend(events)

        # 4) 将 prefill 请求拆成 fresh / incremental
        fresh_prefills = []
        incremental_prefills = []
        incremental_chunk_sizes = {}

        for req in plan.prefill_requests:
            chunk_size = plan.prefill_chunk_sizes.get(req.request_id, 0)
            if chunk_size <= 0:
                continue

            if req.past_key_values is None:
                # fresh prefill：没有历史 cache，可以 batch
                fresh_prefills.append(req)
            else:
                # incremental prefill：有历史 cache，逐请求执行
                incremental_prefills.append(req)
                incremental_chunk_sizes[req.request_id] = chunk_size

        # 5) 执行 batched fresh prefill
        if fresh_prefills:
            fresh_chunk_sizes = {
                req.request_id: plan.prefill_chunk_sizes.get(req.request_id, 0)
                for req in fresh_prefills
            }
            events = self.fresh_prefill_executor.run(fresh_prefills, fresh_chunk_sizes)
            all_events.extend(events)

        # 6) 执行 per-request incremental prefill
        for req in incremental_prefills:
            chunk_size = incremental_chunk_sizes.get(req.request_id, 0)
            if chunk_size <= 0:
                continue
            events = self.incremental_prefill_executor.run_one(req, chunk_size)
            all_events.extend(events)

        # 7) 将完成 prefill 的请求从 active_prefill 移到 active_decode
        #    执行器会直接设置 req.status = "decoding"，但请求仍在 active_prefill 列表中
        #    需要手动迁移
        still_prefilling = []
        for req in self.queue.active_prefill:
            if req.status == "decoding":
                self.queue.active_decode.append(req)
            elif req.status == "finished":
                pass  # 已完成，直接丢弃
            else:
                still_prefilling.append(req)
        self.queue.active_prefill = still_prefilling

        # 8) 清理已完成请求
        self.queue.remove_finished()

        return all_events

    def run_until_complete(self) -> list:
        """持续 step，直到所有请求完成"""
        while self.has_pending():
            self.step()
        return [
            req for req in self.request_index.values()
            if req.status == "finished"
        ]

    def generate(self, request) -> str:
        """兼容单请求 generate() API"""
        self.add_request(request)
        finished_requests = self.run_until_complete()
        final_request = finished_requests[-1]
        return self.tokenizer_adapter.decode(final_request.generated_token_ids)
