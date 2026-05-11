"""推理引擎

第三阶段的 InferenceEngine 支持完整的 chunked prefill 生命周期：
- add_request(): 持续接收请求
- step(): 每次推进一批请求，产出带 event_type 的 StepResult
- run_until_complete(): 跑到所有请求完成

step() 中对 prefill 请求根据 chunked prefill 结果区分不同事件类型：
- PREFILL_PROGRESS: chunk 进行中，未产出 token
- PREFILL_TO_DECODE: prefill 完成，产出首 token
- DECODE_TOKEN: decode 阶段产出一个 token
- FINISHED: 请求完成
"""

from __future__ import annotations

from miniservellm.runtime.outputs import StepResult


class InferenceEngine:
    """最小 serving engine

    Attributes:
        tokenizer_adapter: tokenizer 适配器，用于 token <-> text
        prefill_executor: chunked prefill 执行器
        decode_executor: decode 执行器
        request_queue: 请求队列
        scheduler: 调度器
        kv_cache_manager: KV Cache 管理器
    """

    def __init__(
        self,
        tokenizer_adapter,
        prefill_executor,
        decode_executor,
        request_queue,
        scheduler,
        kv_cache_manager,
    ):
        self.tokenizer_adapter = tokenizer_adapter
        self.prefill_executor = prefill_executor
        self.decode_executor = decode_executor
        self.request_queue = request_queue
        self.scheduler = scheduler
        self.kv_cache_manager = kv_cache_manager

    def add_request(self, request):
        """添加一个新请求到引擎

        会先为请求分配 KV Cache handle 和 block，再加入 waiting queue。
        """
        self.kv_cache_manager.allocate(request.request_id)
        self.request_queue.add_request(request)

    def has_pending(self) -> bool:
        """是否仍有未完成请求"""
        return self.request_queue.has_pending()

    def step(self) -> list[StepResult]:
        """推进引擎一步

        一次 step 会：
        1. 调用 scheduler 生成 BatchPlan（含 prefill chunk 分配）
        2. 逐个执行 plan.prefill_requests 的 chunked prefill
        3. 逐个执行 plan.decode_requests 的 decode step
        4. 根据执行结果将请求放入正确的队列
        5. 返回本步每个请求的 StepResult（含 event_type 和 metadata）

        Returns:
            本步生成结果列表
        """
        plan = self.scheduler.schedule()
        results: list[StepResult] = []

        # --- 1) 执行 prefill 请求 ---
        for req in plan.prefill_requests:
            # 获取调度器为本请求分配的 chunk token 预算
            assigned_chunk_size = plan.prefill_chunk_sizes.get(req.request_id, 0)

            chunk_result = self.prefill_executor.run_chunk(
                req,
                chunk_size=assigned_chunk_size,
            )

            produced_token = chunk_result.produced_token
            finished = chunk_result.finished
            prefill_done = chunk_result.prefill_done
            chunk_processed_tokens = chunk_result.chunk_processed_tokens

            if produced_token is None:
                # 未产出 token：prefill 仍在进行中
                if prefill_done:
                    # prefill 完成但未产出 token（不应出现，但做防御性处理）
                    self.request_queue.requeue_for_decode(req)
                    event_type = "PREFILL_TO_DECODE"
                    metadata = {
                        "chunk_processed_tokens": chunk_processed_tokens,
                        "prefill_cursor": req.prefill_cursor,
                        "prefill_done": req.prefill_done,
                        "assigned_chunk_size": assigned_chunk_size,
                        "queue_action": "requeue_decode",
                    }
                else:
                    # prefill 还在继续，放回 prefilling 队列
                    self.request_queue.requeue_for_prefill(req)
                    event_type = "PREFILL_PROGRESS"
                    metadata = {
                        "chunk_processed_tokens": chunk_processed_tokens,
                        "prefill_cursor": req.prefill_cursor,
                        "prefill_done": req.prefill_done,
                        "assigned_chunk_size": assigned_chunk_size,
                        "queue_action": "requeue_prefill",
                    }

                results.append(
                    StepResult(
                        request_id=req.request_id,
                        next_token_id=None,
                        finished=False,
                        text_delta="",
                        event_type=event_type,
                        metadata=metadata,
                    )
                )
            else:
                # 产出了首 token：prefill 完成
                if finished:
                    # 首 token 就满足终止条件
                    self.request_queue.mark_finished(req)
                    self.kv_cache_manager.free(req.request_id)
                    event_type = "FINISHED"
                    queue_action = "mark_finished"
                else:
                    # 进入 decode 阶段
                    self.request_queue.requeue_for_decode(req)
                    event_type = "PREFILL_TO_DECODE"
                    queue_action = "requeue_decode"

                results.append(
                    StepResult(
                        request_id=req.request_id,
                        next_token_id=produced_token,
                        finished=finished,
                        text_delta=self.tokenizer_adapter.decode([produced_token]),
                        event_type=event_type,
                        metadata={
                            "chunk_processed_tokens": chunk_processed_tokens,
                            "prefill_cursor": req.prefill_cursor,
                            "prefill_done": req.prefill_done,
                            "assigned_chunk_size": assigned_chunk_size,
                            "queue_action": queue_action,
                        },
                    )
                )

        # --- 2) 执行 decode 请求 ---
        for req in plan.decode_requests:
            decode_result = self.decode_executor.step(req)
            next_token_id = decode_result.next_token_id
            finished = decode_result.finished

            if finished:
                self.request_queue.mark_finished(req)
                self.kv_cache_manager.free(req.request_id)
                event_type = "FINISHED"
                queue_action = "mark_finished"
            else:
                self.request_queue.requeue_for_decode(req)
                event_type = "DECODE_TOKEN"
                queue_action = "requeue_decode"

            results.append(
                StepResult(
                    request_id=req.request_id,
                    next_token_id=next_token_id,
                    finished=finished,
                    text_delta=self.tokenizer_adapter.decode([next_token_id]),
                    event_type=event_type,
                    metadata={
                        "prefill_cursor": req.prefill_cursor,
                        "generated_tokens": len(req.generated_token_ids),
                        "queue_action": queue_action,
                    },
                )
            )

        return results

    def run_until_complete(self) -> list:
        """持续 step，直到所有请求完成

        Returns:
            所有完成的 Request 列表
        """
        while self.has_pending():
            self.step()
        return self.request_queue.all_finished_requests()
