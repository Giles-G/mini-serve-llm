"""推理引擎

第二阶段的 InferenceEngine 不再只是单请求 generate()，而是支持：
- add_request(): 持续接收请求
- step(): 每次推进一批请求
- run_until_complete(): 跑到所有请求完成

当前版本的 batch 是调度层 batch，执行层仍逐 request forward。
"""

from miniservellm.runtime.outputs import StepResult


class InferenceEngine:
    """最小 serving engine

    Attributes:
        tokenizer_adapter: tokenizer 适配器，用于 token <-> text
        prefill_executor: prefill 执行器
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

        会先为请求分配 KV Cache handle，再加入 waiting queue。
        """
        self.kv_cache_manager.allocate(request.request_id)
        self.request_queue.add_request(request)

    def has_pending(self) -> bool:
        """是否仍有未完成请求"""
        return self.request_queue.has_pending()

    def step(self) -> list[StepResult]:
        """推进引擎一步

        一次 step 会：
        1. 调用 scheduler 生成 BatchPlan
        2. 逐个执行 plan.prefill_requests
        3. 逐个执行 plan.decode_requests
        4. 将未完成请求重新放回 decode queue
        5. 返回本步每个请求生成的 token 文本增量

        Returns:
            本步生成结果列表
        """
        plan = self.scheduler.schedule()
        results: list[StepResult] = []

        # 执行 prefill 请求：每个请求会产出首 token 并初始化 KV Cache
        for req in plan.prefill_requests:
            next_token_id = self.prefill_executor.run(req)

            finished = (
                len(req.generated_token_ids) >= req.sampling_params.max_new_tokens
                or next_token_id in req.sampling_params.stop_token_ids
            )

            if finished:
                # 如果首 token 就结束，直接标记完成并释放 cache
                req.finish_time = req.first_token_time
                self.request_queue.mark_finished(req)
                self.kv_cache_manager.free(req.request_id)
            else:
                # 未完成则进入 decode queue，后续每轮继续生成
                self.request_queue.requeue_for_decode(req)

            results.append(
                StepResult(
                    request_id=req.request_id,
                    next_token_id=next_token_id,
                    finished=finished,
                    text_delta=self.tokenizer_adapter.decode([next_token_id]),
                )
            )

        # 执行 decode 请求：每个请求推进一个 token
        for req in plan.decode_requests:
            next_token_id, finished = self.decode_executor.step(req)

            if finished:
                self.request_queue.mark_finished(req)
                self.kv_cache_manager.free(req.request_id)
            else:
                self.request_queue.requeue_for_decode(req)

            results.append(
                StepResult(
                    request_id=req.request_id,
                    next_token_id=next_token_id,
                    finished=finished,
                    text_delta=self.tokenizer_adapter.decode([next_token_id]),
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

    def generate(self, request) -> str:
        """兼容第一阶段的单请求 generate() API

        Args:
            request: 单个 Request

        Returns:
            生成文本
        """
        self.add_request(request)
        finished_requests = self.run_until_complete()
        final_request = finished_requests[-1]
        return self.tokenizer_adapter.decode(final_request.generated_token_ids)
