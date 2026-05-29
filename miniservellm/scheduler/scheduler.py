"""调度器

第五阶段：decode-first + token budget 调度，输出 SchedulePlan。
包含 SchedulePlan 和 Scheduler。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from miniservellm.cache.kv_cache import KVCacheManager
from miniservellm.scheduler.request import Request, RequestStatus


@dataclass
class SchedulePlan:
    """单步调度计划

    由 Scheduler.schedule_step() 生成，描述本步需要执行哪些请求以及如何分类。
    引擎根据此计划分别调用不同的 Runner 执行。

    Attributes:
        step_id: 步骤编号，每次 schedule_step() 调用递增
        fresh_prefill_requests: 首次进入 prefill 的请求列表
            这些请求刚从 WAITING 状态转为 RUNNING_PREFILL，还没有任何 KV Cache。
            会由 FreshPrefillRunner 处理。
        incremental_prefill_requests: 增量 prefill 的请求列表
            这些请求已经过上一轮 prefill，有部分 KV Cache，本轮继续处理剩余 prompt。
            会由 IncrementalPrefillRunner 处理。
        decode_requests: decode 请求列表
            这些请求的 prompt 已全部处理完，正在逐 token 生成。
            会由 DecodeRunner 处理。
        prefill_chunks: request_id → 本步分配的 chunk token 数
            对于 fresh 和 incremental prefill 请求，记录本轮分配给该请求的 token 数。
            chunk 大小受 prefill_chunk_size、剩余 prompt 长度、token budget 三者约束。
    """

    step_id: int
    fresh_prefill_requests: list[Request] = field(default_factory=list)
    incremental_prefill_requests: list[Request] = field(default_factory=list)
    decode_requests: list[Request] = field(default_factory=list)
    prefill_chunks: dict[str, int] = field(default_factory=dict)


class Scheduler:
    """请求调度器

    负责在每个 step 决定哪些请求执行 prefill、哪些执行 decode，
    以及每个 prefill 请求分配多少 token 的预算。

    调度策略：
    - decode-first：decode 请求优先调度，因为 decode 每步只生成一个 token，
      延迟对用户体验影响最大，不能被 prefill 阻塞。
    - token budget：prefill 有独立的 token 预算（max_prefill_tokens_per_step），
      防止 prefill 占用过多算力导致 decode 饥饿。
    - chunked prefill：长 prompt 不会一次性处理完，而是按 prefill_chunk_size 分块，
      允许 prefill 和 decode 在同一步内交替执行。

    四种预算约束：
    - budget_batch：总 batch 大小限制（decode + prefill 请求总数）
    - budget_tokens：每步总 token 数限制（decode 1 token/req + prefill chunk tokens）
    - budget_prefill：prefill token 预算（只扣减 prefill 请求的 chunk 大小）
    - budget_decode：decode 请求数限制

    Attributes:
        queue: 请求队列（RequestQueue），维护 waiting/running_prefill/running_decode/finished 四个列表
        max_batch_size: 单步最大参与请求总数（decode + prefill）
        max_tokens_per_step: 单步最大 token 总数（decode 每个 1 token + prefill chunk tokens）
        max_prefill_tokens_per_step: 单步 prefill 可消耗的最大 token 数
        max_decode_requests_per_step: 单步最大 decode 请求数
        prefill_chunk_size: 单个请求每步最多 prefill 的 token 数（chunk 上限）
        step_id: 当前步骤编号，每次 schedule_step() 后递增
    """

    def __init__(
        self,
        queue,
        max_batch_size: int,
        max_tokens_per_step: int,
        max_prefill_tokens_per_step: int,
        max_decode_requests_per_step: int,
        prefill_chunk_size: int,
        kv_cache_manager: Optional[KVCacheManager] = None,
        kv_decode_block_reserve: int = 2,
    ) -> None:
        """初始化调度器

        Args:
            queue: 请求队列实例
            max_batch_size: 单步最大参与请求总数
            max_tokens_per_step: 单步最大 token 总数
            max_prefill_tokens_per_step: 单步 prefill token 预算
            max_decode_requests_per_step: 单步最大 decode 请求数
            prefill_chunk_size: 单个请求每步 prefill chunk 大小上限
            kv_cache_manager: KV Cache 管理器；用于调度前做容量预检查，
                容量不足时请求留在原队列等待后续 step 重试。
            kv_decode_block_reserve: 预留给 decode 增长的最少空闲 block 水位。
                prefill 调度时必须保留该水位，避免 prefill 抢光 KV 导致全局卡死。
        """
        self.queue = queue
        self.kv_cache_manager = kv_cache_manager
        self.kv_decode_block_reserve = max(0, kv_decode_block_reserve)
        self.max_batch_size = max_batch_size
        self.max_tokens_per_step = max_tokens_per_step
        self.max_prefill_tokens_per_step = max_prefill_tokens_per_step
        self.max_decode_requests_per_step = max_decode_requests_per_step
        self.prefill_chunk_size = prefill_chunk_size
        self.step_id = 0

    def add_request(self, req: Request) -> None:
        """将新请求加入调度器的等待队列

        请求状态设为 WAITING，加入 queue.waiting 列表，
        等待后续 schedule_step() 中被调度为 fresh prefill。

        Args:
            req: 新请求对象
        """
        req.status = RequestStatus.WAITING
        self.queue.add_waiting(req)

    def on_prefill_progress(self, req: Request) -> None:
        """通知调度器某个请求的 prefill 取得了进展

        在引擎处理完一个 prefill chunk 后调用。如果该请求的 prompt 已全部处理完
        （remaining_prompt_tokens() == 0），则将其从 running_prefill 移入 running_decode。

        注意：一个请求可能在多步内完成 prefill（chunked prefill），每次处理完一个 chunk
        后都需要调用此方法。只有最后一个 chunk 处理完时才会触发状态转换。

        Args:
            req: 取得 prefill 进展的请求
        """
        if req.remaining_prompt_tokens() == 0:
            # prompt 全部处理完，从 RUNNING_PREFILL → RUNNING_DECODE
            req.status = RequestStatus.RUNNING_DECODE
            self.queue.remove_running_prefill(req)
            if req.can_decode_more():
                # 正常情况：请求进入 decode 阶段
                self.queue.running_decode.append(req)
            else:
                # 边界情况：max_new_tokens=0 或已达上限，直接结束
                req.mark_finished_max_new_tokens()
                self.queue.finished.append(req)

    def on_request_finished(self, req: Request) -> None:
        """通知调度器某个请求已完成（EOS 或达到最大生成长度）

        从所有可能所在的队列中移除该请求，然后加入 finished 列表。
        之所以尝试从所有队列中移除，是因为请求可能在任何阶段结束
        （例如 prefill 首 token 就是 EOS）。

        Args:
            req: 已完成的请求
        """
        self.queue.remove_waiting(req)
        self.queue.remove_running_prefill(req)
        self.queue.remove_running_decode(req)
        if req.finish_reason.name != "NONE":
            self.queue.finished.append(req)

    def schedule_step(self, relax_kv_reserve: bool = False) -> SchedulePlan:
        """生成本步的调度计划

        按以下优先级依次分配预算：
        1. decode first：优先调度 running_decode 中的请求
           - 每个请求消耗 1 token 预算、1 decode 预算、1 batch 预算
        2. incremental prefill：调度 running_prefill 中已有部分 KV Cache 的请求
           - chunk 大小 = min(剩余 prompt, prefill_chunk_size, prefill 预算, 总 token 预算)
           - 每个 chunk 消耗对应 token 数的 prefill 预算和总 token 预算
        3. fresh prefill：从 waiting 队列中取出新请求
           - 请求状态从 WAITING → RUNNING_PREFILL，移入 running_prefill
           - chunk 大小计算同 incremental prefill

        Args:
            relax_kv_reserve: 是否临时放宽 prefill 的 KV 预留水位。
                False（默认）时执行正常水位保护；True 时用于 deadlock breaker，
                允许在单步里忽略 prefill 水位，尝试打破长期无进展状态。

        Returns:
            SchedulePlan 包含本步各类请求列表和 prefill chunk 分配
        """
        plan = SchedulePlan(step_id=self.step_id)
        self.step_id += 1

        # 初始化本步的四种预算
        budget_tokens = self.max_tokens_per_step          # 总 token 预算
        budget_prefill = self.max_prefill_tokens_per_step  # prefill token 预算
        budget_decode = self.max_decode_requests_per_step  # decode 请求数预算
        budget_batch = self.max_batch_size                 # 总 batch 预算
        # KV block 预算也必须按本 step 已纳入计划的请求递减。
        # 不能只对每个请求单独 can_allocate，否则多个请求都看到同一批 free blocks，
        # 最终 aggregate 需求可能超过真实空闲量，Runner 里仍会触发 _allocate_block 异常。
        budget_kv_blocks = (
            self.kv_cache_manager.num_free_blocks()
            if self.kv_cache_manager is not None
            else 0
        )
        kv_prefill_reserve = 0 if relax_kv_reserve else self.kv_decode_block_reserve

        # ---- 阶段 1：decode first ----
        # decode 每个请求每步生成 1 个 token，优先保证 decode 不被 prefill 阻塞。
        # 如果 KV block 不够，跳过该请求，留在 running_decode 队列中等待后续 step 重试；
        # 不能在 KVCache 内阻塞等待，否则释放 block 的后续 step 永远无法发生。
        for req in list(self.queue.running_decode):
            if budget_batch <= 0 or budget_decode <= 0 or budget_tokens <= 0:
                break
            if req.status != RequestStatus.RUNNING_DECODE:
                continue
            if not req.can_decode_more():
                continue
            need_blocks = 0
            if self.kv_cache_manager is not None:
                need_blocks = self.kv_cache_manager.needed_new_blocks(req, 1)
                if need_blocks > budget_kv_blocks:
                    continue
            plan.decode_requests.append(req)
            budget_batch -= 1   # 占一个 batch 槽位
            budget_decode -= 1  # 占一个 decode 槽位
            budget_tokens -= 1  # decode 每步消耗 1 个 token
            budget_kv_blocks -= need_blocks

        # ---- 阶段 2：incremental prefill ----
        # 处理已有部分 KV Cache 的请求，继续处理其剩余 prompt。
        # 容量不足时保留在 running_prefill 队列中，下轮等其它请求释放 block 后再重试。
        for req in list(self.queue.running_prefill):
            if budget_batch <= 0 or budget_prefill <= 0 or budget_tokens <= 0:
                break
            remain = req.remaining_prompt_tokens()
            if remain <= 0:
                continue
            # chunk 大小受四个约束：剩余 prompt 长度、chunk 大小上限、prefill 预算、总 token 预算
            chunk = min(remain, self.prefill_chunk_size, budget_prefill, budget_tokens)
            if chunk <= 0:
                continue
            need_blocks = 0
            if self.kv_cache_manager is not None:
                need_blocks = self.kv_cache_manager.needed_new_blocks(req, chunk)
                # prefill 需要遵守 decode 预留水位，避免 prefill 抢光 KV 导致全局卡死。
                if need_blocks > budget_kv_blocks - kv_prefill_reserve:
                    continue
            plan.incremental_prefill_requests.append(req)
            plan.prefill_chunks[req.request_id] = chunk
            budget_batch -= 1       # 占一个 batch 槽位
            budget_prefill -= chunk  # 消耗 chunk 大小的 prefill 预算
            budget_tokens -= chunk   # 消耗 chunk 大小的总 token 预算
            budget_kv_blocks -= need_blocks

        # ---- 阶段 3：fresh prefill ----
        # 从等待队列中取出新请求，首次开始 prefill。
        # 容量不足时不要改变状态，也不要移出 waiting；这样就是非阻塞排队。
        for req in list(self.queue.waiting):
            if budget_batch <= 0 or budget_prefill <= 0 or budget_tokens <= 0:
                break
            remain = req.remaining_prompt_tokens()
            if remain <= 0:
                continue
            chunk = min(remain, self.prefill_chunk_size, budget_prefill, budget_tokens)
            if chunk <= 0:
                continue
            need_blocks = 0
            if self.kv_cache_manager is not None:
                need_blocks = self.kv_cache_manager.needed_new_blocks(req, chunk)
                # prefill 需要遵守 decode 预留水位，避免 prefill 抢光 KV 导致全局卡死。
                if need_blocks > budget_kv_blocks - kv_prefill_reserve:
                    continue
            # 状态转换：WAITING → RUNNING_PREFILL，并移入 running_prefill 队列
            req.status = RequestStatus.RUNNING_PREFILL
            self.queue.remove_waiting(req)
            self.queue.running_prefill.append(req)

            plan.fresh_prefill_requests.append(req)
            plan.prefill_chunks[req.request_id] = chunk
            budget_batch -= 1
            budget_prefill -= chunk
            budget_tokens -= chunk
            budget_kv_blocks -= need_blocks

        return plan

    def has_pending_work(self) -> bool:
        """是否还有未完成的工作（等待调度或正在执行的请求）

        Returns:
            True 表示还有待处理的工作
        """
        return self.queue.has_pending_work()
