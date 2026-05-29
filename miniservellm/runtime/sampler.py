"""采样器

第六阶段重构：sample_batch 完全向量化。
N 个请求的 logits 堆叠为 [N, V]，temperature/repetition_penalty/top-k/top-p/multinomial
均在 [N, V] 维度上一次完成，避免 Python for 循环。
"""

from __future__ import annotations

from typing import Dict, List

import torch

from miniservellm.scheduler.request import Request


def top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """单行 top-k 过滤（兼容旧接口）。

    将 logits 中除前 top_k 个最大值之外的位置置为 -inf。
    """
    if top_k <= 0 or top_k >= logits.numel():
        return logits
    values, _ = torch.topk(logits, top_k)
    threshold = values[-1]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """单行 top-p（nucleus）过滤（兼容旧接口）。"""
    if top_p >= 1.0:
        return logits
    if top_p <= 0.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits.float(), dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    keep = cumulative <= top_p
    keep[0] = True
    filtered_sorted = sorted_logits.masked_fill(~keep, float("-inf"))
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(0, sorted_indices, filtered_sorted)
    return filtered


class Sampler:
    """Token 采样器

    支持 greedy、temperature、top-k、top-p、repetition_penalty 采样。
    """

    def sample_one(self, logits: torch.Tensor, req: Request) -> int:
        """单请求采样（保留作为兼容接口）。"""
        params = req.sampling_params
        temperature = float(params.temperature)
        top_k = int(params.top_k)
        top_p = float(params.top_p)
        repetition_penalty = float(params.repetition_penalty)

        if temperature <= 0.0:
            return int(torch.argmax(logits).item())

        work = logits
        if temperature != 1.0:
            work = work / temperature

        if repetition_penalty != 1.0 and req.generated_token_ids:
            seen = set(req.generated_token_ids)
            for token_id in seen:
                if token_id < work.numel():
                    if work[token_id] > 0:
                        work[token_id] /= repetition_penalty
                    else:
                        work[token_id] *= repetition_penalty

        if top_k > 0:
            work = top_k_filtering(work, top_k)
        if top_p < 1.0:
            work = top_p_filtering(work, top_p)

        probs = torch.softmax(work.float(), dim=-1)
        token = torch.multinomial(probs, 1)
        return int(token.item())

    def sample_batch(
        self,
        logits_by_request: Dict[str, torch.Tensor],
        requests: List[Request],
    ) -> Dict[str, int]:
        """N 个请求的批量采样（向量化）。

        实现要点：
        1. 把所有请求的 logits 堆成 [N, V]，所有逐 token 算子一次完成
        2. 不同请求的 sampling_params 通过 [N, 1] 张量广播
        3. greedy（temperature<=0）通过 mask 与 argmax 结果合并
        4. repetition_penalty 仅在「确实需要」时才构造 [N, V] mask
        """
        if not requests:
            return {}

        N = len(requests)
        # 堆叠为 [N, V]
        logits = torch.stack([logits_by_request[r.request_id] for r in requests], dim=0)
        V = logits.shape[-1]
        device = logits.device

        # 收集每个请求的采样参数到张量
        temps = torch.tensor(
            [float(r.sampling_params.temperature) for r in requests],
            device=device, dtype=torch.float32,
        )
        top_ks = [int(r.sampling_params.top_k) for r in requests]
        top_ps = torch.tensor(
            [float(r.sampling_params.top_p) for r in requests],
            device=device, dtype=torch.float32,
        )
        rps = torch.tensor(
            [float(r.sampling_params.repetition_penalty) for r in requests],
            device=device, dtype=torch.float32,
        )

        greedy_mask = temps <= 0.0  # [N]
        # 任何模式下都算 argmax，便于在最后用 greedy_mask 选择
        greedy_tokens = torch.argmax(logits, dim=-1)  # [N]

        # Temperature 缩放：greedy 行用 1.0 占位避免除零（结果会被 greedy_tokens 覆盖）
        safe_temps = torch.where(greedy_mask, torch.ones_like(temps), temps)
        work = logits / safe_temps.unsqueeze(-1)  # [N, V]

        # Repetition penalty：只在需要时构造 [N, V] seen mask
        rp_indices = [
            i for i, r in enumerate(requests)
            if float(r.sampling_params.repetition_penalty) != 1.0 and r.generated_token_ids
        ]
        if rp_indices:
            seen_mask = torch.zeros((N, V), device=device, dtype=torch.bool)
            for i in rp_indices:
                ids = list({tid for tid in requests[i].generated_token_ids if tid < V})
                if ids:
                    idx_t = torch.tensor(ids, device=device, dtype=torch.long)
                    seen_mask[i, idx_t] = True
            rp_col = rps.unsqueeze(-1)  # [N, 1]
            positive = work > 0
            # 正值：除以 rp（降低概率）；非正值：乘以 rp（让负的更负）
            work = torch.where(seen_mask & positive, work / rp_col, work)
            work = torch.where(seen_mask & ~positive, work * rp_col, work)

        # Top-k 过滤：以本批次最大 top_k 做一次 topk，再按行阈值掩码
        valid_top_ks = [k for k in top_ks if 0 < k < V]
        if valid_top_ks:
            max_top_k = max(valid_top_ks)
            topk_vals, _ = torch.topk(work, max_top_k, dim=-1)  # [N, max_top_k]
            # 每行阈值：top_ks[i]>0 取 topk_vals[i, k-1]，否则 -inf（不过滤）
            thresholds = torch.full((N,), float("-inf"), device=device)
            for i, k in enumerate(top_ks):
                if 0 < k < V:
                    thresholds[i] = topk_vals[i, k - 1]
            mask = work >= thresholds.unsqueeze(-1)
            work = torch.where(mask, work, torch.full_like(work, float("-inf")))

        # Top-p 过滤：批量 sort + cumsum，按行 top_p 阈值
        if (top_ps < 1.0).any().item():
            sorted_logits, sorted_indices = torch.sort(work, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits.float(), dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            keep = cumulative <= top_ps.unsqueeze(-1)
            keep[:, 0] = True
            # top_p>=1.0 的行全部保留
            full_keep = (top_ps >= 1.0).unsqueeze(-1)
            keep = keep | full_keep
            filtered_sorted = sorted_logits.masked_fill(~keep, float("-inf"))
            work = torch.full_like(work, float("-inf"))
            work.scatter_(1, sorted_indices, filtered_sorted.to(work.dtype))

        # 批量 multinomial 采样
        probs = torch.softmax(work.float(), dim=-1)
        sampled = torch.multinomial(probs, 1).squeeze(-1)  # [N]

        # greedy 行用 argmax 结果覆盖
        final = torch.where(greedy_mask, greedy_tokens, sampled)

        # CPU 同步一次后再返回（避免 N 次 .item() 同步）
        final_list = final.tolist()
        return {requests[i].request_id: int(final_list[i]) for i in range(N)}
