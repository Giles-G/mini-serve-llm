"""采样器

第五阶段重构：新增 top-p 采样、batch 采样接口。
"""

from __future__ import annotations

from typing import Dict, List

import torch

from miniservellm.scheduler.request import Request


def top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.numel():
        return logits
    values, _ = torch.topk(logits, top_k)
    threshold = values[-1]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
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

    支持 greedy、temperature、top-k、top-p 采样。
    """

    def sample_one(self, logits: torch.Tensor, req: Request) -> int:
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

        # Apply repetition penalty based on already generated tokens
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

    def sample_batch(self, logits_by_request: Dict[str, torch.Tensor], requests: List[Request]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for req in requests:
            out[req.request_id] = self.sample_one(logits_by_request[req.request_id], req)
        return out
