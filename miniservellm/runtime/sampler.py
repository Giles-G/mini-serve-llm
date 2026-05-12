"""采样器

从模型输出的 logits 中采样下一个 token，支持 greedy、top-k 等策略。
"""

import torch


class Sampler:
    """Token 采样器

    支持 greedy decoding 和带温度的 top-k 采样。
    后续可扩展 top-p（nucleus）、repetition penalty 等。
    """

    def sample(
        self,
        logits,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> int:
        """从 logits 中采样一个 token

        Args:
            logits: 模型输出的 logits，形状 [vocab_size] 或 [1, vocab_size]
            temperature: 采样温度，0.0 表示贪心解码
            top_k: top-k 采样的候选数，0 表示不限制
            top_p: nucleus sampling 阈值（当前未实现）

        Returns:
            采样得到的 token id
        """
        # 统一为 1D
        if logits.dim() > 1:
            logits = logits.squeeze(0)

        # Greedy decoding：直接取 argmax
        if temperature == 0.0:
            return torch.argmax(logits).item()

        # 带温度的 softmax 概率分布
        probs = torch.softmax(logits / temperature, dim=-1)

        # Top-k 采样：只从概率最高的 k 个 token 中采样
        if top_k > 0:
            values, indices = torch.topk(probs, top_k, dim=-1)
            # 重新归一化，使 top-k 候选概率之和为 1
            values = values / values.sum(dim=-1, keepdim=True)
            sampled = torch.multinomial(values, num_samples=1).item()
            return indices[sampled].item()

        # 无 top-k 约束，直接从全词表分布采样
        sampled = torch.multinomial(probs, num_samples=1).item()
        return sampled
