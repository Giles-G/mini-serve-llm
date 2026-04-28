import torch


class Sampler:
    def sample(
        self,
        logits,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> int:
        """
        logits: [1, vocab_size]
        第一阶段先实现 greedy + 简单 top-k
        """
        if temperature == 0.0:
            return torch.argmax(logits, dim=-1).item()

        probs = torch.softmax(logits / temperature, dim=-1)

        if top_k > 0:
            values, indices = torch.topk(probs, top_k, dim=-1)
            values = values / values.sum(dim=-1, keepdim=True)
            sampled = torch.multinomial(values[0], num_samples=1).item()
            return indices[0, sampled].item()

        sampled = torch.multinomial(probs[0], num_samples=1).item()
        return sampled
