import torch


class DecodeExecutor:
    def __init__(self, model_runner, sampler):
        self.model_runner = model_runner
        self.sampler = sampler

    def step(self, request):
        input_ids = torch.tensor([[request.last_token_id]], dtype=torch.long)

        # attention_mask 长度必须等于 past_key_values 的序列长度 + 当前 token
        prompt_len = len(request.prompt_token_ids)
        generated_len = len(request.generated_token_ids)
        total_len = prompt_len + generated_len
        attention_mask = torch.ones(1, total_len, dtype=torch.long)

        output = self.model_runner.forward_decode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=request.past_key_values,
        )

        next_token_id = self.sampler.sample(
            output.logits[:, -1, :],
            temperature=request.sampling_params.temperature,
            top_k=request.sampling_params.top_k,
            top_p=request.sampling_params.top_p,
        )

        request.past_key_values = output.past_key_values
        request.last_token_id = next_token_id
        request.generated_token_ids.append(next_token_id)

        finished = (
            len(request.generated_token_ids) >= request.sampling_params.max_new_tokens
            or next_token_id in request.sampling_params.stop_token_ids
        )

        return next_token_id, finished
