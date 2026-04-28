import time
import torch


class PrefillExecutor:
    def __init__(self, model_runner, sampler):
        self.model_runner = model_runner
        self.sampler = sampler

    def run(self, request):
        input_ids = torch.tensor([request.prompt_token_ids], dtype=torch.long)

        output = self.model_runner.forward_prefill(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            past_key_values=None,
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

        if request.first_token_time is None:
            request.first_token_time = time.time()

        return next_token_id
