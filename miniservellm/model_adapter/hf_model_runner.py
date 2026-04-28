import torch
from miniservellm.types import ModelForwardOutput


class HFModelRunner:
    def __init__(self, model, device: str):
        self.model = model
        self.device = device

    @torch.no_grad()
    def forward_prefill(self, input_ids, attention_mask=None, past_key_values=None):
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

        return ModelForwardOutput(
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
        )

    @torch.no_grad()
    def forward_decode(self, input_ids, attention_mask=None, past_key_values=None):
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

        return ModelForwardOutput(
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
        )
