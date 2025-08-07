import torch.nn as nn
from torch import Tensor

class Embedder(nn.Module):
    def __init__(self, tokenizer, encoder, max_length: int, output_key: str):
        super().__init__()
        self.max_length = max_length
        self.output_key = output_key

        self.tokenizer = tokenizer
        self.encoder = encoder.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = self.encoder(
            input_ids=batch_encoding["input_ids"].to(self.encoder.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]

class ImageEmbedder(nn.Module):
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        
    def forward(self, image: Tensor) -> Tensor:
        redux_output = self.pipeline(image)
        clip_embed = redux_output["pooled_prompt_embeds"]
        t5_embed = redux_output["prompt_embeds"]

        return clip_embed, t5_embed