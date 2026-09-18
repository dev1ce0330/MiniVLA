import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForCausalLM

class QwenBackbone(nn.Module):

    def __init__(
            self,
            model_name: str = "Qwen/Qwen2.5-0.5B",
            freeze: bool = False,
    ):
        super().__init__()

        self.model_name = model_name
        self.freeze = freeze

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name
        )

        self.hidden_size = self.model.config.hidden_size

        if self.freeze:
            self.model.requires_grad_(False)
            self.model.eval()

    def tokenize(
            self,
            text: str,
    ):
        inputs = self.tokenizer(
            text,
            return_tensors = "pt",
        )

        return inputs

    def embed_tokens(
            self,
            input_ids : torch.Tensor,
    ) -> torch.Tensor:

        embedding_layer = self.model.get_input_embeddings()

        token_embeddings = embedding_layer(
            input_ids
        )

        return token_embeddings