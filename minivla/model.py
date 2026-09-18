import torch
import torch.nn as nn

from minivla.vision_encoder import DINOv2VisionEncoder
from minivla.llm_backbone import QwenBackbone
from minivla.projector import MLPProjector
from minivla.action_tokenizer import ActionTokenizer

class MiniVLA(nn.Module):

    def __init__(
        self,
        vision_model_name: str = "facebook/dinov2-small",
        llm_model_name: str = "Qwen/Qwen2.5-0.5B",
        num_action_bins: int =256,
        freeze_vision: bool = True,
        freeze_llm: bool = False,
    ):

        super().__init__()

        self.vision_encoder = DINOv2VisionEncoder(
            model_name = vision_model_name,
            freeze = freeze_vision,
        )

        self.llm = QwenBackbone(
            model_name = llm_model_name,
            freeze = freeze_llm,
        )

        action_q01 = [
            -0.760714,
            -0.656250,
            -0.937500,
            -0.108214,
            -0.204643,
            -0.186429,
        ]

        action_q99 = [
            0.937500,
            0.873214,
            0.934821,
            0.105000,
            0.175714,
            0.143571,
        ]

        self.action_tokenizer = ActionTokenizer(
            tokenizer=self.llm.tokenizer,
            num_bins=num_action_bins,
            action_q01=action_q01,
            action_q99=action_q99,
        )

        self.llm.model.resize_token_embeddings(
            len(self.llm.tokenizer)
        )

        self.projector = MLPProjector(
            input_dim = self.vision_encoder.embed_dim,
            output_dim = self.llm.hidden_size,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        actions: torch.Tensor,
    ):

        if self.vision_encoder.freeze:
            with torch.no_grad():
                vision_features = self.vision_encoder(
                    pixel_values
                )

        else:
            vision_features = self.vision_encoder(
                pixel_values
            )

        visual_embeddings = self.projector(
            vision_features
        )

        text_embeddings = self.llm.embed_tokens(
            input_ids
        )

        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        action_token_ids = self.action_tokenizer.encode(
            actions
        )

        action_embeddings = self.llm.embed_tokens(
            action_token_ids
        )

        inputs_embeds = torch.cat(
            [
                visual_embeddings,
                text_embeddings,
                action_embeddings,
            ],
            dim = 1,
        )

        batch_size = input_ids.shape[0]

        num_vision_tokens = visual_embeddings.shape[1]
        num_action_tokens = action_embeddings.shape[1]

        vision_attention_mask = torch.ones(
            (
                batch_size,
                num_vision_tokens,
            ),
            dtype = text_attention_mask.dtype,
            device = text_attention_mask.device,
        )

        action_attention_mask = torch.ones(
            (
                batch_size,
                num_action_tokens,
            ),
            dtype = text_attention_mask.dtype,
            device = text_attention_mask.device,
        )

        attention_mask = torch.cat(
            [
                vision_attention_mask,
                text_attention_mask,
                action_attention_mask,
            ],
            dim = 1,
        )

        vision_labels = torch.full(
            (
                batch_size,
                num_vision_tokens,
            ),
            fill_value = -100,
            dtype = torch.long,
            device = input_ids.device,
        )

        text_labels = torch.full(
            input_ids.shape,
            fill_value = -100,
            dtype = torch.long,
            device = input_ids.device,
        )

        labels = torch.cat(
            [
                vision_labels,
                text_labels,
                action_token_ids,
            ],
            dim = 1,
        )

        outputs = self.llm.model(
            inputs_embeds = inputs_embeds,
            attention_mask = attention_mask,
            labels = labels,
            use_cache = False,
        )

        return outputs

    @torch.no_grad()
    def predict_action(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ):

        vision_features = self.vision_encoder(
            pixel_values
        )

        visual_embeddings = self.projector(
            vision_features
        )

        text_embeddings = self.llm.embed_tokens(
            input_ids
        )

        batch_size = input_ids.shape[0]

        generated_action_token_ids = None

        for action_index in range(7):

            if generated_action_token_ids is None:

                inputs_embeds = torch.cat(
                    [
                        visual_embeddings,
                        text_embeddings,
                    ],
                    dim = 1,
                )

            else:

                action_embeddings = (
                    self.llm.embed_tokens(
                        generated_action_token_ids
                    )
                )

                inputs_embeds = torch.cat(
                    [
                        visual_embeddings,
                        text_embeddings,
                        action_embeddings,
                    ],
                    dim = 1,
                )

            num_vision_tokens = (
                visual_embeddings.shape[1]
            )

            vision_attention_mask = torch.ones(
                (
                    batch_size,
                    num_vision_tokens,
                ),
                dtype=text_attention_mask.dtype,
                device=text_attention_mask.device,
            )

            if generated_action_token_ids is None:

                attention_mask = torch.cat(
                    [
                        vision_attention_mask,
                        text_attention_mask,
                    ],
                    dim=1,
                )

            else:

                action_attention_mask = torch.ones(
                    generated_action_token_ids.shape,
                    dtype=text_attention_mask.dtype,
                    device=text_attention_mask.device,
                )

                attention_mask = torch.cat(
                    [
                        vision_attention_mask,
                        text_attention_mask,
                        action_attention_mask,
                    ],
                    dim=1,
                )

            outputs = self.llm.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            )

            next_token_logits = (
                outputs.logits[:, -1, :]
            )

            if action_index < 6:

                allowed_token_ids = torch.tensor(
                    self.action_tokenizer.action_token_ids,
                    dtype=torch.long,
                    device=next_token_logits.device,
                )

            else:

                allowed_token_ids = torch.tensor(
                    [
                        self.action_tokenizer.gripper_open_id,
                        self.action_tokenizer.gripper_close_id,
                    ],
                    dtype=torch.long,
                    device=next_token_logits.device,
                )

            allowed_logits = (
                next_token_logits[
                    :,
                    allowed_token_ids,
                ]
            )

            best_allowed_index = (
                allowed_logits.argmax(
                    dim=-1
                )
            )

            next_token_id = (
                allowed_token_ids[
                    best_allowed_index
                ]
            )

            next_token_id = (
                next_token_id.unsqueeze(-1)
            )

            if generated_action_token_ids is None:

                generated_action_token_ids = (
                    next_token_id
                )

            else:

                generated_action_token_ids = (
                    torch.cat(
                        [
                            generated_action_token_ids,
                            next_token_id,
                        ],
                        dim=1,
                    )
                )

        action = self.action_tokenizer.decode(
            generated_action_token_ids
        )

        return (
            action,
            generated_action_token_ids,
        )    