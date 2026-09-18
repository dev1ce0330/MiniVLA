import torch
import torch.nn as nn

from PIL import Image
from transformers import AutoImageProcessor, AutoModel

class DINOv2VisionEncoder(nn.Module):

    def __init__(
            self,
            model_name: str = "facebook/dinov2-small",
            freeze: bool = True,
    ):
        super().__init__()

        self.model_name = model_name
        self.freeze = freeze

        self.image_processor = AutoImageProcessor.from_pretrained(
            model_name,
            use_fast = False,
        )

        self.vision_model = AutoModel.from_pretrained(
            model_name
        )

        self.embed_dim = self.vision_model.config.hidden_size
        self.patch_size = self.vision_model.config.patch_size


        if self.freeze:
            self.vision_model.requires_grad_(False)
            self.vision_model.eval()

    def preprocess(
            self,
            image: Image.Image
    ) -> torch.Tensor:

        inputs = self.image_processor(
            images = image,
            return_tensors = "pt",
        )

        pixel_values = inputs["pixel_values"]

        return pixel_values

    def forward(
            self,
            pixel_values: torch.Tensor
    ):

        outputs = self.vision_model(
            pixel_values = pixel_values
        )


        hidden_states = outputs.last_hidden_state

        patch_features = hidden_states[:, 1:, :]

        return patch_features

