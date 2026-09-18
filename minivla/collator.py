import torch

class MiniVLACollator:

    def __init__(
            self,
            vision_encoder,
            tokenizer,
    ):
        self.vision_encoder = vision_encoder
        self.tokenizer = tokenizer

    def __call__(
            self,
            samples,
    ):

        images = [
            sample["image"]
            for sample in samples
        ]

        instructions = [
            sample["instruction"]
            for sample in samples
        ]

        actions = torch.stack(
            [
                sample["action"]
                for sample in samples
            ],
            dim = 0,
        )

        image_inputs = self.vision_encoder.image_processor(
            images = images,
            return_tensors = "pt",
        )

        pixel_values = image_inputs[
            "pixel_values"
        ]

        text_inputs = self.tokenizer(
            instructions,
            padding = True,
            return_tensors = "pt",
        )

        input_ids = text_inputs[
            "input_ids"
        ]

        text_attention_mask = text_inputs[
            "attention_mask"
        ]

        return{
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "text_attention_mask": text_attention_mask,
            "actions": actions, 
        }