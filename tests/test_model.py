import torch

from minivla.dataset import LIBEROSpatialDataset
from minivla.model import MiniVLA
from minivla.paths import LIBERO_SPATIAL_DATASET_ROOT

DATASET_ROOT = LIBERO_SPATIAL_DATASET_ROOT

def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:")
    print(device)

    dataset = LIBEROSpatialDataset(
        dataset_dir = DATASET_ROOT
    )

    sample = dataset[0]

    image = sample["image"]
    instruction = sample["instruction"]
    action = sample["action"]

    print("\nInstruction:")
    print(instruction)

    print("\nAction:")
    print(action)

    model = MiniVLA(
        vision_model_name = "facebook/dinov2-small",
        llm_model_name = "Qwen/Qwen2.5-0.5B",
        num_action_bins = 256,
        freeze_vision = True,
        freeze_llm = False,
    )

    model = model.to(device)

    pixel_values = model.vision_encoder.preprocess(
        image
    )
    
    pixel_values = pixel_values.to(device)

    print("\nPixel values shape:")
    print(pixel_values.shape)

    text_inputs = model.llm.tokenize(
        instruction
    )

    input_ids = text_inputs["input_ids"].to(device)

    text_attention_mask = text_inputs[
        "attention_mask"
    ].to(device)

    print("\nInput IDs shape:")
    print(input_ids.shape)

    print("\nText attention mask shape:")
    print(text_attention_mask.shape)

    action = action.to(device)

    print("\nAction shape:")
    print(action.shape)

    outputs = model(
        pixel_values = pixel_values,
        input_ids = input_ids,
        text_attention_mask = text_attention_mask,
        actions = action,
    )

    print("\nLoss:")
    print(outputs.loss)

    print("\nLogits shape:")
    print(outputs.logits.shape)

    model.zero_grad(set_to_none = True)

    outputs.loss.backward()

    print("\nBackward succeeded!")

    first_projector_param = next(
        model.projector.parameters()
    )

    print("\nProjector gradient mean:")
    print(
        first_projector_param.grad.abs().mean()
    )

if __name__ == "__main__":
    main()

