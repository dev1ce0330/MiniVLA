import torch

from torch.utils.data import DataLoader

from minivla.dataset import LIBEROSpatialDataset
from minivla.model import MiniVLA
from minivla.collator import MiniVLACollator
from minivla.paths import (
    LIBERO_SPATIAL_DATASET_ROOT,
)

DATASET_ROOT = LIBERO_SPATIAL_DATASET_ROOT

def main():

    device =torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:")
    print(device)

    model = MiniVLA(
        vision_model_name="facebook/dinov2-small",
        llm_model_name="Qwen/Qwen2.5-0.5B",
        num_action_bins=256,
        freeze_vision=True,
        freeze_llm=False,
    )

    model = model.to(device)

    model.train()

    model.vision_encoder.vision_model.eval()

    dataset = LIBEROSpatialDataset(
        dataset_dir = DATASET_ROOT
    )

    collator = MiniVLACollator(
        vision_encoder = model.vision_encoder,
        tokenizer = model.llm.tokenizer,
    )

    dataloader = DataLoader(
        dataset,
        batch_size = 1,
        shuffle = True,
        collate_fn = collator,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = 1e-5,
    )

    batch = next(
        iter(dataloader)
    )

    pixel_values = batch[
        "pixel_values"
    ].to(device)

    input_ids = batch[
        "input_ids"
    ].to(device)

    text_attention_mask = batch[
        "text_attention_mask"
    ].to(device)

    actions = batch[
        "actions"
    ].to(device)

    first_projector_param = next(
        model.projector.parameters()
    )

    param_before = (
        first_projector_param
        .detach()
        .clone()
    )

    optimizer.zero_grad()

    outputs = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        text_attention_mask=text_attention_mask,
        actions=actions,
    )

    loss = outputs.loss

    print("\nLoss before update:")
    print(loss)

    loss.backward()

    print("\nProjector gradient mean:")

    print(
        first_projector_param
        .grad
        .abs()
        .mean()
    )

    optimizer.step()

    param_after = (
        first_projector_param
        .detach()
        .clone()
    )

    parameter_change = (
        param_after - param_before
    ).abs().mean()

    print("\nProjector parameter change:")
    print(parameter_change)

    if parameter_change > 0:
        print("\nTraining step succeeded!")
    else:
        print("\nWARNING: parameters did not change.")


if __name__ == "__main__":
    main()