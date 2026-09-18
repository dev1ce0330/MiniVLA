from pathlib import Path
import torch
from torch.utils.data import DataLoader

from minivla.dataset import LIBEROSpatialDataset
from minivla.model import MiniVLA
from minivla.collator import MiniVLACollator
from minivla.paths import (
    LIBERO_SPATIAL_DATASET_ROOT,
    CHECKPOINT_ROOT,
)


DATASET_ROOT = LIBERO_SPATIAL_DATASET_ROOT

BATCH_SIZE = 1
LEARNING_RATE = 1e-5
MAX_STEPS = 100

CHECKPOINT_DIR = (
    CHECKPOINT_ROOT
    / "tiny_overfit"
)
SAVE_EVERY = 100

def main():

    CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
    )
    
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)


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
        dataset_dir = DATASET_DIR
    )

    collator = MiniVLACollator(
        vision_encoder=model.vision_encoder,
        tokenizer=model.llm.tokenizer,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
    )

    step = 0
    running_loss = 0.0

    for batch in dataloader:

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

        optimizer.zero_grad()

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            text_attention_mask=text_attention_mask,
            actions=actions,
        )

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        step += 1

        if step % SAVE_EVERY == 0:

            checkpoint_path = (
                CHECKPOINT_DIR
                / f"step_{step:06d}.pt"
            )

            torch.save(
                {
                    "step": step,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "vision_model_name":
                        "facebook/dinov2-small",

                    "llm_model_name":
                        "Qwen/Qwen2.5-0.5B",

                    "num_action_bins":
                        256,
                },
                checkpoint_path,
            )

            print(
                f"\nCheckpoint saved to: "
                f"{checkpoint_path}\n"
            )

        loss_value = loss.item()

        running_loss += loss_value

        print(
            f"Step {step:03d} | "
            f"Loss = {loss_value:.4f}"
        )

        if step % 10 == 0:

            average_loss = (
                running_loss / 10
            )

            print(
                f"Average loss "
                f"[{step - 9}-{step}] "
                f"= {average_loss:.4f}"
            )

            running_loss = 0.0

        if step >= MAX_STEPS:
            break

    print("\nTraining finished!")


if __name__ == "__main__":
    main()