import math
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


# ============================================================
# Configuration
# ============================================================

DATASET_DIR = LIBERO_SPATIAL_DATASET_ROOT

CHECKPOINT_DIR = (
    CHECKPOINT_ROOT
    / "full_train_norm"
)

NUM_EPOCHS = 10

RESUME_FROM = (
    "checkpoints/full_train_norm/"
    "training_step_054474.pt"
)

# 真正一次 forward 的样本数
BATCH_SIZE = 1

# 累积 8 次梯度后再 optimizer.step()
GRAD_ACCUM_STEPS = 8

LEARNING_RATE = 1e-5

# 每多少个 optimizer update 打印一次
LOG_EVERY = 50

MAX_GRAD_NORM = 1.0


def save_inference_checkpoint(
    model,
    global_step,
):
    checkpoint_path = (
        CHECKPOINT_DIR
        / f"inference_step_{global_step:06d}.pt"
    )

    torch.save(
        {
            "step": global_step,
            "model_state_dict": model.state_dict(),
            "vision_model_name":
                "facebook/dinov2-small",
            "llm_model_name":
                "Qwen/Qwen2.5-0.5B",
            "num_action_bins": 256,
        },
        checkpoint_path,
    )

    print(
        f"\n[Checkpoint] inference saved: "
        f"{checkpoint_path}\n"
    )


def save_training_checkpoint(
    model,
    optimizer,
    epoch,
    global_step,
):
    checkpoint_path = (
        CHECKPOINT_DIR
        / f"training_step_{global_step:06d}.pt"
    )

    torch.save(
        {
            "epoch": epoch,
            "step": global_step,

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

            "learning_rate":
                LEARNING_RATE,

            "grad_accum_steps":
                GRAD_ACCUM_STEPS,
        },
        checkpoint_path,
    )

    print(
        f"\n[Checkpoint] training saved: "
        f"{checkpoint_path}\n"
    )


def main():

    # ========================================================
    # 1. Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # ========================================================
    # 2. BF16
    # ========================================================

    use_bf16 = (
        device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    print("BF16:", use_bf16)

    # ========================================================
    # 3. Checkpoint directory
    # ========================================================

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 4. Model
    # ========================================================

    model = MiniVLA(
        vision_model_name="facebook/dinov2-small",
        llm_model_name="Qwen/Qwen2.5-0.5B",
        num_action_bins=256,
        freeze_vision=True,
        freeze_llm=False,
    )

    model = model.to(device)

    model.train()

    # DINO 不训练
    model.vision_encoder.vision_model.eval()

    # ========================================================
    # 5. Dataset
    # ========================================================

    dataset = LIBEROSpatialDataset(
        dataset_dir=DATASET_DIR
    )

    # ========================================================
    # 6. Collator
    # ========================================================

    collator = MiniVLACollator(
        vision_encoder=model.vision_encoder,
        tokenizer=model.llm.tokenizer,
    )

    # ========================================================
    # 7. DataLoader
    # ========================================================

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )

    # ========================================================
    # 8. Optimizer
    # ========================================================

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
    )

    # ========================================================
    # 9. Step information
    # ========================================================

    optimizer_steps_per_epoch = math.ceil(
        len(dataloader)
        / GRAD_ACCUM_STEPS
    )

    print(
        "Samples:",
        len(dataset),
    )

    print(
        "Micro batches per epoch:",
        len(dataloader),
    )

    print(
        "Gradient accumulation:",
        GRAD_ACCUM_STEPS,
    )

    print(
        "Effective batch size:",
        BATCH_SIZE * GRAD_ACCUM_STEPS,
    )

    print(
        "Optimizer steps per epoch:",
        optimizer_steps_per_epoch,
    )

        # ========================================================
    # 10. Resume
    # ========================================================

    start_epoch = 0
    global_step = 0

    optimizer.zero_grad(
        set_to_none=True
    )

    if RESUME_FROM is not None:

        print(
            "\n================================"
        )
        print(
            "Loading training checkpoint"
        )
        print(
            "================================"
        )
        print(
            RESUME_FROM
        )

        checkpoint = torch.load(
            RESUME_FROM,
            map_location="cpu",
        )

        print(
            "Checkpoint keys:",
            checkpoint.keys(),
        )

        # --------------------------------------------
        # Restore model
        # --------------------------------------------

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        print(
            "Model state restored."
        )

        # --------------------------------------------
        # Restore optimizer
        # --------------------------------------------

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        for state in optimizer.state.values():

            for key, value in state.items():

                if torch.is_tensor(value):

                    state[key] = value.to(
                        device
                    )

        print(
            "Optimizer state restored."
        )

        # --------------------------------------------
        # Restore global step
        # --------------------------------------------

        global_step = checkpoint["step"]

        # 旧checkpoint中：
        # epoch=0 表示 Epoch 1 已完成
        completed_epoch_index = (
            checkpoint["epoch"]
        )

        start_epoch = (
            completed_epoch_index + 1
        )

        print(
            "Completed epoch:",
            completed_epoch_index + 1,
        )

        print(
            "Resume from epoch:",
            start_epoch + 1,
        )

        print(
            "Global optimizer step:",
            global_step,
        )

        del checkpoint

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ========================================================
    # 11. Training
    # ========================================================

    for epoch in range(
        start_epoch,
        NUM_EPOCHS,
    ):

        print(
            f"\n========== "
            f"Epoch {epoch + 1}/{NUM_EPOCHS} "
            f"=========="
        )

        model.train()

        # DINO保持eval
        model.vision_encoder.vision_model.eval()

        # 每个epoch重新统计log
        running_loss = 0.0
        running_samples = 0

        num_micro_batches = len(
            dataloader
        )

        for batch_index, batch in enumerate(
            dataloader,
            start=1,
        ):

            pixel_values = (
                batch["pixel_values"]
                .to(device)
            )

            input_ids = (
                batch["input_ids"]
                .to(device)
            )

            text_attention_mask = (
                batch["text_attention_mask"]
                .to(device)
            )

            actions = (
                batch["actions"]
                .to(device)
            )

            # --------------------------------------------
            # Current accumulation group size
            # --------------------------------------------

            group_start = (
                ((batch_index - 1)
                 // GRAD_ACCUM_STEPS)
                * GRAD_ACCUM_STEPS
                + 1
            )

            group_end = min(
                group_start
                + GRAD_ACCUM_STEPS
                - 1,
                num_micro_batches,
            )

            current_accum_steps = (
                group_end
                - group_start
                + 1
            )

            # --------------------------------------------
            # Forward
            # --------------------------------------------

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):

                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    text_attention_mask=
                        text_attention_mask,
                    actions=actions,
                )

                raw_loss = outputs.loss

                loss = (
                    raw_loss
                    / current_accum_steps
                )

            # --------------------------------------------
            # Backward
            # --------------------------------------------

            loss.backward()

            running_loss += (
                raw_loss.item()
            )

            running_samples += 1

            # --------------------------------------------
            # Optimizer update
            # --------------------------------------------

            if batch_index == group_end:

                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    MAX_GRAD_NORM,
                )

                optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                global_step += 1

                if (
                    global_step
                    % LOG_EVERY
                    == 0
                ):

                    average_loss = (
                        running_loss
                        / running_samples
                    )

                    print(
                        f"Epoch "
                        f"{epoch + 1}/"
                        f"{NUM_EPOCHS} | "
                        f"Step "
                        f"{global_step:05d}/"
                        f"{optimizer_steps_per_epoch * NUM_EPOCHS:05d} | "
                        f"Avg Loss = "
                        f"{average_loss:.4f}"
                    )

                    running_loss = 0.0
                    running_samples = 0

        # ====================================================
        # Save after EVERY epoch
        # ====================================================

        print(
            f"\nEpoch {epoch + 1} finished."
        )

        save_inference_checkpoint(
            model=model,
            global_step=global_step,
        )

        save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
        )

    # ========================================================
    # Finished
    # ========================================================

    print(
        "\nTraining finished!"
    )

    print(
        "Final optimizer step:",
        global_step,
    )
if __name__ == "__main__":
    main()