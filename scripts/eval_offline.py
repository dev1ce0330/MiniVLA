import random

import torch

from minivla.dataset import LIBEROSpatialDataset
from minivla.model import MiniVLA
from minivla.paths import (
    LIBERO_SPATIAL_DATASET_ROOT,
)


DATASET_ROOT = LIBERO_SPATIAL_DATASET_ROOT

CHECKPOINT_PATH = (
    "checkpoints/full_train_norm/"
    "inference_step_077820.pt"
)

NUM_EVAL_SAMPLES = 500
RANDOM_SEED = 42
PRINT_EVERY = 50


@torch.no_grad()
def main():

    # ==========================================
    # 1. Device
    # ==========================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # ==========================================
    # 2. Load checkpoint
    # ==========================================

    print("\nLoading checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    print(
        "Checkpoint step:",
        checkpoint["step"],
    )

    # ==========================================
    # 3. Rebuild MiniVLA
    # ==========================================

    model = MiniVLA(
        vision_model_name=checkpoint[
            "vision_model_name"
        ],
        llm_model_name=checkpoint[
            "llm_model_name"
        ],
        num_action_bins=checkpoint[
            "num_action_bins"
        ],
        freeze_vision=True,
        freeze_llm=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    print("Model loaded!")

    # ==========================================
    # 4. Dataset
    # ==========================================

    dataset = LIBEROSpatialDataset(
        dataset_dir=DATASET_DIR
    )

    # 固定随机种子，保证每次评估同样的500条
    rng = random.Random(
        RANDOM_SEED
    )

    eval_indices = rng.sample(
        range(len(dataset)),
        k=min(
            NUM_EVAL_SAMPLES,
            len(dataset),
        ),
    )

    print(
        "\nEvaluation samples:",
        len(eval_indices),
    )

    # ==========================================
    # 5. Metrics
    # ==========================================

    total_correct_tokens = 0
    total_tokens = 0

    exact_correct = 0

    total_action_mae = 0.0

    # 每个位置预测对多少
    per_dim_correct = torch.zeros(
        7,
        dtype=torch.long,
    )

    # 保存所有 GT / prediction token
    all_gt_token_ids = []
    all_pred_token_ids = []

    # ==========================================
    # 6. Evaluate
    # ==========================================

    for eval_i, dataset_index in enumerate(
        eval_indices,
        start=1,
    ):

        sample = dataset[
            dataset_index
        ]

        image = sample["image"]
        instruction = sample[
            "instruction"
        ]

        ground_truth_action = (
            sample["action"]
            .to(device)
            .unsqueeze(0)
        )
        # [1, 7]

        # --------------------------------------
        # Image
        # --------------------------------------

        pixel_values = (
            model
            .vision_encoder
            .preprocess(image)
            .to(device)
        )

        # --------------------------------------
        # Text
        # --------------------------------------

        text_inputs = (
            model.llm.tokenize(
                instruction
            )
        )

        input_ids = (
            text_inputs[
                "input_ids"
            ]
            .to(device)
        )

        text_attention_mask = (
            text_inputs[
                "attention_mask"
            ]
            .to(device)
        )

        # --------------------------------------
        # GT tokens
        # --------------------------------------

        gt_token_ids = (
            model
            .action_tokenizer
            .encode(
                ground_truth_action
            )
        )
        # [1, 7]

        # --------------------------------------
        # Autoregressive prediction
        # --------------------------------------

        (
            predicted_action,
            predicted_token_ids,
        ) = model.predict_action(
            pixel_values=pixel_values,
            input_ids=input_ids,
            text_attention_mask=
                text_attention_mask,
        )

        # --------------------------------------
        # Token metrics
        # --------------------------------------

        token_correct = (
            predicted_token_ids
            == gt_token_ids
        )
        # [1, 7]

        total_correct_tokens += (
            token_correct
            .sum()
            .item()
        )

        total_tokens += 7

        per_dim_correct += (
            token_correct[0]
            .cpu()
            .long()
        )

        if token_correct.all():
            exact_correct += 1

        # --------------------------------------
        # Action MAE
        # --------------------------------------

        action_mae = (
            predicted_action
            - ground_truth_action
        ).abs().mean()

        total_action_mae += (
            action_mae.item()
        )

        # --------------------------------------
        # Save tokens
        # --------------------------------------

        all_gt_token_ids.append(
            gt_token_ids[0]
            .cpu()
        )

        all_pred_token_ids.append(
            predicted_token_ids[0]
            .cpu()
        )

        if eval_i % PRINT_EVERY == 0:

            print(
                f"Evaluated "
                f"{eval_i}/"
                f"{len(eval_indices)}"
            )

    # ==========================================
    # 7. Stack results
    # ==========================================

    all_gt_token_ids = torch.stack(
        all_gt_token_ids,
        dim=0,
    )
    # [N, 7]

    all_pred_token_ids = torch.stack(
        all_pred_token_ids,
        dim=0,
    )
    # [N, 7]

    num_samples = len(
        eval_indices
    )

    # ==========================================
    # 8. Overall metrics
    # ==========================================

    token_accuracy = (
        total_correct_tokens
        / total_tokens
    )

    exact_accuracy = (
        exact_correct
        / num_samples
    )

    average_action_mae = (
        total_action_mae
        / num_samples
    )

    print(
        "\n"
        "================================"
    )

    print(
        "Overall evaluation"
    )

    print(
        "================================"
    )

    print(
        f"Samples: "
        f"{num_samples}"
    )

    print(
        f"Token accuracy: "
        f"{token_accuracy:.4f}"
    )

    print(
        f"Exact action accuracy: "
        f"{exact_accuracy:.4f}"
    )

    print(
        f"Action MAE: "
        f"{average_action_mae:.6f}"
    )

    # ==========================================
    # 9. Per-dimension accuracy
    # ==========================================

    print(
        "\n"
        "================================"
    )

    print(
        "Per-dimension token accuracy"
    )

    print(
        "================================"
    )

    for dim in range(7):

        accuracy = (
            per_dim_correct[dim]
            .item()
            / num_samples
        )

        if dim < 6:
            name = f"dim_{dim}"
        else:
            name = "gripper"

        print(
            f"{name}: "
            f"{accuracy:.4f}"
        )

    # ==========================================
    # 10. ACT_128 analysis
    # ==========================================

    act_128_id = (
        model
        .action_tokenizer
        .action_token_ids[128]
    )

    print(
        "\n"
        "================================"
    )

    print(
        "ACT_128 ratio"
    )

    print(
        "================================"
    )

    for dim in range(6):

        gt_ratio = (
            all_gt_token_ids[
                :, dim
            ]
            == act_128_id
        ).float().mean().item()

        pred_ratio = (
            all_pred_token_ids[
                :, dim
            ]
            == act_128_id
        ).float().mean().item()

        print(
            f"dim_{dim}: "
            f"GT={gt_ratio * 100:6.2f}% | "
            f"Pred={pred_ratio * 100:6.2f}%"
        )

    # ==========================================
    # 11. Top predicted action bins
    # ==========================================

    print(
        "\n"
        "================================"
    )

    print(
        "Top predicted bins"
    )

    print(
        "================================"
    )

    for dim in range(6):

        predicted_ids = (
            all_pred_token_ids[
                :, dim
            ]
        )

        unique_ids, counts = (
            torch.unique(
                predicted_ids,
                return_counts=True,
            )
        )

        order = torch.argsort(
            counts,
            descending=True,
        )

        print(
            f"\n--- dim_{dim} ---"
        )

        top_k = min(
            5,
            len(order),
        )

        for rank in range(
            top_k
        ):

            index = order[rank]

            token_id = (
                unique_ids[
                    index
                ]
                .item()
            )

            count = (
                counts[
                    index
                ]
                .item()
            )

            ratio = (
                count
                / num_samples
            )

            # vocab token ID → ACT bin index
            bin_index = (
                model
                .action_tokenizer
                .id_to_bin[
                    token_id
                ]
            )

            decoded_value = (
                2.0
                * bin_index
                / 255
                - 1.0
            )

            print(
                f"ACT_{bin_index:03d} | "
                f"value≈"
                f"{decoded_value:+.4f} | "
                f"count={count:4d} | "
                f"{ratio * 100:6.2f}%"
            )

    # ==========================================
    # 12. Gripper distribution
    # ==========================================

    print(
        "\n"
        "================================"
    )

    print(
        "Gripper prediction"
    )

    print(
        "================================"
    )

    open_id = (
        model
        .action_tokenizer
        .gripper_open_id
    )

    close_id = (
        model
        .action_tokenizer
        .gripper_close_id
    )

    gt_open_ratio = (
        all_gt_token_ids[:, 6]
        == open_id
    ).float().mean().item()

    pred_open_ratio = (
        all_pred_token_ids[:, 6]
        == open_id
    ).float().mean().item()

    gt_close_ratio = (
        all_gt_token_ids[:, 6]
        == close_id
    ).float().mean().item()

    pred_close_ratio = (
        all_pred_token_ids[:, 6]
        == close_id
    ).float().mean().item()

    print(
        f"GT OPEN:   "
        f"{gt_open_ratio * 100:.2f}%"
    )

    print(
        f"Pred OPEN: "
        f"{pred_open_ratio * 100:.2f}%"
    )

    print(
        f"GT CLOSE:   "
        f"{gt_close_ratio * 100:.2f}%"
    )

    print(
        f"Pred CLOSE: "
        f"{pred_close_ratio * 100:.2f}%"
    )


if __name__ == "__main__":
    main()