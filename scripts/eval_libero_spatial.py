import os

os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path
import json
import csv

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from minivla.model import MiniVLA


# ============================================================
# Configuration
# ============================================================

CHECKPOINT_PATH = (
    "checkpoints/full_train_norm/"
    "inference_step_077820.pt"
)

TASK_SUITE_NAME = "libero_spatial"

NUM_TASKS = 10

# 每个 task 测几个 initial states
NUM_INIT_STATES = 5

MAX_STEPS = 220

NUM_WAIT_STEPS = 10

SAVE_VIDEOS = True

OUTPUT_DIR = Path(
    "rollouts/libero_spatial_eval"
)

VIDEO_DIR = (
    OUTPUT_DIR / "videos"
)

RESULT_JSON = (
    OUTPUT_DIR / "results.json"
)

SUMMARY_CSV = (
    OUTPUT_DIR / "summary.csv"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

VIDEO_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Image preprocessing
# ============================================================

def get_model_image(obs):

    # --------------------------------------------------------
    # IMPORTANT
    #
    # Our alignment experiment showed:
    #
    # simulator raw agentview_image
    #     MAE ≈ 2
    #
    # rotated image
    #     MAE ≈ 70
    #
    # Therefore DO NOT rotate.
    # --------------------------------------------------------

    image = obs[
        "agentview_image"
    ]

    return np.ascontiguousarray(
        image
    )


# ============================================================
# MiniVLA inference
# ============================================================

@torch.no_grad()
def predict_action(
    model,
    image_np,
    instruction,
    device,
):

    image = Image.fromarray(
        image_np
    )

    # --------------------------------------------------------
    # Vision preprocessing
    # --------------------------------------------------------

    pixel_values = (
        model
        .vision_encoder
        .preprocess(
            image
        )
        .to(device)
    )

    # --------------------------------------------------------
    # Language
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Autoregressive action generation
    # --------------------------------------------------------

    (
        action,
        action_token_ids,
    ) = model.predict_action(
        pixel_values=
            pixel_values,

        input_ids=
            input_ids,

        text_attention_mask=
            text_attention_mask,
    )

    action = (
        action[0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    action_token_ids = (
        action_token_ids[0]
        .detach()
        .cpu()
    )

    return (
        action,
        action_token_ids,
    )


# ============================================================
# Save results
# ============================================================

def save_results(results):

    with open(
        RESULT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# One rollout
# ============================================================

@torch.no_grad()
def run_rollout(
    env,
    model,
    initial_state,
    instruction,
    task_id,
    init_id,
    device,
):

    print(
        "\n"
        "================================"
    )

    print(
        f"Task {task_id:02d} "
        f"| Init {init_id:02d}"
    )

    print(
        instruction
    )

    print(
        "================================"
    )

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    env.reset()

    obs = env.set_init_state(
        initial_state
    )

    # --------------------------------------------------------
    # Wait for simulator to stabilize
    # --------------------------------------------------------

    dummy_action = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.0,
        ],
        dtype=np.float32,
    )

    for _ in range(
        NUM_WAIT_STEPS
    ):

        (
            obs,
            reward,
            done,
            info,
        ) = env.step(
            dummy_action
        )

    # --------------------------------------------------------
    # Rollout buffers
    # --------------------------------------------------------

    frames = []

    actions = []

    saturated_count = (
        np.zeros(
            6,
            dtype=np.int64,
        )
    )

    success = False

    success_step = None

    # --------------------------------------------------------
    # Boundary action token IDs
    # --------------------------------------------------------

    act_min_id = (
        model
        .action_tokenizer
        .action_token_ids[0]
    )

    act_max_id = (
        model
        .action_tokenizer
        .action_token_ids[-1]
    )

    # ========================================================
    # Closed-loop rollout
    # ========================================================

    for step in range(
        MAX_STEPS
    ):

        # ----------------------------------------------------
        # Current visual observation
        # ----------------------------------------------------

        image_np = get_model_image(
            obs
        )

        if SAVE_VIDEOS:

            frames.append(
                image_np.copy()
            )

        # ----------------------------------------------------
        # MiniVLA predicts next action
        # ----------------------------------------------------

        (
            action,
            token_ids,
        ) = predict_action(
            model=
                model,

            image_np=
                image_np,

            instruction=
                instruction,

            device=
                device,
        )

        actions.append(
            action.copy()
        )

        # ----------------------------------------------------
        # Boundary diagnostics
        # ----------------------------------------------------

        token_list = (
            token_ids.tolist()
        )

        for dim in range(6):

            if (
                token_list[dim]
                == act_min_id
                or
                token_list[dim]
                == act_max_id
            ):

                saturated_count[
                    dim
                ] += 1

        # ----------------------------------------------------
        # Execute predicted action
        # ----------------------------------------------------

        (
            obs,
            reward,
            done,
            info,
        ) = env.step(
            action.tolist()
        )

        # ----------------------------------------------------
        # LIBERO success
        # ----------------------------------------------------

        current_success = (
            env.check_success()
        )

        if current_success:

            success = True

            success_step = step

            print(
                f"SUCCESS at step "
                f"{step}"
            )

            break

    # ========================================================
    # Diagnostics
    # ========================================================

    actions = np.asarray(
        actions,
        dtype=np.float32,
    )

    num_executed_steps = len(
        actions
    )

    if num_executed_steps > 0:

        action_min = (
            actions.min(
                axis=0
            )
        )

        action_max = (
            actions.max(
                axis=0
            )
        )

        action_mean = (
            actions.mean(
                axis=0
            )
        )

        boundary_ratio = (
            saturated_count
            / num_executed_steps
        )

    else:

        action_min = (
            np.zeros(
                7,
                dtype=np.float32,
            )
        )

        action_max = (
            np.zeros(
                7,
                dtype=np.float32,
            )
        )

        action_mean = (
            np.zeros(
                7,
                dtype=np.float32,
            )
        )

        boundary_ratio = (
            np.zeros(
                6,
                dtype=np.float32,
            )
        )

    # ========================================================
    # Video
    # ========================================================

    video_path = None

    if SAVE_VIDEOS:

        video_path = (
            VIDEO_DIR
            / (
                f"task_{task_id:02d}"
                f"_init_{init_id:02d}"
                f"_success_{success}.mp4"
            )
        )

        imageio.mimsave(
            video_path,
            frames,
            fps=20,
        )

    # ========================================================
    # Print result
    # ========================================================

    print(
        "Result:",
        (
            "SUCCESS"
            if success
            else "FAIL"
        ),
    )

    print(
        "Steps:",
        num_executed_steps,
    )

    print(
        "Boundary ratio:",
        np.round(
            boundary_ratio,
            3,
        ),
    )

    # ========================================================
    # Return
    # ========================================================

    return {

        "task_id":
            int(task_id),

        "init_id":
            int(init_id),

        "instruction":
            instruction,

        "success":
            bool(success),

        "steps":
            int(
                num_executed_steps
            ),

        "success_step":
            (
                int(
                    success_step
                )
                if success_step
                is not None
                else None
            ),

        "video":
            (
                str(video_path)
                if video_path
                is not None
                else None
            ),

        "action_min":
            action_min.tolist(),

        "action_max":
            action_max.tolist(),

        "action_mean":
            action_mean.tolist(),

        "boundary_ratio":
            boundary_ratio.tolist(),
    }


# ============================================================
# Summary
# ============================================================

def print_summary(
    results,
    task_suite,
):

    print(
        "\n"
        "============================================================"
    )

    print(
        "LIBERO-SPATIAL SUMMARY"
    )

    print(
        "============================================================"
    )

    total_success = 0

    total_rollouts = 0

    summary_rows = []

    for task_id in range(
        NUM_TASKS
    ):

        task_results = [

            result

            for result
            in results

            if result[
                "task_id"
            ] == task_id

        ]

        if len(
            task_results
        ) == 0:

            continue

        successes = sum(

            result[
                "success"
            ]

            for result
            in task_results

        )

        num_rollouts = len(
            task_results
        )

        sr = (
            successes
            / num_rollouts
        )

        task = (
            task_suite
            .get_task(
                task_id
            )
        )

        print(
            f"Task {task_id:02d}: "
            f"{successes}/"
            f"{num_rollouts} "
            f"= "
            f"{sr * 100:5.1f}%"
        )

        print(
            f"         "
            f"{task.language}"
        )

        total_success += (
            successes
        )

        total_rollouts += (
            num_rollouts
        )

        successful_steps = [

            result[
                "steps"
            ]

            for result
            in task_results

            if result[
                "success"
            ]

        ]

        avg_success_steps = (

            float(
                np.mean(
                    successful_steps
                )
            )

            if len(
                successful_steps
            ) > 0

            else None
        )

        summary_rows.append(
            {
                "task_id":
                    task_id,

                "instruction":
                    task.language,

                "successes":
                    successes,

                "rollouts":
                    num_rollouts,

                "success_rate":
                    sr,

                "avg_success_steps":
                    avg_success_steps,
            }
        )

    print(
        "------------------------------------------------------------"
    )

    if total_rollouts > 0:

        overall_sr = (
            total_success
            / total_rollouts
        )

        print(
            f"Overall: "
            f"{total_success}/"
            f"{total_rollouts} "
            f"= "
            f"{overall_sr * 100:.1f}%"
        )

    else:

        overall_sr = 0.0

    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    with open(
        SUMMARY_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "instruction",
                "successes",
                "rollouts",
                "success_rate",
                "avg_success_steps",
            ],
        )

        writer.writeheader()

        writer.writerows(
            summary_rows
        )

    print(
        "\nSummary CSV:"
    )

    print(
        SUMMARY_CSV
    )


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    # ========================================================
    # Load MiniVLA
    # ========================================================

    print(
        "\nLoading MiniVLA..."
    )

    model = MiniVLA(
        vision_model_name=
            "facebook/dinov2-small",

        llm_model_name=
            "Qwen/Qwen2.5-0.5B",

        num_action_bins=256,

        freeze_vision=True,

        freeze_llm=False,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    print(
        "Checkpoint step:",
        checkpoint[
            "step"
        ],
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    print(
        "Model loaded!"
    )

    # ========================================================
    # LIBERO suite
    # ========================================================

    benchmark_dict = (
        benchmark
        .get_benchmark_dict()
    )

    task_suite = (
        benchmark_dict[
            TASK_SUITE_NAME
        ]()
    )

    print(
        "\nTask suite:",
        TASK_SUITE_NAME,
    )

    print(
        "Number of tasks:",
        task_suite.n_tasks,
    )

    # ========================================================
    # Resume previous evaluation
    # ========================================================

    if RESULT_JSON.exists():

        print(
            "\nExisting result file found."
        )

        print(
            "Resuming:"
        )

        print(
            RESULT_JSON
        )

        with open(
            RESULT_JSON,
            "r",
            encoding="utf-8",
        ) as f:

            results = json.load(
                f
            )

    else:

        results = []

    completed_pairs = {

        (
            result[
                "task_id"
            ],
            result[
                "init_id"
            ],
        )

        for result
        in results

    }

    print(
        "Already completed:",
        len(
            completed_pairs
        ),
    )

    # ========================================================
    # Evaluate all tasks
    # ========================================================

    num_tasks = min(
        NUM_TASKS,
        task_suite.n_tasks,
    )

    for task_id in range(
        num_tasks
    ):

        task = (
            task_suite
            .get_task(
                task_id
            )
        )

        instruction = (
            task.language
        )

        initial_states = (
            task_suite
            .get_task_init_states(
                task_id
            )
        )

        num_init_states = min(
            NUM_INIT_STATES,
            len(
                initial_states
            ),
        )

        # ----------------------------------------------------
        # Build task environment
        # ----------------------------------------------------

        bddl_file = os.path.join(

            get_libero_path(
                "bddl_files"
            ),

            task.problem_folder,

            task.bddl_file,

        )

        print(
            "\n\n"
            "############################################################"
        )

        print(
            f"TASK {task_id:02d}"
        )

        print(
            instruction
        )

        print(
            "############################################################"
        )

        env = OffScreenRenderEnv(
            bddl_file_name=
                bddl_file,

            camera_heights=
                128,

            camera_widths=
                128,
        )

        env.seed(
            0
        )

        # ----------------------------------------------------
        # Initial states
        # ----------------------------------------------------

        for init_id in range(
            num_init_states
        ):

            pair = (
                task_id,
                init_id,
            )

            if pair in completed_pairs:

                print(
                    f"\nSkipping "
                    f"Task {task_id:02d} "
                    f"Init {init_id:02d} "
                    f"(already completed)"
                )

                continue

            result = run_rollout(
                env=
                    env,

                model=
                    model,

                initial_state=
                    initial_states[
                        init_id
                    ],

                instruction=
                    instruction,

                task_id=
                    task_id,

                init_id=
                    init_id,

                device=
                    device,
            )

            results.append(
                result
            )

            completed_pairs.add(
                pair
            )

            # ------------------------------------------------
            # Save immediately
            # ------------------------------------------------

            save_results(
                results
            )

            # ------------------------------------------------
            # Running overall success rate
            # ------------------------------------------------

            successes = sum(

                result[
                    "success"
                ]

                for result
                in results

            )

            print(
                "\n"
                f"Overall so far: "
                f"{successes}/"
                f"{len(results)} "
                f"= "
                f"{successes / len(results) * 100:.1f}%"
            )

        env.close()

        # ----------------------------------------------------
        # Summary after every task
        # ----------------------------------------------------

        print_summary(
            results,
            task_suite,
        )

    # ========================================================
    # Final
    # ========================================================

    print_summary(
        results,
        task_suite,
    )

    print(
        "\nResults JSON:"
    )

    print(
        RESULT_JSON
    )

    print(
        "\nEvaluation finished."
    )


if __name__ == "__main__":
    main()