import os

# 必须放在 mujoco / libero import 前面
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path

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

# 第一次只跑一个任务
TASK_ID = 0

# 第一次只跑一个固定初始状态
INIT_STATE_ID = 0

# 官方 LIBERO-Spatial eval 使用约 220 control steps
MAX_STEPS = 220

# 开头先让物体稳定
NUM_WAIT_STEPS = 10

VIDEO_DIR = Path(
    "rollouts/minivla"
)

VIDEO_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Convert simulator image -> training image convention
# ============================================================

def get_model_image(obs):

    img = obs[
        "agentview_image"
    ]

    # Our MiniVLA was trained directly on
    # LIBERO HDF5 agentview_rgb.
    #
    # Alignment test shows simulator RAW image
    # matches the training image.
    #
    # DO NOT rotate 180 degrees.

    img = np.ascontiguousarray(
        img
    )

    return img


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

    # --------------------------------------------------------
    # numpy RGB -> PIL
    # --------------------------------------------------------

    image = Image.fromarray(
        image_np
    )

    # --------------------------------------------------------
    # DINO preprocessing
    #
    # [H,W,3]
    #     ↓
    # [1,3,224,224]
    # --------------------------------------------------------

    pixel_values = (
        model.vision_encoder
        .preprocess(
            image
        )
        .to(device)
    )

    # --------------------------------------------------------
    # Instruction -> Qwen tokens
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
    # True autoregressive 7D action prediction
    # --------------------------------------------------------

    action, action_token_ids = (
        model.predict_action(
            pixel_values=
                pixel_values,

            input_ids=
                input_ids,

            text_attention_mask=
                text_attention_mask,
        )
    )

    # [1,7] -> [7]
    action = (
        action[0]
        .detach()
        .float()
        .cpu()
        .numpy()
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
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

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
    # 1. Load MiniVLA
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

    print(
        "Loading checkpoint..."
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    print(
        "Checkpoint step:",
        checkpoint["step"],
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
    # 2. Get LIBERO-Spatial suite
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
    # 3. Select one task
    # ========================================================

    task = (
        task_suite
        .get_task(
            TASK_ID
        )
    )

    instruction = (
        task.language
    )

    print(
        "\n================================"
    )

    print(
        "Task ID:",
        TASK_ID,
    )

    print(
        "Instruction:"
    )

    print(
        instruction
    )

    print(
        "================================"
    )

    # ========================================================
    # 4. Build environment
    # ========================================================

    bddl_file = os.path.join(
        get_libero_path(
            "bddl_files"
        ),

        task.problem_folder,

        task.bddl_file,
    )

    env = OffScreenRenderEnv(
        bddl_file_name=
            bddl_file,

        # 和我们的训练数据一样
        camera_heights=128,

        camera_widths=128,
    )

    # 官方 eval 也固定 seed
    env.seed(
        0
    )

    # ========================================================
    # 5. Fixed initial state
    # ========================================================

    initial_states = (
        task_suite
        .get_task_init_states(
            TASK_ID
        )
    )

    env.reset()

    obs = env.set_init_state(
        initial_states[
            INIT_STATE_ID
        ]
    )

    # ========================================================
    # 6. Wait for simulator stabilization
    # ========================================================

    print(
        "\nWaiting for simulator "
        "to stabilize..."
    )

    for _ in range(
        NUM_WAIT_STEPS
    ):

        # LIBERO no-op:
        #
        # dx dy dz
        # rx ry rz
        # gripper open
        dummy_action = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.0,
        ]

        obs, reward, done, info = (
            env.step(
                dummy_action
            )
        )

    # ========================================================
    # 7. Closed-loop rollout
    # ========================================================

    print(
        "\n================================"
    )

    print(
        "Starting closed-loop rollout"
    )

    print(
        "================================"
    )

    frames = []

    actions = []

    success = False

    saturated_count = (
        np.zeros(
            6,
            dtype=np.int64,
        )
    )

    for step in range(
        MAX_STEPS
    ):

        # ----------------------------------------------------
        # Observation
        # ----------------------------------------------------

        image_np = (
            get_model_image(
                obs
            )
        )

        frames.append(
            image_np
        )

        # ----------------------------------------------------
        # MiniVLA prediction
        # ----------------------------------------------------

        action, token_ids = (
            predict_action(
                model=model,
                image_np=
                    image_np,
                instruction=
                    instruction,
                device=device,
            )
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # 我们训练数据本身就是
        #
        # -1 = gripper open
        # +1 = gripper close
        #
        # ActionTokenizer.decode()已经输出这个格式
        #
        # 所以这里：
        #
        # 不要 invert gripper
        # 不要再次 normalize
        # ----------------------------------------------------

        action = action.astype(
            np.float32
        )

        actions.append(
            action.copy()
        )

        # ----------------------------------------------------
        # saturation diagnostics
        # ----------------------------------------------------

        token_list = (
            token_ids.tolist()
        )

        act_min_id = (
            model.action_tokenizer
            .action_token_ids[0]
        )

        act_max_id = (
            model.action_tokenizer
            .action_token_ids[-1]
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
        # Print some actions
        # ----------------------------------------------------

        if (
            step < 10
            or
            step % 20 == 0
        ):

            print(
                f"Step {step:03d} | "
                f"action = "
                f"{np.round(action, 4)}"
            )

        # ----------------------------------------------------
        # Execute action
        # ----------------------------------------------------

        obs, reward, done, info = (
            env.step(
                action.tolist()
            )
        )

        # ----------------------------------------------------
        # LIBERO done = task success
        # ----------------------------------------------------

        if done:

            success = True

            print(
                f"\nSUCCESS at "
                f"step {step}!"
            )

            break

    # ========================================================
    # 8. Close environment
    # ========================================================

    env.close()

    # ========================================================
    # 9. Save video
    # ========================================================

    video_path = (
        VIDEO_DIR
        / (
            f"task_{TASK_ID:02d}"
            f"_init_{INIT_STATE_ID:02d}"
            f"_success_{success}.mp4"
        )
    )

    imageio.mimsave(
        video_path,
        frames,
        fps=20,
    )

    # ========================================================
    # 10. Diagnostics
    # ========================================================

    actions = np.asarray(
        actions
    )

    print(
        "\n================================"
    )

    print(
        "Rollout result"
    )

    print(
        "================================"
    )

    print(
        "Success:",
        success,
    )

    print(
        "Control steps:",
        len(actions),
    )

    print(
        "Video:",
        video_path,
    )

    if len(actions) > 0:

        print(
            "\nAction min:"
        )

        print(
            np.round(
                actions.min(axis=0),
                4,
            )
        )

        print(
            "\nAction max:"
        )

        print(
            np.round(
                actions.max(axis=0),
                4,
            )
        )

        print(
            "\nAction mean:"
        )

        print(
            np.round(
                actions.mean(axis=0),
                4,
            )
        )

        print(
            "\nBoundary-token ratio:"
        )

        for dim in range(6):

            ratio = (
                saturated_count[dim]
                / len(actions)
            )

            print(
                f"dim_{dim}: "
                f"{ratio * 100:.2f}%"
            )


if __name__ == "__main__":
    main()