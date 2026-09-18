import os

os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from minivla.paths import (
    LIBERO_DATASET_ROOT,
)


# ============================================================
# Configuration
# ============================================================

TASK_SUITE_NAME = "libero_spatial"

TASK_ID = 0

DEMO_NAME = "demo_0"

OUTPUT_DIR = Path(
    "alignment_debug"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Image utilities
# ============================================================

def rotate_180(image):

    image = image[
        ::-1,
        ::-1,
    ]

    return np.ascontiguousarray(
        image
    )


def image_mae(
    image_a,
    image_b,
):

    image_a = image_a.astype(
        np.float32
    )

    image_b = image_b.astype(
        np.float32
    )

    return np.abs(
        image_a
        - image_b
    ).mean()


def save_image(
    image,
    filename,
):

    path = (
        OUTPUT_DIR
        / filename
    )

    Image.fromarray(
        image
    ).save(
        path
    )

    print(
        "Saved:",
        path,
    )


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # 1. Load LIBERO suite
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

    task = task_suite.get_task(
        TASK_ID
    )

    print(
        "\n================================"
    )

    print(
        "Task"
    )

    print(
        "================================"
    )

    print(
        "Task ID:",
        TASK_ID,
    )

    print(
        "Instruction:"
    )

    print(
        task.language
    )

    # ========================================================
    # 2. Find HDF5 demo
    # ========================================================

    demo_relative_path = (
        task_suite
        .get_task_demonstration(
            TASK_ID
        )
    )

    demo_path = (
        LIBERO_DATASET_ROOT
        / demo_relative_path
    )

    print(
        "\nDemo file:"
    )

    print(
        demo_path
    )

    if not demo_path.exists():

        raise FileNotFoundError(
            f"Demo file not found: "
            f"{demo_path}"
        )

    # ========================================================
    # 3. Load HDF5
    # ========================================================

    with h5py.File(
        demo_path,
        "r",
    ) as h5_file:

        data_group = (
            h5_file["data"]
        )

        print(
            "\nAvailable demos:"
        )

        demo_keys = list(
            data_group.keys()
        )

        print(
            demo_keys[:10]
        )

        # ----------------------------------------------------
        # Some dataset versions may use a different demo name
        # ----------------------------------------------------

        if DEMO_NAME in data_group:

            selected_demo_name = (
                DEMO_NAME
            )

        else:

            selected_demo_name = (
                sorted(
                    demo_keys
                )[0]
            )

            print(
                f"\n{DEMO_NAME} "
                "not found."
            )

            print(
                "Using:",
                selected_demo_name,
            )

        demo = data_group[
            selected_demo_name
        ]

        # ----------------------------------------------------
        # Load demo
        # ----------------------------------------------------

        states = (
            demo["states"][:]
        )

        actions = (
            demo["actions"][:]
        )

        demo_images = (
            demo[
                "obs"
            ][
                "agentview_rgb"
            ][:]
        )

        print(
            "\n================================"
        )

        print(
            "Demo information"
        )

        print(
            "================================"
        )

        print(
            "Demo:",
            selected_demo_name,
        )

        print(
            "states:",
            states.shape,
        )

        print(
            "actions:",
            actions.shape,
        )

        print(
            "images:",
            demo_images.shape,
        )

        print(
            "first action:"
        )

        print(
            np.round(
                actions[0],
                4,
            )
        )

        # ====================================================
        # 4. Build simulator
        # ====================================================

        bddl_file = os.path.join(
            get_libero_path(
                "bddl_files"
            ),

            task.problem_folder,

            task.bddl_file,
        )

        print(
            "\nCreating environment..."
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

        # ====================================================
        # 5. Compare HDF5 image 0
        #
        # Candidate A:
        # observation produced directly from states[0]
        #
        # Candidate B:
        # states[0] + action[0]
        #
        # Candidate C:
        # observation directly regenerated from states[1]
        #
        # For every candidate:
        # compare RAW vs ROTATED 180 degrees.
        # ====================================================

        training_image = (
            demo_images[0]
        )

        # ----------------------------------------------------
        # Candidate A:
        # exact states[0]
        # ----------------------------------------------------

        env.reset()

        obs_state0 = (
            env.set_init_state(
                states[0]
            )
        )

        state0_raw = (
            obs_state0[
                "agentview_image"
            ]
        )

        state0_raw = (
            np.ascontiguousarray(
                state0_raw
            )
        )

        state0_rotated = (
            rotate_180(
                state0_raw
            )
        )

        # ----------------------------------------------------
        # Candidate B:
        # states[0] -> action[0]
        # ----------------------------------------------------

        env.reset()

        env.set_init_state(
            states[0]
        )

        (
            obs_after_action0,
            reward,
            done,
            info,
        ) = env.step(
            actions[0]
        )

        after_action0_raw = (
            np.ascontiguousarray(
                obs_after_action0[
                    "agentview_image"
                ]
            )
        )

        after_action0_rotated = (
            rotate_180(
                after_action0_raw
            )
        )

        # ----------------------------------------------------
        # Candidate C:
        # direct states[1]
        # ----------------------------------------------------

        if len(states) > 1:

            env.reset()

            obs_state1 = (
                env.set_init_state(
                    states[1]
                )
            )

            state1_raw = (
                np.ascontiguousarray(
                    obs_state1[
                        "agentview_image"
                    ]
                )
            )

            state1_rotated = (
                rotate_180(
                    state1_raw
                )
            )

        else:

            state1_raw = None
            state1_rotated = None

        # ====================================================
        # 6. Calculate image MAEs
        # ====================================================

        comparisons = {}

        comparisons[
            "state0_raw"
        ] = image_mae(
            training_image,
            state0_raw,
        )

        comparisons[
            "state0_rotated"
        ] = image_mae(
            training_image,
            state0_rotated,
        )

        comparisons[
            "after_action0_raw"
        ] = image_mae(
            training_image,
            after_action0_raw,
        )

        comparisons[
            "after_action0_rotated"
        ] = image_mae(
            training_image,
            after_action0_rotated,
        )

        if state1_raw is not None:

            comparisons[
                "state1_raw"
            ] = image_mae(
                training_image,
                state1_raw,
            )

            comparisons[
                "state1_rotated"
            ] = image_mae(
                training_image,
                state1_rotated,
            )

        print(
            "\n================================"
        )

        print(
            "Image alignment"
        )

        print(
            "================================"
        )

        for name, mae in sorted(
            comparisons.items(),
            key=lambda item:
                item[1],
        ):

            print(
                f"{name:25s}"
                f" MAE = "
                f"{mae:.4f}"
            )

        best_name = min(
            comparisons,
            key=comparisons.get,
        )

        print(
            "\nBEST MATCH:"
        )

        print(
            best_name
        )

        print(
            "MAE:",
            comparisons[
                best_name
            ],
        )

        # ====================================================
        # 7. Save images for visual inspection
        # ====================================================

        print(
            "\nSaving comparison images..."
        )

        save_image(
            training_image,
            "00_training_hdf5.png",
        )

        save_image(
            state0_raw,
            "01_state0_raw.png",
        )

        save_image(
            state0_rotated,
            "02_state0_rotated.png",
        )

        save_image(
            after_action0_raw,
            "03_after_action0_raw.png",
        )

        save_image(
            after_action0_rotated,
            "04_after_action0_rotated.png",
        )

        if state1_raw is not None:

            save_image(
                state1_raw,
                "05_state1_raw.png",
            )

            save_image(
                state1_rotated,
                "06_state1_rotated.png",
            )

        # ====================================================
        # 8. Expert replay
        # ====================================================

        print(
            "\n================================"
        )

        print(
            "Expert replay"
        )

        print(
            "================================"
        )

        env.reset()

        env.set_init_state(
            states[0]
        )

        success = False

        state_errors = []

        success_step = None

        for step, action in enumerate(
            actions
        ):

            (
                obs,
                reward,
                done,
                info,
            ) = env.step(
                action
            )

            # ------------------------------------------------
            # Direct task success check
            # ------------------------------------------------

            current_success = (
                env.check_success()
            )

            if current_success:

                success = True

                success_step = step

            # ------------------------------------------------
            # Check replay state consistency
            #
            # action[t] should approximately bring simulator
            # toward the next stored state.
            # ------------------------------------------------

            if (
                step + 1
                < len(states)
            ):

                current_state = (
                    env.get_sim_state()
                )

                target_state = (
                    states[
                        step + 1
                    ]
                )

                if (
                    current_state.shape
                    == target_state.shape
                ):

                    state_error = (
                        np.linalg.norm(
                            current_state
                            - target_state
                        )
                    )

                    state_errors.append(
                        state_error
                    )

            if (
                step < 10
                or
                step % 20 == 0
            ):

                print(
                    f"Step {step:03d} | "
                    f"reward="
                    f"{float(reward):.3f} | "
                    f"done="
                    f"{done} | "
                    f"success="
                    f"{bool(current_success)}"
                )

            if success:

                print(
                    f"\nExpert SUCCESS "
                    f"at step "
                    f"{success_step}"
                )

                break

        # ====================================================
        # 9. Replay statistics
        # ====================================================

        print(
            "\n================================"
        )

        print(
            "Expert replay result"
        )

        print(
            "================================"
        )

        print(
            "Success:",
            success,
        )

        print(
            "Steps executed:",
            (
                success_step + 1
                if success_step
                is not None
                else len(actions)
            ),
        )

        if len(
            state_errors
        ) > 0:

            state_errors = (
                np.asarray(
                    state_errors
                )
            )

            print(
                "\nState replay error:"
            )

            print(
                "mean =",
                float(
                    state_errors.mean()
                ),
            )

            print(
                "max  =",
                float(
                    state_errors.max()
                ),
            )

            print(
                "last =",
                float(
                    state_errors[-1]
                ),
            )

        env.close()

        print(
            "\nEnvironment closed."
        )


if __name__ == "__main__":
    main()