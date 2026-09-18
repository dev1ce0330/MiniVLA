import os
from pathlib import Path

import h5py
import torch

from PIL import Image
from torch.utils.data import Dataset

from libero.libero import benchmark

class LIBEROSpatialDataset(Dataset):

    def __init__(
        self,
        dataset_dir: str,

    ):
        super().__init__()

        self.dataset_dir = Path(dataset_dir)

        benchmark_dict = benchmark.get_benchmark_dict()

        self.task_suite = benchmark_dict["libero_spatial"]()


        self.samples = []

        for task_id in range(self.task_suite.n_tasks):

            task = self.task_suite.get_task(task_id)

            task_name = task.name
            instruction = task.language

            hdf5_path = (
                self.dataset_dir
                / f"{task_name}_demo.hdf5"
            )

            if not hdf5_path.exists():
                raise FileNotFoundError(
                    f"Cannot find dataset:\n{hdf5_path}"
                )

            with h5py.File(hdf5_path, "r") as f:

                data_group = f["data"]

                demo_names = list(data_group.keys())

                for demo_name in demo_names:

                    demo = data_group[demo_name]

                    trajectory_length = demo["actions"].shape[0]

                    for timestep in range(trajectory_length):

                        self.samples.append(
                            (
                                str(hdf5_path),
                                demo_name,
                                timestep,
                                instruction,
                            )
                        )

        print(
            f"[MiniVLA Dataset]"
            f"Loaded {len(self.samples)} samples" 
        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        (
            hdf5_path,
            demo_name,
            timestep,
            instruction,
        ) = self.samples[index]



        with h5py.File(hdf5_path, "r") as f:

            demo = f["data"][demo_name]

            image_array = (
                demo["obs"]["agentview_rgb"][timestep]
            )

            action = (
                demo["actions"][timestep]
            )

        image = Image.fromarray(image_array)

        action = torch.tensor(
            action,
            dtype = torch.float32,
        )

        return {
            "image": image,
            "instruction": instruction,
            "action": action,
        }