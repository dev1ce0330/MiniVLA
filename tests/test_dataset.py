from minivla.dataset import LIBEROSpatialDataset
from minivla.paths import (
    LIBERO_SPATIAL_DATASET_ROOT,
)

DATASET_ROOT = LIBERO_SPATIAL_DATASET_ROOT


dataset = LIBEROSpatialDataset(
    dataset_dir=DATASET_ROOT
)


print()
print("Dataset size:")
print(len(dataset))


sample = dataset[0]


print()
print("Instruction:")
print(sample["instruction"])


print()
print("Image:")
print(sample["image"])


print()
print("Image size:")
print(sample["image"].size)


print()
print("Action:")
print(sample["action"])


print()
print("Action shape:")
print(sample["action"].shape)


sample["image"].save(
    "/tmp/minivla_dataset_test.png"
)