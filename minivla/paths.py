import os
from pathlib import Path


# MiniVLA repository root
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------
# LIBERO
# ------------------------------------------------------------
#
# Priority:
#   1. LIBERO_ROOT environment variable
#   2. Assume LIBERO is next to MiniVLA
#
# Example:
#
# projects/vla_learning/
# ├── LIBERO/
# └── MiniVLA/
#

LIBERO_ROOT = Path(
    os.environ.get(
        "LIBERO_ROOT",
        PROJECT_ROOT.parent / "LIBERO",
    )
).expanduser()


LIBERO_DATASET_ROOT = Path(
    os.environ.get(
        "LIBERO_DATASET_ROOT",
        LIBERO_ROOT / "datasets",
    )
).expanduser()


LIBERO_SPATIAL_DATASET_ROOT = (
    LIBERO_DATASET_ROOT
    / "libero_spatial"
)


# ------------------------------------------------------------
# Local output directories
# ------------------------------------------------------------

CHECKPOINT_ROOT = Path(
    os.environ.get(
        "MINIVLA_CHECKPOINT_ROOT",
        PROJECT_ROOT / "checkpoints",
    )
).expanduser()