# MiniVLA

A lightweight Vision-Language-Action (VLA) model built from scratch with **DINOv2 + Qwen2.5-0.5B**, trained on **LIBERO-Spatial** demonstrations for autoregressive 7-DoF robot control.

The goal of this project is to build and understand a complete VLA pipeline end-to-end:

- LIBERO demonstration loading
- visual representation learning
- language conditioning
- continuous action discretization
- behavior cloning
- autoregressive action generation
- offline evaluation
- closed-loop simulator evaluation

---

## Overview

MiniVLA follows a token-based action prediction formulation.

```text
RGB Observation
      │
      ▼
   DINOv2
(frozen vision encoder)
      │
      │ [B, 256, 384]
      ▼
 MLP Projector
   384 → 896
      │
      ▼
Visual Embeddings ─────────────┐
                              │
Language Instruction           │
      │                        │
      ▼                        │
Qwen Tokenizer                 │
      │                        │
      ▼                        │
Text Embeddings ───────────────┤
                              │
                              ▼
                    Qwen2.5-0.5B
                              │
                              ▼
                 Autoregressive Action
                        Token Prediction
                              │
                              ▼
                    7-DoF Robot Action
```

The final action consists of:

```text
[x, y, z, roll, pitch, yaw, gripper]
```

The first six continuous dimensions are discretized into **256 action bins**, while the gripper is represented using two discrete tokens:

```text
GRIP_OPEN
GRIP_CLOSE
```

---

## Model Architecture

### Vision Encoder

MiniVLA uses:

```text
facebook/dinov2-small
```

The vision encoder is frozen during training.

For a `224 × 224` input image, DINOv2 produces patch features:

```text
[B, 256, 384]
```

A trainable MLP projector maps them into the Qwen hidden dimension:

```text
384 → 896 → 896
```

### Language Backbone

The language backbone is:

```text
Qwen/Qwen2.5-0.5B
```

The instruction is tokenized using the original Qwen tokenizer.

Additional action tokens are added to the vocabulary and the embedding table is resized accordingly.

### Action Tokenization

For each of the first six action dimensions, MiniVLA applies per-dimension quantile normalization using the 1st and 99th percentiles of the LIBERO-Spatial training data:

```text
raw action
    │
    ▼
clip to [q01, q99]
    │
    ▼
normalize to [-1, 1]
    │
    ▼
quantize into 256 bins
    │
    ▼
ACT_000 ... ACT_255
```

The gripper is represented separately:

```text
-1 → GRIP_OPEN
+1 → GRIP_CLOSE
```

During inference, predicted action tokens are decoded back into continuous LIBERO actions.

---

## Training Objective

MiniVLA is trained with behavior cloning using a causal language modeling objective.

The sequence contains:

```text
[visual tokens]
[text tokens]
[action tokens]
```

Only the 7 action-token positions contribute to the training loss.

```text
visual labels → -100
text labels   → -100
action labels → supervised
```

This allows the Qwen backbone to autoregressively predict the robot action conditioned on the current image and language instruction.

---

## Dataset

The model is trained on **LIBERO-Spatial** demonstrations.

The local dataset used in this project contains:

```text
10 tasks
50 demonstrations per task
500 demonstrations total
62,250 action timesteps
```

Each training sample contains:

```text
RGB observation: 128 × 128 × 3
language instruction
7D continuous robot action
```

---

## Results

### Tiny Overfit Sanity Check

Before full training, the complete pipeline was tested on 32 samples.

```text
Action-token accuracy: 100%
Exact-action accuracy: 100%
Action MAE:            0.000733
Gripper accuracy:      100%
```

This sanity check verifies that the full pipeline can successfully overfit a small dataset.

### Offline Evaluation

After training for 10 epochs on the full LIBERO-Spatial dataset:

| Metric | Result |
|---|---:|
| Action-token accuracy | **94.66%** |
| Exact-action accuracy | **75.20%** |
| Action MAE | **0.003094** |
| Gripper accuracy | **99.80%** |

Per-dimension token accuracy:

| Dimension | Accuracy |
|---|---:|
| dim 0 | 95.8% |
| dim 1 | 93.6% |
| dim 2 | 93.8% |
| dim 3 | 93.2% |
| dim 4 | 92.8% |
| dim 5 | 93.6% |
| gripper | 99.8% |

> These metrics are measured on sampled states from the training dataset and should be interpreted as **seen-state offline evaluation**, not held-out generalization performance.

---

## Closed-Loop LIBERO Evaluation

MiniVLA was connected directly to the LIBERO simulator and evaluated in closed loop:

```text
Current RGB Observation
        │
        ▼
     MiniVLA
        │
        ▼
Predicted 7D Action
        │
        ▼
   env.step(action)
        │
        ▼
New Observation
        │
        └─────────────── repeat
```

A small-scale evaluation was performed using:

```text
10 LIBERO-Spatial tasks
×
5 fixed initial states per task
=
50 closed-loop rollouts
```

Result:

```text
Successful rollouts: 23 / 50
Success Rate:        46.0%
```

This is a **50-rollout experimental evaluation**, rather than the official full LIBERO benchmark protocol.

Detailed results are available in:

```text
results/libero_spatial_50rollout_summary.csv
results/libero_spatial_50rollout_results.json
```

---

## Simulator Alignment Debugging

One important issue encountered during development was a train-test image preprocessing mismatch.

The initial simulator rollout failed despite strong offline performance.

The simulator image had originally been rotated by 180 degrees following another VLA evaluation pipeline. However, MiniVLA was trained directly on LIBERO HDF5 images, which used a different image convention.

To diagnose this, the HDF5 training image was compared with simulator-rendered observations.

Approximate pixel MAE:

```text
Raw simulator image:      ~2
180° rotated image:       ~70
```

The raw simulator observation clearly matched the training distribution.

After removing the incorrect 180-degree rotation, MiniVLA successfully completed the manipulation task in closed loop.

An expert-action replay test was also performed to verify:

```text
MuJoCo environment       ✓
LIBERO task setup        ✓
action convention        ✓
gripper convention       ✓
control interface        ✓
demonstration replay     ✓
```

This debugging process helped isolate the failure to a visual preprocessing mismatch rather than the robot control interface.

---

## Project Structure

```text
MiniVLA/
├── minivla/
│   ├── __init__.py
│   ├── action_tokenizer.py
│   ├── collator.py
│   ├── dataset.py
│   ├── llm_backbone.py
│   ├── model.py
│   ├── paths.py
│   ├── projector.py
│   └── vision_encoder.py
│
├── scripts/
│   ├── train.py
│   ├── overfit_tiny.py
│   ├── eval_offline.py
│   ├── check_libero_alignment.py
│   ├── rollout_libero.py
│   └── eval_libero_spatial.py
│
├── tests/
│   ├── test_action_tokenizer.py
│   ├── test_dataset.py
│   ├── test_model.py
│   └── test_train_step.py
│
├── results/
│   ├── libero_spatial_50rollout_summary.csv
│   └── libero_spatial_50rollout_results.json
│
├── assets/
│   └── minivla_libero_success.mp4
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

### 1. Clone MiniVLA

```bash
git clone <YOUR_MINIVLA_REPOSITORY_URL>
cd MiniVLA
```

### 2. Create a Python environment

```bash
conda create -n minivla python=3.10
conda activate minivla
```

### 3. Install PyTorch

Install PyTorch separately according to your CUDA environment:

https://pytorch.org/get-started/locally/

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

---

## Install LIBERO

Clone the official LIBERO repository next to MiniVLA:

```bash
cd ..
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
```

Recommended directory structure:

```text
workspace/
├── LIBERO/
└── MiniVLA/
```

Add LIBERO to `PYTHONPATH`:

```bash
export LIBERO_ROOT=/path/to/LIBERO
export PYTHONPATH=$PYTHONPATH:$LIBERO_ROOT
```

MiniVLA also supports the following environment variables:

```bash
export LIBERO_ROOT=/path/to/LIBERO
export LIBERO_DATASET_ROOT=/path/to/LIBERO/datasets
export MINIVLA_CHECKPOINT_ROOT=/path/to/checkpoints
```

If `LIBERO_ROOT` is not specified, MiniVLA assumes that `LIBERO/` and `MiniVLA/` are sibling directories.

---

## Dataset

Download the LIBERO datasets following the official LIBERO instructions.

The expected LIBERO-Spatial directory is:

```text
LIBERO/
└── datasets/
    └── libero_spatial/
```

You can verify the dataset path with:

```bash
PYTHONPATH=.:../LIBERO python - <<'PY'
from minivla.paths import LIBERO_SPATIAL_DATASET_ROOT

print(LIBERO_SPATIAL_DATASET_ROOT)
print("Exists:", LIBERO_SPATIAL_DATASET_ROOT.exists())
PY
```

---

## Sanity Tests

Run the action tokenizer test:

```bash
PYTHONPATH=.:../LIBERO \
python tests/test_action_tokenizer.py
```

Dataset test:

```bash
PYTHONPATH=.:../LIBERO \
python tests/test_dataset.py
```

Model forward test:

```bash
PYTHONPATH=.:../LIBERO \
python tests/test_model.py
```

Training-step test:

```bash
PYTHONPATH=.:../LIBERO \
python tests/test_train_step.py
```

---

## Tiny Overfit Test

A useful sanity check before full training is to verify that the model can overfit a very small subset:

```bash
PYTHONPATH=.:../LIBERO \
python scripts/overfit_tiny.py
```

This tests the complete pipeline:

```text
dataset
→ collator
→ DINOv2
→ projector
→ Qwen
→ action tokenizer
→ causal LM loss
→ backward
→ optimizer
```

---

## Training

Run full behavior-cloning training with:

```bash
PYTHONPATH=.:../LIBERO \
python scripts/train.py
```

The default training setup used for the reported experiment included:

```text
Model:                 DINOv2-small + Qwen2.5-0.5B
Dataset:               LIBERO-Spatial
Training samples:      62,250
Epochs:                10
Batch size:            1
Gradient accumulation: 8
Effective batch size:  8
Learning rate:         1e-5
Vision encoder:        frozen
Language model:        trainable
```

Checkpoints are stored locally and are excluded from Git through `.gitignore`.

---

## Offline Evaluation

Run:

```bash
PYTHONPATH=.:../LIBERO \
python scripts/eval_offline.py
```

Set the checkpoint path in the evaluation script to the checkpoint you want to evaluate.

The reported experiment used the final checkpoint at training step:

```text
77,820
```

---

## Single Closed-Loop Rollout

Run one LIBERO closed-loop rollout with:

```bash
MUJOCO_GL=egl \
PYTHONPATH=.:../LIBERO \
python scripts/rollout_libero.py
```

The script performs:

```text
simulator observation
→ MiniVLA inference
→ 7D action
→ environment step
→ next observation
```

and can save the rollout as an MP4 video.

---

## LIBERO-Spatial Evaluation

Run the multi-task evaluation with:

```bash
MUJOCO_GL=egl \
PYTHONPATH=.:../LIBERO \
python scripts/eval_libero_spatial.py
```

The reported result uses:

```text
10 tasks
5 fixed initial states per task
220 maximum control steps
50 total rollouts
```

Evaluation results are saved incrementally so that interrupted runs can be resumed.

---

## Action Normalization

The first six LIBERO action dimensions have significantly different numerical ranges.

Using a single global `[-1, 1]` discretization caused several low-range rotational dimensions to collapse near the center action bin.

MiniVLA therefore uses per-dimension quantile statistics:

```text
q01 = [
    -0.760714,
    -0.656250,
    -0.937500,
    -0.108214,
    -0.204643,
    -0.186429,
]

q99 = [
     0.937500,
     0.873214,
     0.934821,
     0.105000,
     0.175714,
     0.143571,
]
```

Each continuous action dimension is independently normalized before discretization.

This significantly improves effective action resolution for dimensions with small physical ranges.

---

## Limitations

MiniVLA is primarily an educational and experimental implementation rather than a state-of-the-art VLA model.

Current limitations include:

- behavior cloning only
- no action chunking
- no diffusion or flow-matching action head
- no temporal image history
- frozen DINOv2 visual encoder
- relatively small Qwen2.5-0.5B language backbone
- limited closed-loop generalization
- evaluation performed on a small fixed subset of LIBERO initial states
- offline metrics are measured on seen training states

The gap between high offline action accuracy and lower closed-loop success illustrates the effect of **covariate shift and compounding errors** in behavior cloning.

---

## What I Learned

This project was built to understand the full internal workflow of a Vision-Language-Action system rather than treating an existing VLA implementation as a black box.

The implementation covers:

```text
vision encoding
language tokenization
multimodal projection
action discretization
causal action modeling
teacher forcing
behavior cloning
autoregressive inference
robot simulator integration
expert replay
closed-loop evaluation
failure diagnosis
```

---

## Acknowledgements

This project builds on:

- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5)

MiniVLA is an independent educational implementation and is not an official implementation of LIBERO, DINOv2, or Qwen.
