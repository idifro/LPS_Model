# OpenScene — nuScenes Mini: Setup & Run Guide

## 1. Environment Setup

### 1a. Start a GPU Node

```bash
bsub -Is -q inter_a100 -n 4 -M 5120 -W 12:00 -gpu "num=1:mig=4" /bin/bash
```

### 1b. Load Modules & Activate Environment

```bash
module load conda
conda activate openscene
module load cuda/11.0
module load cudnn/8.0.4
```

### 1c. Set Python Path

```bash
export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
```

### 1d. Verify Environment

```bash
python -c "
import torch; print('PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available())
import MinkowskiEngine; print('MinkowskiEngine OK')
import clip; print('CLIP OK')
from nuscenes import NuScenes; print('nuscenes-devkit OK')
from pyquaternion import Quaternion; print('pyquaternion OK')
print('GPU:', torch.cuda.get_device_name(0))
"
```

---

## 2. Preprocessing (CPU — can run on login node)

Converts raw nuScenes mini dataset into OpenScene format.
- **3D**: Aggregates ALL keyframes per scene → `data/nuscenes_3d/{split}/{scene}.pth`
- **2D**: Extracts last keyframe's 6 camera images + poses + intrinsics → `data/nuscenes_2d/{split}/{scene}/`

```bash
python scripts/preprocess/convert_nuscenes_mini.py
```

### Verify

```bash
ls data/nuscenes_3d/train/*.pth | wc -l   # → 8
ls data/nuscenes_3d/val/*.pth | wc -l     # → 2
ls data/nuscenes_2d/val/scene-0061/color/  # → 6 .jpg files
```

---

## 3. Feature Fusion (GPU required)

Projects OpenSeg 2D features onto 3D points via camera geometry.

**Prerequisite**: OpenSeg TF model at `models/openseg/exported_model/`

```bash
# Train split (8 scenes)
python scripts/feature_fusion/nuscenes_openseg.py \
    --data_dir data \
    --output_dir data/nuscenes_multiview_openseg \
    --split train \
    --openseg_model models/openseg/exported_model

# Val split (2 scenes)
python scripts/feature_fusion/nuscenes_openseg.py \
    --data_dir data \
    --output_dir data/nuscenes_multiview_openseg \
    --split val \
    --openseg_model models/openseg/exported_model
```

### Verify

```bash
ls data/nuscenes_multiview_openseg/*.pt | wc -l  # → 10
python -c "
import torch
d = torch.load('data/nuscenes_multiview_openseg/scene-0061.pt')
print('feat:', d['feat'].shape)        # (M, 768)
print('mask:', d['mask_full'].shape)   # (N,) bool
print('coverage:', d['mask_full'].float().mean().item())
"
```

---

## 4. Evaluation — Fusion (no 3D model needed)

Uses only fused 2D OpenSeg features. No model weights required.

```bash
python run/evaluate.py \
    --config config/nuscenes/ours_openseg_mini.yaml \
    TEST.feature_type fusion \
    TEST.save_folder exp/nuscenes_mini_openseg/eval_fusion
```

**Output**: mIoU, mAcc, allAcc for 16 nuScenes classes + `.ply` point clouds + `scene-XXXX_cameras_fusion.png` / `scene-XXXX_cameras_gt.png` (6-camera projection plots).

---

## 5. Evaluation — Distill with Pretrained Weights

Uses 3D MinkUNet18A model with pretrained weights.

### Download Pretrained Weights (from a machine with internet)

```bash
# On a machine with internet access:
wget https://cvg-data.inf.ethz.ch/openscene/models/nuscenes_openseg.pth.tar

# Transfer to HPC:
scp nuscenes_openseg.pth.tar <user>@<hpc>:/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/models/
```

### Run Evaluation

```bash
python run/evaluate.py \
    --config config/nuscenes/ours_openseg_mini_pretrained.yaml \
    TEST.feature_type distill \
    TEST.save_folder exp/nuscenes_mini_openseg/eval_pretrained_distill
```

---

## 6. Evaluation — Ensemble with Pretrained Weights

Confidence-based blend of fusion (2D) + distill (3D) features per point.

```bash
python run/evaluate.py \
    --config config/nuscenes/ours_openseg_mini_pretrained.yaml \
    TEST.feature_type ensemble \
    TEST.save_folder exp/nuscenes_mini_openseg/eval_pretrained_ensemble
```

---

## 7. Smoke Test — Distillation Training (5 epochs)

Trains MinkUNet18A to predict per-point features from geometry alone.
Loss: cosine similarity between 3D network output and fused 2D features.

```bash
python run/distill.py \
    --config config/nuscenes/ours_openseg_mini.yaml
```

### Verify Training Output

```bash
ls exp/nuscenes_mini_openseg/model/
python -c "
import torch
ckpt = torch.load('exp/nuscenes_mini_openseg/model/model_last.pth.tar', map_location='cpu')
print('Epoch:', ckpt['epoch'])
print('Best IoU:', ckpt.get('best_iou', 'N/A'))
"
```

### Evaluate Your Trained Model

```bash
# Distill
python run/evaluate.py \
    --config config/nuscenes/ours_openseg_mini.yaml \
    TEST.feature_type distill \
    TEST.split val \
    TEST.model_path exp/nuscenes_mini_openseg/model/model_best.pth.tar \
    TEST.save_folder exp/nuscenes_mini_openseg/eval_trained_distill \
    TEST.test_workers 0 \
    TEST.test_repeats 1

# Ensemble
python run/evaluate.py \
    --config config/nuscenes/ours_openseg_mini.yaml \
    TEST.feature_type ensemble \
    TEST.split val \
    TEST.model_path exp/nuscenes_mini_openseg/model/model_best.pth.tar \
    TEST.save_folder exp/nuscenes_mini_openseg/eval_trained_ensemble \
    TEST.test_workers 0 \
    TEST.test_repeats 1
```

---

## Quick Reference — Key Paths

| What | Path |
|------|------|
| Raw nuScenes mini | `/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes_mini/` |
| 3D point clouds (.pth) | `data/nuscenes_3d/{train,val}/` |
| 2D camera data | `data/nuscenes_2d/{train,val}/{scene}/` |
| Fused features (.pt) | `data/nuscenes_multiview_openseg/` |
| OpenSeg model | `models/openseg/exported_model/` |
| Experiment outputs | `exp/nuscenes_mini_openseg/` |
| Mini config | `config/nuscenes/ours_openseg_mini.yaml` |
| Pretrained config | `config/nuscenes/ours_openseg_mini_pretrained.yaml` |

## Quick Reference — CLI Overrides

| Override | Purpose |
|----------|---------|
| `TEST.feature_type {fusion,distill,ensemble}` | Evaluation mode |
| `TEST.split {train,val}` | Which data split (train=8 scenes, val=2) |
| `TEST.test_workers 0` | Disable parallel loading (debug) |
| `TEST.test_repeats 1` | Single pass (faster, less stable) |
| `TEST.test_repeats 5` | 5 passes with averaged logits (default, more robust) |
| `TEST.model_path <path>` | Model weights for distill/ensemble |
| `TEST.vis_pred False` | Skip saving prediction `.ply` files |
| `TEST.vis_gt False` | Skip saving ground truth `.ply` files |
| `DATA.voxel_size 0.1` | Coarser voxels (faster, less precise) |

## Mini Dataset Splits

| Split | Scenes | Points/scene |
|-------|--------|-------------|
| train | scene-0061, 0553, 0655, 0757, 0796, 1077, 1094, 1100 | ~1.35-1.42M |
| val | scene-0103, 0916 | ~1.39-1.42M |
