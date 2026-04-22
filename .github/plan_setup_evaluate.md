# OpenScene nuScenes Mini — Setup, Preprocessing & Evaluation

## 1. Environment Setup

### Start a GPU Node
```bash
bsub -Is -q inter_a100 -n 4 -M 5120 -W 12:00 -gpu "num=1:mig=4" /bin/bash
```

### Load Modules & Activate Conda
```bash
module load conda
conda activate openscene
module load cuda/11.0
module load cudnn/8.0.4
```

### Set Python Path
```bash
export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
```

---

## 2. Preprocessing (CPU — login node OK)

Converts raw nuScenes mini into OpenScene format:
- **3D**: aggregates all keyframes per scene → `data/nuscenes_3d/{split}/{scene}.pth`
- **2D**: extracts last keyframe's 6 cameras + poses + intrinsics → `data/nuscenes_2d/{split}/{scene}/`

```bash
python scripts/preprocess/convert_nuscenes_mini.py
```

Verify:
```bash
ls data/nuscenes_3d/train/*.pth | wc -l   # 8
ls data/nuscenes_3d/val/*.pth | wc -l     # 2
```

---

## 3. Feature Fusion (GPU required)

Projects OpenSeg 2D features onto 3D points.
Prerequisite: OpenSeg TF model at `models/openseg/exported_model/`.

```bash
python scripts/feature_fusion/nuscenes_openseg.py \
    --data_dir data --output_dir data/nuscenes_multiview_openseg \
    --split train --openseg_model models/openseg/exported_model

python scripts/feature_fusion/nuscenes_openseg.py \
    --data_dir data --output_dir data/nuscenes_multiview_openseg \
    --split val --openseg_model models/openseg/exported_model
```

Verify:
```bash
ls data/nuscenes_multiview_openseg/*.pt | wc -l   # 10
```

---

## 4. Evaluation

### Run all 3 modes (experiment: nusc_mini_openseg_img_viz)
```bash
bash scripts/run_eval_all.sh
```

### Individual commands
```bash
# Fusion (2D only, no model weights needed)
python run/evaluate.py \
    --config config/nuscenes/ours_openseg_mini.yaml \
    TEST.feature_type fusion \
    TEST.save_folder exp/nusc_mini_openseg_img_viz/eval_fusion

# Distill (3D MinkUNet18A, pretrained weights)
python run/evaluate.py \
    --config config/nuscenes/ours_openseg_mini_pretrained.yaml \
    TEST.feature_type distill \
    TEST.save_folder exp/nusc_mini_openseg_img_viz/eval_distill

# Ensemble (confidence blend of Fusion + Distill)
python run/evaluate.py \
    --config config/nuscenes/ours_openseg_mini_pretrained.yaml \
    TEST.feature_type ensemble \
    TEST.save_folder exp/nusc_mini_openseg_img_viz/eval_ensemble
```

### Pretrained weights
Weights file: `models/nuscenes_openseg.pth.tar`
Download instructions: see `download.md`

---

## 5. Achieved Results (mini val: scene-0103, scene-0916)

| Mode | mIoU | mAcc | allAcc |
|------|------|------|--------|
| Fusion | 12.75% | 17.18% | 67.80% |
| Distill (pretrained) | 35.43% | 43.24% | 83.78% |
| Ensemble | 26.53% | 33.28% | 79.94% |

Distill matches the paper's reported 35.4% mIoU exactly.

---

## 6. Distillation Training (mini, smoke test)

```bash
python run/distill.py --config config/nuscenes/ours_openseg_mini.yaml
```

Outputs: `exp/nuscenes_mini_openseg/model/model_last.pth.tar`, `model_best.pth.tar`

---

## Key Paths

| What | Path |
|------|------|
| Raw nuScenes mini | `/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes_mini/` |
| 3D point clouds | `data/nuscenes_3d/{train,val}/` |
| 2D camera data | `data/nuscenes_2d/{train,val}/{scene}/` |
| Fused features | `data/nuscenes_multiview_openseg/` |
| OpenSeg model | `models/openseg/exported_model/` |
| Pretrained weights | `models/nuscenes_openseg.pth.tar` |
| Mini config | `config/nuscenes/ours_openseg_mini.yaml` |
| Pretrained config | `config/nuscenes/ours_openseg_mini_pretrained.yaml` |
| 4-GPU full config | `config/nuscenes/ours_openseg_full_4gpu.yaml` |
| Eval outputs | `exp/nuscenes_mini_openseg/` |

## Known Issues & Fixes

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: util` | `export PYTHONPATH=<repo_root>` |
| X11 display error (matplotlib) | `matplotlib.use('Agg')` already added to `util/util.py` |
| Workers hang (MIG GPU) | `workers: 0`, `test_workers: 0` in config |
| OOM at epoch 2 (MinkowskiEngine) | `batch_size: 1` on single GPU |
| `typing_extensions` AttributeError | `pip install --upgrade typing_extensions` |
| Mapper index out of bounds (43 classes) | Patched in `run/evaluate.py`: extended mapper tensor |
