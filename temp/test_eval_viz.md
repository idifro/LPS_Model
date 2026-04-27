# Test Commands: Evaluation & Visualization

## Prerequisites
```bash
# Start interactive GPU node
bsub -Is -q batch_a100 -n 4 -M 8192 -W 2:00 -gpu "num=1:mode=exclusive_process:mps=no" /bin/bash

# Activate environment
conda activate openscene

# Navigate to project root
cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
```

## 1. Evaluation (mini val — 81 samples)
```bash
python -u panoptic/scripts/evaluate_full.py \
    --checkpoint panoptic/checkpoints_full/best.pth \
    --version v1.0-mini \
    --dataroot /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes_mini \
    --split val \
    --save_dir panoptic/eval_results
```

**Expected output:**
- `panoptic/eval_results/eval_results.txt` — formatted per-class table
- `panoptic/eval_results/eval_results.json` — machine-readable metrics
- `panoptic/eval_results/confusion_matrix.npy` — 16×16 confusion matrix

## 2. Visualization (3 samples)
```bash
python -u panoptic/scripts/visualize_panoptic.py \
    --checkpoint panoptic/checkpoints_full/best.pth \
    --version v1.0-mini \
    --dataroot /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes_mini \
    --split val \
    --num_samples 3 \
    --save_dir panoptic/viz_panoptic
```

**Expected output:**
```
panoptic/viz_panoptic/
├── sample_000/
│   ├── composite_semantic_pred.png
│   ├── composite_panoptic_pred.png
│   ├── composite_gt.png
│   ├── pointcloud_semantic.ply
│   ├── pointcloud_panoptic.ply
│   ├── pointcloud_gt.ply
│   ├── pointcloud_instance.ply
│   └── bev_comparison.png
├── sample_001/ ...
├── sample_002/ ...
├── legend.png
└── summary.txt
```

## 3. Quick Smoke Test (1 sample, check for import/runtime errors)
```bash
python -u panoptic/scripts/visualize_panoptic.py \
    --checkpoint panoptic/checkpoints_full/best.pth \
    --version v1.0-mini \
    --dataroot /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes_mini \
    --num_samples 1 \
    --save_dir panoptic/viz_panoptic_test
```

## 4. Check Results
```bash
# View eval table
cat panoptic/eval_results/eval_results.txt

# Check viz files exist
ls -lh panoptic/viz_panoptic/sample_000/

# Check PLY file size (should be several MB)
du -sh panoptic/viz_panoptic/sample_000/*.ply
```
