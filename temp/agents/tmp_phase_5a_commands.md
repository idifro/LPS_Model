# Phase 5a (B2b) — All Run Commands

All commands assume CWD = `/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene`

---

## 1. Training

### 1a. Smoke test on mini (3 epochs, ~5 min)

```bash
python -u panoptic/train/train_panoptic_b2b.py \
    --config panoptic/configs/full_nuscenes_b2b.yaml \
    --override data.version=v1.0-mini \
    data.dataroot=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes_mini \
    training.epochs=3 \
    logging.eval_every=1 \
    logging.wandb_enabled=false \
    checkpoint.save_dir=panoptic/checkpoints_b2b_mini
```

**Expected**: loss_ce ≈ 0, loss_mask/loss_dice decreasing, backbone pooling eval runs, novel_voxels logged.

### 1b. Smoke test on full dataset (3 epochs, ~2.5 hr)

```bash
python -u panoptic/train/train_panoptic_b2b.py \
    --config panoptic/configs/full_nuscenes_b2b.yaml \
    --override training.epochs=3 \
    logging.eval_every=1 \
    logging.wandb_run_name=b2b-smoke-full \
    checkpoint.save_dir=panoptic/checkpoints_b2b_smoke
```

**Expected**: Same as mini but with full dataset. Verify no OOM, loss scale is reasonable.

### 1c. Full training via bsub (~17-20 hr)

```bash
bsub < panoptic/scripts/train_b2b_job.sh
```

Monitor:
```bash
bjobs -l        # Check job status
bpeek           # Peek at stdout
tail -f panoptic/logs/train_b2b_*.stdout  # Follow logs
```

---

## 2. Evaluation

### 2a. Evaluate on mini (after smoke test)

```bash
python -u panoptic/scripts/evaluate_b2b.py \
    --config panoptic/configs/full_nuscenes_b2b.yaml \
    --checkpoint panoptic/checkpoints_b2b_mini/best.pth \
    --version v1.0-mini \
    --dataroot /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes_mini \
    --output_dir panoptic/eval_results/b2b_mini/ \
    --obj_thresholds 0.1,0.2,0.3,0.5
```

### 2b. Evaluate on full val (after full training)

```bash
python -u panoptic/scripts/evaluate_b2b.py \
    --config panoptic/configs/full_nuscenes_b2b.yaml \
    --checkpoint panoptic/checkpoints_b2b/best.pth \
    --version v1.0-trainval \
    --dataroot /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes \
    --output_dir panoptic/eval_results/b2b_full/ \
    --obj_thresholds 0.1,0.2,0.3,0.5
```

### 2c. Also run comprehensive eval (all 5 original modes + B2b checkpoint)

```bash
python -u panoptic/scripts/evaluate_comprehensive.py \
    --config panoptic/configs/full_nuscenes_b2b.yaml \
    --checkpoint panoptic/checkpoints_b2b/best.pth \
    --version v1.0-trainval \
    --modes linear,backbone_openvocab \
    --output_dir panoptic/eval_results/b2b_comprehensive/
```

---

## 3. Visualization

### 3a. Visualize on mini (5 samples)

```bash
python -u panoptic/scripts/visualize_b2b.py \
    --config panoptic/configs/full_nuscenes_b2b.yaml \
    --checkpoint panoptic/checkpoints_b2b_mini/best.pth \
    --version v1.0-mini \
    --dataroot /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes_mini \
    --num_samples 5 \
    --save_dir panoptic/viz_b2b_mini \
    --obj_thr 0.3
```

### 3b. Visualize on full val (5 samples)

```bash
python -u panoptic/scripts/visualize_b2b.py \
    --config panoptic/configs/full_nuscenes_b2b.yaml \
    --checkpoint panoptic/checkpoints_b2b/best.pth \
    --version v1.0-trainval \
    --dataroot /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes \
    --num_samples 5 \
    --save_dir panoptic/viz_b2b_full \
    --obj_thr 0.3
```

### 3c. Visualize with original panoptic script (Phase 4 comparison)

```bash
python -u panoptic/scripts/visualize_panoptic.py \
    --checkpoint panoptic/checkpoints_full/best.pth \
    --version v1.0-trainval \
    --dataroot /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes \
    --num_samples 5 \
    --save_dir panoptic/viz_phase4_comparison
```

---

## 4. Output Locations

| Output | Path |
|--------|------|
| Mini checkpoints | `panoptic/checkpoints_b2b_mini/` |
| Full checkpoints | `panoptic/checkpoints_b2b/` |
| Mini eval results | `panoptic/eval_results/b2b_mini/` |
| Full eval results | `panoptic/eval_results/b2b_full/` |
| Mini visualizations | `panoptic/viz_b2b_mini/` |
| Full visualizations | `panoptic/viz_b2b_full/` |
| Training logs | `panoptic/logs/train_b2b_*.stdout` |

---

## 5. Key Metrics to Check

After training, verify:

1. **loss_ce ≈ 0** — confirms weight_ce=0 is working
2. **loss_mask and loss_dice decreasing** — mask quality improving
3. **val/PQ > 0.30** — acceptable known-class quality
4. **val/novel_voxels > 0** — THE key test: novel classes detected
5. **Per-novel-class counts** — which novel classes are detected?
6. **Visualization novel_only images** — do novel detections look reasonable?
