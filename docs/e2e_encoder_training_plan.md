# End-to-End Encoder Training Plan

## Objective

Validate and execute **end-to-end (E2E) encoder unfreezing** for OpenScene distillation on nuScenes, starting with a **mini dataset smoke test** before scaling to the full dataset.

---

## Current Baseline

| Item | Value |
|------|-------|
| Model | DisNet → MinkUNet18A (sparse 3D U-Net) |
| Output dim | 768 (CLIP ViT-L/14@336px) |
| Training loss | Cosine similarity between predicted 3D features and fused 2D OpenSeg features |
| Existing mini config | `config/nuscenes/ours_openseg_mini.yaml` |
| Existing full config | `config/nuscenes/ours_openseg_full_4gpu.yaml` |
| GPU job script | `scripts/train_full_nuscenes.bsub` |
| Training entry | `run/distill.py` via `run/distill.sh` |

---

## Plan: 3 Phases

### Phase 1 — Smoke Test on nuScenes Mini (THIS FIRST)

**Goal:** Confirm training pipeline works, observe loss curve and mIoU, generate PLY visualizations.

**Config:** `config/nuscenes/ours_openseg_mini.yaml` (already exists)

**Steps:**
1. Ensure data exists at `data/nuscenes_3d/` and `data/nuscenes_multiview_openseg/`
2. Run training:
   ```bash
   bash run/distill.sh exp/mini_e2e_smoke config/nuscenes/ours_openseg_mini.yaml
   ```
3. Check outputs:
   - Loss: TensorBoard at `exp/mini_e2e_smoke/`
   - mIoU: printed per epoch (eval_freq=1)
   - PLY files: `exp/mini_e2e_smoke/result/last/` (pred, gt, openseg colored)
4. **Success criteria:** Loss decreases, mIoU > 0 (any non-trivial segmentation)

**Expected runtime:** ~5-10 min on 1 GPU (8 train scenes, 5 epochs, batch=1)

---

### Phase 2 — E2E Mini with Pretrained Init (Encoder Unfreezing Experiment)

**Goal:** Start from the existing pretrained checkpoint (`models/nuscenes_openseg.pth.tar`), unfreeze encoder, fine-tune with lower LR. Compare against frozen baseline.

**New config:** `config/nuscenes/ours_openseg_mini_e2e.yaml`

**Key differences from baseline:**
- `resume: models/nuscenes_openseg.pth.tar` (warm start)
- `base_lr: 0.00001` (10x lower — encoder is already trained)
- `epochs: 10` (more epochs to observe if fine-tuning helps or hurts)
- `save_path: exp/mini_e2e_finetune`

**Analysis after run:**
- Compare mIoU: Phase 1 (from scratch) vs Phase 2 (pretrained + E2E fine-tune)
- Compare PLY visualizations side-by-side
- Check if cosine loss continues to decrease or plateaus

---

### Phase 3 — Full nuScenes E2E Training

**Goal:** Scale the best strategy from Phase 2 to the full dataset on multi-GPU.

**New config:** `config/nuscenes/ours_openseg_full_e2e.yaml`
**New job script:** `scripts/train_full_nuscenes_e2e.bsub`

**Only proceed if Phase 2 shows improvement over Phase 1.**

---

## File Creation Plan

| File | Purpose |
|------|---------|
| `config/nuscenes/ours_openseg_mini_e2e.yaml` | Mini E2E fine-tuning config |
| `config/nuscenes/ours_openseg_full_e2e.yaml` | Full E2E training config |
| `scripts/train_full_nuscenes_e2e.bsub` | LSF batch job for full E2E |

---

## How to Observe / Visualize Results

1. **TensorBoard:**
   ```bash
   tensorboard --logdir exp/ --port 6006
   ```
2. **PLY files:** Open in MeshLab or CloudCompare
   - `openseg_<epoch>.ply` — what the 2D teacher "sees"
   - `pred_<epoch>.ply` — what the 3D student predicts
   - `gt_<epoch>.ply` — ground truth labels
3. **Console logs:** mIoU/mAcc/allAcc printed every `eval_freq` epochs

---

## Key Metrics to Track

| Metric | Where | Good Sign |
|--------|-------|-----------|
| Cosine loss | TensorBoard `loss_train` | Decreasing |
| mIoU | Console + TensorBoard `mIoU_val` | Increasing |
| mAcc | Console | Increasing |
| PLY quality | Visual inspection | Pred matches GT |

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Encoder diverges with fine-tuning | Use very low LR (1e-5) for encoder |
| Memory OOM on full training | Batch=1, gradient accumulation if needed |
| MinkowskiEngine fragmentation | `torch.cuda.empty_cache()` already in distill loop |
| Pretrained checkpoint format mismatch | `strict=True` in `load_state_dict` will catch this |

---

## Execution Order (Copy-Paste Commands)

```bash
# Step 1: Smoke test (existing config, no changes needed)
bash run/distill.sh exp/mini_smoke config/nuscenes/ours_openseg_mini.yaml

# Step 2: E2E fine-tune on mini (after creating config)
bash run/distill.sh exp/mini_e2e_finetune config/nuscenes/ours_openseg_mini_e2e.yaml

# Step 3: Full E2E (submit LSF job)
bsub < scripts/train_full_nuscenes_e2e.bsub
```
