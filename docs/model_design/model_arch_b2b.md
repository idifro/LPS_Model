# Phase 5a — Class-Agnostic Panoptic Segmentation (B2b)

## 1. Overview

Phase 5a trains the **same decoder architecture** as Phase 4, but with
classification loss **disabled** (`weight_ce=0.0`, `cost_class=0.0`). The
decoder learns to predict *mask geometry only*, without learning class-specific
query specialization. At inference, predicted masks are classified via
**backbone feature pooling** against 22 CLIP text prototypes (16 known nuScenes
classes + 6 novel driving-relevant classes), enabling true **open-vocabulary**
panoptic segmentation.

**Hypothesis**: Removing classification loss allows queries to specialize purely
on spatial mask quality, and backbone features (from frozen DisNet/OpenSeg)
retain sufficient semantic information for post-hoc open-vocabulary
classification via CLIP text prototypes.

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Disable CE loss | `weight_ce=0.0` in config | Zero code changes to loss/decoder; weight multiplication produces zero gradient |
| Disable class matching | `cost_class=0.0` in config | Matcher uses only BCE + Dice for query-GT assignment |
| Keep `class_embed` in model | Exists but unused | Backward compatible; no architecture change from Phase 4 |
| Inference classifier | Backbone pooling + cosine | Backbone features are CLIP-aligned (DisNet/OpenSeg), cosine similarity to text prototypes is natural |
| Objectness scoring | `max(sigmoid(mask))` per query | Simple, no extra head needed; high-quality masks have strong sigmoid peaks |
| Novel classes | 6 driving-relevant | pole, bench, trash can, fire hydrant, tree trunk, curb |

---

## 2. End-to-End Data Flow

```
    ┌─────────────────────────────────────────────────────────────────────┐
    │                     TRAINING DATA FLOW                              │
    │                                                                     │
    │  LiDAR (N,5)                                                        │
    │    │                                                                │
    │    ├─ range filter ─► augment ─► voxelize ─► (V,3) coords           │
    │    │                                         (V,3) XYZ feats        │
    │    │                                                                │
    │    ▼                                                                │
    │  Frozen DisNet (MinkUNet18A)                                        │
    │    │                                                                │
    │    ▼                                                                │
    │  backbone_feats (V, 768) ──── backbone_proj(768→256) ───► src       │
    │                          ──── mask_feat_proj(768→256) ──► mask_feats │
    │                                                                     │
    │  queries (106, 256) ──────────────────────────────────────┐         │
    │    │                                                      │         │
    │    ▼                                                      ▼         │
    │  9 decoder layers (cross-attn + self-attn + FFN)                    │
    │    │                                                                │
    │    ├──► class_embed(256→17) → pred_logits ──(IGNORED, zero weight)  │
    │    ├──► mask_embed(256→256) → einsum → pred_masks ──────────────►   │
    │    │                                                                │
    │  LOSS:                                                              │
    │    loss_ce  × 0.0 = 0    (no gradient)                              │
    │    loss_mask × 5.0       (sigmoid BCE on matched masks)             │
    │    loss_dice × 5.0       (Dice loss on matched masks)               │
    │                                                                     │
    │  MATCHER:                                                           │
    │    C = 0.0×class_cost + 5.0×mask_cost + 5.0×dice_cost              │
    │    → Purely geometry-based Hungarian matching                       │
    └─────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │                     INFERENCE DATA FLOW                             │
    │                                                                     │
    │  Forward pass → pred_masks (V, 106)  [pred_logits IGNORED]          │
    │    │                                                                │
    │    ├─ objectness = max(sigmoid(mask_q)) per query                   │
    │    ├─ filter: objectness > 0.3                                      │
    │    │                                                                │
    │    ▼                                                                │
    │  For each kept query q:                                             │
    │    binary_mask = sigmoid(pred_masks[:,q]) >= 0.5                    │
    │    pooled = mean(backbone_feats[binary_mask])  (768-d)              │
    │    pooled = L2_normalize(pooled)                                    │
    │    cos_sim = pooled @ text_protos_22.T   (22 scores)                │
    │    label = argmax(cos_sim)  ← can be 0-15 (known) or 16-21 (novel) │
    │    score = objectness[q] × cos_sim[label]                           │
    │    │                                                                │
    │    ▼                                                                │
    │  Greedy panoptic assignment (sorted by score descending)            │
    │    thing_set = {0..9} ∪ {16..21}                                    │
    │    stuff_set = {10..15}                                             │
    │    │                                                                │
    │    ▼                                                                │
    │  sem_pred (V,)  inst_pred (V,)  ← 22-class panoptic output         │
    └─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Modules

### 3a. PanopticDecoder (unchanged from Phase 4)

Same architecture: 106 queries (100 thing + 6 stuff), 9 decoder layers
(3 blocks × 3 virtual levels), `backbone_proj` (768→256), `mask_feat_proj`
(768→256), `class_embed` (256→17), `mask_embed` MLP(256→256, 3 layers).

`class_embed` still exists and produces `pred_logits` at every layer, but these
are never used at inference and receive zero gradient during training.

| Parameter | Value |
|-----------|-------|
| Trainable parameters | ~10.5M |
| `class_embed` parameters | 256×17 + 17 = 4,369 (wasted, but negligible) |
| Query design | 100 learnable thing + 6 learnable stuff |
| Attention mask | `sigmoid(mask) < 0.5` → True (masked) |

### 3b. Backbone Feature Pooling (inference-only)

Per predicted mask, mean-pool the frozen backbone features:
```
pooled_q = mean(backbone_feats[binary_mask_q])  → (768,)
pooled_q = L2_normalize(pooled_q)
```

This works because the frozen DisNet backbone produces OpenSeg-aligned features
that are semantically meaningful in CLIP's embedding space.

### 3c. Text Prototypes

22-class CLIP text prototypes generated with ViT-L/14@336px:
```
template = "a point cloud of a {class_name}"
```

| Class range | Names | Role |
|-------------|-------|------|
| 0-9 | barrier, bicycle, bus, car, construction, motorcycle, pedestrian, traffic cone, trailer, truck | Known things |
| 10-15 | driveable surface, other flat, sidewalk, terrain, manmade, vegetation | Known stuff |
| 16-21 | pole, bench, trash can, fire hydrant, tree trunk, curb | Novel things |

---

## 4. Loss Computation

### Active losses (mask-only)

| Loss | Formula | Weight | Role |
|------|---------|--------|------|
| `loss_mask` | sigmoid BCE on point-sampled matched masks | 5.0 | Binary mask quality |
| `loss_dice` | Dice loss on point-sampled matched masks | 5.0 | Set overlap quality |

### Disabled losses

| Loss | Weight | Effect |
|------|--------|--------|
| `loss_ce` | **0.0** | Cross-entropy still computes in forward pass but multiplied by zero → no gradient |

### Auxiliary losses

Same structure as Phase 4: loss applied at all 9 intermediate decoder layers +
1 initial prediction = 10 sets of (mask + dice) losses. All weighted equally.

---

## 5. Matcher (geometry-only)

Hungarian matching cost:
```
C = 0.0 × cost_class + 5.0 × cost_mask_bce + 5.0 × cost_dice
```

`cost_class` computes `-softmax(pred_logits)[:, gt_class]` but multiplied by
zero, so the assignment is purely based on mask geometry (BCE + Dice costs).

**Implication**: Queries are matched to ground truth instances based on how well
their predicted mask overlaps the target, without any class preference. This
means a query originally "specialized" for cars can be matched to a pedestrian
if its mask better fits that pedestrian.

---

## 6. Inference Pipeline

1. **Forward pass**: `decoder(backbone_feats, coords)` → `pred_masks (V, 106)`
2. **Objectness**: `objectness_q = max(sigmoid(pred_masks[:, q]))` per query
3. **Filter**: Keep queries with `objectness > 0.3` and `binary_mask.sum() >= 10`
4. **Pool**: For each kept query, mean-pool backbone features at `sigmoid >= 0.5`
5. **Classify**: L2-normalize, cosine sim to 22 prototypes, argmax
6. **Score**: `final_score = objectness × max_cosine_similarity`
7. **Greedy assign**: Sort by score descending, assign non-overlapping masks
   - Things (0-9, 16-21): each instance gets unique ID
   - Stuff (10-15): all instances of same class merged

---

## 7. Evaluation Metrics

### Known-class PQ (primary metric)

Novel predictions (class >= 16) are remapped to background (class 0, inst 0)
before computing PQ against 16-class ground truth. This measures: **how well
does class-agnostic mask learning + backbone pooling preserve known-class quality?**

| Metric | Definition |
|--------|------------|
| PQ (Panoptic Quality) | TP/(TP + 0.5×FP + 0.5×FN) × mean_IoU_of_TPs |
| SQ (Segmentation Quality) | Mean IoU of matched (TP) segments |
| RQ (Recognition Quality) | TP/(TP + 0.5×FP + 0.5×FN) |
| mIoU | Mean IoU across 16 classes |

### Novel class detections

| Metric | Definition |
|--------|------------|
| `novel_voxels` | Total voxels assigned to classes 16-21 |
| Per-class count | Voxels per novel class (pole, bench, etc.) |

### Evaluation modes

| Mode | Prototypes | Objectness | Description |
|------|-----------|------------|-------------|
| `pool22_obj{thr}` | 22-class | Yes, sweep threshold | Primary: objectness × cosine, varying sensitivity |
| `pool22_no_obj` | 22-class | No (all queries) | Pure cosine baseline |
| `pool16_obj0.3` | 16-class only | Yes, thr=0.3 | Upper bound: no novel confusion |

---

## 8. Training Loop

### Warm start

Load Phase 4 best checkpoint (`best.pth`, PQ=0.473) with `strict=True`.
Architecture is identical — all weights transfer exactly.

### Optimizer

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| Weight decay | 0.01 |
| Grad clip | 0.5 |
| Grad accumulation | 8 |
| Mixed precision | Yes (AMP) |

### Scheduler

| Parameter | Value |
|-----------|-------|
| Type | Cosine warmup |
| Warmup epochs | 2 |
| Total epochs | 20 |
| Min LR | 1e-6 |

---

## 9. Training Arithmetic

| Metric | Value |
|--------|-------|
| Training samples | 28,130 (v1.0-trainval train split) |
| Validation samples | 6,019 |
| Batch size | 1 (per GPU) |
| Grad accumulation | 8 → effective batch = 8 |
| Steps per epoch | 28,130 / 8 = 3,516 |
| Total epochs | 20 |
| Total optimizer steps | 70,320 |
| Estimated time per epoch | ~50 min (A100) |
| Estimated total training | ~17 hours |
| Eval every | 2 epochs |
| Checkpoint every | 2 epochs |

---

## 10. Dimension Summary

| Tensor | Shape | Notes |
|--------|-------|-------|
| Raw LiDAR | (N, 5) | x,y,z,intensity,ring |
| After range filter | (~30k, 5) | |
| Voxel coords | (V, 3) int32 | V ≈ 25-35k |
| Voxel features | (V, 3) float32 | XYZ |
| Backbone features | (1, V, 768) | Frozen DisNet output |
| src (projected) | (1, V, 256) | `backbone_proj(768→256)` |
| mask_features | (1, V, 256) | `mask_feat_proj(768→256) + PE` |
| queries | (1, 106, 256) | 100 thing + 6 stuff |
| pred_logits | (1, 106, 17) | **Produced but IGNORED** |
| pred_masks | (1, V, 106) | Used for inference |
| objectness | (106,) | `max(sigmoid(mask))` per query |
| pooled features | (Qv, 768) | Mean backbone feats per valid query |
| text prototypes | (22, 768) | CLIP text embeddings |
| cosine similarity | (Qv, 22) | Classification scores |
| sem_pred | (V,) int64 | 0-21 class labels |
| inst_pred | (V,) int64 | Instance IDs |

---

## 11. Design Choices & Rationale

### Architecture decisions

| Decision | Choice | Alternative | Why |
|----------|--------|-------------|-----|
| Zero CE vs binary objectness head | Zero CE (config-only) | Add Linear(256,2) objectness head | Zero code changes; class_embed wasted params are negligible (4,369 / 10.5M = 0.04%) |
| Objectness signal | `max(sigmoid(mask))` | Separate learned score | Simpler; mask confidence is already a proxy for query quality |
| Pooling strategy | Mean pool per mask | Attention pool, max pool | Mean is simple and works well in evaluate_comprehensive.py Mode E |
| Score combination | `objectness × cosine` | `cosine` only, learned fusion | Product naturally down-weights low-confidence masks |
| Novel class taxonomy | 6 driving-relevant | More classes, different taxonomy | Focused on actionable driving objects missing from nuScenes-16 |

### Why not train a binary objectness head?

- Adds training complexity (new loss, new head, hyperparameter tuning)
- `max(sigmoid(mask))` is free — already computed during mask prediction
- Phase 4's mask quality is high (PQ=0.473); we just need filtering, not scoring

### Why not remove `class_embed` from the code?

- Would require modifying `mask_decoder.py`, `panoptic_loss.py`, and all scripts
- Breaking change for existing checkpoints
- Zero-weight approach is 100% backward compatible

---

## 12. File Map

| File | Purpose |
|------|---------|
| `panoptic/configs/full_nuscenes_b2b.yaml` | Config: weight_ce=0, cost_class=0, warm-start |
| `panoptic/train/train_panoptic_b2b.py` | Training script with class-agnostic eval |
| `panoptic/scripts/evaluate_b2b.py` | Comprehensive eval (3 modes, obj_thr sweep) |
| `panoptic/scripts/visualize_b2b.py` | Per-camera overlay with 22-class panoptic + novel highlights |
| `panoptic/scripts/train_b2b_job.sh` | HPC bsub job script |
| `docs/model_design/model_arch_b2b.md` | This document |
| `panoptic/embeddings/text_prototypes_22.pt` | 22-class CLIP text prototypes |

### Unchanged files (config-driven)

| File | Why unchanged |
|------|---------------|
| `panoptic/models/mask_decoder.py` | `class_embed` exists but weight_ce=0 → no gradient |
| `panoptic/losses/panoptic_loss.py` | `weight_ce=0.0` multiplication → zero gradient |
| `panoptic/losses/matcher.py` | `cost_class=0.0` → geometry-only matching |
| `panoptic/dataset/nuscenes_panoptic.py` | Same data, same format |

---

## 13. Config Reference

```yaml
# Full config: panoptic/configs/full_nuscenes_b2b.yaml
data:
  dataroot: /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes
  version: v1.0-trainval
  voxel_size: 0.05
  point_range: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

model:
  backbone_dim: 768
  hidden_dim: 256
  num_classes: 16
  num_thing_queries: 100
  num_stuff_classes: 6
  use_token_head: false

loss:
  weight_ce: 0.0       # DISABLED
  cost_class: 0.0       # DISABLED
  weight_mask: 5.0
  weight_dice: 5.0
  cost_mask: 5.0
  cost_dice: 5.0

training:
  epochs: 20
  lr: 5.0e-5

warm_start:
  checkpoint: panoptic/checkpoints_full/best.pth

open_vocab:
  prototypes_path: panoptic/embeddings/text_prototypes_22.pt
  objectness_thr: 0.3
  novel_class_names: [pole, bench, trash can, fire hydrant, tree trunk, curb]
```

---

## 14. Experimental Results

*[To be filled after training]*

### Expected outcomes

| Metric | Expected range | Reasoning |
|--------|---------------|-----------|
| PQ_known_16 | 0.35 – 0.42 | Lower than Phase 4 (0.473) due to class-agnostic matching + backbone pooling classification noise |
| novel_voxels | > 0 | The key test: can backbone pooling detect novel classes when CE bias is removed? |
| SQ | ≈ Phase 4 | Mask quality should be similar (same mask/dice loss) |
| RQ | Lower than Phase 4 | More FP/FN without class-guided matching |

### Comparison baselines

| Experiment | PQ_known | Novel | Notes |
|------------|----------|-------|-------|
| Phase 4 (linear head) | 0.473 | N/A | Standard 16-class |
| Phase 5 v2 (token head) | 0.476 | 0 | Token collapsed to closed-set |
| Phase 5 v2 Mode E (backbone pool) | 0.209 | 25M (noise) | Raw backbone pool on Phase 5 checkpoint |
| **Phase 5a (B2b)** | *TBD* | *TBD* | Class-agnostic training |

### WandB metrics logged

| Key | Description |
|-----|-------------|
| `train/loss` | Total loss (mask + dice only, CE ≈ 0) |
| `train/loss_ce` | Should be ≈ 0 or very small |
| `train/loss_mask` | Sigmoid BCE mask loss |
| `train/loss_dice` | Dice loss |
| `train/lr` | Current learning rate |
| `val/PQ` | PQ on 16 known classes (novel → ignore) |
| `val/SQ` | Segmentation quality |
| `val/RQ` | Recognition quality |
| `val/mIoU` | Mean IoU over 16 known |
| `val/novel_voxels` | Total novel voxels detected |
| `val/novel_{class}` | Per-novel-class voxel counts |
| `epoch/gpu_peak_mb` | Peak GPU memory |
