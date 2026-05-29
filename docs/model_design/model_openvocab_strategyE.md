# Strategy E — Geometric Ensemble for Open-Vocabulary Panoptic Segmentation

## 1. Overview

Strategy E enables open-vocabulary panoptic segmentation by combining the strong known-class performance of the linear head (PQ=0.476) with the open-vocabulary capability of backbone-pooled CLIP features. No additional training is required — this is a purely inference-time method that uses the existing checkpoint (`checkpoints_token_full_v2/best.pth`).

**Key insight**: The frozen DisNet backbone produces 768-dim features distilled from CLIP ViT-L/14@336px. By normalizing and soft-pooling these features through predicted masks, we obtain per-mask CLIP embeddings that can be compared to arbitrary text prototypes — including classes never seen during panoptic training.

## 2. Architecture Diagram

```
Frozen DisNet Backbone (768-d)
         │
    ┌────┴────┐
    │         │
    ▼         ▼
 backbone_proj(768→256)    raw_feats (V, 768) ← used by Strategy E
         │
    Transformer Decoder (9 layers)
         │
    ┌────┼────┐
    │    │    │
    ▼    ▼    ▼
class_embed  mask_embed  token_head
(256→17)     MLP(256→256) MLP(256→768)
    │            │
pred_logits  pred_masks (V, Q)
(Q, 17)          │
    │            │
    │      ┌─────┴──────────────────────┐
    │      │                            │
    │      ▼                            ▼
    │   sigmoid → soft weights     raw_feats (V, 768)
    │      │                            │
    │      └─── weighted pool ──────────┘
    │                   │
    │              L2-normalize
    │                   │
    │           pooled_feats (Q, 768)
    │                   │
    │           cos_sim / temperature
    │                   │
    │           softmax → clip_probs (Q, C_all)
    │                   │
    ▼                   ▼
linear_probs^α  ×  clip_probs^(1-α)   [known classes]
                   clip_probs only      [novel classes]
                        │
                  combined scores (Q, C_all)
                        │
                  × objectness (1 - P(no-object))
                        │
                  greedy panoptic assignment
```

## 3. The Bug Fix (Mode E → Mode F)

### What was wrong (PQ=0.209)

The legacy `panoptic_inference_backbone_pool` had three critical bugs:

1. **No pre-normalization**: Raw backbone features have norm range [10, 900]. Mean-pooling without normalization causes large-norm voxels to dominate the average, producing representations biased toward high-activation regions rather than semantic content.

2. **Binary mask threshold**: Using `mask >= 0.5` loses gradient information from soft masks. Voxels with probability 0.49 contribute zero weight while those at 0.51 contribute equally to 1.0.

3. **No objectness filtering**: Every query produces a pooled embedding, even those the decoder classified as "no-object". Background queries get random pooled features that happen to be closest to one prototype, causing massive false-positive predictions (25M voxels as "pole").

### The fix (Mode F: `panoptic_inference_backbone_pool_v2`)

```python
# 1. L2-normalize BEFORE pooling → unit sphere, equal contribution
backbone_normed = F.normalize(backbone_feats.float(), dim=-1)

# 2. Soft sigmoid weights → smooth, differentiable pooling
mask_probs = pred_masks.sigmoid()
weights = mask_probs / mask_area.clamp(min=1e-6).unsqueeze(0)

# 3. Weighted average + re-normalize
pooled = F.normalize(weights.T @ backbone_normed, dim=-1)

# 4. Filter by mask area → objectness proxy
scores[mask_area < min_mask_pts] = -1.0
```

## 4. Geometric Ensemble (Mode G)

### Formulation

For a query with predicted class scores:

```
Known classes (c < 16):
    score(c) = linear_prob(c)^α × clip_prob(c)^(1-α)

Novel classes (c ≥ 16):
    score(c) = clip_prob(c)

Final score = max_c(score(c)) × objectness
objectness = 1 - P(no-object)   [from linear head softmax]
```

### Why geometric mean?

- **Multiplicative**: Both heads must agree for high confidence — reduces false positives
- **Scale-invariant**: Handles the different magnitude scales of linear softmax vs cosine similarity
- **Graceful degradation**: If α=1.0, reverts to pure linear; if α=0.0, reverts to pure CLIP pool
- **Novel class pass-through**: Linear head has no opinion on novel classes, so CLIP probability flows through unattenuated

### Temperature parameter

The CLIP logits are divided by temperature τ before softmax:
```
clip_probs = softmax(cos_sim / τ)
```

- **τ=0.1** (default): Sharp distribution — forces high confidence, reduces ambiguity
- **τ=0.07**: CLIP default, even sharper
- **τ=1.0**: Flat distribution — less discriminative

Lower temperature works better because backbone-pooled features have cosine similarities clustered around 0.2-0.4. Without temperature scaling, the softmax would produce near-uniform distributions.

## 5. Novel Classes

### Default set
```
["police car", "ambulance", "trash cans", "pole", "bench"]
```

Driving-relevant objects not in the nuScenes-16 panoptic label set. Chosen because:
- They appear in driving scenes (potential real detections)
- They are visually distinct from known classes
- CLIP has strong text-image associations for these concepts

### Prototype generation
Uses the same CLIP model (ViT-L/14@336px) and prompt template as the known classes:
```
prompt = "a {class_name} in a scene"
```

Prototypes are generated once and cached to disk. The full prototype file contains `(16 + N_novel, 768)` embeddings.

### Configuration

Three ways to specify novel classes (in priority order):
1. CLI: `--novel-classes "police car,ambulance,trash cans,pole,bench"`
2. Config YAML: `novel_classes: ["police car", "ambulance", ...]`
3. Default: `DEFAULT_NOVEL_CLASS_NAMES` in script

## 6. Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.5 | Geometric blend weight (1.0 = pure linear, 0.0 = pure CLIP) |
| `temperature` | 0.1 | Softmax temperature for CLIP logits |
| `min_mask_pts` | 50 | Minimum soft mask area for valid queries |
| `overlap_thr` | 0.8 | Greedy assignment overlap threshold |

The evaluation script sweeps `alpha ∈ {0.3, 0.5, 0.7}` and reports the best.

## 7. Evaluation Modes (Final — Cleaned 2026-05-27)

Removed legacy/redundant modes (C, D, E). Current eval script has 5 modes:

| Mode | Method | Classes | Purpose |
|------|--------|---------|---------|
| A. Linear | pred_logits softmax | 16 known | Baseline (PQ=0.476) |
| B. Token-16 | pred_tokens cos sim | 16 known | Token head quality diagnostic |
| F. Backbone-Pool-v2 | normalized soft-pool | 16+N | CLIP pool quality (ablation) |
| G. Geometric-Ensemble | linear^α × clip^(1-α) | 16+N | Strategy E v1 |
| **H. Geo-Ensemble-v2** | multi-prompt + bias correction | 16+N | **Strategy E v2 (best)** |

**Removed modes** (2026-05-27):
- C (ensemble): Best at α=0.0 = pure token, blending adds nothing
- D (token_openvocab): Zero novel detections, collapsed to 16 classes
- E (backbone_openvocab): Legacy buggy (PQ=0.209)

## 8. Usage

```bash
# Full evaluation (5 modes)
python -u panoptic/scripts/evaluate_comprehensive.py \
    --config panoptic/configs/full_nuscenes_token_v2.yaml \
    --checkpoint panoptic/checkpoints_token_full_v2/best.pth \
    --novel-classes "police car,ambulance,trash cans,pole,bench"

# Only geo_ensemble_v2 (fastest, best novel detection)
python -u panoptic/scripts/evaluate_comprehensive.py \
    --config panoptic/configs/full_nuscenes_token_v2.yaml \
    --checkpoint panoptic/checkpoints_token_full_v2/best.pth \
    --modes geo_ensemble_v2 \
    --geo_alphas 0.3,0.5,0.7

# Visual verification
python -u panoptic/scripts/visualize_novel_detections.py \
    --config panoptic/configs/full_nuscenes_token_v2.yaml \
    --checkpoint panoptic/checkpoints_token_full_v2/best.pth \
    --version v1.0-trainval \
    --only_with_detections \
    --max_samples 200 --num_samples 30 --alpha 0.3
```

## 9. Results — Run 1 (v1.0-trainval, 6019 samples, 2026-05-26)

### 9.1 Summary Table (geo_ensemble v1 only)

| Mode | PQ | SQ | RQ | mIoU |
|------|-----|-----|-----|------|
| A. Linear | **0.4772** | 0.7926 | **0.5844** | **0.4324** |
| B. Token | 0.4711 | 0.7963 | 0.5771 | 0.4208 |
| F. Backbone-Pool-v2 | 0.2208 | 0.6698 | 0.2600 | 0.1722 |
| **G. Geo-Ensemble α=0.5** | **0.4632** | 0.7943 | 0.5657 | 0.4224 |

### 9.2 Novel Class Detections (Run 1)

| Novel Class | Backbone-Pool-v2 | Geometric Ensemble (α=0.3) |
|-------------|-------------------|---------------------------|
| police car | 0 | **748** |
| ambulance | 0 | 0 |
| trash cans | 42,018 | 7,497 |
| pole | 23,878,854 | 162,804 |
| bench | 0 | 0 |

## 10. Results — Run 2 (v1.0-trainval, 6019 samples, 2026-05-27)

### 10.1 Changes from Run 1
- Added Mode H: `geo_ensemble_v2` with multi-prompt ensembled prototypes + automatic prototype bias correction
- Removed modes C, D, E (cleaned up)
- Fixed OOM: CLIP now loads on GPU before backbone (sequential, not concurrent)
- Fixed novel voxel summary display bug
- Fixed coords bug (reverted — model was trained with vox_feats as coords)

### 10.2 Full Summary Table

| Mode | PQ | SQ | RQ | mIoU | Novel Voxels |
|------|-----|-----|-----|------|------|
| A. Linear | **0.4761** | 0.7930 | 0.5828 | 0.4323 | N/A |
| B. Token | 0.4717 | 0.7969 | 0.5775 | 0.4206 | N/A |
| F. Backbone-Pool-v2 | 0.2207 | 0.6702 | 0.2597 | 0.1722 | 23,909,330 |
| G. Geo-Ensemble α=0.5 | 0.4642 | 0.7960 | 0.5664 | 0.4219 | — |
| **H. Geo-V2 α=0.5** | **0.4681** | 0.7930 | 0.5734 | **0.4283** | **396,229** |

### 10.3 Novel Class Detections (Run 2)

| Novel Class | Backbone-Pool-v2 | Geo-Ensemble v1 | **Geo-V2** |
|-------------|-------------------|-----------------|-----------|
| police car | 0 | 726 | **24,114** |
| ambulance | 0 | 0 | **352,234** |
| trash cans | 41,805 | 7,249 | **17,566** |
| pole | 23,867,525 | 165,142 | **280** |
| bench | 0 | 0 | **2,035** |

### 10.4 Analysis

**Improvements from v2 (multi-prompt + bias correction):**
1. **Pole catastrophe SOLVED**: 165K → 280 voxels (bias correction suppressed generic prototype)
2. **Novel detection increased**: 396K total novel voxels vs 173K in v1
3. **PQ retained at 98.3%** of linear baseline (0.4681/0.4761)
4. **Best overall mode**: geo_v2 α=0.5 gives best PQ (0.4681) + mIoU (0.4283) + novel detections
5. Alpha sensitivity is very low for v2 (0.4679/0.4681/0.4678 across α=0.3/0.5/0.7)

**Multi-prompt prototypes** (5-6 diverse descriptions per novel class):
- Made "police car" 32× more detectable (748 → 24K voxels)
- Enabled "ambulance" detection (0 → 352K voxels)
- Enabled "bench" detection (0 → 2K voxels)

**Bias correction** (per-prototype background affinity subtraction):
- Range: ±0.08 (pole=0.078 highest, bicycle=-0.083 lowest)
- Eliminated pole catastrophe entirely (165K → 280)
- Formula: `corrected_logit = (cos_sim - bias) / temperature`

## 11. Visual Verification Results (2026-05-27)

### Finding: many novel detections are false positives

Visual inspection of geo_ensemble_v2 camera overlays revealed:
- **"Ambulance" (352K voxels)**: White trucks, vans, walls, and buildings misclassified
- **"Police car" (24K voxels)**: Dark cars, SUVs, and vehicles misclassified
- **"Trash cans" (17K voxels)**: Various objects near roads misclassified
- **"Bench" (2K voxels)**: Random structures misclassified

**No genuine novel class detections confirmed across many inspected samples, i have to check for more samples to confirm this.**

### Root Cause

The backbone's CLIP alignment is too weak for fine-grained novel-class discrimination:

1. **Distillation degradation**: CLIP → OpenSeg → 3D projection → DisNet distillation. Each step loses resolution.
2. **Tiny cosine margins**: "car" vs "police car" text proto cosine similarity ≈ 0.85. The backbone cannot resolve this 0.15 margin after 4 degradation steps.
3. **No negative training**: Backbone was never trained to push features AWAY from novel classes. Without explicit negatives, there are no decision boundaries.

### Conclusion

**Inference-only open-vocabulary detection produces only false positives with the current backbone.** The geometric ensemble framework is architecturally sound, but the 3D features lack the discriminative power needed for novel classes that are semantically similar to known classes.

## 12. Code Changes Log (2026-05-27)

| Change | File | Impact |
|--------|------|--------|
| Removed modes C, D, E | evaluate_comprehensive.py | Cleaner eval, faster runtime |
| Removed `panoptic_inference_ensemble`, `panoptic_inference_backbone_pool` (legacy) | evaluate_comprehensive.py | Dead code removed |
| Added `generate_multi_prompt_prototypes()` | evaluate_comprehensive.py | 5-6 diverse prompts per novel class |
| Added `compute_prototype_bias()` | evaluate_comprehensive.py | Automatic per-prototype background affinity |
| Added `panoptic_inference_geometric_ensemble_v2()` | evaluate_comprehensive.py | Bias-corrected logits |
| Moved CLIP loading before backbone (OOM fix) | evaluate_comprehensive.py | Sequential GPU: CLIP first → unload → backbone |
| Fixed novel voxel summary display | evaluate_comprehensive.py | Was always showing N/A |
| Forwarded `temperature` to Mode H | evaluate_comprehensive.py | Bug fix |
| Updated visualize to use geo_v2 only | visualize_novel_detections.py | Only meaningful mode |
| Reverted coords "fix" | evaluate_comprehensive.py | Model was trained with vox_feats as coords |

## 13. Next Steps

### Inference-only (K/L/M TESTED — FAILED, 2026-05-28):

| Strategy | Result | PQ | Novel Voxels |
|----------|--------|-----|-------------|
| K (voxel vote) | Over-detection, PQ degraded | 0.4490 (-0.027) | 1,815,332 |
| L (logit aggregation) | Identical to geo_v2 | 0.4702 (-0.006) | 305,525 |
| M (confidence gated) | Reduces volume, same FPs | 0.4633 (-0.013) | 274,038 |

**Conclusion**: Per-voxel classification produces the same false positives as feature pooling. Strategy L (the theoretically "correct" approach) gives identical results to Strategy E (feature pooling), proving that **pooling degradation is NOT the bottleneck — the backbone features themselves lack discriminative margin between similar classes.**

The inference-only chapter is **CLOSED** (no-bias ablation completed 2026-05-29).

**No-bias ablation results**: Without bias, pole catastrophe returns — voxel_vote explodes to 18.6M novel voxels (PQ=0.3935), logit_agg to 830K (PQ=0.4642), confidence_gated to 364K (PQ=0.4634). Bias prevents pole catastrophe but doesn't fix the fundamental margin issue. L=E confirmed in both ablations.

### Training required (for genuine open-vocab):
1. **Strategy I**: Voxel decomposition loss with 200+ diverse text negatives — teaches explicit inter-class boundaries
2. **Strategy C1**: 200 negatives for token head — prevents collapse
3. **Strategy H + I**: Unfreeze backbone + voxel decomp — strongest combined approach
4. **Strategy D**: Camera CLIP features at inference — bypasses degraded 3D backbone entirely
