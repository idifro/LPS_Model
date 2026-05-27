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

## 7. Evaluation Modes (Updated)

| Mode | Method | Classes | Purpose |
|------|--------|---------|---------|
| A. Linear | pred_logits softmax | 16 known | Baseline |
| B. Token-16 | pred_tokens cos sim | 16 known | Token head quality |
| C. Ensemble | α*linear + (1-α)*token | 16 known | Linear+token blend |
| D. Token-OpenVocab | pred_tokens cos sim | 16+N | Token head open-vocab (failed: collapsed) |
| E. Backbone-Pool | raw mean-pool | 16+N | Legacy (buggy, PQ=0.209) |
| **F. Backbone-Pool-v2** | normalized soft-pool | 16+N | **Bug fix** |
| **G. Geometric-Ensemble** | linear^α × clip^(1-α) | 16+N | **Strategy E proper** |

## 8. Usage

```bash
# Full evaluation (all 7 modes)
python -u panoptic/scripts/evaluate_comprehensive.py \
    --config panoptic/configs/full_nuscenes_token_v2.yaml \
    --checkpoint panoptic/checkpoints_token_full_v2/best.pth \
    --novel-classes "police car,ambulance,trash cans,pole,bench"

# Only new modes
python -u panoptic/scripts/evaluate_comprehensive.py \
    --config panoptic/configs/full_nuscenes_token_v2.yaml \
    --checkpoint panoptic/checkpoints_token_full_v2/best.pth \
    --modes backbone_pool_v2,geometric_ensemble \
    --geo_alphas 0.3,0.5,0.7 \
    --geo_temperature 0.1

# Custom novel classes
python -u panoptic/scripts/evaluate_comprehensive.py \
    --config panoptic/configs/full_nuscenes_token_v2.yaml \
    --checkpoint panoptic/checkpoints_token_full_v2/best.pth \
    --modes geometric_ensemble \
    --novel-classes "motorcycle rider,scooter,stroller,dog,shopping cart"
```

## 9. Actual Results (v1.0-trainval, 6019 samples, 2026-05-26)

### 9.1 Full Summary Table

| Mode | PQ | SQ | RQ | mIoU |
|------|-----|-----|-----|------|
| A. Linear | **0.4772** | 0.7926 | **0.5844** | **0.4324** |
| B. Token | 0.4711 | 0.7963 | 0.5771 | 0.4208 |
| C. Best Ensemble (α=0.0) | 0.4711 | 0.7963 | 0.5771 | 0.4208 |
| D. Token-OpenVocab | 0.4711 | 0.7963 | 0.5771 | 0.4208 |
| E. Backbone-Pool (legacy) | 0.2085 | 0.6672 | 0.2470 | 0.1677 |
| **F. Backbone-Pool-v2** | **0.2208** | 0.6698 | 0.2600 | 0.1722 |
| **G. Geo-Ensemble α=0.3** | 0.4595 | 0.7979 | 0.5598 | 0.4241 |
| **G. Geo-Ensemble α=0.5** | **0.4632** | 0.7943 | 0.5657 | 0.4224 |
| **G. Geo-Ensemble α=0.7** | 0.4628 | 0.7946 | 0.5655 | 0.4215 |

### 9.2 Novel Class Detections

| Novel Class | Token-OV | Backbone-Pool (legacy) | Backbone-Pool-v2 | Geometric Ensemble (α=0.3) |
|-------------|----------|------------------------|-------------------|---------------------------|
| police car | 0 | 0 | 0 | **748** |
| ambulance | 0 | 0 | 0 | 0 |
| trash cans | 0 | 45,202 | 42,018 | **7,497** |
| pole | 0 | **25,360,750** | **23,878,854** | **162,804** |
| bench | 0 | 0 | 0 | 0 |

### 9.3 Analysis

**What worked:**
1. Geometric ensemble preserved 97% of known-class PQ (0.463/0.477) — this validates that the linear head dominates known-class decisions
2. Objectness filtering reduced "pole" catastrophe by 99.4% (25M → 163K)
3. Only geometric ensemble detected "police car" (748 voxels) — genuine CLIP signal
4. Best alpha was 0.5 (balanced) for PQ, but α=0.3 (more CLIP weight) detected more novel voxels

**What failed:**
1. Backbone-Pool-v2 normalization fix was insufficient (+0.012 PQ vs legacy) — the core issue is prototype quality, not pooling math
2. "Pole" prototype remains dominant (163K voxels even with objectness) — text embedding for "pole" has high average similarity to most backbone feature clusters
3. Token-OpenVocab detected zero novel voxels — token head is completely collapsed to 16-class manifold
4. "ambulance" and "bench" = 0 across all modes — either absent from val set or prototypes too generic
5. 748 "police car" voxels across 6019 samples is statistically insignificant — needs visual verification

**Verdict:** Strategy E provides a working framework but novel detection signal is too weak for production use. The geometric ensemble is the right *architecture* but needs:
- Better text prototypes (multi-template, more specific prompts)
- Training-time changes to achieve robust open-vocab

### 9.4 Per-Class Highlights (Geometric Ensemble α=0.5 vs Linear)

| Class | PQ (Linear) | PQ (Geo α=0.5) | Δ | Note |
|-------|-------------|-----------------|---|------|
| motorcycle | 0.654 | **0.672** | +0.018 | CLIP helps rare classes |
| sidewalk | 0.209 | **0.218** | +0.009 | Slight improvement |
| traffic_cone | 0.568 | 0.505 | -0.063 | CLIP hurts (confused with pole?) |
| other_flat | 0.069 | 0.053 | -0.016 | CLIP noise degrades rare stuff |
| car | 0.848 | 0.839 | -0.009 | Minor known-class degradation |

Observation: CLIP helps rare thing classes (motorcycle, construction_vehicle) where the linear head is uncertain, but hurts stuff classes (other_flat, terrain) where CLIP features lack spatial coherence.

## 10. Backward Compatibility

- All existing modes (A-E) produce identical results
- New modes (F, G) are additive — only run if explicitly included in `--modes`
- Default `--modes` includes all 7 modes
- Novel class generation is lazy (only when open-vocab modes are requested)
- Config YAML change is optional (commented out `novel_classes` field)

## 11. File Map

| File | Change |
|------|--------|
| `panoptic/scripts/evaluate_comprehensive.py` | Added `panoptic_inference_backbone_pool_v2`, `panoptic_inference_geometric_ensemble`, Mode F+G in eval loop, configurable novel classes |
| `panoptic/scripts/visualize_novel_detections.py` | Visual verification: projects novel-class voxels onto camera images |
| `panoptic/configs/full_nuscenes_token_v2.yaml` | Added optional `novel_classes` field (commented) |
| `docs/model_design/model_openvocab_strategyE.md` | This document |

## 12. Visual Verification

The geometric ensemble's novel detections (especially 748 "police car" voxels) need visual confirmation. Use:

```bash
python -u panoptic/scripts/visualize_novel_detections.py \
    --config panoptic/configs/full_nuscenes_token_v2.yaml \
    --checkpoint panoptic/checkpoints_token_full_v2/best.pth \
    --version v1.0-trainval \
    --only_with_detections \
    --max_samples 200 \
    --num_samples 30 \
    --alpha 0.3
```

This projects novel-class predictions onto all 6 camera views, enabling visual assessment of:
- Are "police car" detections actually on police-like vehicles?
- Are "pole" detections on actual poles or false positives on buildings/walls?
- Are "trash cans" detections genuine?

## 13. Future Work

1. **Multi-prompt prototypes**: Average 5-10 templates per class ("a police car on the road", "a parked police car", etc.) for more robust embeddings
2. **Temperature tuning**: Per-class temperature; "pole" needs higher τ to reduce dominance
3. **Class-specific alpha**: Known stuff classes (driveable_surface) use α=1.0 (pure linear), rare things use lower α
4. **Class-agnostic mask training** (Strategy A): Remove CE loss, use mask+dice only → unbiased proposals
5. **CLIP-initialized queries**: Initialize decoder queries from CLIP text embeddings instead of random → queries are pre-aligned to semantic concepts
6. **Unfreeze encoder end-to-end**: Joint fine-tuning of DisNet + decoder → features become panoptic-aware while maintaining CLIP alignment via distillation regularization
7. **Voxel decomposition loss**: Per-voxel contrastive loss between backbone features and GT text embeddings → forces feature alignment at point level
8. **Large negative bank (Strategy C1)**: Add 200+ text negatives during token head training → prevents closed-set collapse
