# Current Progress: From Closed-Set v2 to Open-Vocabulary Pipeline

## 1. The v2 Model Architecture (Closed-Set Baseline)

The `checkpoints_token_full_v2/best.pth` model achieves **PQ=0.476** on nuScenes validation with the following architecture:

```
Frozen DisNet Backbone (768-d, CLIP-distilled via OpenSeg)
         │
    backbone_proj (768→256)
         │
    Transformer Decoder (9 layers, 256-d queries, Q=106)
         │
    ┌────┼────────────┐
    │    │            │
    ▼    ▼            ▼
class_embed  mask_embed    token_head
(256→17)     (256→256)     (256→512→512→768)
    │            │              │
pred_logits  pred_masks     pred_embed
(Q, 17)      (V, Q)         (Q, 768)
```

### Training Losses

| Loss | Formula | Purpose |
|------|---------|---------|
| L_CE | CrossEntropy(pred_logits, GT_class) | 17-way classification (16 classes + no-object) |
| L_mask | BinaryCE(pred_masks, GT_masks) | Per-voxel mask prediction |
| L_dice | DiceLoss(pred_masks, GT_masks) | Mask quality / boundary refinement |
| L_token (C1) | InfoNCE(pred_embed, text_protos) | Auxiliary CLIP alignment (16 prototypes) |

Hungarian matching assigns predicted queries to GT instances before loss computation.

### Classification at Inference

Standard closed-set: `scores, labels = pred_logits.softmax(-1).max(-1)` → greedy panoptic assignment with overlap/size thresholds.

The token head was trained simultaneously but only as an auxiliary — the CE classifier (`class_embed`) performs primary classification.

---

## 2. Inference Strategies for Open-Vocabulary (No Retraining)

### Strategy E: Geometric Ensemble

- Pool backbone's raw 768-d features through predicted soft masks → one 768-d vector per mask
- Cosine similarity to text prototypes → CLIP classification scores
- Blend with CE classifier: `linear_probs^α × clip_probs^(1-α)` for known classes
- CLIP scores only for novel classes (CE head has no novel class logits)
- **Result:** PQ_known ~0.39, PQ_novel ~0.05–0.08

### Strategy K: Per-Voxel Majority Vote

- Classify each voxel independently (768-d feature vs text prototypes)
- Aggregate classification decisions within each predicted mask (logit averaging)
- Novel class override only if CLIP confidence exceeds known class by 20% margin
- **Key insight:** `E[f(x)] ≠ f(E[x])` — averaging features before softmax destroys discriminability

### Strategy L: Soft Logit Aggregation

- Same as K but uses geometric ensemble formula on aggregated per-voxel logits
- Preserves signal better than feature pooling

### Strategy M: Hybrid

- Combines per-voxel voting with token head output for known/novel decision

**All strategies achieved limited novel class recognition (PQ_novel < 0.08).**

---

## 3. Voxel Margin Diagnostic: Why Inference Strategies Failed

### Three-Stage Measurement

We developed a diagnostic (`eval_voxel_margin_v1.py`) measuring cosine margin at three points in the processing chain:

| Stage | What It Measures | Margin | Correct |
|-------|-----------------|--------|---------|
| 1. Raw backbone (per-voxel) | DisNet's 768-d output vs text prototypes | **-0.021** | 27.9% |
| 2. Mask-pooled backbone | Predicted mask × backbone feats, then vs prototypes | **-0.019** | 35.4% |
| 3. Token head output | 256→768 learned projection vs prototypes | **+0.225** | 82.3% |

### Per-Class Stage 1 Breakdown

| Class | Margin | Correct | Issue |
|-------|--------|---------|-------|
| car | +0.017 | 64.0% | Only reliably separated thing class |
| pedestrian | +0.032 | 65.0% | Distinctive in CLIP (human shape) |
| vegetation | +0.002 | 58.6% | Barely positive (green texture) |
| driveable surface | -0.015 | 16.2% | Confused with sidewalk, terrain |
| terrain | -0.059 | 0.0% | Completely confused with vegetation/sidewalk |
| other flat | -0.110 | 0.1% | Too generic a concept |
| bus | -0.068 | 4.3% | Confused with truck/car |

### Root Cause Analysis

**The OpenSeg → DisNet distillation lost CLIP geometry.**

DisNet was trained to minimise reconstruction loss (MSE/cosine) against projected 2D CLIP features. This preserves the average direction well but compresses fine-grained angular differences:

- The distillation treats all dimensions equally — it doesn't know that the angular gap between "car" and "truck" matters more than the gap between "car" and "banana"
- Multi-view averaging into 3D blurs view-specific discriminative features
- Boundary voxels receive contributions from multiple classes, drifting toward a generic "outdoor scene" embedding

**Why inference strategies cannot fix this:**

All strategies (E, K, L, M) rely on backbone features being CLIP-aligned at the voxel level. With margin = -0.021, the features simply don't separate classes in CLIP space. No aggregation method (pooling, voting, ensembling, temperature scaling) can create separation that doesn't exist in the underlying features.

### The Token Head Proof

Stage 3 margin = +0.225 (82.3% correct) proves:

1. **The decoder queries carry sufficient semantic information** — the 256-d representations encode class-discriminative features despite being trained for CE
2. **A learned projection can bridge the gap** — a 3-layer MLP successfully maps from decoder space to CLIP space with strong separation
3. **The problem is at the backbone level, not the architecture** — the capacity exists; it needs training-time intervention

---

## 4. The Pivot: Staged Fine-Tuning Pipeline (Job 1a → 1b → 1c)

### Design Principle

Open-vocabulary capability cannot be achieved at inference time alone. Training-time alignment is required. But full retraining is impractical. Solution: staged fine-tuning targeting each stage of the measurement chain.

### Job 1a: Fix Classification Head (Targets Stage 3)

**Goal:** Train a token head to classify in CLIP space using B12/N4 split.

| Aspect | Detail |
|--------|--------|
| Architecture | SingleCLIPEmbeddingHead replaces CE classifier |
| Loss | InfoNCE with 216-class denominator (12 base targets + 4 novel soft negatives + 200 hard negatives) |
| Temperature | Learnable, init at 3.0, clamped [1.0, 5.0], applied as division (sim/τ) |
| Margin penalty | 0.1 subtracted from GT class similarity before temperature scaling |
| Novel handling | {bus, pedestrian, manmade, vegetation} excluded from ALL losses |
| Backbone adaptation | LoRA rank=16 (~41K params) for gradient flow through frozen backbone |
| Warm-start | From closed-set best (PQ=0.476) |

**Expected outcome:** Stage 3 margin > +0.10 for base classes. Novel class generalization through broad negative bank forcing full CLIP geometry learning.

**Key fixes from v1 → v2:**
- Temperature bug: was `sim * τ` (temperature escapes to ceiling) → now `sim / τ` (temperature descends, sharpens)
- Novel class leakage: novel classes had full mask/dice supervision in v1
- AMP gradient norm: read before unscale, giving inflated values

### Job 1b: Fix Backbone Features (Targets Stage 1)

**Goal:** Push backbone voxel features toward CLIP space via dense supervision.

| Aspect | Detail |
|--------|--------|
| Data | Precomputed camera CLIP features (ViT-L/14, stride=112, fp16, 186GB for trainval) |
| L_V (voxel distillation) | 1 - cos_sim(reconstructed_voxel_feat, camera_CLIP_feat) per voxel |
| L_O (object distillation) | Cosine alignment of mask-pooled backbone features to camera CLIP |
| Key property | Class-agnostic — supervises ALL voxels including novel classes |
| Warm-start | From Job 1a best checkpoint |

**Expected outcome:** Stage 1 margin from -0.02 → +0.05–0.10. Since CLIP naturally separates classes, aligning backbone to CLIP should produce class separation as a byproduct.

### Job 1c: Fix Novel Class Masks (Planned, Not Implemented)

**Goal:** Bootstrap novel class mask quality via self-distillation.

| Aspect | Detail |
|--------|--------|
| Method | Run inference on unlabeled data, keep high-confidence novel predictions as pseudo-GT |
| Why needed | Job 1a excludes novel from mask loss → model never learns novel mask shapes |
| Condition | Only needed if after Job 1b, novel classes have good margins but poor SQ (mask quality) |

---

## 5. Current Status

| Component | Status | Result |
|-----------|--------|--------|
| Closed-set baseline | Complete | PQ=0.476 |
| Inference strategies (E, K, L, M) | Complete | PQ_novel < 0.08 (insufficient) |
| Voxel margin diagnostic | Complete | Backbone margin = -0.021, token head = +0.225 |
| Prototype generation (224px) | Complete | 16-class + multiprompt + 200 negative bank |
| Precompute CLIP features | Complete | 186GB, stride=112, fp16, full trainval |
| Job 1a (v3, /τ fix) | Running | Temperature 3.0→2.91 (descending correctly) |
| Job 1b code | Ready | Passed mini smoke test |
| Job 1c | Not implemented | Deferred until after 1b evaluation |
| eval_openvocab_v2.py | Ready | Evaluates token head PQ directly |

### Key Open Questions

1. **After Job 1a completes:** Does Stage 3 margin for base classes reach +0.10? Does novel margin generalize above 0?
2. **After Job 1b completes:** Does Stage 1 margin improve? Does improved backbone help token head generalize to novel classes?
3. **LoRA capacity:** Is rank=16 sufficient for Job 1b's dense alignment, or is rank=32/64 needed?

---

## 6. Chain of Evidence (Summary)

```
Inference strategies failed (PQ_novel < 0.08)
    ↓ Why?
Voxel margin diagnostic: backbone margin = -0.02 (no class separation at source)
    ↓ But...
Token head proved: +0.22 margin achievable with learned projection (architecture works)
    ↓ Therefore...
Need TRAINING-TIME intervention at multiple stages:
    Job 1a → Fix Stage 3 (classification head, B12/N4 split)
    Job 1b → Fix Stage 1 (backbone features, dense CLIP distillation)
    Job 1c → Fix novel masks (self-distillation if needed)
```

The fundamental insight: **the distillation chain (CLIP → OpenSeg → DisNet) degrades cosine geometry below usable margins.** No inference-time trick can recover what was lost during distillation. Only training-time realignment — either through the token head (Job 1a) or directly at the backbone level (Job 1b) — can restore the angular separation needed for open-vocabulary classification.
