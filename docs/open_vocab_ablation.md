# Open-Vocabulary Panoptic Segmentation via Backbone Feature Pooling

## Document Version
- **v1.0** — 2026-04-26
- **Status**: Design complete, pending implementation
- **Parent**: `docs/master_plan.md` Phase 5 extension

---

## 1. Motivation

Our panoptic decoder predicts **class-agnostic binary masks** via a Hungarian-matched transformer decoder. Classification is done by two heads:
- **Linear head**: `nn.Linear(256, 17)` — closed-vocabulary, 16 classes + background
- **Token head** (Phase 5): MLP(256→768) projecting decoder queries into CLIP space — trained on 16 classes but operates in a shared language-vision space

Neither head is truly open-vocabulary because both are **supervised only on nuScenes 16 classes**. The token head projects *decoder query embeddings* (256-dim, learned) into CLIP space, which means it can only surface information the decoder queries have learned to capture.

However, our frozen backbone (OpenScene DisNet, distilled from CLIP ViT-L/14@336px) produces **768-dim CLIP-aligned features at every voxel** that encode the full visual-semantic richness of CLIP — not just 16 classes. These features are currently used only as key/value inputs to the decoder (after projection to 256-dim), discarding most of their open-vocabulary information.

**Option A** recovers this information: instead of classifying masks via the linear or token head, we **pool the raw 768-dim backbone features within each predicted mask** and classify via cosine similarity to arbitrary text prompts.

---

## 2. Core Idea

```
                     ┌──────────────────────────────────────────────┐
                     │           Existing Pipeline                  │
                     │                                              │
  LiDAR voxels ──→ DisNet ──→ feats (V, 768) ──→ decoder ──→ masks │
       backbone        │           ↓ proj(256)     queries    (V,Q) │
                       │                                            │
                       │  ┌─── linear head: query → 17 logits       │
                       │  └─── token head:  query → 768 CLIP embed  │
                       │                                            │
                       └──────────────────────────────────────────────┘

                     ┌──────────────────────────────────────────────┐
                     │         Option A: Open-Vocab Path            │
                     │                                              │
  feats (V, 768)  ───┤  For each predicted mask m_q (Q masks):     │
    raw backbone      │    1. Select voxels: v_idx = (mask_q > 0)   │
                      │    2. Pool: f_q = mean(feats[v_idx])        │
                      │    3. Normalize: f_q = f_q / ||f_q||        │
                      │    4. Score: sim = f_q · text_prototypes.T  │
                      │    5. Class = argmax(sim)                   │
                      └──────────────────────────────────────────────┘
```

**Key insight**: The masks come from the decoder (which is good at segmenting objects). The classification comes from the backbone features (which are good at open-vocabulary recognition). We decouple segmentation from classification.

---

## 3. Why This Works

1. **The backbone features are already in CLIP space.** OpenScene distills ViT-L/14@336px features into 3D voxels. These features encode semantic meaning for *any* visual concept CLIP knows about — far beyond 16 nuScenes classes.

2. **Mean pooling per mask aggregates evidence.** A single voxel's feature may be noisy, but averaging across an entire mask region (50-5000 voxels) gives a robust representation of what that region contains.

3. **The decoder is class-agnostic in practice.** While trained with class supervision, the decoder's primary learned skill is spatial segmentation (grouping voxels into coherent instances). The mask predictions generalize to novel objects that have coherent spatial structure.

4. **Text prompts are free.** CLIP text encoder can encode any phrase. Adding "fire hydrant" costs nothing and requires zero retraining.

---

## 4. Implementation Plan

### 4.1 Files to Create

| File | Purpose |
|---|---|
| `panoptic/scripts/evaluate_open_vocab.py` | Main evaluation script for Option A |
| `panoptic/scripts/generate_open_vocab_prototypes.py` | Generate text prototypes for 16 known + N novel classes |
| `panoptic/configs/open_vocab_eval.yaml` | Eval config with novel class list and thresholds |

### 4.2 Files to Modify

None. Option A is inference-only — no changes to the model, decoder, or training code.

---

## 5. Detailed Implementation

### 5.1 Text Prototype Generation (`generate_open_vocab_prototypes.py`)

```python
# Known nuScenes 16 classes (same as training)
KNOWN_CLASSES = [
    "barrier", "bicycle", "bus", "car", "construction_vehicle",
    "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck",
    "driveable_surface", "other_flat", "sidewalk", "terrain",
    "manmade", "vegetation"
]

# Novel classes — driving-relevant objects NOT in nuScenes 16
NOVEL_CLASSES = [
    "pole", "bench", "trash can", "fire hydrant", "tree trunk", "curb"
]

# Combined vocabulary
ALL_CLASSES = KNOWN_CLASSES + NOVEL_CLASSES  # 22 total
```

- Use `clip.load("ViT-L/14@336px")` (same as backbone distillation)
- Prompt template: `"a {label} in a scene"` (same as `util/util.py:extract_text_feature()`)
- Output: `panoptic/clip_prototypes/open_vocab_prototypes.pt`
  - `embeddings`: (22, 768) L2-normalized
  - `class_names`: list of 22 strings
  - `num_known`: 16
  - `num_novel`: 6

### 5.2 Open-Vocab Inference Function

```python
def panoptic_inference_open_vocab(
    pred_masks,          # (V, Q) — raw mask logits from decoder
    backbone_feats,      # (V, 768) — raw DisNet features (NOT projected)
    text_prototypes,     # (C, 768) — L2-normalized text embeddings
    num_thing_classes=10,
    num_stuff_classes=6,
    num_known=16,
    overlap_thr=0.8,
    score_thr=0.1,       # minimum cosine similarity to accept
    min_points=50,
):
    """
    1. Sigmoid on pred_masks → (V, Q) soft masks
    2. For each query q:
       a. binary_mask = (soft_masks[:, q] > 0.5)
       b. if sum(binary_mask) < min_points: skip
       c. masked_feats = backbone_feats[binary_mask]  # (N_q, 768)
       d. pooled = masked_feats.mean(dim=0)           # (768,)
       e. pooled = F.normalize(pooled, dim=0)         # L2-norm
       f. sim = pooled @ text_prototypes.T             # (C,)
       g. class_id = sim.argmax()
       h. score = sim[class_id]
       i. if score < score_thr: skip
    3. Sort queries by score (descending)
    4. Greedy assignment with overlap_thr (same as existing panoptic_inference)
    5. Return sem_pred (V,), inst_pred (V,)
    """
```

**Important**: The first 10 classes (indices 0-9) are thing classes, 10-15 are stuff classes. Novel classes (indices 16-21) are treated as **thing classes** (they could be individual objects like poles, benches).

### 5.3 Evaluation Script (`evaluate_open_vocab.py`)

**Flow:**

```
Load config, backbone, decoder, text_prototypes (22 classes)
For each val sample:
    feats, coords = extract_backbone_features(backbone, batch)  # (1, V, 768)
    outputs = decoder(feats, coords)
    pred_masks = outputs["pred_masks"][0]  # (V, 106)
    backbone_feats = feats[0]              # (V, 768) ← KEY: raw features

    # Mode 1: Closed-vocab linear (baseline)
    sem_lin, inst_lin = panoptic_inference(outputs["pred_logits"][0], pred_masks, ...)

    # Mode 2: Open-vocab backbone pooling (Option A)
    sem_ov, inst_ov = panoptic_inference_open_vocab(pred_masks, backbone_feats, text_protos, ...)

    # Evaluate both (PanopticEval on known 16 classes)
    eval_linear.addBatch(sem_lin, inst_lin, gt_sem, gt_inst)
    eval_open_vocab.addBatch(
        remap_to_known(sem_ov),  # map novel predictions to ignore for fair comparison
        inst_ov, gt_sem, gt_inst
    )

    # Also log: for each query, what novel classes were predicted (qualitative)
    log_novel_detections(sem_ov, inst_ov, novel_class_indices)
```

**Three output reports:**

1. **Quantitative (known classes):** PQ/SQ/RQ/mIoU comparing linear-head vs open-vocab on the standard 16 classes
2. **Novel class detections:** Per-sample count of queries classified as novel classes, with top-k cosine similarity scores
3. **Qualitative:** Save PLY files with novel class colors for 5-10 random samples

### 5.4 Backbone Feature Access

The raw 768-dim features are already available in the eval loop. In all training/eval scripts, the flow is:

```python
feats, coords = extract_backbone_features(backbone, batch, device)
# feats = (1, V, 768) — this is what we need
outputs = decoder(feats, coords)
# decoder internally projects feats to 256-dim, but we keep the original
```

**No model changes needed.** We just keep the `feats` tensor alongside the decoder outputs.

---

## 6. Evaluation Design

### 6.1 Ablation Table (thesis artifact)

| Method | Classification Source | Vocabulary | PQ (16 cls) | mIoU (16 cls) | Novel Detections |
|---|---|---|---|---|---|
| Phase 4: Linear head | `class_embed(query)` softmax | Closed (16) | **X.XX** | **X.XX** | N/A |
| Phase 5: Token head | `token_head(query)` · text_protos | Semi-open (16 trained) | **X.XX** | **X.XX** | Possible but weak |
| Option A: Backbone pooling | `mean(backbone_feats[mask])` · text_protos | **Open (any text)** | **X.XX** | **X.XX** | **Yes** |

### 6.2 Known-Class Evaluation (Quantitative)

Run on nuScenes v1.0-trainval val split (6019 samples) or v1.0-mini val (81 samples):
- Standard PQ/SQ/RQ/mIoU on 16 known classes
- Per-class breakdown to see which classes benefit or hurt

**Expected outcome:** Option A may score slightly lower on known-class PQ than the linear head (the linear head was optimized for these 16 classes). The gap size is the "cost" of going open-vocabulary.

### 6.3 Novel-Class Evaluation (Qualitative)

Since there are no GT annotations for novel classes, this is qualitative:

1. **Detection frequency**: How often does the model predict novel classes? (Should be non-zero for driving scenes)
2. **Confidence scores**: What's the average cosine similarity for novel class predictions? (Should be >0.15 to be meaningful)
3. **Visual inspection**: Save 10 samples with PLY files colored by open-vocab predictions. Manually inspect whether poles, benches, etc. are correctly identified.
4. **Confusion analysis**: When a novel class is predicted, what was the GT label? (Helps identify if "pole" was GT "manmade" or "barrier")

### 6.4 Score Threshold Sensitivity

Run with `score_thr` ∈ {0.05, 0.10, 0.15, 0.20} and report:
- Known-class PQ vs threshold
- Novel detection count vs threshold
- False positive rate vs threshold

---

## 7. Novel Class Selection Rationale

| Novel Class | Why Selected | Expected Source in nuScenes |
|---|---|---|
| pole | Ubiquitous in driving, not a nuScenes class. Often part of "manmade". | Likely labeled as background or manmade |
| bench | Common urban object, distinct shape | Likely labeled as manmade or unlabeled |
| trash can | Common urban object | Likely manmade or unlabeled |
| fire hydrant | Small but distinctive, good CLIP recognition | Likely manmade or unlabeled |
| tree trunk | Distinct from "vegetation" (canopy) | Likely labeled as vegetation |
| curb | Edge between road and sidewalk | Likely labeled as sidewalk or other_flat |

These classes test different aspects:
- **pole, fire hydrant**: Small objects with strong visual features — tests CLIP feature quality
- **bench, trash can**: Medium objects — tests mask quality for unseen categories
- **tree trunk, curb**: Tests fine-grained distinction within known super-categories

---

## 8. Config (`open_vocab_eval.yaml`)

```yaml
# ── Open-vocab evaluation config ──
data:
  dataroot: /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes
  version: v1.0-mini  # switch to v1.0-trainval for full eval
  voxel_size: 0.05
  point_range: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

model:
  backbone_ckpt: models/nuscenes_openseg.pth.tar
  backbone_dim: 768
  hidden_dim: 256
  num_classes: 16
  num_thing_queries: 100
  num_stuff_classes: 6
  nheads: 8
  dim_feedforward: 1024
  num_blocks: 3
  num_levels: 3
  dropout: 0.0
  use_token_head: true
  token_dim: 768

eval:
  checkpoint: panoptic/checkpoints_token_full/best.pth
  prototype_path: panoptic/clip_prototypes/open_vocab_prototypes.pt
  score_thr: 0.10
  overlap_thr: 0.8
  min_points: 50
  num_workers: 0
  save_qualitative: true
  num_qualitative: 10
  output_dir: panoptic/eval_results/open_vocab/

novel_classes:
  - pole
  - bench
  - trash can
  - fire hydrant
  - tree trunk
  - curb
```

---

## 9. Implementation Order

| Step | Task | Depends On | Time |
|---|---|---|---|
| 1 | Write `generate_open_vocab_prototypes.py` | CLIP model available | 10 min |
| 2 | Run prototype generation | Step 1 | 1 min (GPU) |
| 3 | Write `panoptic_inference_open_vocab()` function | None | 30 min |
| 4 | Write `evaluate_open_vocab.py` | Steps 1-3 | 30 min |
| 5 | Write `open_vocab_eval.yaml` | None | 5 min |
| 6 | Run on mini val (81 samples) | Trained checkpoint + steps 1-4 | 5 min (GPU) |
| 7 | Analyze results, tune score_thr | Step 6 | 15 min |
| 8 | Run on full val (6019 samples) | Step 7 + full checkpoint | 30 min (GPU) |
| 9 | Generate qualitative PLY + camera overlay visualizations | Step 8 | 15 min (GPU) |

**Total estimated work**: ~2 hours coding + ~1 hour GPU eval

---

## 10. Acceptance Criteria

- [ ] Open-vocab evaluation runs end-to-end on mini val without errors
- [ ] Known-class PQ is within 0.05 of linear head PQ (backbone features should be competitive)
- [ ] At least some novel class detections appear in driving scenes (count > 0)
- [ ] Cosine similarity scores for novel predictions are > 0.10 (not random noise)
- [ ] Qualitative PLY files show visually plausible novel class predictions
- [ ] Ablation table populated with all three methods (linear, token, open-vocab)

---

## 11. Potential Issues and Mitigations

| Issue | Cause | Mitigation |
|---|---|---|
| Open-vocab PQ much lower than linear | Backbone features are noisy per-voxel; mean pooling may not be discriminative enough | Try weighted pooling (weight by mask confidence), or increase `min_points` threshold |
| No novel detections | Novel objects too rare, or features collapse to known classes | Lower `score_thr`, add more diverse novel prompts, inspect raw cosine similarities |
| Novel classes steal known-class predictions | "pole" may outscore "manmade" for building features | Use a known-class priority: if a known class scores within 0.05 of a novel class, prefer known |
| Memory issues on full val | Keeping (V, 768) backbone features in memory | Features are already loaded (used by decoder), no extra memory needed |

---

## 12. Future Work: Option B — Class-Agnostic Training (SAL-like)

### Concept

Option A uses masks from a decoder trained with class supervision. This means the decoder queries may have class-specific biases (e.g., query #53 always predicts "car"). This limits generalization to novel objects whose spatial structure differs from known classes.

**Option B** trains the decoder **without class supervision**, making it truly class-agnostic:

1. **Remove CE classification loss** from `PanopticLoss`
2. **Keep only mask + dice loss** — queries learn to segment "any coherent region" without class identity
3. **At inference**: use backbone feature pooling (Option A) for all classification
4. **Benefit**: Queries are not biased toward 16 known classes, so they can segment any object

### Design Sketch

```
Modified loss:
  weight_ce: 0.0    # disable class supervision
  weight_mask: 5.0   # keep mask BCE
  weight_dice: 5.0   # keep dice loss

Modified matcher:
  cost_class: 0.0   # no class cost in Hungarian matching
  cost_mask: 5.0     # match purely on mask overlap
  cost_dice: 5.0

Modified decoder:
  Remove class_embed head (or keep but ignore during training)
  Remove token_head (not needed — classification via backbone pooling)
```

### Training Strategy (SAL-inspired)

1. **Warm-start** from existing Phase 4/5 checkpoint (decoder already knows spatial segmentation)
2. **Fine-tune 5-10 epochs** with class-agnostic losses only
3. **Evaluate** with backbone feature pooling on both known and novel classes

### Expected Impact

- **Known-class PQ**: May decrease slightly (no class supervision to guide queries)
- **Novel-class detection**: Should improve significantly (queries not biased toward 16 classes)
- **Mask quality**: Should remain similar (mask loss unchanged)

### Why This Is Future Work

- Requires retraining (another 5-25 hours of GPU time)
- Need to validate Option A first to prove backbone feature pooling works at all
- If Option A shows competitive known-class PQ, the marginal benefit of Option B is smaller
- Option B is the natural next step if the thesis timeline permits

### Connection to SAL (Segment Anything LiDAR)

SAL uses a similar two-stage approach:
1. Class-agnostic mask proposals from a trained segmenter
2. CLIP-based classification via feature aggregation per mask

Our approach differs in that we distill CLIP features **into the 3D backbone** (via OpenScene), so we don't need 2D camera images at inference time. SAL requires projecting 3D masks back to 2D for CLIP classification. Our approach is simpler and fully 3D.

---

## 13. Summary

Option A is a **zero-cost evaluation-time experiment** that leverages the existing CLIP-distilled backbone features for open-vocabulary classification. It requires:
- No model changes
- No retraining
- Only a new evaluation script + text prototypes

It provides the thesis with:
1. A quantitative ablation comparing closed-vocab vs open-vocab on known classes
2. A qualitative demonstration of novel class detection
3. A proof-of-concept for the full vision: supervised panoptic segmentation with open-vocabulary capability
4. A clear path to future Option B (class-agnostic training) for even stronger generalization
