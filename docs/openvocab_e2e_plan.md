# Open-Vocabulary E2E Encoder Training Plan

## Document Purpose

Single source of truth for the **open-vocabulary + end-to-end backbone training** phase of this project.  
Builds directly on the closed-set panoptic baseline (PQ=0.473 full nuScenes, PQ=0.228 mini).

Evaluation protocol aligned with **"3D Open-Vocabulary Panoptic Segmentation with 2D-3D Vision-Language Distillation"** (Xiao et al., ECCV 2024, arXiv:2401.02402).  
We adopt their **B12/N4** split and their two novel loss functions (L_O + L_V).

---

## 1. Where We Are (Baseline State)

### Current best checkpoint: `panoptic/checkpoints_token_full_v2/best.pth`

| Mode | PQ | SQ | RQ | mIoU | Notes |
|---|---|---|---|---|---|
| Linear | **0.4759** | 0.7912 | 0.5841 | 0.4311 | Best closed-set |
| Token-16 | 0.4684 | 0.7939 | 0.5751 | 0.4190 | Token head, closed-set |
| Ensemble α=0.0 | 0.4684 | 0.7939 | 0.5751 | 0.4190 | Same as token |
| Token-OpenVocab | 0.4684 | — | — | — | **0 novel detections** |
| Backbone-Pool | 0.2090 | 0.6655 | 0.2479 | 0.1688 | 25M voxels as "pole" |

### Critical Diagnosis

**The token head failed at open-vocabulary** for two reasons:

1. **Contrastive loss collapsed**: Trained against 16 fixed prototypes → token head learned 16-class softmax, not a general CLIP encoder. Every query maps to one of 16 known prototypes regardless of input.

2. **Region features are per-voxel, not per-region**: Backbone outputs per-voxel CLIP features. Mask-pooling of these was noisy (25M voxels labeled "pole") because features weren't trained to be coherent within instance boundaries.

3. **Backbone was frozen throughout**: Features could not adapt to region-level semantics. A "car" voxel on the door and a "car" voxel on the roof have similar per-voxel features — but a region pooling of both should give a clear "car" descriptor. Frozen features don't achieve this.

---

## 2. Evaluation Framework (Paper-Aligned)

### 2.1 Reference Paper

**"3D Open-Vocabulary Panoptic Segmentation with 2D-3D Vision-Language Distillation"**  
Xiao et al., ECCV 2024 · arXiv:2401.02402  
GitHub: `https://github.com/nvlabs/3d-ovps` (Waymo / Google Brain / NVIDIA)

Their method = P3Former backbone + frozen OpenSeg CLIP features fused at voxel level + two distillation losses.  
**Our method differs** in that our backbone (DisNet/MinkUNet) is being fine-tuned E2E — we don't have a separate frozen CLIP camera branch fused with LiDAR. We distill through a single backbone that already outputs 768-d CLIP-aligned features.

---

### 2.2 The B12/N4 Split (nuScenes 16 classes → 12 base + 4 novel)

The paper uses a **3:1 ratio**: 12 base (seen) + 4 novel (unseen).

**Split used in their main Table 1 (our primary target split)**:

| Category | Type | Split |
|---|---|---|
| barrier | thing | **BASE** |
| bicycle | thing | **NOVEL** |
| bus | thing | **NOVEL** |
| car | thing | **BASE** |
| construction_vehicle | thing | **BASE** |
| motorcycle | thing | **NOVEL** |
| pedestrian | thing | **BASE** |
| traffic_cone | thing | **BASE** |
| trailer | thing | **BASE** |
| truck | thing | **BASE** |
| driveable_surface | stuff | **BASE** |
| other_flat | stuff | **BASE** |
| sidewalk | stuff | **BASE** |
| terrain | stuff | **BASE** |
| manmade | stuff | **BASE** |
| vegetation | stuff | **NOVEL** |

> **Novel things** = `bicycle`, `bus`, `motorcycle` (3 classes, index: 2, 3, 6)  
> **Novel stuff** = `vegetation` (1 class, index: 16)

The model is **trained on only the 12 base classes**. At test time, all 16 class prototypes are passed to the classification head (12 base + 4 novel text embeddings). Novel classes receive **zero training supervision** — they can only be detected if the CLIP embedding space generalizes.

---

### 2.3 Metrics to Report

Following the paper exactly, report separately for:

| Metric | Description |
|---|---|
| **PQ** | Overall panoptic quality (all 16 classes at test) |
| **PQ_B** | Base-class PQ (12 classes) |
| **PQ_N** | Novel-class PQ (4 classes) |
| **PQ_NTh** | Novel things PQ (bicycle, bus, motorcycle) |
| **PQ_NSt** | Novel stuff PQ (vegetation) |
| **SQ / RQ** | Segmentation quality / Recognition quality (for each of the above) |
| **mIoU** | Semantic segmentation mIoU (all 16 classes) |

**Reference numbers from the paper (B12/N4 split)**:

| Method | PQ | PQ_NTh | PQ_NSt | mIoU | Notes |
|---|---|---|---|---|---|
| P3Former (closed-set, full data) | 73.4 / 80.5 (Th/St) | — | — | — | Oracle ceiling |
| PFC baseline (open-vocab) | 54.8 | 37.3 | 0.5 | 55.5 | Weak on novel stuff |
| **Paper's method (ours to beat)** | **62.0** | **49.6** | **35.2** | **60.1** | LiDAR+Camera fusion |
| **Our current baseline** | 47.59 | ~0 (novel) | ~0 (novel) | 43.11 | No novel detection |

Our closed-set PQ=0.4759 is lower than the paper's PFC (0.548) largely because the paper uses P3Former (stronger backbone) and camera CLIP fusion. Our target is to **match PFC (54.8) on base classes** and get **meaningful novel PQ > 0** with our distillation approach.

---

### 2.4 What Our Eval Code Needs to Implement

**`panoptic/eval/eval_b12n4.py`** — ✅ Created. Computes:

1. **`PQ_B`** — average PQ over the 12 base class indices
2. **`PQ_NTh`** — average PQ over [bicycle, bus, motorcycle] = indices [1, 2, 5] (0-indexed)
3. **`PQ_NSt`** — PQ for [vegetation] = index [15] (0-indexed)
4. **`PQ_N`** = average of all 4 novel class PQs
5. Side-by-side comparison vs PFC baseline and Xiao et al. numbers

**`dataset/label_constants.py`** — ✅ Updated. Added:
- `NUSCENES_B12N4_NOVEL_IDX`, `NUSCENES_B12N4_BASE_IDX`
- `NUSCENES_B12N4_NOVEL_THINGS_IDX`, `NUSCENES_B12N4_NOVEL_STUFF_IDX`
- `NUSCENES_B11N5_*`, `NUSCENES_B12N4_V2_*` (alternative splits)

Training: **mask the 4 novel classes from the classification loss** (treat them as "no object" during training).  
Inference: **pass all 16 text embeddings** — novel classes discovered via CLIP similarity only.

---

### 2.5 Alternative Splits (to run as ablation after main split)

The paper also tests:
- **B11/N5**: Novel = bicycle, car, construction_vehicle, terrain, manmade
- **B12/N4 variant 2**: Novel = construction_vehicle, traffic_cone, other_flat, manmade

Run these only after the primary B12/N4 split is working.

---

## 3. What Needs to Change (Architecture)

### 3.0 Why the Backbone Encoder (conv1-conv4) Stays Frozen

The MinkUNet backbone has two halves:

```
Encoder: conv1 → conv2 → conv3 → conv4 (stride 2→4→8→16)
Decoder: convtr4 → convtr3 → convtr2 → convtr1 → final
```

**The encoder is frozen. Here is the exact reasoning:**

#### Reason 1: CLIP alignment is the supervision source for L_O and L_V

The open-vocab losses depend on backbone features being in CLIP space:
- `L_O`: `1 - cos(v_q, w_q)` where `w_q = mask_pool(backbone_feats)` — if backbone drifts from CLIP, `w_q` is garbage and this loss teaches nothing
- `L_V`: `L1(F_rec, F_backbone)` — if backbone drifts, the voxel reconstruction target is meaningless

The encoder was pre-trained by distilling from OpenSeg/CLIP on millions of frames. Those weights encode the precise 768-d directions in which "car" voxels must point to be near CLIP's text embedding for "car". Panoptic task gradients from 323 mini scenes would overwrite this in a few epochs.

#### Reason 2: 323 samples ≠ enough to retrain 8M encoder params

conv1-conv4 has ~8M parameters. nuScenes mini has 323 training scenes. With this ratio, the encoder would memorise training scenes and destroy the generalisation that lets it recognise novel classes (bicycle, vegetation) via CLIP similarity at test time.

#### Reason 3: SceneContextEncoder already solves the real problem

The motivation for training the encoder was to make "car near black building" work — i.e., each voxel needs scene context. The `SceneContextEncoder` (Transformer at the bottleneck) does this **without touching the encoder weights**:

```
conv4 (frozen) → bottleneck (stable CLIP-aligned)
                      ↓
         SceneContextEncoder (TRAINABLE, NEW)
         — car voxel attends to building voxels
         — feature now encodes "car in context of building"
                      ↓
         convtr4-final (TRAINABLE, backbone_lr=5e-6)
```

This gives you context-aware features while preserving CLIP alignment.

#### When to unfreeze the encoder (future step, not now)

| Condition | What to unfreeze | LR |
|---|---|---|
| Gate 4 (full 28k frames) converged, PQ_NTh < 30 | conv4 only | 1e-7 |
| L_O + L_V plateaued on full data | conv3 + conv4 | 1e-7 |
| Encoder warmstarted from E2E closed-set run | conv3 + conv4 | 5e-7 |
| **Never** | conv1 + conv2 | — low-level geometry, no benefit |

---

```
CLOSED-SET (current):
  query → Linear(256 → 17) → CE loss → argmax → fixed label

PAPER METHOD (Xiao et al.):
  query → class_embed_head(256 → 768) → cosine_sim with text[C_B + C_N] → class score
  mask  → pool(frozen_CLIP_voxel_feats) → w_q               [object-level distillation target]
  L_O = 1 - cos(v_q, w_q)                                   [object-level distillation loss]
  F_rec = M_Q^T @ F_Q_emb → L_V = L1(F_rec, F_vclip)       [voxel-level distillation loss]

OUR APPROACH (same principle, single backbone):
  query → RegionCLIPHead(256 → 768) → cosine_sim with text[all 16 at test] → class score
  mask  → RegionPooler(backbone_CLIP_feats) → w_q            [object-level distillation target]
  L_O = 1 - cos(v_q, w_q)                                   [same loss, no camera branch]
  F_rec = M_Q^T @ F_Q_emb → L_V = L1(F_rec, F_backbone)    [voxel-level distillation]
  TRAINING: only 12 base class text embeds for cls loss
  INFERENCE: pass all 16 class text embeds (novel classes discovered via CLIP sim)
```

### 3.2 Backbone Change (E2E)

```
FROZEN (current):
  DisNet backbone → no gradients → fixed 768-d per-voxel features

E2E decoder_only mode:
  DisNet encoder stages (conv1-conv4)  → STILL FROZEN  (preserve spatial hierarchy)
  DisNet decoder stages (convtr1-4, final) → TRAINABLE  (adapt feature projection)
  LR: backbone 5e-6, decoder 1e-4  (10-20x differential)
```

### 3.3 Supervision Change

```
CLOSED-SET training:
  text_prototypes = ["a car in a scene", ...]  ← 1 phrase per class
  
OPEN-VOCAB training:
  text_prototypes[car] = [
    "a car in a scene", "an SUV", "a sedan", "a red car",
    "a car near a building", "a car at an intersection", ...
  ]  ← 30+ phrases per class, randomly sampled per batch
```

---

## 4. New Files Created

| File | Role | Status |
|---|---|---|
| `panoptic/models/context_encoder.py` | **SceneContextEncoder** (MHSA at bottleneck) + ContextualRegionPooler + ContextAwareBackboneWrapper | ✅ Created |
| `panoptic/models/open_vocab_head.py` | RegionCLIPHead + ObjectnessHead | ✅ Created |
| `panoptic/models/diverse_prompts.py` | 30+ prompt templates per class | ✅ Created |
| `panoptic/losses/open_vocab_loss.py` | **L_O + L_V** (paper-aligned) + B12/N4 novel masking | ✅ Updated |
| `panoptic/train/train_panoptic_e2e.py` | E2E training (closed-set, backbone unfrozen) | ✅ Created |
| `panoptic/train/train_openvocab_e2e.py` | Full open-vocab + E2E + context encoder | ✅ Updated |
| `panoptic/configs/mini_openvocab_e2e.yaml` | Config: open-vocab E2E, mini, B12/N4 | ✅ Updated |
| `panoptic/configs/mini_e2e_encoder.yaml` | Config: closed-set E2E, mini | ✅ Created |
| `panoptic/configs/full_nuscenes_e2e.yaml` | Config: closed-set E2E, full | ✅ Created |
| `panoptic/scripts/train_e2e_mini_job.sh` | Cluster job: closed-set E2E, mini | ✅ Created |
| `panoptic/scripts/train_openvocab_mini_job.sh` | Cluster job: open-vocab E2E, mini | ✅ Created |
| `panoptic/eval/eval_b12n4.py` | B12/N4 paper-aligned evaluation script | ✅ Created |
| `dataset/label_constants.py` | B12/N4 / B11/N5 split index constants | ✅ Updated |

---

## 5. Loss Components (Paper-Aligned)

Paper uses: **L = w_α·L_cls + w_β·L_mask + w_λ·L_O + w_γ·L_V**  
Paper weights: w_α=1, w_β=1, w_λ=1, w_γ=0.1

| Loss | Symbol | Weight (paper) | What it does |
|---|---|---|---|
| `cls` (focal) | L_cls | 1.0 | Focal loss: query class embed vs 12 base-class text embeds. **B12 only at train.** |
| `mask` | L_mask | 1.0 | Per-voxel mask BCE + Dice. Unchanged. |
| `object_distill` | L_O | 1.0 | `1 - cos(v_q, w_q)` for matched queries. Pulls query embed toward mask-pooled backbone CLIP. Key for novel **things**. |
| `voxel_distill` | L_V | 0.1 | `L1(M_Q^T @ F_Qemb, F_backbone)`. Reconstruct per-voxel CLIP from query weighted sum. Key for novel **stuff**. Applied to ALL queries. |

**Why L_V is the most important loss**: In the paper's ablation, adding L_V improved novel stuff PQ from 0.2 → 35.2. This loss works because it doesn't require ground truth matches — unmatched queries (which cover novel class regions) are supervised indirectly through the CLIP voxel reconstruction objective.

**Key difference from our current `open_vocab_loss.py`**: Our current implementation uses an InfoNCE contrastive loss. We need to replace/augment it with L_O (cosine distance) + L_V (L1 voxel reconstruction), matching the paper exactly.

---

## 6. Inference Change (Paper-Aligned)

```python
# CLOSED-SET (current):
pred_class = argmax(query @ class_embed.T)       # 16 known options, always correct class

# PAPER-ALIGNED OPEN-VOCAB (target):
# At TRAIN time: only 12 base class text embeds used in cls loss
text_train = clip.encode_text(BASE_12_CLASSES)    # shape [12, 768]
logits_train = (v_q @ text_train.T) / T           # [N_queries, 12]

# At TEST time: pass all 16 text embeds (novel 4 discovered via similarity)
text_test = clip.encode_text(ALL_16_CLASSES)       # shape [16, 768]
logits_test = (v_q @ text_test.T) / T              # [N_queries, 16]
pred_class = argmax(logits_test)                   # can now produce novel class predictions
```

---

## 7. Execution Plan (Gate-by-Gate)

### Gate 0 — Environment + Eval Framework Ready ← YOU ARE HERE
- [ ] `opensc_env` conda env activated
- [ ] MinkowskiEngine imports correctly
- [ ] `panoptic/checkpoints/best.pth` exists (warm-start source)
- [ ] **`panoptic/eval/eval_b12n4.py` created** — new eval script that computes PQ_B, PQ_NTh, PQ_NSt (see Section 2.4)
- [ ] B12/N4 class split constants defined in `dataset/label_constants.py`

### Gate 1 — Closed-Set E2E on Mini
**Goal**: Verify backbone unfreezing doesn't destabilize training. Compare to frozen baseline.

**Command**:
```bash
python -u panoptic/train/train_panoptic_e2e.py \
    --config panoptic/configs/mini_e2e_encoder.yaml
```

**Accept if**: PQ ≥ 0.228 (match frozen baseline). Abort if PQ < 0.15 (training unstable).

**Expected**: ~4-6 GB GPU, ~2-3 min/epoch, 50 epochs.

**What to watch**:
- Loss should decrease steadily
- LR log should show `lr_dec` and `lr_bb` (backbone LR appears at epoch 5 when unfrozen)
- GPU memory should not OOM (if it does: set `mixed_precision: true` or reduce `num_mask_pts`)

---

### Gate 2 — Open-Vocab E2E on Mini (Paper Losses)
**Goal**: Validate L_O + L_V distillation losses train without collapse. Evaluate with B12/N4 split.

**Command**:
```bash
python -u panoptic/train/train_openvocab_e2e.py \
    --config panoptic/configs/mini_openvocab_e2e.yaml
```

**Evaluation with B12/N4 split**:
```bash
python panoptic/eval/eval_b12n4.py \
    --checkpoint panoptic/checkpoints_openvocab_mini/best.pth \
    --split b12n4_main \
    --version v1.0-mini
```

**Accept if**:
- `voxel_distill` (L_V) loss decreases steadily
- `object_distill` (L_O) loss decreases for matched queries
- **PQ_NTh > 0** — at least some bicycle/bus/motorcycle detections
- **PQ_NSt > 0** — at least some vegetation detections

---

### Gate 3 — Open-Vocab Qualitative Test
Run a quick qualitative check on mini val with custom text queries:

```bash
python panoptic/scripts/eval_openvocab_custom.py \
    --checkpoint panoptic/checkpoints_openvocab_mini/best.pth \
    --queries "car" "red SUV" "car near a building" "person crossing" "large truck"
```

*(This eval script to be created after Gate 2 passes)*

**Accept if**: "car" and "person" detect reasonably. Novel queries ("red SUV") detect something plausible.

---

### Gate 4 — Full nuScenes Open-Vocab E2E
**Only run after Gate 3 passes.**

Config to be created: `panoptic/configs/full_nuscenes_openvocab_e2e.yaml`  
(copy of `full_nuscenes_e2e.yaml` + open_vocab section from `mini_openvocab_e2e.yaml`)

**Expected**: ~8-12 GB GPU, ~60-90 min/epoch, 30 epochs.

---

## 8. Ablation Table to Fill

All entries use **B12/N4 split** (novel = bicycle, bus, motorcycle, vegetation).

| Experiment | PQ | PQ_B | PQ_NTh | PQ_NSt | mIoU | Config |
|---|---|---|---|---|---|---|
| **Paper: PFC baseline** | 54.8 | — | 37.3 | 0.5 | 55.5 | — |
| **Paper: Xiao et al.** | **62.0** | — | **49.6** | **35.2** | **60.1** | — |
| Our closed-set (all 16, existing) | 47.59 | — | ~0 | ~0 | 43.11 | `full_nuscenes_token_v2.yaml` |
| Our closed-set B12 eval (existing ckpt) | — | — | 0 | 0 | — | run `eval_b12n4.py` |
| E2E decoder_only, mini | — | — | — | — | — | `mini_e2e_encoder.yaml` |
| Open-vocab E2E B12/N4, mini | — | — | — | — | — | `mini_openvocab_e2e.yaml` |
| Open-vocab E2E B12/N4, full nuScenes | — | — | — | — | — | `full_nuscenes_openvocab_e2e.yaml` |

---

## 9. Key Hyperparameter Decisions (Paper + Our Adjustments)

| Parameter | Paper Value | Our Value | Reason |
|---|---|---|---|
| `w_cls` | 1.0 | 1.0 | Match paper. Focal loss on B12 only. |
| `w_mask` | 1.0 | 1.0 | Unchanged. |
| `w_object_distill` (L_O) | 1.0 | 1.0 | Match paper. |
| `w_voxel_distill` (L_V) | 0.1 | 0.1 | Low weight — larger causes training instability. |
| `backbone_lr` | N/A (frozen) | 5e-6 | 20× lower than decoder. **E2E advantage over paper.** |
| `unfreeze_after_epoch` | N/A | 3-5 | Decoder warms up first. |
| `unfreeze_mode` | N/A | `decoder_only` | Safest. Preserves encoder CLIP alignment. |
| `temperature` | learnable | learnable | InfoNCE temperature adapts. |
| `optimizer` | AdamW, wd=0.01, lr=8e-4 | AdamW, lr=1e-4 | Paper uses higher LR on P3Former. |
| `epochs` | 40 | 30-50 | Paper uses 40. |
| Training classes | B12 | B12 | Novel classes excluded from cls loss. |
| Inference classes | B12 + N4 = 16 | 16 | All 16 text embeds at test time. |

---

## 10. Risk Registry

| Risk | Likelihood | Mitigation |
|---|---|---|
| Token/region_clip collapse | Medium | Lower `weight_region_clip`, check cosine sim distribution |
| Backbone forgetting CLIP space | Low | `backbone_lr` is 20× lower; `decoder_only` mode |
| OOM during E2E | Low-Medium | AMP enabled; `num_mask_pts=500`; batch=1 |
| Open-vocab PQ lower than closed-set | High (expected) | Open-vocab is harder at inference — PQ is not the right metric. Check novel detections. |
| Mini data too small | Medium | Mini has only 323 train samples. Full run needed for real eval. Gate 2 is only a smoke test. |

---

## 11. Do-Not-Do Rules

- **Do not** run full nuScenes E2E before mini Gate 1 and Gate 2 pass.
- **Do not** set `backbone_lr > 1e-4` (will catastrophically forget CLIP alignment).
- **Do not** unfreeze `all` backbone layers on mini (15.6M params, too few training samples).
- **Do not** compare open-vocab PQ directly to closed-set PQ — they use different inference heads.
- **Do not** increase `weight_region_clip > 6.0` (masks will degrade to serve CLIP, not vice versa).

---

## 12. Cluster Run Instructions

### Interactive (debugging):
```bash
bsub -Is -q batch_a100 -n 4 -M 8192 -W 12:00 \
  -gpu "num=1:mode=exclusive_process:mps=no" /bin/bash

module load conda && conda activate opensc_env
module load cuda/11.0.2 && module load cudnn/11.0_v8.0
export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model
cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model
```

### Gate 1 — Closed-Set E2E Mini (batch):
```bash
bsub < panoptic/scripts/train_e2e_mini_job.sh
```

### Gate 2 — Open-Vocab E2E Mini (batch):
```bash
bsub < panoptic/scripts/train_openvocab_mini_job.sh
```

### Monitor:
```bash
bjobs -l                          # job status
tail -f panoptic/logs/e2e_mini_JOBID.stdout   # live logs
```
