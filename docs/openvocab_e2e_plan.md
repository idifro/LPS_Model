# Open-Vocabulary Decoder Fine-Tuning Plan (nuScenes)

## Document Purpose

Single source of truth for the **open-vocabulary + decoder fine-tuning** phase of this project.  
Builds directly on the closed-set panoptic baseline (PQ=0.473 full nuScenes, PQ=0.228 mini).

> **Naming correction (Jun 2, 2026):** This plan was previously titled "E2E Encoder Training." That was misleading. The encoder (conv1–conv4) stays **frozen throughout Gates 0–4**. What is being trained is the **decoder + SceneContextEncoder + query heads**. True encoder unfreezing is a conditional Gate 5 step requiring precomputed external CLIP anchors and full nuScenes data. See Section 0 for the complete conceptual justification before reading anything else.

Evaluation protocol aligned with **"3D Open-Vocabulary Panoptic Segmentation with 2D-3D Vision-Language Distillation"** (Xiao et al., ECCV 2024, arXiv:2401.02402).  
We adopt their **B12/N4** split and their two novel loss functions (L_O + L_V).

---

## 0. Conceptual Foundation — Read This First

### 0.1 How the Backbone Became CLIP-Aligned

The backbone (DisNet/MinkUNet18A) was trained **before** any panoptic work, on a completely separate pre-distillation task:

```
For every LiDAR frame in a large corpus:
  1. Take 6 surround-view camera images
  2. Run OpenSeg (a CLIP-based 2D segmentor) on each image
     → 768-d CLIP feature vector per pixel
  3. Project 2D pixel features onto the corresponding 3D LiDAR voxels
     → fused 768-d CLIP target per voxel  (F_openseg)
  4. Train the 3D backbone to match those targets:

     Loss = 1 - cos( backbone_output_voxel,  F_openseg )

     This ran on millions of frames from the OpenScene dataset.
```

**Result:** every voxel the backbone produces is a 768-d vector that lives in the same geometric space as CLIP's text embeddings. A "car" voxel output points in the same direction in 768-d space as the CLIP text embedding for `"a photo of a car"`. This is not learned during panoptic training — **it was already there**.

Confirmed by `run/distill.py`:
```python
optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
# Loss = 1 - cos(backbone_output, fused_openseg_feature)  [in distill() function]
```

### 0.2 Why That Enables Novel Class Detection (The Core Idea)

CLIP's text encoder was trained on billions of image-text pairs. Its embedding space has a critical property: **semantic similarity = geometric proximity**:

```
"bicycle"    ── close to ──  "motorcycle"     (both two-wheeled vehicles)
"vegetation" ── close to ──  "tree", "bush"
"car"        ── far from  ──  "vegetation"
```

Because the 3D backbone output is in this same space, at test time:

```python
text_embed  = clip.encode_text("a bicycle")    # 768-d text vector
voxel_feat  = backbone(lidar_point)            # 768-d 3D voxel vector

similarity  = cos(voxel_feat, text_embed)      # HIGH if it is a bicycle
```

The model **never sees a bicycle during panoptic training** (bicycle is a novel class, excluded from all training labels). But because the voxel features already live in CLIP space from pre-distillation, the cosine similarity to the text "bicycle" is meaningful. That is the entire mechanism for novel class detection.

### 0.3 What Is and Is Not CLIP-Aligned in the Architecture

```
Encoder: conv1 → conv2 → conv3 → conv4   (stride 2→4→8→16)
         ↓ produces standard sparse conv geometry features (NOT CLIP vectors)
                                     ↓
                         Final output layer (768-d projection)
                         ← THIS is where CLIP alignment lives (F_backbone)
                                     ↓
Decoder: convtr4 → convtr3 → convtr2 → convtr1 → final
         ↓ upsamples back to voxel resolution, was never trained for region tasks
                                     ↓
                         Per-voxel 768-d CLIP-aligned output
```

**What is CLIP-aligned:** The backbone's **final-layer output** `F_backbone` — 768-d per voxel. This is what `L_O` and `L_V` use as the distillation target.

**What is NOT CLIP-aligned:** Intermediate encoder stages (conv1–conv4) — standard sparse conv feature maps in arbitrary dimensional spaces.

### 0.4 Why the Current Model Fails at Open-Vocabulary (0 Novel Detections)

Three compounding failures:

1. **Classification head learned a fixed lookup table**: Trained with CE loss against 16 fixed class prototypes → head memorised 16-class softmax. Every query maps to one of 16 known classes regardless of CLIP similarity. Cannot generalise to "bicycle" at test time.

2. **Mask-pooling of per-voxel features was noisy**: When you pool thousands of voxels inside an instance mask to get a region descriptor, you get a noisy average because the decoder was never trained to make voxels *within the same instance* point coherently. The "25M voxels as pole" result (PQ=0.209, Backbone-Pool mode) shows this directly.

3. **Argmax always favours base classes**: At test time, even if CLIP similarity to "bicycle" is high, the 12 base classes were trained with explicit supervision and always produce higher logits. Novel classes never win the argmax without a threshold correction.

### 0.5 Why Decoder Fine-Tuning (Not Encoder Unfreezing) Is the Fix

**The encoder output `F_backbone` serves as the distillation target for both `L_O` and `L_V`**:
- `L_O`: `1 - cos(query_embed, mask_pool(F_backbone))` — target is `F_backbone`
- `L_V`: `L1(reconstructed_voxels, F_backbone)` — target is `F_backbone`

If you unfreeze the encoder, `F_backbone` changes every training step:
```
Unfrozen encoder → F_backbone moves each step
L_V target = F_backbone  ← moving target
→ loss is chasing itself → collapses
```

The encoder must stay frozen **as long as it is its own supervision signal**. This is not a limitation — it is architecturally intentional. The decoder is fine-tuned to **aggregate and preserve** the CLIP alignment that already exists in `F_backbone`, producing instance-level CLIP descriptors instead of per-voxel ones.

The complete training signal flow:
```
TRAINING (12 base classes, encoder frozen):

LiDAR → Encoder (FROZEN) → F_backbone [768-d per voxel, fixed, CLIP-aligned]
                                  ↓ detached — serves as fixed supervision target
              SceneContextEncoder (TRAINABLE) → cross-voxel context
                                  ↓
                       Decoder (TRAINABLE)
                                  ↓
                    Query embeddings [768-d per instance]
                     ↙                         ↘
   L_O: cos(query, mask_pool(F_backbone))     L_V: reconstruct F_backbone from queries
   [matched queries only, base classes]        [ALL queries, incl. novel regions]
              ↓
   L_cls: focal vs 12 base class text embeds (bicycle/bus/motorcycle/vegetation EXCLUDED)


INFERENCE (all 16 classes):

Query embedding [768-d] → cos_sim with text["car","bicycle","bus",..."vegetation"]
                                                   ↑ all 16 at test time
                          Novel classes detected via CLIP similarity — no training labels needed
```

### 0.6 How L_V Enables Novel Class Detection Without Any Labels

`L_V` is the key mechanism. It says: the weighted sum of all query embeddings, when projected back to voxel space, must reconstruct `F_backbone` per voxel.

```
F_rec = Σ_q  mask_q(v) * query_embed_q     for each voxel v
L_V   = L1(F_rec, F_backbone)
```

This applies to **every** query — including queries that cover vegetation, bicycle, and bus regions. These queries receive:
- **Zero L_cls supervision** (novel classes excluded from classification loss)
- **Zero L_O supervision** (L_O only runs on GT-matched queries)
- **Full L_V supervision** — the query must still reconstruct the CLIP-aligned `F_backbone` in those voxels

So every query over a bicycle region is forced to stay in the CLIP manifold near "bicycle" — even though the word "bicycle" never appears in training. At test time, when you compute cosine similarity with the text "bicycle", those queries produce high scores.

**Paper ablation proof:** Removing L_V drops novel stuff PQ from 35.2 → 0.2. Adding L_V is what makes vegetation detectable at test time with zero vegetation training labels.

---

---

## 0.7 Will We Hit PQ=0.5? Realistic Trajectory

### Closed-Set (Gates 0–1): Yes, Likely

| Step | PQ | Risk |
|---|---|---|
| Frozen baseline (current) | 0.473 | — |
| + decoder fine-tune + SceneContextEncoder (Gate 1) | **~0.49–0.50** | Low |

Decoder fine-tune alone should close the +0.027 gap. Low risk.

### Open-Vocab B12/N4 (Gates 2–4): PQ=0.5 Is the Optimistic End

Open-vocab PQ will **initially drop below 0.473** because novel classes compete at inference with weaker signal. Realistic trajectory with full plan implemented:

| Stage | PQ | PQ_NTh | PQ_NSt | Notes |
|---|---|---|---|---|
| Open-vocab head + L_O + L_V (Gate 2, mini) | ~0.42–0.46 | >0 | >0 | Smoke test only, noisy |
| + τ threshold + L_bg | ~0.44–0.48 | improves | stable | Already coded in |
| + density-stratified L_V | ~0.45–0.49 | sparse things improve | — | Already coded in |
| + full nuScenes (Gate 4) | **~0.47–0.52** | 0.10–0.25 | 0.20–0.35 | Real eval |
| + pseudo-labels (after Gate 2 ckpt) | **~0.48–0.52** | 0.15–0.25 | 0.25–0.35 | Pushes NTh most |

**PQ=0.5 open-vocab is at the optimistic end.** Most likely landing: **0.45–0.48** without pseudo-labels, **0.48–0.52** with pseudo-labels on full nuScenes.

### What Actually Drives PQ (PQ = SQ × RQ)

Current SQ = **0.79** (already good — mask quality is fine). The bottleneck is **RQ**:

| Component | Problem | Fix (coded?) |
|---|---|---|
| Base RQ | Novel classes steal predictions via τ threshold | τ tuning on val ✅ |
| Novel thing RQ | Sparse classes (bicycle=80pts) near-zero L_V gradient | Density-stratified L_V ✅ |
| Novel stuff RQ | L_V works but unmatched queries unsupervised | L_bg ✅ |
| Novel thing RQ on full data | Zero mask supervision for novel things | Pseudo-labels (after Gate 2) ⏳ |

### Three Risks That Block PQ=0.5

| Risk | Signal to watch | What to do |
|---|---|---|
| **Risk 1**: L_V flat after epoch 10 at Gate 2 | `loss_voxel_distill` in logs | Stop, debug `F_backbone.detach()`, check encoder frozen |
| **Risk 2**: τ not wired into Gate 2 eval | `PQ_NTh=0` after training | Use `--novel_threshold 0.4` in `eval_b12n4.py` call ✅ |
| **Risk 3**: L_bg missing from Gate 2 | Unmatched queries drift | Already added to `open_vocab_loss.py` ✅ |

### Honest Publication Target

Do **not** use PQ=0.5 as the primary open-vocab metric. Frame instead as:

> *"Camera-free, single-backbone LiDAR open-vocabulary panoptic segmentation via voxel-level CLIP distillation. Achieves competitive novel stuff detection (PQ_NSt > 20) and novel thing detection (PQ_NTh > 10) without any novel-class training labels and without camera fusion."*

This is defensible and differentiates from Xiao et al. (camera+LiDAR fusion, stronger backbone, PQ=62.0). Chasing their absolute number is the wrong target — our contribution is the camera-free single-backbone approach.

**Recommended targets for paper table:**

| Metric | Target | Confidence |
|---|---|---|
| Closed-set PQ | ≥ 0.50 | High |
| PQ_B (base 12 open-vocab) | ≥ 0.46 | High |
| PQ_NSt (vegetation) | > 20 | Medium — L_V does this |
| PQ_NTh (bicycle/bus/motorcycle) | > 10 | Medium — needs pseudo-labels |
| Novel detections exist | Yes | High — τ + L_bg coded in |

---

## 1. Where We Are (Baseline State)

### Current best checkpoint: `/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/panoptic/checkpoints_token_full_v2/best.pth`

| Mode | PQ | SQ | RQ | mIoU | Notes |
|---|---|---|---|---|---|
| Linear | **0.4759** | 0.7912 | 0.5841 | 0.4311 | Best closed-set |
| Token-16 | 0.4684 | 0.7939 | 0.5751 | 0.4190 | Token head, closed-set |
| Ensemble α=0.0 | 0.4684 | 0.7939 | 0.5751 | 0.4190 | Same as token |
| Token-OpenVocab | 0.4684 | — | — | — | **0 novel detections** |
| Backbone-Pool | 0.2090 | 0.6655 | 0.2479 | 0.1688 | 25M voxels as "pole" |

### Critical Diagnosis

**The token head failed at open-vocabulary** for three reasons:

1. **Contrastive loss collapsed**: Trained against 16 fixed prototypes → token head learned 16-class softmax, not a general CLIP encoder. Every query maps to one of 16 known prototypes regardless of input.

2. **Region features are per-voxel, not per-region**: Backbone outputs per-voxel CLIP features. Mask-pooling of these was noisy (25M voxels labeled "pole") because features weren't trained to be coherent within instance boundaries.

3. **Decoder was frozen throughout**: The decoder (`convtr1–4`, `final`) was never fine-tuned under region-level supervision. It cannot produce instance-coherent CLIP features from the per-voxel encoder output — only per-voxel CLIP features propagate unchanged. The **fix is to fine-tune the decoder** under `L_O + L_V`, not to unfreeze the encoder.

---

## 2. Evaluation Framework (Paper-Aligned)

### 2.1 Reference Paper

**"3D Open-Vocabulary Panoptic Segmentation with 2D-3D Vision-Language Distillation"**  
Xiao et al., ECCV 2024 · arXiv:2401.02402  
GitHub: `https://github.com/nvlabs/3d-ovps` (Waymo / Google Brain / NVIDIA)

Their method = P3Former backbone + frozen OpenSeg CLIP features fused at voxel level + two distillation losses.  
**Our method differs** in that we have a single DisNet/MinkUNet backbone whose **final-layer output is already 768-d CLIP-aligned** (pre-distilled from OpenSeg). We do not have a separate frozen camera CLIP branch. We fine-tune the decoder and query heads to aggregate that per-voxel CLIP alignment into region-level descriptors.

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

### 3.0 CLIP Alignment: What Is and Is Not Aligned

The MinkUNet backbone has two halves:

```
Encoder: conv1 → conv2 → conv3 → conv4   (stride 2→4→8→16)
                                    ↓
                        bottleneck (768-d, CLIP-aligned final output)
                                    ↓
Decoder: convtr4 → convtr3 → convtr2 → convtr1 → final   (768-d output per voxel)
```

**What is CLIP-aligned:** The backbone's **final-layer output** (768-d per-voxel) is in CLIP space. It was trained with:
```
Loss = 1 - cos(backbone_output_voxel, fused_openseg_2d_feature)
```
on millions of frames of OpenSeg 2D features. This is the `F_backbone` used as the distillation target in `L_V` and `L_O`.

**What is NOT CLIP-aligned:** The intermediate encoder stages (conv1–conv4) produce standard sparse conv feature maps — not CLIP vectors. Only the decoder's final projection head lifts features into 768-d CLIP space.

**Why this matters:** `L_V` and `L_O` both use `F_backbone` (the frozen encoder's final output) as the supervision signal. If the encoder drifts, `F_backbone` becomes meaningless and both losses teach nothing. The encoder must stay frozen.

---

### 3.1 Why the Encoder (conv1–conv4) Stays Frozen

**The encoder is frozen throughout Gates 0–4. Here is the exact reasoning:**

#### Reason 1: Encoder output is the distillation target for L_O and L_V
- `L_O`: `1 - cos(v_q, w_q)` where `w_q = mask_pool(F_backbone)` — if `F_backbone` drifts from CLIP, `w_q` is garbage
- `L_V`: `L1(F_rec, F_backbone)` — if `F_backbone` drifts, the voxel reconstruction target is meaningless

The encoder was pre-distilled from OpenSeg/CLIP on a large corpus. Panoptic task gradients from 323 mini scenes would corrupt this in a few epochs.

#### Reason 2: 323 samples cannot retrain 8M encoder parameters
conv1–conv4 has ~8M parameters. nuScenes mini has 323 training scenes. The encoder would overfit, losing the CLIP generalisation that allows novel class detection at test time (bicycle, vegetation receive zero training labels).

#### Reason 3: SceneContextEncoder solves the actual problem without touching the encoder
The real problem is that per-voxel features lack instance-level context. The `SceneContextEncoder` (Transformer at the bottleneck) adds this:

```
conv4 (frozen) → bottleneck F_backbone (stable CLIP-aligned)
                      ↓
         SceneContextEncoder (TRAINABLE, NEW)
         — car voxel attends to building voxels
         — feature now encodes "car in context of building"
                      ↓
         convtr4–final (TRAINABLE, decoder_lr=1e-4)
                      ↓
         RegionCLIPHead / ObjectnessHead (TRAINABLE)
```

This gives context-aware, region-coherent CLIP features without touching the encoder.

#### When encoder unfreezing is considered (Gate 5, future, conditional)

| Condition | What to unfreeze | LR |
|---|---|---|
| Gate 4 (full 28k frames) converged AND PQ_NTh < 20 AND L_V plateaued | conv4 only | 1e-7 |
| Above + still no improvement after 10 epochs | conv3 + conv4 | 1e-7 |
| **Never** | conv1 + conv2 | low-level geometry, zero open-vocab benefit |
| **Never** | any encoder layer on mini (323 scenes) | insufficient data |

Monitor `cos(F_backbone, CLIP_text_embed)` — if it drops after unfreezing, re-freeze immediately.

---

### 3.2 What IS Being Trained (Decoder Fine-Tuning)

```
FROZEN (original closed-set baseline):
  DisNet full backbone → no gradients → fixed 768-d per-voxel features → Linear(256→17) → CE loss

CURRENT PLAN (decoder_only fine-tuning):
  DisNet encoder (conv1–conv4)          → FROZEN   (preserve CLIP alignment of F_backbone)
  SceneContextEncoder (bottleneck)      → TRAINABLE (new, cross-voxel attention)
  DisNet decoder (convtr1–4, final)     → TRAINABLE (decoder_lr=1e-4)
  RegionCLIPHead (256→768)              → TRAINABLE (replaces Linear(256→17))
  ObjectnessHead                        → TRAINABLE
```

**LR hierarchy:**  
- Decoder: `1e-4`  
- SceneContextEncoder: `5e-5`  
- Encoder: `0` (frozen, no gradient)

> Note: `backbone_lr=5e-6` in config refers to the SceneContextEncoder LR — not the encoder conv stages. The encoder conv stages receive zero gradient in `decoder_only` mode.

---

### 3.3 Supervision Change (Open-Vocab vs Closed-Set)

```
CLOSED-SET training:
  text_prototypes = ["a car in a scene", ...]  ← 1 phrase per class
  
OPEN-VOCAB training:
  text_prototypes[car] = [
    "a car in a scene", "an SUV", "a sedan", "a red car",
    "a car near a building", "a car at an intersection", ...
  ]  ← 30+ phrases per class, randomly sampled per batch
  Novel classes (bicycle, bus, motorcycle, vegetation) → excluded from cls loss at train
  Novel classes → included at test (16 text embeds passed to classification head)
```

---

## 4. Files — Status

| File | Role | Status |
|---|---|---|
| `panoptic/models/context_encoder.py` | **SceneContextEncoder** (MHSA at bottleneck) + ContextualRegionPooler + ContextAwareBackboneWrapper | ✅ Created |
| `panoptic/models/open_vocab_head.py` | RegionCLIPHead + ObjectnessHead | ✅ Created |
| `panoptic/models/diverse_prompts.py` | 30+ prompt templates per class | ✅ Created |
| `panoptic/losses/open_vocab_loss.py` | **L_O + L_V** (paper-aligned) + B12/N4 novel masking | ✅ Updated |
| `panoptic/train/train_panoptic_e2e.py` | Decoder fine-tuning, closed-set (misnamed; encoder stays frozen) | ✅ Created |
| `panoptic/train/train_openvocab_e2e.py` | Full open-vocab + decoder fine-tune + context encoder | ✅ Updated |
| `panoptic/configs/mini_openvocab_e2e.yaml` | Config: open-vocab, decoder fine-tune, mini, B12/N4 | ✅ Updated |
| `panoptic/configs/mini_e2e_encoder.yaml` | Config: closed-set, decoder fine-tune, mini | ✅ Created |
| `panoptic/configs/full_nuscenes_e2e.yaml` | Config: closed-set, decoder fine-tune, full | ✅ Created |
| `panoptic/scripts/train_e2e_mini_job.sh` | Cluster job: closed-set decoder fine-tune, mini | ✅ Created |
| `panoptic/scripts/train_openvocab_mini_job.sh` | Cluster job: open-vocab decoder fine-tune, mini | ✅ Created |
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

## 7. Execution Plan — nuScenes Mini (Gate-by-Gate)

> **Scope of this plan:** nuScenes mini (323 train / 81 val scenes). Full nuScenes runs only after Gates 1–3 pass.  
> **What "E2E" means here:** Decoder + SceneContextEncoder + query heads are trainable. Encoder (conv1–conv4) is frozen throughout Gates 0–4.

---

### Gate 0 — Environment & Eval Framework

**Goal:** Confirm environment works and B12/N4 eval is ready before any training.

```bash
# 1. Get an interactive GPU node
bsub -Is -q batch_a100 -n 4 -M 8192 -W 2:00 \
  -gpu "num=1:mode=exclusive_process:mps=no" /bin/bash

# 2. Load environment
module load conda && conda activate opensc_env
module load cuda/11.0.2 && module load cudnn/11.0_v8.0
export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model
cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model

# 3. Verify MinkowskiEngine
python -c "import MinkowskiEngine as ME; print(ME.__version__)"

# 4. Verify eval script runs on existing checkpoint
python panoptic/eval/eval_b12n4.py \
    --checkpoint /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/panoptic/checkpoints_token_full_v2/best.pth \
    --split b12n4_main \
    --version v1.0-mini
```

**Accept if:** B12/N4 eval runs and prints PQ_B, PQ_NTh, PQ_NSt (expect ~0 for novel classes on existing closed-set ckpt). This is your baseline.

**Checklist:**
- [ ] `opensc_env` activates without errors
- [ ] MinkowskiEngine imports correctly
- [ ] `/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/panoptic/checkpoints_token_full_v2/best.pth` exists
- [ ] `eval_b12n4.py` runs and reports 0 novel class detections (confirms closed-set baseline)
- [ ] B12/N4 class split constants exist in `dataset/label_constants.py`

---

### Gate 1 — Closed-Set Decoder Fine-Tuning on Mini

**Goal:** Verify decoder unfreezing (+ SceneContextEncoder) does not destabilise training. Confirm PQ at least matches the frozen baseline.

**What trains:** `convtr1–4`, `final`, `SceneContextEncoder`, `PanopticDecoder` query heads.  
**What stays frozen:** `conv1–conv4` (encoder). Zero gradient flows into encoder weights.

```bash
# Submit batch job
bsub < panoptic/scripts/train_e2e_mini_job.sh

# OR run interactively
python -u panoptic/train/train_panoptic_e2e.py \
    --config panoptic/configs/mini_e2e_encoder.yaml

# Monitor
bjobs -l
tail -f panoptic/logs/e2e_mini_<JOBID>.stdout
```

**What to watch:**
- `lr_dec` (decoder LR) and `lr_ctx` (context encoder LR) appear in logs — encoder LR should be **absent** or `0.0`
- Total loss decreases by epoch 5
- GPU memory stays under 8 GB (if OOM: add `mixed_precision: true` to config, reduce `num_mask_pts`)

**Accept if:** `PQ ≥ 0.228` (matches frozen baseline on mini val).  
**Abort if:** `PQ < 0.15` after 30 epochs — decoder fine-tuning is destabilising something; check LR values.

**Expected runtime:** ~2–3 min/epoch × 50 epochs ≈ 2–2.5 hrs on 1 A100.

---

### Gate 2 — Open-Vocab Decoder Fine-Tuning on Mini (L_O + L_V)

**Goal:** Validate that paper losses (L_O + L_V) train without collapse and produce at least some novel class detections.

**What trains:** Same as Gate 1 + `RegionCLIPHead` (replaces `Linear(256→17)`) + open-vocab loss.  
**What stays frozen:** Encoder (conv1–conv4). `F_backbone` used as fixed distillation target.

```bash
# Submit batch job
bsub < panoptic/scripts/train_openvocab_mini_job.sh

# OR run interactively
python -u panoptic/train/train_openvocab_e2e.py \
    --config panoptic/configs/mini_openvocab_e2e.yaml
```

**Evaluate with B12/N4 split:**
```bash
python panoptic/eval/eval_b12n4.py \
    --checkpoint panoptic/checkpoints_openvocab_mini/best.pth \
    --split b12n4_main \
    --version v1.0-mini
```

**What to watch (in order of importance):**
1. `loss_voxel_distill` (L_V) decreases steadily → decoder queries stay on CLIP manifold ✅
2. `loss_object_distill` (L_O) decreases for matched queries → region pooling is working ✅
3. `loss_cls` decreases on B12 classes → base-class classification working ✅
4. `loss_mask` stable → mask quality not degrading ✅

**Accept if:**
- `PQ_NTh > 0` — at least some bicycle / bus / motorcycle detections
- `PQ_NSt > 0` — at least some vegetation detections
- L_V is decreasing (this is the critical signal — if L_V is flat or increasing, novel detection will fail)

**Note:** Open-vocab PQ will likely be **lower than closed-set PQ** — this is expected. Novel detection is the only metric that matters at Gate 2. Mini is too small for reliable absolute numbers.

**Expected runtime:** ~3–4 min/epoch × 50 epochs ≈ 3 hrs on 1 A100.

---

### Gate 3 — Open-Vocab Qualitative Check on Mini Val

**Goal:** Sanity-check novel detection with custom text queries.

```bash
# (Script to be created after Gate 2 passes)
python panoptic/scripts/eval_openvocab_custom.py \
    --checkpoint panoptic/checkpoints_openvocab_mini/best.pth \
    --queries "car" "bicycle" "bus" "person crossing" "vegetation" "large truck"
```

**Accept if:** "car" and "pedestrian" detect correctly. At least one novel class query ("bicycle", "vegetation") returns non-empty predictions.

---

### Gate 4 — Full nuScenes Open-Vocab (Decoder Fine-Tune)

**Only proceed after Gates 1, 2, 3 pass.**

```bash
# Config to be created:
# panoptic/configs/full_nuscenes_openvocab_e2e.yaml
# (copy mini_openvocab_e2e.yaml, update data path, epochs=30, batch_size=2)

bsub < panoptic/scripts/train_openvocab_full_job.sh
```

**Accept if:** `PQ_NTh > 20`, `PQ_NSt > 10` on full val.

**Expected runtime:** ~60–90 min/epoch × 30 epochs on 1–4 A100s.

---

### Gate 5 — [CONDITIONAL] Partial Encoder Unfreezing

**Only consider after Gate 4, only if PQ_NTh < 20 AND L_V has plateaued for 10+ epochs.**

```
Unfreeze: conv4 only (deepest encoder stage)
LR: 1e-7 (100× lower than decoder)
Monitor: cos(F_backbone, CLIP_text["car"]) each epoch
         → if cosine similarity drops > 5% from baseline: re-freeze immediately
```

**Never unfreeze:** `conv1`, `conv2` (low-level geometry, zero open-vocab benefit).  
**Never do this on mini** (323 scenes cannot support 8M encoder params).

---

## 8. Ablation Table to Fill

All entries use **B12/N4 split** (novel = bicycle, bus, motorcycle, vegetation).

| Experiment | PQ | PQ_B | PQ_NTh | PQ_NSt | mIoU | What it tests | Config |
|---|---|---|---|---|---|---|---|
| **Paper: PFC baseline** | 54.8 | — | 37.3 | 0.5 | 55.5 | — | — |
| **Paper: Xiao et al.** | **62.0** | — | **49.6** | **35.2** | **60.1** | — | — |
| Our closed-set (all 16, existing) | 47.59 | — | ~0 | ~0 | 43.11 | Frozen baseline | `full_nuscenes_token_v2.yaml` |
| Our closed-set B12 eval (existing ckpt) | — | — | 0 | 0 | — | Confirm 0 novel | run `eval_b12n4.py` |
| Decoder fine-tune only, mini | — | — | — | — | — | Gate 1 stability | `mini_e2e_encoder.yaml` |
| Open-vocab decoder fine-tune, mini | — | — | — | — | — | Gate 2 smoke test | `mini_openvocab_e2e.yaml` |
| + base suppression threshold τ | — | — | — | — | — | Inference-only, no retrain | τ tuned on mini val |
| + background CLIP anchor L_bg | — | — | — | — | — | Unmatched query supervision | extend `open_vocab_loss.py` |
| + density-stratified L_V | — | — | — | — | — | Sparse novel thing coverage | extend `open_vocab_loss.py` |
| + range-gated attention | — | — | — | — | — | Structured scene context | extend `context_encoder.py` |
| Open-vocab decoder fine-tune, full nuScenes | — | — | — | — | — | Gate 4 real eval | `full_nuscenes_openvocab_e2e.yaml` |
| + pseudo-label novel classes | — | — | — | — | — | Weak novel supervision | after Gate 2 ckpt exists |
| True E2E (conv4 unfreeze, external anchor) | — | — | — | — | — | Gate 5, conditional | external CLIP anchor required |

---

## 9. Key Hyperparameter Decisions (Paper + Our Adjustments)

| Parameter | Paper Value | Our Value | Reason |
|---|---|---|---|
| `w_cls` | 1.0 | 1.0 | Match paper. Focal loss on B12 only. |
| `w_mask` | 1.0 | 1.0 | Unchanged. |
| `w_object_distill` (L_O) | 1.0 | 1.0 | Match paper. |
| `w_voxel_distill` (L_V) | 0.1 | 0.1 | Low weight — larger values cause training instability. |
| `decoder_lr` | N/A (P3Former head) | 1e-4 | Fine-tunes decoder to produce region-coherent CLIP features. |
| `context_encoder_lr` | N/A | 5e-5 | SceneContextEncoder gets 2× lower LR than decoder. |
| `encoder_lr` | N/A (frozen in paper too) | **0 (frozen)** | Encoder stays frozen. No gradient. CLIP alignment preserved. |
| `unfreeze_after_epoch` | N/A | 3–5 | Decoder + query heads warm up first before context encoder activates. |
| `unfreeze_mode` | N/A | `decoder_only` | Safest. Encoder CLIP alignment fully preserved. |
| `temperature` | learnable | learnable | Classification temperature adapts. |
| `optimizer` | AdamW, wd=0.01, lr=8e-4 | AdamW, lr=1e-4 | Paper uses higher LR on P3Former (different backbone). |
| `epochs` | 40 | 50 (mini) / 30 (full) | More epochs on mini since data is small. |
| Training classes | B12 | B12 | Novel classes excluded from cls loss. |
| Inference classes | B12 + N4 = 16 | 16 | All 16 text embeds at test time. |

> **Important clarification:** `backbone_lr=5e-6` in older configs referred to the SceneContextEncoder, not the encoder conv stages. In updated configs this is renamed `context_encoder_lr` to avoid confusion.

---

## 10. Risk Registry

| Risk | Likelihood | Mitigation |
|---|---|---|
| L_V flat / not decreasing | Medium | Check `F_backbone` is detached from decoder graph; confirm encoder is frozen |
| Region-CLIP head collapse | Medium | Lower `w_object_distill`, check cosine sim distribution of `v_q` |
| CLIP alignment drift (encoder unfrozen accidentally) | Low | Verify `encoder_lr=0`; check `requires_grad=False` on conv1–conv4 in logs |
| OOM during training | Low-Medium | AMP enabled; `num_mask_pts=500`; `batch_size=1` |
| Open-vocab PQ lower than closed-set PQ | High (expected) | This is normal — novel inference is harder. L_V trend is the real metric at Gate 2. |
| Mini data too small for reliable PQ numbers | High | 323 train scenes gives noisy numbers. Gate 2 is a smoke test only. Gate 4 (full) gives real eval. |
| Gate 5 encoder unfreeze corrupts CLIP alignment | Medium | Monitor `cos(F_backbone, CLIP_text)` per epoch; re-freeze if it drops |

## 11. Do-Not-Do Rules

- **Do not** run full nuScenes before Gates 1 and 2 pass on mini.
- **Do not** set `decoder_lr > 5e-4` (will overwrite CLIP alignment propagated by L_V).
- **Do not** set any `encoder_lr > 0` during Gates 0–4.
- **Do not** unfreeze `conv1` or `conv2` — ever. Low-level geometry features, zero open-vocab benefit.
- **Do not** compare open-vocab PQ directly to closed-set PQ — different inference heads, different class count at test time.
- **Do not** increase `w_voxel_distill (L_V) > 0.5` — masks will degrade to serve CLIP reconstruction rather than segmentation.
- **Do not** proceed to Gate 5 (encoder unfreezing) if Gate 4 shows PQ_NTh > 20. It's working — don't risk breaking it.

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

---

## 13. Improvements On Top of Current Plan

These five additions are independently ablatable, all implementable on top of existing files, and **none are in Xiao et al.** — they form the paper's novel contribution table.

### 13.1 Base-Class Suppression Threshold τ ← Do This First (1 day, no retraining)

**Problem:** At test time `argmax` over 16 text embeds always favours base classes — they have stronger training signal. Novel classes never win the argmax. This is why `Token-OpenVocab` shows **0 novel detections** even though CLIP alignment exists.

**Fix (inference only, tune τ on mini val after Gate 2):**
```python
base_logits  = logits[:, BASE_12_IDX]   # [N_queries, 12]
novel_logits = logits[:, NOVEL_4_IDX]   # [N_queries, 4]

max_base_conf = base_logits.max(dim=-1).values
pred = torch.where(
    max_base_conf < τ,              # τ = 0.4, tune on mini val
    NOVEL_4_IDX[novel_logits.argmax(-1)],
    BASE_12_IDX[base_logits.argmax(-1)]
)
```
**File to change:** `panoptic/eval/eval_b12n4.py` — add `--threshold τ` argument.  
**Expected gain:** Immediate PQ_NTh > 0 without any retraining.

---

### 13.2 Background CLIP Anchor Loss L_bg (1 day, extend open_vocab_loss.py)

**Problem:** `L_O` only supervises **matched** queries (those with a GT instance). Unmatched queries — which will fire on novel classes at test time — get zero object-level supervision. Only `L_V` touches them, and only indirectly.

**Fix:** Pull unmatched queries toward the CLIP centroid of the voxels they attend to:
```python
# In open_vocab_loss.py, after matched query loss
unmatched_mask = ~matched_mask                         # [N_queries]
v_q_unmatched  = query_embeds[unmatched_mask]          # [N_unmatched, 768]
bg_anchor      = mask_pool(F_backbone, attn_weights[unmatched_mask])  # per-query centroid

L_bg = 1 - F.cosine_similarity(v_q_unmatched, bg_anchor.detach(), dim=-1).mean()
L_total += w_bg * L_bg   # w_bg = 0.3
```
No new architecture needed. Costs one extra mask-pool operation per batch.

---

### 13.3 Density-Stratified L_V Weighting (1 day, extend open_vocab_loss.py)

**Problem:** On full nuScenes, `L_V` is dominated by dense stuff regions (vegetation = 50k pts, driveable surface = 200k pts). Sparse novel things (bicycle = 80 pts, motorcycle = 120 pts) contribute near-zero gradient — exactly the novel classes we need to detect.

**Fix:** Weight each voxel's L_V contribution by inverse local point density:
```python
# In open_vocab_loss.py
density_weight = 1.0 / (local_point_count.float() + 1.0)   # [N_voxels]
density_weight = density_weight / density_weight.mean()      # normalise

L_V = (F.l1_loss(F_rec, F_backbone.detach(), reduction='none') 
       * density_weight.unsqueeze(-1)).mean()
```
`local_point_count` can be approximated from the SparseTensor coordinate neighbourhood within a fixed radius. Sparse novel things get 10–100× higher L_V weight than dense stuff.

---

### 13.4 Range-Gated Attention in SceneContextEncoder (3 days, extend context_encoder.py)

**Problem:** Standard MHSA in `SceneContextEncoder` attends to all voxels uniformly. For outdoor LiDAR scenes this is noisy — a bicycle 80m away is irrelevant context for a nearby car.

**Fix:** Two attention heads with different radial gates:
```python
# In context_encoder.py
pairwise_dist = torch.cdist(voxel_coords, voxel_coords)   # [N, N]

# Head group 1: local context (things — bicycle near car)
local_mask = pairwise_dist < R_local    # R_local = 10m
# Head group 2: scene context (stuff — vegetation, driveable surface)
scene_mask = pairwise_dist < R_scene    # R_scene = 50m

attn_weight_local = attn_weight_local.masked_fill(~local_mask, float('-inf'))
attn_weight_scene = attn_weight_scene.masked_fill(~scene_mask, float('-inf'))
```
Use 4 heads for local, 4 heads for scene. Directly motivated by LiDAR scene structure. Better gradient signal than uniform global attention.

---

### 13.5 Pseudo-Label Novel Classes (After Gate 2 Checkpoint Exists)

**Problem:** Novel classes (bicycle, bus, motorcycle, vegetation) get **zero mask supervision** during training. `L_V` is the only signal. On full nuScenes, this may be insufficient for sparse novel things.

**Fix:** Use the Gate 2 checkpoint to generate pseudo-labels for novel classes on the full nuScenes **training set**, then use them as weak supervision in Gate 4:
```bash
# Step 1: Generate pseudo-labels (after Gate 2 passes)
python panoptic/scripts/generate_pseudo_labels.py \
    --checkpoint panoptic/checkpoints_openvocab_mini/best.pth \
    --version v1.0 \                # full nuScenes train split
    --novel-only \                  # only output predictions for novel classes
    --confidence-threshold 0.6 \    # high threshold to keep quality
    --output /scratch/.../pseudo_labels_novel/

# Step 2: Gate 4 training uses pseudo-labels as weak supervision
# In config: pseudo_label_dir: /scratch/.../pseudo_labels_novel/
# In loss: novel pseudo-label instances get weight=0.3 (vs 1.0 for real GT)
```
This breaks the "zero novel supervision" constraint without human labels. Not in Xiao et al.

---

## 14. True E2E Encoder Training Plan (Gates 5+)

> This section explains what real end-to-end encoder training requires — why the current plan cannot do it, and what the exact steps are to enable it.

### 14.1 Why You Cannot Do True E2E Encoder Training Yet

The current plan uses `F_backbone` (encoder output) as the distillation target for both `L_O` and `L_V`. This creates a **self-referential loop** if you unfreeze the encoder:

```
If encoder is unfrozen:
  F_backbone changes every step
  L_V = L1(F_rec, F_backbone)  ←  target is moving
  L_O: w_q = mask_pool(F_backbone)  ←  supervision signal is moving
  → both losses become meaningless → collapse
```

The encoder **must** stay frozen as long as it serves as its own distillation target.

### 14.2 The Missing Piece: Precomputed External CLIP Anchors

To unfreeze the encoder, replace the moving `F_backbone` with **offline-precomputed OpenSeg features** as the fixed distillation target:

```bash
# Run ONCE on full nuScenes (prerequisite for Gate 5)
python tools/precompute_openseg_features.py \
    --version v1.0 \
    --output /scratch/.../openseg_anchors/ \
    --model openseg_vitl16_448
# Output: {scene_token}_{frame_token}.npy, shape [N_voxels, 768]
# Storage estimate: ~50 GB for full nuScenes
```

Once these exist, `L_V` and `L_O` targets become fixed regardless of encoder state:
```python
F_clip_frozen = load_precomputed(frame_id)     # never changes
L_V = L1(F_rec, F_clip_frozen)                # stable target
w_q = mask_pool(F_clip_frozen, mask)          # stable supervision
L_O = 1 - cos(v_q, w_q)                      # meaningful even as encoder trains
```

### 14.3 Staged Encoder Unfreezing (Gate 5 Procedure)

```
Prerequisites: Gate 4 converged + PQ_NTh < 20 + external CLIP anchors precomputed

Epoch 0–N    : Load Gate 4 best.pth. Switch L_V/L_O targets to F_clip_frozen.
               Keep encoder frozen. Run 5 epochs to confirm loss is stable with new target.

Epoch N+1–10 : Unfreeze conv4 only. lr_conv4 = 5e-7.
               Monitor: cos(F_backbone_conv4_out, F_clip_frozen) per epoch.
               Accept if: PQ_NTh improves after 5 epochs.
               Abort if: cosine similarity drops > 5% → re-freeze conv4.

Epoch 11–20  : [If conv4 stable] Unfreeze conv3. lr_conv3 = 1e-7.
               Same monitoring. More conservative — conv3 is further from CLIP space.

Never unfreeze: conv1, conv2  ← low-level geometry, no semantic content to adapt.
```

Add encoder alignment loss to keep encoder honest during fine-tuning:
```python
L_enc = 1 - F.cosine_similarity(F_backbone_current, F_clip_frozen, dim=-1).mean()
L_total = L_cls + L_mask + L_O + L_V + w_enc * L_enc   # w_enc = 0.5
```

### 14.4 What True E2E Encoder Training Gains

| Component | Frozen encoder | E2E encoder (Gates 5+) |
|---|---|---|
| Per-voxel CLIP quality | Fixed at pre-distil level | Adapts to panoptic task |
| Novel class detection | Relies entirely on L_V | Encoder actively adapts to region boundaries |
| Instance boundary sharpness | Decoder must do all the work | Encoder learns instance-coherent features |
| L_O quality | Noisy mask-pool from per-voxel feats | Cleaner — encoder feats cluster within instances |
| Risk | Low | Medium — needs external CLIP anchor |
| Data requirement | 323 mini scenes ok | Full 28k frames mandatory |

### 14.5 Implementation Order for True E2E

```
Priority 1  Precompute OpenSeg anchors (full nuScenes)     ← unblocks everything
Priority 2  Swap L_V/L_O targets to F_clip_frozen          ← 1 day code change
Priority 3  Gate 4 convergence                             ← current plan
Priority 4  conv4 unfreeze + L_enc monitoring              ← Gate 5
Priority 5  conv3 unfreeze if Gate 5 is stable             ← Gate 6
```

---
