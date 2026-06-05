# Model Architecture: Dense Camera-CLIP Supervision (Job 1)

**Purpose:** Train the panoptic segmentation model with dense CLIP supervision to achieve open-vocabulary 3D panoptic segmentation on nuScenes.
**Primary setup:** B12/N4 (12 base classes for training, 4 novel classes evaluated via CLIP at inference).
**Secondary ablation:** All-16 classes (closed-set baseline comparison with SAL/MaskPLS).
**Mathematical formulation:** See `model_math_dense_clip.md` for full notation.

---

## 1. Why This Architecture Exists

### The Problem (Confirmed by Strategies E, K, L, M)

The frozen DisNet backbone (MinkUNet18A, distilled from OpenSeg) produces per-voxel 768-d features with **noise-level cosine margin** (~0.02) between similar classes. This means:
- cos_sim(car_voxel, "car") ≈ cos_sim(car_voxel, "police car") ± 0.02
- No inference-only strategy can reliably distinguish novel concepts

**Root cause:** The distillation chain `CLIP text → OpenSeg pixel → multi-view projection → DisNet 3D` degrades alignment at each step. By the time features reach the 3D backbone, fine-grained CLIP-space structure is destroyed.

### The Solution

Train the model with multiple CLIP-space losses that:
1. **Sharpen per-voxel feature margins** (L7 — root cause fix via LoRA adapters)
2. **Align decoder outputs to camera CLIP features** (L_O object-level, L_V voxel-level)
3. **Expand the contrastive denominator** from 16 to 216+ classes (L5/C1)
4. **Handle ambiguous stuff classes** with context-aware targets (CASD)

---

## 2. Architecture Overview

```
Input: Sparse 3D point cloud (V voxels, 3 channels: XYZ)
                    │
                    ▼
┌──────────────────────────────────────────┐
│  Frozen DisNet Backbone (MinkUNet18A)    │
│  15.6M params, frozen                    │
│  Output: SparseTensor (V, 768)           │
└──────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────┐
│  LoRA Sparse Conv Adapters (B1)          │
│  Inserted after: block5-8, final         │
│  ~50K trainable params (rank=16)         │
│  Standard nn.Linear — bypasses ME backward│
└──────────────────────────────────────────┘
                    │
                    ▼  backbone_feats (V, 768)
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌───────────────┐     ┌─────────────────────┐
│ Panoptic      │     │ L7: VoxelDecomp     │
│ Decoder       │     │ (per-voxel text     │
│ (3 blocks,    │     │  contrastive)       │
│  Q=106 queries│     │ Uses backbone_feats │
│  mask head)   │     │ directly            │
└───────────────┘     └─────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│ SingleCLIPEmbeddingHead (A4)              │
│ MLP: 256 → 512 → 512 → 768, L2-norm     │
│ Learnable temperature τ                   │
│ Replaces dual linear+token head           │
└───────────────────────────────────────────┘
        │
        ▼  pred_embed (B, Q, 768) + pred_masks (B, V, Q)
        │
        ├──→ Hungarian Matching (mask+dice cost only, no CE)
        │         │
        │         ▼  matched_indices
        │
        ├──→ L_mask + L_dice (standard mask losses)
        ├──→ L_obj (binary objectness BCE)
        ├──→ L5/C1: TokenLoss with 216+ negatives
        ├──→ L_O: Object-level distillation (matched queries → camera CLIP)
        └──→ L_V + CASD: Voxel-level distillation (reconstructed features → camera CLIP)
```

---

## 3. Component Details

### 3.1 Frozen Backbone + LoRA Adapters (B1)

**Backbone:** MinkUNet18A (PLANES = 32, 64, 128, 256, 128, 128, 96, 96)
**Frozen:** All 15.6M params have `requires_grad=False`
**LoRA adapters:** Inserted via forward hooks after backbone decoder blocks:

| Layer | Channels | LoRA rank | Trainable params |
|---|---|---|---|
| net3d.block5 | 128 | 16 | 4,224 |
| net3d.block6 | 128 | 16 | 4,224 |
| net3d.block7 | 96 | 16 | 3,168 |
| net3d.block8 | 96 | 16 | 3,168 |
| net3d.final | 768 | 16 | 24,960 |
| **Total** | | | **~41K** |

Each adapter: `feats → down(Linear C→16) → ReLU → Dropout(0.05) → up(Linear 16→C) → feats + delta`

**Design intuition:**
- Standard `nn.Linear` layers — bypasses MinkowskiEngine backward entirely (no hash map issues)
- Operates on dense `.F` tensor extracted from SparseTensor, returns new SparseTensor with same coordinates
- Zero-initialized `up.weight` ensures **exact identity** at epoch 0 (no disruption to pretrained features)
- No LayerNorm — removed after discovering it re-scaled backbone features even at epoch 0, breaking warm-start compatibility with existing decoder
- Dropout(0.05) prevents adapter overfitting on small feature subsets

**Why LoRA not full unfreeze:** ME gradient check confirmed backward through sparse conv is safe, but LoRA is preferred because:
1. Only ~41K params vs 15.6M (much faster convergence)
2. No risk of catastrophic forgetting in encoder blocks
3. Lower GPU memory (no optimizer states for frozen params)
4. All losses that backprop through decoder also reach LoRA (not just L7)

**Gradient flow:** All losses (mask, dice, token C1, L_O, L_V) flow through the decoder → backbone_proj → LoRA adapters. L7 additionally provides a direct per-voxel path.

### 3.2 Single CLIP Embedding Head (A4)

Replaces the dual-head architecture (linear classification + CLIPTokenHead):

```
Old: decoder_output → Linear(256, 17) → class logits (closed-set)
     decoder_output → MLP(256→768) → CLIP embeddings (open-vocab)

New: decoder_output → MLP(256→512→512→768) → L2-norm → CLIP embeddings
     decoder_output → Linear(256, 1) → objectness score (sigmoid)
```

**Classification at inference:** `argmax(pred_embed @ text_protos.T / τ)`
**Objectness:** Binary head trained with BCE against has-GT-match.

**Tradeoff:** −0.008 PQ vs linear head (confirmed Phase 5). Required for true open-vocab — linear head creates closed-set shortcut.

### 3.3 Panoptic Decoder (Unchanged)

- 3 cross-attention blocks, 8 heads, FFN dim 1024
- Q = 100 thing queries + 6 stuff queries = 106
- Mask head: dot product between query features and backbone voxel features
- Hungarian matching uses mask + dice cost ONLY (no CE cost with single CLIP head)

### 3.4 Camera CLIP Feature Infrastructure

**Model:** CLIP ViT-L/14@224 (OpenAI `clip` package) — same embedding space as text prototypes
**Extraction:** Strategy B — Dense single-scale with 50% overlap (stride=112, tile=224)
- Each 1600×900 camera image → ~84 overlapping 224×224 tiles
- Each tile → CLIP patch tokens (256 patches, 768-d each)
- Tiles stitched with 2-4× overlap averaging → smoother, higher-quality features
- 6 cameras × 84 tiles = ~504 CLIP forward passes per sample

**Precomputed output:** `{sample_token}.npz` per sample containing:
- `obj_feats`: (N_inst, 768) float16 — mask-pooled CLIP per GT instance (for L_O)
- `vox_feats`: (4096, 768) float16 — randomly sampled voxel CLIP features (for L_V)
- `vox_indices`: (4096,) int32 — voxel indices into point cloud
- Total storage: **186 GB** (fp16) for trainval

**At training time:** Load precomputed features per sample, project to matched queries (L_O) or selected voxels (L_V).

---

## 4. Loss Functions

### 4.1 Standard Losses (from baseline)

| Loss | Weight | Purpose |
|---|---|---|
| L_mask (BCE) | 5.0 | Binary mask quality per query |
| L_dice | 5.0 | Mask overlap quality |
| L_obj (BCE) | 2.0 | Objectness: matched=1, unmatched=0 |

### 4.2 L5/C1: Token Contrastive with 216+ Negatives (v2)

**Problem solved:** The old 16-class InfoNCE denominator caused closed-set collapse — the token head learned 16 narrow tunnels in CLIP space, making novel-class detection impossible.

**Fix:** Expand denominator to 216+ concepts:
- 16 class prototypes (all 16 in denominator as soft negatives; only 12 base used as targets)
- 200 negatives: driving confusors (van, SUV, pickup), infrastructure (fire hydrant, bollard), generic objects

**v2 improvements (post Job 1a analysis):**
- **Temperature clamp:** `min=1.0, max=5.0` — prevents saturation at 14+ where gradients vanish
- **Margin penalty (0.1):** Subtracts margin from GT similarity before softmax: `logits[gt] -= margin`. Forces minimum cosine separation > margin between GT and confusers
- **Soft negatives:** Novel protos (bus, pedestrian, manmade, vegetation) remain in denominator but are never targets. Pushes embeddings away from novel clusters without rewarding them

**Temperature:** Learnable, initialized at 1/0.07 ≈ 14.3, clamped to [1.0, 5.0]. Expected to converge to ~3-5 range for optimal discrimination.

### 4.3 L_O: Object-Level Camera CLIP Distillation

**Intuition:** Each matched decoder query represents one object/stuff segment. Its CLIP embedding should match what CLIP sees for that object in the camera images.

**Computation:**
1. For each matched (query, GT mask) pair
2. Soft-pool camera CLIP features within the predicted mask: `w_q = softmax(mask) @ fv_clip`
3. Loss = `1 - cos_sim(pred_embed_q, w_q)`

**Why it works:** Forces the decoder to produce CLIP-aligned query embeddings that reflect what the actual CLIP model sees for each object. This is the standard 3D2D distillation approach at the object level.

**Applied to:** Matched queries only (unmatched → noisy random pooling).
**Warmup:** 0→1.0 over 2 epochs (phase 2 only).

### 4.4 L_V: Voxel-Level Camera CLIP Distillation

**Intuition:** Reconstruct per-voxel CLIP features from ALL query embeddings. If the mask prediction assigns voxel v to query q, then the reconstructed feature for v should match the camera CLIP feature at v.

**Computation:**
1. Reconstruct: `F_rec = M_Q^T @ F_Q` where M_Q = sigmoid(mask logits), F_Q = query embeddings
2. Loss = `L1(F_rec[covered], fv_clip[covered])`

**Why it works:** Provides dense voxel-level supervision. Even unmatched queries contribute — they must reconstruct the CLIP feature field correctly across the entire scene.

**Weight:** 0.1 (deliberately small — noisy early masks create noisy F_rec).
**Warmup:** 0→0.1 over 5 epochs.

### 4.5 CASD: Context-Aware Stuff Distillation

**Problem:** Stuff classes like "sidewalk" and "other_flat" are locally ambiguous — a patch of flat gray could be anything. CLIP distinguishes them by context (adjacent buildings = sidewalk, adjacent road = parking lot).

**Fix:** For 3 ambiguous stuff classes, replace L_V targets with context-expanded crop CLS tokens:
- other_flat (idx=11): 3.0x bbox expansion
- sidewalk (idx=12): 2.5x expansion
- terrain (idx=13): 2.0x expansion

**Online computation (no precompute):**
1. Find voxels of CASD class in GT
2. Project to best camera → compute bbox of visible region
3. Expand bbox by class-specific scale factor
4. Crop camera image → resize to 224x224 → CLIP forward → CLS token
5. Broadcast CLS token to all voxels of that class

**Cost:** ~3 CLIP forward passes per sample (one per CASD class present). Adds ~0.1s/sample.

### 4.6 L7: Voxel Decomposition Loss (Root Cause Fix)

**Problem:** Backbone features lack inter-class margin. This is THE root cause of inference failure.

**Fix:** Directly train backbone features (via LoRA) against 216+ text prototypes using per-voxel InfoNCE.

**Computation:**
1. Take backbone_feats for valid voxels (subsample 10K for memory)
2. Compute cosine similarity to all 216 text concepts / τ
3. Cross-entropy loss with GT class as target (indexed into first 16 positions)

**Why it works:** Every "car" voxel must score higher for "car" than for "van", "SUV", "police car", etc. This directly creates the inter-class margin that was missing.

**Gradient path:** L7 → backbone_feats → LoRA adapters (the ONLY trainable params in the backbone path). Without LoRA, this loss has no gradient destination.

**Weight:** 0→0.5 over 5 epochs (ramped to prevent destabilizing early training).

### 4.7 T4: Multi-Prompt Template Augmentation

During training, randomly sample from 10 templates per class for the text prototypes used in L5/C1 and L7:
```
"a {class} in a scene", "a photo of a {class}", "a {class} on a road",
"a {class} in an urban environment", "a {class} in a driving scene", ...
```

**Why it helps:** Prevents overfitting to a single text embedding per class. CLIP's text space has many valid phrasings — training against all of them creates more robust alignment.

---

## 5. Training Strategy

### 5.1 Three-Job Pipeline

```
Job 1a (20 epochs, ~12h) → Job 1b (15 epochs, ~20h) → Job 1c (10 epochs, ~14h)
```

| Job | Components | What it trains | Primary target |
|---|---|---|---|
| 1a | LoRA + A4 + C1 (margin+soft neg) + T4 | Foundation: mask decoder → CLIP space | PQ_base, margin > 0 |
| 1b | + L_O + L_V (camera distillation) | Embed alignment to real CLIP | SQ improvement, begin novel |
| 1c | + CASD + L7 (voxel InfoNCE) | Per-voxel margin, stuff disambiguation | PQ_novel > 0 |

Each job warm-starts from the previous job's best checkpoint.

### 5.2 Two-Phase Training (Within Each Job)

**Phase 1** (first 2 epochs): Only LoRA + new heads train. Decoder base frozen. Establishes stable gradient flow without destabilizing pretrained mask quality.
**Phase 2** (remaining epochs): All components train with differential LRs. Distillation losses activate with warmup schedules.

### 5.3 Differential Learning Rates

| Parameter group | LR | Rationale |
|---|---|---|
| Decoder (existing) | 5e-5 | Lower — preserve mask quality |
| SingleCLIPEmbeddingHead | 1e-4 | Higher — new architecture |
| LoRA adapters | 1e-4 | Higher — starting from zero |
| token_criterion.log_temperature | 1e-4 | Shared with head group |

### 5.4 B12/N4 Class Split

**Base 12 (training classes):** barrier, bicycle, car, construction_vehicle, motorcycle, traffic_cone, trailer, truck, driveable_surface, other_flat, sidewalk, terrain
**Novel 4 (held-out, CLIP-only at inference):** bus(2), pedestrian(6), vegetation(15), manmade(14)

**Critical:** Novel class GT masks are **excluded** from all losses (mask, dice, token). Novel text protos remain in the C1 denominator as soft negatives. PQ_novel can only improve through genuine CLIP-space generalization.

---

## 6. Bug Fixes and Design Revisions (v1 → v2)

| Issue | Root Cause | Fix |
|---|---|---|
| Temperature stuck at 14.3 (clamped to 5.0) | Init at log(1/0.07)=2.66 → exp=14.3, above clamp, zero gradient | Init at log(3.0)=1.1 → exp=3.0, inside clamp range |
| LoRA gnorm = 914K (appeared unstable) | Read AMP-scaled gradients before unscale | Moved computation after `scaler.unscale_()`. True norm ~0.2 |
| PQ_novel=0.47 at epoch 1 (impossibly high) | Novel classes had full mask/dice supervision | Filter NOVEL_IDS_TRAIN={2,6,14,15} from targets |
| Voxel margin = -0.03 after Job 1a v1 | Temperature saturation → no gradient pressure for margins | Margin penalty (0.1) + temp clamp [1,5] + soft negatives |
| Two independent temperatures (train vs inference) | decoder.clip_embed_head.temperature trained by nothing | Inference now uses token_criterion.temperature |
| LayerNorm in LoRA adapter | Re-scaled features at epoch 0, breaking warm-start | Removed entirely — just feats + delta |
| Matcher KeyError in single_clip mode | aux_outputs lack pred_logits | Skip class cost when cost_class=0 |

---

## 7. Job 1b Design: Camera CLIP Distillation

### 7.1 Overview

Job 1b adds two distillation losses that align the model's CLIP embeddings with what the actual CLIP model sees in camera images:

```
Job 1a checkpoint (mask quality + C1 alignment)
    │
    ├── L_O: Object-level (query embed → mask-pooled camera CLIP)
    └── L_V: Voxel-level (reconstructed features → per-voxel camera CLIP)
```

### 7.2 Data Pipeline

Precomputed features loaded per sample from `/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features/v1.0-trainval/`:
- `obj_feats` (N_inst, 768) → used by L_O for each matched query
- `vox_feats` (4096, 768) + `vox_indices` (4096,) → used by L_V for sampled voxels

### 7.3 Why Camera Distillation Works

Text protos are single-point summaries ("a car in a scene"). Camera CLIP features capture the ACTUAL visual appearance of each specific object — color, size, context, occlusion state. By distilling from camera features:
1. Query embeddings learn object-specific CLIP representations (not just class-average)
2. The model implicitly learns what CLIP "looks at" for novel objects
3. Even without novel class labels, camera CLIP features for novel objects push embeddings toward novel proto regions

### 7.4 Warmup Schedule

- L_O: weight ramps 0 → 1.0 over 2 epochs
- L_V: weight ramps 0 → 0.1 over 5 epochs (kept small — noisy masks create noisy reconstructions early)

---

## 8. Current Progress and Results

### 8.1 Job 1a v2 (In Progress)

**Status:** Running on full trainval, 20 epochs
**Warm-start:** Closed-set best checkpoint (PQ=0.476)
**Key changes from v1:** Margin penalty, soft negatives, temperature init fix

Mini smoke test results (3 epochs, 404 samples):
```
Epoch 0: loss=90→52, lora_gnorm=0.2, token_loss=4.0
Epoch 2: PQ_base=0.10, PQ_novel=0.00, token_loss=3.8
Zero-shot: ambulance=7.3%, police_van=12.2%, trash_can=0%, fire_hydrant=0%
```

### 8.2 Precompute (Complete)

**Status:** ✅ Done — 34,149 npz files, 186 GB, all verified
**Strategy:** Dense single-scale, stride=112, 50% overlap, fp16

### 8.3 Expected Full Trainval Results

| Metric | Job 1a (expected) | Job 1b (expected) | Job 1c (expected) |
|---|---|---|---|
| PQ_all | 0.35-0.45 | 0.40-0.50 | 0.45-0.55 |
| PQ_base | 0.40-0.48 | 0.45-0.52 | 0.45-0.52 |
| PQ_novel | 0.00 | 0.00-0.10 | 0.10-0.25 |
| SQ | 0.55-0.65 | 0.65-0.75 | 0.65-0.75 |
| Voxel margin | >0.05 | >0.08 | >0.12 |
| Temperature | 3.0-5.0 | 3.0-5.0 | 3.0-5.0 |

---

## 6. Inference Strategy

### 6.1 Closed-Set Inference (All 16 Classes)

Same as baseline: `class = argmax(pred_embed @ text_protos_16.T / τ)` for each query.
Panoptic assignment via greedy merge (overlap threshold, min points filter).

### 6.2 Open-Vocabulary Inference (B12/N4 + Novel-5)

1. Load B12/N4 checkpoint
2. Construct text prototype matrix: base 12 + novel 4 + 5 additional novel concepts (police car, ambulance, van, fire hydrant, dog)
3. Classify: `class = argmax(pred_embed @ text_protos_21.T / τ)`
4. Panoptic assignment as usual

**Expected:** Novel-4 things PQ 25-45, novel stuff PQ 10-25 (after full pipeline).

### 6.3 Voxel Margin Diagnostic

Post-training validation: compute per-voxel cosine similarity margins.
- For car voxels: margin = cos_sim("car") - cos_sim("police car")
- Target: margin > 0.05 (was ~0.02 before training)

---

## 7. Prerequisites and Execution Plan

### P0: Environment Setup

```bash
# Install open_clip
pip install open_clip_torch

# Verify
python -c "import open_clip; print(open_clip.__version__)"
python -c "import torch, MinkowskiEngine as ME; print('OK')"
```

### P1: CLIP Model Verification

Run `panoptic/scripts/verify_clip_model.py`:
- Confirm CLS shape (1, 768)
- Confirm patch token extraction (257 tokens, strip CLS → 256)
- Record GPU memory (~1.5 GB)

### P2: Camera Projection Verification

Run `panoptic/scripts/verify_camera_projection.py`:
- Confirm 35-60% LiDAR points visible per camera
- Confirm positive depths, valid UV coords

### P3: Generate Negative Bank

Run `panoptic/scripts/generate_negative_bank.py`:
- Produces `panoptic/clip_prototypes/negative_bank_200.pt` (200, 768)

### P4: Precompute Camera CLIP Cache

Submit batch job (~8 hours):
```bash
python panoptic/scripts/precompute_camera_clip_features.py \
    --dataroot /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes \
    --version v1.0-trainval \
    --output_dir /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/camera_clip_cache/vitl14_224 \
    --model ViT-L-14 --pretrained openai --split all
```

### P5: Mini Smoke Test (Per Job)

Before each full training run, 3 epochs on v1.0-mini:
- All losses must be finite
- LoRA grad norms in [1e-4, 10]
- GPU peak < 40 GB
- No shape errors

---

## 8. Execution Timeline

```
DAY 0-1: Pre-flight
├── P0: pip install open_clip, verify env
├── P1: verify CLIP model
├── P2: verify camera projection
├── P3: generate negative bank
├── Smoke test Job 1a on mini (3 epochs, ~20 min)
└── On PASS → submit:
    ├── Batch A: CLIP precompute (8h, parallel with 1a)
    └── Batch B: Job 1a full (25h)

DAY 1-2: Code Jobs 1b and 1c while 1a runs

DAY 2-3: Job 1a finishes
├── Evaluate: PQ ≥ 0.47?
├── Smoke test Job 1b on mini
└── Submit Job 1b (15h, warm-start from 1a)

DAY 3-4: Job 1b finishes
├── Evaluate: B12/N4 novel things PQ > 0?
├── Smoke test Job 1c on mini
└── Submit Job 1c (10h, warm-start from 1b)

DAY 4-5: Job 1c finishes
├── Final evaluation (all metrics)
├── Voxel margin diagnostic
└── Ablation table complete
```

---

## 9. Expected Results

| Setup | PQ | PQ_novel_things | PQ_novel_stuff | Key change |
|---|---|---|---|---|
| Baseline (PQ=0.476) | 0.476 | ~0 | ~0 | — |
| + LoRA + L5 + A4 (Job 1a) | 0.48-0.50 | 3-5 | ~0 | Foundation |
| + L_O + L_V (Job 1b) | 0.52-0.55 | 30-45 | 10-20 | Camera distillation |
| + CASD + L7 (Job 1c) | 0.54-0.58 | 35-50 | 20-35 | Novel contributions |

---

## 10. Failure Modes

| Symptom | Likely cause | Fix |
|---|---|---|
| PQ < 0.46 after Job 1a | A4 head change destabilized masks | Increase phase1_epochs to 4 |
| loss_L_O stuck > 0.5 | Camera coverage too low | Check projection code, lower coverage threshold |
| loss_L7 not decreasing | LoRA grads vanishing | Check hook wiring, increase LoRA rank to 32 |
| NaN in loss_L_V | Near-zero masks → degenerate F_rec | Increase L_V warmup to 8 epochs |
| Novel PQ still ~0 after 1b | CLIP alignment not improving | Check text proto model matches camera model (both ViT-L/14 openai) |

---

## 11. Files Created/Modified

### New files
```
panoptic/scripts/verify_clip_model.py
panoptic/scripts/verify_camera_projection.py
panoptic/scripts/precompute_camera_clip_features.py
panoptic/scripts/generate_negative_bank.py
panoptic/scripts/check_voxel_margin.py
panoptic/models/lora_backbone.py
panoptic/losses/distillation_loss.py
panoptic/utils/camera_clip_loader.py
panoptic/utils/casd_crop_extractor.py
panoptic/train/train_panoptic_job1.py
panoptic/configs/job1_dense_clip.yaml
panoptic/configs/job1_dense_clip_mini.yaml
panoptic/scripts/train_job1_dense_clip.sh
```

### Modified files
```
panoptic/models/mask_decoder.py  ← SingleCLIPEmbeddingHead + objectness_head
panoptic/losses/token_loss.py   ← TokenLossC1 (216+ negatives)
panoptic/losses/panoptic_loss.py ← integrate new losses
```

### Unchanged
```
panoptic/dataset/nuscenes_panoptic.py  ← no changes
panoptic/models/disnet.py              ← backbone stays frozen
models/mink_unet.py                    ← no changes
```

---

## Optional Improvements (Post Job 1a v1 Observations)

### ~~Soft Negatives in C1 Loss~~ (IMPLEMENTED in v2)

**Status:** ✅ Done — integrated into `token_loss_c1.py` along with margin penalty and temperature clamping.

### Temperature Warmup (Optional)

If temperature still saturates at max=5.0 too quickly, consider:
- Start at temp=1.0 for first 5 epochs (linear warmup), then allow learning
- This ensures early gradients are strong enough to establish margins

### Negative Mining (Optional)

Replace static 200-negative bank with dynamic hard negatives:
- Each step, find top-5 confuser concepts per class
- Update negative bank every N epochs
- Expected benefit: harder negatives → sharper boundaries

