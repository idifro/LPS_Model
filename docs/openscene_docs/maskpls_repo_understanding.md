# MaskPLS Repository Understanding

> **Paper**: "Mask-Based Panoptic LiDAR Segmentation for Autonomous Driving" (Marcuzzi et al., RA-L 2023)
> **Repo**: `/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/third_party/MaskPLS`
> **Origin**: Adapted from Facebook's Mask2Former for 3D LiDAR with MinkowskiEngine sparse convolutions.

---

## 1. Complete File Map

| File | Purpose | Pick for us? |
|---|---|---|
| `mask_pls/models/mask_model.py` | **Main LightningModule**: forward, loss, panoptic inference, optimizer | **REFERENCE** — study forward/inference logic, reimplement without Lightning |
| `mask_pls/models/mink.py` | **Backbone**: MinkUNet encoder-decoder (5-stage) | **NO** — we use frozen OpenScene DisNet instead |
| `mask_pls/models/decoder.py` | **Masked Transformer Decoder** (Mask2Former-style) | **YES** — core component to adapt |
| `mask_pls/models/blocks.py` | Transformer blocks: SelfAttn, CrossAttn, FFN, MLP | **YES** — reuse directly |
| `mask_pls/models/loss.py` | MaskLoss (CE+Dice+BCE) and SemLoss (CE+Lovász) | **YES** — adapt MaskLoss; skip SemLoss (we don't have semantic head) |
| `mask_pls/models/matcher.py` | Hungarian matcher (scipy `linear_sum_assignment`) | **YES** — reuse directly |
| `mask_pls/models/positional_encoder.py` | 3D sinusoidal positional encoding | **YES** — reuse directly |
| `mask_pls/datasets/semantic_dataset.py` | SemanticKITTI/nuScenes dataloader + augmentations + mask construction | **REFERENCE** — we already have our own `NuScenesPanoptic`; study mask construction patterns |
| `mask_pls/datasets/semantic-kitti.yaml` | KITTI label map (20 classes), splits, color map | **NO** — we use nuScenes |
| `mask_pls/datasets/semantic-nuscenes.yaml` | nuScenes label map (17 classes), splits | **REFERENCE** — compare class mapping with ours |
| `mask_pls/scripts/train_model.py` | Training entry (PyTorch Lightning Trainer) | **REFERENCE** — study callbacks/logging; reimplement in plain PyTorch |
| `mask_pls/scripts/evaluate_model.py` | Evaluation/test entry | **REFERENCE** |
| `mask_pls/utils/eval_np.py` | PanopticEval: PQ/SQ/RQ computation (numpy) | **YES** — reuse for PQ evaluation |
| `mask_pls/utils/evaluate_panoptic.py` | PanopticEvaluator wrapper | **YES** — reuse or adapt |
| `mask_pls/utils/interpolate.py` | KNN upsampling via pykeops (voxel→point) | **MAYBE** — useful if we need point-level predictions; pykeops is an extra dep |
| `mask_pls/utils/misc.py` | Distributed utils, `pad_stack`, `sample_points` | **YES** — `pad_stack` and `sample_points` are useful |
| `mask_pls/utils/testing.py` | Save results (.label / .npz format) | **REFERENCE** |
| `mask_pls/config/model.yaml` | Training hyperparams, loss weights | **REFERENCE** — use as defaults |
| `mask_pls/config/backbone.yaml` | Backbone channel dimensions | **NO** — our backbone is different |
| `mask_pls/config/decoder.yaml` | Decoder config (hidden_dim, heads, queries) | **YES** — use as starting hyperparams |
| `requirements.txt` | Dependencies | **REFERENCE** |
| `setup.py` | Package setup | **NO** |

---

## 2. Architecture Summary

### 2.1 Overall Flow

```
LiDAR points (N, 4: xyz+intensity)
    │
    ▼
┌───────────────────────────┐
│  Backbone: MinkUNet        │  ← we replace this with frozen OpenScene DisNet
│  Input: SparseTensor(4-d)  │
│  Output: 4 feature levels  │
│    y1 (256-d), y2 (128-d)  │
│    y3 (96-d),  y4 (96-d)   │
│  + semantic logits (96→20) │
└───────────┬───────────────┘
            │
    ▼ KNN interpolate voxel→point ▼
            │
┌───────────────────────────────────┐
│  Transformer Decoder (Mask2Former) │
│  3 blocks × 3 levels = 9 layers   │
│  100 learnable queries             │
│  Per layer:                        │
│    1. Cross-attn (queries ← points)│
│    2. Self-attn (queries ← queries)│
│    3. FFN                          │
│  Prediction heads per layer:       │
│    - class_embed: Linear(256→C+1)  │
│    - mask_embed:  MLP(256→256)     │
│    - mask = einsum(mask_embed, pts) │
└───────────┬───────────────────────┘
            │
            ▼
┌────────────────────────────┐
│  Panoptic Inference         │
│  1. Filter no-object queries│
│  2. score × mask → assignment│
│  3. Overlap threshold (0.8) │
│  4. Merge stuff segments    │
│  5. Output: sem + inst IDs  │
└────────────────────────────┘
```

### 2.2 Backbone: `MinkEncoderDecoder` (mink.py) — NOT USED BY US

5-stage sparse U-Net with MinkowskiEngine:

**Encoder** (each: strided conv + 2× ResidualBlock):
```
Stem:   4 → 32 → 32
Stage1: 32 → 32   (stride 2)
Stage2: 32 → 64   (stride 2)
Stage3: 64 → 128  (stride 2)
Stage4: 128 → 256 (stride 2)
```

**Decoder** (each: transposed conv + skip concat + 2× ResidualBlock):
```
up1: 256 → 256, cat(128) → 256
up2: 256 → 128, cat(64)  → 128
up3: 128 → 96,  cat(32)  → 96
up4: 96  → 96,  cat(32)  → 96
```

Produces 4 feature levels (y1=256-d, y2=128-d, y3=96-d, y4=96-d).
Semantic head: `Linear(96 → NUM_CLASSES)` on finest level.

**Why we don't use this:** We replace it with frozen OpenScene DisNet (768-d output). Our adapter will project 768→256 to match the decoder's expected hidden_dim.

### 2.3 Transformer Decoder: `MaskedTransformerDecoder` (decoder.py) — CORE COMPONENT

Adapted from Mask2Former for point clouds.

**Key dimensions:**
| Param | Value |
|---|---|
| `hidden_dim` | 256 |
| `nheads` | 8 |
| `dim_feedforward` | 1024 |
| `dec_layers` | 9 (3 blocks × 3 feature levels) |
| `num_queries` | 100 |
| Feature levels used | 3 (y1, y2, y3) |
| Mask features | y4 (finest level, 96→256 projected) |

**Learnable parameters:**
- `query_feat`: (100, 256) — query content embeddings
- `query_embed`: (100, 256) — query positional embeddings
- `level_embed`: (3, 256) — per-level embeddings added to backbone features

**Input projections:** Linear layers project each backbone level to `hidden_dim=256`:
- Level 0 (y1, 256-d): identity
- Level 1 (y2, 128-d): Linear(128→256)
- Level 2 (y3, 96-d): Linear(96→256)

**Mask features:** Finest level y4 (96-d) → Linear(96→256) + 3D positional encoding

**Per-layer forward (9 layers total):**
1. **Masked cross-attention**: queries attend to backbone point features at level `i % 3`. Attention mask = predicted mask thresholded at 0.5 (points below threshold are masked out).
2. **Self-attention**: queries attend to each other
3. **FFN**: 256→1024→256 with ReLU

**Prediction heads (applied after each layer):**
- `class_embed`: `Linear(256 → NUM_CLASSES+1)` — per-query class logits (+1 for no-object)
- `mask_embed`: `MLP(256 → 256 → 256)` — per-query mask embedding (3 layers)
- Mask logits: `einsum("bqc,bpc→bpq", mask_embed, mask_features)` — dot product

**Outputs per layer:** `pred_logits` [B, 100, C+1], `pred_masks` [B, P, 100]
**Total outputs:** 10 predictions (1 final + 9 auxiliary)

### 2.4 Positional Encoding (positional_encoder.py) — REUSE

3D sinusoidal encoding for point coordinates:
- Normalize: x,y by ÷48, z by ÷4
- Log-spaced frequency bands: 1 to 5000, base=2
- `num_bands` = floor(256 / 3 / 2) = 42 bands
- Output: 42 × 2 × 3 = 252 dims, zero-padded to 256

### 2.5 Transformer Blocks (blocks.py) — REUSE

- **SelfAttentionLayer**: Pre-norm. q=k=embed+pos, v=embed. `nn.MultiheadAttention(batch_first=True)`.
- **CrossAttentionLayer**: Pre-norm on queries. `key = value = bb_feat + pos`. Supports masked attention.
- **FFNLayer**: Pre-norm. `Linear(256→1024) → ReLU → Dropout → Linear(1024→256) + residual`.
- **MLP**: 3-layer `256→256→256` for mask embedding.

---

## 3. Loss Functions

### 3.1 MaskLoss — REUSE (adapt)

Hungarian matching + three losses on matched query-GT pairs:

**Loss weights (from model.yaml):**
| Loss | Weight |
|---|---|
| `loss_ce` (class cross-entropy) | 2.0 |
| `loss_dice` (dice on masks) | 5.0 |
| `loss_mask` (sigmoid BCE on masks) | 5.0 |

**Key parameters:**
| Param | Value |
|---|---|
| `eos_coef` | 0.1 (no-object class weight in CE) |
| `NUM_POINTS` | 50,000 (total sampled points for mask loss) |
| `NUM_MASK_PTS` | 500 (points sampled per matched mask) |
| `P_RATIO` | 0.4 (fraction of points used during Hungarian matching) |

**Process:**
1. Hungarian matcher finds optimal query↔GT assignment
2. **Class CE**: Weight vector (1.0 for real classes, 0.1 for no-object, 0.0 for ignore)
3. **Mask losses**: Sample 500 points per matched mask + random fill to 50k total. Compute dice + sigmoid BCE.
4. **Auxiliary**: Same computation on all 9 intermediate decoder outputs (same weights)

### 3.2 SemLoss — SKIP

Applied to backbone semantic head:
- CE (weight 2) + Lovász softmax (weight 6)
- We don't have a separate semantic head — our backbone is frozen.

### 3.3 Hungarian Matcher — REUSE

Cost matrix: `2.0 × cost_class + 5.0 × cost_mask + 5.0 × cost_dice`
- Subsamples 40% of points for efficiency
- Uses `scipy.optimize.linear_sum_assignment`
- JIT-compiled cost functions

---

## 4. Training Strategy

### 4.1 Optimizer & Scheduler
| Param | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Scheduler | StepLR (step=80, gamma=0.1) |
| Gradient clipping | 0.5 (max norm) |
| Gradient accumulation | 4 steps (effective batch=4) |

### 4.2 Training Config
| Param | Value |
|---|---|
| Epochs | 100 |
| Batch size | 1 |
| Workers | 6 |
| GPUs | all available (ddp) |
| SyncBatchNorm | Yes (converted via ME) |
| Framework | PyTorch Lightning |

### 4.3 Augmentations (training only)
1. **Random rotation**: Full 360° around Z axis
2. **Random flip**: 4 options (none, flip-x, flip-y, flip-both)
3. **Random scale**: Uniform [0.95, 1.05] on x,y
4. **Random translation**: Gaussian σ=[0.1, 0.1, 0.1]

### 4.4 Point Subsampling
- KITTI: Subsample to 80,000 points during training
- nuScenes: Subsample to 50,000 points during training
- Evaluation: no subsampling

### 4.5 Coordinate Bounds
- KITTI: x,y ∈ [-48, 48], z ∈ [-4, 1.5]
- nuScenes: x,y ∈ [-50, 50], z ∈ [-5, 3]

### 4.6 Callbacks
- LearningRateMonitor (per step)
- ModelCheckpoint on `metrics/iou` (max, save_last)
- ModelCheckpoint on `metrics/pq` (max, save_last)
- TensorBoard logging

---

## 5. Evaluation Pipeline

### 5.1 Panoptic Inference (`mask_model.py:panoptic_inference`)

```python
# 1. Get per-query scores and class labels
scores, labels = pred_logits.softmax(-1).max(-1)   # (B, 100)

# 2. Sigmoid on mask predictions, remove padding
masks = pred_masks.sigmoid()  # (B, P, 100)

# 3. Filter out no-object queries (label == num_classes)
keep = labels != num_classes

# 4. For each remaining query (by score):
#    - mask = (argmax over queries == this query) AND (mask >= 0.5)
#    - skip if overlap_ratio = mask_area / original_area < 0.8
#    - stuff: merge same-class segments
#    - things: assign unique instance IDs

# 5. Output: per-point semantic_pred, instance_pred
```

**Key threshold:** overlap_ratio = 0.8 (reject masks that lose too much area from argmax competition)

### 5.2 PQ Computation (`eval_np.py:PanopticEval`)

Standard SemanticKITTI panoptic quality:
- **min_points**: 50 (minimum points to count an instance)
- Instance ID offset: 1000 × semantic_id + instance_id
- TP: IoU > 0.5 between pred and GT instance
- **PQ = SQ × RQ** where SQ = mean IoU of matched pairs, RQ = 2×TP / (2×TP + FP + FN)
- Also reports PQ† (PQ-dagger): uses IoU instead of SQ×RQ for stuff classes

### 5.3 Results Output
- **KITTI**: `.label` files (uint32: lower 16 bits = semantic, upper 16 bits = instance)
- **nuScenes**: `.npz` files with `data = 1000×sem + instance` as uint16

---

## 6. MaskPLS nuScenes Class Convention

From `semantic-nuscenes.yaml`:

**17 classes (1-indexed, 0=ignore):**

| ID | Name | Thing/Stuff |
|---|---|---|
| 1 | noise | stuff |
| 2 | barrier | thing |
| 3 | bicycle | thing |
| 4 | bus | thing |
| 5 | car | thing |
| 6 | construction_vehicle | thing |
| 7 | motorcycle | thing |
| 8 | other_flat | stuff |
| 9 | pedestrian | thing |
| 10 | traffic_cone | thing |
| 11 | driveable_surface | stuff |
| 12 | sidewalk | stuff |
| 13 | terrain | stuff |
| 14 | trailer | stuff* |
| 15 | truck | stuff* |
| 16 | manmade | stuff |
| 17 | vegetation | stuff |

> **Note:** MaskPLS's nuScenes convention differs from ours. They have 17 classes with different ordering and thing/stuff assignment. We use 16 classes (no "noise" class — it's ignored). **Our `GENERAL_TO_PANOPTIC_16` mapping is the correct one for standard nuScenes panoptic.**

---

## 7. What to Pick vs What Not to Pick

### PICK (adapt for our project)

| Component | File | Reason |
|---|---|---|
| Masked transformer decoder | `decoder.py` | Core architecture — adapt input projections for our backbone |
| Transformer blocks | `blocks.py` | Self-attn, cross-attn, FFN — clean, reusable |
| Positional encoding | `positional_encoder.py` | 3D sinusoidal PE — works as-is |
| Hungarian matcher | `matcher.py` | Standard, well-tested |
| MaskLoss (CE+Dice+BCE) | `loss.py` (MaskLoss class) | Core panoptic loss — adapt class count |
| PanopticEval | `eval_np.py` | PQ/SQ/RQ computation — reuse directly |
| PanopticEvaluator wrapper | `evaluate_panoptic.py` | Convenience wrapper |
| `pad_stack`, `sample_points` | `misc.py` | Useful utilities |
| Decoder config | `config/decoder.yaml` | Starting hyperparameters |
| Loss weights | `config/model.yaml` | Starting hyperparameters |

### DO NOT PICK

| Component | File | Reason |
|---|---|---|
| Backbone (`MinkEncoderDecoder`) | `mink.py` | We use frozen OpenScene DisNet |
| Semantic loss (SemLoss) | `loss.py` | No separate semantic head in our design |
| SemanticKITTI dataloader | `semantic_dataset.py` | We have our own `NuScenesPanoptic` |
| PyTorch Lightning wrapper | `mask_model.py` | Reimplement in plain PyTorch for clarity |
| KNN interpolation via pykeops | `interpolate.py` | Extra dependency; use simpler voxel→point mapping |
| Backbone config | `backbone.yaml` | Not relevant |

### REFERENCE ONLY

| Component | File | Why reference |
|---|---|---|
| `mask_model.py` forward + inference | `mask_model.py` | Study the forward flow and panoptic inference logic |
| Mask construction in dataloader | `semantic_dataset.py` | Compare with our mask construction approach |
| nuScenes class mapping | `semantic-nuscenes.yaml` | Cross-reference with our 16-class convention |
| Training callbacks | `train_model.py` | Study checkpoint/logging patterns |

---

## 8. Key Adaptation Notes for Our Project

### 8.1 Backbone interface change

MaskPLS backbone outputs 4 feature levels + KNN-interpolated point features.
Our frozen DisNet outputs a single 768-d feature vector per voxel.

**Adaptation needed:**
- Single-level decoder (or project 768→256 and use as all levels)
- No KNN interpolation needed (we stay in voxel space, use `inv_idx` for point mapping)
- Add adapter: `Linear(768→256) + LayerNorm`

### 8.2 Multi-level vs single-level features

MaskPLS uses 3 feature levels for cross-attention cycling.
Options for us:
1. **Use single level everywhere** — simplest, may lose multi-scale benefit
2. **Create fake multi-scale** from backbone intermediate layers (requires modifying DisNet to return intermediates)
3. **Use single level but keep the 3-block cycling** — mask features and cross-attn features are the same

Recommendation: **Option 1** for smoke test, consider Option 2 later.

### 8.3 Class count

MaskPLS nuScenes: 17 classes + 1 no-object = 18
Our convention: 16 classes + 1 no-object = 17

Update `class_embed` output dimension accordingly.

### 8.4 Point subsampling

MaskPLS subsamples to 50K points for nuScenes training.
Our voxels are typically ~20K per frame — no subsampling needed.

### 8.5 No semantic head

MaskPLS has a separate semantic head on the backbone (SemLoss).
We skip this entirely — our backbone is frozen, semantic comes from panoptic masks.

### 8.6 Batch handling

MaskPLS uses `pad_stack` to pad variable-length point sets to the same size in a batch.
We use MinkowskiEngine sparse batching for backbone, then need to handle variable-length features for the decoder.
With batch_size=1, this is trivial.

---

## 9. Published Results

### SemanticKITTI (from paper)

| Method | PQ | PQ† | SQ | RQ | mIoU |
|---|---|---|---|---|---|
| MaskPLS | 53.0 | 59.0 | — | — | 62.6 |

### nuScenes (from paper)

| Method | PQ | mIoU |
|---|---|---|
| MaskPLS | 52.2 | 66.2 |

---

## 10. Dependencies Comparison

| Dependency | MaskPLS | Our env (`openscene`) |
|---|---|---|
| PyTorch | 1.10.1+cu113 | 1.7.1+cu110 |
| MinkowskiEngine | 0.5.4 | (older, in conda env) |
| Python | 3.8 | 3.8 |
| pytorch_lightning | 1.5.6 | not installed |
| pykeops | 2.1 | not installed |
| scipy | 1.7.3 | likely available |

**Key implication:** We cannot import MaskPLS modules directly — different PyTorch version. Must copy/adapt the code we need.
