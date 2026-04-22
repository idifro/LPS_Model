# P3Former Repository Understanding

> **Paper**: "Position-Guided Point Cloud Panoptic Segmentation Transformer" (CVPR 2023)
> **Authors**: Xiao, Zhang, Wang, Chen Change Loy, Lin, Pang (SmartBot / PJLab)
> **Repo**: `/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/third_party/P3Former`
> **Key idea**: Three position-guided components — MPE, PA-Seg, MFA — for panoptic LiDAR segmentation.

---

## 1. Complete File Map

| File | Purpose | Pick for us? |
|---|---|---|
| `p3former/decode_heads/p3former_head.py` | **Core head**: `_P3FormerHead`, `_Masked_Focal_Attention`, `_Transformer_Decoder`, MLP | **REFERENCE** — study MPE, PA-Seg, MFA concepts |
| `p3former/backbones/cylinder3d.py` | Backbone: `_Asymm3DSpconv` with cylindrical voxelization | **NO** — we use frozen DisNet |
| `p3former/segmentors/p3former.py` | Top-level segmentor extending Cylinder3D | **REFERENCE** — study forward flow |
| `p3former/data_preprocessors/data_preprocessor.py` | `_Det3DDataPreprocessor` with cylindrical voxelization | **NO** — different voxelization scheme |
| `p3former/task_modules/samplers/mask_pseduo_sampler.py` | `_MaskPseudoSampler` (from K-Net) | **NO** — tied to mmdet3d |
| `datasets/semantickitti_dataset.py` | SemanticKITTI dataset | **NO** — we use nuScenes |
| `datasets/transforms/loading.py` | Panoptic label loading | **REFERENCE** |
| `datasets/transforms/transforms_3d.py` | **PolarMix + LaserMix** augmentations (panoptic-aware) | **MAYBE** — novel augmentations, but tied to mmdet3d |
| `datasets/transforms/formating.py` | Test submission formatting | **NO** |
| `evaluation/functional/panoptic_seg_eval.py` | `EvalPanoptic`: PQ/SQ/RQ/mIoU computation | **YES** — reuse for PQ eval (standalone numpy) |
| `evaluation/metrics/panoptic_seg_metric.py` | Metric wrapper + submission formatting | **REFERENCE** |
| `configs/_base_/models/p3former.py` | Base model config | **REFERENCE** — key hyperparameters |
| `configs/p3former/p3former_8xb2_3x_semantickitti.py` | Main training config | **REFERENCE** |
| `configs/_base_/datasets/semantickitti_panoptic_lpmix.py` | Panoptic dataset config | **REFERENCE** |
| `configs/_base_/default_runtime.py` | Runtime defaults | **NO** |
| `tools/create_data.py` | Data preparation entry | **NO** |
| `tools/dataset_converters/semantickitti_converter.py` | Creates info pkl files | **NO** |
| `train.py` | Training entry (mmengine Runner) | **NO** — mmengine-specific |
| `test.py` | Test entry | **NO** |

---

## 2. Architecture Summary

### 2.1 Overall Pipeline

```
Points (x,y,z,intensity)
    │
    ▼  Cylindrical voxelization
    │  (rho, phi, z) → grid [480, 360, 32]
    │
    ▼
┌───────────────────────────────┐
│  Backbone: Cylinder3D          │  ← we replace with frozen DisNet
│  _Asymm3DSpconv                │
│  Input: 16-d (from SegVFE)     │
│  4 down blocks + 4 up blocks   │
│  Channels: 32→64→128→256→512   │
│  + DDCM context modeling        │
│  Output: 256-d per voxel       │
└───────────┬───────────────────┘
            │
            ▼
┌─────────────────────────────────────┐
│  P3Former Head                       │
│                                      │
│  ┌─ MPE (Mixed Positional Embedding)│
│  │  polar_proj(3→256) + cart_proj    │
│  │  (3→256) → added to features     │
│  │                                   │
│  ├─ 128 thing queries (learnable)    │
│  ├─ 11 stuff queries (from sem head) │
│  │  Total: 139 queries               │
│  │                                   │
│  ├─ Transformer Decoder (6 layers)   │
│  │  Per layer:                       │
│  │    1. MFA (Masked Focal Attention)│
│  │    2. Self-attention (8 heads)    │
│  │    3. FFN (256→2048→256)          │
│  │                                   │
│  ├─ PA-Seg (Position-Aware Seg)      │
│  │  mask = feat_mask + pos_mask      │
│  │                                   │
│  └─ Class head: MLP(256→256→20)      │
│     Mask head: MLP(256→256→256→256)  │
└─────────────────────────────────────┘
```

### 2.2 Backbone: `_Asymm3DSpconv` — NOT USED BY US

Cylinder3D-based backbone with:
- **Cylindrical voxelization**: (x,y,z) → (rho, phi, z), grid [480, 360, 32]
- **Asymmetric residual blocks**: Uses (1,3,3) and (3,1,3) kernels instead of standard (3,3,3)
- **DDCM**: Dimension-Decomposition Context Modeling — applies (3,1,1), (1,3,1), (1,1,3) convolutions with sigmoid gating
- **NOT MinkowskiEngine** — uses mmcv.ops SparseConv3d (spconv-based)
- Output: 256-d per voxel

**Why we don't use this:** Different sparse conv library (spconv vs ME), different voxelization scheme (cylindrical vs Cartesian), and we have frozen DisNet.

### 2.3 The Three "P3" Novel Components

#### MPE — Mixed-Parameterized Positional Embedding

```python
# Compute both coordinate systems
polar_coords = (rho, phi, z)          # normalized
cart_coords  = (rho*cos(phi), rho*sin(phi), z)

# Separate projections
polar_pe = pe_conv(polar_proj(polar_coords))  # Linear(3→256) + LN + ReLU
cart_pe  = pe_conv(cart_proj(cart_coords))     # Linear(3→256) + LN + ReLU

# Also process backbone features through same pe_conv
features = pe_conv(features)

# Add both PEs to features
pe = polar_pe + cart_pe
features = features + pe  # [V, 256]
```

**Key insight:** Dual coordinate representation (polar + Cartesian) captures different spatial patterns. Polar helps with range-based patterns, Cartesian with absolute position.

**For us:** Our DisNet operates in Cartesian space. We could add a simplified positional encoding (Cartesian only) to our adapter. The dual polar+Cartesian is tied to cylindrical voxelization which we don't use.

#### PA-Seg — Position-Aware Segmentation

```python
# Standard feature-based mask
feat_mask = einsum('nc,vc->nv', fc_mask(queries), features)

# Position-based mask (uses MPE directly)
pos_mask  = einsum('nc,vc->nv', fc_coor_mask(queries), mpe)

# Final mask is the sum
mask_pred = feat_mask + pos_mask
```

**Key insight:** Instances in 3D are strongly correlated with position (a car occupies a spatial region). Adding a position-only mask branch directly leverages this. Separate loss on `pos_mask` with weight 0.2.

**For us:** Interesting idea. Could add a positional mask branch using xyz coordinates of voxels. Low implementation cost, potentially useful for spatial disambiguation.

#### MFA — Masked Focal Attention

```python
# Pool features within predicted mask region
binary_masks = (masks.sigmoid() > 0.5).float()
masked_feats = einsum('nv,vc->nc', binary_masks, features)  # [N_queries, C]

# Gated update (from K-Net)
param_out  = dynamic_params(queries)        # query-derived
input_out  = input_transform(masked_feats)  # feature-derived
update_gate = sigmoid(gate_param(queries))
input_gate  = sigmoid(gate_input(masked_feats))
output = update_gate * param_out + input_gate * input_out
```

**Key insight:** Instead of standard cross-attention (all queries attend to all points), each query only pools features from its predicted mask region. This is more focused and computationally cheaper than full attention.

**For us:** This replaces standard cross-attention. Could be useful but adds complexity. The gating mechanism from K-Net is an extra engineering step. Standard cross-attention (as in MaskPLS) is simpler for initial implementation.

### 2.4 Query Design

| Type | Count | Initialization |
|---|---|---|
| Thing queries | 128 | Learnable `SubMConv3d(256, 128, 1)` weight |
| Stuff queries | 11 | From semantic head weights `Conv3d(256, 20, 1)` for stuff class rows |
| **Total** | **139** | |

**Stuff query trick:** Stuff queries are initialized from the semantic segmentation head's weight matrix. Each stuff query corresponds to exactly one stuff class, enforced by `sem_stuff_weights = eye(num_stuff_classes)`.

**For us:** MaskPLS uses 100 unified queries. P3Former's split (128 thing + 11 stuff) is more structured. Consider this for our design — could use 100 thing + 6 stuff = 106 queries for our 16-class nuScenes.

### 2.5 Decoder Structure

| Param | Value |
|---|---|
| Decoder layers | 6 |
| Embed dims | 256 |
| Self-attention heads | 8 |
| FFN dim | 2048 |
| Per-layer flow | MFA → Self-attn → FFN |
| Prediction heads per layer | class logits + mask + pos_mask |

**Comparison with MaskPLS:**
- MaskPLS: 9 layers (3 blocks × 3 levels), multi-level cross-attention cycling
- P3Former: 6 layers, single-level MFA, but with PA-Seg and MPE

---

## 3. Loss Functions

### 3.1 Per Decoder Layer (layers 1-6)

| Loss | Type | Weight |
|---|---|---|
| `loss_cls` | FocalLoss (γ=4.0, α=0.25) | 1.0 |
| `loss_mask` | FocalLoss (γ=2.0, α=0.25, binary) | 1.0 |
| `loss_dice` | DiceLoss | 2.0 |
| `loss_dice_pos` | DiceLoss × `pa_seg_weight` | 0.2 |

**Layer 0** has no class loss (identity matching only uses mask costs).

### 3.2 Semantic Auxiliary Loss

| Loss | Weight |
|---|---|
| CrossEntropyLoss | 1.0 |
| LovaszLoss | 1.0 |

Applied to per-voxel semantic predictions from `sem_queries`.

### 3.3 Matching Strategy

**Layer 0 (initial):** HungarianAssigner with:
- FocalLossCost (binary, γ=2.0, α=0.25, weight=1.0)
- DiceCost (weight=2.0)
- Only thing queries matched

**Layers 1-6:** HungarianAssigner with:
- FocalLossCost (classification, γ=4.0, α=0.25, weight=1.0)
- FocalLossCost (binary mask, γ=2.0, α=0.25, weight=1.0)
- DiceCost (weight=2.0)

**Key difference from MaskPLS:** P3Former uses FocalLoss-based costs instead of CE+Dice+BCE. Also has separate layer-0 matching (no class cost in first layer).

---

## 4. Training Strategy

| Param | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 0.0008 |
| Weight decay | 0.01 |
| Scheduler | MultiStepLR (milestones=[24, 32], γ=0.2) |
| Epochs | 36 |
| Batch size | 2/GPU × 8 GPUs = 16 total |
| Distributed | torch.distributed.launch |
| Checkpoint interval | Every 5 epochs |
| Backbone pretrained | Yes (Cylinder3D semantic pretraining recommended) |

### 4.1 Augmentations (panoptic-aware)

1. **LaserMix** (prob 0.5): Divides by pitch angle into 3-6 areas, alternates between 2 scans
2. **PolarMix** (prob 0.5): Swaps azimuth sectors + rotate-paste instance points
3. **RandomFlip3D**: horizontal 0.5, vertical 0.5
4. **GlobalRotScaleTrans**: rotation [-π/4, π/4], scale [0.95, 1.05], translation σ=0.1

**Note:** LaserMix and PolarMix propagate panoptic labels correctly — they are instance-aware augmentations. These are stronger than MaskPLS's simpler augmentations.

### 4.2 Coordinate Bounds

- Cylindrical: rho ∈ [0, 50], phi ∈ [-π, π], z ∈ [-4, 2]
- Grid: [480, 360, 32]

---

## 5. Evaluation Pipeline

### 5.1 Panoptic Prediction Assembly

```python
# Score computation
thing_scores = sigmoid(cls_pred[:, thing_classes]).max(dim=1) * 2  # scaled up
stuff_scores = diagonal of sigmoid(cls_pred[:, stuff_classes])

# Filter by score_thr = 0.4
# Weighted mask assignment: argmax(mask * score) per point
# Overlap filter: skip if mask_area / original_area < iou_thr (0.8)
# Stuff: one segment per class
# Things: unique instance IDs
```

### 5.2 PQ Computation (`EvalPanoptic`)

Standard panoptic quality:
- TP threshold: IoU > 0.5
- min_num_points: 50
- PQ = SQ × RQ
- PQ† = mean(thing PQ + stuff mIoU)
- Reports: PQ, PQ†, SQ, RQ, mIoU, PQ_things, PQ_stuff

---

## 6. Dependencies

| Dependency | Version |
|---|---|
| Python | 3.8 |
| PyTorch | 1.10.1+cu111 |
| mmengine | 0.7.4 |
| mmcv | 2.0.0rc4 |
| mmdet | 3.0.0 |
| mmdet3d | 1.1.0 |
| torch_scatter | 2.0.9 |
| **MinkowskiEngine** | **NOT USED** — uses mmcv.ops sparse convs |

**Key implication:** P3Former's mmdet3d ecosystem is very different from our ME-based setup. We cannot import any modules directly — everything must be studied and reimplemented if desired.

---

## 7. Published Results

### SemanticKITTI

| Split | PQ | PQ† | SQ | RQ | mIoU | PQ_Th | PQ_St |
|---|---|---|---|---|---|---|---|
| test | 65.3 | 67.8 | 86.6 | 74.9 | — | 67.4 | 63.7 |
| val | 62.6 | 66.2 | 76.2 | 72.4 | — | 69.4 | 57.7 |

**Comparison:** MaskPLS achieves 53.0 PQ on SemanticKITTI test. P3Former is +12.3 PQ better.

---

## 8. Comparison: P3Former vs MaskPLS

| Aspect | P3Former | MaskPLS |
|---|---|---|
| **PQ (KITTI test)** | **65.3** | 53.0 |
| **Backbone** | Cylinder3D (spconv, cylindrical) | MinkUNet (ME, Cartesian) |
| **Sparse conv library** | mmcv.ops (spconv) | MinkowskiEngine |
| **Voxelization** | Cylindrical (rho, phi, z) | Cartesian (x, y, z) |
| **Cross-attention** | MFA (gated, mask-pooled) | Standard masked cross-attn |
| **Positional encoding** | MPE: dual polar+Cartesian | 3D sinusoidal |
| **Position-aware masks** | PA-Seg (yes) | No |
| **Queries** | 128 thing + 11 stuff = 139 | 100 unified |
| **Decoder layers** | 6 | 9 (3×3 levels) |
| **Multi-level features** | Single level | 3 levels cycled |
| **Embed dims** | 256 | 256 |
| **FFN dim** | 2048 | 1024 |
| **Class loss** | FocalLoss | CrossEntropy |
| **Augmentations** | PolarMix + LaserMix (strong) | Rotation, flip, scale (simple) |
| **Framework** | mmdet3d (mmengine) | PyTorch Lightning |
| **Epochs** | 36 | 100 |
| **Training GPUs** | 8 × (batch 2) = 16 | 1+ (batch 1, accum 4) |

---

## 9. What's Useful for Our Project

### Concepts to borrow (implement ourselves)

| Concept | Source | Benefit for us |
|---|---|---|
| **PA-Seg (positional mask branch)** | `p3former_head.py` | Low-cost addition: separate `pos_mask = einsum(queries, xyz_embed)` added to feature mask. Could help spatial disambiguation. |
| **Thing/stuff query split** | `p3former_head.py` | More structured than MaskPLS's unified queries. Use 100 thing + 6 stuff for our 16 classes. |
| **Stuff queries from semantic weights** | `p3former_head.py` | Clever initialization: stuff query = semantic head weight for that class. |
| **Layer-0 mask-only matching** | config | First layer matches by mask cost only (no class cost). Helps early convergence. |
| **FocalLoss for classification** | config | Better for imbalanced classes than CE. |
| **PQ eval code** | `panoptic_seg_eval.py` | Standalone numpy PQ — can adapt without mmdet3d. |

### Concepts NOT to borrow

| Concept | Reason |
|---|---|
| Cylindrical voxelization | Our backbone (DisNet) was trained with Cartesian 0.05m voxels |
| MFA (Masked Focal Attention) | Too complex for initial implementation; standard cross-attn is simpler to debug |
| MPE dual polar+Cartesian PE | Tied to cylindrical voxelization; simpler Cartesian PE suffices for us |
| mmdet3d framework | Completely different ecosystem from our ME-based setup |
| Backbone (Cylinder3D) | We use frozen DisNet |
| PolarMix/LaserMix augmentations | Tied to mmdet3d transforms; reimplementing is low priority |

---

## 10. Summary: Design Decisions Informed by Both Repos

| Decision | MaskPLS approach | P3Former approach | **Our choice** |
|---|---|---|---|
| Decoder type | Mask2Former cross-attn | MFA (gated, mask-pooled) | **Mask2Former style** (simpler, well-tested) |
| Queries | 100 unified | 128 thing + 11 stuff | **Split: ~100 thing + 6 stuff** |
| Positional encoding | 3D sinusoidal | MPE (dual polar+Cartesian) | **3D sinusoidal** (matches Cartesian voxels) |
| Position-aware masks | No | PA-Seg | **Consider adding** (low cost, potential benefit) |
| Class loss | CE | FocalLoss | **FocalLoss** (handles imbalance better) |
| Mask loss | BCE + Dice | FocalLoss + Dice | **BCE + Dice** (simpler, well-tested) |
| Multi-level features | 3 levels cycled | Single level | **Single level** (our backbone is single-output) |
| Augmentations | Simple (rotate, flip, scale) | PolarMix + LaserMix | **Simple first**, add stronger later |
| PQ evaluation | SemanticKITTI eval_np | EvalPanoptic | **Either** — both are numpy-based |
