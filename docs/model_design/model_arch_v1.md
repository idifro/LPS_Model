# Model Architecture v1 - Panoptic 3D Segmentation on nuScenes

**Version:** v1.1 (updated for full nuScenes training)
**Date:** 2026-04-23
**Files covered:** All files under `panoptic/`

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [End-to-End Data Flow Diagram](#2-end-to-end-data-flow-diagram)
3. [Stage 1: Data Loading & Preprocessing](#3-stage-1-data-loading--preprocessing)
4. [Stage 2: Backbone Encoder (Frozen)](#4-stage-2-backbone-encoder-frozen)
5. [Stage 3: Panoptic Decoder](#5-stage-3-panoptic-decoder)
6. [Stage 4: Loss Computation](#6-stage-4-loss-computation)
7. [Stage 5: Panoptic Inference](#7-stage-5-panoptic-inference)
8. [Stage 6: Evaluation Metrics](#8-stage-6-evaluation-metrics)
9. [Training Loop & Optimizer](#9-training-loop--optimizer)
10. [Training Arithmetic: Full nuScenes](#10-training-arithmetic-full-nuscenes)
11. [Dimension Summary Table](#11-dimension-summary-table)
12. [Design Choices & Rationale](#12-design-choices--rationale)
13. [File Map](#13-file-map)
14. [Config Reference](#14-config-reference)

---

## 1. System Overview

```
                         FROZEN                          TRAINABLE
                    ┌──────────────┐            ┌─────────────────────────┐
  nuScenes LiDAR   │   DisNet      │            │   PanopticDecoder       │
  point cloud ────>│  (MinkUNet18A)│──768-d───>│  (Mask2Former-style)    │──> PQ Eval
  (N, 5)           │  CLIP-distilled│  per-voxel │  100 thing + 6 stuff    │
                    │  15.6M params │  features  │  queries, 9 layers      │
                    └──────────────┘            └─────────────────────────┘
                           |                              |
                     OpenScene pretrained          Hungarian Matching
                     nuscenes_openseg.pth.tar       + CE + BCE + Dice Loss
```

**Goal:** Supervised panoptic 3D segmentation of nuScenes LiDAR scans using:
- A frozen OpenScene backbone that outputs CLIP-aligned 768-d features per voxel
- A trainable Mask2Former-style transformer decoder that predicts class labels + binary masks

---

## 2. End-to-End Data Flow Diagram

```
                                    TRAINING FORWARD PASS
                                    =====================

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ NuScenesPanoptic Dataset (__getitem__)                                      │
  │                                                                             │
  │  LiDAR .bin ──> (N,5) ──range_filter──> (N',5) ──augment──> (N',5)        │
  │                                                                             │
  │  panoptic .npz ──> pan_data (N) ──range_filter──> (N')                     │
  │       │                                                                     │
  │       ├── sem = pan // 1000 ──GENERAL_TO_PANOPTIC_16──> 16-class (1-idx)   │
  │       └── inst = pan % 1000                                                 │
  │                                                                             │
  │  Voxelize @ 0.05m:  (N',5) ──np.unique──> vox_coords (V,3), vox_feats (V,3)│
  │                                                                             │
  │  Build GT:  thing masks (per-instance) + stuff masks (per-class)           │
  │             gt_masks: (M, V) bool,  gt_classes: (M,) int64 [0-indexed]     │
  │             gt_sem: (V,) int64 [0-15, 255=ignore]                          │
  └────────────────────────────────┬────────────────────────────────────────────┘
                                   │
                                   ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ extract_backbone_features()                                                 │
  │                                                                             │
  │  vox_coords (V,3) + vox_feats (V,3)                                        │
  │       │                                                                     │
  │       ├── Prepend batch dim: (V,4) = [0, x_vox, y_vox, z_vox]             │
  │       ├── ME.SparseTensor(feats=vox_feats, coords=batch_coords)            │
  │       │                                                                     │
  │       └── DisNet forward (frozen, no_grad):                                │
  │              SparseTensor(V, 3) ─── MinkUNet18A ───> Tensor(V, 768)        │
  │                                                                             │
  │  Output:  backbone_feats (1, V, 768),  coords_xyz (1, V, 3)               │
  └────────────────────────────────┬────────────────────────────────────────────┘
                                   │
                                   ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ PanopticDecoder.forward(backbone_feats, coords)                             │
  │                                                                             │
  │  ┌─── Projections ────────────────────────────────────────────────────┐     │
  │  │  backbone_proj:  (B,V,768) ──Linear+LN──> src       (B,V,256)    │     │
  │  │  mask_feat_proj: (B,V,768) ──Linear+LN──> mask_feat (B,V,256)    │     │
  │  │  pe_layer:       (B,V,3)   ──SinPE──────> pe        (B,V,256)    │     │
  │  │  mask_features = mask_feat + pe                      (B,V,256)    │     │
  │  └───────────────────────────────────────────────────────────────────┘     │
  │                                                                             │
  │  ┌─── Query Init ────────────────────────────────────────────────────┐     │
  │  │  thing_query_feat:  Embedding(100, 256)  ─┐                       │     │
  │  │  stuff_query_feat:  Embedding(6, 256)    ─┤──> output      (B,106,256)│ │
  │  │  thing_query_embed: Embedding(100, 256)  ─┐                       │     │
  │  │  stuff_query_embed: Embedding(6, 256)    ─┤──> query_embed (B,106,256)│ │
  │  └───────────────────────────────────────────────────────────────────┘     │
  │                                                                             │
  │  ┌─── Initial Prediction (Layer 0) ──────────────────────────────────┐     │
  │  │  pred_heads(output, mask_features) ──> class_0, mask_0, attn_mask │     │
  │  └───────────────────────────────────────────────────────────────────┘     │
  │                                                                             │
  │  ┌─── Decoder Layers (i = 0..8) ─────────────────────────────────────┐     │
  │  │  level_idx = i % 3                                                 │     │
  │  │  src_level = src + level_embed[level_idx]           (B,V,256)     │     │
  │  │                                                                     │     │
  │  │  ┌── Cross-Attention ──────────────────────────────────────────┐   │     │
  │  │  │  Q = output + query_embed    (B,106,256) ──transpose──> (106,B,256)│ │
  │  │  │  K = src_level + pe          (B,V,256)   ──transpose──> (V,B,256)  │ │
  │  │  │  V = src_level + pe          (B,V,256)   ──transpose──> (V,B,256)  │ │
  │  │  │  attn_mask: (B*8, 106, V)  from previous pred_heads     │   │     │
  │  │  │  MultiheadAttention(256, 8 heads, head_dim=32)           │   │     │
  │  │  │  output = output + dropout(attn_out)                     │   │     │
  │  │  └─────────────────────────────────────────────────────────────┘   │     │
  │  │                                                                     │     │
  │  │  ┌── Self-Attention ───────────────────────────────────────────┐   │     │
  │  │  │  Q = K = output + query_embed    (B,106,256)               │   │     │
  │  │  │  V = output                      (B,106,256)               │   │     │
  │  │  │  MultiheadAttention(256, 8 heads)                          │   │     │
  │  │  │  output = output + dropout(attn_out)                       │   │     │
  │  │  │  output = LayerNorm(output)                                │   │     │
  │  │  └─────────────────────────────────────────────────────────────┘   │     │
  │  │                                                                     │     │
  │  │  ┌── FFN ──────────────────────────────────────────────────────┐   │     │
  │  │  │  output = LayerNorm(output)                                 │   │     │
  │  │  │  output = output + Linear(1024→256, ReLU(Linear(256→1024))) │   │     │
  │  │  └─────────────────────────────────────────────────────────────┘   │     │
  │  │                                                                     │     │
  │  │  ┌── Prediction Heads ─────────────────────────────────────────┐   │     │
  │  │  │  decoder_norm = LayerNorm(output)           (B,106,256)    │   │     │
  │  │  │  class_logits = Linear(256→17)              (B,106,17)     │   │     │
  │  │  │  mask_embed = MLP(256→256→256, 3 layers)    (B,106,256)    │   │     │
  │  │  │  mask_logits = einsum(mask_embed, mask_features)           │   │     │
  │  │  │              = "bqc,bpc->bpq"               (B,V,106)     │   │     │
  │  │  │  attn_mask = (sigmoid(mask_logits) < 0.5)   (B*8,106,V)  │   │     │
  │  │  └─────────────────────────────────────────────────────────────┘   │     │
  │  └─────────────────────────────────────────────────────────────────────┘     │
  │                                                                             │
  │  Output dict:                                                               │
  │    pred_logits: (B, 106, 17)   ← final layer class predictions             │
  │    pred_masks:  (B, V, 106)    ← final layer mask predictions              │
  │    aux_outputs: list of 9 dicts, each with pred_logits + pred_masks        │
  └────────────────────────────────┬────────────────────────────────────────────┘
                                   │
                                   ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ PanopticLoss.forward(outputs, targets)                                      │
  │                                                                             │
  │  For EACH prediction layer (1 final + 9 auxiliary = 10 total):             │
  │                                                                             │
  │  ┌── Hungarian Matcher ───────────────────────────────────────────────┐     │
  │  │  Cost matrix C (106, M) =                                          │     │
  │  │    2.0 * class_cost  (neg softmax prob for GT class)               │     │
  │  │  + 5.0 * bce_cost    (sigmoid CE between pred/GT mask samples)     │     │
  │  │  + 5.0 * dice_cost   (1 - dice between pred/GT mask samples)       │     │
  │  │  Points subsampled: 75% of V (p_ratio=0.75), without replacement   │     │
  │  │  scipy.linear_sum_assignment(C) ──> (pred_idx, gt_idx) pairs       │     │
  │  └────────────────────────────────────────────────────────────────────┘     │
  │                                                                             │
  │  ┌── Classification Loss (CE) ────────────────────────────────────────┐     │
  │  │  Matched queries get GT class label (0-15)                         │     │
  │  │  Unmatched queries get no-object label (16)                        │     │
  │  │  F.cross_entropy(pred_logits, target, weight=[1..1, 0.1])         │     │
  │  │  no-object class weight = eos_coef = 0.1                          │     │
  │  │  loss_ce *= 2.0 (weight)                                          │     │
  │  └────────────────────────────────────────────────────────────────────┘     │
  │                                                                             │
  │  ┌── Mask BCE Loss ───────────────────────────────────────────────────┐     │
  │  │  For each matched (pred_query, gt_mask) pair:                      │     │
  │  │  Sigmoid BCE per-voxel, mean over voxels, sum over pairs / M      │     │
  │  │  Point sampling: min(50000, V), without replacement                │     │
  │  │  loss_mask *= 5.0 (weight)                                         │     │
  │  └────────────────────────────────────────────────────────────────────┘     │
  │                                                                             │
  │  ┌── Dice Loss ───────────────────────────────────────────────────────┐     │
  │  │  pred_sig = sigmoid(pred_mask_logits)                              │     │
  │  │  dice = 2*sum(pred_sig * gt) / (sum(pred_sig) + sum(gt))          │     │
  │  │  loss_dice = (1 - dice), sum over pairs / M                       │     │
  │  │  loss_dice *= 5.0 (weight)                                        │     │
  │  └────────────────────────────────────────────────────────────────────┘     │
  │                                                                             │
  │  Total = sum of all 30 loss terms (3 losses x 10 layers)                   │
  └────────────────────────────────┬────────────────────────────────────────────┘
                                   │
                                   ▼
                            backward() + AdamW step
```

---

## 3. Stage 1: Data Loading & Preprocessing

**File:** `panoptic/dataset/nuscenes_panoptic.py`

### 3.1 Input Data

| Source | Shape | Description |
|--------|-------|-------------|
| LiDAR .bin | (N, 5) | x, y, z, intensity, ring_index |
| Panoptic .npz | (N,) | Encoded as `semantic * 1000 + instance_id` |

### 3.2 Processing Pipeline

```
Raw points (N, 5)
    │
    ├── Range filter: [-51.2, -51.2, -5.0] to [51.2, 51.2, 3.0]
    │   Keeps points within 51.2m XY, -5 to +3m Z
    │   Typical: N ≈ 34K → N' ≈ 30K after filter
    │
    ├── Augmentation (training only):
    │   ├── Random rotation Z: uniform [0, 2*pi]
    │   ├── Random flip X: 50% chance
    │   ├── Random flip Y: 50% chance
    │   ├── Random scale XY: uniform [0.95, 1.05]
    │   └── Random translate: Gaussian std=0.1m
    │
    ├── Voxelization @ 0.05m:
    │   coords_f = pts[:, :3] / 0.05
    │   coords_i = floor(coords_f)
    │   np.unique(coords_i) → vox_idx, inv_idx
    │   Typical: N' ≈ 30K → V ≈ 20-22K voxels
    │
    └── Panoptic label decoding:
        sem_general = pan // 1000          (nuScenes general 0-31)
        inst = pan % 1000                  (instance ID)
        sem_16 = GENERAL_TO_PANOPTIC_16[sem_general]  (mapped to 1-16)
        vox_sem = sem_16 - 1               (0-indexed: 0-15, ignore=255)
```

### 3.3 GT Mask Construction

```
Thing masks (classes 0-9): One binary mask per unique (class, instance_id) pair
  Example: 3 cars → 3 separate masks, each (V,) bool

Stuff masks (classes 10-15): One binary mask per class present
  Example: driveable_surface → 1 mask covering all driveable voxels

Total M = num_thing_instances + num_stuff_classes_present
Typical M ≈ 15-30 per scene
```

### 3.4 16-Class Panoptic Taxonomy

| Index (0-idx) | Name | Type |
|---|---|---|
| 0 | barrier | thing |
| 1 | bicycle | thing |
| 2 | bus | thing |
| 3 | car | thing |
| 4 | construction_vehicle | thing |
| 5 | motorcycle | thing |
| 6 | pedestrian | thing |
| 7 | traffic_cone | thing |
| 8 | trailer | thing |
| 9 | truck | thing |
| 10 | driveable_surface | stuff |
| 11 | other_flat | stuff |
| 12 | sidewalk | stuff |
| 13 | terrain | stuff |
| 14 | manmade | stuff |
| 15 | vegetation | stuff |

### 3.5 Output Tensors

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `vox_coords` | (V, 3) | int32 | Voxel grid coordinates |
| `vox_feats` | (V, 3) | float32 | XYZ in meters (backbone input) |
| `gt_masks` | (M, V) | bool | Binary masks per segment |
| `gt_classes` | (M,) | int64 | 0-indexed class per mask |
| `gt_sem` | (V,) | int64 | Per-voxel semantic (255=ignore) |
| `inv_idx` | (N',) | int64 | Point → voxel mapping |

---

## 4. Stage 2: Backbone Encoder (Frozen)

**File:** `models/disnet.py` (from OpenScene), loaded in `train_panoptic.py:load_backbone()`

### 4.1 Architecture: DisNet wrapping MinkUNet18A

```
                    DisNet (Frozen, 15.6M params)
                    ═════════════════════════════

  Input: ME.SparseTensor
         feats = vox_feats (V, 3) [x, y, z coordinates]
         coords = [batch_idx, vox_x, vox_y, vox_z] (V, 4)

         ┌────────────────────────────────────────┐
         │           MinkUNet18A                   │
         │                                         │
         │  Encoder:  3→32→64→128→256             │
         │  4 sparse conv blocks with              │
         │  residual connections                   │
         │                                         │
         │  Decoder: 256→128→128→96               │
         │  4 sparse deconv blocks with            │
         │  skip connections from encoder          │
         │                                         │
         │  Final: Conv(96, 768, kernel=1)         │
         │         net3d.final.kernel: [96, 768]   │
         └────────────────────────────────────────┘

  Output: Tensor (V, 768)
          768-d CLIP-space features per voxel
          (NOT a SparseTensor — DisNet extracts .F internally)
```

### 4.2 Key Properties

| Property | Value |
|---|---|
| Input channels | 3 (XYZ coordinates) |
| Output channels | 768 (CLIP embedding dim) |
| Total parameters | 15,575,696 |
| Trainable | 0 (all frozen) |
| Voxel size | 0.05m (must match training) |
| Forward time | ~37ms on A100 |
| GPU memory | ~207MB peak |
| Checkpoint | `models/nuscenes_openseg.pth.tar` |
| State dict prefix | `net3d.*` keys |

### 4.3 Why Frozen?

The backbone was distilled from OpenSeg/CLIP — its features already encode rich semantic information. Fine-tuning would:
1. Destroy CLIP alignment (needed for future open-vocabulary inference)
2. Require massive GPU memory (15.6M params + activations)
3. Risk overfitting on nuScenes dataset

---

## 5. Stage 3: Panoptic Decoder

**File:** `panoptic/models/mask_decoder.py`
**Supporting:** `panoptic/models/blocks.py`, `panoptic/models/positional_encoder.py`

### 5.1 High-Level Architecture

```
  backbone_feats (B,V,768)     coords (B,V,3)
        │                           │
        ├── backbone_proj ──────> src (B,V,256)
        ├── mask_feat_proj ─────> mask_feat (B,V,256)
        │                           │
        │                     pe_layer (sinusoidal)
        │                           │
        │                     pe (B,V,256)
        │                           │
        │                     mask_features = mask_feat + pe  (B,V,256)
        │
        │     ┌── Query Init ──────────────────────────┐
        │     │  thing_feat:  (100, 256) learnable      │
        │     │  stuff_feat:  (6, 256) learnable        │
        │     │  thing_embed: (100, 256) learnable      │
        │     │  stuff_embed: (6, 256) learnable        │
        │     │                                         │
        │     │  output = cat(thing_feat, stuff_feat)   │
        │     │         = (B, 106, 256)                 │
        │     │  query_embed = cat(thing_embed, stuff)  │
        │     │              = (B, 106, 256)            │
        │     └─────────────────────────────────────────┘
        │
        │     pred_heads(output, mask_features) ──> predictions[0]
        │
        │     ┌── 9 Decoder Layers (i=0..8) ───────────┐
        │     │                                         │
        │     │  level = i % 3                          │
        │     │  src_level = src + level_embed[level]   │
        │     │                                         │
        │     │  Cross-Attn(output, src_level, pe)      │
        │     │  Self-Attn(output, query_embed)         │
        │     │  FFN(output)                            │
        │     │  pred_heads(output, mask_features)      │
        │     │       → predictions[i+1]                │
        │     │                                         │
        │     └─────────────────────────────────────────┘
        │
        └──> Final: pred_logits (B,106,17), pred_masks (B,V,106)
             Aux:   9 intermediate predictions (same shapes)
```

### 5.2 Projection Layers

```python
backbone_proj = Sequential(
    Linear(768, 256),    # Project CLIP features to hidden dim
    LayerNorm(256),
)
mask_feat_proj = Sequential(
    Linear(768, 256),    # Separate projection for mask features
    LayerNorm(256),
)
```

**Why two separate projections?** `src` (for cross-attention keys/values) and `mask_features` (for mask dot-product) serve different purposes. Separate projections let each learn its own transformation of the 768-d features.

### 5.3 Positional Encoding

**File:** `panoptic/models/positional_encoder.py`

```
Input: coords (B, V, 3) — xyz in meters

Normalize:
  x /= 50.0   (nuScenes range: [-51.2, 51.2])
  y /= 50.0
  z /= 4.0    (nuScenes range: [-5.0, 3.0])

Sinusoidal encoding:
  num_bands = floor(256 / 3 / 2) = 42  bands per dimension
  scales = logspace(0, log2(2500), 42, base=2)
  For each coordinate dimension:
    sin(coord * scale * pi), cos(coord * scale * pi)

  Concat all: 3 dims * 42 bands * 2 (sin+cos) = 252 features
  Zero-pad left to 256: pad = 256 - 252 = 4

Output: (B, V, 256)
```

### 5.4 Query Design: 100 Thing + 6 Stuff

```
  ┌─── Thing Queries (0-99) ──────────────────────────────┐
  │  100 learnable embeddings                              │
  │  Each can match ANY thing instance of ANY thing class  │
  │  thing_query_feat:  (100, 256) — content               │
  │  thing_query_embed: (100, 256) — positional             │
  │  During matching: compete to cover all instances       │
  └────────────────────────────────────────────────────────┘

  ┌─── Stuff Queries (100-105) ───────────────────────────┐
  │  6 learnable embeddings, one per stuff class           │
  │  Query 100 → driveable_surface (class 10)             │
  │  Query 101 → other_flat (class 11)                    │
  │  ...                                                   │
  │  Query 105 → vegetation (class 15)                    │
  │  stuff_query_feat:  (6, 256) — content                 │
  │  stuff_query_embed: (6, 256) — positional              │
  └────────────────────────────────────────────────────────┘

  Total: Q = 106 queries
```

**Why split?** Stuff classes are non-countable (exactly one instance per class). Dedicated queries prevent thing queries from wasting capacity on stuff prediction.

### 5.5 Decoder Layer Detail

Each of the 9 layers follows this sequence:

```
Layer i (i = 0..8):
  level_idx = i % 3     ← cycles through 3 "virtual" feature levels

  ┌── Cross-Attention ────────────────────────────────┐
  │  Pre-norm: q_embed = LayerNorm(output)             │
  │                                                     │
  │  Query = (q_embed + query_embed)                   │
  │  Key   = (src_level + pe)                          │
  │  Value = (src_level + pe)                          │
  │                                                     │
  │  Masked MHA: attn_mask (B*8, 106, V)              │
  │    Only attend to voxels where sigmoid(mask) >= 0.5 │
  │    This is the "masked cross-attention" from        │
  │    Mask2Former — each query attends only to its     │
  │    predicted foreground region                      │
  │                                                     │
  │  nn.MultiheadAttention(256, 8 heads)               │
  │  head_dim = 256/8 = 32                             │
  │                                                     │
  │  Residual: output = output + dropout(attn_out)     │
  └─────────────────────────────────────────────────────┘

  ┌── Self-Attention ─────────────────────────────────┐
  │  Q = K = (output + query_embed)                    │
  │  Value = output                                    │
  │  No masking (all 106 queries attend to each other) │
  │  Residual + LayerNorm                              │
  └─────────────────────────────────────────────────────┘

  ┌── FFN ────────────────────────────────────────────┐
  │  Pre-norm: LayerNorm(output)                       │
  │  Linear(256 → 1024) → ReLU → Dropout              │
  │  Linear(1024 → 256) → Dropout                     │
  │  Residual: output = output + ffn_out               │
  └─────────────────────────────────────────────────────┘

  ┌── Prediction Heads ───────────────────────────────┐
  │  decoder_norm = LayerNorm(output)     (B,106,256) │
  │  class_logits = Linear(decoder_norm)  (B,106,17)  │
  │  mask_embed = MLP(decoder_norm)       (B,106,256) │
  │  mask_logits = einsum("bqc,bpc->bpq",            │
  │                mask_embed, mask_features)          │
  │              = (B, V, 106)                         │
  │                                                    │
  │  attn_mask = sigmoid(mask_logits) < 0.5           │
  │  Reshape: (B,V,106) → (B*8, 106, V)              │
  └─────────────────────────────────────────────────────┘
```

### 5.6 Level Embeddings (Virtual Multi-Scale)

```
We have single-level features (all from backbone final layer).
To mimic multi-scale (like MaskPLS with 3 FPN levels), we add
3 learnable level embeddings that cycle every 3 layers:

  Layer 0,3,6 → level_embed[0]
  Layer 1,4,7 → level_embed[1]
  Layer 2,5,8 → level_embed[2]

src_level = src + level_embed[level_idx]

This gives the decoder different "views" of the same features,
encouraging diverse attention patterns.
```

### 5.7 Attention Mask Mechanism (Masked Cross-Attention)

```
From Mask2Former: Each query only attends to voxels where its
previous mask prediction is positive (sigmoid > 0.5).

  mask_logits (B, V, Q)
       │
  sigmoid → binary threshold at 0.5
       │
  attn_mask (B, V, Q) bool: True where masked OUT
       │
  Expand for multi-head: (B, 1, V, Q) → (B, 8, V, Q)
       │
  Flatten batch+heads: (B*8, V, Q) → permute → (B*8, Q, V)
       │
  Safety: if ALL voxels masked for a query, unmask all
          (prevents NaN in softmax)

This focuses each query on its relevant spatial region,
dramatically improving efficiency and convergence.
```

### 5.8 Parameter Count

| Component | Parameters |
|---|---|
| backbone_proj (768→256 + LN) | 197,120 + 512 |
| mask_feat_proj (768→256 + LN) | 197,120 + 512 |
| thing_query_feat (100×256) | 25,600 |
| thing_query_embed (100×256) | 25,600 |
| stuff_query_feat (6×256) | 1,536 |
| stuff_query_embed (6×256) | 1,536 |
| level_embed (3×256) | 768 |
| 9× CrossAttn (256, 8 heads) | 9 × 263,168 = 2,368,512 |
| 9× SelfAttn (256, 8 heads) | 9 × 263,168 = 2,368,512 |
| 9× FFN (256→1024→256) | 9 × 525,568 = 4,730,112 |
| decoder_norm (LN 256) | 512 |
| class_embed (256→17) | 4,369 |
| mask_embed MLP (256→256→256, 3 layers) | 131,328 |
| **Total (approximate)** | **~10M trainable** |

---

## 6. Stage 4: Loss Computation

**Files:** `panoptic/losses/panoptic_loss.py`, `panoptic/losses/matcher.py`

### 6.1 Hungarian Matching

```
For each prediction layer (10 total = 1 final + 9 aux):

  pred_logits (B, 106, 17)     targets: gt_classes (M,), gt_masks (M, V)
       │                              │
       ▼                              │
  Cost Matrix C (106, M):            │
  ┌───────────────────────────────────┘
  │
  │  class_cost = -softmax(pred_logits)[:, gt_class]     (106, M)
  │     Negated probability of correct class
  │     Weight: 2.0
  │
  │  Subsample 75% of V voxels (randperm, no replacement)
  │
  │  bce_cost = sigmoid_ce(pred_mask, gt_mask)           (106, M)
  │     Weight: 5.0
  │
  │  dice_cost = 1 - dice(sigmoid(pred_mask), gt_mask)   (106, M)
  │     Weight: 5.0
  │
  │  C = 2.0 * class_cost + 5.0 * bce_cost + 5.0 * dice_cost
  │
  └──> scipy.linear_sum_assignment(C)
       Returns optimal (pred_idx, gt_idx) pairs
       Minimum: min(106, M) matches
```

### 6.2 Classification Loss

```
Cross-Entropy with per-class weights (matching MaskPLS):

  pred: (B, 106, 17) logits
  target: (B, 106) class indices
    - Matched queries → GT class (0-15)
    - Unmatched queries → 16 (no-object)

  weights: [1.0, 1.0, ..., 1.0, 0.1]
                16 real classes    no-obj
                                   (eos_coef)

  loss_ce = F.cross_entropy(pred, target, weight=weights)
  loss_ce *= 2.0  (loss weight)
```

**Why CE instead of Focal Loss?**
- MaskPLS uses CE — proven to work for LiDAR panoptic
- Our earlier sigmoid focal loss with γ=4.0 suppressed loss_ce to ~0.008 (no gradient)
- CE with eos_coef=0.1 gives balanced gradients between real and no-object classes

### 6.3 Mask BCE Loss

```
For each matched (pred_query_i, gt_mask_j) pair:

  pred_mask: (V,) raw logits
  gt_mask:   (V,) binary {0.0, 1.0}

  Point sampling: n_sample = min(50000, V)
    If n_sample >= V: use all points (no sampling)
    Else: randperm(V)[:n_sample] (without replacement)

  BCE = -[gt * log(sigmoid(pred)) + (1-gt) * log(1-sigmoid(pred))]
  Per voxel, averaged across voxels, summed across pairs, / num_masks

  loss_mask *= 5.0  (loss weight)
```

### 6.4 Dice Loss

```
  pred_sig = sigmoid(pred_mask)     (n_sample,) per matched pair

  numerator   = 2 * sum(pred_sig * gt_mask)
  denominator = sum(pred_sig) + sum(gt_mask)
  dice_coeff  = (numerator + 1) / (denominator + 1)  ← Laplace smooth
  dice_loss   = (1 - dice_coeff)

  Summed over all matched pairs, / num_masks

  loss_dice *= 5.0  (loss weight)
```

### 6.5 Loss Summary Diagram

```
  10 prediction layers (1 final + 9 aux)
       │
       │  Each produces:
       │    loss_ce   × 2.0
       │    loss_mask × 5.0
       │    loss_dice × 5.0
       │
       └──> total_loss = sum of all 30 terms

  Naming: "loss_ce", "loss_mask", "loss_dice" (final layer)
          "0_loss_ce", "0_loss_mask", "0_loss_dice" (aux layer 0)
          ...
          "8_loss_ce", "8_loss_mask", "8_loss_dice" (aux layer 8)
```

### 6.6 Why Auxiliary Losses?

```
Without aux losses, only the final (10th) prediction layer gets
direct supervision. Gradients must backprop through 9 transformer
layers to reach early layers — vanishing gradient problem.

With aux losses, EVERY layer gets direct loss signal:
  Layer 0: pred_heads → loss → gradient directly to layer 0 weights
  Layer 1: pred_heads → loss → gradient directly to layer 1 weights
  ...
  Layer 9: pred_heads → loss → gradient directly to layer 9 weights

This is critical for training deep Mask2Former-style decoders.
Each aux layer has its OWN Hungarian matching (may differ from final).
```

---

## 7. Stage 5: Panoptic Inference

**File:** `panoptic/train/train_panoptic.py:panoptic_inference()`

```
Input: pred_logits (106, 17), pred_masks (V, 106) — from final layer

  Step 1: Classification
    scores, labels = softmax(pred_logits).max(-1)    (106,)
    Filter: keep queries where label != 16 (no-object)

  Step 2: Sort by confidence (descending)

  Step 3: Greedy mask assignment (score-sorted)
    For each query (highest score first):
      binary_mask = sigmoid(mask) >= 0.5
      Remove already-assigned voxels
      If remaining / original < overlap_thr (0.8): skip

      If thing class (0-9):
        Assign new instance_id
      If stuff class (10-15):
        Merge with existing stuff instance or create new

  Output:
    sem_pred:  (V,) int — class per voxel (0-indexed)
    inst_pred: (V,) int — instance ID per voxel
```

---

## 8. Stage 6: Evaluation Metrics

**File:** `panoptic/eval/eval_pq.py`

### 8.1 Panoptic Quality (PQ)

```
  PQ = SQ x RQ     (per class, then averaged over 16 classes)

  For each class c:
    Match pred segments to GT segments where IoU > 0.5
    (Each pred/GT can match at most once; segments < 50 points ignored)

    TP = matched pairs
    FP = unmatched predictions (>= min_points)
    FN = unmatched GT segments (>= min_points)

    SQ(c) = mean IoU of TP pairs          ← mask quality
    RQ(c) = 2*TP / (2*TP + FP + FN)       ← detection F1-score
    PQ(c) = SQ(c) * RQ(c)

  PQ = mean(PQ[c] for c in 0..15)
```

### 8.2 Semantic mIoU

```
  Confusion matrix: (16, 16) — predicted vs GT class per voxel
  Per-class IoU = TP / (TP + FP + FN)
  mIoU = mean over 16 classes
```

### 8.3 What Each Metric Tells You

```
  ┌────────┬────────────────────────┬───────────────────────┐
  │ Metric │ Measures               │ Low means...          │
  ├────────┼────────────────────────┼───────────────────────┤
  │ PQ     │ Overall panoptic       │ Bad segmentation OR   │
  │        │ quality                │ bad detection         │
  ├────────┼────────────────────────┼───────────────────────┤
  │ SQ     │ Mask boundary quality  │ Sloppy/wrong masks    │
  │        │ (range 0.5-1.0)       │ for found objects     │
  ├────────┼────────────────────────┼───────────────────────┤
  │ RQ     │ Detection F1-score     │ Missing objects or    │
  │        │                        │ hallucinating objects │
  ├────────┼────────────────────────┼───────────────────────┤
  │ mIoU   │ Semantic accuracy      │ Wrong class labels    │
  │        │ (ignores instances)    │ per voxel             │
  └────────┴────────────────────────┴───────────────────────┘
```

---

## 9. Training Loop & Optimizer

**File:** `panoptic/train/train_panoptic.py`

### 9.1 Optimizer: AdamW

```
AdamW(decoder.parameters(), lr=1e-4, weight_decay=0.01)

Only decoder parameters are optimized (backbone is frozen).
AdamW decouples weight decay from gradient update — better than
Adam+L2 for transformer architectures.
```

### 9.2 Learning Rate Scheduler: Cosine Warmup

```
  LR
  │  1e-4  ─────────────╲
  │  /                     ╲
  │ / warmup                 ╲  cosine decay
  │/  (2 epochs)               ╲
  │                               ╲
  │  1e-6  ─────────────────────────╲───
  └─────────────────────────────────────── epochs
  0     2                            30

  IMPORTANT: steps_per_epoch = len(dataloader) // grad_accum
  This counts OPTIMIZER STEPS (not raw batches), so warmup and
  cosine decay are computed correctly for gradient accumulation.

  warmup_steps = warmup_epochs × steps_per_epoch
  total_steps = epochs × steps_per_epoch
  Steps per lr update: every optimizer step (step-level)

  Alternatives available via YAML:
  - cosine_annealing: CosineAnnealingLR (epoch-level)
  - step: StepLR(step_size=25, gamma=0.1) (epoch-level)
```

### 9.3 Gradient Accumulation & Clipping

```
  grad_accum = 8: accumulate gradients over 8 samples before
                   optimizer step (effective batch size = 8)

  grad_clip = 0.5: clip_grad_norm_ to prevent gradient explosion
                    (common in transformer training)

  Sequence per 8 steps:
    step 0: forward + backward (accumulate)
    step 1: forward + backward (accumulate)
    ...
    step 7: forward + backward → clip_grad_norm → optimizer.step → zero_grad
```

### 9.4 Mixed Precision (AMP)

```
  torch.cuda.amp.autocast():
    Decoder forward + loss computation in float16
    Backbone is already float32 (frozen, no_grad)

  torch.cuda.amp.GradScaler():
    Scales loss before backward to prevent underflow in float16
    Unscales gradients before clipping and optimizer step

  Benefit: ~30-50% memory reduction, ~20% speed improvement
```

### 9.5 Training Loop Structure

```
  for epoch in range(start_epoch, 30):
    for step, batch in dataloader:
      │
      ├── Check kill signal (SIGTERM/SIGUSR1)
      ├── extract_backbone_features (frozen, no_grad)
      ├── decoder forward (AMP autocast)
      ├── PanopticLoss (matching + CE + BCE + Dice)
      ├── backward (scaled if AMP)
      ├── Every grad_accum steps: clip + optimizer.step + scheduler.step
      └── Log to wandb every log_every steps
    │
    ├── Evaluate every eval_every epochs (every 2)
    ├── Save best.pth if PQ improved
    └── Save latest.pth every save_every epochs (every 2)
```

### 9.6 Checkpointing

```
  Saved in checkpoint:
    decoder state_dict
    optimizer state_dict
    scheduler state_dict
    scaler state_dict (if AMP)
    epoch number
    best_pq
    full config dict
    metrics (in best.pth)

  Auto-resume: on startup, checks for latest.pth or interrupted.pth
  Signal handler: SIGTERM/SIGUSR1 → save interrupted.pth → exit
```

---

## 10. Training Arithmetic: Full nuScenes

**Config:** `panoptic/configs/full_nuscenes.yaml`

### 10.1 Dataset Size

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  nuScenes v1.0-trainval                                         │
  │                                                                 │
  │  Total keyframes:   ~28,130 training  +  ~6,019 validation     │
  │                                                                 │
  │  850 scenes × 40 keyframes = 34,000 total                      │
  │    700 scenes → train split ≈ 28,130 keyframes                 │
  │    150 scenes → val split   ≈  6,019 keyframes                 │
  │                                                                 │
  │  vs. mini:  323 train  +  81 val  (10 scenes)                  │
  │  Scale-up:  ~87× more training data                            │
  └─────────────────────────────────────────────────────────────────┘
```

### 10.2 Per-Sample Statistics

```
  Raw LiDAR points:    N ≈ 34,000 per scan
  After range filter:  N' ≈ 30,000
  After voxelization:  V ≈ 20,000-22,000 voxels
  GT segments:         M ≈ 15-30 per scene (thing instances + stuff classes)
```

### 10.3 Iteration & Step Arithmetic

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  Config Values                                                  │
  │    batch_size   = 1       (samples per GPU per step)            │
  │    grad_accum   = 8       (accumulate before optimizer step)    │
  │    epochs       = 30                                            │
  │    num_workers  = 4       (dataloader workers)                  │
  │                                                                 │
  │  Derived Quantities                                             │
  │    effective_batch_size = batch_size × grad_accum = 1 × 8 = 8  │
  │                                                                 │
  │    batches_per_epoch    = 28,130 / 1 = 28,130 forward passes   │
  │    optimizer_steps/epoch = 28,130 / 8 = 3,516 optimizer steps  │
  │                                                                 │
  │    total_batches        = 28,130 × 30 = 843,900 forward passes │
  │    total_optimizer_steps = 3,516 × 30 = 105,480 steps          │
  │                                                                 │
  │  Scheduler Steps                                                │
  │    warmup_epochs = 2                                            │
  │    warmup_steps  = 2 × 3,516 = 7,032 optimizer steps           │
  │    decay_steps   = 105,480 - 7,032 = 98,448 cosine steps       │
  │                                                                 │
  │  LR Schedule                                                    │
  │    Steps 0 → 7,032:     linear ramp 0 → 1e-4                  │
  │    Steps 7,032 → 105,480: cosine decay 1e-4 → 1e-6            │
  └─────────────────────────────────────────────────────────────────┘
```

### 10.4 Augmentation Multiplier

```
  Each epoch sees all 28,130 samples with DIFFERENT augmentation:
    - Random rotation Z:    uniform [0, 2π]    → infinite variants
    - Random flip X:        50% chance
    - Random flip Y:        50% chance
    - Random scale:         uniform [0.95, 1.05]
    - Random translate:     Gaussian σ=0.1m

  Over 30 epochs:
    28,130 × 30 = 843,900 augmented samples seen
    Each unique scene appears 30 times, each differently augmented

  With mini (323 × 100 epochs = 32,300) we had 26× FEWER augmented samples.
  Full training provides 843,900 augmented samples — 26× more gradient signal.
```

### 10.5 Evaluation Cadence

```
  eval_every = 2 epochs
  Evaluations at: epoch 2, 4, 6, 8, ..., 28, 30
  Total evaluations: 15

  Each evaluation:
    6,019 val samples × panoptic_inference → PQ/SQ/RQ/mIoU
    Estimated time: ~15 min per eval (inference-only, no backward)

  Checkpointing:
    save_every = 2 epochs → latest.pth every 2 epochs
    best.pth saved when PQ improves
```

### 10.6 Time & Throughput Estimates

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  Per-sample time (from mini training):                          │
  │    Backbone forward:   ~37 ms                                   │
  │    Decoder forward:    ~15 ms                                   │
  │    Loss + backward:    ~25 ms                                   │
  │    Overhead:           ~10 ms                                   │
  │    ─────────────────────────                                    │
  │    Total per sample:   ~87 ms ≈ 0.09 sec                       │
  │                                                                 │
  │  Per epoch:                                                     │
  │    28,130 × 0.09 sec ≈ 2,532 sec ≈ 42 min                     │
  │                                                                 │
  │  Total training (30 epochs):                                    │
  │    Training:     30 × 42 min = 1,260 min ≈ 21 hours            │
  │    Evaluations:  15 × 15 min = 225 min ≈ 3.75 hours            │
  │    ──────────────────────────                                   │
  │    Estimated total: ~25 hours                                   │
  │                                                                 │
  │  Wall-time budget: 48 hours (bsub -W 48:00)                    │
  │    Comfortable margin for slower epochs, IO, etc.               │
  └─────────────────────────────────────────────────────────────────┘
```

### 10.7 Expected Results (Based on Mini + Literature)

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │  Mini Results (323 train, 25 epochs):                                  │
  │    PQ = 0.228,  SQ = 0.663,  RQ = 0.288,  mIoU = 0.317              │
  │                                                                        │
  │  Full nuScenes Expectations:                                           │
  │    The 87× data increase should significantly improve PQ.              │
  │    Conservative targets for 30 epochs:                                 │
  │      PQ:   0.35 - 0.45  (vs 0.228 on mini)                           │
  │      SQ:   0.70 - 0.80  (mask quality improves with more instances)   │
  │      RQ:   0.45 - 0.55  (detection improves with rare class exposure) │
  │      mIoU: 0.50 - 0.65  (semantic accuracy benefits most from data)   │
  │                                                                        │
  │  Reference: MaskPLS (Mask2Former on full nuScenes LiDAR):             │
  │      PQ = 0.53,  SQ = 0.80,  RQ = 0.64,  mIoU = 0.71                │
  │    Our model uses frozen OpenScene backbone (not jointly trained),      │
  │    so expect ~60-80% of MaskPLS numbers as realistic ceiling.          │
  │                                                                        │
  │  Key improvements from mini → full:                                    │
  │    1. Rare class coverage: bicycle, motorcycle, construction_vehicle   │
  │       had 0 PQ on mini — full dataset has many more instances          │
  │    2. Object diversity: more pose/size/occlusion variation             │
  │    3. Scene diversity: different lighting, weather, locations          │
  │    4. Stuff boundary quality: more examples refine edges              │
  └────────────────────────────────────────────────────────────────────────┘
```

### 10.8 Mini vs Full Comparison Table

```
  ┌───────────────────────┬────────────────────┬─────────────────────┐
  │ Parameter             │ Mini (default.yaml)│ Full (full_nuscenes)│
  ├───────────────────────┼────────────────────┼─────────────────────┤
  │ Dataset version       │ v1.0-mini          │ v1.0-trainval       │
  │ Train samples         │ 323                │ ~28,130             │
  │ Val samples           │ 81                 │ ~6,019              │
  │ Epochs                │ 100                │ 30                  │
  │ Effective batch size  │ 1 × 4 = 4          │ 1 × 8 = 8          │
  │ Batches/epoch         │ 323                │ 28,130              │
  │ Optimizer steps/epoch │ 323 / 4 = 80       │ 28,130 / 8 = 3,516 │
  │ Total optimizer steps │ 80 × 100 = 8,000   │ 3,516 × 30 =105,480│
  │ Warmup steps          │ 2 × 80 = 160       │ 2 × 3,516 = 7,032  │
  │ Augmented samples     │ 32,300              │ 843,900             │
  │ Eval every            │ 5 epochs           │ 2 epochs            │
  │ Log every             │ 10 steps           │ 50 steps            │
  │ num_workers           │ 0                  │ 4                   │
  │ Checkpoint dir        │ panoptic/checkpoints│ panoptic/checkpoints_full│
  │ Est. epoch time       │ ~30 sec            │ ~42 min             │
  │ Est. total time       │ ~50 min            │ ~25 hours           │
  │ Wall-time limit       │ N/A (interactive)  │ 48 hours (bsub)     │
  └───────────────────────┴────────────────────┴─────────────────────┘
```

---

## 11. Dimension Summary Table

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │                     DIMENSION FLOW                                   │
  ├──────────────────────────────────────────────────────────────────────┤
  │                                                                      │
  │  Raw LiDAR:           (N, 5)      N ≈ 34,000 points                │
  │  After range filter:  (N', 5)     N' ≈ 30,000                      │
  │  After voxelization:  (V, 3)      V ≈ 20,000 voxels               │
  │                                                                      │
  │  Backbone input:      (V, 3)      XYZ features                     │
  │  Backbone output:     (V, 768)    CLIP-space features              │
  │                                                                      │
  │  After unsqueeze:     (1, V, 768) batched                          │
  │                                                                      │
  │  backbone_proj:       (1, V, 256) src features                     │
  │  mask_feat_proj:      (1, V, 256) mask features                    │
  │  pe_layer:            (1, V, 256) positional encoding              │
  │  mask_features:       (1, V, 256) mask_feat + pe                   │
  │                                                                      │
  │  Queries:             (1, 106, 256) = 100 thing + 6 stuff          │
  │  query_embed:         (1, 106, 256) positional queries             │
  │                                                                      │
  │  Cross-Attn:          Q(106,1,256) × K(V,1,256) → (106,1,256)     │
  │    attn_mask:         (8, 106, V)  per-head mask                   │
  │                                                                      │
  │  Self-Attn:           Q(106,1,256) × K(106,1,256) → (106,1,256)   │
  │                                                                      │
  │  class_embed:         (1, 106, 17) = 16 classes + 1 no-object      │
  │  mask_embed MLP:      (1, 106, 256)                                │
  │  mask_logits:         (1, V, 106) = einsum(queries, mask_features) │
  │                                                                      │
  │  GT masks:            (M, V)      M ≈ 15-30 segments              │
  │  GT classes:          (M,)        0-indexed                        │
  │  Hungarian match:     (min(106,M),) pairs                          │
  │                                                                      │
  │  Inference output:    sem_pred (V,), inst_pred (V,)                │
  └──────────────────────────────────────────────────────────────────────┘
```

---

## 12. Design Choices & Rationale

### 12.1 Architecture Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Backbone | Frozen DisNet/MinkUNet18A | Preserves CLIP alignment; saves GPU memory |
| Decoder style | Mask2Former (from MaskPLS) | SOTA for panoptic LiDAR; well-tested |
| Query split | 100 thing + 6 stuff | Stuff = non-countable, 1 per class; thing = learnable pool |
| Decoder layers | 9 (3 blocks × 3 levels) | MaskPLS default; good balance of depth vs speed |
| Hidden dim | 256 | Standard for Mask2Former decoders |
| Attention heads | 8 (head_dim=32) | Standard transformer config |
| FFN dim | 1024 (4x hidden) | Standard expansion ratio |
| Masked cross-attn | sigmoid(mask) < 0.5 | Each query focuses on its mask region |
| Two projections | backbone_proj + mask_feat_proj | Separate representations for attention vs mask computation |
| Level embeddings | 3 learnable, cycling | Simulates multi-scale without actual FPN |

### 12.2 Loss Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Classification | Cross-Entropy | Matches MaskPLS; focal loss was too aggressive (loss→0) |
| No-object weight | eos_coef=0.1 | 94/106 queries are unmatched — downweight them |
| Mask loss 1 | Sigmoid BCE | Standard per-voxel binary classification loss |
| Mask loss 2 | Dice | Size-invariant overlap loss; complements BCE |
| Loss weights | CE=2, BCE=5, Dice=5 | From MaskPLS defaults |
| Auxiliary losses | Same weights, all 9 layers | Direct supervision at every layer prevents vanishing gradients |
| Matcher p_ratio | 0.75 | Higher than MaskPLS default for matching stability |
| Point sampling | randperm (no replacement) | Reduces noise vs randint (with replacement) |

### 12.3 Training Decisions

| Decision | Choice (Full) | Rationale |
|---|---|---|
| Optimizer | AdamW (lr=1e-4, wd=0.01) | Standard for transformers; decoupled weight decay |
| Scheduler | Cosine warmup (2 epoch warmup) | Avoids early instability; smooth decay |
| Grad accumulation | 8 steps (eff. batch=8) | Larger effective batch stabilizes training on 28K samples |
| Grad clipping | max_norm=0.5 | Prevents transformer gradient explosion |
| Mixed precision | AMP (float16 forward) | Memory savings; backbone already float32 |
| Epochs | 30 (vs 100 on mini) | ~13× more steps/epoch; 30 epochs yields 105K optimizer steps |
| Eval frequency | Every 2 epochs | ~84 min between evals; frequent enough to track convergence |
| Log frequency | Every 50 steps | ~4.4 sec apart; enough resolution without log spam |
| Voxel size | 0.05m | Must match OpenScene checkpoint; ~20K voxels/scene |
| Augmentation | Rotate Z, flip X/Y, scale, translate | Standard LiDAR augmentation; applied before voxelization |

---

## 13. File Map

```
panoptic/
├── configs/
│   ├── default.yaml              ← Mini config (v1.0-mini, 323 samples)
│   └── full_nuscenes.yaml        ← Full config (v1.0-trainval, ~28K samples)
│
├── dataset/
│   └── nuscenes_panoptic.py      ← Dataset: loading, voxelization, GT mask construction
│                                    Classes: NuScenesPanoptic
│                                    Key: GENERAL_TO_PANOPTIC_16, augment_points()
│
├── models/
│   ├── blocks.py                 ← Transformer building blocks
│   │                                Classes: SelfAttentionLayer, CrossAttentionLayer,
│   │                                         FFNLayer, MLP
│   │                                Note: PyTorch 1.7.1 compatible (manual transpose)
│   │
│   ├── positional_encoder.py     ← 3D sinusoidal positional encoding
│   │                                Class: PositionalEncoder
│   │                                42 frequency bands per dimension
│   │
│   └── mask_decoder.py           ← Main decoder architecture
│                                    Class: PanopticDecoder
│                                    106 queries, 9 layers, prediction heads
│
├── losses/
│   ├── matcher.py                ← Hungarian matching (scipy LSA)
│   │                                Class: HungarianMatcher
│   │                                Functions: _batch_dice_cost, _batch_sigmoid_ce_cost
│   │
│   └── panoptic_loss.py          ← CE + BCE + Dice loss with auxiliary
│                                    Class: PanopticLoss
│
├── eval/
│   └── eval_pq.py                ← PQ/SQ/RQ and mIoU computation
│                                    Class: PanopticEval
│                                    offset=2^32 for segment matching
│
├── train/
│   └── train_panoptic.py         ← Full training loop
│                                    Backbone loading, AMP, wandb, resume,
│                                    signal handler, panoptic_inference()
│
├── checkpoints/                  ← Mini training weights
│   ├── best.pth
│   └── latest.pth
│
├── checkpoints_full/             ← Full training weights
│   ├── best.pth
│   └── latest.pth
│
├── scripts/
│   ├── train_full_job.sh         ← bsub job script for full nuScenes
│   ├── verify_matcher_and_eval.py
│   ├── verify_phase3_semantic_proxy.py
│   ├── inspect_openscene_checkpoint.py
│   └── validate_per_sample_forward.py
│
└── logs/                         ← Training stdout/stderr from bsub
```

---

## 14. Config Reference

### 14.1 Full nuScenes Config (`full_nuscenes.yaml`)

```yaml
data:
  dataroot: /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes
  version: v1.0-trainval
  voxel_size: 0.05
  point_range: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

augmentation:
  enabled: true
  random_rotate_z: true
  random_flip_x: true
  random_flip_y: true
  random_scale_min: 0.95
  random_scale_max: 1.05
  random_translate_std: 0.1

model:
  backbone_ckpt: models/nuscenes_openseg.pth.tar
  backbone_dim: 768
  hidden_dim: 256
  num_classes: 16            # 10 thing + 6 stuff
  num_thing_queries: 100
  num_stuff_classes: 6
  nheads: 8                  # head_dim = 256/8 = 32
  dim_feedforward: 1024      # 4× expansion
  num_blocks: 3
  num_levels: 3
  dropout: 0.0

loss:
  eos_coef: 0.1              # no-object class weight
  weight_ce: 2.0
  weight_mask: 5.0
  weight_dice: 5.0
  cost_class: 2.0            # matcher weights
  cost_mask: 5.0
  cost_dice: 5.0
  num_points: 50000          # max points for mask loss
  p_ratio: 0.75              # matcher subsampling ratio

training:
  epochs: 30
  lr: 1.0e-4                 # peak lr
  weight_decay: 0.01
  grad_clip: 0.5
  grad_accum: 8              # effective batch = 8
  batch_size: 1
  num_workers: 4
  seed: 42
  scheduler: cosine_warmup
  warmup_epochs: 2           # 7,032 warmup steps
  min_lr: 1.0e-6             # cosine floor
  mixed_precision: true

logging:
  log_every: 50
  eval_every: 2
  wandb_project: openscene-panoptic
  wandb_run_name: full-nuscenes-v1

checkpoint:
  save_dir: panoptic/checkpoints_full
  save_every: 2
  auto_resume: true

inference:
  overlap_thr: 0.8
  min_points: 50
```

### 14.2 Mini Config Differences (`default.yaml`)

```yaml
# Only showing values that differ from full_nuscenes.yaml:
data:
  dataroot: .../nuscenes_mini
  version: v1.0-mini

training:
  epochs: 100                # more epochs to compensate for tiny dataset
  grad_accum: 4              # smaller effective batch (4 vs 8)
  num_workers: 0             # single-threaded for tiny dataset

logging:
  log_every: 10              # log more often (fewer steps/epoch)
  eval_every: 5

checkpoint:
  save_dir: panoptic/checkpoints
  save_every: 5
```
