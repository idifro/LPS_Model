# OpenScene: End-to-End Architecture & Pipeline

## Overview

OpenScene performs open-vocabulary 3D scene understanding by distilling 2D vision-language features (from OpenSeg/CLIP) into 3D point cloud representations. The system operates in three modes: **feature fusion**, **distillation**, and **ensemble**.

---

## 1. Model Architecture

### DisNet (`models/disnet.py`)
A thin wrapper around MinkUNet:
```python
class DisNet(nn.Module):
    def __init__(self, cfg):
        self.net3d = MinkUNet18A(in_channels=3, out_channels=768)  # 768 for OpenSeg
    def forward(self, sparse_3d):
        return self.net3d(sparse_3d)
```

### MinkUNet (`models/mink_unet.py`)
A 3D sparse convolutional U-Net using MinkowskiEngine:

**Encoder** (4 stages):
- Each stage: stride-2 3D sparse convolution -> BN -> ReLU -> residual blocks
- PLANES for MinkUNet18A: (32, 64, 128, 256, 128, 128, 96, 96)
- First 4 values = encoder channels, last 4 = decoder channels

**Decoder** (4 stages):
- Each stage: transposed convolution (upsample) -> concatenate skip connection -> residual blocks

**Final layer**: 1x1 convolution projecting to `out_channels` (768 for distillation, num_classes for supervised)

**Input**: SparseTensor with coordinates (N, 4) [batch_idx + xyz] and features (N, 3) [colors or zeros]
**Output**: Dense feature matrix (N, out_channels)

---

## 2. Loss Functions for Distillation

Defined in `run/distill.py`:

### Cosine Similarity Loss (default)
```python
loss = (1 - CosineSimilarity()(output_3d, feat_3d)).mean()
```
Minimizes angular distance between predicted 3D features and target fused 2D features.

### L1 Loss (alternative)
```python
loss = L1Loss()(output_3d, feat_3d)
```
Minimizes absolute difference between predicted and target features.

The target `feat_3d` is the pre-computed multi-view fused OpenSeg feature for each point.

---

## 3. How OpenSeg Works on Images

Implemented in `scripts/feature_fusion/fusion_util.py`, function `extract_openseg_img_feature()`:

1. Load raw image bytes
2. Feed to TensorFlow SavedModel: `openseg_model.signatures['serving_default'](inp_image_bytes=..., inp_text_emb=zeros(1,1,768))`
3. Extract `ppixel_ave_feat` -- per-pixel 768-dim CLIP-aligned features (region-averaged)
4. Crop to valid image region using `image_info`
5. Resize to target size (800x450 for nuScenes) via nearest-neighbor
6. Result: tensor of shape **(768, H, W)** -- dense per-pixel language-aligned features

OpenSeg produces features that live in the same embedding space as CLIP text features, enabling open-vocabulary classification via dot product with arbitrary text prompts.

---

## 4. The Three Modes

### Mode 1: Feature Fusion

**Pipeline** (`scripts/feature_fusion/nuscenes_openseg.py`):
1. For each scene, load the 3D point cloud
2. For each of 6 cameras (front, front_left, front_right, back, back_left, back_right):
   - Project 3D points to 2D via camera intrinsics/extrinsics
   - Extract OpenSeg features from the image
   - Assign pixel features to visible 3D points
3. Average features across all views where a point is visible
4. Save: `{feat: (M, 768), mask_full: (N,) bool}` per scene

**Point-to-Image Mapping** (`PointCloudToImageMapper`):
- Transform: `p_cam = K @ (world_to_camera @ p_world)`
- Filter: depth > 0, within image bounds, optional occlusion check
- Returns: (N, 3) mapping with (row, col, visibility)

**At evaluation**: directly use fused features as point descriptors. No neural network inference needed.

### Mode 2: Distillation

**Goal**: Train a 3D network (DisNet) to predict the fused features from raw 3D geometry alone, enabling inference without 2D images at test time.

**Training**: DisNet takes 3D point cloud -> predicts 768-dim features -> supervised by cosine loss against fused features.

**At evaluation**: run DisNet on the point cloud to get per-point features.

### Mode 3: Ensemble

Combines fusion and distillation at test time:
1. Compute both fusion features and distillation features
2. L2-normalize both
3. Compute classification logits for both: `logits = features @ text_features.T`
4. Per-point, select the source with **higher max confidence**:
   ```python
   mask = pred_distill.max(dim=-1)[0] < pred_fusion.max(dim=-1)[0]
   feat_ensemble[mask] = feat_fusion[mask]  # fusion wins
   ```
5. Classify ensemble features

Rationale: Fusion is accurate where multi-view coverage is good; distillation fills gaps where points lack image coverage.

---

## 5. Evaluation Script (`run/evaluate.py`)

### Setup
1. Load config, build DisNet model (or skip for fusion-only)
2. Load `FusedFeatureLoader` dataset in eval mode
3. Pre-compute CLIP text features for all class labels:
   - Optionally use prompt engineering: `"a {class_name} in a scene"`
   - For nuScenes: can use 43 detailed sub-labels mapped to 16 classes

### Per-Scene Evaluation Loop
```
For each scene:
  1. Load (coords, feats, labels, feat_3d, mask) from FusedFeatureLoader
  2. Repeat `test_repeats` times (to average voxelization randomness):
     a. Create SparseTensor
     b. Based on feature_type:
        - fusion: predictions = feat_3d[inds_reverse]
        - distill: predictions = model(sinput)[inds_reverse]
        - ensemble: combine both (confidence-based selection)
     c. Classify: logits = predictions @ text_features.T; pred = argmax(logits)
     d. Accumulate predictions
  3. Final prediction = mode across repeats
  4. Update confusion matrix
```

### Metrics (`util/metric.py`)
- Confusion matrix (C x C)
- Per-class IoU = TP / (TP + FP + FN)
- mIoU = mean of per-class IoUs
- Handles `UNKNOWN_ID=255` (ignored in eval), `NO_FEATURE_ID=256` (no feature assigned)

### Visualization
- `export_pointcloud()`: saves .ply colored by predicted labels
- `plot_camera_views()`: projects predictions onto 6 camera images, creates 1x3 front/back grids with legend

---

## 6. Training Script (`run/distill.py`) -- Full Pipeline

### A. Data Loading

**FusedFeatureLoader** (`dataset/feature_loader.py`):
- Loads `.pth` files: `(locs [N,3], feats [N,3], labels [N])`
- Loads fused features `.pt`: `{feat: (M, 768), mask_full: (N,) bool}`
- Applies Voxelizer (voxel_size=0.05m): quantizes coordinates to grid
- Re-indexes fused features to match voxelized point ordering
- Returns: `(coords [V,4], feats [V,3], labels [V], feat_3d [V,768], mask [V])`
  - `mask`: indicates which voxelized points have valid fused features

**Voxelizer** (from `dataset/voxelizer.py`):
- Quantizes: `coords_int = floor(coords / voxel_size)`
- Random offset within voxel for augmentation
- Deduplicates overlapping points (keeps first or random)

**Augmentation**:
- Elastic distortion
- Random horizontal flip
- Chromatic jitter (on color features)
- Random voxel offset

**Collation** (`dataset/point_loader.py`, `collation_fn`):
- Batches variable-length point clouds by prepending batch index to coordinates

### B. Model Setup
```python
model = DisNet(cfg)  # MinkUNet18A, in=3, out=768
optimizer = Adam(model.parameters(), lr=0.0001)
scheduler = PolyLR (power=0.9)
```

### C. Training Loop (per epoch)
```python
for (coords, feat, label_3d, feat_3d, mask) in train_loader:
    # 1. Random coordinate offset (augmentation)
    coords[:, 1:] += (torch.rand(3) * 100).int()

    # 2. Create sparse tensor
    sinput = SparseTensor(feat.float(), coords.int(), device)

    # 3. Forward pass
    output_3d = model(sinput)  # (V, 768)

    # 4. Mask to valid points (those with fused features)
    output_3d = output_3d[mask]
    feat_3d = feat_3d[mask]

    # 5. Compute loss
    loss = (1 - CosineSimilarity()(output_3d, feat_3d)).mean()

    # 6. Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
```

### D. Validation (during training)
- Runs model on `Point3DLoader` (no fused features needed)
- Classifies: `pred = argmax(output @ text_features.T)`
- Computes mIoU against ground truth labels
- Saves best checkpoint by mIoU

### E. Backpropagation Details
- Gradient flows through: cosine loss -> MinkUNet final layer -> decoder stages -> encoder stages
- Only the 3D network is trained; OpenSeg/CLIP are frozen (features pre-computed)
- The target (fused features) is fixed -- pure regression/alignment objective

---

## 7. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE FUSION (offline)                       │
│                                                                   │
│  3D Points ──┐                                                   │
│              ├──> Project to 2D ──> Sample OpenSeg features      │
│  Camera K,E ─┘       per view         per visible point          │
│                                              │                    │
│                                    Average across views           │
│                                              │                    │
│                                    Save: feat(M,768) + mask(N)   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DISTILLATION (training)                        │
│                                                                   │
│  3D Points ──> Voxelize ──> SparseTensor ──> MinkUNet18A         │
│                                                    │              │
│                                              output (V, 768)     │
│                                                    │              │
│  Fused Features (target) ────────> Cosine Loss <───┘              │
│                                        │                          │
│                                   Backprop → Update weights       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION / INFERENCE                         │
│                                                                   │
│  3D Points ──> Model ──> features (N, 768)                       │
│                                    │                              │
│  Text Labels ──> CLIP ──> text_features (C, 768)                 │
│                                    │                              │
│               logits = features @ text_features.T                │
│               pred = argmax(logits, dim=-1)                      │
│                                    │                              │
│                          mIoU evaluation                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Configuration (`config/nuscenes/ours_openseg.yaml`)

```yaml
DATA:
  dataset: nuscenes
  data_root: data/nuscenes_3d          # .pth point clouds
  data_root_2d_fused_feature: data/nuscenes_multiview_openseg  # fused .pt files
  feature_2d_extractor: openseg
  classes: 16
  voxel_size: 0.05                     # 5cm voxels
  num_workers: 4

DISTILL:
  arch_3d: MinkUNet18A
  loss_type: cosine
  base_lr: 0.0001
  epochs: 100
  batch_size: 16

TEST:
  feature_type: ensemble               # fusion | distill | ensemble
  map_nuscenes_details: True           # use 43 detailed labels → 16 classes
  mark_no_feature_to_unknown: True     # points without features → "unknown"
  test_repeats: 5                      # average over random voxelizations
```

---

## 9. Key Insights

1. **No 2D network training**: OpenSeg is used as a frozen feature extractor. All training is 3D-only.
2. **Open vocabulary**: Classification is done via dot product with arbitrary CLIP text embeddings -- no fixed label set at training time.
3. **Voxelization randomness**: Repeated evaluation with different random offsets improves robustness.
4. **Sparse convolutions**: MinkowskiEngine enables efficient processing of large-scale 3D point clouds (millions of points) by only computing at occupied voxels.
5. **Multi-view aggregation**: Averaging OpenSeg features across camera views provides robust dense features even for partially occluded points.

---

## 10. File Reference

| Component | Path |
|-----------|------|
| DisNet model | `models/disnet.py` |
| MinkUNet | `models/mink_unet.py` |
| Distill training | `run/distill.py` |
| Evaluation (3 modes) | `run/evaluate.py` |
| Point dataloader | `dataset/point_loader.py` |
| Fused feature dataloader | `dataset/feature_loader.py` |
| Feature fusion script | `scripts/feature_fusion/nuscenes_openseg.py` |
| Fusion utilities + OpenSeg | `scripts/feature_fusion/fusion_util.py` |
| Metrics | `util/metric.py` |
| Config | `config/nuscenes/ours_openseg.yaml` |
| Label constants | `dataset/label_constants.py` |
