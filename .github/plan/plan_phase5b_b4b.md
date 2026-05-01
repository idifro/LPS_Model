# Phase 5b (B4b) — Camera Feature Concatenation Implementation Plan

## Context

Phase 4 achieved PQ=0.473 using only LiDAR backbone features (frozen DisNet, 768-d).
Camera images contain rich semantic information (texture, color, fine structure) that
LiDAR alone misses. This experiment concatenates frozen CLIP ViT-L/14@336px camera
patch features with backbone features before the decoder projection, testing whether
2D visual features improve 3D panoptic segmentation quality.

**Prerequisite**: Phase 5a (B2b) training and evaluation must be complete first.

---

## Architecture

### Data Flow

```
6 camera images (900×1600)
    │
    ▼
Frozen CLIP ViT-L/14@336px
    │
    ▼
(6, 576, 768) patch tokens → reshape → (6, 24, 24, 768) patch feature maps
    │
    │    LiDAR voxels (V, 3) xyz in lidar frame
    │        │
    │        ├── lidar_sensor2ego (4×4)
    │        ├── lidar_ego2global (4×4)
    │        ├── inv(cam_ego2global) (4×4)
    │        ├── inv(cam_sensor2ego) (4×4)
    │        ▼
    │    Point in camera frame (x_c, y_c, z_c)
    │        │
    │        ├── u = fx * x_c/z_c + cx
    │        ├── v = fy * y_c/z_c + cy
    │        ├── valid = z_c > 0 AND 0 ≤ u < W AND 0 ≤ v < H
    │        ▼
    │    Pixel (u, v) → patch grid (u/14, v/14)
    │        │
    ▼        ▼
Bilinear sample from (24, 24, 768) at patch coords
    │
    ▼
cam_feats (V, 768)  [mean over visible cameras per voxel]
    │
    │    backbone_feats (V, 768)  [frozen DisNet]
    │        │
    ▼        ▼
torch.cat → (V, 1536)
    │
    ├── backbone_proj(1536 → 256) → src
    ├── mask_feat_proj(1536 → 256) → mask_features
    │
    ▼
Decoder (unchanged 256-d, 106 queries, 9 layers)
    │
    ├── pred_logits (106, 17)    [standard 16-class classification]
    ├── pred_masks  (V, 106)     [mask predictions]
    │
    ▼
Standard panoptic inference (softmax → greedy assign)
```

### Key Dimensions

| Tensor | Shape | Notes |
|--------|-------|-------|
| Camera images (raw) | 6 × (900, 1600, 3) | nuScenes surround view |
| Camera images (CLIP input) | (6, 3, 336, 336) | Resized + normalized |
| Patch features | (6, 24, 24, 768) | 576 patches per image |
| Per-voxel camera features | (V, 768) | After projection + sampling |
| Backbone features | (V, 768) | Frozen DisNet |
| Combined features | (V, 1536) | Concatenation |
| `backbone_proj` input | 1536 | Was 768 in Phase 4 |
| `mask_feat_proj` input | 1536 | Was 768 in Phase 4 |
| Decoder hidden | 256 | Unchanged |

---

## Files to Create (6 new)

### 1. `panoptic/models/camera_feature_extractor.py` — NEW

Frozen CLIP ViT-L/14@336px encoder + 3D→2D projection + bilinear sampling.

```python
class CameraFeatureExtractor(nn.Module):
    """Frozen CLIP ViT-L/14@336px → per-voxel camera features via 3D→2D projection."""

    def __init__(self, clip_model="ViT-L/14@336px", device="cuda"):
        # Load frozen CLIP visual encoder
        # Register hook on visual transformer to extract patch tokens
        # Freeze all parameters
        # patch_grid = 24, feat_dim = 768

    @torch.no_grad()
    def extract_patch_features(self, images):
        """(6, 3, 336, 336) → (6, 24, 24, 768)"""
        # Preprocess: resize to 336×336, CLIP normalization
        # Forward through CLIP visual encoder
        # Extract patch tokens (before final pooling) via hook
        # Reshape: (6, 576, 768) → (6, 24, 24, 768)

    def project_and_sample(self, voxel_xyz, patch_features, calibration):
        """(V, 3) + (6, 24, 24, 768) + calib → (V, 768)"""
        # For each of 6 cameras:
        #   1. Transform voxel coords: lidar → ego → global → cam_ego → cam_sensor
        #   2. Project to pixel: intrinsics
        #   3. Filter valid: z > 0, 0 ≤ u < W, 0 ≤ v < H
        #   4. Convert pixel to patch grid: u/14, v/14
        #   5. Bilinear sample from (24, 24, 768) using grid_sample
        # For each voxel: pick camera with smallest positive depth
        # Voxels not seen by any camera → zero vector

    def forward(self, cam_images_or_paths, voxel_xyz, calibration):
        """Full pipeline: images → patch features → project → sample → (V, 768)"""
```

**Key implementation details**:
- CLIP hook: Register forward hook on `model.visual.transformer.resblocks[-1]` to capture patch tokens before the final layer norm + projection
- Multi-camera fusion: Each voxel picks the closest camera (smallest positive depth). Alternative: mean over all visible cameras
- Batch processing: Process all 6 images in one forward pass through CLIP (`batch_size=6`)
- Memory: ViT-L/14 is ~1.2GB frozen; patch features are 6×24×24×768 = 6.3MB per sample

### 2. `panoptic/configs/full_nuscenes_b4b.yaml` — NEW

```yaml
data:
  dataroot: /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes
  version: v1.0-trainval
  voxel_size: 0.05
  point_range: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
  return_calibration: true        # NEW: return camera calibration data

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
  input_dim: 1536                 # NEW: backbone(768) + camera(768)
  hidden_dim: 256
  num_classes: 16
  num_thing_queries: 100
  num_stuff_classes: 6
  nheads: 8
  dim_feedforward: 1024
  num_blocks: 3
  num_levels: 3
  dropout: 0.0
  use_token_head: false

camera:                           # NEW section
  enabled: true
  clip_model: "ViT-L/14@336px"
  image_size: 336
  feat_dim: 768
  fusion: closest                 # closest camera per voxel (vs "mean")
  dropout: 0.0                    # probability of zeroing camera features during training

loss:
  focal_gamma: 4.0
  focal_alpha: 0.25
  eos_coef: 0.1
  weight_ce: 2.0
  weight_mask: 5.0
  weight_dice: 5.0
  cost_class: 2.0
  cost_mask: 5.0
  cost_dice: 5.0
  num_points: 50000
  num_mask_pts: 500
  p_ratio: 0.75

training:
  epochs: 20
  lr: 5.0e-5
  weight_decay: 0.01
  grad_clip: 0.5
  grad_accum: 8
  batch_size: 1
  num_workers: 4
  seed: 42
  scheduler: cosine_warmup
  warmup_epochs: 2
  min_lr: 1.0e-6
  cosine_T_max: 20
  cosine_eta_min: 1.0e-6
  mixed_precision: true

warm_start:
  checkpoint: panoptic/checkpoints_full/best.pth   # Phase 4 PQ=0.473
  strict: false           # backbone_proj shape mismatch (768→1536), will reinitialize

logging:
  log_every: 50
  eval_every: 2
  wandb_project: openscene-panoptic
  wandb_entity: null
  wandb_enabled: true
  wandb_run_name: full-nuscenes-b4b-camera-concat

checkpoint:
  save_dir: panoptic/checkpoints_b4b
  save_every: 2
  resume: null
  auto_resume: true

inference:
  overlap_thr: 0.8
  min_points: 50
```

### 3. `panoptic/train/train_panoptic_camera.py` — NEW

Based on `train_panoptic.py`. Key changes:

```python
# Init camera encoder (frozen, alongside backbone)
cam_extractor = CameraFeatureExtractor(cfg["camera"]["clip_model"]).to(device).eval()

# In training loop:
feats, coords = extract_backbone_features(backbone, batch, device)  # (1, V, 768)

# Extract camera features
t_cam = time.time()
cam_feats = cam_extractor(
    batch["cam_paths"],
    batch["vox_feats"].to(device),   # (V, 3) lidar XYZ
    batch["calibration"],
)  # (V, 768)
cam_time_ms = (time.time() - t_cam) * 1000

# Concatenate
combined = torch.cat([feats[0], cam_feats], dim=-1).unsqueeze(0)  # (1, V, 1536)

# Forward through decoder (backbone_proj now takes 1536)
outputs = decoder(combined, coords)

# Loss is identical to Phase 4
```

**Wandb logging**: All standard metrics + `train/cam_time_ms` per step.

**Camera dropout**: With probability `camera.dropout`, replace `cam_feats` with zeros during training. This prevents the decoder from over-relying on camera features and improves robustness when cameras are occluded or unavailable.

### 4. `panoptic/scripts/evaluate_b4b.py` — NEW

Based on `evaluate_comprehensive.py`. Must load CameraFeatureExtractor for forward pass.

Modes:
- `linear`: Standard softmax inference (with camera features in forward pass)
- `linear_no_cam`: Same checkpoint but zero out camera features (ablation)

### 5. `panoptic/scripts/visualize_b4b.py` — NEW

Based on `visualize_panoptic.py` + camera forward pass. Outputs:
- Per-camera panoptic overlays with legends (same as `visualize_panoptic.py`)
- PLY files (semantic, panoptic, GT, instance)
- BEV comparison
- **Camera visibility heatmap**: Per-camera image showing which projected LiDAR points received camera features (green = visible, red = no valid projection)

### 6. `docs/model_design/model_arch_b4b.md` — NEW

14-section architecture doc (same structure as `model_arch_v1.md` and `model_arch_b2b.md`).

---

## Files to Modify (2, backward compatible)

### 1. `panoptic/models/mask_decoder.py`

Add `input_dim=None` parameter to `__init__`:

```python
def __init__(self, ..., input_dim=None):
    ...
    proj_input_dim = input_dim if input_dim is not None else backbone_dim
    self.backbone_proj = nn.Sequential(
        nn.Linear(proj_input_dim, hidden_dim),  # 1536→256 (was 768→256)
        nn.LayerNorm(hidden_dim),
    )
    self.mask_feat_proj = nn.Sequential(
        nn.Linear(proj_input_dim, hidden_dim),  # 1536→256 (was 768→256)
        nn.LayerNorm(hidden_dim),
    )
```

**Backward compatible**: When `input_dim` is None (default), uses `backbone_dim=768`.
All existing configs/checkpoints work unchanged.

### 2. `panoptic/dataset/nuscenes_panoptic.py`

Add `return_calibration=False` parameter and `_get_calibration()` method:

```python
class NuScenesPanoptic(Dataset):
    def __init__(self, ..., return_calibration=False):
        ...
        self.return_calibration = return_calibration

    def __getitem__(self, idx):
        ...  # existing code unchanged
        result = { ... existing dict ... }

        if self.return_calibration:
            result["calibration"] = self._get_calibration(sample)

        return result

    def _get_calibration(self, sample):
        """Return camera + lidar calibration for 3D→2D projection."""
        from pyquaternion import Quaternion
        lidar_sd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        lidar_cs = self.nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        ego_lidar = self.nusc.get("ego_pose", lidar_sd["ego_pose_token"])

        cameras = {}
        for cam_name in CAMERAS:
            cam_sd = self.nusc.get("sample_data", sample["data"][cam_name])
            cam_cs = self.nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
            ego_cam = self.nusc.get("ego_pose", cam_sd["ego_pose_token"])
            cameras[cam_name] = {
                "intrinsic": np.array(cam_cs["camera_intrinsic"], dtype=np.float32),   # (3, 3)
                "rotation": Quaternion(cam_cs["rotation"]).rotation_matrix.astype(np.float32),  # (3, 3)
                "translation": np.array(cam_cs["translation"], dtype=np.float32),      # (3,)
                "ego_rotation": Quaternion(ego_cam["rotation"]).rotation_matrix.astype(np.float32),
                "ego_translation": np.array(ego_cam["translation"], dtype=np.float32),
                "image_size": (cam_sd["height"], cam_sd["width"]),                     # (H, W)
            }
        return {
            "cameras": cameras,
            "lidar_rotation": Quaternion(lidar_cs["rotation"]).rotation_matrix.astype(np.float32),
            "lidar_translation": np.array(lidar_cs["translation"], dtype=np.float32),
            "ego_lidar_rotation": Quaternion(ego_lidar["rotation"]).rotation_matrix.astype(np.float32),
            "ego_lidar_translation": np.array(ego_lidar["translation"], dtype=np.float32),
        }
```

**Backward compatible**: `return_calibration=False` by default. Existing callers see no change.

---

## 3D→2D Projection Detail

The nuScenes coordinate transform chain:

```
Lidar point p_L (3,) in lidar sensor frame
    → p_ego_L = R_lidar @ p_L + t_lidar         (lidar sensor → ego at lidar time)
    → p_global = R_ego_L @ p_ego_L + t_ego_L     (ego at lidar time → global)
    → p_ego_C = R_ego_C_inv @ (p_global - t_ego_C) (global → ego at camera time)
    → p_cam = R_cam_inv @ (p_ego_C - t_cam)       (ego at camera time → camera sensor)

Where:
    R_lidar, t_lidar    = calibrated_sensor for LIDAR_TOP
    R_ego_L, t_ego_L    = ego_pose at LIDAR_TOP timestamp
    R_ego_C, t_ego_C    = ego_pose at camera timestamp
    R_cam, t_cam         = calibrated_sensor for camera

Then project:
    u = fx * p_cam[0] / p_cam[2] + cx
    v = fy * p_cam[1] / p_cam[2] + cy
    valid = (p_cam[2] > 0) AND (0 ≤ u < W) AND (0 ≤ v < H)

Convert to patch grid:
    patch_u = u * (336 / W) / 14    # rescale to CLIP input then to patch
    patch_v = v * (336 / H) / 14

Bilinear sample from patch_features[cam_idx] at (patch_u, patch_v)
```

For multi-camera fusion per voxel:
- **Option A (recommended)**: Pick camera with smallest positive depth `p_cam[2]`
- **Option B**: Mean across all valid cameras

---

## Warm-Start Strategy

Phase 4 checkpoint has:
- `backbone_proj.0.weight`: shape (256, 768)
- `mask_feat_proj.0.weight`: shape (256, 768)

B4b needs:
- `backbone_proj.0.weight`: shape (256, 1536)
- `mask_feat_proj.0.weight`: shape (256, 1536)

Using `strict=False`:
- These 2 projection layers fail to load → **randomly reinitialized** (Xavier default)
- All other weights load successfully: queries, decoder layers (cross-attn, self-attn, FFN), class_embed, mask_embed, layer norm, level embed
- The 2 projection layers should converge within 2 warmup epochs (they're just linear layers)

---

## Memory & Speed Estimate

| Component | Memory | Time per sample |
|-----------|--------|-----------------|
| Backbone (frozen DisNet) | ~207 MB | ~50 ms |
| Camera encoder (frozen ViT-L/14) | ~1.2 GB | ~200 ms (6 images) |
| Projection + bilinear sampling | ~50 MB | ~20 ms |
| Decoder (with 1536-d proj) | ~44 MB | ~80 ms |
| Per-sample activations | ~500 MB | — |
| **Total** | **~3-4 GB** | **~350 ms** |

Fits easily on A100 (80 GB). Training time estimate: ~25 hours for 20 epochs.

---

## Implementation Steps

| Step | Task | File | Effort |
|------|------|------|--------|
| 1 | Add `input_dim` to PanopticDecoder | `mask_decoder.py` | 15 min |
| 2 | Add `return_calibration` to dataset | `nuscenes_panoptic.py` | 45 min |
| 3 | Create CameraFeatureExtractor | `camera_feature_extractor.py` | 2 hr |
| 4 | Create B4b config | `full_nuscenes_b4b.yaml` | 15 min |
| 5 | Create camera training script | `train_panoptic_camera.py` | 2 hr |
| 6 | Create B4b eval script | `evaluate_b4b.py` | 1 hr |
| 7 | Create B4b viz script | `visualize_b4b.py` | 1 hr |
| 8 | Create B4b arch doc | `model_arch_b4b.md` | 1 hr |
| 9 | Create HPC job script | `train_b4b_job.sh` | 15 min |
| 10 | Smoke test mini | User runs | ~10 min |
| 11 | Full training | User runs | ~25 hr |
| 12 | Evaluate + visualize | User runs | ~3 hr |

---

## Verification Plan

### Smoke test (mini, 3 epochs)

```bash
python -u panoptic/train/train_panoptic_camera.py \
    --config panoptic/configs/full_nuscenes_b4b.yaml \
    --override data.version=v1.0-mini \
    data.dataroot=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes_mini \
    training.epochs=3 \
    logging.eval_every=1 \
    logging.wandb_enabled=false \
    checkpoint.save_dir=panoptic/checkpoints_b4b_mini
```

**Check**:
- Camera features extracted without error (~200ms/sample)
- Combined features shape is (1, V, 1536)
- Loss is decreasing
- No OOM
- `train/cam_time_ms` logged

### Full training

```bash
bsub < panoptic/scripts/train_b4b_job.sh
```

### Evaluation

```bash
python -u panoptic/scripts/evaluate_b4b.py \
    --config panoptic/configs/full_nuscenes_b4b.yaml \
    --checkpoint panoptic/checkpoints_b4b/best.pth \
    --version v1.0-trainval
```

### Visualization

```bash
python -u panoptic/scripts/visualize_b4b.py \
    --checkpoint panoptic/checkpoints_b4b/best.pth \
    --version v1.0-trainval \
    --dataroot /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes \
    --num_samples 5
```

---

## Expected Results

| Metric | Expected | Reasoning |
|--------|----------|-----------|
| PQ | 0.48 – 0.51 | Camera features add texture/color info, should improve over Phase 4 (0.473) |
| SQ | ≥ Phase 4 | Better features → more precise masks |
| RQ | ≥ Phase 4 | Better features → fewer misclassifications |
| Camera ablation (zero cam) | ~0.35 – 0.42 | Without camera, proj layers are randomly initialized, should underperform |

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Camera feature extraction too slow | Training time doubles | Precompute and cache camera features to disk |
| Projection errors (coordinate frame mismatch) | Garbage camera features | Validate projection against `project_points_to_camera()` which is known-correct |
| OOM with ViT-L/14 + decoder | Training fails | Camera dropout, reduce num_workers, gradient checkpointing |
| Camera features dominate backbone | Backbone learning ignored | Camera dropout regularization (e.g., 20% zero-out) |
| Warm-start proj layers too slow to converge | Poor early training | Extended warmup (3 epochs), or pad old weights with zeros instead of reinit |

---

## Critical Files Reference

| File | Purpose | Status |
|------|---------|--------|
| `panoptic/models/mask_decoder.py` | Add `input_dim` param | To modify |
| `panoptic/dataset/nuscenes_panoptic.py` | Add `return_calibration` | To modify |
| `panoptic/scripts/visualize_lidar_camera_overlay.py` | Contains `project_points_to_camera()` — reference for projection logic | Read-only |
| `panoptic/train/train_panoptic.py` | Base training script pattern | Read-only |
| `panoptic/scripts/evaluate_comprehensive.py` | Eval script pattern | Read-only |
| `panoptic/scripts/visualize_panoptic.py` | Viz script pattern | Read-only |
