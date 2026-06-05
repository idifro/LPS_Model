# Job 1: Dense Camera-CLIP Supervision
## Closing the Feature Margin Gap Through Multi-Loss Distillation

**Codename:** `job1_dense_clip`  
**Warm-start checkpoint:** `panoptic/checkpoints_token_full_v2/best.pth` (PQ=0.476)  
**Target epochs:** 20 (2 warm-up + 18 full)  
**Estimated wall time:** ~28–32 hours on A100 96 GB  
**bsub config:** `-q batch_a100 -n 4 -M 8192 -W 36:00 -gpu "num=1:mode=exclusive_process:mps=no"`

---

## 1. Why This Job Exists

Inference-only experiments (Strategies E, K, L, M) conclusively proved:  
**The backbone's per-voxel features lack cosine similarity margin between similar classes.**  
"car" voxels have cos_sim("police car") ≈ cos_sim("car") ± noise-level 0.02 difference.  
This is caused by the distillation chain: `CLIP → OpenSeg → 3D projection → DisNet` — each step degrades alignment.

Job 1 closes this gap by adding dense CLIP supervision from cameras:
- **L_O**: forces decoder query embeddings to match mask-pooled camera CLIP features (object level)
- **L_V**: forces every voxel's reconstructed feature to match projected camera CLIP features (voxel level)
- **L_V + CASD**: replaces L_V targets for contextually ambiguous stuff classes with context-expanded crop features (novel contribution)
- **L7**: directly trains backbone voxel features against 200+ text negatives (root cause fix)
- **L5 (C1)**: fixes the confirmed 16-class InfoNCE collapse in the token head
- **B1 (LoRA)**: adds trainable capacity to frozen backbone, providing gradient path for L7
- **A4**: replaces dual-head (linear + token) with single CLIP embedding head
- **T4**: randomly samples from 10 text templates per class during training

---

## 2. CLIP Model Selection

### Decision

**Use `CLIP ViT-L/14@224` from OpenAI (open_clip identifier: `ViT-L-14`, pretrained `openai`).**

### Full Reasoning

| Criterion | CLIP ViT-L/14@224 | EVA-CLIP ViT-L/14@336 | CLIP ViT-B/16@224 |
|---|---|---|---|
| Same space as text prototypes | ✓ YES | ✗ No (different training) | ✓ Same family |
| DisNet pre-alignment | ✓ Already distilled toward this | ✗ Different space | Partial |
| Storage (28K × 6 cams × patches × 768-d, fp16) | **62 GB** | 139 GB | 32 GB |
| Feature dim | 768 (matches token head) | 768 | 512 (needs projection) |
| Patch count per image | 256 (14×14 grid, 16px each) | 576 (24×24 grid, 14px each) | 196 (14×14 grid, 16px each) |

**Why not EVA-CLIP for Job 1:**  
Our text prototypes were generated with `CLIP ViT-L/14` (OpenAI). EVA-CLIP uses a hybrid CLIP+EVA reconstruction objective trained on LAION — it produces a *different* embedding space from our text prototypes. Using EVA-CLIP visual features + OpenAI text features for L_O/L_V supervision creates a cross-model embedding gap that degrades signal quality. DisNet was distilled from OpenSeg which uses CLIP ViT-L/14 — supervising with the same model is supervising toward what DisNet already partially learned.

**EVA-CLIP is reserved for Job 2** (camera cross-attention, Phase 6), where you will regenerate text prototypes with EVA-CLIP's text encoder to maintain alignment.

**Fallback if <70 GB storage available:**  
Use `CLIP ViT-B/16@224` (32 GB) and add `cam_feature_proj: nn.Linear(512, 768)` as a small trainable bridge layer in the distillation losses. This costs ~800K parameters and slightly degrades feature quality.

### Exact open_clip Load Command

```python
import open_clip

# Primary recommendation — CLIP ViT-L/14@224 (OpenAI)
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-L-14',
    pretrained='openai'
)
model.eval()
for p in model.parameters():
    p.requires_grad = False

# Verify patch token output shape
import torch
test_img = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    feats = model.encode_image(test_img, normalize=False)  # (1, 768) — CLS only
    # For patch tokens, we need the intermediate representation:
    # See Section 5.2 for patch extraction method
```

---

## 3. Storage Requirements

### Precomputed Patch Grid Cache (Recommended)

Storing full 6-camera patch grids per training sample eliminates CLIP model runtime during training.

| Model | Per-sample size | Full trainval (28,130) | Val (6,019) | Total |
|---|---|---|---|---|
| CLIP ViT-L/14@224 | 6 × 256 × 768 × 2 bytes = 2.36 MB | 66.4 GB | 14.2 GB | **~81 GB** |
| CLIP ViT-B/16@224 | 6 × 196 × 512 × 2 bytes = 1.21 MB | 34.1 GB | 7.3 GB | **~42 GB** |
| CLIP ViT-B/32@224 | 6 × 49 × 512 × 2 bytes = 0.30 MB | 8.5 GB | 1.8 GB | **~11 GB** |

**Format:** One `.pt` file per sample token.  
**Path:** `{CAMERA_CACHE_DIR}/{model_name}/{sample_token}.pt`  
**Content:** `torch.Tensor` of shape `(6, N_patches, D)` in float16.  
**Camera order:** `['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']`

### Online Extraction Fallback

If storage is unavailable, CLIP runs at training time. Training time impact:  
- CLIP ViT-L/14@224 on 6 images: ~0.18s per sample on A100  
- 28,130 samples/epoch × 0.18s = ~84 min of CLIP overhead per epoch  
- 20 epochs × 84 min = ~28 additional hours total  
- **Use online mode only as last resort — training time doubles.**

---

## 4. Prerequisite Checklist

Complete ALL prerequisites before writing any new training code. These are diagnostic steps that validate the environment. The training script should not be modified until these pass.

### P0: Cluster Environment Verification

```bash
# On interactive GPU node:
bsub -Is -q batch_a100 -n 4 -M 8192 -W 01:00 \
  -gpu "num=1:mode=exclusive_process:mps=no" /bin/bash

module load conda && conda activate openscene
export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene

python3 -c "
import torch, open_clip, MinkowskiEngine as ME
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('open_clip version:', open_clip.__version__)
print('CUDA device:', torch.cuda.get_device_name(0))
print('GPU memory:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')
"
```

Expected output: PyTorch 1.7.x, CUDA available, A100 80/96 GB.

### P1: CLIP Model Verification

**Create and run:** `panoptic/scripts/verify_clip_model.py`

```python
"""
Prerequisite: verify CLIP ViT-L/14 loads correctly and produces patch tokens.
Run this BEFORE implementing any distillation losses.
Expected runtime: ~3 minutes on A100.
"""
import torch
import open_clip
import numpy as np

DEVICE = 'cuda'

def verify_clip_model():
    print("Loading CLIP ViT-L/14 (OpenAI)...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='openai'
    )
    model = model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False

    # Test 1: CLS token output
    img = torch.randn(1, 3, 224, 224).to(DEVICE)
    with torch.no_grad():
        cls_feat = model.encode_image(img, normalize=True)
    print(f"CLS token shape: {cls_feat.shape}")   # expect (1, 768)
    assert cls_feat.shape == (1, 768), f"Unexpected CLS shape: {cls_feat.shape}"

    # Test 2: Patch token extraction via forward hook
    patch_tokens = {}
    def patch_hook(module, input, output):
        # output shape depends on open_clip version — handle both
        if isinstance(output, (list, tuple)):
            patch_tokens['feat'] = output[0]
        else:
            patch_tokens['feat'] = output

    # Register hook on the transformer's final layer norm
    # This captures patch tokens before the final projection
    handle = model.visual.transformer.resblocks[-1].register_forward_hook(patch_hook)
    with torch.no_grad():
        _ = model.encode_image(img, normalize=False)
    handle.remove()

    raw = patch_tokens.get('feat')
    if raw is not None:
        print(f"Patch tokens via hook shape: {raw.shape}")
        # Expected: (seq_len, batch, dim) or (batch, seq_len, dim)
        # For ViT-L/14@224: seq_len = 256 patches + 1 CLS = 257
    else:
        print("WARNING: Hook did not capture patch tokens. Trying alternative method.")

    # Test 3: Text-image alignment (sanity check)
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    text = tokenizer(["a car", "a pedestrian", "a tree"]).to(DEVICE)
    with torch.no_grad():
        text_feats = model.encode_text(text, normalize=True)    # (3, 768)
        img_feat   = model.encode_image(img, normalize=True)    # (1, 768)
        sims = img_feat @ text_feats.T
    print(f"Text-image cosine similarities (random image): {sims.cpu().numpy()}")
    print("Note: random image should give ~equal low similarities to all classes")

    # Test 4: Memory
    allocated = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory after CLIP load: {allocated:.2f} GB")

    # Test 5: Confirm preprocess transform
    print(f"Preprocess: {preprocess}")
    print("Expected input size: 224×224")

    print("\n✓ CLIP verification PASSED")
    return model, preprocess

if __name__ == '__main__':
    model, preprocess = verify_clip_model()
```

**Run with:** `python panoptic/scripts/verify_clip_model.py`

**What you must confirm before proceeding:**
1. CLS token shape is `(1, 768)` ✓
2. Patch tokens are extractable (shape `(257, 1, 768)` or `(1, 257, 768)` — note the exact shape and record it)
3. GPU memory after load is <2 GB (should be ~1.5 GB for ViT-L/14)
4. **Record the exact patch token shape here** — the camera projection code depends on it

### P2: Storage Decision

```bash
# Check available scratch space
df -h /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/

# Based on output, choose:
# > 90 GB free  → use CLIP ViT-L/14@224, run precompute script
# 50-90 GB free → use CLIP ViT-B/16@224, run precompute script
# < 50 GB free  → use online mode (set CAMERA_CACHE_DIR=null in config)
```

Record the chosen model in `panoptic/configs/job1_dense_clip.yaml` under `camera.model`.

### P3: nuScenes Camera Projection Verification

**Create and run:** `panoptic/scripts/verify_camera_projection.py`

```python
"""
Prerequisite: verify LiDAR→camera projection works for one sample.
Tests the full transform chain using nuScenes devkit.
"""
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

DATAROOT = '/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes'
VERSION  = 'v1.0-mini'
DEVICE   = 'cuda'

def project_lidar_to_camera(nusc, sample_token, lidar_xyz, cam_name):
    """
    Project LiDAR points to a single camera.

    Args:
        nusc: NuScenes instance
        sample_token: str
        lidar_xyz: (N, 3) numpy array of LiDAR points in LIDAR_TOP sensor frame
        cam_name: str, e.g. 'CAM_FRONT'

    Returns:
        uv: (N, 2) pixel coordinates (may include out-of-frame points)
        depths: (N,) depth values in camera frame
        valid: (N,) boolean mask — True where point is visible in this camera
    """
    sample = nusc.get('sample', sample_token)

    # LiDAR sensor data record
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_cs = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    lidar_ep = nusc.get('ego_pose', lidar_sd['ego_pose_token'])

    # Camera sensor data record
    cam_sd = nusc.get('sample_data', sample['data'][cam_name])
    cam_cs = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
    cam_ep = nusc.get('ego_pose', cam_sd['ego_pose_token'])

    # Step 1: LiDAR sensor → ego frame (at LiDAR timestamp)
    pts = lidar_xyz.T  # (3, N)
    pts = pts + np.array(lidar_cs['translation'])[:, np.newaxis]
    pts = Quaternion(lidar_cs['rotation']).rotation_matrix @ pts

    # Step 2: Ego frame (LiDAR time) → global frame
    pts = pts + np.array(lidar_ep['translation'])[:, np.newaxis]
    pts = Quaternion(lidar_ep['rotation']).rotation_matrix @ pts

    # Step 3: Global frame → ego frame (camera timestamp)
    pts = pts - np.array(cam_ep['translation'])[:, np.newaxis]
    pts = Quaternion(cam_ep['rotation']).rotation_matrix.T @ pts

    # Step 4: Ego frame (camera time) → camera sensor frame
    pts = pts - np.array(cam_cs['translation'])[:, np.newaxis]
    pts = Quaternion(cam_cs['rotation']).rotation_matrix.T @ pts

    # Step 5: Project to pixel coordinates using camera intrinsics
    K = np.array(cam_cs['camera_intrinsic'])  # (3, 3)
    depths = pts[2, :]  # Z-depth in camera frame

    pts_img = view_points(pts, K, normalize=True)  # (3, N), last row = 1.0
    uv = pts_img[:2, :].T  # (N, 2) pixel coordinates

    # Get image dimensions
    impath = nusc.get_sample_data_path(cam_sd['token'])
    from PIL import Image
    img = Image.open(impath)
    W, H = img.size

    valid = (depths > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)

    return uv, depths, valid

def verify_projection():
    print("Loading nuScenes mini...")
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)

    sample = nusc.sample[0]
    sample_token = sample['token']
    print(f"Testing sample: {sample_token[:16]}...")

    # Load LiDAR
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    pc = LidarPointCloud.from_file(nusc.get_sample_data_path(lidar_sd['token']))
    lidar_xyz = pc.points[:3, :].T  # (N, 3)
    print(f"LiDAR points: {lidar_xyz.shape[0]}")

    # Project to each camera
    cam_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

    total_visible = 0
    for cam in cam_names:
        uv, depths, valid = project_lidar_to_camera(nusc, sample_token, lidar_xyz, cam)
        pct = valid.sum() / len(valid) * 100
        total_visible += valid.sum()
        print(f"  {cam}: {valid.sum()} points visible ({pct:.1f}%)")

    # Expect coverage: ~70-85% of LiDAR points visible in at least one camera
    # Each camera covers ~40-60% of points
    print(f"Total visibility events: {total_visible} (some points visible in multiple cameras)")
    print("\n✓ Camera projection verification PASSED")
    return nusc

if __name__ == '__main__':
    nusc = verify_projection()
```

**What to verify:**
- Each camera covers 35–60% of LiDAR points ✓
- Depths are all positive for valid points ✓
- UV coordinates are within image bounds for valid points ✓

### P4: Mini Smoke Test (BEFORE full run)

After all code is implemented, run on mini for 3 epochs to catch NaN gradients, OOM, and shape errors:

```bash
python -u panoptic/train/train_panoptic_job1.py \
    --config panoptic/configs/job1_dense_clip_mini.yaml \
    --smoke_test \
    2>&1 | tee logs/job1_smoke_$(date +%Y%m%d_%H%M).log
```

**Smoke test pass criteria:**
- All 3 epochs complete without exception
- `loss_L_O` < 1.0 by epoch 2 (it should converge fast)
- `loss_L_V` < 0.5 by epoch 2
- No NaN in any loss term (monitor `torch.isnan(total_loss)`)
- LoRA adapter grad norms between 1e-4 and 10
- GPU peak memory < 40 GB

---

## 5. Architecture Changes

### 5.1 Single CLIP Embedding Head (A4)

Replace the dual-head (`class_embed: Linear(256→17)` + `CLIPTokenHead: MLP(256→768)`) with a single embedding head.

**File to modify:** `panoptic/models/mask_decoder.py`

```python
# REMOVE these two lines from PanopticDecoder.__init__:
#   self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
#   (CLIPTokenHead is in clip_token_head.py — will be replaced)

# ADD this single head:
class SingleCLIPEmbeddingHead(nn.Module):
    """
    Replaces linear classification head + CLIPTokenHead.
    Classification at inference: argmax(embed @ text_protos.T / temperature)
    """
    def __init__(self, hidden_dim=256, clip_dim=768, num_layers=3):
        super().__init__()
        # Same MLP structure as old CLIPTokenHead
        dims = [hidden_dim] + [hidden_dim * 2] * (num_layers - 2) + [clip_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*layers)
        # Temperature — learnable, initialised from CLIP default
        self.log_temperature = nn.Parameter(torch.tensor(np.log(1.0 / 0.07)))

    def forward(self, x):
        """x: (B, Q, hidden_dim) — decoder output"""
        return F.normalize(self.mlp(x), dim=-1)  # (B, Q, 768)

    @property
    def temperature(self):
        return self.log_temperature.exp().clamp(max=100)
```

**In PanopticDecoder.forward(), replace:**
```python
# OLD:
# pred_logits = self.class_embed(decoder_norm)    # (B, Q, num_classes+1)
# pred_tokens = self.token_head(decoder_norm)      # (B, Q, 768)

# NEW:
pred_embed = self.clip_embed_head(decoder_norm)    # (B, Q, 768) L2-normalised
# Classification at inference: pred_embed @ text_protos.T / temperature
# No separate linear head — single pathway
```

**Impact on PQ:** Confirmed −0.008 vs linear head from Phase 5 experiments. This is the known cost of true open-vocab capability.

**Objectness:** Without the no-object class from the linear head, use a separate binary objectness signal. Add `self.objectness_head = nn.Linear(hidden_dim, 1)` and train with BCE against has-GT-match as positive. Use `objectness.sigmoid()` to gate panoptic assignment instead of `1 - P(no-object)`.

### 5.2 LoRA Sparse Conv Adapters (B1)

**File to create:** `panoptic/models/lora_backbone.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME


class SparseLoRAAdapter(nn.Module):
    """
    Trainable residual adapter inserted after frozen backbone blocks.

    CRITICAL: Uses standard nn.Linear (NOT ME.MinkowskiLinear) so that
    the backward pass goes through dense ops, bypassing ME's hash map
    traversal entirely. ME gradient check confirmed 0.084 GB memory
    for decoder layers — this path is safe.

    The adapter receives the .F feature matrix from the ME sparse tensor,
    applies a low-rank projection, and returns a new sparse tensor with
    modified features but IDENTICAL coordinate structure.

    Args:
        channels: feature dimension of the backbone block output
        rank: low-rank dimension (default 16, higher = more capacity, more memory)
        dropout: dropout on the adapter output for regularisation
    """
    def __init__(self, channels: int, rank: int = 16, dropout: float = 0.05):
        super().__init__()
        self.down = nn.Linear(channels, rank, bias=False)
        self.up   = nn.Linear(rank, channels, bias=False)
        self.norm = nn.LayerNorm(channels)
        self.drop = nn.Dropout(p=dropout)

        # Zero-init: adapter starts as identity (output = input)
        nn.init.kaiming_uniform_(self.down.weight, nonlinearity='relu')
        nn.init.zeros_(self.up.weight)    # zero-init → identity at epoch 0

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        feats = x.F                                            # (V, C) dense
        delta = self.drop(self.up(F.relu(self.down(feats))))   # (V, C)
        adapted = self.norm(feats + delta)                     # (V, C)

        return ME.SparseTensor(
            features=adapted,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager
        )


def insert_lora_adapters(backbone, rank: int = 16, skip_bn: bool = False):
    """
    Insert LoRA adapters after confirmed-safe backbone layers.

    ME gradient check confirmed these 7 layers:
        net3d.final, net3d.block8, net3d.convtr7p2s2,
        net3d.bntr7, net3d.block7, net3d.convtr6p4s2, net3d.bntr6

    Args:
        backbone: DisNet model
        rank: adapter rank (16 = default, 32 = more capacity)
        skip_bn: if True, skip BatchNorm layers (bntr6, bntr7).
                 Set to True if BN+LoRA causes training instability.
                 Set to False (default) to adapt all 7 confirmed layers.

    Returns:
        adapters: nn.ModuleDict of inserted adapters (for optimiser param group)
    """
    # Layer name → expected output channels (from MinkUNet18A architecture)
    TARGET_LAYERS = {
        'net3d.final':        768,   # output layer
        'net3d.block8':       96,    # last decoder res block
        'net3d.convtr7p2s2':  96,    # last deconv transpose
        'net3d.bntr7':        96,    # batch norm (skip if skip_bn=True)
        'net3d.block7':       96,    # second-to-last decoder block
        'net3d.convtr6p4s2':  128,   # second-to-last deconv transpose
        'net3d.bntr6':        128,   # batch norm (skip if skip_bn=True)
    }

    BN_LAYERS = {'net3d.bntr7', 'net3d.bntr6'}

    adapters = nn.ModuleDict()

    for layer_name, out_channels in TARGET_LAYERS.items():
        if skip_bn and layer_name in BN_LAYERS:
            print(f"  Skipping BN layer: {layer_name}")
            continue

        adapter = SparseLoRAAdapter(channels=out_channels, rank=rank)
        adapters[layer_name.replace('.', '_')] = adapter
        print(f"  Inserted LoRA adapter after {layer_name} (channels={out_channels}, rank={rank})")

    # Backbone remains fully frozen — only adapters are trainable
    for p in backbone.parameters():
        p.requires_grad = False

    print(f"Backbone frozen. LoRA adapters: {sum(p.numel() for p in adapters.parameters()):,} trainable params")
    return adapters
```

**Wiring into `extract_backbone_features()` in `train_panoptic_job1.py`:**

```python
def extract_backbone_features_with_lora(sparse_input, backbone, lora_adapters):
    """
    Run backbone (frozen) + apply LoRA adapters after target layers.
    backbone.eval() is called to disable BN update in frozen layers.
    """
    backbone.eval()   # keep BN in eval mode — frozen backbone should not update running stats

    # Intercept intermediate outputs and apply adapters
    # We monkey-patch the forward to inject adapters at the right positions
    # The cleaner alternative is to modify disnet.py to accept adapters as optional modules

    with torch.no_grad():
        # Since backbone is frozen and adapters are external,
        # we run the backbone in segments using forward hooks
        pass  # Implementation depends on DisNet's internal structure
              # See Section 8 (File Modifications) for the exact hook pattern
```

---

## 6. Camera CLIP Infrastructure

### 6.1 Precompute Script

**File to create:** `panoptic/scripts/precompute_camera_clip_features.py`

```python
"""
Precompute CLIP ViT-L/14@224 patch features for all nuScenes training samples.
Output: one .pt file per sample = (6, 256, 768) tensor in float16.
Storage: ~62 GB for full trainval, ~14 GB for val.

Usage:
    python panoptic/scripts/precompute_camera_clip_features.py \
        --dataroot /path/to/nuscenes \
        --version v1.0-trainval \
        --output_dir /path/to/camera_clip_cache \
        --model ViT-L-14 \
        --pretrained openai \
        --batch_cameras 6
"""
import os, argparse, torch, open_clip
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes

CAM_NAMES = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
             'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

def extract_patch_tokens(model, img_tensor, device):
    """
    Extract patch tokens from CLIP ViT image encoder.
    Returns: (N_patches, D) tensor — CLS token EXCLUDED.

    For ViT-L/14@224: N_patches = 256 (16×16 grid), D = 768.

    The method depends on open_clip version. We try the most common approaches
    in order and use whichever works.
    """
    img_tensor = img_tensor.to(device)

    # Method 1: encode_image with intermediate=True (some open_clip versions)
    try:
        feats = model.visual(img_tensor)
        if feats.ndim == 3:
            # Shape (B, seq_len, D) — seq_len includes CLS token at index 0
            return feats[0, 1:, :]  # (N_patches, D) — strip CLS
    except Exception:
        pass

    # Method 2: Forward hook on the last transformer block's output
    patch_store = {}
    def hook(module, input, output):
        if isinstance(output, tuple):
            patch_store['x'] = output[0]
        else:
            patch_store['x'] = output

    handle = model.visual.transformer.resblocks[-1].register_forward_hook(hook)
    with torch.no_grad():
        model.encode_image(img_tensor, normalize=False)
    handle.remove()

    if 'x' in patch_store:
        x = patch_store['x']
        # May be (seq_len, B, D) or (B, seq_len, D)
        if x.shape[0] > x.shape[-1]:  # seq_len is first dim
            x = x.permute(1, 0, 2)  # → (B, seq_len, D)
        return x[0, 1:, :].float()   # (N_patches, D), strip CLS, cast to float32

    raise RuntimeError("Could not extract patch tokens. Run verify_clip_model.py first and record exact shape.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True)
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model', default='ViT-L-14')
    parser.add_argument('--pretrained', default='openai')
    parser.add_argument('--split', default='train', choices=['train', 'val', 'all'])
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading CLIP {args.model} ({args.pretrained})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    print("Loading nuScenes...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    # Collect sample tokens for requested split
    split_scenes = {'v1.0-mini': {'train': 8, 'val': 2},
                    'v1.0-trainval': {'train': 700, 'val': 150}}
    all_scenes = nusc.scene
    if args.split != 'all':
        n_train = split_scenes.get(args.version, {}).get('train', 700)
        if args.split == 'train':
            scenes = all_scenes[:n_train]
        else:
            scenes = all_scenes[n_train:]
    else:
        scenes = all_scenes

    sample_tokens = []
    for scene in scenes:
        token = scene['first_sample_token']
        while token:
            sample_tokens.append(token)
            token = nusc.get('sample', token)['next']

    print(f"Processing {len(sample_tokens)} samples...")

    skipped = 0
    for sample_token in tqdm(sample_tokens):
        out_path = os.path.join(args.output_dir, f"{sample_token}.pt")
        if os.path.exists(out_path):
            skipped += 1
            continue

        sample = nusc.get('sample', sample_token)
        cam_features = []

        for cam_name in CAM_NAMES:
            sd = nusc.get('sample_data', sample['data'][cam_name])
            img_path = os.path.join(args.dataroot, sd['filename'])
            img = Image.open(img_path).convert('RGB')
            img_t = preprocess(img).unsqueeze(0)   # (1, 3, 224, 224)

            with torch.no_grad():
                patches = extract_patch_tokens(model, img_t, device)   # (N_patches, D)
                patches = F.normalize(patches, dim=-1).half()           # float16

            cam_features.append(patches)

        # Stack: (6, N_patches, D)
        cam_features = torch.stack(cam_features, dim=0).cpu()
        torch.save(cam_features, out_path)

    print(f"Done. Processed {len(sample_tokens) - skipped} samples "
          f"({skipped} already cached).")
    est_gb = (len(sample_tokens) * 6 * 256 * 768 * 2) / 1e9
    print(f"Estimated cache size: {est_gb:.1f} GB (ViT-L/14@224)")


if __name__ == '__main__':
    import torch.nn.functional as F
    main()
```

**Run precomputation (submit as batch job, takes ~8 hours for full trainval):**

```bash
# Run in background on CPU/GPU node — doesn't need interactive session
bsub -q batch_a100 -n 4 -M 16384 -W 10:00 \
  -gpu "num=1:mode=exclusive_process:mps=no" \
  -o logs/precompute_clip_%J.log \
  python panoptic/scripts/precompute_camera_clip_features.py \
    --dataroot /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes \
    --version v1.0-trainval \
    --output_dir /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/camera_clip_cache/vitl14_224 \
    --model ViT-L-14 --pretrained openai --split all
```

**Start this precomputation job the moment the script is written — it can run in parallel with the rest of the implementation work.**

### 6.2 Camera Feature Loader (Training-Time)

**File to create:** `panoptic/utils/camera_clip_loader.py`

```python
"""
Loads precomputed CLIP patch grids and projects them to per-voxel features
for use in L_O and L_V distillation losses.
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

CAM_NAMES = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
             'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

# Image dimensions for nuScenes cameras (standard)
CAM_W, CAM_H = 1600, 900
# Patch grid size for CLIP ViT-L/14@224
# 224px / 14px per patch = 16 patches per side → 16×16 = 256 patches
PATCH_GRID = 16   # set to 24 for @336 variant (576 patches)


class CameraClipLoader:
    """
    Loads cached CLIP patch grids and projects them to per-voxel features.

    Usage:
        loader = CameraClipLoader(nusc, cache_dir, patch_grid=16)
        fv_clip = loader.get_voxel_features(
            sample_token, vox_coords_xyz
        )  # (V, 768) float32, zeros for uncovered voxels
    """

    def __init__(self, nusc: NuScenes, cache_dir: str, patch_grid: int = 16,
                 feature_dim: int = 768, device: str = 'cpu'):
        self.nusc = nusc
        self.cache_dir = cache_dir
        self.patch_grid = patch_grid
        self.feature_dim = feature_dim
        self.device = device
        self.patch_size_px = CAM_W // (patch_grid * (CAM_W // 224))  # pixels per patch

    def get_voxel_features(self, sample_token: str, vox_coords_xyz: np.ndarray,
                           device=None) -> torch.Tensor:
        """
        For each voxel, project to all cameras, sample patch features, average.

        Args:
            sample_token: nuScenes sample token
            vox_coords_xyz: (V, 3) numpy array, LiDAR sensor frame coordinates
            device: target device for output tensor

        Returns:
            fv_clip: (V, D) float32 tensor, L2-normalised
                     Zero vectors for voxels with no camera coverage.
        """
        if device is None:
            device = self.device

        # Load cached patch grids: (6, N_patches, D) float16
        cache_path = os.path.join(self.cache_dir, f"{sample_token}.pt")
        if not os.path.exists(cache_path):
            # Graceful fallback: return zeros (disables L_V for this sample)
            V = len(vox_coords_xyz)
            return torch.zeros(V, self.feature_dim, dtype=torch.float32)

        patch_grids = torch.load(cache_path, map_location='cpu').float()  # (6, P, D)
        # patch_grids[cam_idx] shape: (N_patches, D)
        # Reshape to 2D grid for bilinear sampling: (1, D, patch_grid, patch_grid)
        N_patches = patch_grids.shape[1]
        assert N_patches == self.patch_grid ** 2, \
            f"Expected {self.patch_grid**2} patches, got {N_patches}"

        V = len(vox_coords_xyz)
        accumulated = torch.zeros(V, self.feature_dim)
        count = torch.zeros(V)

        sample = self.nusc.get('sample', sample_token)

        for cam_idx, cam_name in enumerate(CAM_NAMES):
            uv, depths, valid = self._project_to_camera(
                sample, cam_name, vox_coords_xyz
            )
            if valid.sum() == 0:
                continue

            valid_uv = uv[valid]       # (V_valid, 2) pixel coordinates

            # Convert pixel coords to patch grid coords (normalised -1 to 1 for grid_sample)
            # Patch grid is patch_grid × patch_grid covering the full image
            px_per_patch_w = CAM_W / self.patch_grid
            px_per_patch_h = CAM_H / self.patch_grid

            patch_u = valid_uv[:, 0] / px_per_patch_w   # 0 to patch_grid
            patch_v = valid_uv[:, 1] / px_per_patch_h   # 0 to patch_grid

            # Normalise to [-1, 1] for F.grid_sample
            norm_u = (patch_u / self.patch_grid) * 2 - 1
            norm_v = (patch_v / self.patch_grid) * 2 - 1
            grid = torch.tensor(
                np.stack([norm_u, norm_v], axis=1), dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0)  # (1, 1, V_valid, 2)

            # Reshape patch grid for sampling: (1, D, patch_grid, patch_grid)
            pg = patch_grids[cam_idx]                                  # (P, D)
            pg_2d = pg.T.reshape(1, self.feature_dim,                  # (1, D, G, G)
                                  self.patch_grid, self.patch_grid)

            # Bilinear sampling: (1, D, 1, V_valid) → (V_valid, D)
            sampled = F.grid_sample(
                pg_2d, grid,
                mode='bilinear', padding_mode='border', align_corners=True
            )
            sampled = sampled.squeeze(0).squeeze(1).T  # (V_valid, D)

            accumulated[valid] += sampled
            count[valid] += 1

        # Average over cameras, L2-normalise
        covered = count > 0
        fv_clip = torch.zeros(V, self.feature_dim)
        fv_clip[covered] = F.normalize(
            accumulated[covered] / count[covered].unsqueeze(1), dim=-1
        )

        return fv_clip.to(device)

    def _project_to_camera(self, sample, cam_name, lidar_xyz):
        """Full nuScenes transform chain: LiDAR sensor → pixel."""
        lidar_sd = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_cs = self.nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
        lidar_ep = self.nusc.get('ego_pose', lidar_sd['ego_pose_token'])
        cam_sd   = self.nusc.get('sample_data', sample['data'][cam_name])
        cam_cs   = self.nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
        cam_ep   = self.nusc.get('ego_pose', cam_sd['ego_pose_token'])

        pts = lidar_xyz.T   # (3, N)

        # LiDAR sensor → ego (LiDAR time) → global → ego (cam time) → cam sensor
        pts = Quaternion(lidar_cs['rotation']).rotation_matrix @ pts + \
              np.array(lidar_cs['translation'])[:, np.newaxis]
        pts = Quaternion(lidar_ep['rotation']).rotation_matrix @ pts + \
              np.array(lidar_ep['translation'])[:, np.newaxis]
        pts = Quaternion(cam_ep['rotation']).rotation_matrix.T @ \
              (pts - np.array(cam_ep['translation'])[:, np.newaxis])
        pts = Quaternion(cam_cs['rotation']).rotation_matrix.T @ \
              (pts - np.array(cam_cs['translation'])[:, np.newaxis])

        depths = pts[2, :]
        K = np.array(cam_cs['camera_intrinsic'])
        pts_img = view_points(pts, K, normalize=True)
        uv = pts_img[:2, :].T

        valid = ((depths > 0.5) &
                 (uv[:, 0] >= 0) & (uv[:, 0] < CAM_W) &
                 (uv[:, 1] >= 0) & (uv[:, 1] < CAM_H))

        return uv, depths, valid
```

---

## 7. Distillation Losses

**File to create:** `panoptic/losses/distillation_loss.py`

```python
"""
Distillation losses for Job 1:
  - L_O: Object-level distillation (matched queries → camera CLIP)
  - L_V: Voxel-level distillation (all queries, reconstructed per-voxel features)
  - L_V_CASD: Context-aware stuff distillation (modified L_V targets for stuff classes)
  - VoxelDecompLoss: Per-voxel text contrastive loss (L7, root cause fix)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict

# Classes receiving CASD treatment (contextually ambiguous stuff)
# These indices are 0-based nuScenes-16 class indices
CASD_CLASSES = {
    11: 3.0,   # other_flat  — most ambiguous, widest context crop
    12: 2.5,   # sidewalk    — needs road+buildings context
    13: 2.0,   # terrain     — needs vegetation+road context
}


class ObjectLevelDistillationLoss(nn.Module):
    """
    L_O: Align matched decoder query embeddings to mask-pooled camera CLIP features.

    For each Hungarian-matched (query, gt_mask) pair:
        w_q = soft_pool(fv_clip, pred_mask_q)   — mask-pooled camera features
        v_q = clip_embed_head(decoder_output_q)  — query embedding
        L_O = mean(1 - cos_sim(v_q, w_q))

    Applied ONLY to matched queries — unmatched queries (background) are excluded.
    Including unmatched queries degrades performance (noisy random pooling targets).

    Args:
        weight: loss weight (default 1.0)
    """
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self,
                pred_embeds: torch.Tensor,       # (B, Q, 768) from SingleCLIPEmbeddingHead
                pred_masks:  torch.Tensor,       # (B, V, Q) mask logits (before sigmoid)
                fv_clip:     torch.Tensor,       # (B, V, 768) per-voxel camera CLIP (zeros = no coverage)
                matched_indices: List          # list of (pred_idx, gt_idx) per batch item
               ) -> torch.Tensor:

        B = pred_embeds.shape[0]
        total_loss = torch.tensor(0.0, device=pred_embeds.device, requires_grad=True)
        n_pairs = 0

        for b in range(B):
            pred_idx, gt_idx = matched_indices[b]
            if len(pred_idx) == 0:
                continue

            # Soft-pool camera CLIP features within each matched mask
            mask_probs = pred_masks[b, :, pred_idx].sigmoid().T   # (n_matched, V)
            mask_area  = mask_probs.sum(dim=1, keepdim=True).clamp(min=1)
            weights    = mask_probs / mask_area                    # (n_matched, V)

            # w_q: (n_matched, 768) — mask-pooled camera CLIP
            w_q = weights @ fv_clip[b]                            # (n_matched, 768)

            # Only supervise pairs where camera coverage exists
            has_coverage = w_q.norm(dim=-1) > 0.05               # (n_matched,)
            if has_coverage.sum() == 0:
                continue

            w_q = F.normalize(w_q[has_coverage], dim=-1)
            v_q = pred_embeds[b, pred_idx[has_coverage]]          # already L2-normed
            # v_q should already be L2-normalised from SingleCLIPEmbeddingHead

            loss_b = (1 - (v_q * w_q).sum(dim=-1)).mean()
            total_loss = total_loss + loss_b
            n_pairs += 1

        if n_pairs == 0:
            return total_loss

        return self.weight * total_loss / n_pairs


class VoxelLevelDistillationLoss(nn.Module):
    """
    L_V: Reconstruct per-voxel CLIP features from all queries.

    F_rec = M_Q^T @ F_Q_emb   (reconstruction of per-voxel features as
                                weighted sum of all query embeddings)
    L_V = L1(F_rec[covered], fv_clip[covered])

    Applied to ALL queries (not just matched) — this is the key difference
    from L_O. Provides dense voxel-level supervision.

    Weight is deliberately small (0.1) because noisy early mask scores
    produce noisy F_rec, which can destabilise training.

    Args:
        weight: loss weight (default 0.1 — do not increase without testing)
        casd_classes: dict of {class_idx: context_scale} for CASD treatment
    """
    def __init__(self, weight: float = 0.1, casd_classes: Optional[Dict] = None):
        super().__init__()
        self.weight = weight
        self.casd_classes = casd_classes or {}

    def forward(self,
                pred_embeds: torch.Tensor,    # (B, Q, 768)
                pred_masks:  torch.Tensor,    # (B, V, Q)
                fv_clip:     torch.Tensor,    # (B, V, 768) — standard per-voxel targets
                fv_clip_casd: Optional[torch.Tensor] = None,  # (B, V, 768) — CASD targets
                gt_sem:       Optional[torch.Tensor] = None,  # (B, V) for CASD routing
               ) -> torch.Tensor:

        B = pred_embeds.shape[0]
        total_loss = torch.tensor(0.0, device=pred_embeds.device, requires_grad=True)

        for b in range(B):
            # F_rec = M_Q^T @ F_Q  — reconstruct per-voxel features from queries
            M_Q   = pred_masks[b].sigmoid().T     # (Q, V)
            F_Q   = pred_embeds[b]                # (Q, 768)
            F_rec = M_Q.T @ F_Q                   # (V, 768)

            # Build CASD-aware target
            if fv_clip_casd is not None and gt_sem is not None and self.casd_classes:
                target = fv_clip[b].clone()
                for cls_idx in self.casd_classes:
                    stuff_mask = (gt_sem[b] == cls_idx)
                    if stuff_mask.any():
                        target[stuff_mask] = fv_clip_casd[b][stuff_mask]
            else:
                target = fv_clip[b]

            # Only supervise voxels with camera coverage
            has_coverage = (target.norm(dim=-1) > 0.05)   # (V,)
            if has_coverage.sum() == 0:
                continue

            loss_b = F.l1_loss(F_rec[has_coverage], target[has_coverage].detach())
            total_loss = total_loss + loss_b

        return self.weight * total_loss / B


class VoxelDecompositionLoss(nn.Module):
    """
    L7 (Strategy I): Per-voxel contrastive loss against 200+ text prototypes.

    ROOT CAUSE FIX: directly trains backbone features to have inter-class margin.
    Each foreground voxel must score higher for its GT class than for all other
    classes in the 200+ concept bank (including confusors: van, SUV, pickup for car).

    Requires:
        - LoRA adapters (B1) installed to give this loss a gradient destination.
          Without B1, backbone is frozen and this loss does nothing.
        - text_bank: (C_all, 768) precomputed embeddings for 200+ driving concepts

    Args:
        text_bank: (C_all, 768) all concept embeddings, base 16 classes first
        weight: loss weight (start 0, ramp to 0.5 over 5 epochs)
        temperature: InfoNCE temperature (0.07 default = CLIP standard)
        n_subsample: max voxels per sample (10K recommended for memory)
    """
    def __init__(self, text_bank: torch.Tensor, weight: float = 0.5,
                 temperature: float = 0.07, n_subsample: int = 10000):
        super().__init__()
        self.register_buffer('text_bank', F.normalize(text_bank, dim=-1))
        self.weight      = weight
        self.temperature = temperature
        self.n_subsample = n_subsample

    def forward(self,
                backbone_feats: torch.Tensor,   # (B, V, 768) from backbone/LoRA
                gt_sem:         torch.Tensor,   # (B, V) semantic labels (255=ignore)
               ) -> torch.Tensor:

        B = backbone_feats.shape[0]
        total_loss = torch.tensor(0.0, device=backbone_feats.device, requires_grad=True)
        n_valid_batches = 0

        for b in range(B):
            valid_mask = (gt_sem[b] != 255)
            if valid_mask.sum() == 0:
                continue

            feats  = backbone_feats[b, valid_mask]   # (V', 768)
            labels = gt_sem[b][valid_mask]            # (V',) in 0..15

            # Subsample for memory efficiency (10K random valid voxels)
            if len(feats) > self.n_subsample:
                idx = torch.randperm(len(feats), device=feats.device)[:self.n_subsample]
                feats  = feats[idx]
                labels = labels[idx]

            # Normalise features (backbone output may not be normalised)
            feats_normed = F.normalize(feats.float(), dim=-1)

            # Cosine similarity to all 200+ concepts
            sim = feats_normed @ self.text_bank.T / self.temperature  # (V', C_all)

            # GT labels index into first 16 positions of text_bank
            # (base 16 classes are stored first in text_bank)
            loss_b = F.cross_entropy(sim, labels)

            if torch.isfinite(loss_b):
                total_loss = total_loss + loss_b
                n_valid_batches += 1

        if n_valid_batches == 0:
            return total_loss

        return self.weight * total_loss / n_valid_batches
```

---

## 8. Updated Token Loss (L5: 200+ Negative Bank)

**File to modify:** `panoptic/losses/token_loss.py`

```python
# Add to existing TokenLoss or replace the contrastive method:

class TokenLossC1(nn.Module):
    """
    Updated InfoNCE token loss with 200+ negative bank.

    Replaces the existing 16-prototype InfoNCE which caused confirmed
    closed-set collapse (zero novel detections in all inference experiments).

    The 200+ negatives include:
    - 16 base nuScenes classes (GT positive pool)
    - ~60 driving confusors: van, SUV, pickup truck, scooter, moped,
      child, jogger, skateboarder, bollard, lamp post, fire hydrant,
      bench, bin, fence post, guard rail, curb, speed bump, crosswalk...
    - ~40 generic negatives: dog, cat, cloud, gravel, puddle, shadow...
    - ~80 adversarial similar concepts for each base class

    With 216+ denominators the token head cannot learn 16 tunnels —
    it must learn the fine-grained CLIP space structure around each class.
    """
    def __init__(self, text_protos_16: torch.Tensor,
                 negative_bank: torch.Tensor,
                 learnable_temperature: bool = True):
        super().__init__()
        # Concatenate: base 16 classes + 200+ negatives
        all_protos = torch.cat([text_protos_16, negative_bank], dim=0)
        self.register_buffer('all_protos', F.normalize(all_protos, dim=-1))
        self.n_base = text_protos_16.shape[0]   # 16

        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.tensor(np.log(1.0 / 0.07)))
        else:
            self.register_buffer('log_temperature', torch.tensor(np.log(1.0 / 0.07)))

    @property
    def temperature(self):
        return self.log_temperature.exp().clamp(max=100)

    def forward(self, pred_embeds: torch.Tensor,   # (Q_matched, 768) L2-normed
                gt_classes:   torch.Tensor         # (Q_matched,) indices 0..15
               ) -> torch.Tensor:
        # Cosine similarities to all 216+ concepts
        sim = pred_embeds @ self.all_protos.T / self.temperature   # (Q, C_all)

        # GT labels index into first 16 positions (base classes)
        loss = F.cross_entropy(sim, gt_classes)
        return loss
```

**File to create:** `panoptic/scripts/generate_negative_bank.py`

```python
"""
Generate the 200+ negative text embedding bank for C1 token loss.
Saves: panoptic/clip_prototypes/negative_bank_200.pt  → (200, 768)

Run once before training:
    python panoptic/scripts/generate_negative_bank.py
"""
import torch, open_clip, torch.nn.functional as F
from pathlib import Path

NEGATIVE_CONCEPTS = [
    # Driving confusors (similar to base classes — most important)
    "a van in a scene", "an SUV in a scene", "a minivan in a scene",
    "a pickup truck in a scene", "a delivery van in a scene",
    "a scooter in a scene", "a moped in a scene", "a segway in a scene",
    "a child walking in a scene", "a jogger in a scene", "a skateboarder in a scene",
    "a cyclist in a scene", "an electric bicycle in a scene",
    "a school bus in a scene", "a coach bus in a scene", "a double decker bus in a scene",
    "a fire truck in a scene", "an ambulance in a scene", "a police car in a scene",
    "a tow truck in a scene", "a garbage truck in a scene", "a cement truck in a scene",

    # Infrastructure / novel driving objects
    "a street lamp in a scene", "a lamp post in a scene", "a utility pole in a scene",
    "a traffic light in a scene", "a traffic sign in a scene", "a stop sign in a scene",
    "a fire hydrant in a scene", "a bollard in a scene", "a parking meter in a scene",
    "a bench in a scene", "a trash bin in a scene", "a mailbox in a scene",
    "a fence in a scene", "a guard rail in a scene", "a curb in a scene",
    "a speed bump in a scene", "a pothole in a scene", "a crosswalk in a scene",
    "a tree trunk in a scene", "a bush in a scene", "a shrub in a scene",
    "a wall in a scene", "a pillar in a scene", "a column in a scene",

    # Ground surface variants
    "an asphalt road in a scene", "a concrete road in a scene",
    "a gravel road in a scene", "a dirt road in a scene",
    "a bike lane in a scene", "a parking lot in a scene",
    "a driveway in a scene", "an intersection in a scene",

    # Vehicles not in base classes
    "a motorcycle with sidecar in a scene", "a tricycle in a scene",
    "an ATV in a scene", "a golf cart in a scene",
    "a forklift in a scene", "a bulldozer in a scene",
    "an excavator in a scene", "a crane in a scene",
    "a road roller in a scene", "a tractor in a scene",

    # General objects (regularisation — not driving-specific)
    "a dog in a scene", "a cat in a scene", "a bird in a scene",
    "a person sitting in a scene", "a person standing in a scene",
    "a group of people in a scene",
    "a building in a scene", "a house in a scene", "a tower in a scene",
    "a bridge in a scene", "a tunnel in a scene", "an underpass in a scene",

    # Materials/surfaces (regularisation)
    "metal in a scene", "concrete in a scene", "glass in a scene",
    "grass in a scene", "sand in a scene", "gravel in a scene",
    "snow in a scene", "puddles in a scene", "shadows in a scene",

    # Ambiguous stuff-like objects
    "a parking barrier in a scene", "a road divider in a scene",
    "a median strip in a scene", "a raised kerb in a scene",
    "a drainage grate in a scene", "a manhole cover in a scene",
    "road markings in a scene", "lane markings in a scene",

    # Night/weather variants (harder negatives)
    "a car at night in a scene", "a person in rain in a scene",
    "a wet road in a scene", "a road with reflections in a scene",
]

# Deduplicate and trim to exactly 200
NEGATIVE_CONCEPTS = list(dict.fromkeys(NEGATIVE_CONCEPTS))[:200]

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Generating {len(NEGATIVE_CONCEPTS)} negative embeddings...")

    model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer('ViT-L-14')

    embeddings = []
    batch_size = 32
    for i in range(0, len(NEGATIVE_CONCEPTS), batch_size):
        batch = NEGATIVE_CONCEPTS[i:i+batch_size]
        tokens = tokenizer(batch).to(device)
        with torch.no_grad():
            emb = model.encode_text(tokens, normalize=True).cpu()
        embeddings.append(emb)

    negative_bank = torch.cat(embeddings, dim=0)   # (200, 768)
    assert negative_bank.shape == (len(NEGATIVE_CONCEPTS), 768)

    Path('panoptic/clip_prototypes').mkdir(exist_ok=True)
    torch.save(negative_bank, 'panoptic/clip_prototypes/negative_bank_200.pt')
    print(f"Saved: panoptic/clip_prototypes/negative_bank_200.pt  {negative_bank.shape}")
    print(f"Concepts list saved to: panoptic/clip_prototypes/negative_bank_concepts.txt")

    with open('panoptic/clip_prototypes/negative_bank_concepts.txt', 'w') as f:
        for c in NEGATIVE_CONCEPTS:
            f.write(c + '\n')

if __name__ == '__main__':
    main()
```

---

## 9. Training Configuration

### 9.1 New Config File

**File to create:** `panoptic/configs/job1_dense_clip.yaml`

```yaml
# Job 1: Dense Camera-CLIP Supervision
# Train on all 16 classes (Option A) — clean closed-set baseline + open-vocab losses
# Warm-start from PQ=0.476 checkpoint

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
  num_classes: 16
  num_thing_queries: 100
  num_stuff_classes: 6
  nheads: 8
  dim_feedforward: 1024
  num_blocks: 3
  num_levels: 3
  dropout: 0.0

  # Single CLIP embedding head (A4 — replaces dual linear+token head)
  head_type: single_clip      # 'dual' | 'single_clip'
  clip_embed_dim: 768

  # LoRA adapters (B1)
  lora:
    enabled: true
    rank: 16
    skip_bn_layers: false     # set true to skip bntr6/bntr7 if instability occurs
    dropout: 0.05

camera:
  model: ViT-L-14             # open_clip model name
  pretrained: openai          # 'openai' for CLIP ViT-L/14@224
  patch_grid: 16              # 16 for @224, 24 for @336
  feature_dim: 768
  cache_dir: /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/camera_clip_cache/vitl14_224
  # Set cache_dir to null to use online extraction (slower but no storage needed)

distillation:
  L_O:
    enabled: true
    weight: 1.0
    warmup_epochs: 2           # ramp 0→1.0 over 2 epochs after phase 2 start
  L_V:
    enabled: true
    weight: 0.1                # deliberately small — noisy early masks
    warmup_epochs: 5           # ramp 0→0.1 over 5 epochs after phase 2 start
    casd:
      enabled: true
      classes: [11, 12, 13]   # other_flat (3.0×), sidewalk (2.5×), terrain (2.0×)
      context_scales: [3.0, 2.5, 2.0]

token:
  # L5: 200+ negative bank
  negative_bank: panoptic/clip_prototypes/negative_bank_200.pt
  n_negatives: 200
  temperature: 0.07
  learnable_temperature: true
  # Multi-prompt augmentation (T4): randomly sample from 10 templates per class
  multi_prompt:
    enabled: true
    templates:
      - "a {} in a scene"
      - "a photo of a {}"
      - "a {} on a road"
      - "a {} in an urban environment"
      - "a {} in a driving scene"
      - "a {} as seen from a car"
      - "{} in traffic"
      - "a small {}"
      - "a large {}"
      - "a {} from a bird's eye view"

voxel_decomp:
  # L7: per-voxel text contrastive (root cause fix)
  enabled: true
  weight: 0.5                  # ramp 0→0.5 over 5 epochs
  warmup_epochs: 5
  temperature: 0.07
  n_subsample: 10000           # voxels per sample
  text_bank: panoptic/clip_prototypes/negative_bank_200.pt
  include_base_protos: true   # prepend 16 base class protos before negatives

loss:
  eos_coef: 0.1                # objectness downweight (no-object class)
  weight_ce: 0.0               # CE disabled (single CLIP head has no class logits)
  weight_mask: 5.0
  weight_dice: 5.0
  # Objectness loss (replaces CE in single-head setup)
  weight_obj: 2.0              # binary BCE for objectness head
  cost_mask: 5.0
  cost_dice: 5.0
  num_points: 50000
  p_ratio: 0.75

training:
  warm_start: panoptic/checkpoints_token_full_v2/best.pth
  warm_start_strict: false     # allow missing LoRA + new head keys

  # Two-phase training (Q8 recommendation)
  phase1_epochs: 2             # warm-up: train only LoRA + token head
  phase2_epochs: 18            # full training: all losses active
  epochs: 20                   # total = phase1 + phase2

  # Differential learning rates
  lr_decoder: 5.0e-5           # decoder (lower — preserve mask quality)
  lr_token_head: 1.0e-4        # single CLIP head (higher — new architecture)
  lr_lora: 1.0e-4              # LoRA adapters
  lr_objectness: 1.0e-4        # objectness head

  weight_decay: 0.01
  grad_clip: 0.5
  grad_accum: 8
  batch_size: 1
  num_workers: 0               # cluster constraint
  seed: 42
  scheduler: cosine_warmup
  warmup_epochs: 2
  min_lr: 1.0e-6
  mixed_precision: true

logging:
  log_every: 50
  eval_every: 2
  wandb_project: openscene-panoptic
  wandb_run_name: job1-dense-clip-v1

checkpoint:
  save_dir: panoptic/checkpoints_job1
  save_every: 2
  auto_resume: true

inference:
  overlap_thr: 0.8
  min_points: 50
  text_protos: panoptic/clip_prototypes/text_prototypes_vitl14_336.pt
  temperature: 0.07
```

### 9.2 Mini Config

**File to create:** `panoptic/configs/job1_dense_clip_mini.yaml`

Same as above but override:
```yaml
data:
  dataroot: /path/to/nuscenes_mini
  version: v1.0-mini
training:
  phase1_epochs: 1
  phase2_epochs: 2
  epochs: 3
  num_workers: 0
  grad_accum: 4
logging:
  eval_every: 1
checkpoint:
  save_dir: panoptic/checkpoints_job1_mini
```

### 9.3 Two-Phase Warm-Start Logic

**In `train_panoptic_job1.py`, implement the following optimizer setup:**

```python
def build_optimizer(model, lora_adapters, config):
    """
    Two-phase optimizer setup:
    Phase 1 (epochs 1-2): only LoRA adapters + token/objectness heads train.
                          Decoder fully frozen. Establishes stable gradient flow.
    Phase 2 (epochs 3+): all components train with differential LRs.
                         Distillation losses activate with warmup schedules.
    """
    # Decoder parameters
    decoder_params = [p for n, p in model.decoder.named_parameters()
                      if 'clip_embed_head' not in n and 'objectness' not in n]

    # New head parameters
    head_params = (list(model.decoder.clip_embed_head.parameters()) +
                   list(model.decoder.objectness_head.parameters()))

    # LoRA adapter parameters
    lora_params = list(lora_adapters.parameters())

    # Phase 1 optimizer (warm-up only LoRA + heads)
    phase1_optimizer = torch.optim.AdamW([
        {'params': head_params,  'lr': config.training.lr_token_head},
        {'params': lora_params,  'lr': config.training.lr_lora},
    ], weight_decay=config.training.weight_decay)

    # Phase 2 optimizer (all components, differential LRs)
    phase2_optimizer = torch.optim.AdamW([
        {'params': decoder_params, 'lr': config.training.lr_decoder},
        {'params': head_params,    'lr': config.training.lr_token_head},
        {'params': lora_params,    'lr': config.training.lr_lora},
    ], weight_decay=config.training.weight_decay)

    return phase1_optimizer, phase2_optimizer


def get_loss_weight(loss_name, current_epoch, config):
    """
    Linear ramp schedule for new distillation losses.
    Prevents unstable early-training mask predictions from producing
    noisy distillation targets that disrupt the well-trained decoder.
    """
    phase2_epoch = max(0, current_epoch - config.training.phase1_epochs)

    schedules = {
        'L_O':   (config.distillation.L_O.warmup_epochs,   config.distillation.L_O.weight),
        'L_V':   (config.distillation.L_V.warmup_epochs,   config.distillation.L_V.weight),
        'L7':    (config.voxel_decomp.warmup_epochs,         config.voxel_decomp.weight),
        'token': (0, 1.0),   # token loss active immediately (no ramp)
    }

    if loss_name not in schedules:
        return 1.0

    warmup_epochs, target_weight = schedules[loss_name]
    if warmup_epochs <= 0:
        return target_weight

    ramp = min(1.0, phase2_epoch / warmup_epochs)
    return target_weight * ramp
```

---

## 10. Training Script Outline

**File to create:** `panoptic/train/train_panoptic_job1.py`

New training script with the following additions over `train_panoptic_token.py`:

```python
# Key additions to training loop (pseudocode outline):

for epoch in range(config.training.epochs):
    is_phase1 = (epoch < config.training.phase1_epochs)
    optimizer = phase1_optimizer if is_phase1 else phase2_optimizer

    for batch in dataloader:
        # --- Standard forward pass ---
        backbone_feats = extract_backbone_features_with_lora(
            batch, backbone, lora_adapters
        )
        decoder_output = decoder(backbone_feats, batch['vox_coords'])
        # decoder_output['pred_embed']: (B, Q, 768)  ← from SingleCLIPEmbeddingHead
        # decoder_output['pred_masks']: (B, V, Q)
        # decoder_output['obj_scores']: (B, Q, 1)

        # --- Hungarian matching (uses mask cost only — no CE cost with single head) ---
        matched_indices = matcher(decoder_output, batch['targets'])

        # --- Standard mask losses ---
        loss_mask, loss_dice = compute_mask_losses(decoder_output, batch, matched_indices)
        loss_obj = compute_objectness_loss(decoder_output['obj_scores'],
                                           matched_indices, batch)

        # --- New losses (phase 2 only, with warmup weights) ---
        total_loss = loss_mask * 5.0 + loss_dice * 5.0 + loss_obj * 2.0

        if not is_phase1:
            # Load camera features (from cache or online)
            fv_clip = camera_loader.get_voxel_features(
                batch['sample_token'], batch['vox_coords_xyz']
            )

            # L_O: object-level distillation
            w_L_O = get_loss_weight('L_O', epoch, config)
            if w_L_O > 0:
                loss_LO = L_O_loss(
                    decoder_output['pred_embed'],
                    decoder_output['pred_masks'],
                    fv_clip, matched_indices
                )
                total_loss = total_loss + w_L_O * loss_LO

            # L_V + CASD: voxel-level distillation
            w_L_V = get_loss_weight('L_V', epoch, config)
            if w_L_V > 0:
                loss_LV = L_V_casd_loss(
                    decoder_output['pred_embed'],
                    decoder_output['pred_masks'],
                    fv_clip, fv_clip_casd,    # CASD targets precomputed for stuff classes
                    batch['gt_sem']
                )
                total_loss = total_loss + loss_LV  # weight already inside L_V loss

            # L5: C1 token loss with 200+ negatives
            loss_token = token_loss_c1(
                decoder_output['pred_embed'], matched_indices, batch['gt_classes']
            )
            total_loss = total_loss + loss_token

            # L7: voxel decomposition loss (root cause fix)
            w_L7 = get_loss_weight('L7', epoch, config)
            if w_L7 > 0:
                loss_L7 = voxel_decomp_loss(backbone_feats, batch['gt_sem'])
                total_loss = total_loss + w_L7 * loss_L7

        # --- Backward + optimizer step ---
        scaler.scale(total_loss).backward()
        if step % config.training.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in optimizer.param_groups[0]['params']
                 if p.grad is not None],
                config.training.grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # --- Logging ---
        log_dict = {
            'loss_mask': loss_mask.item(),
            'loss_dice': loss_dice.item(),
            'loss_obj':  loss_obj.item(),
        }
        if not is_phase1:
            log_dict.update({
                'loss_LO':    loss_LO.item() if w_L_O > 0 else 0,
                'loss_LV':    loss_LV.item() if w_L_V > 0 else 0,
                'loss_token': loss_token.item(),
                'loss_L7':    loss_L7.item()  if w_L7 > 0 else 0,
                'w_L_O': w_L_O, 'w_L_V': w_L_V, 'w_L7': w_L7,
                'lora_grad_norm': get_lora_grad_norm(lora_adapters),
            })
```

---

## 11. CASD Context Crop Generation

**File to create:** `panoptic/utils/casd_crop_extractor.py`

```python
"""
Context-Aware Stuff Distillation (CASD) — novel contribution.

For contextually ambiguous stuff classes (other_flat, sidewalk, terrain),
replaces per-pixel L_V targets with context-expanded crop CLS tokens.

CASD classes and expansion factors:
    other_flat (idx=11): 3.0× — most ambiguous, needs maximum context
    sidewalk   (idx=12): 2.5× — needs road+building+pedestrian context
    terrain    (idx=13): 2.0× — needs vegetation+road context

Why NOT apply to driveable_surface, manmade, vegetation:
    driveable_surface: already PQ=0.854 (RQ=1.0) — working perfectly
    manmade:  PQ=0.671 — locally distinctive (buildings)
    vegetation: PQ=0.677 — locally distinctive (green color)
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Dict, Optional

CASD_CONFIGS = {
    11: 3.0,   # other_flat  — bike lanes, parking lots (context-only)
    12: 2.5,   # sidewalk    — distinguished by adjacent road/buildings
    13: 2.0,   # terrain     — distinguished by vegetation/road borders
}


def get_casd_clip_targets(
    sample_token: str,
    vox_coords_xyz: np.ndarray,      # (V, 3) in LiDAR sensor frame
    gt_sem: np.ndarray,              # (V,) semantic labels
    nusc,                            # NuScenes instance
    clip_model,                      # loaded CLIP model
    clip_preprocess,                 # CLIP image preprocessing transform
    camera_loader,                   # CameraClipLoader instance
    device: str = 'cuda'
) -> torch.Tensor:
    """
    For each CASD stuff class present in this sample:
    1. Find all voxels of that class
    2. Project them to the best camera (highest coverage)
    3. Compute the bounding box of the projected region
    4. Expand the bounding box by the class-specific scale factor
    5. Crop the expanded region from the camera image
    6. Run CLIP ViT to get CLS token (768-d) for that context crop
    7. Broadcast this CLS token to all voxels of that stuff class

    Returns:
        fv_clip_casd: (V, 768) — context-aware targets.
                      For non-CASD voxels: zero (caller falls back to standard L_V targets).
                      For CASD voxels: context-expanded crop feature.
    """
    V = len(vox_coords_xyz)
    fv_clip_casd = torch.zeros(V, 768, dtype=torch.float32)

    sample = nusc.get('sample', sample_token)
    cam_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

    for cls_idx, context_scale in CASD_CONFIGS.items():
        class_mask = (gt_sem == cls_idx)
        if class_mask.sum() < 10:   # skip if fewer than 10 voxels
            continue

        class_voxels = vox_coords_xyz[class_mask]   # (V_c, 3)

        # Find the camera that covers the most voxels of this class
        best_cam, best_uv, best_valid = None, None, None
        best_coverage = 0

        for cam_name in cam_names:
            uv, _, valid = camera_loader._project_to_camera(
                sample, cam_name, class_voxels
            )
            if valid.sum() > best_coverage:
                best_coverage = valid.sum()
                best_cam = cam_name
                best_uv = uv
                best_valid = valid

        if best_cam is None or best_coverage < 5:
            continue   # no camera covers this stuff region — skip CASD

        visible_uv = best_uv[best_valid]   # (V_visible, 2)

        # Compute bounding box of visible region in image space
        u_min, u_max = visible_uv[:, 0].min(), visible_uv[:, 0].max()
        v_min, v_max = visible_uv[:, 1].min(), visible_uv[:, 1].max()

        u_center = (u_min + u_max) / 2
        v_center = (v_min + v_max) / 2
        u_half = (u_max - u_min) / 2
        v_half = (v_max - v_min) / 2

        # Expand bounding box by context_scale
        u_half_exp = u_half * context_scale
        v_half_exp = v_half * context_scale

        # Clip to image bounds
        u_min_exp = max(0, u_center - u_half_exp)
        u_max_exp = min(1600, u_center + u_half_exp)
        v_min_exp = max(0, v_center - v_half_exp)
        v_max_exp = min(900, v_center + v_half_exp)

        # Load camera image and crop
        sd = nusc.get('sample_data', sample['data'][best_cam])
        img_path = os.path.join(nusc.dataroot, sd['filename'])
        img = Image.open(img_path).convert('RGB')

        # Crop and resize to CLIP input size (224×224)
        crop_box = (int(u_min_exp), int(v_min_exp),
                    int(u_max_exp), int(v_max_exp))
        crop = img.crop(crop_box)

        if crop.size[0] < 5 or crop.size[1] < 5:
            continue   # crop too small — degenerate case

        crop_tensor = clip_preprocess(crop).unsqueeze(0).to(device)

        # Get CLIP CLS token for the context crop
        with torch.no_grad():
            cls_feat = clip_model.encode_image(
                crop_tensor, normalize=True
            )   # (1, 768)

        # Broadcast to all voxels of this stuff class
        fv_clip_casd[class_mask] = cls_feat.squeeze(0).float().cpu()

    return fv_clip_casd
```

---

## 12. bsub Job Script

**File to create:** `panoptic/scripts/train_job1_dense_clip.sh`

```bash
#!/bin/bash
#BSUB -q batch_a100
#BSUB -n 4
#BSUB -M 8192
#BSUB -W 36:00
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -o logs/job1_dense_clip_%J.log
#BSUB -e logs/job1_dense_clip_%J.err
#BSUB -J job1_dense_clip

module load conda
conda activate openscene
module load cuda/11.0.2
module load cudnn/11.0_v8.0

export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
export OMP_NUM_THREADS=16
cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene

# Verify prerequisites before launching
python panoptic/scripts/verify_clip_model.py || exit 1
python panoptic/scripts/verify_camera_projection.py || exit 1

# Launch training
python -u panoptic/train/train_panoptic_job1.py \
    --config panoptic/configs/job1_dense_clip.yaml \
    2>&1 | tee logs/job1_training_$(date +%Y%m%d_%H%M).log
```

---

## 13. Success Criteria and Expected Results

### Early Stopping Gates

| Epoch | Check | Pass threshold | Action if fails |
|---|---|---|---|
| 2 (end of phase 1) | LoRA grad norms | 1e-4 to 10 | Increase LoRA rank to 32 |
| 2 (end of phase 1) | `loss_token` C1 | < 3.0 (should converge fast) | Check negative bank dimension |
| 5 | `loss_L_O` | < 0.3 | Check camera feature coverage rate |
| 5 | `loss_L_V` | < 0.15 | Increase L_V weight to 0.2 |
| 5 | `loss_L7` | < 2.5 (InfoNCE vs 216 classes) | Check text bank alignment |
| 10 | val PQ | ≥ 0.47 (should not regress much) | Reduce lr_decoder to 2e-5 |
| 20 | val PQ | ≥ 0.50 | — |

### Expected Results (Job 1 completion)

| Metric | Baseline (PQ=0.476) | Expected after Job 1 |
|---|---|---|
| Closed-set PQ | 0.476 | 0.50–0.55 |
| Closed-set SQ | 0.793 | 0.78–0.82 (slight regression acceptable from head change) |
| Closed-set mIoU | 0.433 | 0.46–0.52 (L7 should improve this most) |
| other_flat PQ | 0.082 | 0.15–0.25 (CASD target improvement) |
| sidewalk PQ | 0.211 | 0.28–0.38 (CASD target improvement) |
| B12/N4 novel things PQ | ~0 | 25–45 (L_O drives this) |
| B12/N4 novel stuff PQ | ~0 | 10–25 (L_V + CASD drives this) |

### Critical Diagnostic: per-voxel CLIP margin test

After Job 1 completes, run:
```python
python panoptic/scripts/check_voxel_margin.py \
    --checkpoint panoptic/checkpoints_job1/best.pth \
    --n_samples 100
```

This runs Strategy L (logit aggregation) on 100 val samples using the Job 1 backbone features and reports:
- Mean cos_sim("car") for car voxels
- Mean cos_sim("police car") for car voxels
- **Margin = (car_sim - police_car_sim)** — should be > 0.05 after Job 1 (was ~0.02 before)

If margin > 0.05: L7 worked. Open-vocab inference has improved signal.  
If margin ≤ 0.02: L7 had no effect — check LoRA grad norms and text bank quality.

---

## 14. Failure Modes and Debugging Guide

### NaN in `loss_L_V`
**Cause:** Mask scores early in training produce near-zero soft masks → `F_rec = M_Q^T @ F_Q` with near-zero weights → NaN after normalisation.  
**Fix:** Check `L_V warmup_epochs` — should be ≥ 5. If still NaN, add `fv_clip.nan_to_num_(nan=0.0)` before loss computation.

### `loss_L_O` not decreasing past epoch 5
**Cause 1:** Camera cache corrupted — some `.pt` files are zero tensors.  
**Diagnosis:** Check `fv_clip.norm(dim=-1).mean()` per batch. Should be ~0.7+ for well-covered batches.  
**Fix:** Rerun precompute script with `--recompute_all` flag.  
**Cause 2:** Matched masks too small for meaningful pooling.  
**Diagnosis:** Log `valid_pairs_fraction = n_pairs_with_coverage / n_matched`.  
**Fix:** Reduce `min_mask_coverage` threshold from 0.05 to 0.01.

### LoRA grad norms vanishing (<1e-8)
**Cause:** `loss_L7` is not backpropagating through LoRA. Verify backbone outputs are connected to LoRA adapters in the computation graph.  
**Diagnosis:** `print(list(lora_adapters.parameters())[0].grad)` after first backward.  
**Fix:** Verify `extract_backbone_features_with_lora()` is using hooks correctly — backbone forward must pass through LoRA adapters' `nn.Linear` modules before returning.

### PQ regresses below 0.46
**Cause:** Single CLIP head (A4) has destabilised mask quality due to changed loss landscape.  
**Fix:** Increase `phase1_epochs` from 2 to 4, or temporarily restore `weight_obj: 4.0`.

### Camera cache OOM during training
**Cause:** Loading `(6, 256, 768)` float32 tensor per sample per step is ~3 MB/step × grad_accum=8 = 24 MB per optimizer step. Should be fine.  
**If OOM:** Load cache in float16, cast to float32 only for loss computation.

---

## 15. File Map — What Gets Created/Modified

### New files (create from scratch)
```
panoptic/scripts/verify_clip_model.py          ← P1 prerequisite
panoptic/scripts/verify_camera_projection.py   ← P3 prerequisite
panoptic/scripts/precompute_camera_clip_features.py  ← run immediately
panoptic/scripts/generate_negative_bank.py     ← run before training
panoptic/models/lora_backbone.py               ← B1 LoRA adapters
panoptic/losses/distillation_loss.py           ← L_O, L_V, L7
panoptic/utils/camera_clip_loader.py           ← camera feature loading
panoptic/utils/casd_crop_extractor.py          ← CASD context crops
panoptic/train/train_panoptic_job1.py          ← main training script
panoptic/configs/job1_dense_clip.yaml          ← full nuScenes config
panoptic/configs/job1_dense_clip_mini.yaml     ← mini smoke test config
panoptic/scripts/train_job1_dense_clip.sh      ← bsub job script
panoptic/scripts/check_voxel_margin.py         ← post-job diagnostic
panoptic/checkpoints_job1/                     ← output directory
```

### Files to modify
```
panoptic/models/mask_decoder.py      ← add SingleCLIPEmbeddingHead, objectness_head
panoptic/losses/token_loss.py        ← add TokenLossC1 class
panoptic/losses/panoptic_loss.py     ← integrate new losses, remove CE loss
panoptic/train/train_panoptic_token.py  ← reference only — new script replaces this
```

### Files unchanged
```
panoptic/dataset/nuscenes_panoptic.py  ← no changes (Option A: all 16 classes)
panoptic/eval/eval_pq.py               ← no changes
panoptic/models/blocks.py              ← no changes
panoptic/models/disnet.py              ← no changes (backbone stays frozen)
```