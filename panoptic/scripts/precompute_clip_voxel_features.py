"""Precompute per-voxel CLIP features from camera images for Job 1b.

Strategy B: Dense single-scale with 50% overlap
  - Object-level: mask-pooled CLIP features per GT instance (for L_O)
  - Voxel-level: 4096 randomly sampled valid voxels with CLIP features (for L_V)

Multi-tile: each 1600x900 camera image is tiled into overlapping 224x224 crops
(stride=112, 50% overlap), processed by frozen CLIP ViT-L/14, and stitched into
a dense feature map at patch resolution (14px per patch). Each pixel gets 2-4
contributing tiles for smoother, higher-quality features.

Output per sample: {sample_token}.npz with:
  obj_feats:   (N_inst, 768) float16   mask-pooled CLIP per GT instance
  obj_classes: (N_inst,) int32         GT class per instance
  vox_feats:   (4096, 768) float16     sampled voxel CLIP features
  vox_indices: (4096,) int32           voxel indices (into un-augmented voxel array)
  n_valid:     int                     total valid voxels with projection

Usage:
    python panoptic/scripts/precompute_clip_voxel_features.py \
        --dataroot /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes \
        --version v1.0-trainval \
        --output_dir /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features_v1_diag \
        --device cuda
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene")

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.splits import create_splits_scenes
from panoptic.scripts.verify_camera_projection import project_lidar_to_camera

# Constants (must match nuscenes_panoptic.py exactly)
DEFAULT_RANGE = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
CAMERAS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]
CAM_W, CAM_H = 1600, 900

# Panoptic label mapping (from nuscenes_panoptic.py)
GENERAL_TO_PANOPTIC_16 = np.zeros(32, dtype=np.int32)
GENERAL_TO_PANOPTIC_16[9]  = 1   # barrier
GENERAL_TO_PANOPTIC_16[14] = 2   # bicycle
GENERAL_TO_PANOPTIC_16[15] = 3   # bus.bendy
GENERAL_TO_PANOPTIC_16[16] = 3   # bus.rigid
GENERAL_TO_PANOPTIC_16[17] = 4   # car
GENERAL_TO_PANOPTIC_16[18] = 5   # construction
GENERAL_TO_PANOPTIC_16[21] = 6   # motorcycle
GENERAL_TO_PANOPTIC_16[2]  = 7   # pedestrian.adult
GENERAL_TO_PANOPTIC_16[3]  = 7   # pedestrian.child
GENERAL_TO_PANOPTIC_16[4]  = 7   # pedestrian.construction_worker
GENERAL_TO_PANOPTIC_16[6]  = 7   # pedestrian.police_officer
GENERAL_TO_PANOPTIC_16[12] = 8   # trafficcone
GENERAL_TO_PANOPTIC_16[22] = 9   # trailer
GENERAL_TO_PANOPTIC_16[23] = 10  # truck
GENERAL_TO_PANOPTIC_16[24] = 11  # driveable_surface
GENERAL_TO_PANOPTIC_16[25] = 12  # other_flat
GENERAL_TO_PANOPTIC_16[26] = 13  # sidewalk
GENERAL_TO_PANOPTIC_16[27] = 14  # terrain
GENERAL_TO_PANOPTIC_16[28] = 15  # manmade
GENERAL_TO_PANOPTIC_16[30] = 16  # vegetation

THING_IDS = list(range(1, 11))
STUFF_IDS = list(range(11, 17))

N_VOX_SAMPLE = 4096  # number of voxels to sample for L_V


class CLIPPatchExtractor:
    """Extract dense CLIP patch features from images using multi-tile approach."""

    def __init__(self, device="cuda", tile_size=224, stride=112):
        import clip
        self.device = device
        self.tile_size = tile_size
        self.stride = stride

        # Load CLIP
        self.model, self.preprocess = clip.load("ViT-L/14", device=device, jit=False)
        self.model.eval()

        # Hook to capture patch tokens from last resblock
        self._patch_tokens = None
        self.model.visual.transformer.resblocks[-1].register_forward_hook(self._hook)

        # Precompute normalization (from CLIP preprocess)
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                                 device=device).half().view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                                device=device).half().view(1, 3, 1, 1)

    def _hook(self, module, input, output):
        self._patch_tokens = output

    def _preprocess_tile(self, tile_img):
        """Preprocess a 224x224 PIL tile to CLIP input tensor."""
        # Convert to tensor, normalize
        tile_t = torch.from_numpy(np.array(tile_img)).float().permute(2, 0, 1) / 255.0
        tile_t = tile_t.unsqueeze(0).to(self.device)
        tile_t = (tile_t - self.mean) / self.std
        return tile_t.half()  # CLIP model expects float16

    def extract_dense_features(self, image_path, batch_size=32):
        """Extract dense CLIP features for a full camera image.

        Returns:
            feat_map: (H_patches, W_patches, 768) numpy array
            H_patches, W_patches: dimensions of patch grid
        """
        img = Image.open(image_path).convert("RGB")
        img_w, img_h = img.size  # 1600, 900

        # Compute tile grid
        cols = list(range(0, img_w - self.tile_size + 1, self.stride))
        if cols[-1] + self.tile_size < img_w:
            cols.append(img_w - self.tile_size)
        rows = list(range(0, img_h - self.tile_size + 1, self.stride))
        if rows[-1] + self.tile_size < img_h:
            rows.append(img_h - self.tile_size)

        # Dense feature map at pixel-to-patch resolution
        # Each patch covers 14x14 pixels in the original tile
        patch_px = 14  # 224 / 16
        H_patches = (img_h + patch_px - 1) // patch_px
        W_patches = (img_w + patch_px - 1) // patch_px

        feat_accum = np.zeros((H_patches, W_patches, 768), dtype=np.float32)
        feat_count = np.zeros((H_patches, W_patches), dtype=np.int32)

        # Collect all tiles
        tiles = []
        tile_positions = []
        for r in rows:
            for c in cols:
                tile = img.crop((c, r, c + self.tile_size, r + self.tile_size))
                tiles.append(tile)
                tile_positions.append((r, c))

        # Process in batches
        for batch_start in range(0, len(tiles), batch_size):
            batch_tiles = tiles[batch_start:batch_start + batch_size]
            batch_pos = tile_positions[batch_start:batch_start + batch_size]

            # Stack tiles
            batch_tensor = torch.cat(
                [self._preprocess_tile(t) for t in batch_tiles], dim=0
            )

            # Forward through CLIP visual
            with torch.no_grad():
                _ = self.model.visual(batch_tensor)
                # _patch_tokens shape: (seq_len=257, batch, 1024)
                raw = self._patch_tokens
                # Skip CLS token, get patches
                patches = raw[1:, :, :]  # (256, B, 1024)
                patches = patches.permute(1, 0, 2)  # (B, 256, 1024)
                # Apply ln_post + proj
                patches = self.model.visual.ln_post(patches)
                patches = patches @ self.model.visual.proj  # (B, 256, 768)
                # L2 normalize
                patches = F.normalize(patches, dim=-1)
                patches = patches.cpu().numpy()  # (B, 256, 768)

            # Scatter into dense map
            for i, (r, c) in enumerate(batch_pos):
                patch_grid = patches[i].reshape(16, 16, 768)  # (16, 16, 768)
                for pi in range(16):
                    for pj in range(16):
                        # Pixel range this patch covers in original image
                        py = r + pi * patch_px
                        px = c + pj * patch_px
                        # Map to dense grid
                        gy = py // patch_px
                        gx = px // patch_px
                        if gy < H_patches and gx < W_patches:
                            feat_accum[gy, gx] += patch_grid[pi, pj]
                            feat_count[gy, gx] += 1

        # Average overlapping contributions
        valid_patches = feat_count > 0
        feat_accum[valid_patches] /= feat_count[valid_patches, np.newaxis]

        return feat_accum, H_patches, W_patches


def voxelize_sample(nusc, sample_token, dataroot, voxel_size=0.05):
    """Voxelize a sample identically to NuScenesPanoptic (no augmentation).

    Returns:
        vox_xyz: (V, 3) voxel XYZ coordinates (for projection)
        vox_idx: (V,) indices of representative points
        inv_idx: (N,) point-to-voxel mapping
        sem: (N,) panoptic labels (1-indexed, 0=ignore)
        inst: (N,) instance IDs
        N: number of points after range filter
    """
    sample = nusc.get("sample", sample_token)
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    pc = LidarPointCloud.from_file(os.path.join(dataroot, lidar_sd["filename"]))
    pts = pc.points.T.astype(np.float32)  # (N, 5)

    # Load panoptic labels
    lidar_token = lidar_sd["token"]
    pan_path = os.path.join(dataroot, "panoptic", nusc.version, f"{lidar_token}_panoptic.npz")
    pan_data_full = np.load(pan_path)["data"]

    # Range filter
    r = DEFAULT_RANGE
    keep = (
        (pts[:, 0] >= r[0]) & (pts[:, 0] <= r[3]) &
        (pts[:, 1] >= r[1]) & (pts[:, 1] <= r[4]) &
        (pts[:, 2] >= r[2]) & (pts[:, 2] <= r[5])
    )
    pts = pts[keep]
    pan_data = pan_data_full[keep]
    N = pts.shape[0]

    # Decode panoptic
    sem = GENERAL_TO_PANOPTIC_16[(pan_data // 1000).astype(np.int32).clip(0, 31)]
    inst = (pan_data % 1000).astype(np.int32)

    # Voxelize (NO augmentation)
    coords_f = pts[:, :3] / voxel_size
    coords_i = np.floor(coords_f).astype(np.int32)
    _, vox_idx, inv_idx = np.unique(coords_i, axis=0, return_index=True, return_inverse=True)

    vox_xyz = pts[vox_idx, :3]  # (V, 3)

    return vox_xyz, vox_idx, inv_idx, sem, inst, N


def build_gt_masks(sem, inst, inv_idx, V):
    """Build GT instance masks (same logic as dataset).

    Returns list of (mask, class_id) tuples.
    """
    instances = []

    # Thing instances
    for sid in THING_IDS:
        pt_mask_class = sem == sid
        if not pt_mask_class.any():
            continue
        for iid in np.unique(inst[pt_mask_class]):
            if iid == 0:
                continue
            pt_mask = (sem == sid) & (inst == iid)
            vox_mask = np.zeros(V, dtype=bool)
            vox_mask[inv_idx[pt_mask]] = True
            if vox_mask.sum() < 1:
                continue
            instances.append((vox_mask, sid - 1))  # 0-indexed class

    # Stuff classes
    for sid in STUFF_IDS:
        pt_mask = sem == sid
        if not pt_mask.any():
            continue
        vox_mask = np.zeros(V, dtype=bool)
        vox_mask[inv_idx[pt_mask]] = True
        if vox_mask.sum() < 1:
            continue
        instances.append((vox_mask, sid - 1))  # 0-indexed class

    return instances


def process_sample(nusc, sample_token, dataroot, extractor, device):
    """Process one nuScenes sample: extract and save CLIP features."""

    # 1. Voxelize
    vox_xyz, vox_idx, inv_idx, sem, inst, N = voxelize_sample(
        nusc, sample_token, dataroot
    )
    V = len(vox_idx)

    # 2. Extract dense CLIP features for all 6 cameras
    sample = nusc.get("sample", sample_token)
    cam_feat_maps = {}
    for cam_name in CAMERAS:
        cam_sd = nusc.get("sample_data", sample["data"][cam_name])
        cam_path = os.path.join(dataroot, cam_sd["filename"])
        feat_map, H_p, W_p = extractor.extract_dense_features(cam_path)
        cam_feat_maps[cam_name] = (feat_map, H_p, W_p)

    # 3. Project voxels to cameras, assign features
    voxel_feats_accum = np.zeros((V, 768), dtype=np.float32)
    voxel_cam_count = np.zeros(V, dtype=np.int32)
    patch_px = 14

    for cam_name in CAMERAS:
        uv, depths, valid = project_lidar_to_camera(
            nusc, sample_token, vox_xyz, cam_name
        )
        feat_map, H_p, W_p = cam_feat_maps[cam_name]

        for vi in np.where(valid)[0]:
            u, v = int(uv[vi, 0]), int(uv[vi, 1])
            gy = min(v // patch_px, H_p - 1)
            gx = min(u // patch_px, W_p - 1)
            voxel_feats_accum[vi] += feat_map[gy, gx]
            voxel_cam_count[vi] += 1

    # Average multi-camera, L2-normalize
    valid_voxels = voxel_cam_count > 0
    voxel_feats_accum[valid_voxels] /= voxel_cam_count[valid_voxels, np.newaxis]
    norms = np.linalg.norm(voxel_feats_accum[valid_voxels], axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    voxel_feats_accum[valid_voxels] /= norms

    n_valid = int(valid_voxels.sum())

    # 4. Object-level: mask-pool per GT instance
    # Save ALL GT instances in original order with obj_valid flag.
    # Instances with no camera-visible voxels get zero-norm feat + valid=False.
    # This lets Script A compute true coverage = valid.sum() / len(gt_instances).
    gt_instances = build_gt_masks(sem, inst, inv_idx, V)
    n_inst = len(gt_instances)

    obj_feats   = np.zeros((n_inst, 768), dtype=np.float16)
    obj_classes = np.zeros(n_inst,        dtype=np.int32)
    obj_valid   = np.zeros(n_inst,        dtype=np.bool_)

    for k, (mask, cls_id) in enumerate(gt_instances):
        obj_classes[k] = cls_id
        mask_valid = mask & valid_voxels
        if mask_valid.sum() < 1:
            continue
        pooled = voxel_feats_accum[mask_valid].mean(axis=0)
        pooled = pooled / max(np.linalg.norm(pooled), 1e-8)
        obj_feats[k]  = pooled.astype(np.float16)
        obj_valid[k]  = True

    # 5. Voxel-level: sample 4096 valid voxels
    valid_indices = np.where(valid_voxels)[0]
    if len(valid_indices) >= N_VOX_SAMPLE:
        sampled = np.random.choice(valid_indices, N_VOX_SAMPLE, replace=False)
    elif len(valid_indices) > 0:
        # Pad with repetition if fewer valid voxels
        sampled = np.random.choice(valid_indices, N_VOX_SAMPLE, replace=True)
    else:
        sampled = np.zeros(N_VOX_SAMPLE, dtype=np.int64)

    vox_feats = voxel_feats_accum[sampled].astype(np.float16)
    vox_indices = sampled.astype(np.int32)

    return {
        "obj_feats":   obj_feats,
        "obj_classes": obj_classes,
        "obj_valid":   obj_valid,
        "vox_feats":   vox_feats,
        "vox_indices": vox_indices,
        "n_valid":     n_valid,
        "n_voxels":    np.int32(V),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--output_dir", default="/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features_v1_diag")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", default="train", help="train or val or both")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--resume", action="store_true", help="Skip already processed samples")
    args = parser.parse_args()

    # Output directory
    out_dir = os.path.join(args.output_dir, args.version)
    os.makedirs(out_dir, exist_ok=True)

    # Load nuScenes
    print(f"Loading nuScenes {args.version} from {args.dataroot}...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    # Get sample tokens for split
    splits = create_splits_scenes()
    if "mini" in args.version:
        if args.split == "both":
            scene_names = set(splits["mini_train"]) | set(splits["mini_val"])
        else:
            key = f"mini_{args.split}"
            scene_names = set(splits[key])
    else:
        if args.split == "both":
            scene_names = set(splits["train"]) | set(splits["val"])
        else:
            scene_names = set(splits[args.split])

    tokens = [
        s["token"] for s in nusc.sample
        if nusc.get("scene", s["scene_token"])["name"] in scene_names
    ]
    print(f"Processing {len(tokens)} samples ({args.split} split)")

    # Initialize CLIP extractor
    extractor = CLIPPatchExtractor(device=args.device, tile_size=224, stride=112)

    # Process
    skipped = 0
    for i, token in enumerate(tqdm(tokens, desc="Precomputing CLIP features")):
        out_path = os.path.join(out_dir, f"{token}.npz")
        if args.resume and os.path.exists(out_path):
            skipped += 1
            continue

        try:
            result = process_sample(nusc, token, args.dataroot, extractor, args.device)
            np.savez_compressed(out_path, **result)
        except Exception as e:
            print(f"\nERROR on sample {token[:16]}: {e}")
            continue

        if (i + 1) % 100 == 0:
            print(f"\n  Processed {i+1}/{len(tokens)}, "
                  f"last n_valid={result['n_valid']}, "
                  f"obj_feats={result['obj_feats'].shape}")

    print(f"\nDone. Saved to {out_dir}/")
    if skipped:
        print(f"  Skipped {skipped} already-processed samples (--resume)")


if __name__ == "__main__":
    main()
