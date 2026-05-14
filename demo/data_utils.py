"""Data utilities for NuScenes mini scene/sample navigation.

Provides:
  - NuScenes instance loading (lru_cache — one instance per dataroot+version)
  - Scene listing filtered by split
  - Ordered sample listing for a scene
  - Single-sample LiDAR loading + voxelization (no GT masks needed for demo)

The voxelization logic mirrors NuScenesPanoptic.__getitem__ exactly so that
vox_coords/vox_feats format matches what inference.extract_backbone_features
expects.
"""

import os
from functools import lru_cache

import numpy as np
import torch

# Camera order for 2x3 display grid (row1: front, row2: back)
CAMERAS_ORDERED = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]

DEFAULT_RANGE = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], dtype=np.float32)


@lru_cache(maxsize=4)
def load_nusc(dataroot, version):
    """Load and cache a NuScenes instance (keyed by dataroot + version string).

    lru_cache keeps the object alive for the Python process lifetime, so
    Streamlit rerenders reuse the same in-memory index.
    """
    from nuscenes.nuscenes import NuScenes
    print(f"[NuScenes] Loading {version} from {dataroot} ...", flush=True)
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    print(
        f"[NuScenes] Ready. {len(nusc.scene)} scenes, {len(nusc.sample)} samples.",
        flush=True,
    )
    return nusc


def get_scenes_for_split(nusc, version, split):
    """Return list of (scene_token, scene_name, n_samples) for the given split.

    For v1.0-mini the NuScenes split keys are 'mini_train' / 'mini_val'.
    For v1.0-trainval they are 'train' / 'val'.
    """
    from nuscenes.utils.splits import create_splits_scenes

    splits = create_splits_scenes()
    if "mini" in version:
        key = "mini_train" if split == "train" else "mini_val"
    else:
        key = split

    scene_names = set(splits.get(key, []))
    result = []
    for scene in nusc.scene:
        if scene["name"] in scene_names:
            result.append((scene["token"], scene["name"], scene["nbr_samples"]))
    result.sort(key=lambda x: x[1])
    return result


def get_samples_for_scene(nusc, scene_token):
    """Return ordered list of (sample_token, timestamp_us) for a scene.

    Timestamps are in microseconds (nuScenes convention).
    """
    scene = nusc.get("scene", scene_token)
    samples = []
    token = scene["first_sample_token"]
    while token != "":
        s = nusc.get("sample", token)
        samples.append((s["token"], s["timestamp"]))
        token = s["next"]
    return samples


def load_sample_for_inference(nusc, sample_token, voxel_size=0.05, point_range=None):
    """Load a single LIDAR sweep, apply range filter, and voxelize.

    Mirrors NuScenesPanoptic.__getitem__ (without GT mask building) so that
    the returned tensors are accepted directly by inference.extract_backbone_features.

    Returns a dict:
        vox_coords:    (V, 3) int32 tensor  — voxel integer grid coordinates
        vox_feats:     (V, 3) float32 tensor — raw XYZ per voxel (LIDAR frame)
                                              same tensor used for camera projection
        cam_paths:     list of 6 str — absolute paths in CAMERAS_ORDERED order
        sample_record: nuScenes sample dict (needed for camera projection transforms)
    """
    from nuscenes.utils.data_classes import LidarPointCloud

    if point_range is None:
        point_range = DEFAULT_RANGE

    sample = nusc.get("sample", sample_token)

    # ── Load LiDAR point cloud ──────────────────────────────────────────────
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, lidar_sd["filename"]))
    pts = pc.points.T.astype(np.float32)  # (N, 5) — x, y, z, intensity, ring

    # ── Range filter ────────────────────────────────────────────────────────
    r = point_range
    keep = (
        (pts[:, 0] >= r[0]) & (pts[:, 0] <= r[3]) &
        (pts[:, 1] >= r[1]) & (pts[:, 1] <= r[4]) &
        (pts[:, 2] >= r[2]) & (pts[:, 2] <= r[5])
    )
    pts = pts[keep]

    # ── Voxelize ─────────────────────────────────────────────────────────────
    coords_f = pts[:, :3] / voxel_size
    coords_i = np.floor(coords_f).astype(np.int32)
    _, vox_idx, _ = np.unique(coords_i, axis=0, return_index=True, return_inverse=True)

    vox_coords = coords_i[vox_idx]        # (V, 3) int32
    vox_feats = pts[vox_idx, :3].copy()   # (V, 3) float32 — raw XYZ of voxel rep. points

    # ── Camera image paths (CAMERAS_ORDERED order) ───────────────────────────
    cam_paths = []
    for cam in CAMERAS_ORDERED:
        sd = nusc.get("sample_data", sample["data"][cam])
        cam_paths.append(os.path.join(nusc.dataroot, sd["filename"]))

    return {
        "vox_coords": torch.from_numpy(vox_coords).int(),     # (V, 3)
        "vox_feats": torch.from_numpy(vox_feats).float(),     # (V, 3)
        "cam_paths": cam_paths,
        "sample_record": sample,
    }
