"""Per-sample nuScenes panoptic dataset for MaskPLS-style training.

Each item = one nuScenes sample token → one LIDAR_TOP sweep + panoptic labels + 6 camera paths.
Does NOT aggregate across keyframes (unlike OpenScene semantic pipeline).

Panoptic decoding: semantic = pan // 1000, instance = pan % 1000
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.splits import create_splits_scenes

# ── nuScenes general category (0-31) → 16-class panoptic (1-16) mapping ──
# Maps general index to panoptic class (1-indexed). 0 = ignored.
# Things: 1-10, Stuff: 11-16
GENERAL_TO_PANOPTIC_16 = np.zeros(32, dtype=np.int32)  # 0 = ignore by default
GENERAL_TO_PANOPTIC_16[9]  = 1   # movable_object.barrier
GENERAL_TO_PANOPTIC_16[14] = 2   # vehicle.bicycle
GENERAL_TO_PANOPTIC_16[15] = 3   # vehicle.bus.bendy
GENERAL_TO_PANOPTIC_16[16] = 3   # vehicle.bus.rigid
GENERAL_TO_PANOPTIC_16[17] = 4   # vehicle.car
GENERAL_TO_PANOPTIC_16[18] = 5   # vehicle.construction
GENERAL_TO_PANOPTIC_16[21] = 6   # vehicle.motorcycle
GENERAL_TO_PANOPTIC_16[2]  = 7   # human.pedestrian.adult
GENERAL_TO_PANOPTIC_16[3]  = 7   # human.pedestrian.child
GENERAL_TO_PANOPTIC_16[4]  = 7   # human.pedestrian.construction_worker
GENERAL_TO_PANOPTIC_16[6]  = 7   # human.pedestrian.police_officer
GENERAL_TO_PANOPTIC_16[12] = 8   # movable_object.trafficcone
GENERAL_TO_PANOPTIC_16[22] = 9   # vehicle.trailer
GENERAL_TO_PANOPTIC_16[23] = 10  # vehicle.truck
GENERAL_TO_PANOPTIC_16[24] = 11  # flat.driveable_surface
GENERAL_TO_PANOPTIC_16[25] = 12  # flat.other
GENERAL_TO_PANOPTIC_16[26] = 13  # flat.sidewalk
GENERAL_TO_PANOPTIC_16[27] = 14  # flat.terrain
GENERAL_TO_PANOPTIC_16[28] = 15  # static.manmade
GENERAL_TO_PANOPTIC_16[30] = 16  # static.vegetation

THING_IDS = list(range(1, 11))   # panoptic 16-class things (1-indexed)
STUFF_IDS = list(range(11, 17))  # panoptic 16-class stuff (1-indexed)
NUM_CLASSES = 16
IGNORE_LABEL = 255

NUSCENES_CLASS_NAMES = [
    "barrier", "bicycle", "bus", "car", "construction_vehicle",
    "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck",
    "driveable_surface", "other_flat", "sidewalk", "terrain", "manmade", "vegetation",
]

CAMERAS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]

DEFAULT_RANGE = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])


def augment_points(pts, aug_cfg):
    """Apply geometric augmentations to raw point cloud (N, 3+).

    Operates on xyz columns in-place. Labels are unaffected since
    augmentations are rigid/affine transforms.
    """
    xyz = pts[:, :3].copy()

    # Random rotation around Z axis (full 360°)
    if aug_cfg.get("random_rotate_z", True):
        theta = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        xyz = xyz @ rot.T

    # Random flip X
    if aug_cfg.get("random_flip_x", True) and np.random.rand() > 0.5:
        xyz[:, 0] = -xyz[:, 0]

    # Random flip Y
    if aug_cfg.get("random_flip_y", True) and np.random.rand() > 0.5:
        xyz[:, 1] = -xyz[:, 1]

    # Random scale on X, Y
    scale_min = aug_cfg.get("random_scale_min", 0.95)
    scale_max = aug_cfg.get("random_scale_max", 1.05)
    scale = np.random.uniform(scale_min, scale_max, size=2).astype(np.float32)
    xyz[:, 0] *= scale[0]
    xyz[:, 1] *= scale[1]

    # Random translation (Gaussian)
    trans_std = aug_cfg.get("random_translate_std", 0.1)
    if trans_std > 0:
        xyz += np.random.normal(0, trans_std, size=3).astype(np.float32)

    pts[:, :3] = xyz
    return pts


class NuScenesPanoptic(Dataset):
    """Per-keyframe nuScenes panoptic dataset.

    Returns per sample:
        vox_coords:    (V, 3) int32       voxel coordinates
        vox_feats:     (V, 3) float32     XYZ features (matches DisNet in_channels=3)
        gt_masks:      list of (V,) bool  one mask per thing instance + stuff class
        gt_classes:    (M,) int64         0-indexed class per mask
        gt_sem:        (V,) int64         per-voxel semantic label (0-indexed, 255=ignore)
        inv_idx:       (N,) int64         point→voxel mapping for un-voxelizing
        cam_paths:     list of 6 str      absolute camera image paths
        token:         str                sample token
    """

    def __init__(
        self,
        dataroot: str,
        version: str = "v1.0-mini",
        split: str = "train",
        voxel_size: float = 0.05,
        point_range: np.ndarray = None,
        augment: bool = False,
        aug_cfg: dict = None,
        nusc: "NuScenes" = None,
    ):
        self.dataroot = dataroot
        self.version = version
        self.voxel_size = voxel_size
        self.point_range = point_range if point_range is not None else DEFAULT_RANGE
        self.augment = augment
        self.aug_cfg = aug_cfg or {}

        self.nusc = nusc if nusc is not None else NuScenes(version=version, dataroot=dataroot, verbose=False)

        # Build sample list for split
        splits = create_splits_scenes()
        if "mini" in version:
            key = "mini_train" if split == "train" else "mini_val"
        else:
            key = split
        scene_names = set(splits[key])

        self.tokens = [
            s["token"]
            for s in self.nusc.sample
            if self.nusc.get("scene", s["scene_token"])["name"] in scene_names
        ]
        print(f"[NuScenesPanoptic] {split}: {len(self.tokens)} samples from {version}")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        token = self.tokens[idx]
        sample = self.nusc.get("sample", token)

        # ── 1. Load LiDAR (single sweep only) ──
        lidar_sd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        pc = LidarPointCloud.from_file(
            os.path.join(self.dataroot, lidar_sd["filename"])
        )
        pts = pc.points.T.astype(np.float32)  # (N, 5) x,y,z,intensity,ring

        # ── 1b. Load panoptic labels BEFORE augmentation (keep correspondence) ──
        lidar_token = lidar_sd["token"]
        pan_path = os.path.join(
            self.dataroot, "panoptic", self.version, f"{lidar_token}_panoptic.npz"
        )
        pan_data_full = np.load(pan_path)["data"]  # (N_full,)

        # Range filter (before augmentation — on original coords)
        r = self.point_range
        keep = (
            (pts[:, 0] >= r[0]) & (pts[:, 0] <= r[3]) &
            (pts[:, 1] >= r[1]) & (pts[:, 1] <= r[4]) &
            (pts[:, 2] >= r[2]) & (pts[:, 2] <= r[5])
        )
        pts = pts[keep]
        pan_data = pan_data_full[keep]

        # ── 1c. Augment points (training only) ──
        if self.augment and self.aug_cfg.get("enabled", True):
            pts = augment_points(pts, self.aug_cfg)

        N = pts.shape[0]

        # ── 2. Decode panoptic labels ──
        sem_general = (pan_data // 1000).astype(np.int32)
        inst = (pan_data % 1000).astype(np.int32)
        sem_general = np.clip(sem_general, 0, 31)
        sem = GENERAL_TO_PANOPTIC_16[sem_general]  # (N,) 0-16, 0=ignore

        # ── 3. Voxelize ──
        coords_f = pts[:, :3] / self.voxel_size
        coords_i = np.floor(coords_f).astype(np.int32)
        _, vox_idx, inv_idx = np.unique(
            coords_i, axis=0, return_index=True, return_inverse=True
        )
        V = len(vox_idx)
        vox_coords = coords_i[vox_idx]              # (V, 3)
        vox_feats = pts[vox_idx, :3].copy()          # (V, 3) XYZ as features
        vox_sem_raw = sem[vox_idx]                    # (V,) raw semantic per voxel

        # Per-voxel semantic: 0-indexed (subtract 1), noise(0) → IGNORE
        vox_sem = np.full(V, IGNORE_LABEL, dtype=np.int64)
        valid = vox_sem_raw > 0
        vox_sem[valid] = vox_sem_raw[valid] - 1      # 1-16 → 0-15

        # ── 4. Build GT masks (things + stuff) ──
        gt_masks = []
        gt_classes = []

        # Thing instances: one mask per unique (sem, inst) pair
        for sid in THING_IDS:
            pt_mask_class = sem == sid
            if not pt_mask_class.any():
                continue
            for iid in np.unique(inst[pt_mask_class]):
                if iid == 0:
                    continue  # skip noise instances
                pt_mask = (sem == sid) & (inst == iid)
                vox_mask = np.zeros(V, dtype=bool)
                vox_mask[inv_idx[pt_mask]] = True
                if vox_mask.sum() < 1:
                    continue
                gt_masks.append(vox_mask)
                gt_classes.append(sid - 1)  # 0-indexed

        # Stuff classes: one mask per class present
        for sid in STUFF_IDS:
            pt_mask = sem == sid
            if not pt_mask.any():
                continue
            vox_mask = np.zeros(V, dtype=bool)
            vox_mask[inv_idx[pt_mask]] = True
            if vox_mask.sum() < 1:
                continue
            gt_masks.append(vox_mask)
            gt_classes.append(sid - 1)  # 0-indexed

        # Stack masks
        if len(gt_masks) > 0:
            gt_masks_arr = np.stack(gt_masks, axis=0)   # (M, V)
            gt_classes_arr = np.array(gt_classes, dtype=np.int64)
        else:
            gt_masks_arr = np.zeros((0, V), dtype=bool)
            gt_classes_arr = np.array([], dtype=np.int64)

        # ── 5. Camera paths ──
        cam_paths = []
        for cam in CAMERAS:
            sd = self.nusc.get("sample_data", sample["data"][cam])
            cam_paths.append(os.path.join(self.dataroot, sd["filename"]))

        # ── 6. Convert to tensors ──
        return {
            "vox_coords": torch.from_numpy(vox_coords).int(),         # (V, 3)
            "vox_feats": torch.from_numpy(vox_feats).float(),         # (V, 3)
            "gt_masks": torch.from_numpy(gt_masks_arr).bool(),        # (M, V)
            "gt_classes": torch.from_numpy(gt_classes_arr).long(),     # (M,)
            "gt_sem": torch.from_numpy(vox_sem).long(),                # (V,)
            "inv_idx": torch.from_numpy(inv_idx.astype(np.int64)),     # (N,)
            "num_points": N,
            "num_voxels": V,
            "cam_paths": cam_paths,
            "token": token,
        }
