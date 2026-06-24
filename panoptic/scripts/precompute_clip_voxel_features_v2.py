"""Precompute per-instance crop-CLS CLIP features for Job 1b-v2 (L_O).

Two-pass workflow
-----------------
Pass 1  (--no_vox_feats)        : encode bbox crops with CLIP CLS token → obj_feats.
                                  vox_feats written as zeros. ~3 GB, ~12-15 h trainval.
Pass 2  (--vox_feats_only --resume): load existing .npz, add patch-pooled vox_feats.
                                  ~180 GB additional. Only needed for ablation B.
Single-pass (no flags)          : both passes in one run (slower, rarely needed).

Output schema per sample ({sample_token}.npz)
---------------------------------------------
  obj_feats:          (N_gt, 768) float16  crop-CLS CLIP per GT instance (zero = invalid)
  obj_classes:        (N_gt,)     int32    0-indexed GT class
  obj_valid:          (N_gt,)     bool     True if ≥1 valid crop contributed
  obj_num_views:      (N_gt,)     int16    cameras with a valid crop
  obj_num_points:     (N_gt,)     int32    total LiDAR pts projected across cameras
  obj_margin:         (N_gt,)     float16  sim_own - sim_best_base (0 if no --proto_path)
  vox_feats:          (4096, 768) float16  patch-pooled CLIP (zeros if --no_vox_feats)
  vox_indices:        (4096,)     int32    voxel indices into un-augmented array
  vox_gt_instance_id: (4096,)     int64    GT instance index per sampled voxel (-1=none)
  n_valid:            int         voxels with ≥1 camera projection (patch-pooled count)
  n_voxels:           int32       total voxels in this sample
"""
import argparse
import hashlib
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene")

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.splits import create_splits_scenes
from panoptic.scripts.verify_camera_projection import project_lidar_to_camera

# ── Constants (must match nuscenes_panoptic.py exactly) ───────────────────────
DEFAULT_RANGE = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
CAMERAS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK",  "CAM_BACK_LEFT",  "CAM_BACK_RIGHT",
]
CAM_W, CAM_H = 1600, 900

GENERAL_TO_PANOPTIC_16 = np.zeros(32, dtype=np.int32)
GENERAL_TO_PANOPTIC_16[9]  = 1   # barrier
GENERAL_TO_PANOPTIC_16[14] = 2   # bicycle
GENERAL_TO_PANOPTIC_16[15] = 3   # bus.bendy
GENERAL_TO_PANOPTIC_16[16] = 3   # bus.rigid
GENERAL_TO_PANOPTIC_16[17] = 4   # car
GENERAL_TO_PANOPTIC_16[18] = 5   # construction vehicle
GENERAL_TO_PANOPTIC_16[21] = 6   # motorcycle
GENERAL_TO_PANOPTIC_16[2]  = 7   # pedestrian.adult
GENERAL_TO_PANOPTIC_16[3]  = 7   # pedestrian.child
GENERAL_TO_PANOPTIC_16[4]  = 7   # pedestrian.construction_worker
GENERAL_TO_PANOPTIC_16[6]  = 7   # pedestrian.police_officer
GENERAL_TO_PANOPTIC_16[12] = 8   # traffic cone
GENERAL_TO_PANOPTIC_16[22] = 9   # trailer
GENERAL_TO_PANOPTIC_16[23] = 10  # truck
GENERAL_TO_PANOPTIC_16[24] = 11  # driveable_surface
GENERAL_TO_PANOPTIC_16[25] = 12  # other_flat
GENERAL_TO_PANOPTIC_16[26] = 13  # sidewalk
GENERAL_TO_PANOPTIC_16[27] = 14  # terrain
GENERAL_TO_PANOPTIC_16[28] = 15  # manmade
GENERAL_TO_PANOPTIC_16[30] = 16  # vegetation

THING_IDS  = list(range(1, 11))   # 1-indexed panoptic ids
STUFF_IDS  = list(range(11, 17))

N_VOX_SAMPLE   = 4096
MIN_PTS_THING  = 5
MIN_PTS_STUFF  = 20
N_STUFF_CROPS  = 6
STUFF_HALF_PX  = 112   # half-side for 224×224 stuff crops

CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


# ── CLIP encoders ─────────────────────────────────────────────────────────────

class CLIPCropEncoder:
    """Encode 2D bbox crops with CLIP encode_image() (CLS token)."""

    def __init__(self, model, device):
        self.model  = model
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ])


class CLIPPatchExtractor:
    """Extract dense CLIP patch features using multi-tile overlap (same as v1)."""

    def __init__(self, model, device, tile_size=224, stride=112):
        self.model     = model
        self.device    = device
        self.tile_size = tile_size
        self.stride    = stride

        self._patch_tokens = None
        model.visual.transformer.resblocks[-1].register_forward_hook(self._hook)

        self.mean = torch.tensor(CLIP_MEAN, device=device).half().view(1, 3, 1, 1)
        self.std  = torch.tensor(CLIP_STD,  device=device).half().view(1, 3, 1, 1)

    def _hook(self, module, input, output):
        self._patch_tokens = output

    def _preprocess_tile(self, tile_img):
        t = torch.from_numpy(np.array(tile_img)).float().permute(2, 0, 1) / 255.0
        return ((t.unsqueeze(0).to(self.device) - self.mean) / self.std).half()

    def extract_dense_features(self, image_path, batch_size=32):
        """Returns (H_p, W_p, 768) float32 patch feature map."""
        img = Image.open(image_path).convert("RGB")
        img_w, img_h = img.size

        cols = list(range(0, img_w - self.tile_size + 1, self.stride))
        if cols[-1] + self.tile_size < img_w:
            cols.append(img_w - self.tile_size)
        rows = list(range(0, img_h - self.tile_size + 1, self.stride))
        if rows[-1] + self.tile_size < img_h:
            rows.append(img_h - self.tile_size)

        patch_px = 14
        H_p = (img_h + patch_px - 1) // patch_px
        W_p = (img_w + patch_px - 1) // patch_px
        feat_accum = np.zeros((H_p, W_p, 768), dtype=np.float32)
        feat_count = np.zeros((H_p, W_p),       dtype=np.int32)

        tiles, positions = [], []
        for r in rows:
            for c in cols:
                tiles.append(img.crop((c, r, c + self.tile_size, r + self.tile_size)))
                positions.append((r, c))

        for bs in range(0, len(tiles), batch_size):
            batch_t = torch.cat(
                [self._preprocess_tile(t) for t in tiles[bs:bs + batch_size]], dim=0
            )
            with torch.no_grad():
                _ = self.model.visual(batch_t)
                raw     = self._patch_tokens            # (257, B, 1024)
                patches = raw[1:].permute(1, 0, 2)     # (B, 256, 1024)
                patches = self.model.visual.ln_post(patches)
                patches = patches @ self.model.visual.proj   # (B, 256, 768)
                patches = F.normalize(patches, dim=-1).cpu().numpy()

            for i, (r, c) in enumerate(positions[bs:bs + batch_size]):
                pg = patches[i].reshape(16, 16, 768)
                for pi in range(16):
                    for pj in range(16):
                        gy = min((r + pi * patch_px) // patch_px, H_p - 1)
                        gx = min((c + pj * patch_px) // patch_px, W_p - 1)
                        feat_accum[gy, gx] += pg[pi, pj]
                        feat_count[gy, gx] += 1

        valid = feat_count > 0
        feat_accum[valid] /= feat_count[valid, np.newaxis]
        return feat_accum, H_p, W_p


# ── Crop helpers ──────────────────────────────────────────────────────────────

def _make_sq_crop(uv_v, img_w, img_h, expand=1.3):
    """Return clamped square crop (x0,y0,x1,y1) from projected 2D points."""
    u_min, u_max = float(uv_v[:, 0].min()), float(uv_v[:, 0].max())
    v_min, v_max = float(uv_v[:, 1].min()), float(uv_v[:, 1].max())
    cx = (u_min + u_max) / 2.0
    cy = (v_min + v_max) / 2.0
    half = max((u_max - u_min) / 2.0, (v_max - v_min) / 2.0) * expand
    half = max(half, 10.0)
    x0 = int(max(0.0, cx - half))
    y0 = int(max(0.0, cy - half))
    x1 = int(min(float(img_w), cx + half))
    y1 = int(min(float(img_h), cy + half))
    return x0, y0, x1, y1


def batch_encode_crops(crop_encoder, crop_items, max_batch=128):
    """Batch-encode crop items with CLIPCropEncoder.

    crop_items: list of {item_id: int, crop: PIL Image or None}
    Returns:    {item_id: (768,) float32 numpy}
    """
    valid = [(item["item_id"], item["crop"])
             for item in crop_items if item["crop"] is not None]
    feat_by_item = {}
    for start in range(0, len(valid), max_batch):
        chunk    = valid[start:start + max_batch]
        item_ids = [iid for iid, _ in chunk]
        batch_t  = torch.stack(
            [crop_encoder.transform(crop) for _, crop in chunk], dim=0
        ).to(crop_encoder.device).half()
        with torch.no_grad():
            feats = crop_encoder.model.encode_image(batch_t)   # (B, 768)
            feats = F.normalize(feats.float(), dim=-1).cpu().numpy()
        for iid, feat in zip(item_ids, feats):
            feat_by_item[iid] = feat
    return feat_by_item


# ── nuScenes helpers ──────────────────────────────────────────────────────────

def voxelize_sample(nusc, sample_token, dataroot, voxel_size=0.05):
    """Voxelize one sample without augmentation (matches NuScenesPanoptic).

    Returns: vox_xyz (V,3), vox_idx (V,), inv_idx (N,), sem (N,), inst (N,)
    """
    sample  = nusc.get("sample", sample_token)
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    pc       = LidarPointCloud.from_file(os.path.join(dataroot, lidar_sd["filename"]))
    pts      = pc.points.T.astype(np.float32)

    lidar_token = lidar_sd["token"]
    pan_path = os.path.join(
        dataroot, "panoptic", nusc.version, f"{lidar_token}_panoptic.npz"
    )
    pan_data_full = np.load(pan_path)["data"]

    r    = DEFAULT_RANGE
    keep = (
        (pts[:, 0] >= r[0]) & (pts[:, 0] <= r[3]) &
        (pts[:, 1] >= r[1]) & (pts[:, 1] <= r[4]) &
        (pts[:, 2] >= r[2]) & (pts[:, 2] <= r[5])
    )
    pts      = pts[keep]
    pan_data = pan_data_full[keep]

    sem  = GENERAL_TO_PANOPTIC_16[(pan_data // 1000).astype(np.int32).clip(0, 31)]
    inst = (pan_data % 1000).astype(np.int32)

    coords_i = np.floor(pts[:, :3] / voxel_size).astype(np.int32)
    _, vox_idx, inv_idx = np.unique(
        coords_i, axis=0, return_index=True, return_inverse=True
    )
    vox_xyz = pts[vox_idx, :3]
    return vox_xyz, vox_idx, inv_idx, sem, inst


def build_gt_instances(sem, inst, inv_idx, V):
    """Return list of (vox_mask: bool[V], cls_id: int 0-indexed)."""
    instances = []
    for sid in THING_IDS:
        cls_pts = sem == sid
        if not cls_pts.any():
            continue
        for iid in np.unique(inst[cls_pts]):
            if iid == 0:
                continue
            pt_mask  = (sem == sid) & (inst == iid)
            vox_mask = np.zeros(V, dtype=bool)
            vox_mask[inv_idx[pt_mask]] = True
            if vox_mask.sum() >= 1:
                instances.append((vox_mask, sid - 1))
    for sid in STUFF_IDS:
        pt_mask = sem == sid
        if not pt_mask.any():
            continue
        vox_mask = np.zeros(V, dtype=bool)
        vox_mask[inv_idx[pt_mask]] = True
        if vox_mask.sum() >= 1:
            instances.append((vox_mask, sid - 1))
    return instances


# ── Core processing ───────────────────────────────────────────────────────────

def process_sample_obj(nusc, sample_token, dataroot, crop_encoder, proto_embs=None):
    """Compute crop-CLS obj_feats + vox metadata for all GT instances.

    Returns the full output dict (vox_feats is zeros; fill via add_vox_feats or inline).
    """
    vox_xyz, vox_idx, inv_idx, sem, inst = voxelize_sample(
        nusc, sample_token, dataroot
    )
    V = len(vox_idx)

    sample       = nusc.get("sample", sample_token)
    gt_instances = build_gt_instances(sem, inst, inv_idx, V)
    n_inst       = len(gt_instances)

    obj_classes   = np.array([cls_id for _, cls_id in gt_instances], dtype=np.int32)
    obj_feats_acc = np.zeros((n_inst, 768), dtype=np.float64)
    obj_weight    = np.zeros(n_inst,        dtype=np.float64)
    obj_num_views = np.zeros(n_inst,        dtype=np.int32)
    obj_num_pts   = np.zeros(n_inst,        dtype=np.int32)

    # Per-voxel GT instance assignment
    vox_inst_id = np.full(V, -1, dtype=np.int64)
    for k, (mask, _) in enumerate(gt_instances):
        vox_inst_id[mask] = k

    # Build crop items across all cameras
    crop_items = []   # list of {item_id, crop}
    item_meta  = []   # list of {item_id, inst_k, n_pts}
    item_ctr   = 0

    for cam_idx, cam_name in enumerate(CAMERAS):
        img_sd   = nusc.get("sample_data", sample["data"][cam_name])
        img_path = os.path.join(dataroot, img_sd["filename"])
        img      = Image.open(img_path).convert("RGB")

        uv_all, _, valid_all = project_lidar_to_camera(
            nusc, sample_token, vox_xyz, cam_name
        )

        for k, (vox_mask, cls_id) in enumerate(gt_instances):
            valid_k = valid_all & vox_mask
            n_pts   = int(valid_k.sum())
            is_stuff = (cls_id + 1) in STUFF_IDS
            min_pts  = MIN_PTS_STUFF if is_stuff else MIN_PTS_THING

            if n_pts < min_pts:
                continue

            uv_v = uv_all[valid_k]   # (n_pts, 2)

            if is_stuff:
                # N_STUFF_CROPS random crops around sampled LiDAR centers
                seed_bytes = hashlib.md5(
                    f"{sample_token}_{k}_{cam_idx}".encode()
                ).digest()
                rng = np.random.default_rng(
                    np.frombuffer(seed_bytes[:8], dtype=np.uint8).astype(np.uint64)
                )
                n_centers = min(N_STUFF_CROPS, len(uv_v))
                ci_arr    = rng.choice(len(uv_v), size=n_centers, replace=False)
                for ci in ci_arr:
                    cx = float(uv_v[ci, 0])
                    cy = float(uv_v[ci, 1])
                    x0 = int(max(0,      cx - STUFF_HALF_PX))
                    y0 = int(max(0,      cy - STUFF_HALF_PX))
                    x1 = int(min(CAM_W,  cx + STUFF_HALF_PX))
                    y1 = int(min(CAM_H,  cy + STUFF_HALF_PX))
                    if (x1 - x0) < 5 or (y1 - y0) < 5:
                        continue
                    crop_items.append({"item_id": item_ctr, "crop": img.crop((x0, y0, x1, y1))})
                    item_meta.append( {"item_id": item_ctr, "inst_k": k, "n_pts": n_pts})
                    item_ctr += 1
            else:
                # Things: 1 expanded square crop from 2D bbox
                x0, y0, x1, y1 = _make_sq_crop(uv_v, CAM_W, CAM_H, expand=1.3)
                if (x1 - x0) * (y1 - y0) < 100:
                    continue
                crop_items.append({"item_id": item_ctr, "crop": img.crop((x0, y0, x1, y1))})
                item_meta.append( {"item_id": item_ctr, "inst_k": k, "n_pts": n_pts})
                item_ctr += 1

    # Batch encode all crops across all cameras
    feat_by_item = batch_encode_crops(crop_encoder, crop_items, max_batch=128)

    # Accumulate weighted by n_pts
    for meta in item_meta:
        iid = meta["item_id"]
        if iid not in feat_by_item:
            continue
        k     = meta["inst_k"]
        n_pts = meta["n_pts"]
        obj_feats_acc[k] += feat_by_item[iid].astype(np.float64) * n_pts
        obj_weight[k]    += n_pts
        obj_num_views[k] += 1
        obj_num_pts[k]   += n_pts

    # Finalize per-instance
    obj_feats = np.zeros((n_inst, 768), dtype=np.float16)
    obj_valid = np.zeros(n_inst,        dtype=bool)
    for k in range(n_inst):
        if obj_weight[k] > 0:
            feat = obj_feats_acc[k] / obj_weight[k]
            obj_feats[k] = (feat / max(np.linalg.norm(feat), 1e-8)).astype(np.float16)
            obj_valid[k] = True

    # Optional per-instance margin vs best base
    obj_margin = np.zeros(n_inst, dtype=np.float16)
    if proto_embs is not None:
        valid_ks  = np.where(obj_valid)[0]
        if len(valid_ks) > 0:
            feats_f = obj_feats[valid_ks].astype(np.float32)
            sims    = feats_f @ proto_embs.T   # (n_valid, 16)
            for i, k in enumerate(valid_ks):
                cls_id  = int(obj_classes[k])
                sim_own = float(sims[i, cls_id])
                tmp     = sims[i].copy()
                tmp[cls_id] = -1.0
                sim_base    = float(tmp.max())
                obj_margin[k] = np.float16(sim_own - sim_base)

    # Voxel sampling (uniform over all voxels)
    all_vox = np.arange(V, dtype=np.int64)
    if V >= N_VOX_SAMPLE:
        sampled = np.random.choice(all_vox, N_VOX_SAMPLE, replace=False)
    elif V > 0:
        sampled = np.random.choice(all_vox, N_VOX_SAMPLE, replace=True)
    else:
        sampled = np.zeros(N_VOX_SAMPLE, dtype=np.int64)

    return {
        "obj_feats":          obj_feats,
        "obj_classes":        obj_classes,
        "obj_valid":          obj_valid,
        "obj_num_views":      obj_num_views.astype(np.int16),
        "obj_num_points":     obj_num_pts,
        "obj_margin":         obj_margin,
        "vox_feats":          np.zeros((N_VOX_SAMPLE, 768), dtype=np.float16),
        "vox_indices":        sampled.astype(np.int32),
        "vox_gt_instance_id": vox_inst_id[sampled],
        "n_valid":            0,   # filled by add_vox_feats or inline
        "n_voxels":           np.int32(V),
    }


def compute_vox_feats(nusc, sample_token, dataroot, vox_xyz, vox_indices, patch_extractor):
    """Compute patch-pooled vox_feats for sampled voxels. Returns (4096,768) float16, n_valid."""
    V = len(vox_xyz)
    sample = nusc.get("sample", sample_token)

    voxel_acc   = np.zeros((V, 768), dtype=np.float32)
    voxel_count = np.zeros(V,        dtype=np.int32)
    patch_px    = 14

    for cam_name in CAMERAS:
        cam_sd   = nusc.get("sample_data", sample["data"][cam_name])
        cam_path = os.path.join(dataroot, cam_sd["filename"])
        feat_map, H_p, W_p = patch_extractor.extract_dense_features(cam_path)
        uv, _, valid = project_lidar_to_camera(nusc, sample_token, vox_xyz, cam_name)
        for vi in np.where(valid)[0]:
            u_px, v_px = int(uv[vi, 0]), int(uv[vi, 1])
            gy = min(v_px // patch_px, H_p - 1)
            gx = min(u_px // patch_px, W_p - 1)
            voxel_acc[vi]   += feat_map[gy, gx]
            voxel_count[vi] += 1

    valid_v = voxel_count > 0
    voxel_acc[valid_v] /= voxel_count[valid_v, np.newaxis]
    norms = np.linalg.norm(voxel_acc[valid_v], axis=1, keepdims=True)
    voxel_acc[valid_v] /= np.maximum(norms, 1e-8)

    return voxel_acc[vox_indices].astype(np.float16), int(valid_v.sum())


def add_vox_feats(nusc, sample_token, dataroot, patch_extractor, npz_path):
    """Pass 2: compute patch-pooled vox_feats and merge into existing .npz."""
    data        = dict(np.load(npz_path, allow_pickle=False))
    vox_indices = data["vox_indices"]

    vox_xyz, _, _, _, _ = voxelize_sample(nusc, sample_token, dataroot)
    vox_feats, n_valid  = compute_vox_feats(
        nusc, sample_token, dataroot, vox_xyz, vox_indices, patch_extractor
    )
    data["vox_feats"] = vox_feats
    data["n_valid"]   = n_valid
    np.savez_compressed(npz_path, **data)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot",   default="/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes")
    parser.add_argument("--version",    default="v1.0-trainval")
    parser.add_argument("--output_dir", default="/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features_v2")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--split",      default="both", help="train | val | both")
    parser.add_argument("--resume",     action="store_true",
                        help="Skip already-processed samples")
    parser.add_argument("--no_vox_feats",   action="store_true",
                        help="Pass 1: write zero vox_feats (fast, for ablation A)")
    parser.add_argument("--vox_feats_only", action="store_true",
                        help="Pass 2: add patch-pooled vox_feats to existing .npz files")
    parser.add_argument("--proto_path", default="",
                        help="Optional path to text_prototypes v2.pt for obj_margin")
    args = parser.parse_args()

    import clip
    print(f"Loading CLIP ViT-L/14 on {args.device}...")
    model, _ = clip.load("ViT-L/14", device=args.device, jit=False)
    model.eval()

    crop_encoder    = None
    patch_extractor = None

    if not args.vox_feats_only:
        crop_encoder = CLIPCropEncoder(model, args.device)
        print("  CLIPCropEncoder ready (encode_image / CLS token)")

    # CLIPPatchExtractor registers a forward hook — only instantiate if needed.
    need_patch = args.vox_feats_only or not args.no_vox_feats
    if need_patch:
        patch_extractor = CLIPPatchExtractor(model, args.device)
        print("  CLIPPatchExtractor ready (patch tokens / multi-tile)")

    # Optional proto embeddings for obj_margin computation
    proto_embs = None
    if args.proto_path and os.path.isfile(args.proto_path):
        ckpt       = torch.load(args.proto_path, map_location="cpu")
        proto_embs = ckpt["embeddings"].float().numpy()   # (16, 768)
        print(f"  Proto embeddings loaded: {proto_embs.shape}")

    out_dir = os.path.join(args.output_dir, args.version)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading nuScenes {args.version} from {args.dataroot}...")
    nusc   = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    splits = create_splits_scenes()

    if "mini" in args.version:
        if args.split == "both":
            scene_names = set(splits["mini_train"]) | set(splits["mini_val"])
        else:
            scene_names = set(splits[f"mini_{args.split}"])
    else:
        if args.split == "both":
            scene_names = set(splits["train"]) | set(splits["val"])
        else:
            scene_names = set(splits[args.split])

    tokens = [
        s["token"] for s in nusc.sample
        if nusc.get("scene", s["scene_token"])["name"] in scene_names
    ]
    print(f"Processing {len(tokens)} samples ({args.split})")

    skipped = 0
    for i, token in enumerate(tqdm(tokens, desc="Precomputing")):
        out_path = os.path.join(out_dir, f"{token}.npz")

        # ── Pass 2: merge vox_feats into existing file ──
        if args.vox_feats_only:
            if not os.path.exists(out_path):
                skipped += 1
                continue
            if args.resume:
                try:
                    d = np.load(out_path, allow_pickle=False)
                    if int(d["n_valid"]) > 0:
                        skipped += 1
                        continue
                except Exception:
                    pass
            try:
                add_vox_feats(nusc, token, args.dataroot, patch_extractor, out_path)
            except Exception as e:
                print(f"\nERROR (vox_feats_only) {token[:16]}: {e}")
            continue

        # ── Pass 1 / single-pass: obj_feats ──
        if args.resume and os.path.exists(out_path):
            skipped += 1
            continue

        try:
            result = process_sample_obj(
                nusc, token, args.dataroot, crop_encoder, proto_embs
            )

            # Inline vox_feats (single-pass mode; skipped if --no_vox_feats)
            if not args.no_vox_feats:
                vox_xyz, _, _, _, _ = voxelize_sample(nusc, token, args.dataroot)
                vox_feats, n_valid  = compute_vox_feats(
                    nusc, token, args.dataroot,
                    vox_xyz, result["vox_indices"], patch_extractor
                )
                result["vox_feats"] = vox_feats
                result["n_valid"]   = n_valid

            np.savez_compressed(out_path, **result)

        except Exception as e:
            print(f"\nERROR {token[:16]}: {e}")
            continue

        if (i + 1) % 100 == 0:
            n_v = int(result["obj_valid"].sum())
            print(
                f"\n  [{i+1}/{len(tokens)}] obj_valid={n_v}/{result['obj_feats'].shape[0]}, "
                f"n_voxels={result['n_voxels']}, "
                f"vox_gt_assigned={(result['vox_gt_instance_id'] >= 0).sum()}/{N_VOX_SAMPLE}"
            )

    print(f"\nDone. Output: {out_dir}/")
    if skipped:
        print(f"  Skipped {skipped} samples (resume or missing)")


if __name__ == "__main__":
    main()
