"""Diagnostic: verify that online CASD crops give positive margins for STUFF classes.

For each sampled stuff voxel in the mini .npz cache:
  - Re-project the voxel's 3D position to all 6 cameras
  - Take a crop centered at the BEST projection (scored by area + border margin - depth)
  - Encode with CLIP encode_image() → CLS token (same mechanism as CASD at training time)
  - Compute margin = sim_own - sim_best_other vs v2 text prototypes

Comparison printed alongside the broadcast baseline (from diag_broadcast_vox_check):
  broadcast margin (obj_feats target) vs CASD margin (per-voxel crop-CLS target)

Key question: does CASD produce positive margins for vegetation / terrain / sidewalk?
If yes → CASD is a valid L_V teacher for STUFF → proceed with job1c plan.
If no  → CASD crops are too noisy at this voxel density → investigate crop size.

Requires GPU (~5 min on A100 for 50 samples, 30 STUFF voxels/class).

Usage:
    python panoptic/scripts/diag_casd_margin_check.py --device cuda
    python panoptic/scripts/diag_casd_margin_check.py --device cuda --n_samples 20 --max_vox_per_class 20
    python panoptic/scripts/diag_casd_margin_check.py --device cuda --crop_half_px 168
"""
import argparse
import hashlib
import os
import sys
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene")

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from panoptic.scripts.verify_camera_projection import project_lidar_to_camera

# ── Constants (must match precompute_clip_voxel_features_v2.py) ───────────────
DEFAULT_RANGE = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
CAMERAS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK",  "CAM_BACK_LEFT",  "CAM_BACK_RIGHT",
]
CAM_W, CAM_H = 1600, 900
GENERAL_TO_PANOPTIC_16 = np.zeros(32, dtype=np.int32)
GENERAL_TO_PANOPTIC_16[9]  = 1;  GENERAL_TO_PANOPTIC_16[14] = 2
GENERAL_TO_PANOPTIC_16[15] = 3;  GENERAL_TO_PANOPTIC_16[16] = 3
GENERAL_TO_PANOPTIC_16[17] = 4;  GENERAL_TO_PANOPTIC_16[18] = 5
GENERAL_TO_PANOPTIC_16[21] = 6;  GENERAL_TO_PANOPTIC_16[2]  = 7
GENERAL_TO_PANOPTIC_16[3]  = 7;  GENERAL_TO_PANOPTIC_16[4]  = 7
GENERAL_TO_PANOPTIC_16[6]  = 7;  GENERAL_TO_PANOPTIC_16[12] = 8
GENERAL_TO_PANOPTIC_16[22] = 9;  GENERAL_TO_PANOPTIC_16[23] = 10
GENERAL_TO_PANOPTIC_16[24] = 11; GENERAL_TO_PANOPTIC_16[25] = 12
GENERAL_TO_PANOPTIC_16[26] = 13; GENERAL_TO_PANOPTIC_16[27] = 14
GENERAL_TO_PANOPTIC_16[28] = 15; GENERAL_TO_PANOPTIC_16[30] = 16

STUFF_IDS_0IDX = set(range(10, 16))   # 0-indexed

CLASS_NAMES = [
    "barrier", "bicycle", "bus", "car", "construction_vehicle",
    "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck",
    "driveable_surface", "other_flat", "sidewalk", "terrain", "manmade", "vegetation",
]
NOVEL_IDS = {2, 5, 6, 15}

# Broadcast margins from diag_broadcast_vox_check (2026-06-14, mini, 404 samples)
BROADCAST_MARGINS = {
    10: +0.0025, 11: -0.0109, 12: -0.0123,
    13: -0.0197, 14: +0.0060, 15: -0.0282,
}

CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


# ── Helpers ───────────────────────────────────────────────────────────────────

def voxelize_sample(nusc, sample_token, dataroot, voxel_size=0.05):
    sample   = nusc.get("sample", sample_token)
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    pc       = LidarPointCloud.from_file(os.path.join(dataroot, lidar_sd["filename"]))
    pts      = pc.points.T.astype(np.float32)
    lidar_token = lidar_sd["token"]
    pan_path = os.path.join(dataroot, "panoptic", nusc.version,
                            f"{lidar_token}_panoptic.npz")
    pan_data_full = np.load(pan_path)["data"]
    r    = DEFAULT_RANGE
    keep = ((pts[:, 0] >= r[0]) & (pts[:, 0] <= r[3]) &
            (pts[:, 1] >= r[1]) & (pts[:, 1] <= r[4]) &
            (pts[:, 2] >= r[2]) & (pts[:, 2] <= r[5]))
    pts      = pts[keep]
    pan_data = pan_data_full[keep]
    sem = GENERAL_TO_PANOPTIC_16[(pan_data // 1000).astype(np.int32).clip(0, 31)]
    coords_i = np.floor(pts[:, :3] / voxel_size).astype(np.int32)
    _, vox_idx, inv_idx = np.unique(coords_i, axis=0,
                                    return_index=True, return_inverse=True)
    vox_xyz = pts[vox_idx, :3]
    vox_sem = np.zeros(len(vox_idx), dtype=np.int32)
    for vi in range(len(vox_idx)):
        pt_mask = inv_idx == vi
        modes   = sem[pt_mask]
        if len(modes):
            vox_sem[vi] = int(np.bincount(modes.astype(np.int64)).argmax())
    return vox_xyz, vox_idx, inv_idx, vox_sem


def pad_square(crop):
    """Pad a potentially-rectangular crop to square with grey fill before resize.

    Near image boundaries the crop window gets clipped, producing a rectangle.
    Resizing directly to 224×224 squishes/stretches the content. Square padding
    preserves the aspect ratio exactly as the CASD training logic will do.
    """
    w, h = crop.size
    if w == h:
        return crop
    side = max(w, h)
    sq = Image.new("RGB", (side, side), (128, 128, 128))
    sq.paste(crop, ((side - w) // 2, (side - h) // 2))
    return sq


def best_camera_crop(nusc, sample_token, pt_xyz, half_px=112):
    """Return (PIL crop, cam_name) from the camera that gives the best view.

    Scored by: crop_area + 10 * border_margin - 0.01 * depth
      - crop_area:     larger = more of the requested window is inside the image
      - border_margin: distance of projected point from image edge; high = point
                       is well-centred, low = point near edge → clipped crop
      - depth penalty: prefer closer projections (less noise/occlusion)
    Returns (None, None) if no camera produces a usable crop (≥16×16 px).
    """
    sample     = nusc.get("sample", sample_token)
    best_crop  = None
    best_cam   = None
    best_score = -1e9

    for cam_name in CAMERAS:
        uv, depths, valid = project_lidar_to_camera(
            nusc, sample_token, pt_xyz[np.newaxis], cam_name
        )
        if not valid[0]:
            continue

        u     = float(uv[0, 0])
        v     = float(uv[0, 1])
        depth = float(depths[0])

        x0 = max(0, int(u - half_px))
        y0 = max(0, int(v - half_px))
        x1 = min(CAM_W, int(u + half_px))
        y1 = min(CAM_H, int(v + half_px))
        crop_w = x1 - x0
        crop_h = y1 - y0
        if crop_w < 16 or crop_h < 16:
            continue

        border_margin = min(u, CAM_W - u, v, CAM_H - v)
        score = float(crop_w * crop_h) + 10.0 * border_margin - 0.01 * depth

        if score > best_score:
            img_sd   = nusc.get("sample_data", sample["data"][cam_name])
            img_path = os.path.join(nusc.dataroot, img_sd["filename"])
            img      = Image.open(img_path).convert("RGB")
            best_crop  = img.crop((x0, y0, x1, y1))
            best_cam   = cam_name
            best_score = score

    return best_crop, best_cam


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    print(f"Loading CLIP ViT-L/14 on {args.device}...")
    import clip
    model, _ = clip.load("ViT-L/14", device=args.device, jit=False)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])

    # Fix 5: handle both plain tensor and dict checkpoint formats
    proto_data = torch.load(args.proto_path, map_location="cpu")
    proto_tensor = proto_data["embeddings"] if isinstance(proto_data, dict) else proto_data
    protos = F.normalize(proto_tensor.float(), dim=-1).numpy()   # (16, 768)

    npz_paths = sorted(glob.glob(f"{args.feat_dir}/*.npz"))
    if args.n_samples and args.n_samples < len(npz_paths):
        rng = np.random.default_rng(42)
        npz_paths = [npz_paths[i] for i in rng.choice(
            len(npz_paths), args.n_samples, replace=False)]

    print(f"Loading nuScenes {args.version} from {args.dataroot}...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    print(f"Processing {len(npz_paths)} samples  "
          f"(crop_half_px={args.crop_half_px} → {2*args.crop_half_px}×{2*args.crop_half_px} "
          f"before square-pad + resize to 224)\n")

    casd_margins = {cls_id: [] for cls_id in STUFF_IDS_0IDX}
    n_vox_seen   = {cls_id: 0  for cls_id in STUFF_IDS_0IDX}
    n_vox_encoded= {cls_id: 0  for cls_id in STUFF_IDS_0IDX}

    for npz_path in tqdm(npz_paths, desc="Samples"):
        token = os.path.basename(npz_path).replace(".npz", "")
        try:
            d = np.load(npz_path, allow_pickle=False)
        except Exception as e:
            print(f"  SKIP {token[:16]}: {e}")
            continue

        vox_inst    = d["vox_gt_instance_id"]             # (4096,)
        vox_indices = d["vox_indices"].astype(np.int64)   # (4096,)
        obj_classes = d["obj_classes"].astype(np.int32)   # (N_inst,)

        # Sanity check: catch degenerate .npz where vox_indices are all zeros
        n_assigned   = int((vox_inst >= 0).sum())
        n_unique_idx = len(np.unique(vox_indices))
        if n_unique_idx < 10:
            print(f"  WARN {token[:16]}: only {n_unique_idx} unique vox_indices "
                  f"— npz may be from --no_vox_feats run with dummy indices. Skipping.")
            continue

        # Re-voxelize to recover 3D positions
        try:
            vox_xyz, vox_idx_arr, _, _ = voxelize_sample(nusc, token, args.dataroot)
        except Exception as e:
            print(f"  SKIP voxelize {token[:16]}: {e}")
            continue

        V = len(vox_idx_arr)

        # Collect stuff voxels: {cls_id: [(sample_vi, pt_xyz), ...]}
        to_encode = {cls_id: [] for cls_id in STUFF_IDS_0IDX}
        for vi in range(len(vox_inst)):
            gi = int(vox_inst[vi])
            if gi < 0 or gi >= len(obj_classes):
                continue
            cls_id = int(obj_classes[gi])
            if cls_id not in STUFF_IDS_0IDX:
                continue
            n_vox_seen[cls_id] += 1
            raw_idx = int(vox_indices[vi])
            if raw_idx >= V:
                continue
            to_encode[cls_id].append((vi, vox_xyz[raw_idx]))

        # Subsample per class with deterministic seed derived from token
        # Fix 3: use hashlib so seed is process-independent
        seed = int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "little")
        rng  = np.random.default_rng(seed)

        crops_batch = []   # list of (sample_vi, cls_id, PIL crop)
        for cls_id, items in to_encode.items():
            if not items:
                continue
            n_take = min(args.max_vox_per_class, len(items))
            chosen = (items[:n_take] if n_take >= len(items)
                      else [items[i] for i in rng.choice(len(items), n_take, replace=False)])
            for vi, pt in chosen:
                # Fix 1: pass args.crop_half_px directly (no monkey-patch)
                crop, _ = best_camera_crop(nusc, token, pt,
                                           half_px=args.crop_half_px)
                if crop is not None:
                    crops_batch.append((vi, cls_id, crop))

        if not crops_batch:
            continue

        # Fix 4: square-pad before resize to avoid aspect-ratio distortion
        # Fix 1 (continued): crop_half_px already applied above via best_camera_crop call
        try:
            imgs = torch.stack(
                [preprocess(pad_square(item[2])) for item in crops_batch], dim=0
            ).to(args.device).half()
            with torch.no_grad():
                feats = model.encode_image(imgs)          # (N, 768)
                feats = F.normalize(feats.float(), dim=-1).cpu().numpy()
        except Exception as e:
            print(f"  encode error {token[:16]}: {e}")
            continue

        sims = feats @ protos.T   # (N, 16)
        for i, (vi, cls_id, _) in enumerate(crops_batch):
            sim_own  = float(sims[i, cls_id])
            tmp      = sims[i].copy(); tmp[cls_id] = -1.0
            sim_best = float(tmp.max())
            casd_margins[cls_id].append(sim_own - sim_best)
            n_vox_encoded[cls_id] += 1

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\n{'Class':<22} {'Novel':<6} "
          f"{'Seen':>6} {'Encoded':>8} "
          f"{'Broadcast':>12}  {'CASD margin':>14}  {'Delta':>8}  Verdict")
    print("-" * 105)

    for cls_id in sorted(STUFF_IDS_0IDX):
        cname  = CLASS_NAMES[cls_id]
        novel  = "NOVEL" if cls_id in NOVEL_IDS else "base"
        bc_m   = BROADCAST_MARGINS.get(cls_id, float("nan"))
        seen   = n_vox_seen[cls_id]
        enc    = n_vox_encoded[cls_id]

        if casd_margins[cls_id]:
            arr    = np.array(casd_margins[cls_id])
            casd_m = arr.mean()
            casd_s = arr.std()
            delta  = casd_m - bc_m
            better  = ("✅ better"  if casd_m > bc_m + 0.01 else
                       "⚠️  similar" if abs(casd_m - bc_m) <= 0.01 else "❌ worse")
            sign    = "✅ positive" if casd_m > 0 else "❌ negative"
            print(f"  [{cls_id:2d}] {cname:<18} {novel:<6} "
                  f"{seen:6d} {enc:8d} "
                  f"  {bc_m:+.4f}     {casd_m:+.4f}±{casd_s:.4f}"
                  f"  {delta:+.4f}  {sign} / {better}")
        else:
            print(f"  [{cls_id:2d}] {cname:<18} {novel:<6} "
                  f"{seen:6d} {enc:8d}  {bc_m:+.4f}      N/A")

    print()
    print("═" * 60)
    print(f"GATE 5e-CASD  (crop_half_px={args.crop_half_px})")
    veg_pass = (bool(casd_margins[15]) and np.mean(casd_margins[15]) > 0)
    all_better = all(
        (bool(casd_margins[i]) and np.mean(casd_margins[i]) > BROADCAST_MARGINS.get(i, 0))
        for i in STUFF_IDS_0IDX
    )
    print(f"  Vegetation CASD margin > 0   : {'✅ PASS' if veg_pass else '❌ FAIL'}")
    print(f"  All STUFF CASD > broadcast   : {'✅ PASS' if all_better else '⚠️  partial'}")

    if veg_pass:
        print("\n  ✅ CASD is a valid L_V teacher for STUFF including vegetation.")
        print("     Proceed: job1c = L_O + L_V(broadcast THING) + CASD(online STUFF).")
    else:
        print("\n  ❌ Vegetation CASD margin still negative.")
        print(f"     Try: --crop_half_px {args.crop_half_px + 56} for larger context.")
        print("     Or: check best_camera_crop is selecting well-centred projections.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_dir",
        default="/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features_v2/v1.0-mini")
    parser.add_argument("--dataroot",
        default="/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes_mini")
    parser.add_argument("--version",   default="v1.0-mini")
    parser.add_argument("--proto_path",
        default="panoptic/clip_prototypes/text_prototypes_vitl14_224_v2.pt")
    parser.add_argument("--device",    default="cuda")
    parser.add_argument("--n_samples", type=int, default=50,
        help="Samples to process (default 50 for quick run; omit for all 404)")
    parser.add_argument("--max_vox_per_class", type=int, default=30,
        help="Max STUFF voxels per class per sample (default 30)")
    parser.add_argument("--crop_half_px", type=int, default=112,
        help="Half-side of crop window in pixels: 112→224px, 168→336px (default 112)")
    args = parser.parse_args()
    run(args)
