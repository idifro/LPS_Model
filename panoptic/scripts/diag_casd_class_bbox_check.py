"""Diagnostic: test class-level bbox CASD for vegetation.

Per-voxel CASD failed (diag_casd_margin_check.py): a single voxel's crop captures
global urban scene context, not the specific class. Larger crops made it worse.

This script tests the class-level bbox approach instead:
  For each stuff class in each sample:
    1. Project ALL voxels of that class visible in a camera
    2. Find the 2D bounding box of projected points
    3. Expand bbox by a factor → crop covering the whole class region
    4. Encode with CLIP encode_image() → CLS
    5. Average across cameras (weighted by n_visible_points)

Hypothesis: vegetation voxels cluster in tree canopies / roadsides.
Their 2D bbox in a camera = a region dominated by trees → CLS should align
with the "vegetation" text prototype (positive margin).

Also tests text prototype as L_V target for base stuff (guaranteed positive margin
by construction from v2 prototypes — confirms this as a zero-compute alternative).

Usage:
    python panoptic/scripts/diag_casd_class_bbox_check.py --device cuda
    python panoptic/scripts/diag_casd_class_bbox_check.py --device cuda --expand 2.5
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

STUFF_IDS_0IDX = set(range(10, 16))
CLASS_NAMES = [
    "barrier", "bicycle", "bus", "car", "construction_vehicle",
    "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck",
    "driveable_surface", "other_flat", "sidewalk", "terrain", "manmade", "vegetation",
]
NOVEL_IDS = {2, 5, 6, 15}

BROADCAST_MARGINS = {
    10: +0.0025, 11: -0.0109, 12: -0.0123,
    13: -0.0197, 14: +0.0060, 15: -0.0282,
}
# From diag_casd_margin_check at half_px=112 (per-voxel, failed)
PER_VOX_MARGINS = {
    10: -0.0072, 11: -0.0217, 12: -0.0203,
    13: -0.0309, 14: -0.0063, 15: -0.0395,
}

CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


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
    # Majority-vote class per voxel
    vox_sem = np.zeros(len(vox_idx), dtype=np.int32)
    for vi in range(len(vox_idx)):
        modes = sem[inv_idx == vi]
        if len(modes):
            vox_sem[vi] = int(np.bincount(modes.astype(np.int64)).argmax())
    return vox_xyz, vox_sem


def pad_square(crop):
    w, h = crop.size
    if w == h:
        return crop
    side = max(w, h)
    sq = Image.new("RGB", (side, side), (128, 128, 128))
    sq.paste(crop, ((side - w) // 2, (side - h) // 2))
    return sq


def class_bbox_crop(nusc, sample_token, class_vox_xyz, expand, min_pts=10):
    """Project all voxels of one class across 6 cameras. For each camera that
    sees >= min_pts voxels, compute 2D bbox → expand → crop. Return list of
    (PIL crop, n_pts) for multi-view averaging. Crops are square-padded.
    """
    sample  = nusc.get("sample", sample_token)
    results = []   # (PIL crop, n_pts)

    for cam_name in CAMERAS:
        uv, depths, valid = project_lidar_to_camera(
            nusc, sample_token, class_vox_xyz, cam_name
        )
        n_pts = int(valid.sum())
        if n_pts < min_pts:
            continue

        uv_v = uv[valid]   # (n_pts, 2)
        u_min, u_max = float(uv_v[:, 0].min()), float(uv_v[:, 0].max())
        v_min, v_max = float(uv_v[:, 1].min()), float(uv_v[:, 1].max())
        cx = (u_min + u_max) / 2.0
        cy = (v_min + v_max) / 2.0
        half_w = (u_max - u_min) / 2.0 * expand
        half_h = (v_max - v_min) / 2.0 * expand
        # Keep the crop square (use the larger half)
        half = max(half_w, half_h, 20.0)

        x0 = int(max(0,     cx - half))
        y0 = int(max(0,     cy - half))
        x1 = int(min(CAM_W, cx + half))
        y1 = int(min(CAM_H, cy + half))
        if (x1 - x0) < 16 or (y1 - y0) < 16:
            continue

        img_sd   = nusc.get("sample_data", sample["data"][cam_name])
        img_path = os.path.join(nusc.dataroot, img_sd["filename"])
        img      = Image.open(img_path).convert("RGB")
        crop     = img.crop((x0, y0, x1, y1))
        results.append((crop, n_pts))

    return results


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

    proto_data   = torch.load(args.proto_path, map_location="cpu")
    proto_tensor = proto_data["embeddings"] if isinstance(proto_data, dict) else proto_data
    protos       = F.normalize(proto_tensor.float(), dim=-1)         # (16, 768) torch
    protos_np    = protos.numpy()

    # Also test text prototype as target (Option 2 baseline)
    # For each base stuff class: target = text_prototype (perfect margin by construction)
    # For novel vegetation: would be cheating if used at training time (excluded from L_V novel)
    # We report it here as the guaranteed upper bound for base stuff.
    text_margins = {}
    for cls_id in STUFF_IDS_0IDX:
        sim_own  = float(protos_np[cls_id] @ protos_np[cls_id])   # always 1.0
        tmp      = protos_np[cls_id] @ protos_np.T                # (16,)
        tmp2     = tmp.copy(); tmp2[cls_id] = -1.0
        text_margins[cls_id] = float(1.0 - tmp2.max())            # 1 - sim_best_other

    npz_paths = sorted(glob.glob(f"{args.feat_dir}/*.npz"))
    if args.n_samples and args.n_samples < len(npz_paths):
        rng = np.random.default_rng(42)
        npz_paths = [npz_paths[i] for i in rng.choice(
            len(npz_paths), args.n_samples, replace=False)]

    print(f"Loading nuScenes {args.version} from {args.dataroot}...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    print(f"Processing {len(npz_paths)} samples  "
          f"(class-level bbox expand={args.expand}×, min_pts_per_cam={args.min_pts})\n")

    bbox_margins = {cls_id: [] for cls_id in STUFF_IDS_0IDX}
    n_samples_with_class = {cls_id: 0 for cls_id in STUFF_IDS_0IDX}

    for npz_path in tqdm(npz_paths, desc="Samples"):
        token = os.path.basename(npz_path).replace(".npz", "")
        try:
            vox_xyz, vox_sem = voxelize_sample(nusc, token, args.dataroot)
        except Exception as e:
            print(f"  SKIP voxelize {token[:16]}: {e}")
            continue

        # For each stuff class, collect class-level crops and batch encode
        crops_batch = []   # (cls_id, PIL crop, n_pts)
        for cls_id in sorted(STUFF_IDS_0IDX):
            # 0-indexed class: add 1 to match 1-indexed vox_sem
            class_mask = (vox_sem == cls_id + 1)
            if class_mask.sum() < args.min_pts:
                continue
            n_samples_with_class[cls_id] += 1
            class_xyz   = vox_xyz[class_mask]
            view_crops  = class_bbox_crop(nusc, token, class_xyz,
                                          expand=args.expand,
                                          min_pts=args.min_pts)
            for crop, n_pts in view_crops:
                crops_batch.append((cls_id, crop, n_pts))

        if not crops_batch:
            continue

        try:
            imgs = torch.stack(
                [preprocess(pad_square(item[1])) for item in crops_batch], dim=0
            ).to(args.device).half()
            with torch.no_grad():
                feats = model.encode_image(imgs)          # (N, 768)
                feats = F.normalize(feats.float(), dim=-1).cpu().numpy()
        except Exception as e:
            print(f"  encode error {token[:16]}: {e}")
            continue

        # Multi-view weighted average per class in this sample
        class_feat_acc  = {}   # cls_id → (768,) accumulator
        class_wt_acc    = {}   # cls_id → float

        for i, (cls_id, _, n_pts) in enumerate(crops_batch):
            if cls_id not in class_feat_acc:
                class_feat_acc[cls_id] = np.zeros(768, dtype=np.float64)
                class_wt_acc[cls_id]   = 0.0
            class_feat_acc[cls_id] += feats[i].astype(np.float64) * n_pts
            class_wt_acc[cls_id]   += n_pts

        for cls_id, acc in class_feat_acc.items():
            norm = np.linalg.norm(acc / class_wt_acc[cls_id])
            if norm < 1e-8:
                continue
            feat = acc / class_wt_acc[cls_id] / norm   # L2-normalised average
            sims    = feat @ protos_np.T                 # (16,)
            sim_own = float(sims[cls_id])
            tmp     = sims.copy(); tmp[cls_id] = -1.0
            margin  = sim_own - float(tmp.max())
            bbox_margins[cls_id].append(margin)

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\n{'Class':<22} {'Novel':<6} "
          f"{'Broadcast':>12}  {'PerVox(fail)':>13}  {'BboxCASD':>12}  "
          f"{'TextProto':>10}  Verdict")
    print("-" * 110)

    for cls_id in sorted(STUFF_IDS_0IDX):
        cname  = CLASS_NAMES[cls_id]
        novel  = "NOVEL" if cls_id in NOVEL_IDS else "base"
        bc_m   = BROADCAST_MARGINS.get(cls_id, float("nan"))
        pv_m   = PER_VOX_MARGINS.get(cls_id, float("nan"))
        tp_m   = text_margins.get(cls_id, float("nan"))

        if bbox_margins[cls_id]:
            arr      = np.array(bbox_margins[cls_id])
            bbox_m   = arr.mean()
            bbox_s   = arr.std()
            sign     = "✅" if bbox_m > 0 else "❌"
            better   = "✅ better" if bbox_m > bc_m + 0.01 else "⚠️  similar"
            novel_str = "text=cheating" if cls_id in NOVEL_IDS else f"text={tp_m:+.4f}"
            print(f"  [{cls_id:2d}] {cname:<18} {novel:<6} "
                  f"  {bc_m:+.6f}    {pv_m:+.6f}    {sign}{bbox_m:+.4f}±{bbox_s:.4f}"
                  f"  {novel_str:>16}  {'✅ PASS' if bbox_m > 0 else '❌ FAIL'}")
        else:
            print(f"  [{cls_id:2d}] {cname:<18} {novel:<6} "
                  f"  {bc_m:+.6f}    {pv_m:+.6f}     N/A  n={n_samples_with_class[cls_id]}")

    print()
    print("Text prototype margins for base STUFF (Option 2 upper bound):")
    for cls_id in sorted(STUFF_IDS_0IDX):
        if cls_id not in NOVEL_IDS:
            print(f"  {CLASS_NAMES[cls_id]:<22}: sim_own=1.000 vs best_other  "
                  f"margin={text_margins[cls_id]:+.4f}  (guaranteed positive)")

    print()
    print("═" * 60)
    veg_pass = bool(bbox_margins[15]) and np.mean(bbox_margins[15]) > 0
    print(f"Vegetation class-bbox CASD margin > 0  : {'✅ PASS' if veg_pass else '❌ FAIL'}")

    if veg_pass:
        print("\n  ✅ Class-level bbox CASD works for vegetation.")
        print("     Recommended L_V design for job1c:")
        print("       THING voxels:        broadcast from obj_feats (done)")
        print("       Base STUFF voxels:   text prototype target (guaranteed positive)")
        print("       Novel veg voxels:    class-level bbox CASD (confirmed here)")
    else:
        print("\n  ❌ Class-level bbox also fails for vegetation.")
        print("     Fallback: use text prototype for base stuff only.")
        print("     Novel vegetation: rely on L7 backbone margins + C1 boundary learning.")
        print("     Retry with --expand 3.0 or --min_pts 5 to check threshold sensitivity.")


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
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--expand",    type=float, default=1.5,
        help="Bbox expansion factor for class-level crop (default 1.5)")
    parser.add_argument("--min_pts",   type=int, default=10,
        help="Min projected voxels per camera to include that view (default 10)")
    args = parser.parse_args()
    run(args)
