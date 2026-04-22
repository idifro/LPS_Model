"""Visualize LiDAR-to-camera overlay with per-class CLIP semantic proxy coloring.

Projects voxel 3D points onto all 6 camera images using nuScenes extrinsics/intrinsics.
Applies frustum filtering (behind camera, outside FOV) and depth cutoff (50m default).
Colors each projected point by its zero-shot CLIP class prediction.

Usage (GPU node — needs backbone + CLIP):
    python -u panoptic/scripts/visualize_lidar_camera_overlay.py

Usage (CPU — if .npy files already saved from verify_phase3_semantic_proxy.py):
    python -u panoptic/scripts/visualize_lidar_camera_overlay.py --from_saved panoptic/vis_phase3/sample_000

Folder structure created:
    panoptic/viz/
    ├── sample_000/
    │   ├── data/                    # Raw numpy arrays
    │   │   ├── coords_xyz.npy
    │   │   ├── gt_sem.npy
    │   │   ├── pred_sem.npy
    │   │   ├── similarity.npy
    │   │   ├── confidence.npy
    │   │   └── confusion.npy
    │   ├── overlay_pred/            # Predicted class colored overlays
    │   │   ├── CAM_FRONT.png
    │   │   ├── CAM_FRONT_LEFT.png
    │   │   ├── CAM_FRONT_RIGHT.png
    │   │   ├── CAM_BACK.png
    │   │   ├── CAM_BACK_LEFT.png
    │   │   └── CAM_BACK_RIGHT.png
    │   ├── overlay_gt/             # GT class colored overlays
    │   │   └── ... (same 6 cameras)
    │   ├── bev/                    # Bird's eye view plots
    │   │   ├── bev_pred.png
    │   │   └── bev_gt.png
    │   └── legend.png              # Class color legend
    ├── sample_001/
    │   └── ...
    └── summary.txt
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene")

# ── Lazy imports for GPU-optional execution ──
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from panoptic.dataset.nuscenes_panoptic import (
    NUSCENES_CLASS_NAMES, NUM_CLASSES, CAMERAS, IGNORE_LABEL,
)


# ══════════════════════════════════════════════════════════════════════════════
# 16-class color palette (distinct, colorblind-friendly where possible)
# ══════════════════════════════════════════════════════════════════════════════
CLASS_COLORS = np.array([
    [255, 120,  50],   # 0  barrier         - orange
    [255, 192, 203],   # 1  bicycle          - pink
    [255, 255,   0],   # 2  bus              - yellow
    [  0, 150, 245],   # 3  car              - blue
    [  0, 255, 255],   # 4  construction_veh - cyan
    [200, 180,   0],   # 5  motorcycle       - dark yellow
    [255,   0,   0],   # 6  pedestrian       - red
    [255,   0, 255],   # 7  traffic_cone     - magenta
    [100, 150, 200],   # 8  trailer          - steel blue
    [  0, 255,   0],   # 9  truck            - green
    [128, 128, 128],   # 10 driveable_surf   - gray
    [235, 200, 150],   # 11 other_flat        - tan
    [ 75,   0,  75],   # 12 sidewalk          - dark purple
    [150, 240,  80],   # 13 terrain           - lime
    [230, 230, 250],   # 14 manmade           - lavender
    [ 34, 139,  34],   # 15 vegetation        - forest green
], dtype=np.uint8)

IGNORE_COLOR = np.array([40, 40, 40], dtype=np.uint8)  # dark gray for ignore


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", default="/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes_mini")
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--ckpt", default="models/nuscenes_openseg.pth.tar")
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--split", default="val")
    p.add_argument("--save_dir", default="panoptic/viz")
    p.add_argument("--max_depth", type=float, default=50.0)
    p.add_argument("--point_size", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.7, help="Point opacity (0-1)")
    p.add_argument("--from_saved", default=None, help="Load from saved .npy instead of running backbone")
    p.add_argument("--clip_model", default="ViT-L/14@336px")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Projection
# ══════════════════════════════════════════════════════════════════════════════

def project_points_to_camera(points_3d, nusc, sample, cam_name, max_depth=50.0):
    """Project 3D points (ego frame) onto a camera image.

    Args:
        points_3d: (N, 3) xyz in ego frame at the LiDAR timestamp
        nusc: NuScenes instance
        sample: sample record dict
        cam_name: e.g. "CAM_FRONT"
        max_depth: discard points beyond this depth (meters)

    Returns:
        uv: (K, 2) pixel coordinates of valid projected points
        valid_idx: (K,) indices into original points_3d
        cam_path: str, absolute path to camera image
    """
    from pyquaternion import Quaternion

    # Get camera sample_data
    cam_sd = nusc.get("sample_data", sample["data"][cam_name])
    cam_path = os.path.join(nusc.dataroot, cam_sd["filename"])

    # Get LiDAR sample_data
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])

    # ── Transform: ego @ lidar_time → global → ego @ cam_time → camera ──

    # Ego pose at LiDAR time
    ego_lidar = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    R_ego_lidar = Quaternion(ego_lidar["rotation"]).rotation_matrix
    t_ego_lidar = np.array(ego_lidar["translation"])

    # LiDAR calibration (lidar → ego)
    lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    R_lidar = Quaternion(lidar_cs["rotation"]).rotation_matrix
    t_lidar = np.array(lidar_cs["translation"])

    # Ego pose at camera time
    ego_cam = nusc.get("ego_pose", cam_sd["ego_pose_token"])
    R_ego_cam = Quaternion(ego_cam["rotation"]).rotation_matrix
    t_ego_cam = np.array(ego_cam["translation"])

    # Camera calibration (camera → ego)
    cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    R_cam = Quaternion(cam_cs["rotation"]).rotation_matrix
    t_cam = np.array(cam_cs["translation"])
    intrinsic = np.array(cam_cs["camera_intrinsic"])  # (3, 3)

    # Our voxel_feats are already xyz in ego frame (at lidar time,
    # since we used raw pts[:, :3] which is in lidar frame, but
    # actually nuscenes LidarPointCloud is in lidar frame, not ego).
    # Need to check: vox_feats = pts[vox_idx, :3] where pts comes from
    # LidarPointCloud.from_file — which is in LIDAR frame.

    # So points_3d are in LIDAR frame. Transform chain:
    # lidar → ego_lidar → global → ego_cam → camera

    # Step 1: lidar → ego (at lidar time)
    pts_ego = (R_lidar @ points_3d.T).T + t_lidar  # (N, 3)

    # Step 2: ego_lidar → global
    pts_global = (R_ego_lidar @ pts_ego.T).T + t_ego_lidar  # (N, 3)

    # Step 3: global → ego_cam
    pts_ego_cam = (R_ego_cam.T @ (pts_global - t_ego_cam).T).T  # (N, 3)

    # Step 4: ego_cam → camera
    pts_cam = (R_cam.T @ (pts_ego_cam - t_cam).T).T  # (N, 3)

    # ── Frustum filtering ──
    depth = pts_cam[:, 2]

    # Filter 1: behind camera (depth <= 0)
    valid = depth > 0.5  # small margin to avoid division issues

    # Filter 2: max depth
    valid &= depth <= max_depth

    # Project to image plane
    pts_img = (intrinsic @ pts_cam[valid].T).T  # (K, 3)
    uv = pts_img[:, :2] / pts_img[:, 2:3]  # (K, 2)

    # Filter 3: inside image bounds
    img_h, img_w = cam_sd["height"], cam_sd["width"]
    in_bounds = (
        (uv[:, 0] >= 0) & (uv[:, 0] < img_w) &
        (uv[:, 1] >= 0) & (uv[:, 1] < img_h)
    )

    valid_indices = np.where(valid)[0][in_bounds]
    uv_valid = uv[in_bounds]

    return uv_valid, valid_indices, cam_path, (img_h, img_w)


# ══════════════════════════════════════════════════════════════════════════════
# Drawing
# ══════════════════════════════════════════════════════════════════════════════

def draw_overlay(cam_path, uv, colors, point_size=3, alpha=0.7):
    """Draw colored points on camera image.

    Args:
        cam_path: path to camera image
        uv: (K, 2) pixel coordinates
        colors: (K, 3) uint8 RGB colors
        point_size: radius of each point
        alpha: blending factor

    Returns:
        PIL Image with overlay
    """
    img = Image.open(cam_path).convert("RGB")
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)

    for (u, v), color in zip(uv.astype(int), colors):
        r = point_size
        draw.ellipse([u - r, v - r, u + r, v + r],
                     fill=tuple(color.tolist()))

    # Alpha blend
    result = Image.blend(img, overlay, alpha)
    return result


def draw_legend(save_path):
    """Draw a class color legend."""
    if not HAS_MPL:
        print("    [SKIP] matplotlib not available for legend")
        return

    fig, ax = plt.subplots(1, 1, figsize=(4, 6))
    patches = []
    for i in range(NUM_CLASSES):
        color = CLASS_COLORS[i] / 255.0
        patches.append(mpatches.Patch(color=color, label=f"{i}: {NUSCENES_CLASS_NAMES[i]}"))
    patches.append(mpatches.Patch(color=IGNORE_COLOR / 255.0, label="ignore (255)"))

    ax.legend(handles=patches, loc="center", fontsize=10, frameon=False)
    ax.axis("off")
    ax.set_title("Class Color Legend", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def draw_bev(coords_xyz, labels, save_path, title="BEV"):
    """Draw bird's eye view scatter plot."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    for c in range(NUM_CLASSES):
        mask = labels == c
        if mask.sum() == 0:
            continue
        color = CLASS_COLORS[c] / 255.0
        ax.scatter(coords_xyz[mask, 0], coords_xyz[mask, 1],
                   c=[color], s=0.5, alpha=0.6, label=NUSCENES_CLASS_NAMES[c])

    # Ignore
    mask_ign = labels == IGNORE_LABEL
    if mask_ign.sum() > 0:
        ax.scatter(coords_xyz[mask_ign, 0], coords_xyz[mask_ign, 1],
                   c=[IGNORE_COLOR / 255.0], s=0.3, alpha=0.2, label="ignore")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=6, markerscale=10)
    ax.set_xlim(-55, 55)
    ax.set_ylim(-55, 55)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if not HAS_PIL:
        print("ABORT: Pillow required. pip install Pillow")
        sys.exit(1)

    os.makedirs(args.save_dir, exist_ok=True)

    # Load nuScenes (needed for projection transforms)
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    # Determine if we need GPU (backbone+CLIP) or can use saved data
    if args.from_saved:
        print(f"Loading from saved: {args.from_saved}")
        # Just visualize one saved sample
        _visualize_from_saved(args, nusc)
        return

    # ── Full pipeline: backbone + CLIP + projection ──
    import torch
    import torch.nn.functional as F
    import MinkowskiEngine as ME
    import clip

    from models.disnet import DisNet, state_dict_remove_moudle
    from panoptic.dataset.nuscenes_panoptic import NuScenesPanoptic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load backbone
    class Cfg:
        feature_2d_extractor = "openseg"
        arch_3d = "MinkUNet18A"
    backbone = DisNet(Cfg()).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = state_dict_remove_moudle(ckpt["state_dict"])
    backbone.load_state_dict(state, strict=True)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    print("[Backbone] Loaded and frozen.")

    # Load CLIP text prototypes
    PROMPT_TEMPLATES = [
        "a photo of a {}.",
        "a photo of a {} in a driving scene.",
        "a {} in a street scene.",
        "there is a {} in the scene.",
        "a photo showing {}.",
    ]
    print(f"[CLIP] Loading {args.clip_model}...")
    clip_model, _ = clip.load(args.clip_model, device=device)
    clip_model.eval()

    prototypes = []
    for cls_name in NUSCENES_CLASS_NAMES:
        prompts = [t.format(cls_name) for t in PROMPT_TEMPLATES]
        tokens = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            tf = clip_model.encode_text(tokens).float()
            tf = F.normalize(tf, dim=-1).mean(dim=0)
            tf = F.normalize(tf, dim=0)
        prototypes.append(tf)
    text_prototypes = torch.stack(prototypes, dim=0)  # (16, 768)
    print(f"[CLIP] Text prototypes: {text_prototypes.shape}")

    # Load dataset
    dataset = NuScenesPanoptic(
        args.dataroot, args.version, split=args.split, augment=False
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=lambda x: x[0],
    )

    # Draw legend once
    draw_legend(os.path.join(args.save_dir, "legend.png"))

    summary_lines = []

    for i, batch in enumerate(loader):
        if i >= args.num_samples:
            break

        token = batch["token"]
        sample = nusc.get("sample", token)
        gt_sem = batch["gt_sem"].numpy()
        coords_xyz = batch["vox_feats"].numpy()  # (V, 3) raw XYZ in lidar frame
        V = coords_xyz.shape[0]

        # Extract backbone features
        vox_coords = batch["vox_coords"]
        vox_feats = batch["vox_feats"]
        batch_coords = torch.cat([
            torch.zeros(V, 1, dtype=torch.int32), vox_coords
        ], dim=1)
        sinput = ME.SparseTensor(
            vox_feats.float(), coordinates=batch_coords.int(), device=device
        )
        with torch.no_grad():
            output = backbone(sinput)
        feats = output.F if hasattr(output, "F") else output

        # Semi-cosine similarity (OpenScene convention): raw features @ L2-normalized text
        sim = torch.mm(feats.float().half(), text_prototypes.half().T).float().cpu().numpy()
        pred_sem = sim.argmax(axis=1)
        confidence = sim.max(axis=1)

        # Accuracy
        valid = gt_sem != IGNORE_LABEL
        acc = (pred_sem[valid] == gt_sem[valid]).mean() if valid.sum() > 0 else 0.0

        print(f"\nSample {i} (token: {token[:12]}...) V={V}, acc={acc:.4f}")

        # ── Create folder structure ──
        sample_dir = os.path.join(args.save_dir, f"sample_{i:03d}")
        data_dir = os.path.join(sample_dir, "data")
        pred_dir = os.path.join(sample_dir, "overlay_pred")
        gt_dir = os.path.join(sample_dir, "overlay_gt")
        bev_dir = os.path.join(sample_dir, "bev")

        for d in [data_dir, pred_dir, gt_dir, bev_dir]:
            os.makedirs(d, exist_ok=True)

        # Save data
        np.save(os.path.join(data_dir, "coords_xyz.npy"), coords_xyz)
        np.save(os.path.join(data_dir, "gt_sem.npy"), gt_sem)
        np.save(os.path.join(data_dir, "pred_sem.npy"), pred_sem)
        np.save(os.path.join(data_dir, "similarity.npy"), sim)
        np.save(os.path.join(data_dir, "confidence.npy"), confidence)

        # Confusion matrix
        if valid.sum() > 0:
            conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
            for gc, pc in zip(gt_sem[valid], pred_sem[valid]):
                conf[int(gc), int(pc)] += 1
            np.save(os.path.join(data_dir, "confusion.npy"), conf)

        # ── BEV plots ──
        draw_bev(coords_xyz, pred_sem, os.path.join(bev_dir, "bev_pred.png"),
                 title=f"BEV Predicted (acc={acc:.3f})")
        draw_bev(coords_xyz, gt_sem, os.path.join(bev_dir, "bev_gt.png"),
                 title="BEV Ground Truth")

        # ── Camera overlays ──
        for cam_name in CAMERAS:
            try:
                uv, valid_idx, cam_path, (img_h, img_w) = project_points_to_camera(
                    coords_xyz, nusc, sample, cam_name, max_depth=args.max_depth
                )
            except Exception as e:
                print(f"  {cam_name}: projection failed — {e}")
                continue

            n_proj = len(valid_idx)
            print(f"  {cam_name}: {n_proj} points projected (of {V})")

            if n_proj == 0:
                continue

            # Predicted class overlay
            pred_colors = CLASS_COLORS[pred_sem[valid_idx]]
            img_pred = draw_overlay(cam_path, uv, pred_colors,
                                    point_size=args.point_size, alpha=args.alpha)
            img_pred.save(os.path.join(pred_dir, f"{cam_name}.png"))

            # GT class overlay
            gt_labels = gt_sem[valid_idx]
            gt_colors = np.where(
                gt_labels[:, None] == IGNORE_LABEL,
                IGNORE_COLOR[None, :],
                CLASS_COLORS[np.clip(gt_labels, 0, NUM_CLASSES - 1)]
            )
            img_gt = draw_overlay(cam_path, uv, gt_colors,
                                  point_size=args.point_size, alpha=args.alpha)
            img_gt.save(os.path.join(gt_dir, f"{cam_name}.png"))

        summary_lines.append(
            f"sample_{i:03d}: token={token[:16]}  V={V}  acc={acc:.4f}  "
            f"pred_classes={np.unique(pred_sem).tolist()}"
        )
        print(f"  Saved: {sample_dir}/")

    # Write summary
    with open(os.path.join(args.save_dir, "summary.txt"), "w") as f:
        f.write("Phase 3.3 Semantic Proxy Visualization Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Max depth: {args.max_depth}m\n")
        f.write(f"CLIP model: {args.clip_model}\n\n")
        for line in summary_lines:
            f.write(line + "\n")

    print(f"\nAll visualizations saved to: {args.save_dir}/")
    print("Done.")


def _visualize_from_saved(args, nusc):
    """Visualize from pre-saved .npy files (CPU-only, no backbone needed)."""
    saved_dir = args.from_saved
    coords_xyz = np.load(os.path.join(saved_dir, "coords_xyz.npy"))
    gt_sem = np.load(os.path.join(saved_dir, "gt_sem.npy"))
    pred_sem = np.load(os.path.join(saved_dir, "pred_sem.npy"))

    # We need the sample token to get camera calibration — try to find it
    # from the dataset
    from panoptic.dataset.nuscenes_panoptic import NuScenesPanoptic
    dataset = NuScenesPanoptic(
        args.dataroot, args.version, split=args.split, augment=False
    )

    # Guess sample index from dir name (sample_000 → idx 0)
    basename = os.path.basename(saved_dir.rstrip("/"))
    idx = int(basename.split("_")[-1])
    token = dataset.tokens[idx]
    sample = nusc.get("sample", token)

    V = coords_xyz.shape[0]
    valid = gt_sem != IGNORE_LABEL
    acc = (pred_sem[valid] == gt_sem[valid]).mean() if valid.sum() > 0 else 0.0
    print(f"Sample idx={idx}, token={token[:12]}..., V={V}, acc={acc:.4f}")

    out_dir = os.path.join(args.save_dir, basename)
    pred_dir = os.path.join(out_dir, "overlay_pred")
    gt_dir = os.path.join(out_dir, "overlay_gt")
    bev_dir = os.path.join(out_dir, "bev")
    for d in [pred_dir, gt_dir, bev_dir]:
        os.makedirs(d, exist_ok=True)

    draw_legend(os.path.join(args.save_dir, "legend.png"))
    draw_bev(coords_xyz, pred_sem, os.path.join(bev_dir, "bev_pred.png"), title=f"BEV Predicted (acc={acc:.3f})")
    draw_bev(coords_xyz, gt_sem, os.path.join(bev_dir, "bev_gt.png"), title="BEV Ground Truth")

    for cam_name in CAMERAS:
        try:
            uv, valid_idx, cam_path, _ = project_points_to_camera(
                coords_xyz, nusc, sample, cam_name, max_depth=args.max_depth
            )
        except Exception as e:
            print(f"  {cam_name}: failed — {e}")
            continue

        n_proj = len(valid_idx)
        print(f"  {cam_name}: {n_proj} points")
        if n_proj == 0:
            continue

        pred_colors = CLASS_COLORS[pred_sem[valid_idx]]
        img_pred = draw_overlay(cam_path, uv, pred_colors, point_size=args.point_size, alpha=args.alpha)
        img_pred.save(os.path.join(pred_dir, f"{cam_name}.png"))

        gt_labels = gt_sem[valid_idx]
        gt_colors = np.where(
            gt_labels[:, None] == IGNORE_LABEL,
            IGNORE_COLOR[None, :],
            CLASS_COLORS[np.clip(gt_labels, 0, NUM_CLASSES - 1)]
        )
        img_gt = draw_overlay(cam_path, uv, gt_colors, point_size=args.point_size, alpha=args.alpha)
        img_gt.save(os.path.join(gt_dir, f"{cam_name}.png"))

    print(f"Saved to: {out_dir}/")


if __name__ == "__main__":
    main()
