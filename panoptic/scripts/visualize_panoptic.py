"""Panoptic visualization: per-camera overlay images + PLY point clouds + BEV.

For each sample, generates per-camera images for:
  A) Panoptic prediction: instance-aware coloring with per-instance legend
  B) Ground truth:        instance-aware coloring with per-instance legend

Also generates PLY files (semantic, panoptic, GT, instance) and BEV comparison.

Usage:
    python -u panoptic/scripts/visualize_panoptic.py \
        --checkpoint panoptic/checkpoints_full/best.pth \
        --version v1.0-trainval \
        --dataroot /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes \
        --num_samples 3
"""
import argparse
import colorsys
import gc
import os
import sys

import numpy as np
import torch
import open3d as o3d

sys.path.insert(0, "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene")

from PIL import Image, ImageDraw

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from nuscenes.nuscenes import NuScenes

from panoptic.dataset.nuscenes_panoptic import (
    NuScenesPanoptic, NUSCENES_CLASS_NAMES, NUM_CLASSES, CAMERAS, IGNORE_LABEL,
)
from panoptic.models.mask_decoder import PanopticDecoder
from panoptic.train.train_panoptic import (
    load_backbone, extract_backbone_features, panoptic_inference, load_config,
)
from panoptic.scripts.visualize_lidar_camera_overlay import (
    CLASS_COLORS, IGNORE_COLOR, project_points_to_camera,
)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    """Parse command-line arguments for the panoptic visualization script.

    Called at startup to configure visualization parameters such as
    checkpoint path, dataset version, number of samples, and rendering
    options (point size, alpha blending, max projection depth).

    Steps:
        1. Creates an ArgumentParser with relevant options.
        2. Adds args for config, checkpoint, dataroot, version, split,
           num_samples, save_dir, max_depth, point_size, and alpha.
        3. Parses sys.argv and returns the namespace.

    Returns:
        argparse.Namespace: Parsed arguments controlling the visualization run.
    """
    p = argparse.ArgumentParser(description="Panoptic visualization")
    p.add_argument("--config", default="panoptic/configs/full_nuscenes.yaml")
    p.add_argument("--checkpoint", default="panoptic/checkpoints_full/best.pth")
    p.add_argument("--dataroot", default=None, help="Override config dataroot")
    p.add_argument("--version", default="v1.0-mini", help="Override config version")
    p.add_argument("--split", default="val")
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--save_dir", default="panoptic/viz_panoptic")
    p.add_argument("--max_depth", type=float, default=50.0)
    p.add_argument("--point_size", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.7)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Panoptic coloring
# ══════════════════════════════════════════════════════════════════════════════

def generate_panoptic_colors(sem_pred, inst_pred, class_colors=CLASS_COLORS):
    """Stuff = class color. Thing = per-instance HSV-shifted shade of class color."""
    colors = class_colors[sem_pred].copy()
    for cls in range(10):
        cls_mask = sem_pred == cls
        if not cls_mask.any():
            continue
        unique_insts = np.unique(inst_pred[cls_mask])
        base_rgb = class_colors[cls] / 255.0
        h, s, v = colorsys.rgb_to_hsv(float(base_rgb[0]), float(base_rgb[1]), float(base_rgb[2]))
        for idx, iid in enumerate(unique_insts):
            if iid == 0:
                continue
            inst_mask = cls_mask & (inst_pred == iid)
            h_new = (h + idx * 0.103) % 1.0
            v_new = max(0.4, min(1.0, v + (idx % 3 - 1) * 0.15))
            r, g, b = colorsys.hsv_to_rgb(h_new, s, v_new)
            colors[inst_mask] = np.array([r * 255, g * 255, b * 255], dtype=np.uint8)
    return colors


def generate_gt_panoptic_colors(gt_sem, gt_masks, gt_classes, class_colors=CLASS_COLORS):
    """Generate panoptic colors from GT masks: each GT mask = one instance."""
    V = len(gt_sem)
    colors = np.where(
        gt_sem[:, None] == IGNORE_LABEL,
        IGNORE_COLOR[None, :],
        class_colors[np.clip(gt_sem, 0, NUM_CLASSES - 1)]
    )
    colors = colors.copy()
    # Build per-voxel instance ID from GT masks
    gt_inst = np.zeros(V, dtype=np.int32)
    for m_idx in range(len(gt_classes)):
        cls = gt_classes[m_idx]
        mask = gt_masks[m_idx]
        if cls < 10:  # thing class — assign instance ID
            gt_inst[mask] = m_idx + 1

    # Color thing instances with HSV shifts (same logic as pred)
    for cls in range(10):
        cls_mask = gt_sem == cls
        if not cls_mask.any():
            continue
        unique_insts = np.unique(gt_inst[cls_mask])
        base_rgb = class_colors[cls] / 255.0
        h, s, v = colorsys.rgb_to_hsv(float(base_rgb[0]), float(base_rgb[1]), float(base_rgb[2]))
        for idx, iid in enumerate(unique_insts):
            if iid == 0:
                continue
            inst_mask = cls_mask & (gt_inst == iid)
            h_new = (h + idx * 0.103) % 1.0
            v_new = max(0.4, min(1.0, v + (idx % 3 - 1) * 0.15))
            r, g, b = colorsys.hsv_to_rgb(h_new, s, v_new)
            colors[inst_mask] = np.array([r * 255, g * 255, b * 255], dtype=np.uint8)
    return colors, gt_inst


def generate_instance_colors(inst_pred):
    """Random distinct color per instance ID (no class relationship)."""
    V = len(inst_pred)
    colors = np.full((V, 3), 40, dtype=np.uint8)
    for iid in np.unique(inst_pred):
        if iid == 0:
            continue
        hue = (iid * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors[inst_pred == iid] = np.array([r * 255, g * 255, b * 255], dtype=np.uint8)
    return colors


# ══════════════════════════════════════════════════════════════════════════════
# Drawing utilities
# ══════════════════════════════════════════════════════════════════════════════

def draw_overlay_fast(cam_path, uv, colors, point_size=3, alpha=0.7):
    """Draw colored points on camera image using vectorized numpy blending."""
    img = np.array(Image.open(cam_path).convert("RGB"))
    overlay = img.copy()
    h, w = img.shape[:2]
    uv_int = uv.astype(np.int32)
    colors_uint8 = colors.astype(np.uint8)
    for dy in range(-point_size, point_size + 1):
        for dx in range(-point_size, point_size + 1):
            if dx * dx + dy * dy > point_size * point_size:
                continue
            yy = uv_int[:, 1] + dy
            xx = uv_int[:, 0] + dx
            valid = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
            overlay[yy[valid], xx[valid]] = colors_uint8[valid]
    blended = (alpha * overlay.astype(np.float32) + (1 - alpha) * img.astype(np.float32)).astype(np.uint8)
    return Image.fromarray(blended)


def write_ply(path, xyz, rgb):
    """Write colored point cloud to PLY using open3d (fast binary)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(path, pcd)


def make_panoptic_legend(sem, inst, colors, title="Panoptic"):
    """Build matplotlib legend image with per-instance entries for thing classes."""
    patches = []
    for cls in range(10):
        cls_mask = sem == cls
        if not cls_mask.any():
            continue
        unique_insts = np.unique(inst[cls_mask])
        for iid in unique_insts:
            if iid == 0:
                continue
            inst_mask = cls_mask & (inst == iid)
            c = colors[np.where(inst_mask)[0][0]] / 255.0
            patches.append(mpatches.Patch(
                color=c, label=f"{NUSCENES_CLASS_NAMES[cls]} #{iid} ({inst_mask.sum()}pt)"))
    for cls in range(10, NUM_CLASSES):
        cls_mask = sem == cls
        if not cls_mask.any():
            continue
        c = colors[np.where(cls_mask)[0][0]] / 255.0
        patches.append(mpatches.Patch(
            color=c, label=f"{NUSCENES_CLASS_NAMES[cls]} ({cls_mask.sum()}pt)"))
    if not patches:
        return None
    ncol = min(8, len(patches))
    nrow = (len(patches) + ncol - 1) // ncol
    fig, ax = plt.subplots(1, 1, figsize=(14, 0.5 * nrow + 0.5))
    ax.legend(handles=patches, loc="center", ncol=ncol, fontsize=6, frameon=False)
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    plt.tight_layout(pad=0.2)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close()
    return Image.fromarray(buf)


def save_overlay_with_legend(overlay_img, legend_img, path):
    """Stack camera overlay image and legend vertically, save as single PNG."""
    if legend_img is None:
        overlay_img.save(path)
        return
    ow, oh = overlay_img.size
    lw, lh = legend_img.size
    # Resize legend to match overlay width
    scale = ow / lw
    legend_resized = legend_img.resize((ow, int(lh * scale)), Image.LANCZOS)
    lh2 = legend_resized.size[1]
    combined = Image.new("RGB", (ow, oh + lh2), (255, 255, 255))
    combined.paste(overlay_img, (0, 0))
    combined.paste(legend_resized, (0, oh))
    combined.save(path)


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_decoder(cfg, checkpoint_path, device):
    """Load the PanopticDecoder from a saved checkpoint file.

    Used during visualization to reconstruct the trained decoder model
    so that panoptic inference can be run on selected samples.

    Steps:
        1. Reads model hyperparameters from the config dict.
        2. Instantiates a PanopticDecoder with those hyperparameters.
        3. Loads the 'decoder' state dict from the checkpoint file.
        4. Sets the model to eval mode and prints load confirmation.

    Args:
        cfg (dict): Full YAML config dictionary containing a 'model' key
            with decoder architecture hyperparameters.
        checkpoint_path (str): Path to the .pth checkpoint file containing
            a 'decoder' state dict entry.
        device (torch.device): Device to place the model on (cpu or cuda).

    Returns:
        PanopticDecoder: The loaded decoder model in eval mode.
    """
    mcfg = cfg["model"]
    decoder = PanopticDecoder(
        backbone_dim=mcfg["backbone_dim"],
        hidden_dim=mcfg["hidden_dim"],
        num_classes=mcfg["num_classes"],
        num_thing_queries=mcfg["num_thing_queries"],
        num_stuff_classes=mcfg["num_stuff_classes"],
        nheads=mcfg["nheads"],
        dim_feedforward=mcfg["dim_feedforward"],
        num_blocks=mcfg["num_blocks"],
        num_levels=mcfg["num_levels"],
        dropout=mcfg["dropout"],
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()
    print(f"[Decoder] Loaded from {checkpoint_path} (epoch={ckpt.get('epoch', '?')})")
    return decoder


CAMERA_GRID_ORDER = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT",
]


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Generate panoptic visualization outputs for selected nuScenes samples.

    This is the top-level entry point for the visualization pipeline. It
    produces per-camera overlay images, PLY point clouds, and BEV comparison
    plots for both predictions and ground truth.

    Steps:
        1. Parses arguments and loads config, backbone, and decoder.
        2. Randomly selects samples across different scenes.
        3. For each sample, runs panoptic inference and generates colors.
        4. Saves PLY files (semantic, panoptic, GT, instance).
        5. Renders BEV comparison plot (predicted vs GT).
        6. Projects 3D points to each camera and saves overlay images
           with per-instance panoptic legends.
        7. Writes a summary text file listing all visualized samples.

    Returns:
        None
    """
    args = parse_args()
    cfg = load_config(args.config)

    if args.dataroot:
        cfg["data"]["dataroot"] = args.dataroot
    if args.version:
        cfg["data"]["version"] = args.version

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    os.makedirs(args.save_dir, exist_ok=True)

    dcfg = cfg["data"]
    nusc = NuScenes(version=dcfg["version"], dataroot=dcfg["dataroot"], verbose=False)

    backbone = load_backbone(cfg["model"]["backbone_ckpt"], device)
    decoder = load_decoder(cfg, args.checkpoint, device)

    dataset = NuScenesPanoptic(
        dcfg["dataroot"], dcfg["version"], split=args.split,
        voxel_size=dcfg["voxel_size"],
        point_range=np.array(dcfg["point_range"]),
        augment=False,
    )

    num_classes = cfg["model"]["num_classes"]
    overlap_thr = cfg["inference"]["overlap_thr"]
    summary_lines = []

    # ── Randomly select num_samples (scene, keyframe) pairs ──
    # Group dataset indices by scene
    from collections import defaultdict
    scene_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        token = dataset.tokens[idx]
        scene_token = nusc.get("sample", token)["scene_token"]
        scene_to_indices[scene_token].append(idx)
    scene_tokens = list(scene_to_indices.keys())

    rng = np.random.RandomState()  # truly random each run
    rng.shuffle(scene_tokens)

    # Pick num_samples pairs: cycle through scenes, random keyframe each
    selected = []
    scene_cycle = 0
    while len(selected) < args.num_samples:
        st = scene_tokens[scene_cycle % len(scene_tokens)]
        idx = rng.choice(scene_to_indices[st])
        selected.append(idx)
        scene_cycle += 1

    n_scenes = len(scene_tokens)
    print(f"Visualizing {len(selected)} samples from {n_scenes} scenes "
          f"({dcfg['version']} {args.split})...", flush=True)

    for vi, dataset_idx in enumerate(selected):
        batch = dataset[dataset_idx]
        token = batch["token"]
        sample = nusc.get("sample", token)
        coords_xyz = batch["vox_feats"].numpy()
        gt_sem = batch["gt_sem"].numpy()
        gt_masks = batch["gt_masks"].numpy()   # (M, V)
        gt_classes = batch["gt_classes"].numpy()  # (M,)
        V = coords_xyz.shape[0]

        # Inference
        with torch.no_grad():
            feats, coords = extract_backbone_features(backbone, batch, device)
            outputs = decoder(feats, coords)
        pred_logits = outputs["pred_logits"][0].cpu()
        pred_masks = outputs["pred_masks"][0].cpu()
        sem_pred, inst_pred = panoptic_inference(pred_logits, pred_masks, num_classes, overlap_thr)

        n_things = len(set(inst_pred[sem_pred < 10]) - {0})
        n_stuff = len(np.unique(sem_pred[sem_pred >= 10]))
        scene_name = nusc.get("scene", sample["scene_token"])["name"]
        print(f"\nSample {vi} (scene: {scene_name}, token: {token[:12]}...) "
              f"V={V}, things={n_things}, stuff={n_stuff}", flush=True)

        # Generate colors
        panoptic_colors = generate_panoptic_colors(sem_pred, inst_pred)
        gt_panoptic_colors, gt_inst = generate_gt_panoptic_colors(
            gt_sem, gt_masks, gt_classes)
        semantic_colors = CLASS_COLORS[sem_pred]
        instance_colors = generate_instance_colors(inst_pred)

        sample_dir = os.path.join(args.save_dir, f"sample_{vi:03d}_{scene_name}")
        os.makedirs(sample_dir, exist_ok=True)

        # ── PLY files ──
        write_ply(os.path.join(sample_dir, "pointcloud_semantic.ply"), coords_xyz, semantic_colors)
        write_ply(os.path.join(sample_dir, "pointcloud_panoptic.ply"), coords_xyz, panoptic_colors)
        write_ply(os.path.join(sample_dir, "pointcloud_gt.ply"), coords_xyz, gt_panoptic_colors)
        write_ply(os.path.join(sample_dir, "pointcloud_instance.ply"), coords_xyz, instance_colors)
        print(f"  PLY files saved", flush=True)

        # ── BEV comparison ──
        fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=100)
        for ax, labels, title in [(axes[0], sem_pred, "Predicted"), (axes[1], gt_sem, "Ground Truth")]:
            bev_c = CLASS_COLORS[np.clip(labels, 0, NUM_CLASSES - 1)] / 255.0
            ax.scatter(coords_xyz[:, 0], coords_xyz[:, 1],
                       c=bev_c, s=0.3, alpha=0.6, rasterized=True)
            ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_title(title)
            ax.set_aspect("equal"); ax.set_xlim(-55, 55); ax.set_ylim(-55, 55)
            ax.grid(True, alpha=0.3)
            patches = [mpatches.Patch(color=CLASS_COLORS[c]/255.0, label=NUSCENES_CLASS_NAMES[c])
                       for c in np.unique(labels) if 0 <= c < NUM_CLASSES]
            ax.legend(handles=patches, loc="upper right", fontsize=5)
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, "bev_comparison.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  BEV saved", flush=True)

        # ── Per-camera overlay images (legend scoped to visible points) ──
        cam_dir = os.path.join(sample_dir, "cameras")
        os.makedirs(cam_dir, exist_ok=True)

        for cam_name in CAMERA_GRID_ORDER:
            try:
                uv, valid_idx, cam_path, (img_h, img_w) = project_points_to_camera(
                    coords_xyz, nusc, sample, cam_name, max_depth=args.max_depth
                )
            except Exception as e:
                print(f"  {cam_name}: projection failed — {e}", flush=True)
                continue
            if len(valid_idx) == 0:
                continue

            cam_short = cam_name.replace("CAM_", "").lower()

            # Subset labels/colors to only what's visible in this camera
            cam_sem_pred = sem_pred[valid_idx]
            cam_inst_pred = inst_pred[valid_idx]
            cam_pan_colors = panoptic_colors[valid_idx]
            cam_gt_sem = gt_sem[valid_idx]
            cam_gt_inst = gt_inst[valid_idx]
            cam_gt_colors = gt_panoptic_colors[valid_idx]

            # Panoptic prediction overlay + camera-scoped legend
            pred_img = draw_overlay_fast(
                cam_path, uv, cam_pan_colors,
                point_size=args.point_size, alpha=args.alpha)
            pred_legend = make_panoptic_legend(
                cam_sem_pred, cam_inst_pred, cam_pan_colors, "Panoptic Prediction")
            save_overlay_with_legend(
                pred_img, pred_legend,
                os.path.join(cam_dir, f"{cam_short}_panoptic.png"))

            # GT overlay + camera-scoped legend
            gt_img = draw_overlay_fast(
                cam_path, uv, cam_gt_colors,
                point_size=args.point_size, alpha=args.alpha)
            gt_legend_img = make_panoptic_legend(
                cam_gt_sem, cam_gt_inst, cam_gt_colors, "Ground Truth")
            save_overlay_with_legend(
                gt_img, gt_legend_img,
                os.path.join(cam_dir, f"{cam_short}_gt.png"))

            print(f"  {cam_name}: {len(valid_idx)} pts → saved", flush=True)

        print(f"  Sample {vi} done.", flush=True)

        summary_lines.append(
            f"sample_{vi:03d}_{scene_name}: token={token[:16]}  V={V}  "
            f"things={n_things}  stuff={n_stuff}  "
            f"pred_classes={sorted(np.unique(sem_pred).tolist())}"
        )

        # Free memory before loading next sample (critical on 5GB RAM)
        del panoptic_colors, gt_panoptic_colors, gt_inst, semantic_colors, instance_colors
        del coords_xyz, gt_sem, gt_masks, gt_classes, feats, coords, outputs
        del pred_logits, pred_masks, sem_pred, inst_pred
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    with open(os.path.join(args.save_dir, "summary.txt"), "w") as f:
        f.write("Panoptic Visualization Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset: {dcfg['version']} {args.split}\n")
        f.write(f"Samples visualized: {len(selected)} from {n_scenes} scenes\n\n")
        for line in summary_lines:
            f.write(line + "\n")

    print(f"\nAll visualizations saved to: {args.save_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
