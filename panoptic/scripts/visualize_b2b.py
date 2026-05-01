"""Phase 5a (B2b) visualization: class-agnostic panoptic overlay on camera images.

For each sample: per-camera panoptic overlays (pred + GT), PLY files, BEV, novel class highlights.
Uses backbone feature pooling for classification (22 classes: 16 known + 6 novel).

Usage:
    python -u panoptic/scripts/visualize_b2b.py \
        --checkpoint panoptic/checkpoints_b2b/best.pth \
        --version v1.0-trainval \
        --dataroot /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes \
        --num_samples 5
"""
import argparse
import colorsys
import gc
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene")

from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from nuscenes.nuscenes import NuScenes

from panoptic.dataset.nuscenes_panoptic import (
    NuScenesPanoptic, NUSCENES_CLASS_NAMES, NUM_CLASSES, CAMERAS, IGNORE_LABEL,
)
from panoptic.train.train_panoptic import (
    load_backbone, extract_backbone_features, load_config,
)
from panoptic.scripts.visualize_lidar_camera_overlay import (
    CLASS_COLORS, IGNORE_COLOR, project_points_to_camera,
)
from panoptic.scripts.visualize_panoptic import (
    load_decoder, generate_gt_panoptic_colors, generate_instance_colors,
    draw_overlay_fast, write_ply, make_panoptic_legend, save_overlay_with_legend,
    CAMERA_GRID_ORDER,
)
from panoptic.train.train_panoptic_b2b import (
    load_text_prototypes_22, panoptic_inference_class_agnostic,
    NOVEL_CLASS_NAMES, ALL_22_CLASS_NAMES,
)


# ── Extended color palette (22 classes) ─────────────────────────────────────
NOVEL_COLORS = np.array([
    [128,   0, 128],   # 16 pole         - purple
    [139,  69,  19],   # 17 bench        - brown
    [  0, 128, 128],   # 18 trash can    - teal
    [220,  20,  60],   # 19 fire hydrant - crimson
    [ 34, 139,  34],   # 20 tree trunk   - forest green
    [169, 169, 169],   # 21 curb         - dark gray
], dtype=np.uint8)

CLASS_COLORS_22 = np.concatenate([CLASS_COLORS, NOVEL_COLORS], axis=0)
CLASS_NAMES_22 = list(NUSCENES_CLASS_NAMES) + NOVEL_CLASS_NAMES


# ── Panoptic coloring (22 classes) ──────────────────────────────────────────

def generate_panoptic_colors_22(sem_pred, inst_pred, num_known=16):
    """Panoptic coloring with novel class support."""
    colors = CLASS_COLORS_22[np.clip(sem_pred, 0, len(CLASS_COLORS_22) - 1)].copy()
    # Thing instances get HSV-shifted colors (known things: 0-9, novel things: 16-21)
    thing_classes = list(range(10)) + list(range(num_known, len(CLASS_COLORS_22)))
    for cls in thing_classes:
        cls_mask = sem_pred == cls
        if not cls_mask.any():
            continue
        unique_insts = np.unique(inst_pred[cls_mask])
        base_rgb = CLASS_COLORS_22[cls] / 255.0
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


def make_panoptic_legend_22(sem, inst, colors, title="Panoptic"):
    """Build legend with 22-class support."""
    patches = []
    # Thing classes (known + novel)
    thing_classes = list(range(10)) + list(range(16, 22))
    for cls in thing_classes:
        cls_mask = sem == cls
        if not cls_mask.any():
            continue
        unique_insts = np.unique(inst[cls_mask])
        for iid in unique_insts:
            if iid == 0:
                continue
            inst_mask = cls_mask & (inst == iid)
            c = colors[np.where(inst_mask)[0][0]] / 255.0
            name = CLASS_NAMES_22[cls] if cls < len(CLASS_NAMES_22) else f"class_{cls}"
            suffix = " [NOVEL]" if cls >= 16 else ""
            patches.append(mpatches.Patch(
                color=c, label=f"{name}{suffix} #{iid} ({inst_mask.sum()}pt)"))
    # Stuff classes (10-15)
    for cls in range(10, 16):
        cls_mask = sem == cls
        if not cls_mask.any():
            continue
        c = colors[np.where(cls_mask)[0][0]] / 255.0
        patches.append(mpatches.Patch(
            color=c, label=f"{CLASS_NAMES_22[cls]} ({cls_mask.sum()}pt)"))

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


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 5a (B2b) visualization")
    p.add_argument("--config", default="panoptic/configs/full_nuscenes_b2b.yaml")
    p.add_argument("--checkpoint", default="panoptic/checkpoints_b2b/best.pth")
    p.add_argument("--dataroot", default=None)
    p.add_argument("--version", default="v1.0-trainval")
    p.add_argument("--split", default="val")
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--save_dir", default="panoptic/viz_b2b")
    p.add_argument("--max_depth", type=float, default=50.0)
    p.add_argument("--point_size", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--obj_thr", type=float, default=0.3)
    return p.parse_args()


def main():
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
        nusc=nusc,
    )

    # Load prototypes
    ov_cfg = cfg.get("open_vocab", {})
    proto_path = ov_cfg.get("prototypes_path", "panoptic/embeddings/text_prototypes_22.pt")
    text_protos_22 = load_text_prototypes_22(proto_path, device)

    overlap_thr = cfg["inference"]["overlap_thr"]
    summary_lines = []

    # Select samples across scenes
    from collections import defaultdict
    scene_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        token = dataset.tokens[idx]
        scene_token = nusc.get("sample", token)["scene_token"]
        scene_to_indices[scene_token].append(idx)
    scene_tokens = list(scene_to_indices.keys())

    rng = np.random.RandomState()
    rng.shuffle(scene_tokens)

    selected = []
    scene_cycle = 0
    while len(selected) < args.num_samples:
        st = scene_tokens[scene_cycle % len(scene_tokens)]
        idx = rng.choice(scene_to_indices[st])
        selected.append(idx)
        scene_cycle += 1

    print(f"Visualizing {len(selected)} samples ({dcfg['version']} {args.split})...", flush=True)

    for vi, dataset_idx in enumerate(selected):
        batch = dataset[dataset_idx]
        token = batch["token"]
        sample = nusc.get("sample", token)
        coords_xyz = batch["vox_feats"].numpy()
        gt_sem = batch["gt_sem"].numpy()
        gt_masks = batch["gt_masks"].numpy()
        gt_classes = batch["gt_classes"].numpy()
        V = coords_xyz.shape[0]

        # Inference (class-agnostic)
        with torch.no_grad():
            feats, coords = extract_backbone_features(backbone, batch, device)
            outputs = decoder(feats, coords)
        pred_masks = outputs["pred_masks"][0].cpu()
        backbone_feats = feats[0].cpu()

        sem_pred, inst_pred = panoptic_inference_class_agnostic(
            backbone_feats, pred_masks, text_protos_22.cpu(),
            overlap_thr=overlap_thr, obj_thr=args.obj_thr,
        )

        n_things = len(set(inst_pred[sem_pred < 10]) - {0})
        n_novel = (sem_pred >= 16).sum()
        scene_name = nusc.get("scene", sample["scene_token"])["name"]
        print(f"\nSample {vi} (scene: {scene_name}, token: {token[:12]}...) "
              f"V={V}, things={n_things}, novel_voxels={n_novel}", flush=True)

        # Colors
        panoptic_colors = generate_panoptic_colors_22(sem_pred, inst_pred)
        gt_panoptic_colors, gt_inst = generate_gt_panoptic_colors(
            gt_sem, gt_masks, gt_classes)
        semantic_colors = CLASS_COLORS_22[np.clip(sem_pred, 0, len(CLASS_COLORS_22) - 1)]
        instance_colors = generate_instance_colors(inst_pred)

        sample_dir = os.path.join(args.save_dir, f"sample_{vi:03d}_{scene_name}")
        os.makedirs(sample_dir, exist_ok=True)

        # PLY files
        write_ply(os.path.join(sample_dir, "pointcloud_semantic.ply"), coords_xyz, semantic_colors)
        write_ply(os.path.join(sample_dir, "pointcloud_panoptic.ply"), coords_xyz, panoptic_colors)
        write_ply(os.path.join(sample_dir, "pointcloud_gt.ply"), coords_xyz, gt_panoptic_colors)
        write_ply(os.path.join(sample_dir, "pointcloud_instance.ply"), coords_xyz, instance_colors)

        # Novel-only PLY (only novel class points colored, rest gray)
        novel_mask = sem_pred >= 16
        novel_colors = np.full((V, 3), 40, dtype=np.uint8)
        novel_colors[novel_mask] = CLASS_COLORS_22[sem_pred[novel_mask]]
        write_ply(os.path.join(sample_dir, "pointcloud_novel_only.ply"), coords_xyz, novel_colors)
        print(f"  PLY files saved", flush=True)

        # BEV comparison
        fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=100)
        for ax, labels, title in [(axes[0], sem_pred, "Predicted (B2b)"), (axes[1], gt_sem, "Ground Truth")]:
            if title.startswith("Predicted"):
                bev_c = CLASS_COLORS_22[np.clip(labels, 0, len(CLASS_COLORS_22) - 1)] / 255.0
            else:
                bev_c = CLASS_COLORS[np.clip(labels, 0, NUM_CLASSES - 1)] / 255.0
            ax.scatter(coords_xyz[:, 0], coords_xyz[:, 1],
                       c=bev_c, s=0.3, alpha=0.6, rasterized=True)
            ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_title(title)
            ax.set_aspect("equal"); ax.set_xlim(-55, 55); ax.set_ylim(-55, 55)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, "bev_comparison.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  BEV saved", flush=True)

        # Per-camera overlay images
        cam_dir = os.path.join(sample_dir, "cameras")
        os.makedirs(cam_dir, exist_ok=True)

        for cam_name in CAMERA_GRID_ORDER:
            try:
                uv, valid_idx, cam_path, (img_h, img_w) = project_points_to_camera(
                    coords_xyz, nusc, sample, cam_name, max_depth=args.max_depth
                )
            except Exception as e:
                print(f"  {cam_name}: projection failed - {e}", flush=True)
                continue
            if len(valid_idx) == 0:
                continue

            cam_short = cam_name.replace("CAM_", "").lower()

            cam_sem_pred = sem_pred[valid_idx]
            cam_inst_pred = inst_pred[valid_idx]
            cam_pan_colors = panoptic_colors[valid_idx]
            cam_gt_sem = gt_sem[valid_idx]
            cam_gt_inst = gt_inst[valid_idx]
            cam_gt_colors = gt_panoptic_colors[valid_idx]

            # Panoptic prediction overlay
            pred_img = draw_overlay_fast(
                cam_path, uv, cam_pan_colors,
                point_size=args.point_size, alpha=args.alpha)
            pred_legend = make_panoptic_legend_22(
                cam_sem_pred, cam_inst_pred, cam_pan_colors, "B2b Prediction")
            save_overlay_with_legend(
                pred_img, pred_legend,
                os.path.join(cam_dir, f"{cam_short}_panoptic.png"))

            # GT overlay
            gt_img = draw_overlay_fast(
                cam_path, uv, cam_gt_colors,
                point_size=args.point_size, alpha=args.alpha)
            gt_legend_img = make_panoptic_legend(
                cam_gt_sem, cam_gt_inst, cam_gt_colors, "Ground Truth")
            save_overlay_with_legend(
                gt_img, gt_legend_img,
                os.path.join(cam_dir, f"{cam_short}_gt.png"))

            # Novel-only overlay (only novel classes colored)
            cam_novel_mask = cam_sem_pred >= 16
            if cam_novel_mask.any():
                novel_only_colors = np.full_like(cam_pan_colors, 128)
                novel_only_colors[cam_novel_mask] = cam_pan_colors[cam_novel_mask]
                novel_img = draw_overlay_fast(
                    cam_path, uv, novel_only_colors,
                    point_size=args.point_size + 1, alpha=0.9)
                novel_img.save(os.path.join(cam_dir, f"{cam_short}_novel_only.png"))

            print(f"  {cam_name}: {len(valid_idx)} pts, novel={cam_novel_mask.sum()}", flush=True)

        # Novel class summary
        novel_summary = {}
        for j, name in enumerate(NOVEL_CLASS_NAMES):
            cnt = int((sem_pred == (16 + j)).sum())
            if cnt > 0:
                novel_summary[name] = cnt
        if novel_summary:
            print(f"  Novel classes: {novel_summary}", flush=True)

        summary_lines.append(
            f"sample_{vi:03d}_{scene_name}: token={token[:16]}  V={V}  "
            f"things={n_things}  novel_voxels={n_novel}  "
            f"pred_classes={sorted(np.unique(sem_pred).tolist())}"
        )

        del panoptic_colors, gt_panoptic_colors, gt_inst, semantic_colors, instance_colors
        del coords_xyz, gt_sem, gt_masks, gt_classes, feats, coords, outputs
        del pred_masks, backbone_feats, sem_pred, inst_pred
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    with open(os.path.join(args.save_dir, "summary.txt"), "w") as f:
        f.write("Phase 5a (B2b) Visualization Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset: {dcfg['version']} {args.split}\n")
        f.write(f"Objectness threshold: {args.obj_thr}\n")
        f.write(f"Samples visualized: {len(selected)}\n\n")
        for line in summary_lines:
            f.write(line + "\n")

    print(f"\nAll visualizations saved to: {args.save_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
