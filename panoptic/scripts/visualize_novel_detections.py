"""Visualize novel class detections by projecting predicted voxels onto camera images.

Runs the geometric ensemble (Strategy E) on selected samples, identifies voxels
predicted as novel classes, and overlays them on all 6 camera images with
distinct colors. This enables visual verification of whether novel-class
detections are genuine or false positives.

Also produces bird's eye view (BEV) plots showing novel predictions in spatial context.

Usage:
    # Visualize specific samples by index
    python -u panoptic/scripts/visualize_novel_detections.py \
        --config panoptic/configs/full_nuscenes_token_v2.yaml \
        --checkpoint panoptic/checkpoints_token_full_v2/best.pth \
        --version v1.0-trainval \
        --novel-classes "police car,ambulance,trash cans,pole,bench" \
        --num_samples 20 \
        --alpha 0.3

    # Visualize only samples that have novel detections
    python -u panoptic/scripts/visualize_novel_detections.py \
        --config panoptic/configs/full_nuscenes_token_v2.yaml \
        --checkpoint panoptic/checkpoints_token_full_v2/best.pth \
        --version v1.0-trainval \
        --only_with_detections \
        --max_samples 50
"""
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene")

from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from panoptic.dataset.nuscenes_panoptic import (
    NuScenesPanoptic, NUM_CLASSES, NUSCENES_CLASS_NAMES, CAMERAS,
)
from panoptic.scripts.visualize_lidar_camera_overlay import (
    project_points_to_camera, draw_overlay, CLASS_COLORS,
)
from panoptic.scripts.evaluate_comprehensive import (
    load_config, load_backbone, extract_backbone_features,
    load_decoder_flexible, load_or_generate_22_prototypes,
    generate_multi_prompt_prototypes, compute_prototype_bias,
    panoptic_inference_geometric_ensemble_v2,
    panoptic_inference_linear, DEFAULT_NOVEL_CLASS_NAMES, THING_CLASSES_16,
)

# Novel class colors (bright, distinct from 16-class palette)
NOVEL_COLORS = np.array([
    [255,   0, 128],   # police car - hot pink
    [  0, 255, 128],   # ambulance - spring green
    [128,   0, 255],   # trash cans - purple
    [255, 165,   0],   # pole - orange
    [  0, 191, 255],   # bench - deep sky blue
    [255, 255,   0],   # extra slot 1 - yellow
    [255, 105, 180],   # extra slot 2 - hot pink 2
    [  0, 250, 154],   # extra slot 3 - medium spring green
], dtype=np.uint8)

KNOWN_COLOR_DIM = np.array([80, 80, 80], dtype=np.uint8)  # dim gray for known class points


def parse_args():
    p = argparse.ArgumentParser(description="Visualize novel class detections on camera images")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--version", default=None)
    p.add_argument("--dataroot", default=None)
    p.add_argument("--novel-classes", default=None)
    p.add_argument("--alpha", type=float, default=0.3, help="Geometric ensemble alpha")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--mode", default="geo_ensemble_v2",
                   choices=["geo_ensemble_v2"],
                   help="Inference mode (geo_ensemble_v2 with multi-prompt + bias correction)")
    p.add_argument("--num_samples", type=int, default=20, help="Number of samples to process")
    p.add_argument("--start_idx", type=int, default=0, help="Starting sample index")
    p.add_argument("--only_with_detections", action="store_true",
                   help="Only save visualization for samples with novel detections")
    p.add_argument("--max_samples", type=int, default=100,
                   help="Max samples to scan when --only_with_detections")
    p.add_argument("--save_dir", default="panoptic/viz_novel_detections")
    p.add_argument("--point_size", type=int, default=4)
    p.add_argument("--overlay_alpha", type=float, default=0.8)
    p.add_argument("--max_depth", type=float, default=50.0)
    p.add_argument("--show_known", action="store_true",
                   help="Also overlay known-class predictions (dimmed)")
    return p.parse_args()


def run_inference(pred_logits, pred_masks, backbone_feats, text_protos_mp,
                  proto_bias, num_known, overlap_thr, alpha, temperature, min_mask_pts):
    """Run geo_ensemble_v2 inference (multi-prompt + bias correction)."""
    return panoptic_inference_geometric_ensemble_v2(
        pred_logits, pred_masks, backbone_feats, text_protos_mp, proto_bias,
        num_known=num_known, overlap_thr=overlap_thr,
        alpha=alpha, temperature=temperature, min_mask_pts=min_mask_pts,
    )


def draw_novel_overlay(cam_path, uv, sem_labels, novel_class_names, num_known,
                       point_size=4, alpha=0.8, show_known=False):
    """Draw novel-class colored points on camera image.

    Novel classes get bright distinct colors. Known classes get dim gray (if shown).
    Background/unassigned points are not drawn.
    """
    img = Image.open(cam_path).convert("RGB")
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)

    num_novel = len(novel_class_names)

    for (u, v), label in zip(uv.astype(int), sem_labels):
        if label >= num_known and label < num_known + num_novel:
            # Novel class - bright color
            color_idx = label - num_known
            color = tuple(NOVEL_COLORS[color_idx].tolist())
            r = point_size + 1  # slightly larger for novel
            draw.ellipse([u - r, v - r, u + r, v + r], fill=color)
        elif show_known and label > 0:
            # Known class - dim
            color = tuple(CLASS_COLORS[label].tolist())
            r = point_size - 1
            draw.ellipse([u - r, v - r, u + r, v + r], fill=color)

    result = Image.blend(img, overlay, alpha)
    return result


def draw_bev_novel(coords_xyz, sem_pred, novel_class_names, num_known, save_path,
                   xlim=(-51.2, 51.2), ylim=(-51.2, 51.2)):
    """Bird's eye view of novel detections."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_facecolor("black")

    # Draw all points as dim gray
    ax.scatter(coords_xyz[:, 0], coords_xyz[:, 1], c="gray", s=0.1, alpha=0.1)

    # Highlight novel detections
    num_novel = len(novel_class_names)
    legend_patches = []
    for i, name in enumerate(novel_class_names):
        cls_id = num_known + i
        mask = sem_pred == cls_id
        if mask.sum() > 0:
            color = NOVEL_COLORS[i] / 255.0
            ax.scatter(coords_xyz[mask, 0], coords_xyz[mask, 1],
                       c=[color], s=3, alpha=0.9, label=name)
            legend_patches.append(mpatches.Patch(
                color=color, label=f"{name} ({mask.sum()} vox)"))

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("BEV - Novel Class Detections")
    ax.set_aspect("equal")
    if legend_patches:
        ax.legend(handles=legend_patches, loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def draw_legend(novel_class_names, save_path):
    """Draw a legend for novel class colors."""
    fig, ax = plt.subplots(figsize=(4, 0.5 * len(novel_class_names) + 0.5))
    ax.axis("off")
    patches = []
    for i, name in enumerate(novel_class_names):
        color = NOVEL_COLORS[i] / 255.0
        patches.append(mpatches.Patch(color=color, label=name))
    ax.legend(handles=patches, loc="center", fontsize=12, frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve novel classes
    if args.novel_classes:
        novel_class_names = [n.strip() for n in args.novel_classes.split(",")]
    else:
        novel_class_names = cfg.get("novel_classes", DEFAULT_NOVEL_CLASS_NAMES)

    num_known = cfg["model"]["num_classes"]
    overlap_thr = cfg["inference"]["overlap_thr"]
    min_points = cfg["inference"].get("min_points", 50)

    version = args.version or cfg["data"]["version"]
    if args.dataroot:
        cfg["data"]["dataroot"] = args.dataroot
    dataroot = cfg["data"]["dataroot"]

    # Output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.save_dir) / f"{args.mode}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Novel Class Visualization")
    print(f"  Mode:         {args.mode}")
    print(f"  Alpha:        {args.alpha}")
    print(f"  Novel:        {novel_class_names}")
    print(f"  Output:       {out_dir}")
    print()

    # Load prototypes (CLIP on GPU before backbone)
    token_cfg = cfg.get("token", {})
    proto_path = token_cfg.get(
        "prototype_path", "panoptic/clip_prototypes/text_prototypes_vitl14_336.pt"
    )
    _, text_protos_mp = generate_multi_prompt_prototypes(
        proto_path, device, novel_class_names
    )
    if text_protos_mp is None:
        print("ERROR: Cannot generate multi-prompt prototypes.")
        return
    print(f"Multi-prompt prototypes: {text_protos_mp.shape}", flush=True)

    # Clear CLIP from GPU before loading panoptic models
    torch.cuda.empty_cache()

    # Load models
    backbone = load_backbone(cfg["model"]["backbone_ckpt"], device)
    decoder, has_token_head = load_decoder_flexible(cfg, args.checkpoint, device)

    # Load dataset and nuScenes DB
    from nuscenes.nuscenes import NuScenes as NuSc
    dcfg = cfg["data"]
    print("Loading NuScenes database...", flush=True)
    nusc = NuSc(version=version, dataroot=dataroot, verbose=False)
    val_ds = NuScenesPanoptic(
        dataroot, version, split="val",
        voxel_size=dcfg["voxel_size"],
        point_range=np.array(dcfg["point_range"]),
        augment=False, aug_cfg=None, nusc=nusc,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=lambda x: x[0],
    )
    print(f"Val: {len(val_ds)} samples\n", flush=True)

    # Compute prototype bias (calibration)
    print("Computing prototype bias for visualization...", flush=True)
    proto_bias = compute_prototype_bias(backbone, val_loader, text_protos_mp, device)

    # Save legend
    draw_legend(novel_class_names, str(out_dir / "legend.png"))

    # Evaluation
    samples_with_novel = 0
    total_novel_voxels = {name: 0 for name in novel_class_names}
    summary_lines = []

    sample_indices = range(args.start_idx, min(args.start_idx + args.max_samples, len(val_ds)))

    with torch.no_grad():
        saved_count = 0
        for i in sample_indices:
            if saved_count >= args.num_samples:
                break

            batch = val_ds[i]
            feats, coords = extract_backbone_features(backbone, batch, device)
            outputs = decoder(feats, coords)

            pred_logits = outputs["pred_logits"][0]
            pred_masks = outputs["pred_masks"][0]
            backbone_feats_v = feats[0]

            # Run inference
            sem_pred, inst_pred = run_inference(
                pred_logits, pred_masks, backbone_feats_v,
                text_protos_mp, proto_bias, num_known, overlap_thr,
                args.alpha, args.temperature, min_points,
            )

            # Also run linear for comparison
            sem_linear, _ = panoptic_inference_linear(
                pred_logits.cpu(), pred_masks.cpu(), num_known, overlap_thr
            )

            # Count novel detections
            novel_counts = {}
            total_novel_this_sample = 0
            for j, name in enumerate(novel_class_names):
                cls_id = num_known + j
                n_vox = int((sem_pred == cls_id).sum())
                novel_counts[name] = n_vox
                total_novel_voxels[name] += n_vox
                total_novel_this_sample += n_vox

            if args.only_with_detections and total_novel_this_sample == 0:
                if (i + 1) % 100 == 0:
                    print(f"  Scanned {i+1} samples, {samples_with_novel} with novel detections", flush=True)
                continue

            if total_novel_this_sample > 0:
                samples_with_novel += 1

            # Create sample directory
            sample_dir = out_dir / f"sample_{i:04d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            # Get coordinates for projection and BEV
            coords_xyz = batch["vox_feats"].numpy()  # (V, 3) in LIDAR frame
            sample_token = batch["token"]
            sample_rec = nusc.get("sample", sample_token)

            # Save BEV plot
            draw_bev_novel(coords_xyz, sem_pred, novel_class_names, num_known,
                           str(sample_dir / "bev_novel.png"))

            # Project onto cameras
            for cam_name in CAMERAS:
                uv, valid_idx, cam_path, (img_h, img_w) = project_points_to_camera(
                    coords_xyz, nusc, sample_rec, cam_name, max_depth=args.max_depth
                )

                if len(valid_idx) == 0:
                    continue

                # Get semantic labels for projected points
                proj_sem = sem_pred[valid_idx]

                # Check if any novel points in this view
                novel_in_view = sum(
                    (proj_sem == num_known + j).sum()
                    for j in range(len(novel_class_names))
                )

                # Draw overlay
                overlay_img = draw_novel_overlay(
                    cam_path, uv, proj_sem, novel_class_names, num_known,
                    point_size=args.point_size, alpha=args.overlay_alpha,
                    show_known=args.show_known,
                )

                # Add text annotation
                draw = ImageDraw.Draw(overlay_img)
                text = f"Sample {i} | {cam_name} | Novel: {novel_in_view} vox"
                draw.text((10, 10), text, fill=(255, 255, 0))

                # Also annotate which novel classes are present
                y_offset = 30
                for j, name in enumerate(novel_class_names):
                    cls_id = num_known + j
                    n_in_view = int((proj_sem == cls_id).sum())
                    if n_in_view > 0:
                        color = tuple(NOVEL_COLORS[j].tolist())
                        draw.text((10, y_offset), f"  {name}: {n_in_view}", fill=color)
                        y_offset += 18

                overlay_img.save(str(sample_dir / f"{cam_name}.png"))

            # Also save a "linear only" comparison for reference (CAM_FRONT)
            uv_front, valid_idx_front, cam_path_front, _ = project_points_to_camera(
                coords_xyz, nusc, sample_rec, "CAM_FRONT", max_depth=args.max_depth
            )
            if len(valid_idx_front) > 0:
                proj_sem_linear = sem_linear[valid_idx_front]
                colors_linear = np.array([CLASS_COLORS[l] if 0 <= l < 16 else [0, 0, 0]
                                          for l in proj_sem_linear], dtype=np.uint8)
                linear_img = draw_overlay(cam_path_front, uv_front, colors_linear,
                                          point_size=args.point_size - 1, alpha=0.6)
                linear_img.save(str(sample_dir / "CAM_FRONT_linear_comparison.png"))

            # Summary info
            novel_str = ", ".join(f"{n}:{novel_counts[n]}" for n in novel_class_names if novel_counts[n] > 0)
            if not novel_str:
                novel_str = "(no novel)"
            summary_line = f"Sample {i:4d}: {novel_str}"
            summary_lines.append(summary_line)
            print(f"  [{saved_count+1}] {summary_line}", flush=True)

            saved_count += 1

    # Write summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Samples with novel detections: {samples_with_novel}")
    print(f"  Total novel voxels per class:")
    for name, count in total_novel_voxels.items():
        print(f"    {name:<20s}: {count:>10,d}")

    with open(out_dir / "summary.txt", "w") as f:
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Alpha: {args.alpha}\n")
        f.write(f"Temperature: {args.temperature}\n")
        f.write(f"Novel classes: {novel_class_names}\n")
        f.write(f"Samples with novel detections: {samples_with_novel}\n\n")
        f.write("Per-class totals:\n")
        for name, count in total_novel_voxels.items():
            f.write(f"  {name}: {count}\n")
        f.write(f"\nPer-sample details:\n")
        for line in summary_lines:
            f.write(f"  {line}\n")

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
