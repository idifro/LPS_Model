"""Visualisation helpers: camera overlays, BEV scatter plots, legends.

Camera projection code is adapted from
panoptic/scripts/visualize_lidar_camera_overlay.py.

Public API:
  build_text_query_colors(matched_queries, V)       → (V, 3) uint8
  build_class_colors(sem_pred)                      → (V, 3) uint8
  get_raw_camera_images(cam_paths)                  → dict[cam_name → PIL.Image]
  make_camera_overlays(raw_xyz, colors, nusc, ...)  → dict[cam_name → PIL.Image]
  make_bev_figure(raw_xyz, colors, title)           → matplotlib Figure
  make_text_query_legend_fig(...)                   → matplotlib Figure
  make_class_legend_fig(sem_pred)                   → matplotlib Figure
  make_query_score_fig(...)                         → matplotlib Figure
"""

import io
import os

import numpy as np
from PIL import Image, ImageDraw

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Camera order (matches data_utils.CAMERAS_ORDERED) ────────────────────────
CAMERAS_ORDERED = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]

CAM_LABELS = {
    "CAM_FRONT_LEFT":  "Front Left",
    "CAM_FRONT":       "Front",
    "CAM_FRONT_RIGHT": "Front Right",
    "CAM_BACK_LEFT":   "Back Left",
    "CAM_BACK":        "Back",
    "CAM_BACK_RIGHT":  "Back Right",
}

# ── 16-class colour palette (0-indexed, mirrors visualize_lidar_camera_overlay) ──
CLASS_COLORS = np.array([
    [255, 120,  50],   # 0  barrier
    [255, 192, 203],   # 1  bicycle
    [255, 255,   0],   # 2  bus
    [  0, 150, 245],   # 3  car
    [  0, 255, 255],   # 4  construction_vehicle
    [200, 180,   0],   # 5  motorcycle
    [255,   0,   0],   # 6  pedestrian
    [255,   0, 255],   # 7  traffic_cone
    [100, 150, 200],   # 8  trailer
    [  0, 255,   0],   # 9  truck
    [128, 128, 128],   # 10 driveable_surface
    [235, 200, 150],   # 11 other_flat
    [ 75,   0,  75],   # 12 sidewalk
    [150, 240,  80],   # 13 terrain
    [230, 230, 250],   # 14 manmade
    [ 34, 139,  34],   # 15 vegetation
], dtype=np.uint8)

IGNORE_COLOR    = np.array([40,  40,  40],  dtype=np.uint8)   # background / ignored
UNMATCHED_COLOR = np.array([55,  55,  55],  dtype=np.uint8)   # voxels not matched by text query

# ── Rank-based colours for text-query matched instances ───────────────────────
# Rank 0 = best cosine match = brightest red; fades through the spectrum.
RANK_COLORS = np.array([
    [255,  51,  51],   # rank 0 – red
    [255, 136,   0],   # rank 1 – orange
    [255, 221,   0],   # rank 2 – yellow
    [128, 220,   0],   # rank 3 – yellow-green
    [  0, 204,   0],   # rank 4 – green
    [  0, 204, 153],   # rank 5 – green-teal
    [  0, 180, 255],   # rank 6 – light blue
    [  0, 120, 255],   # rank 7 – blue
    [100,   0, 255],   # rank 8 – purple
    [200,   0, 200],   # rank 9 – magenta
], dtype=np.uint8)

NUSCENES_CLASS_NAMES = [
    "barrier", "bicycle", "bus", "car", "construction_vehicle",
    "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck",
    "driveable_surface", "other_flat", "sidewalk", "terrain", "manmade", "vegetation",
]

THING_IDS = list(range(0, 10))   # 0-indexed
STUFF_IDS = list(range(10, 16))  # 0-indexed


# ══════════════════════════════════════════════════════════════════════════════
# Camera projection  (adapted from visualize_lidar_camera_overlay.py)
# ══════════════════════════════════════════════════════════════════════════════

def _project_points_to_camera(points_3d, nusc, sample, cam_name, max_depth=50.0):
    """Project 3D LIDAR-frame points onto one camera image.

    Full transform chain:
      LIDAR frame → ego @ lidar_time → global → ego @ cam_time → camera

    Args:
        points_3d: (N, 3) float32 XYZ in LIDAR frame
        nusc:      NuScenes instance
        sample:    nuScenes sample dict
        cam_name:  e.g. "CAM_FRONT"
        max_depth: discard points beyond this depth (metres)

    Returns:
        uv:         (K, 2) float pixel coordinates of valid projected points
        valid_idx:  (K,)   indices into points_3d
        cam_path:   str    absolute path to camera image
        (img_h, img_w): image dimensions
    """
    from pyquaternion import Quaternion

    cam_sd   = nusc.get("sample_data", sample["data"][cam_name])
    cam_path = os.path.join(nusc.dataroot, cam_sd["filename"])
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])

    # Ego pose at LiDAR time
    ego_lidar  = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    R_ego_lidar = Quaternion(ego_lidar["rotation"]).rotation_matrix
    t_ego_lidar = np.array(ego_lidar["translation"])

    # LiDAR → ego calibration
    lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    R_lidar  = Quaternion(lidar_cs["rotation"]).rotation_matrix
    t_lidar  = np.array(lidar_cs["translation"])

    # Ego pose at camera time
    ego_cam  = nusc.get("ego_pose", cam_sd["ego_pose_token"])
    R_ego_cam = Quaternion(ego_cam["rotation"]).rotation_matrix
    t_ego_cam = np.array(ego_cam["translation"])

    # Camera → ego calibration
    cam_cs    = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    R_cam     = Quaternion(cam_cs["rotation"]).rotation_matrix
    t_cam     = np.array(cam_cs["translation"])
    intrinsic = np.array(cam_cs["camera_intrinsic"])  # (3, 3)

    # ── Transform: lidar → ego_lidar → global → ego_cam → camera ──
    pts_ego     = (R_lidar     @ points_3d.T).T + t_lidar     # lidar → ego@lidar
    pts_global  = (R_ego_lidar @ pts_ego.T).T    + t_ego_lidar # ego@lidar → global
    pts_ego_cam = (R_ego_cam.T @ (pts_global - t_ego_cam).T).T # global → ego@cam
    pts_cam     = (R_cam.T     @ (pts_ego_cam - t_cam).T).T    # ego@cam → camera

    depth = pts_cam[:, 2]
    valid = (depth > 0.5) & (depth <= max_depth)

    if valid.sum() == 0:
        return np.zeros((0, 2)), np.array([], dtype=np.int64), cam_path, \
               (cam_sd["height"], cam_sd["width"])

    pts_img = (intrinsic @ pts_cam[valid].T).T       # (K, 3)
    uv      = pts_img[:, :2] / pts_img[:, 2:3]       # (K, 2)

    img_h, img_w = cam_sd["height"], cam_sd["width"]
    in_bounds = (
        (uv[:, 0] >= 0) & (uv[:, 0] < img_w) &
        (uv[:, 1] >= 0) & (uv[:, 1] < img_h)
    )

    valid_indices = np.where(valid)[0][in_bounds]
    uv_valid      = uv[in_bounds]
    return uv_valid, valid_indices, cam_path, (img_h, img_w)


def _draw_overlay(cam_path, uv, colors, point_size=4, alpha=0.75):
    """Draw coloured ellipses on a camera image and blend with the original.

    Args:
        cam_path:   path to camera JPEG
        uv:         (K, 2) float pixel coordinates
        colors:     (K, 3) uint8 RGB colours
        point_size: radius of each point in pixels
        alpha:      blending factor (0=original, 1=full overlay)

    Returns:
        PIL Image (RGB)
    """
    img     = Image.open(cam_path).convert("RGB")
    overlay = img.copy()
    draw    = ImageDraw.Draw(overlay)

    for (u, v), color in zip(uv.astype(int), colors):
        r = point_size
        draw.ellipse(
            [u - r, v - r, u + r, v + r],
            fill=tuple(int(c) for c in color),
        )

    return Image.blend(img, overlay, alpha)


# ══════════════════════════════════════════════════════════════════════════════
# Colour builders
# ══════════════════════════════════════════════════════════════════════════════

def build_text_query_colors(matched_queries, V):
    """Build per-voxel (V, 3) uint8 colour array for text-query overlay.

    Matched queries are coloured by rank (rank 0 = best cosine match = red).
    All other voxels are dark gray (UNMATCHED_COLOR).

    If multiple matched queries overlap on the same voxel, the lower rank
    (better match) wins because we iterate in rank order.

    Args:
        matched_queries: list of (query_idx, cos_sim, binary_mask) sorted desc by score
        V:               number of voxels

    Returns:
        (V, 3) uint8 numpy array
    """
    colors = np.full((V, 3), UNMATCHED_COLOR, dtype=np.uint8)
    for rank, (qidx, score, binary_mask) in enumerate(matched_queries):
        color = RANK_COLORS[min(rank, len(RANK_COLORS) - 1)]
        colors[binary_mask] = color
    return colors


def build_class_colors(sem_pred):
    """Build per-voxel (V, 3) uint8 colour array for all-16-classes overlay.

    Args:
        sem_pred: (V,) int64 np.ndarray — 0-indexed class labels

    Returns:
        (V, 3) uint8 numpy array
    """
    V      = sem_pred.shape[0]
    colors = np.full((V, 3), IGNORE_COLOR, dtype=np.uint8)
    valid  = (sem_pred >= 0) & (sem_pred < 16)
    colors[valid] = CLASS_COLORS[sem_pred[valid]]
    return colors


# ══════════════════════════════════════════════════════════════════════════════
# Camera image rendering
# ══════════════════════════════════════════════════════════════════════════════

def get_raw_camera_images(cam_paths):
    """Load raw (no overlay) camera images from disk.

    Args:
        cam_paths: list of 6 absolute paths in CAMERAS_ORDERED order

    Returns:
        dict: cam_name → PIL Image (RGB)
    """
    result = {}
    for cam_name, path in zip(CAMERAS_ORDERED, cam_paths):
        try:
            result[cam_name] = Image.open(path).convert("RGB")
        except Exception:
            result[cam_name] = Image.new("RGB", (1600, 900), (30, 30, 30))
    return result


def make_camera_overlays(
    raw_xyz, per_voxel_colors, nusc, sample_record,
    max_depth=50.0, point_size=4, alpha=0.75,
):
    """Project voxel points onto all 6 cameras and draw coloured overlays.

    Args:
        raw_xyz:          (V, 3) float32 np.ndarray — XYZ in LIDAR frame (= vox_feats)
        per_voxel_colors: (V, 3) uint8  np.ndarray — RGB colour per voxel
        nusc:             NuScenes instance
        sample_record:    nuScenes sample dict (from data_utils)
        max_depth:        projection depth cutoff (metres)
        point_size:       point radius in pixels
        alpha:            overlay blending factor

    Returns:
        dict: cam_name → PIL Image (RGB)
    """
    result = {}
    for cam_name in CAMERAS_ORDERED:
        try:
            uv, valid_idx, cam_path, _ = _project_points_to_camera(
                raw_xyz, nusc, sample_record, cam_name, max_depth=max_depth
            )
        except Exception:
            result[cam_name] = Image.new("RGB", (1600, 900), (30, 30, 30))
            continue

        if len(valid_idx) == 0:
            try:
                result[cam_name] = Image.open(cam_path).convert("RGB")
            except Exception:
                result[cam_name] = Image.new("RGB", (1600, 900), (30, 30, 30))
            continue

        point_colors = per_voxel_colors[valid_idx]
        result[cam_name] = _draw_overlay(
            cam_path, uv, point_colors, point_size=point_size, alpha=alpha
        )

    return result


# ══════════════════════════════════════════════════════════════════════════════
# BEV scatter plot
# ══════════════════════════════════════════════════════════════════════════════

def make_bev_figure(raw_xyz, per_voxel_colors, title="Bird's Eye View"):
    """Create a top-down BEV scatter plot.

    X axis = forward (LIDAR +X), Y axis = left (+Y).
    Ego vehicle marked with a white star at (0, 0).

    Args:
        raw_xyz:          (V, 3) float32 np.ndarray — XYZ in LIDAR frame
        per_voxel_colors: (V, 3) uint8  np.ndarray
        title:            plot title string

    Returns:
        matplotlib Figure (caller responsible for closing with plt.close(fig))
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    colors_float = per_voxel_colors.astype(np.float32) / 255.0

    ax.scatter(
        raw_xyz[:, 0], raw_xyz[:, 1],
        c=colors_float, s=0.6, alpha=0.8, linewidths=0,
    )

    ax.plot(0, 0, "w*", markersize=14, zorder=5, label="Ego")
    ax.set_xlabel("X (m) — forward", color="white", fontsize=9)
    ax.set_ylabel("Y (m) — left",    color="white", fontsize=9)
    ax.set_title(title, color="white", fontsize=10, fontweight="bold")
    ax.set_xlim(-55, 55)
    ax.set_ylim(-55, 55)
    ax.set_aspect("equal")
    ax.set_facecolor("#111111")
    ax.tick_params(colors="white", labelsize=7)
    ax.grid(True, alpha=0.18, color="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.legend(fontsize=8, labelcolor="white", framealpha=0.2)

    fig.patch.set_facecolor("#111111")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Legend figures
# ══════════════════════════════════════════════════════════════════════════════

def make_text_query_legend_fig(matched_queries, pred_tokens_cpu, text_protos_cpu):
    """Legend for text-query mode: rank | colour | cosine score | nearest class.

    Args:
        matched_queries: list of (query_idx, score, binary_mask) sorted desc
        pred_tokens_cpu: (Q, 768) float32 CPU tensor (for nearest-class lookup)
        text_protos_cpu: (16, 768) float32 CPU tensor

    Returns:
        matplotlib Figure
    """
    import torch as _torch

    if not matched_queries:
        fig, ax = plt.subplots(figsize=(5, 1))
        ax.text(0.5, 0.5, "No queries matched", ha="center", va="center",
                color="white", fontsize=11)
        ax.axis("off")
        fig.patch.set_facecolor("#1e1e1e")
        return fig

    n   = len(matched_queries)
    fh  = max(1.5, n * 0.45 + 0.6)
    fig, ax = plt.subplots(figsize=(6.5, fh))
    ax.axis("off")
    fig.patch.set_facecolor("#1e1e1e")

    # Column headers
    for x_pos, header in [
        (0.04, "Rank"), (0.18, "Color"), (0.36, "Score"),
        (0.55, "Nearest class"), (0.82, "Voxels"),
    ]:
        ax.text(x_pos, 1.02, header, color="#aaaaaa", fontsize=8,
                va="bottom", transform=ax.transAxes)

    for rank, (qidx, score, binary_mask) in enumerate(matched_queries):
        y = 1.0 - (rank + 1) * (0.92 / (n + 0.5))

        color_u8  = RANK_COLORS[min(rank, len(RANK_COLORS) - 1)]
        color_f   = tuple(color_u8.astype(float) / 255.0)

        # Nearest prototype class via cosine similarity
        if pred_tokens_cpu is not None and text_protos_cpu is not None:
            sim       = pred_tokens_cpu[qidx] @ text_protos_cpu.T  # (16,)
            cls_idx   = sim.argmax().item()
            cls_name  = NUSCENES_CLASS_NAMES[cls_idx]
            cls_sim   = sim[cls_idx].item()
            cls_label = f"{cls_name} ({cls_sim:.2f})"
        else:
            cls_label = "—"

        n_vox = int(binary_mask.sum())

        ax.text(0.04, y, f"#{rank + 1}", color="white", fontsize=8,
                va="center", transform=ax.transAxes)
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.16, y - 0.025), 0.07, 0.05,
            boxstyle="round,pad=0.005",
            facecolor=color_f, edgecolor="white", linewidth=0.5,
            transform=ax.transAxes,
        ))
        ax.text(0.36, y, f"{score:.3f}", color="white", fontsize=8,
                va="center", transform=ax.transAxes)
        ax.text(0.55, y, cls_label, color="white", fontsize=8,
                va="center", transform=ax.transAxes)
        ax.text(0.82, y, str(n_vox), color="#aaaaaa", fontsize=8,
                va="center", transform=ax.transAxes)

    plt.tight_layout()
    return fig


def make_class_legend_fig(sem_pred):
    """Compact colour legend for all-16-classes mode.

    Only shows classes that are actually present in the current prediction.

    Args:
        sem_pred: (V,) int64 np.ndarray

    Returns:
        matplotlib Figure
    """
    present = sorted({int(c) for c in sem_pred if 0 <= c < 16})
    if not present:
        fig, ax = plt.subplots(figsize=(4, 1))
        ax.text(0.5, 0.5, "No classes detected", ha="center", va="center",
                color="white", fontsize=10)
        ax.axis("off")
        fig.patch.set_facecolor("#1e1e1e")
        return fig

    n_cols  = 2
    n_rows  = (len(present) + n_cols - 1) // n_cols
    fig_h   = max(1.5, n_rows * 0.35 + 0.4)
    fig, ax = plt.subplots(figsize=(6, fig_h))
    ax.axis("off")
    fig.patch.set_facecolor("#1e1e1e")

    patches = []
    for cls in present:
        color = CLASS_COLORS[cls].astype(float) / 255.0
        kind  = "thing" if cls in THING_IDS else "stuff"
        label = f"{NUSCENES_CLASS_NAMES[cls]}  [{kind}]"
        patches.append(mpatches.Patch(
            facecolor=color, edgecolor="white", linewidth=0.4, label=label,
        ))

    legend = ax.legend(
        handles=patches, loc="center",
        fontsize=8, frameon=False, ncol=n_cols,
    )
    for text in legend.get_texts():
        text.set_color("white")

    plt.tight_layout()
    return fig


def make_query_score_fig(
    active_q_indices, cos_sims, matched_queries,
    threshold, pred_tokens_cpu=None, text_protos_cpu=None,
):
    """Horizontal bar chart: cosine similarity for every active query.

    Bars above threshold are shown in the matched-rank colour; below in gray.
    A yellow dashed line marks the current threshold.

    Args:
        active_q_indices: list of int   — original query indices
        cos_sims:         list of float — cosine sim scores, parallel to active_q_indices
        matched_queries:  list of (query_idx, score, mask) — queries that passed threshold
        threshold:        float
        pred_tokens_cpu:  (Q, 768) optional, for class name labelling
        text_protos_cpu:  (16, 768) optional

    Returns:
        matplotlib Figure
    """
    if not active_q_indices:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No active queries", ha="center", va="center",
                color="white", fontsize=10)
        ax.axis("off")
        fig.patch.set_facecolor("#1e1e1e")
        return fig

    matched_q_set = {qidx for qidx, _, _ in matched_queries}

    # Build y-axis labels (nearest class if tokens available)
    labels      = []
    bar_colors  = []
    rank_lookup = {qidx: rank for rank, (qidx, _, _) in enumerate(matched_queries)}

    for i, q in enumerate(active_q_indices):
        if pred_tokens_cpu is not None and text_protos_cpu is not None:
            sim     = pred_tokens_cpu[q] @ text_protos_cpu.T
            cls_idx = sim.argmax().item()
            labels.append(f"Q{q} ({NUSCENES_CLASS_NAMES[cls_idx]})")
        else:
            labels.append(f"Q{q}")

        if q in matched_q_set:
            rank = rank_lookup[q]
            c    = RANK_COLORS[min(rank, len(RANK_COLORS) - 1)].astype(float) / 255.0
            bar_colors.append(tuple(c))
        else:
            bar_colors.append((0.35, 0.35, 0.35))

    n      = len(labels)
    fig_h  = max(4.0, n * 0.28)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")

    ax.barh(range(n), cos_sims, color=bar_colors, edgecolor="none", height=0.65)
    ax.axvline(x=threshold, color="#FFDD00", linestyle="--", linewidth=1.4,
               label=f"Threshold ({threshold:.2f})")

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=7, color="white")
    ax.set_xlabel("Cosine Similarity", color="white", fontsize=9)
    ax.set_title(
        "Active query scores  (coloured = matched, grey = below threshold)",
        color="white", fontsize=9,
    )
    ax.tick_params(colors="white", labelsize=7)
    ax.set_xlim(-0.15, 1.05)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.legend(fontsize=8, labelcolor="white", framealpha=0.15)

    plt.tight_layout()
    return fig
