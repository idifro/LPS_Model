"""Phase 3.3: Semantic proxy check — zero-shot classification using CLIP text vs DisNet features.

WITHOUT any panoptic decoder, use raw backbone features to compute a rough semantic proxy:
  - Encode nuScenes class names with CLIP text encoder (ViT-L/14@336px → 768-d)
  - Compute dot product (semi-cosine) between voxel features and text prototypes
  - Supports both 16-class and 43-detail modes (matching OpenScene evaluate.py)
  - Save per-sample visualizations as numpy arrays

Usage (GPU node):
    python -u panoptic/scripts/verify_phase3_semantic_proxy.py
    python -u panoptic/scripts/verify_phase3_semantic_proxy.py --mode details --num_samples 10
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import MinkowskiEngine as ME

sys.path.insert(0, "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene")

from models.disnet import DisNet, state_dict_remove_moudle
from panoptic.dataset.nuscenes_panoptic import (
    NuScenesPanoptic, NUM_CLASSES, IGNORE_LABEL,
)

# ── Use EXACT label strings from OpenScene (dataset/label_constants.py) ──
# These must match what the backbone was distilled with.
NUSCENES_LABELS_16 = (
    'barrier', 'bicycle', 'bus', 'car', 'construction vehicle',
    'motorcycle', 'person', 'traffic cone',
    'trailer', 'truck', 'drivable surface', 'other flat',
    'sidewalk', 'terrain', 'manmade', 'vegetation',
)

# 43 fine-grained labels — more discriminative for CLIP
NUSCENES_LABELS_DETAILS = (
    'barrier', 'barricade', 'bicycle', 'bus', 'car',
    'bulldozer', 'excavator', 'concrete mixer', 'crane', 'dump truck',
    'motorcycle', 'person', 'pedestrian', 'traffic cone',
    'trailer', 'semi trailer', 'cargo container', 'shipping container', 'freight container',
    'truck', 'road', 'curb', 'traffic island', 'traffic median',
    'sidewalk', 'grass', 'grassland', 'lawn', 'meadow', 'turf', 'sod',
    'building', 'wall', 'pole', 'awning',
    'tree', 'trunk', 'tree trunk', 'bush', 'shrub', 'plant', 'flower', 'woods',
)

# Maps each of the 43 detail labels → 16-class index (0-indexed)
MAPPING_NUSCENES_DETAILS = (
    0, 0, 1, 2, 3, 4, 4, 4, 4, 4,
    5, 6, 6, 7, 8, 8, 8, 8, 8,
    9, 10, 11, 11, 11, 12, 13, 13, 13, 13, 13, 13,
    14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15,
)

# For display (matches our dataset convention)
DISPLAY_NAMES = [
    "barrier", "bicycle", "bus", "car", "construction_vehicle",
    "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck",
    "driveable_surface", "other_flat", "sidewalk", "terrain", "manmade", "vegetation",
]

# ── Try importing CLIP ──
try:
    import clip
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
    print("[WARN] clip not installed. Install: pip install git+https://github.com/openai/CLIP.git")


def parse_args():
    """Parse command-line arguments for Phase 3.3 semantic proxy verification.

    Configures the data paths, CLIP model variant, label mode (16 coarse vs 43 detailed),
    number of evaluation samples, and output directory for visualization artifacts.

    Steps:
        1. Creates an ArgumentParser with defaults for nuscenes_mini and ViT-L/14@336px.
        2. Adds arguments for dataroot, version, ckpt, num_samples, split, save_dir,
           clip_model, mode, and prompt_eng flag.
        3. Parses and returns the namespace object.

    Returns:
        argparse.Namespace: Parsed arguments including mode ('16' or 'details'),
            clip_model (str), and prompt_eng (bool).
    """
    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", default="/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes_mini")
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--ckpt", default="models/nuscenes_openseg.pth.tar")
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--split", default="val")
    p.add_argument("--save_dir", default="panoptic/vis_phase3")
    p.add_argument("--clip_model", default="ViT-L/14@336px",
                   help="CLIP model name (ViT-L/14@336px for 768-d to match OpenSeg)")
    p.add_argument("--mode", default="details", choices=["16", "details"],
                   help="'16' = 16 coarse labels, 'details' = 43 fine-grained → mapped to 16 (recommended)")
    p.add_argument("--prompt_eng", action="store_true", default=True,
                   help="Use OpenScene prompt template: 'a {label} in a scene'")
    return p.parse_args()


def encode_text_prototypes(class_names, clip_model_name, device, prompt_eng=True):
    """Encode class names to CLIP text embeddings.

    Matches OpenScene's extract_text_feature() exactly:
    - If prompt_eng: wraps as "a {label} in a scene"
    - L2-normalize text features (point features stay raw)
    - Uses single prompt per class (matching OpenScene, not multi-prompt averaging)

    Returns: (C, D) L2-normalized text prototypes.
    """
    label_list = list(class_names)
    if prompt_eng:
        label_list = ["a " + label + " in a scene" for label in label_list]
        print(f"\n  Prompt engineering: 'a {{label}} in a scene'")
        print(f"  Example: '{label_list[0]}'")

    print(f"  Encoding {len(label_list)} labels with CLIP {clip_model_name}...")
    clip_model, _ = clip.load(clip_model_name, device=device)
    clip_model.eval()

    tokens = clip.tokenize(label_list).to(device)
    with torch.no_grad():
        text_feats = clip_model.encode_text(tokens)  # (C, D)
        # L2 normalize — matching OpenScene util.py line 47
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    print(f"  Text prototype shape: {text_feats.shape}")
    return text_feats.float()


def load_backbone(ckpt_path, device):
    """Load the pretrained DisNet backbone with all parameters frozen for inference.

    Instantiates and configures a DisNet (MinkUNet18A + openseg) model, loads
    pretrained weights, and disables all gradient computation for downstream use.

    Steps:
        1. Constructs DisNet with openseg/MinkUNet18A config and moves to device.
        2. Loads checkpoint, strips 'module.' prefix, and applies strict state dict loading.
        3. Sets model to eval mode and freezes all parameters.

    Args:
        ckpt_path (str): Path to the pretrained OpenScene checkpoint file.
        device (torch.device): Target device (cuda or cpu).

    Returns:
        DisNet: Frozen backbone model ready for feature extraction.
    """
    class Cfg:
        feature_2d_extractor = "openseg"
        arch_3d = "MinkUNet18A"
    model = DisNet(Cfg()).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = state_dict_remove_moudle(ckpt["state_dict"])
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def extract_features(model, batch, device):
    """Extract 768-d backbone features from a single dataset sample.

    Converts the batch dictionary's voxel coordinates and features into a
    MinkowskiEngine SparseTensor and runs the frozen backbone forward pass.

    Steps:
        1. Reads vox_coords (V, 3) and vox_feats (V, 3) from the batch dict.
        2. Prepends a zero batch-index column to form ME-compatible coordinates.
        3. Constructs a SparseTensor on the target device and runs forward pass.
        4. Extracts dense feature tensor from the sparse output.

    Args:
        model (DisNet): Frozen backbone model.
        batch (dict): Single sample dict with 'vox_coords' and 'vox_feats' tensors.
        device (torch.device): Target device for computation.

    Returns:
        torch.Tensor: Voxel features of shape (V, 768).
    """
    vox_coords = batch["vox_coords"]
    vox_feats = batch["vox_feats"]
    V = vox_coords.shape[0]
    batch_coords = torch.cat([
        torch.zeros(V, 1, dtype=torch.int32), vox_coords
    ], dim=1)
    sinput = ME.SparseTensor(
        vox_feats.float(), coordinates=batch_coords.int(), device=device
    )
    with torch.no_grad():
        output = model(sinput)
    feats = output.F if hasattr(output, "F") else output
    return feats  # (V, 768)


def main():
    """Run the Phase 3.3 semantic proxy check: zero-shot classification via CLIP text alignment.

    Validates that the frozen DisNet backbone produces features with meaningful semantic
    structure by computing dot-product similarity against CLIP text prototypes and
    measuring per-class accuracy and mIoU without any learned decoder.

    Steps:
        1. Parses args, loads dataset and frozen backbone, encodes CLIP text prototypes.
        2. For each sample, extracts backbone features and computes semi-cosine similarity
           against text prototypes (L2-normalized text, raw point features).
        3. In 'details' mode, argmaxes over 43 fine-grained labels then maps to 16 classes.
        4. Accumulates per-class accuracy, confusion matrix, and mIoU across samples.
        5. Saves per-sample visualization arrays (coords, predictions, similarity, confusion)
           and a global results JSON to save_dir.

    Returns:
        None: Prints metrics to stdout and saves artifacts to disk.
    """
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: {args.mode} ({'43 detailed → 16 mapped' if args.mode == 'details' else '16 coarse'})")

    if not HAS_CLIP:
        print("ABORT: clip package required. pip install git+https://github.com/openai/CLIP.git")
        sys.exit(1)

    os.makedirs(args.save_dir, exist_ok=True)

    # Load dataset
    dataset = NuScenesPanoptic(
        args.dataroot, args.version, split=args.split, augment=False
    )

    # Load backbone
    backbone = load_backbone(args.ckpt, device)

    # ── Encode text prototypes (matching OpenScene evaluate.py exactly) ──
    if args.mode == "details":
        labelset = NUSCENES_LABELS_DETAILS  # 43 labels
        mapper = np.array(MAPPING_NUSCENES_DETAILS, dtype=np.int64)  # 43 → 16
        print(f"\n  Using 43 detailed labels → mapped to 16 classes")
    else:
        labelset = NUSCENES_LABELS_16  # 16 labels
        mapper = None
        print(f"\n  Using 16 coarse labels directly")

    text_prototypes = encode_text_prototypes(
        labelset, args.clip_model, device, prompt_eng=args.prompt_eng
    )
    # details: (43, 768), 16-mode: (16, 768)

    # Save prototypes
    proto_path = os.path.join(args.save_dir, "text_prototypes.npy")
    np.save(proto_path, text_prototypes.cpu().numpy())
    print(f"  Saved: {proto_path}")

    # ── Process samples ──
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=lambda x: x[0],
    )

    print(f"\n{'='*60}")
    print(f"Semantic Proxy Check: {args.num_samples} samples, mode={args.mode}")
    print(f"{'='*60}")

    total_correct = 0
    total_valid = 0
    per_class_correct = np.zeros(NUM_CLASSES, dtype=np.int64)
    per_class_total = np.zeros(NUM_CLASSES, dtype=np.int64)
    # mIoU confusion matrix
    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for i, batch in enumerate(loader):
        if i >= args.num_samples:
            break

        token = batch["token"]
        gt_sem = batch["gt_sem"].numpy()  # (V,) 0-indexed, 255=ignore
        vox_feats_xyz = batch["vox_feats"].numpy()  # (V, 3)

        # Extract backbone features
        feats = extract_features(backbone, batch, device)  # (V, 768)

        # Semi-cosine: text features are L2-normalized, point features are RAW
        # This matches OpenScene evaluate.py line 497-500:
        #   pred = predictions.half() @ text_features.t()
        sim = (feats.half() @ text_prototypes.half().t()).float()  # (V, C_labels)

        if mapper is not None:
            # 43-detail mode: argmax over 43, then map to 16
            fine_pred = sim.argmax(dim=-1).cpu().numpy()  # (V,) range 0-42
            pred_classes = mapper[fine_pred]  # (V,) range 0-15
        else:
            pred_classes = sim.argmax(dim=-1).cpu().numpy()  # (V,) range 0-15

        # Compute accuracy on valid voxels
        valid = gt_sem != IGNORE_LABEL
        n_valid = valid.sum()
        if n_valid > 0:
            correct = (pred_classes[valid] == gt_sem[valid]).sum()
            acc = correct / n_valid
            total_correct += correct
            total_valid += n_valid

            for c in range(NUM_CLASSES):
                mask_c = gt_sem[valid] == c
                per_class_correct[c] += (pred_classes[valid][mask_c] == c).sum()
                per_class_total[c] += mask_c.sum()

            # Accumulate confusion matrix
            for gt_c, pred_c in zip(gt_sem[valid], pred_classes[valid]):
                conf_matrix[int(gt_c), int(pred_c)] += 1
        else:
            acc = 0.0

        # Top-K class distribution
        pred_counts = np.bincount(pred_classes, minlength=NUM_CLASSES)
        gt_counts = np.bincount(gt_sem[valid].astype(np.int64), minlength=NUM_CLASSES) if n_valid > 0 else np.zeros(NUM_CLASSES)

        print(f"\n  Sample {i} (token: {token[:12]}...)")
        print(f"    Voxels: {len(gt_sem)}, Valid: {n_valid}, Accuracy: {acc:.4f}")
        print(f"    Top-5 predicted classes:")
        top5_pred = pred_counts.argsort()[::-1][:5]
        for rank, cls_idx in enumerate(top5_pred):
            print(f"      {rank+1}. {DISPLAY_NAMES[cls_idx]:25s} pred={pred_counts[cls_idx]:>5} gt={int(gt_counts[cls_idx]):>5}")

        # Similarity statistics
        sim_np = sim.cpu().numpy()
        print(f"    Sim stats: min={sim_np.min():.1f}, max={sim_np.max():.1f}, mean={sim_np.mean():.1f}")

        # Confidence (max sim per voxel)
        max_sim = sim_np.max(axis=1)
        print(f"    Confidence: min={max_sim.min():.1f}, max={max_sim.max():.1f}, "
              f"mean={max_sim.mean():.1f}")

        # ── Save visualization data ──
        sample_dir = os.path.join(args.save_dir, f"sample_{i:03d}")
        os.makedirs(sample_dir, exist_ok=True)

        np.save(os.path.join(sample_dir, "coords_xyz.npy"), vox_feats_xyz)
        np.save(os.path.join(sample_dir, "gt_sem.npy"), gt_sem)
        np.save(os.path.join(sample_dir, "pred_sem.npy"), pred_classes)
        np.save(os.path.join(sample_dir, "similarity.npy"), sim_np)
        np.save(os.path.join(sample_dir, "confidence.npy"), max_sim)

        # Save per-sample confusion
        if n_valid > 0:
            conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
            for gt_c, pred_c in zip(gt_sem[valid], pred_classes[valid]):
                conf[int(gt_c), int(pred_c)] += 1
            np.save(os.path.join(sample_dir, "confusion.npy"), conf)

        print(f"    Saved: {sample_dir}/")

    # ── Global summary ──
    print(f"\n{'='*60}")
    print(f"GLOBAL SEMANTIC PROXY SUMMARY (mode={args.mode})")
    print(f"{'='*60}")

    global_acc = total_correct / max(total_valid, 1)
    print(f"\n  Overall accuracy: {total_correct}/{total_valid} = {global_acc:.4f}")

    # Compute mIoU from confusion matrix
    tp = conf_matrix.diagonal().astype(np.float64)
    fp = conf_matrix.sum(axis=0).astype(np.float64) - tp
    fn = conf_matrix.sum(axis=1).astype(np.float64) - tp
    union = tp + fp + fn
    iou = np.where(union > 0, tp / union, 0.0)
    miou = np.mean(iou[union > 0]) if (union > 0).any() else 0.0

    print(f"  mIoU: {miou:.4f}")

    print(f"\n  Per-class accuracy + IoU:")
    print(f"    {'Class':25s} {'Correct':>8} {'Total':>8} {'Acc':>8} {'IoU':>8}")
    print(f"    {'-'*62}")
    accs_valid = []
    for c in range(NUM_CLASSES):
        if per_class_total[c] > 0:
            c_acc = per_class_correct[c] / per_class_total[c]
            accs_valid.append(c_acc)
        else:
            c_acc = float('nan')
        c_iou = iou[c]
        print(f"    {DISPLAY_NAMES[c]:25s} {per_class_correct[c]:>8} {per_class_total[c]:>8} "
              f"{'N/A' if per_class_total[c] == 0 else f'{c_acc:.4f}':>8} {c_iou:.4f}")

    mean_class_acc = np.nanmean(accs_valid) if accs_valid else 0.0
    print(f"\n  Mean class accuracy: {mean_class_acc:.4f}")
    print(f"  mIoU: {miou:.4f}")

    # Interpretation
    print(f"\n  Interpretation:")
    if miou > 0.25:
        print(f"    STRONG: Features have clear semantic structure in CLIP space.")
        print(f"    (OpenScene reports ~35% mIoU with distill on full nuScenes)")
    elif miou > 0.10:
        print(f"    MODERATE: Features show semantic alignment. Expected range for per-keyframe.")
    elif miou > 0.0625:
        print(f"    WEAK: Above random ({1/NUM_CLASSES:.4f}).")
    else:
        print(f"    POOR: Near random chance.")

    print(f"\n  Visualizations saved to: {args.save_dir}/")

    # Save global results
    import json
    results = {
        "mode": args.mode,
        "prompt_eng": args.prompt_eng,
        "global_accuracy": float(global_acc),
        "mean_class_accuracy": float(mean_class_acc),
        "mIoU": float(miou),
        "per_class_accuracy": {
            DISPLAY_NAMES[c]: float(per_class_correct[c] / max(per_class_total[c], 1))
            for c in range(NUM_CLASSES)
        },
        "per_class_iou": {
            DISPLAY_NAMES[c]: float(iou[c]) for c in range(NUM_CLASSES)
        },
        "per_class_total": {DISPLAY_NAMES[c]: int(per_class_total[c]) for c in range(NUM_CLASSES)},
        "num_samples": args.num_samples,
        "clip_model": args.clip_model,
    }
    with open(os.path.join(args.save_dir, "semantic_proxy_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results JSON: {args.save_dir}/semantic_proxy_results.json")

    # Save confusion matrix
    np.save(os.path.join(args.save_dir, "global_confusion.npy"), conf_matrix)

    print(f"\n  PASS: 3.3 — Semantic proxy check complete.")


if __name__ == "__main__":
    main()
