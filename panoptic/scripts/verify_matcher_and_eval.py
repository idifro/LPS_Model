"""Verify matcher produces non-empty matches and evaluation runs to completion.

Master plan acceptance gates:
  - matcher produces non-empty matches in most frames
  - evaluation script runs to completion

Usage (on GPU node):
    python -u panoptic/scripts/verify_matcher_and_eval.py
"""
import sys
import numpy as np
import torch
import MinkowskiEngine as ME

sys.path.insert(0, "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene")

from models.disnet import DisNet, state_dict_remove_moudle
from panoptic.models.mask_decoder import PanopticDecoder
from panoptic.losses.matcher import HungarianMatcher
from panoptic.eval.eval_pq import PanopticEval
from panoptic.dataset.nuscenes_panoptic import (
    NuScenesPanoptic, NUM_CLASSES, THING_IDS, STUFF_IDS,
)


def load_backbone(ckpt_path, device):
    """Load the pretrained DisNet backbone with all parameters frozen for inference.

    Used to provide 768-d CLIP-aligned voxel features to the panoptic decoder
    during matcher and evaluation verification.

    Steps:
        1. Constructs a DisNet with MinkUNet18A architecture and openseg config.
        2. Loads checkpoint weights with 'module.' prefix stripping (strict mode).
        3. Sets model to eval mode and disables gradient computation on all params.

    Args:
        ckpt_path (str): Path to the pretrained OpenScene checkpoint (.pth.tar).
        device (torch.device): Target device (cuda or cpu).

    Returns:
        DisNet: The frozen backbone model ready for feature extraction.
    """
    class Cfg:
        feature_2d_extractor = "openseg"
        arch_3d = "MinkUNet18A"
    model = DisNet(Cfg()).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    from models.disnet import state_dict_remove_moudle
    state = state_dict_remove_moudle(ckpt["state_dict"])
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def extract_features(backbone, batch, device):
    """Extract 768-d backbone features from a single-sample batch dictionary.

    Prepares voxel coordinates and features from the dataset batch format into a
    MinkowskiEngine SparseTensor and runs the frozen backbone forward pass.

    Steps:
        1. Reads vox_coords and vox_feats from the batch dictionary.
        2. Prepends a zero batch-index column to coordinates for ME SparseTensor.
        3. Runs the backbone in no_grad mode and extracts dense features.
        4. Returns features and coordinates both unsqueezed to (1, V, D) for decoder input.

    Args:
        backbone (DisNet): Frozen backbone model.
        batch (dict): Single sample dict with 'vox_coords' (V, 3) and 'vox_feats' (V, 3).
        device (torch.device): Target device for tensor placement.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (feats, coords) each shaped (1, V, D) and (1, V, 3).
    """
    vox_coords = batch["vox_coords"]
    vox_feats = batch["vox_feats"]
    V = vox_coords.shape[0]
    batch_coords = torch.cat([torch.zeros(V, 1, dtype=torch.int32), vox_coords], dim=1)
    sinput = ME.SparseTensor(vox_feats.float(), coordinates=batch_coords.int(), device=device)
    with torch.no_grad():
        output = backbone(sinput)
    feats = output.F if hasattr(output, "F") else output
    return feats.unsqueeze(0), vox_feats.to(device).unsqueeze(0)


def panoptic_inference(pred_logits, pred_masks, num_classes=16):
    """Convert raw decoder predictions into panoptic segmentation (semantic + instance maps).

    Implements greedy mask-first panoptic fusion: iterates over predicted masks sorted by
    confidence, assigns each to a semantic class and instance ID while respecting overlap rules.

    Steps:
        1. Computes per-query class scores via softmax and filters out no-object predictions.
        2. Sorts remaining queries by descending confidence score.
        3. For each query, binarizes its mask at threshold 0.5 and checks overlap with
           already-assigned voxels (requires >= 80% of mask to be unoccupied).
        4. Assigns thing classes unique instance IDs; merges stuff classes into shared IDs.

    Args:
        pred_logits (torch.Tensor): Query classification logits, shape (Q, num_classes+1).
        pred_masks (torch.Tensor): Query mask logits, shape (V, Q).
        num_classes (int): Number of semantic classes (default 16).

    Returns:
        tuple[np.ndarray, np.ndarray]: (sem_pred, inst_pred) both of shape (V,) with
            semantic class labels and instance IDs respectively.
    """
    scores, labels = pred_logits.softmax(-1).max(-1)
    masks = pred_masks.sigmoid()
    V = masks.shape[0]
    keep = labels != num_classes
    if keep.sum() == 0:
        return np.zeros(V, dtype=np.int64), np.zeros(V, dtype=np.int64)
    scores, labels, masks = scores[keep], labels[keep], masks[:, keep]
    sorted_idx = scores.argsort(descending=True)
    sem_pred = np.zeros(V, dtype=np.int64)
    inst_pred = np.zeros(V, dtype=np.int64)
    inst_id = 1
    thing_set = set(range(10))
    stuff_assigned = {}
    for idx in sorted_idx:
        label = labels[idx].item()
        binary = masks[:, idx] >= 0.5
        if binary.sum() == 0:
            continue
        assigned = torch.from_numpy(inst_pred).to(binary.device) > 0
        final_mask = binary & ~assigned
        if final_mask.sum().item() / max(binary.sum().item(), 1) < 0.8:
            continue
        mask_np = final_mask.cpu().numpy().astype(bool)
        if label in thing_set:
            sem_pred[mask_np] = label
            inst_pred[mask_np] = inst_id
            inst_id += 1
        else:
            if label in stuff_assigned:
                inst_pred[mask_np] = stuff_assigned[label]
            else:
                stuff_assigned[label] = inst_id
                inst_pred[mask_np] = inst_id
                inst_id += 1
            sem_pred[mask_np] = label
    return sem_pred, inst_pred


def main():
    """Run matcher and evaluation acceptance tests for the panoptic pipeline.

    Serves as the integration verification script confirming that the Hungarian matcher
    produces non-empty matches and that the PanopticEval evaluator runs to completion.

    Steps:
        1. Loads the val split of NuScenesPanoptic and initializes backbone + decoder.
        2. TEST 1: Runs the matcher on 20 frames, checking that >= 80% have non-empty matches.
        3. TEST 2: Runs full panoptic inference and feeds results into PanopticEval,
           reporting PQ, SQ, RQ, and mIoU metrics per-class.
        4. Prints a summary of pass/fail status for both tests.

    Returns:
        None: Prints test results and metrics to stdout.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    dataroot = "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes_mini"
    ds = NuScenesPanoptic(dataroot, "v1.0-mini", split="val", augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x[0])

    # Load models
    backbone = load_backbone("models/nuscenes_openseg.pth.tar", device)
    decoder = PanopticDecoder(
        backbone_dim=768, hidden_dim=256, num_classes=NUM_CLASSES,
        num_thing_queries=100, num_stuff_classes=len(STUFF_IDS),
        nheads=8, dim_feedforward=1024, num_blocks=3, num_levels=3,
    ).to(device)

    # Load trained weights if available
    import os
    ckpt_path = "panoptic/checkpoints/best.pth"
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        decoder.load_state_dict(ckpt["decoder"])
        print(f"[Loaded] {ckpt_path} (epoch {ckpt['epoch']}, PQ={ckpt.get('best_pq', '?')})")
    else:
        print("[WARN] No checkpoint found, using random weights")

    decoder.eval()
    matcher = HungarianMatcher(cost_class=2.0, cost_mask=5.0, cost_dice=5.0, p_ratio=0.75)

    # ═══════════════════════════════════════════════════════════════
    # TEST 1: Matcher produces non-empty matches
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("TEST 1: Matcher non-empty matches")
    print(f"{'='*60}")

    total_frames = 0
    non_empty_matches = 0
    match_counts = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            feats, coords = extract_features(backbone, batch, device)
            outputs = decoder(feats, coords)

            targets = {
                "classes": [batch["gt_classes"].to(device)],
                "masks": [batch["gt_masks"].float().to(device)],
            }

            indices = matcher(
                {"pred_logits": outputs["pred_logits"], "pred_masks": outputs["pred_masks"]},
                targets,
            )

            n_matched = len(indices[0][0])
            n_gt = len(batch["gt_classes"])
            total_frames += 1

            if n_matched > 0:
                non_empty_matches += 1

            match_counts.append(n_matched)
            print(f"  Frame {i}: GT={n_gt} masks, Matched={n_matched}/{min(106, n_gt)}")

            if i >= 19:  # Check 20 frames
                break

    pct = 100 * non_empty_matches / total_frames
    print(f"\nResult: {non_empty_matches}/{total_frames} frames with non-empty matches ({pct:.0f}%)")
    print(f"Match counts: min={min(match_counts)}, max={max(match_counts)}, "
          f"mean={np.mean(match_counts):.1f}")

    if pct >= 80:
        print("PASS: Matcher produces non-empty matches in most frames")
    else:
        print("FAIL: Too many frames with empty matches")

    # ═══════════════════════════════════════════════════════════════
    # TEST 2: Evaluation script runs to completion
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("TEST 2: Evaluation runs to completion")
    print(f"{'='*60}")

    evaluator = PanopticEval(n_classes=NUM_CLASSES, min_points=50)
    eval_frames = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            feats, coords = extract_features(backbone, batch, device)
            outputs = decoder(feats, coords)

            pred_logits = outputs["pred_logits"][0].cpu()
            pred_masks = outputs["pred_masks"][0].cpu()
            sem_pred, inst_pred = panoptic_inference(pred_logits, pred_masks)

            gt_sem = batch["gt_sem"].numpy()
            gt_masks_np = batch["gt_masks"].numpy()
            gt_classes_np = batch["gt_classes"].numpy()
            V = gt_sem.shape[0]
            gt_inst = np.zeros(V, dtype=np.int64)
            for m_idx in range(len(gt_classes_np)):
                gt_inst[gt_masks_np[m_idx]] = m_idx + 1

            valid = gt_sem != 255
            if valid.sum() == 0:
                continue

            evaluator.addBatch(sem_pred[valid], inst_pred[valid], gt_sem[valid], gt_inst[valid])
            eval_frames += 1

    PQ, SQ, RQ, pq_all, sq_all, rq_all = evaluator.getPQ()
    iou_mean, iou_all = evaluator.getSemIoU()

    print(f"\nEvaluated {eval_frames} frames")
    print(f"PQ={PQ:.4f}  SQ={SQ:.4f}  RQ={RQ:.4f}  mIoU={iou_mean:.4f}")
    print(f"\nPer-class PQ:")
    from panoptic.dataset.nuscenes_panoptic import NUSCENES_CLASS_NAMES
    for c in range(NUM_CLASSES):
        print(f"  {NUSCENES_CLASS_NAMES[c]:25s}  PQ={pq_all[c]:.4f}  SQ={sq_all[c]:.4f}  RQ={rq_all[c]:.4f}  IoU={iou_all[c]:.4f}")

    print(f"\nPASS: Evaluation script ran to completion")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Matcher non-empty matches: {pct:.0f}% of frames ({'PASS' if pct >= 80 else 'FAIL'})")
    print(f"  Evaluation completion:     PASS")
    print(f"  PQ={PQ:.4f}  SQ={SQ:.4f}  RQ={RQ:.4f}  mIoU={iou_mean:.4f}")


if __name__ == "__main__":
    main()
