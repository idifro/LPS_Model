"""Comprehensive panoptic evaluation with per-class breakdown.

Loads a trained checkpoint, runs inference on the val set, and reports
PQ/SQ/RQ/mIoU per class with thing/stuff/overall means.

Usage:
    python -u panoptic/scripts/evaluate_full.py \
        --checkpoint panoptic/checkpoints_full/best.pth \
        --version v1.0-mini \
        --dataroot /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes_mini
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene")

from panoptic.dataset.nuscenes_panoptic import (
    NuScenesPanoptic, NUSCENES_CLASS_NAMES, NUM_CLASSES,
)
from panoptic.eval.eval_pq import PanopticEval
from panoptic.models.mask_decoder import PanopticDecoder
from panoptic.train.train_panoptic import (
    load_backbone, extract_backbone_features, panoptic_inference, load_config,
)


def parse_args():
    p = argparse.ArgumentParser(description="Full panoptic evaluation")
    p.add_argument("--config", default="panoptic/configs/full_nuscenes.yaml")
    p.add_argument("--checkpoint", default="panoptic/checkpoints_full/best.pth")
    p.add_argument("--dataroot", default=None, help="Override config dataroot")
    p.add_argument("--version", default="v1.0-mini", help="Override config version")
    p.add_argument("--split", default="val")
    p.add_argument("--save_dir", default="panoptic/eval_results")
    return p.parse_args()


def load_decoder(cfg, checkpoint_path, device):
    """Load decoder from checkpoint."""
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

    epoch = ckpt.get("epoch", "?")
    best_pq = ckpt.get("best_pq", "?")
    print(f"[Decoder] Loaded from {checkpoint_path} (epoch={epoch}, best_pq={best_pq})")
    return decoder


def format_table(pq_all, sq_all, rq_all, iou_all):
    """Format per-class evaluation results as a table string."""
    lines = []
    sep = "-" * 65
    header = f"{'Class':<22s} | {'PQ':>6s} | {'SQ':>6s} | {'RQ':>6s} | {'IoU':>6s}"
    lines.append(header)
    lines.append(sep)

    for i in range(NUM_CLASSES):
        name = NUSCENES_CLASS_NAMES[i]
        tag = " (T)" if i < 10 else " (S)"
        lines.append(
            f"{name + tag:<22s} | {pq_all[i]:6.4f} | {sq_all[i]:6.4f} | "
            f"{rq_all[i]:6.4f} | {iou_all[i]:6.4f}"
        )

    lines.append(sep)
    thing_idx = list(range(10))
    stuff_idx = list(range(10, 16))
    all_idx = list(range(16))

    for label, idx_list in [("MEAN (all 16)", all_idx), ("MEAN (thing 0-9)", thing_idx), ("MEAN (stuff 10-15)", stuff_idx)]:
        lines.append(
            f"{label:<22s} | {np.mean(pq_all[idx_list]):6.4f} | {np.mean(sq_all[idx_list]):6.4f} | "
            f"{np.mean(rq_all[idx_list]):6.4f} | {np.mean(iou_all[idx_list]):6.4f}"
        )

    return "\n".join(lines)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.dataroot:
        cfg["data"]["dataroot"] = args.dataroot
    if args.version:
        cfg["data"]["version"] = args.version

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Load models
    backbone = load_backbone(cfg["model"]["backbone_ckpt"], device)
    decoder = load_decoder(cfg, args.checkpoint, device)

    # Load dataset
    dcfg = cfg["data"]
    dataset = NuScenesPanoptic(
        dcfg["dataroot"], dcfg["version"], split=args.split,
        voxel_size=dcfg["voxel_size"],
        point_range=np.array(dcfg["point_range"]),
        augment=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=lambda x: x[0],
    )
    print(f"Evaluating {len(dataset)} {args.split} samples...")

    # Evaluate
    num_classes = cfg["model"]["num_classes"]
    overlap_thr = cfg["inference"]["overlap_thr"]
    min_points = cfg["inference"]["min_points"]
    evaluator = PanopticEval(n_classes=num_classes, min_points=min_points)

    t_start = time.time()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            feats, coords = extract_backbone_features(backbone, batch, device)
            outputs = decoder(feats, coords)
            pred_logits = outputs["pred_logits"][0].cpu()
            pred_masks = outputs["pred_masks"][0].cpu()
            sem_pred, inst_pred = panoptic_inference(pred_logits, pred_masks, num_classes, overlap_thr)

            # Build GT instance map
            gt_sem = batch["gt_sem"].numpy()
            gt_masks = batch["gt_masks"].numpy()
            gt_classes = batch["gt_classes"].numpy()
            V = gt_sem.shape[0]
            gt_inst = np.zeros(V, dtype=np.int64)
            for m_idx in range(len(gt_classes)):
                gt_inst[gt_masks[m_idx]] = m_idx + 1

            valid = gt_sem != 255
            if valid.sum() == 0:
                continue
            evaluator.addBatch(
                sem_pred[valid], inst_pred[valid],
                gt_sem[valid], gt_inst[valid],
            )

            if (i + 1) % 100 == 0 or i == 0:
                print(f"  [{i+1}/{len(dataset)}] ...")

    elapsed = time.time() - t_start
    print(f"Inference done in {elapsed:.1f}s ({elapsed/len(dataset):.2f}s/sample)")

    # Get results
    PQ, SQ, RQ, pq_all, sq_all, rq_all = evaluator.getPQ()
    iou_mean, iou_all = evaluator.getSemIoU()

    # Format and print table
    table = format_table(pq_all, sq_all, rq_all, iou_all)
    print(f"\n{table}\n")

    # Save text
    txt_path = os.path.join(args.save_dir, "eval_results.txt")
    with open(txt_path, "w") as f:
        f.write(f"Panoptic Evaluation Results\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset: {dcfg['version']} {args.split} ({len(dataset)} samples)\n")
        f.write(f"Time: {elapsed:.1f}s\n\n")
        f.write(table + "\n")
    print(f"Saved: {txt_path}")

    # Save JSON
    results = {
        "checkpoint": args.checkpoint,
        "dataset": dcfg["version"],
        "split": args.split,
        "num_samples": len(dataset),
        "PQ": float(PQ), "SQ": float(SQ), "RQ": float(RQ), "mIoU": float(iou_mean),
        "per_class": {},
    }
    for i in range(NUM_CLASSES):
        results["per_class"][NUSCENES_CLASS_NAMES[i]] = {
            "PQ": float(pq_all[i]), "SQ": float(sq_all[i]),
            "RQ": float(rq_all[i]), "IoU": float(iou_all[i]),
        }
    json_path = os.path.join(args.save_dir, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {json_path}")

    # Save confusion matrix
    conf_path = os.path.join(args.save_dir, "confusion_matrix.npy")
    np.save(conf_path, evaluator.px_iou_conf_matrix)
    print(f"Saved: {conf_path}")


if __name__ == "__main__":
    main()
