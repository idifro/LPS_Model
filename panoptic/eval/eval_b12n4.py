"""
B12/N4 Open-Vocabulary Evaluation Script
=========================================
Evaluates a panoptic segmentation checkpoint using the B12/N4 class split from:
  "3D Open-Vocabulary Panoptic Segmentation with 2D-3D Vision-Language Distillation"
  Xiao et al., ECCV 2024, arXiv:2401.02402

Split (nuScenes 16 classes):
  BASE  (12): barrier, car, construction_vehicle, person, traffic_cone, trailer,
              truck, driveable_surface, other_flat, sidewalk, terrain, manmade
  NOVEL things (3): bicycle, bus, motorcycle
  NOVEL stuff  (1): vegetation

Outputs (matching paper Table 1):
  PQ, SQ, RQ (all 16), PQ_B (base 12), PQ_N (novel 4),
  PQ_NTh (novel things), PQ_NSt (novel stuff), mIoU

Novel class detection mechanism:
  The backbone was pre-distilled from OpenSeg/CLIP → its output F_backbone is
  in CLIP space. L_V during training forces ALL query embeddings (including those
  over novel regions) to reconstruct F_backbone, keeping them in CLIP space.
  At inference, cos(query_embed, text["bicycle"]) is high for bicycle queries
  even though bicycle never appeared in training labels.

  Problem: base classes always win argmax because they have explicit training signal.
  Fix: base-class suppression threshold τ (--novel_threshold):
    if max_base_confidence < τ → predict the best novel class instead
    Tune τ on mini val after Gate 2. Default τ=0.4.

Usage:
  python panoptic/eval/eval_b12n4.py \\
      --checkpoint panoptic/checkpoints_openvocab_mini/best.pth \\
      --split b12n4_main \\
      --version v1.0-mini \\
      --dataroot /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes_mini \\
      --novel_threshold 0.4
"""

import sys
import os
import argparse
import json
import numpy as np
import torch

sys.path.insert(0, "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model")

from dataset.label_constants import (
    NUSCENES_LABELS_16,
    NUSCENES_B12N4_NOVEL_IDX,
    NUSCENES_B12N4_BASE_IDX,
    NUSCENES_B12N4_NOVEL_THINGS_IDX,
    NUSCENES_B12N4_NOVEL_STUFF_IDX,
    NUSCENES_B11N5_NOVEL_IDX,
    NUSCENES_B11N5_BASE_IDX,
    NUSCENES_B12N4_V2_NOVEL_IDX,
    NUSCENES_B12N4_V2_BASE_IDX,
)
from panoptic.eval.eval_pq import PanopticEval

# ─── Split registry ────────────────────────────────────────────────────────────
SPLITS = {
    "b12n4_main": {
        "name": "B12/N4 Main (Xiao et al. Table 1)",
        "novel_idx": NUSCENES_B12N4_NOVEL_IDX,
        "base_idx": NUSCENES_B12N4_BASE_IDX,
        "novel_things_idx": NUSCENES_B12N4_NOVEL_THINGS_IDX,
        "novel_stuff_idx": NUSCENES_B12N4_NOVEL_STUFF_IDX,
        "desc": "Novel things: bicycle, bus, motorcycle | Novel stuff: vegetation",
    },
    "b11n5": {
        "name": "B11/N5 (Xiao et al. Table 5)",
        "novel_idx": NUSCENES_B11N5_NOVEL_IDX,
        "base_idx": NUSCENES_B11N5_BASE_IDX,
        "novel_things_idx": [idx for idx in NUSCENES_B11N5_NOVEL_IDX if idx < 10],
        "novel_stuff_idx":  [idx for idx in NUSCENES_B11N5_NOVEL_IDX if idx >= 10],
        "desc": "Novel things: bicycle, car, construction_vehicle | Novel stuff: terrain, manmade",
    },
    "b12n4_v2": {
        "name": "B12/N4 Variant 2 (Xiao et al. Appendix Tab 7/8)",
        "novel_idx": NUSCENES_B12N4_V2_NOVEL_IDX,
        "base_idx": NUSCENES_B12N4_V2_BASE_IDX,
        "novel_things_idx": [idx for idx in NUSCENES_B12N4_V2_NOVEL_IDX if idx < 10],
        "novel_stuff_idx":  [idx for idx in NUSCENES_B12N4_V2_NOVEL_IDX if idx >= 10],
        "desc": "Novel things: construction_vehicle, traffic_cone | Novel stuff: other_flat, manmade",
    },
}


def compute_split_pq(pq_per_class, sq_per_class, rq_per_class, iou_per_class, indices):
    """Compute mean PQ/SQ/RQ/mIoU over a subset of class indices."""
    if not indices:
        return 0.0, 0.0, 0.0, 0.0
    pq  = float(np.mean([pq_per_class[i]  for i in indices]))
    sq  = float(np.mean([sq_per_class[i]  for i in indices]))
    rq  = float(np.mean([rq_per_class[i]  for i in indices]))
    iou = float(np.mean([iou_per_class[i] for i in indices]))
    return pq, sq, rq, iou


def print_split_results(label, indices, pq_c, sq_c, rq_c, iou_c):
    pq, sq, rq, iou = compute_split_pq(pq_c, sq_c, rq_c, iou_c, indices)
    names = [NUSCENES_LABELS_16[i] for i in indices]
    print(f"\n  {label}: {', '.join(names)}")
    print(f"    PQ={pq:.4f}  SQ={sq:.4f}  RQ={rq:.4f}  mIoU={iou:.4f}")
    for i in indices:
        print(f"    {NUSCENES_LABELS_16[i]:<25}  PQ={pq_c[i]:.4f}  SQ={sq_c[i]:.4f}  RQ={rq_c[i]:.4f}  IoU={iou_c[i]:.4f}")
    return pq, sq, rq, iou


def evaluate_checkpoint(args):
    """Load checkpoint, run inference on nuScenes val, compute B12/N4 metrics."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_cfg = SPLITS[args.split]

    print("=" * 70)
    print(f"  B12/N4 Open-Vocabulary Evaluation")
    print(f"  Split    : {split_cfg['name']}")
    print(f"  {split_cfg['desc']}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Version  : {args.version}")
    print("=" * 70)

    # ── Load model ────────────────────────────────────────────────────────────
    print("\nLoading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location=device)

    # Try to load PanopticDecoder (handles both e2e and base checkpoints)
    try:
        from panoptic.models.panoptic_decoder import PanopticDecoder
        model_cfg = ckpt.get("config", {})
        model = PanopticDecoder(model_cfg)
        model.load_state_dict(ckpt["model"], strict=False)
    except Exception as e:
        print(f"  [WARN] Could not load PanopticDecoder: {e}")
        print("  Running in feature-only mode (backbone pooling eval).")
        model = None

    # ── Load CLIP text embeddings for ALL 16 classes ───────────────────────
    print("Encoding CLIP text embeddings for all 16 classes...")
    try:
        import clip
        clip_model, _ = clip.load("ViT-L/14@336px", device=device)
        text_embeds = []
        for cls_name in NUSCENES_LABELS_16:
            tokens = clip.tokenize([f"a {cls_name} in a scene"]).to(device)
            with torch.no_grad():
                feat = clip_model.encode_text(tokens)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            text_embeds.append(feat)
        text_embeds = torch.cat(text_embeds, dim=0)  # [16, 768]
        print(f"  Text embeddings: {text_embeds.shape}")
    except ImportError:
        print("  [WARN] CLIP not available. Loading from cached prototypes.")
        proto_path = "panoptic/clip_prototypes/text_prototypes_vitl14_336_16class.pt"
        text_embeds = torch.load(proto_path, map_location=device)

    # ── Load nuScenes dataset ─────────────────────────────────────────────
    print(f"\nLoading nuScenes {args.version} val split...")
    try:
        from panoptic.dataset.nuscenes_panoptic import NuScenesPanopticDataset
        dataset = NuScenesPanopticDataset(
            dataroot=args.dataroot,
            version=args.version,
            split="val",
        )
        print(f"  Val samples: {len(dataset)}")
    except Exception as e:
        print(f"  [ERROR] Could not load dataset: {e}")
        print("  Make sure dataroot and version are correct.")
        return None

    # ── Run evaluation ────────────────────────────────────────────────────
    evaluator = PanopticEval(n_classes=16, ignore=None, min_points=50)

    if model is not None:
        model.eval().to(device)

    print(f"\nEvaluating {len(dataset)} samples...")
    with torch.no_grad():
        for idx, sample in enumerate(dataset):
            if (idx + 1) % 500 == 0:
                print(f"  [{idx+1}/{len(dataset)}]")

            # Get predictions — handles both model and backbone-pool paths
            if model is not None:
                pred_sem, pred_inst = _model_inference(model, sample, text_embeds, device, split_cfg, args.novel_threshold)
            else:
                pred_sem, pred_inst = _backbone_pool_inference(sample, text_embeds, device, split_cfg, args.novel_threshold)

            gt_sem  = sample["gt_semantic"].numpy().astype(np.int32)   # [N]
            gt_inst = sample["gt_instance"].numpy().astype(np.int32)   # [N]

            evaluator.addBatch(
                pred_sem.astype(np.int32),
                pred_inst.astype(np.int32),
                gt_sem,
                gt_inst,
            )

    # ── Compute metrics ───────────────────────────────────────────────────
    pq_c, sq_c, rq_c, _ = evaluator.getPQ()
    iou_c, _             = evaluator.getSemIoU()

    pq_all  = float(np.mean(pq_c))
    sq_all  = float(np.mean(sq_c))
    rq_all  = float(np.mean(rq_c))
    miou_all = float(np.mean(iou_c))

    # ── Print results ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    print(f"\n  OVERALL (16 classes):")
    print(f"    PQ={pq_all:.4f}  SQ={sq_all:.4f}  RQ={rq_all:.4f}  mIoU={miou_all:.4f}")
    print(f"\n  Per-class breakdown:")
    print(f"  {'Class':<28} {'PQ':>6} {'SQ':>6} {'RQ':>6} {'IoU':>6}")
    print("  " + "-" * 55)
    for i, name in enumerate(NUSCENES_LABELS_16):
        marker = "  [NOVEL]" if i in split_cfg["novel_idx"] else ""
        print(f"  {name:<28} {pq_c[i]:6.4f} {sq_c[i]:6.4f} {rq_c[i]:6.4f} {iou_c[i]:6.4f}{marker}")

    pq_b,   sq_b,   rq_b,   iou_b   = print_split_results("BASE  (12 classes)", split_cfg["base_idx"],          pq_c, sq_c, rq_c, iou_c)
    pq_n,   sq_n,   rq_n,   iou_n   = print_split_results("NOVEL  (4 classes)", split_cfg["novel_idx"],         pq_c, sq_c, rq_c, iou_c)
    pq_nth, sq_nth, rq_nth, iou_nth = print_split_results("NOVEL THINGS       ", split_cfg["novel_things_idx"],  pq_c, sq_c, rq_c, iou_c)
    pq_nst, sq_nst, rq_nst, iou_nst = print_split_results("NOVEL STUFF        ", split_cfg["novel_stuff_idx"],   pq_c, sq_c, rq_c, iou_c)

    # ── Paper comparison ──────────────────────────────────────────────────
    if args.split == "b12n4_main":
        print("\n" + "=" * 70)
        print("  COMPARISON vs PAPER (B12/N4 Main Split)")
        print("  " + "-" * 55)
        print(f"  {'Method':<30} {'PQ':>6} {'PQ_NTh':>8} {'PQ_NSt':>8} {'mIoU':>6}")
        print(f"  {'PFC baseline (paper)':30} {'54.8':>6} {'37.3':>8} {'0.5':>8} {'55.5':>6}")
        print(f"  {'Xiao et al. (paper)':30} {'62.0':>6} {'49.6':>8} {'35.2':>8} {'60.1':>6}")
        print(f"  {'Ours':30} {pq_all*100:6.1f} {pq_nth*100:8.1f} {pq_nst*100:8.1f} {miou_all*100:6.1f}")
        print("=" * 70)

    # ── Save results ──────────────────────────────────────────────────────
    results = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "version": args.version,
        "overall": {"PQ": pq_all, "SQ": sq_all, "RQ": rq_all, "mIoU": miou_all},
        "base":   {"PQ": pq_b,   "SQ": sq_b,   "RQ": rq_b,   "mIoU": iou_b},
        "novel":  {"PQ": pq_n,   "SQ": sq_n,   "RQ": rq_n,   "mIoU": iou_n},
        "novel_things": {"PQ": pq_nth, "SQ": sq_nth, "RQ": rq_nth, "mIoU": iou_nth},
        "novel_stuff":  {"PQ": pq_nst, "SQ": sq_nst, "RQ": rq_nst, "mIoU": iou_nst},
        "per_class": {
            NUSCENES_LABELS_16[i]: {
                "PQ": float(pq_c[i]), "SQ": float(sq_c[i]),
                "RQ": float(rq_c[i]), "IoU": float(iou_c[i]),
                "is_novel": i in split_cfg["novel_idx"],
            }
            for i in range(16)
        },
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to: {args.output}")

    return results


def _model_inference(model, sample, text_embeds, device, split_cfg, threshold=0.0):
    """Run model forward pass and return (pred_sem, pred_inst) as numpy arrays."""
    # Placeholder — actual implementation depends on PanopticDecoder API
    # Will be completed when train_openvocab_e2e.py inference path is finalized
    raise NotImplementedError(
        "Model inference path not yet implemented. "
        "Use backbone_pool mode (--use_backbone_pool) for now."
    )


def _apply_novel_threshold(logits, base_idx, novel_idx, threshold):
    """Base-class suppression threshold for novel class detection.

    Problem: at test time argmax always favours base classes — they have 12× more
    training signal than novel classes. Novel classes never win even when CLIP
    similarity is meaningful. This is why Token-OpenVocab shows 0 novel detections.

    Fix: if the best base class confidence is below threshold τ, predict the best
    novel class instead. This gives novel classes a fair chance without retraining.

    Args:
        logits:    (N, 16) cosine similarity logits for all 16 classes
        base_idx:  list of 12 base class indices
        novel_idx: list of 4 novel class indices
        threshold: τ — if max base confidence < τ, use novel prediction.
                   Tune on mini val. Typical range: 0.3–0.6.

    Returns:
        pred_class: (N,) predicted class indices (0–15)
    """
    if threshold <= 0.0:
        # No threshold — standard argmax (base classes dominate)
        return logits.argmax(dim=-1).cpu().numpy()

    base_idx_t  = torch.tensor(base_idx,  dtype=torch.long, device=logits.device)
    novel_idx_t = torch.tensor(novel_idx, dtype=torch.long, device=logits.device)

    base_logits  = logits[:, base_idx_t]   # (N, 12)
    novel_logits = logits[:, novel_idx_t]  # (N, 4)

    max_base_conf = base_logits.max(dim=-1).values  # (N,)
    best_base  = base_idx_t[base_logits.argmax(dim=-1)]    # (N,) actual class index
    best_novel = novel_idx_t[novel_logits.argmax(dim=-1)]  # (N,) actual class index

    pred = torch.where(max_base_conf < threshold, best_novel, best_base)
    return pred.cpu().numpy()


def _backbone_pool_inference(sample, text_embeds, device, split_cfg, threshold=0.0):
    """Use backbone CLIP features + cosine similarity for open-vocab classification.

    The backbone output is in CLIP space (pre-distilled from OpenSeg).
    Cosine similarity with all 16 text embeddings gives per-voxel class scores.
    Novel classes (bicycle, vegetation etc.) are detectable via CLIP similarity
    even though they had zero training labels.

    Base-class suppression threshold τ: applied to give novel classes a fair shot.
    """
    backbone_feats = sample["backbone_features"].to(device)  # [N, 768]
    backbone_feats = backbone_feats / (backbone_feats.norm(dim=-1, keepdim=True) + 1e-8)
    logits = backbone_feats @ text_embeds.T  # [N, 16]

    pred_sem = _apply_novel_threshold(
        logits,
        base_idx=split_cfg["base_idx"],
        novel_idx=split_cfg["novel_idx"],
        threshold=threshold,
    )
    pred_inst = np.zeros_like(pred_sem)
    return pred_sem, pred_inst


def main():
    parser = argparse.ArgumentParser(description="B12/N4 Open-Vocab PQ Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pth file")
    parser.add_argument("--split", type=str, default="b12n4_main",
                        choices=list(SPLITS.keys()),
                        help="Class split to use (default: b12n4_main)")
    parser.add_argument("--version", type=str, default="v1.0-mini",
                        choices=["v1.0-mini", "v1.0-trainval"],
                        help="nuScenes version")
    parser.add_argument("--dataroot", type=str,
                        default="/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes_mini",
                        help="Path to nuScenes data directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save JSON results (optional)")
    parser.add_argument("--use_backbone_pool", action="store_true",
                        help="Use backbone feature pooling instead of full model")
    parser.add_argument("--novel_threshold", type=float, default=0.0,
                        help=(
                            "Base-class suppression threshold τ (0=disabled). "
                            "If max base-class confidence < τ, predict best novel class. "
                            "Fixes 0-novel-detection problem. Tune on mini val. "
                            "Typical range: 0.3-0.6. Start with 0.4."
                        ))
    args = parser.parse_args()

    if args.version == "v1.0-trainval":
        args.dataroot = args.dataroot.replace("nuscenes_mini", "nuscenes")

    results = evaluate_checkpoint(args)
    return results


if __name__ == "__main__":
    main()
