"""Phase 5a (B2b) evaluation: class-agnostic panoptic with backbone pooling.

Modes:
  1. backbone_pool_22   – Pool backbone feats per mask → cosine to 22 protos
  2. objectness_cosine  – Objectness * cosine ranking, sweep obj_thr
  3. backbone_pool_16   – Pool backbone feats → cosine to 16 known protos only (upper bound)

Usage:
    python -u panoptic/scripts/evaluate_b2b.py \
        --config panoptic/configs/full_nuscenes_b2b.yaml \
        --checkpoint panoptic/checkpoints_b2b/best.pth \
        --version v1.0-trainval
"""
import argparse
import sys
import os
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import MinkowskiEngine as ME
import yaml

sys.path.insert(0, "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene")

from panoptic.eval.eval_pq import PanopticEval
from panoptic.dataset.nuscenes_panoptic import (
    NuScenesPanoptic, NUM_CLASSES, NUSCENES_CLASS_NAMES,
)
from panoptic.scripts.evaluate_comprehensive import (
    TeeLogger, load_backbone, load_config, extract_backbone_features,
    load_decoder_flexible, _greedy_assign, format_table,
    count_novel_detections, remap_novel_to_ignore,
    NOVEL_CLASS_NAMES, ALL_22_CLASS_NAMES, THING_CLASSES_16,
)
from panoptic.train.train_panoptic_b2b import load_text_prototypes_22


# ── B2b inference functions ──────────────────────────────────────────────────

def panoptic_inference_b2b(
    backbone_feats, pred_masks, text_protos,
    overlap_thr=0.8, obj_thr=0.3, min_mask_pts=10,
    use_objectness_weighting=True,
):
    """Class-agnostic: objectness from mask, class from backbone pooling."""
    masks = pred_masks.sigmoid()  # (V, Q)
    V, Q = masks.shape

    objectness = masks.max(dim=0).values  # (Q,)

    pooled = []
    valid_queries = []
    for q in range(Q):
        if objectness[q] < obj_thr:
            continue
        binary = masks[:, q] >= 0.5
        if binary.sum() < min_mask_pts:
            continue
        feat = F.normalize(backbone_feats[binary].mean(dim=0), dim=0)
        pooled.append(feat)
        valid_queries.append(q)

    if len(pooled) == 0:
        return np.zeros(V, dtype=np.int64), np.zeros(V, dtype=np.int64)

    pooled = torch.stack(pooled)
    cos_sim = torch.mm(pooled, text_protos.t())
    scores, labels = cos_sim.max(-1)

    if use_objectness_weighting:
        obj_scores = objectness[valid_queries]
        scores = obj_scores * scores

    valid_masks = masks[:, valid_queries]
    num_protos = text_protos.shape[0]
    thing_set = THING_CLASSES_16 | set(range(16, num_protos)) if num_protos > 16 else THING_CLASSES_16
    return _greedy_assign(scores, labels, valid_masks, V, overlap_thr, thing_set)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 5a (B2b) evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--version", default=None)
    parser.add_argument("--dataroot", default=None)
    parser.add_argument("--output_dir", default="panoptic/eval_results/b2b/")
    parser.add_argument("--obj_thresholds", default="0.1,0.2,0.3,0.5")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj_thrs = [float(t) for t in args.obj_thresholds.split(",")]

    version = args.version or cfg["data"]["version"]
    if args.dataroot:
        cfg["data"]["dataroot"] = args.dataroot

    ckpt_stem = Path(args.checkpoint).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_tag = version.replace("v1.0-", "")
    run_name = f"b2b_{ckpt_stem}_{version_tag}_{timestamp}"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tee = TeeLogger(str(out_dir / "eval_log.txt"))
    sys.stdout = tee

    print(f"{'='*70}", flush=True)
    print(f"  Phase 5a (B2b) Class-Agnostic Evaluation", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Config:     {args.config}", flush=True)
    print(f"  Checkpoint: {args.checkpoint}", flush=True)
    print(f"  Version:    {version}", flush=True)
    print(f"  Obj thrs:   {obj_thrs}", flush=True)
    print(f"  Output:     {out_dir}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # ── Data ──
    dcfg = cfg["data"]
    from nuscenes.nuscenes import NuScenes as NuSc
    nusc = NuSc(version=version, dataroot=dcfg["dataroot"], verbose=False)
    val_ds = NuScenesPanoptic(
        dcfg["dataroot"], version, split="val",
        voxel_size=dcfg["voxel_size"],
        point_range=np.array(dcfg["point_range"]),
        augment=False, nusc=nusc,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
        collate_fn=lambda x: x[0],
    )
    print(f"Val: {len(val_ds)} samples\n", flush=True)

    # ── Models ──
    backbone = load_backbone(cfg["model"]["backbone_ckpt"], device)
    decoder, _ = load_decoder_flexible(cfg, args.checkpoint, device)

    # ── Prototypes ──
    ov_cfg = cfg.get("open_vocab", {})
    proto_path = ov_cfg.get("prototypes_path", "panoptic/embeddings/text_prototypes_22.pt")
    text_protos_22 = load_text_prototypes_22(proto_path, device)
    text_protos_16 = text_protos_22[:16]  # first 16 are known
    print(f"22-class prototypes: {text_protos_22.shape}", flush=True)
    print(f"16-class prototypes: {text_protos_16.shape}\n", flush=True)

    num_classes = cfg["model"]["num_classes"]
    overlap_thr = cfg["inference"]["overlap_thr"]
    min_points = cfg["inference"].get("min_points", 50)

    # ── Setup evaluators ──
    evaluators = {}
    # Mode 1: backbone_pool_22 with each obj_thr
    for thr in obj_thrs:
        evaluators[f"pool22_obj{thr:.1f}"] = PanopticEval(n_classes=num_classes, min_points=min_points)
    # Mode 2: backbone_pool_22 without objectness weighting
    evaluators["pool22_no_obj"] = PanopticEval(n_classes=num_classes, min_points=min_points)
    # Mode 3: backbone_pool_16 (known-only, upper bound)
    evaluators["pool16_obj0.3"] = PanopticEval(n_classes=num_classes, min_points=min_points)

    novel_stats = {k: {name: 0 for name in NOVEL_CLASS_NAMES} for k in evaluators}

    # ── Eval loop ──
    print(f"Evaluating {len(val_ds)} samples...\n", flush=True)
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            feats, coords = extract_backbone_features(backbone, batch, device)
            outputs = decoder(feats, coords)
            pred_masks = outputs["pred_masks"][0]  # (V, Q) on device
            backbone_feats = feats[0]  # (V, 768) on device

            gt_sem = batch["gt_sem"].numpy()
            gt_masks_arr = batch["gt_masks"].numpy()
            gt_classes = batch["gt_classes"].numpy()
            V = gt_sem.shape[0]
            gt_inst = np.zeros(V, dtype=np.int64)
            for m_idx in range(len(gt_classes)):
                gt_inst[gt_masks_arr[m_idx]] = m_idx + 1
            valid = gt_sem != 255
            if valid.sum() == 0:
                continue

            pm = pred_masks.cpu()
            bf = backbone_feats.cpu()

            # Mode 1: pool22 with varying obj_thr
            for thr in obj_thrs:
                sem, inst = panoptic_inference_b2b(
                    bf, pm, text_protos_22.cpu(),
                    overlap_thr=overlap_thr, obj_thr=thr,
                    use_objectness_weighting=True,
                )
                key = f"pool22_obj{thr:.1f}"
                # Count novel voxels
                for j, name in enumerate(NOVEL_CLASS_NAMES):
                    novel_stats[key][name] += int((sem == (16 + j)).sum())
                # Remap novel for PQ
                sem_k, inst_k = remap_novel_to_ignore(sem, inst, num_classes)
                evaluators[key].addBatch(sem_k[valid], inst_k[valid], gt_sem[valid], gt_inst[valid])

            # Mode 2: no objectness weighting (pure cosine)
            sem, inst = panoptic_inference_b2b(
                bf, pm, text_protos_22.cpu(),
                overlap_thr=overlap_thr, obj_thr=0.0,
                use_objectness_weighting=False,
            )
            key = "pool22_no_obj"
            for j, name in enumerate(NOVEL_CLASS_NAMES):
                novel_stats[key][name] += int((sem == (16 + j)).sum())
            sem_k, inst_k = remap_novel_to_ignore(sem, inst, num_classes)
            evaluators[key].addBatch(sem_k[valid], inst_k[valid], gt_sem[valid], gt_inst[valid])

            # Mode 3: pool16 (known-only protos)
            sem, inst = panoptic_inference_b2b(
                bf, pm, text_protos_16.cpu(),
                overlap_thr=overlap_thr, obj_thr=0.3,
                use_objectness_weighting=True,
            )
            evaluators["pool16_obj0.3"].addBatch(sem[valid], inst[valid], gt_sem[valid], gt_inst[valid])

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(val_ds)}]", flush=True)

    # ── Results ──
    print(f"\n{'='*70}", flush=True)
    print(f"  RESULTS", flush=True)
    print(f"{'='*70}\n", flush=True)

    results = {}
    summary_rows = []

    for name, ev in evaluators.items():
        PQ, SQ, RQ, pq_all, sq_all, rq_all = ev.getPQ()
        iou_mean, iou_all = ev.getSemIoU()
        results[name] = {
            "PQ": float(PQ), "SQ": float(SQ), "RQ": float(RQ), "mIoU": float(iou_mean),
            "per_class_pq": [float(x) for x in pq_all],
        }
        novel_total = sum(novel_stats.get(name, {}).values())
        results[name]["novel_voxels"] = novel_total
        summary_rows.append((name, PQ, SQ, RQ, iou_mean, novel_total))

        print(f"  Mode: {name}", flush=True)
        print(f"  PQ={PQ:.4f}  SQ={SQ:.4f}  RQ={RQ:.4f}  mIoU={iou_mean:.4f}  novel={novel_total:,}", flush=True)
        table = format_table(pq_all, sq_all, rq_all, iou_all, NUSCENES_CLASS_NAMES)
        print(table, flush=True)
        print(flush=True)

    # ── Summary ──
    print(f"\n{'='*70}", flush=True)
    print(f"  SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  {'Mode':<25s} {'PQ':>8s} {'SQ':>8s} {'RQ':>8s} {'mIoU':>8s} {'Novel':>12s}", flush=True)
    print(f"  {'-'*70}", flush=True)
    for name, pq, sq, rq, iou, novel in summary_rows:
        print(f"  {name:<25s} {pq:8.4f} {sq:8.4f} {rq:8.4f} {iou:8.4f} {novel:>12,d}", flush=True)

    # ── Novel class breakdown for best mode ──
    best_mode = max(
        [r for r in summary_rows if r[0].startswith("pool22")],
        key=lambda r: r[1]  # best PQ
    )[0]
    print(f"\n  Novel class breakdown ({best_mode}):", flush=True)
    for name in NOVEL_CLASS_NAMES:
        cnt = novel_stats[best_mode][name]
        print(f"    {name:<20s}: {cnt:>10,d} voxels", flush=True)

    # ── Save ──
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "novel_stats.json", "w") as f:
        json.dump(novel_stats, f, indent=2)
    print(f"\nSaved to {out_dir}", flush=True)

    sys.stdout = tee.terminal
    tee.close()


if __name__ == "__main__":
    main()
