"""Open-Vocab Evaluation: Token Head v2 Model.

Evaluates the token_v2 model's open-vocabulary panoptic segmentation capability
by using the CLIPTokenHead for classification (prototype matching) instead of
the CE classifier (pred_logits argmax).

Reports:
  - PQ/SQ/RQ overall, per-class, base vs novel split
  - Comparison: CE classifier vs token head (same masks, different classification)

Usage:
    python panoptic/scripts/eval_openvocab_v2.py \
        --checkpoint panoptic/checkpoints_token_full_v2/best.pth \
        --config panoptic/configs/full_nuscenes_token_v2.yaml \
        --device cuda --n_samples 200
"""
import argparse
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene")

from panoptic.train.train_panoptic_job1 import (
    load_backbone, extract_backbone_features, get_nested
)
from panoptic.models.mask_decoder import PanopticDecoder
from panoptic.models.lora_backbone import insert_lora_adapters
from panoptic.dataset.nuscenes_panoptic import NuScenesPanoptic
from panoptic.eval.eval_pq import PanopticEval

import yaml


NUSCENES_CLASS_NAMES = [
    "barrier", "bicycle", "bus", "car", "construction vehicle",
    "motorcycle", "pedestrian", "traffic cone", "trailer", "truck",
    "driveable surface", "other flat", "sidewalk", "terrain", "manmade", "vegetation",
]

NUM_CLASSES = 16
THING_IDS = list(range(10))  # 0-9 are thing classes
STUFF_IDS = list(range(10, 16))  # 10-15 are stuff classes
NOVEL_IDS = {2, 6, 14, 15}  # bus, pedestrian, manmade, vegetation
BASE_IDS = [i for i in range(16) if i not in NOVEL_IDS]


def panoptic_inference_ce(pred_logits, pred_masks, overlap_thr=0.8, min_pts=50):
    """Standard closed-set panoptic inference using CE classifier."""
    scores, labels = pred_logits.max(dim=-1)  # (Q,)
    mask_probs = pred_masks.sigmoid()  # (V, Q)
    V = mask_probs.shape[0]
    device = mask_probs.device

    pan_seg = torch.zeros(V, dtype=torch.long, device=device)
    used = torch.zeros(V, dtype=torch.bool, device=device)
    instance_id = 0

    # Sort by score
    sorted_idx = scores.argsort(descending=True)

    for idx in sorted_idx:
        label = labels[idx].item()
        if label == NUM_CLASSES:  # no-object
            continue

        mask = mask_probs[:, idx] > 0.5
        mask = mask & ~used

        if mask.sum() < min_pts:
            continue

        # Check overlap
        overlap = (mask & used).sum().float() / (mask.sum().float() + 1e-6)
        if overlap > overlap_thr:
            continue

        instance_id += 1
        pan_seg[mask] = label * 1000 + instance_id
        used[mask] = True

    return pan_seg


def panoptic_inference_clip(pred_tokens, pred_masks, text_protos, temperature=1.0,
                            overlap_thr=0.8, min_pts=50):
    """Open-vocab panoptic inference using token head + prototype matching."""
    # Classify queries via cosine similarity to text prototypes
    pred_tokens_norm = F.normalize(pred_tokens.float(), dim=-1)  # (Q, 768)
    text_protos_norm = F.normalize(text_protos.float(), dim=-1)  # (C, 768)

    sim = pred_tokens_norm @ text_protos_norm.T  # (Q, C)
    sim = sim / temperature

    scores, labels = sim.max(dim=-1)  # (Q,) - best class per query

    mask_probs = pred_masks.sigmoid()  # (V, Q)
    V = mask_probs.shape[0]
    device = mask_probs.device

    pan_seg = torch.zeros(V, dtype=torch.long, device=device)
    used = torch.zeros(V, dtype=torch.bool, device=device)
    instance_id = 0

    # Sort by similarity score
    sorted_idx = scores.argsort(descending=True)

    for idx in sorted_idx:
        label = labels[idx].item()
        if label >= NUM_CLASSES:
            continue

        mask = mask_probs[:, idx] > 0.5
        mask = mask & ~used

        if mask.sum() < min_pts:
            continue

        overlap = (mask & used).sum().float() / (mask.sum().float() + 1e-6)
        if overlap > overlap_thr:
            continue

        instance_id += 1
        pan_seg[mask] = label * 1000 + instance_id
        used[mask] = True

    return pan_seg


def decode_panoptic(pan_seg):
    """Decode panoptic map to semantic and instance labels."""
    sem = (pan_seg // 1000).cpu().numpy().astype(np.int32)
    inst = (pan_seg % 1000).cpu().numpy().astype(np.int32)
    # Unlabeled voxels (pan_seg == 0) get class 0, instance 0
    # Mark them as ignore by setting sem to a special value
    unlabeled = pan_seg.cpu().numpy() == 0
    sem[unlabeled] = NUM_CLASSES  # will be ignored by evaluator
    return sem, inst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for prototype matching (lower=sharper)")
    parser.add_argument("--overlap_thr", type=float, default=0.8)
    parser.add_argument("--min_pts", type=int, default=50)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)

    # Load text prototypes
    proto_path = get_nested(cfg, "token.prototype_path",
                            "panoptic/clip_prototypes/text_prototypes_vitl14_336.pt")
    proto_data = torch.load(proto_path, map_location="cpu")
    text_protos = proto_data["embeddings"].float().to(device)  # (16, 768)
    text_protos = F.normalize(text_protos, dim=-1)
    print(f"Loaded prototypes from: {proto_path}")

    # Load model
    ckpt_path = get_nested(cfg, "model.backbone_ckpt", "models/nuscenes_openseg.pth.tar")
    backbone = load_backbone(ckpt_path, device)
    lora_adapters = None
    hooks = []
    if get_nested(cfg, "lora.enabled", False):
        lora_adapters, hooks = insert_lora_adapters(
            backbone, rank=get_nested(cfg, "lora.rank", 16)
        )
        lora_adapters.to(device)

    decoder = PanopticDecoder(
        backbone_dim=get_nested(cfg, "model.backbone_dim", 768),
        hidden_dim=get_nested(cfg, "model.hidden_dim", 256),
        num_classes=get_nested(cfg, "model.num_classes", 16),
        num_thing_queries=get_nested(cfg, "model.num_thing_queries", 100),
        num_stuff_classes=get_nested(cfg, "model.num_stuff_classes", 6),
        nheads=get_nested(cfg, "model.nheads", 8),
        dim_feedforward=get_nested(cfg, "model.dim_feedforward", 1024),
        num_blocks=get_nested(cfg, "model.num_blocks", 3),
        num_levels=get_nested(cfg, "model.num_levels", 3),
        dropout=get_nested(cfg, "model.dropout", 0.0),
        use_token_head=get_nested(cfg, "model.use_token_head", False),
        token_dim=get_nested(cfg, "model.token_dim", 768),
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "decoder_state" in ckpt:
        state = ckpt["decoder_state"]
    elif "decoder" in ckpt:
        state = ckpt["decoder"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    decoder.load_state_dict(state, strict=False)
    if lora_adapters is not None and "lora_state" in ckpt:
        lora_adapters.load_state_dict(ckpt["lora_state"])

    decoder.to(device)
    decoder.eval()

    # Check if token head loaded
    has_token_head = decoder.token_head is not None
    print(f"Token head present: {has_token_head}")
    if has_token_head:
        # Check if weights are non-random by looking at output norm
        test_input = torch.randn(1, 106, 256).to(device)
        with torch.no_grad():
            test_out = decoder.token_head(test_input)
        print(f"Token head output shape: {test_out.shape}, mean norm: {test_out.norm(dim=-1).mean():.3f}")

    # Dataset
    dataroot = get_nested(cfg, "data.dataroot")
    version = get_nested(cfg, "data.version", "v1.0-trainval")
    voxel_size = get_nested(cfg, "data.voxel_size", 0.05)
    val_dataset = NuScenesPanoptic(
        dataroot=dataroot, version=version, split="val", voxel_size=voxel_size
    )

    n_samples = min(args.n_samples, len(val_dataset))
    print(f"\nEvaluating {n_samples} val samples...")
    print(f"Temperature: {args.temperature}")
    print(f"Novel classes: {[NUSCENES_CLASS_NAMES[i] for i in sorted(NOVEL_IDS)]}\n")

    # Evaluators
    eval_ce = PanopticEval(n_classes=NUM_CLASSES, min_points=args.min_pts)
    eval_clip = PanopticEval(n_classes=NUM_CLASSES, min_points=args.min_pts)

    with torch.no_grad():
        for sample_idx in range(n_samples):
            batch = val_dataset[sample_idx]

            feats, coords = extract_backbone_features(
                backbone, batch, device, lora_adapters
            )
            outputs = decoder(feats, coords)

            pred_masks = outputs["pred_masks"][0]  # (V, Q)
            pred_logits = outputs["pred_logits"][0]  # (Q, 17)
            V = pred_masks.shape[0]

            # Get token head output
            token_output = outputs.get("pred_embed", None)
            if token_output is None:
                token_output = outputs.get("pred_tokens", None)

            # GT
            gt_masks_raw = batch["gt_masks"]
            gt_classes_raw = batch["gt_classes"].numpy().flatten()
            if gt_masks_raw.dim() == 3:
                gt_masks_raw = gt_masks_raw[0]
            if gt_masks_raw.dim() == 1:
                gt_masks_raw = gt_masks_raw.unsqueeze(0)

            # Build GT panoptic map
            gt_sem = np.full(V, NUM_CLASSES, dtype=np.int32)  # unlabeled
            gt_inst = np.zeros(V, dtype=np.int32)
            for m_idx in range(len(gt_classes_raw)):
                cls = int(gt_classes_raw[m_idx])
                if cls >= NUM_CLASSES:
                    continue
                mask = gt_masks_raw[m_idx]
                if mask.shape[0] > V:
                    mask = mask[:V]
                elif mask.shape[0] < V:
                    continue
                mask_np = mask.bool().numpy()
                gt_sem[mask_np] = cls
                gt_inst[mask_np] = m_idx + 1

            # --- CE inference ---
            pan_ce = panoptic_inference_ce(
                pred_logits, pred_masks,
                overlap_thr=args.overlap_thr, min_pts=args.min_pts
            )
            sem_ce, inst_ce = decode_panoptic(pan_ce)
            eval_ce.addBatch(sem_ce, inst_ce, gt_sem, gt_inst)

            # --- CLIP inference ---
            if token_output is not None:
                tokens = token_output[0]  # (Q, 768)
                pan_clip = panoptic_inference_clip(
                    tokens, pred_masks, text_protos,
                    temperature=args.temperature,
                    overlap_thr=args.overlap_thr, min_pts=args.min_pts
                )
                sem_clip, inst_clip = decode_panoptic(pan_clip)
                eval_clip.addBatch(sem_clip, inst_clip, gt_sem, gt_inst)

            if (sample_idx + 1) % 50 == 0:
                pq, sq, rq, pq_per, _, _ = eval_clip.getPQ()
                print(f"  [{sample_idx+1}/{n_samples}] CLIP PQ={pq:.4f} SQ={sq:.4f} RQ={rq:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # Results
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "█" * 70)
    print("  OPEN-VOCAB EVALUATION RESULTS")
    print("█" * 70)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Samples: {n_samples}")
    print(f"  Temperature: {args.temperature}")

    # CE results
    pq_ce, sq_ce, rq_ce, pq_per_ce, sq_per_ce, rq_per_ce = eval_ce.getPQ()
    # CLIP results
    pq_clip, sq_clip, rq_clip, pq_per_clip, sq_per_clip, rq_per_clip = eval_clip.getPQ()

    print(f"\n{'═'*70}")
    print(f"  OVERALL COMPARISON")
    print(f"{'═'*70}")
    print(f"  {'Method':<25} {'PQ':>8} {'SQ':>8} {'RQ':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'CE Classifier (closed)':<25} {pq_ce:>8.4f} {sq_ce:>8.4f} {rq_ce:>8.4f}")
    print(f"  {'Token Head (open-vocab)':<25} {pq_clip:>8.4f} {sq_clip:>8.4f} {rq_clip:>8.4f}")

    # Base vs Novel split
    def split_metrics(pq_per, sq_per, rq_per):
        base_pq = np.mean([pq_per[i] for i in BASE_IDS if i < len(pq_per)])
        novel_pq = np.mean([pq_per[i] for i in NOVEL_IDS if i < len(pq_per)])
        base_rq = np.mean([rq_per[i] for i in BASE_IDS if i < len(rq_per)])
        novel_rq = np.mean([rq_per[i] for i in NOVEL_IDS if i < len(rq_per)])
        return base_pq, novel_pq, base_rq, novel_rq

    base_pq_ce, novel_pq_ce, base_rq_ce, novel_rq_ce = split_metrics(pq_per_ce, sq_per_ce, rq_per_ce)
    base_pq_clip, novel_pq_clip, base_rq_clip, novel_rq_clip = split_metrics(pq_per_clip, sq_per_clip, rq_per_clip)

    print(f"\n{'═'*70}")
    print(f"  BASE vs NOVEL SPLIT")
    print(f"{'═'*70}")
    print(f"  {'Metric':<25} {'CE':>12} {'Token Head':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    print(f"  {'PQ (base classes)':<25} {base_pq_ce:>12.4f} {base_pq_clip:>12.4f}")
    print(f"  {'PQ (novel classes)':<25} {novel_pq_ce:>12.4f} {novel_pq_clip:>12.4f}")
    print(f"  {'RQ (base classes)':<25} {base_rq_ce:>12.4f} {base_rq_clip:>12.4f}")
    print(f"  {'RQ (novel classes)':<25} {novel_rq_ce:>12.4f} {novel_rq_clip:>12.4f}")

    # Per-class breakdown
    print(f"\n{'═'*70}")
    print(f"  PER-CLASS PQ BREAKDOWN")
    print(f"{'═'*70}")
    print(f"  {'Class':<22} {'CE PQ':>8} {'CLIP PQ':>8} {'Delta':>8} {'Type':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for cls in range(NUM_CLASSES):
        if cls < len(pq_per_ce) and cls < len(pq_per_clip):
            ce_val = pq_per_ce[cls]
            clip_val = pq_per_clip[cls]
            delta = clip_val - ce_val
            tag = "NOVEL" if cls in NOVEL_IDS else "base"
            print(f"  {NUSCENES_CLASS_NAMES[cls]:<22} {ce_val:>8.4f} {clip_val:>8.4f} {delta:>+8.4f} {tag:>8}")

    # Thing vs Stuff
    thing_pq_ce = np.mean([pq_per_ce[i] for i in THING_IDS if i < len(pq_per_ce)])
    thing_pq_clip = np.mean([pq_per_clip[i] for i in THING_IDS if i < len(pq_per_clip)])
    stuff_pq_ce = np.mean([pq_per_ce[i] for i in STUFF_IDS if i < len(pq_per_ce)])
    stuff_pq_clip = np.mean([pq_per_clip[i] for i in STUFF_IDS if i < len(pq_per_clip)])

    print(f"\n  {'Thing PQ':<22} {thing_pq_ce:>8.4f} {thing_pq_clip:>8.4f} {thing_pq_clip-thing_pq_ce:>+8.4f}")
    print(f"  {'Stuff PQ':<22} {stuff_pq_ce:>8.4f} {stuff_pq_clip:>8.4f} {stuff_pq_clip-stuff_pq_ce:>+8.4f}")

    # Cleanup
    for h in hooks:
        h.remove()

    print(f"\n{'═'*70}")
    print("  INTERPRETATION GUIDE")
    print(f"{'═'*70}")
    print("  - If CLIP PQ ≈ CE PQ: token head learned same classification as CE")
    print("  - If CLIP PQ_novel > 0: model has open-vocab capability already")
    print("  - If CLIP RQ << CE RQ: token head misclassifies (masks are same)")
    print("  - SQ should be identical (same masks, different labels)")
    print()


if __name__ == "__main__":
    main()
