"""Three-Stage Voxel Margin Diagnostic v1.

Measures cosine margin between GT class and nearest confuser at three points
in the processing chain:

  Stage 1: Raw backbone per-voxel features (768-d DisNet output)
  Stage 2: Mask-pooled backbone features (decoder predicted masks × backbone feats)
  Stage 3: Token head output (256→768 projection from decoder queries)

This produces a chain of accountability:
  - Stage 1 → what distillation preserved
  - Stage 2 → what spatial aggregation achieves
  - Stage 3 → what the learned projection does

Usage:
    python panoptic/scripts/eval_voxel_margin_v1.py \
        --checkpoint panoptic/checkpoints_job1a/best.pth \
        --config panoptic/configs/job1a.yaml \
        --device cuda --n_samples 100
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

import yaml


NUSCENES_CLASS_NAMES = [
    "barrier", "bicycle", "bus", "car", "construction vehicle",
    "motorcycle", "pedestrian", "traffic cone", "trailer", "truck",
    "driveable surface", "other flat", "sidewalk", "terrain", "manmade", "vegetation",
]

NOVEL_IDS = {2, 6, 14, 15}  # bus, pedestrian, manmade, vegetation
BASE_IDS = [i for i in range(16) if i not in NOVEL_IDS]


def compute_margin(embedding, prototypes, gt_class):
    """Compute cosine margin for a single embedding.

    Args:
        embedding: (768,) L2-normalized
        prototypes: (16, 768) L2-normalized text prototypes
        gt_class: int, ground-truth class index

    Returns:
        margin: float, sim(GT) - max(sim(others))
        correct: bool, whether GT class has highest sim
    """
    sims = (embedding @ prototypes.T)  # (16,)
    gt_sim = sims[gt_class].item()
    other_sims = sims.clone()
    other_sims[gt_class] = -2.0
    nearest_confuser = other_sims.max().item()
    margin = gt_sim - nearest_confuser
    return margin, margin > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--max_voxels_per_sample", type=int, default=5000,
                        help="Max voxels to evaluate per sample for Stage 1")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)

    # Load text prototypes (16 classes)
    proto_path = get_nested(cfg, "token.prototype_path",
                            "panoptic/clip_prototypes/text_prototypes_vitl14_224.pt")
    proto_data = torch.load(proto_path, map_location="cpu")
    text_protos = proto_data["embeddings"].float().to(device)  # (16, 768)
    text_protos = F.normalize(text_protos, dim=-1)

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

    head_type = get_nested(cfg, "model.head_type", "dual")
    if head_type == "single_clip":
        decoder.enable_single_clip_head(get_nested(cfg, "model.clip_embed_dim", 768))

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    # Support different checkpoint formats
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

    # Dataset
    dataroot = get_nested(cfg, "data.dataroot")
    version = get_nested(cfg, "data.version", "v1.0-trainval")
    voxel_size = get_nested(cfg, "data.voxel_size", 0.05)
    val_dataset = NuScenesPanoptic(
        dataroot=dataroot, version=version, split="val", voxel_size=voxel_size
    )

    n_samples = min(args.n_samples, len(val_dataset))
    print(f"Running 3-stage margin diagnostic on {n_samples} val samples...")
    print(f"Novel classes: {[NUSCENES_CLASS_NAMES[i] for i in sorted(NOVEL_IDS)]}")
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # Accumulators
    # ═══════════════════════════════════════════════════════════════════════
    # Stage 1: per-voxel
    s1_margins = {i: [] for i in range(16)}
    # Stage 2: mask-pooled backbone
    s2_margins = {i: [] for i in range(16)}
    # Stage 3: token head output
    s3_margins = {i: [] for i in range(16)}

    with torch.no_grad():
        for sample_idx in range(n_samples):
            batch = val_dataset[sample_idx]

            # Do NOT add batch dims — extract_backbone_features expects raw tensors
            # vox_coords: (V, 3), vox_feats: (V, 3), gt_masks: (N_inst, V)

            feats, coords = extract_backbone_features(
                backbone, batch, device, lora_adapters
            )
            outputs = decoder(feats, coords)

            # Raw backbone features: (V, 768)
            raw_feats = feats[0]  # remove batch dim
            V = raw_feats.shape[0]

            # Decoder outputs
            pred_masks = outputs["pred_masks"][0]  # (V, Q) logits
            pred_masks_soft = pred_masks.sigmoid()  # (V, Q) probabilities

            # Token head output (either pred_embed or pred_tokens)
            token_output = outputs.get("pred_embed", None)
            if token_output is None:
                token_output = outputs.get("pred_tokens", None)
            # token_output: (B, Q, 768) or None

            # GT
            gt_classes = batch["gt_classes"].numpy().flatten()
            gt_masks_raw = batch["gt_masks"]  # (N_inst, V)
            if gt_masks_raw.dim() == 3:
                gt_masks_raw = gt_masks_raw[0]  # remove batch dim if present
            if gt_masks_raw.dim() == 1:
                # Single instance: (V,) → (1, V)
                gt_masks_raw = gt_masks_raw.unsqueeze(0)

            # ───────────────────────────────────────────────────────────────
            # Stage 1: Per-voxel backbone margin
            # ───────────────────────────────────────────────────────────────
            # Build per-voxel GT label map
            voxel_labels = torch.full((V,), -1, dtype=torch.long, device=device)
            for m_idx in range(len(gt_classes)):
                cls = int(gt_classes[m_idx])
                if cls >= 16:
                    continue
                mask = gt_masks_raw[m_idx]
                if mask.shape[0] > V:
                    mask = mask[:V]
                elif mask.shape[0] < V:
                    continue
                voxel_labels[mask.bool().to(device)] = cls

            # Evaluate labeled voxels (subsample if too many)
            labeled_mask = voxel_labels >= 0
            labeled_indices = labeled_mask.nonzero(as_tuple=True)[0]
            if len(labeled_indices) > args.max_voxels_per_sample:
                perm = torch.randperm(len(labeled_indices))[:args.max_voxels_per_sample]
                labeled_indices = labeled_indices[perm]

            if len(labeled_indices) > 0:
                vox_feats = raw_feats[labeled_indices]  # (K, 768)
                vox_feats_norm = F.normalize(vox_feats.float(), dim=-1)
                vox_labels = voxel_labels[labeled_indices]

                # Batch cosine similarity: (K, 16)
                all_sims = vox_feats_norm @ text_protos.T
                for k in range(len(labeled_indices)):
                    cls = vox_labels[k].item()
                    gt_sim = all_sims[k, cls].item()
                    other_sims = all_sims[k].clone()
                    other_sims[cls] = -2.0
                    margin = gt_sim - other_sims.max().item()
                    s1_margins[cls].append(margin)

            # ───────────────────────────────────────────────────────────────
            # Stage 2: Mask-pooled backbone features
            # ───────────────────────────────────────────────────────────────
            # For each GT instance, find best matching predicted query,
            # then pool backbone features using that query's predicted soft mask
            for m_idx in range(len(gt_classes)):
                cls = int(gt_classes[m_idx])
                if cls >= 16:
                    continue

                gt_mask = gt_masks_raw[m_idx]
                if gt_mask.shape[0] != V:
                    continue
                gt_mask_dev = gt_mask.float().to(device)
                if gt_mask_dev.sum() < 50:
                    continue

                # Find query with best IoU to GT
                overlaps = (pred_masks_soft * gt_mask_dev.unsqueeze(1)).sum(0)  # (Q,)
                best_q = overlaps.argmax().item()

                # Pool backbone features using predicted mask for best_q
                mask_weights = pred_masks_soft[:, best_q]  # (V,)
                # Only use voxels with >0.3 confidence to avoid noise
                confident = mask_weights > 0.3
                if confident.sum() < 10:
                    continue

                weighted_feats = raw_feats[confident].float() * mask_weights[confident].unsqueeze(1)
                pooled = weighted_feats.sum(0) / (mask_weights[confident].sum() + 1e-8)  # (768,)
                pooled_norm = F.normalize(pooled.unsqueeze(0), dim=-1).squeeze(0)

                margin, _ = compute_margin(pooled_norm, text_protos, cls)
                s2_margins[cls].append(margin)

            # ───────────────────────────────────────────────────────────────
            # Stage 3: Token head output
            # ───────────────────────────────────────────────────────────────
            if token_output is not None:
                token_embs = token_output[0]  # (Q, 768)
                token_embs_norm = F.normalize(token_embs.float(), dim=-1)

                for m_idx in range(len(gt_classes)):
                    cls = int(gt_classes[m_idx])
                    if cls >= 16:
                        continue

                    gt_mask = gt_masks_raw[m_idx]
                    if gt_mask.shape[0] != V:
                        continue
                    gt_mask_dev = gt_mask.float().to(device)
                    if gt_mask_dev.sum() < 50:
                        continue

                    # Find best matching query
                    overlaps = (pred_masks_soft * gt_mask_dev.unsqueeze(1)).sum(0)
                    best_q = overlaps.argmax().item()

                    emb = token_embs_norm[best_q]
                    margin, _ = compute_margin(emb, text_protos, cls)
                    s3_margins[cls].append(margin)

            if (sample_idx + 1) % 20 == 0:
                s1_all = [m for cls_list in s1_margins.values() for m in cls_list]
                s2_all = [m for cls_list in s2_margins.values() for m in cls_list]
                s3_all = [m for cls_list in s3_margins.values() for m in cls_list]
                print(f"  [{sample_idx+1}/{n_samples}] "
                      f"S1={np.mean(s1_all):.4f} "
                      f"S2={np.mean(s2_all) if s2_all else 'N/A':.4f} "
                      f"S3={np.mean(s3_all) if s3_all else 'N/A'}")

    # ═══════════════════════════════════════════════════════════════════════
    # Report
    # ═══════════════════════════════════════════════════════════════════════
    def report_stage(name, margins_dict):
        all_margins = [m for cls_list in margins_dict.values() for m in cls_list]
        base_margins = [m for cls in BASE_IDS for m in margins_dict[cls]]
        novel_margins = [m for cls in NOVEL_IDS for m in margins_dict[cls]]

        print(f"\n{'═'*70}")
        print(f"  {name}")
        print(f"{'═'*70}")

        if not all_margins:
            print("  [NO DATA]")
            return

        print(f"\n  Overall ({len(all_margins)} samples):")
        print(f"    Mean margin:   {np.mean(all_margins):.4f}")
        print(f"    Median margin: {np.median(all_margins):.4f}")
        print(f"    Std:           {np.std(all_margins):.4f}")
        print(f"    Correct (>0):  {100*np.mean(np.array(all_margins)>0):.1f}%")

        if base_margins:
            print(f"\n  Base classes ({len(base_margins)} samples):")
            print(f"    Mean margin:   {np.mean(base_margins):.4f}")
            print(f"    Correct (>0):  {100*np.mean(np.array(base_margins)>0):.1f}%")

        if novel_margins:
            print(f"\n  Novel classes ({len(novel_margins)} samples):")
            print(f"    Mean margin:   {np.mean(novel_margins):.4f}")
            print(f"    Correct (>0):  {100*np.mean(np.array(novel_margins)>0):.1f}%")

        print(f"\n  Per-class breakdown:")
        for cls in range(16):
            if margins_dict[cls]:
                m = np.mean(margins_dict[cls])
                n = len(margins_dict[cls])
                pct = 100 * np.mean(np.array(margins_dict[cls]) > 0)
                tag = " [NOVEL]" if cls in NOVEL_IDS else ""
                print(f"    {NUSCENES_CLASS_NAMES[cls]:22s}: "
                      f"margin={m:+.4f}  correct={pct:5.1f}%  (n={n}){tag}")

    print("\n" + "█" * 70)
    print("  THREE-STAGE VOXEL MARGIN DIAGNOSTIC")
    print("█" * 70)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Samples: {n_samples}")

    report_stage("STAGE 1: Raw Backbone Per-Voxel Features (768-d DisNet)", s1_margins)
    report_stage("STAGE 2: Mask-Pooled Backbone Features (pred_mask × backbone)", s2_margins)
    report_stage("STAGE 3: Token Head Output (256→768 projection)", s3_margins)

    # Summary comparison
    s1_all = [m for cls_list in s1_margins.values() for m in cls_list]
    s2_all = [m for cls_list in s2_margins.values() for m in cls_list]
    s3_all = [m for cls_list in s3_margins.values() for m in cls_list]

    print(f"\n{'═'*70}")
    print("  SUMMARY COMPARISON")
    print(f"{'═'*70}")
    print(f"  {'Stage':<45} {'Margin':>8} {'Correct':>8}")
    print(f"  {'-'*45} {'-'*8} {'-'*8}")
    if s1_all:
        print(f"  {'1. Raw backbone (per-voxel)':<45} {np.mean(s1_all):>+8.4f} {100*np.mean(np.array(s1_all)>0):>7.1f}%")
    if s2_all:
        print(f"  {'2. Mask-pooled backbone':<45} {np.mean(s2_all):>+8.4f} {100*np.mean(np.array(s2_all)>0):>7.1f}%")
    if s3_all:
        print(f"  {'3. Token head output':<45} {np.mean(s3_all):>+8.4f} {100*np.mean(np.array(s3_all)>0):>7.1f}%")

    print()
    if s1_all and s2_all:
        delta = np.mean(s2_all) - np.mean(s1_all)
        print(f"  Spatial pooling gain (S2-S1): {delta:+.4f}"
              f"  {'[pooling helps]' if delta > 0.01 else '[pooling neutral/hurts]'}")
    if s2_all and s3_all:
        delta = np.mean(s3_all) - np.mean(s2_all)
        print(f"  Token head gain (S3-S2):      {delta:+.4f}"
              f"  {'[head adds value]' if delta > 0.01 else '[head neutral/hurts]'}")

    # Cleanup hooks
    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
