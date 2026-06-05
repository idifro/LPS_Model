"""Voxel Margin Check: Measure cosine separation after Job 1a.

Compares per-query CLIP embeddings against all text prototypes to verify
that Job 1a training improved the cosine margin between GT class and
nearest confuser (baseline was ~0.02 from raw DisNet).

Usage:
    python panoptic/scripts/eval_voxel_margin.py \
        --checkpoint panoptic/checkpoints_job1a/best.pth \
        --config panoptic/configs/job1a.yaml \
        --device cuda --n_samples 100
"""
import argparse
import sys
import numpy as np
import torch

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

# Extra novel concepts to test margin against
EXTRA_CONCEPTS = [
    "ambulance", "trash can", "fire hydrant", "police van",
    "scooter", "skateboard", "shopping cart", "bench",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_samples", type=int, default=100)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)

    # Load text prototypes (16 classes)
    proto_path = get_nested(cfg, "token.prototype_path",
                            "panoptic/clip_prototypes/text_prototypes_vitl14_224.pt")
    proto_data = torch.load(proto_path, map_location="cpu")
    text_protos = proto_data["embeddings"].to(device)  # (16, 768)

    # Generate extra concept embeddings
    import clip
    clip_model, _ = clip.load("ViT-L/14", device=device, jit=False)
    clip_model.eval()
    extra_tokens = clip.tokenize([f"a {c} in a scene" for c in EXTRA_CONCEPTS]).to(device)
    with torch.no_grad():
        extra_embs = clip_model.encode_text(extra_tokens).float()
        extra_embs = extra_embs / extra_embs.norm(dim=-1, keepdim=True)
    del clip_model

    # All protos: 16 base + extras
    all_protos = torch.cat([text_protos, extra_embs], dim=0)  # (16+N, 768)
    all_names = NUSCENES_CLASS_NAMES + EXTRA_CONCEPTS

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
    state = ckpt["decoder_state"] if "decoder_state" in ckpt else ckpt
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
    print(f"Running margin check on {n_samples} val samples...")

    # Collect margins
    margins_gt_vs_nearest = []  # margin between GT class and nearest non-GT
    margins_gt_vs_novel = []    # margin between GT class and nearest novel concept
    per_class_margins = {i: [] for i in range(16)}

    # Raw backbone margins (for comparison)
    backbone_margins = []
    backbone_per_class = {i: [] for i in range(16)}

    with torch.no_grad():
        for i in range(n_samples):
            batch = val_dataset[i]
            # Add batch dim
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].unsqueeze(0) if batch[k].dim() < 2 else batch[k]

            feats, coords = extract_backbone_features(
                backbone, batch, device, lora_adapters
            )
            outputs = decoder(feats, coords)

            pred_embed = outputs.get("pred_embed", None) or outputs.get("pred_tokens", None)
            pred_masks = outputs["pred_masks"][0].sigmoid() if "pred_masks" in outputs else None  # (V, Q)

            # Raw backbone features for comparison (already 768-d, CLIP-aligned from OpenSeg)
            raw_feats = feats[0]  # (V, 768) — remove batch dim
            V = raw_feats.shape[0]

            gt_classes = batch["gt_classes"].numpy().flatten()

            # For each GT instance, get the best matching query
            for m_idx in range(len(gt_classes)):
                cls = int(gt_classes[m_idx])
                if cls >= 16:
                    continue

                gt_mask = batch["gt_masks"][m_idx] if batch["gt_masks"].dim() == 2 else batch["gt_masks"][0, m_idx]
                if gt_mask.sum() < 50:
                    continue

                gt_mask_dev = gt_mask.float().to(device)

                # --- Raw backbone margin ---
                # Average backbone features over GT mask voxels
                mask_bool = gt_mask.bool()
                # Ensure mask aligns with feature dim
                if mask_bool.shape[0] > V:
                    mask_bool = mask_bool[:V]
                elif mask_bool.shape[0] < V:
                    mask_bool = torch.nn.functional.pad(mask_bool, (0, V - mask_bool.shape[0]))

                if mask_bool.sum() > 0:
                    avg_backbone_feat = raw_feats[mask_bool].float().mean(dim=0, keepdim=True)  # (1, 768)
                    avg_backbone_feat = torch.nn.functional.normalize(avg_backbone_feat, dim=-1)
                    bb_sims = (avg_backbone_feat @ all_protos.T).squeeze(0)  # (16+N,)
                    bb_gt_sim = bb_sims[cls].item()
                    bb_base_sims = bb_sims[:16].clone()
                    bb_base_sims[cls] = -1.0
                    bb_nearest = bb_base_sims.max().item()
                    bb_margin = bb_gt_sim - bb_nearest
                    backbone_margins.append(bb_margin)
                    backbone_per_class[cls].append(bb_margin)

                # --- Decoder query margin ---
                if pred_embed is None or pred_masks is None:
                    continue

                pred_emb = pred_embed[0]  # (Q, 768)
                # Find query with best overlap to this GT mask
                mask_for_overlap = gt_mask_dev
                if mask_for_overlap.shape[0] > pred_masks.shape[0]:
                    mask_for_overlap = mask_for_overlap[:pred_masks.shape[0]]
                elif mask_for_overlap.shape[0] < pred_masks.shape[0]:
                    mask_for_overlap = torch.nn.functional.pad(mask_for_overlap, (0, pred_masks.shape[0] - mask_for_overlap.shape[0]))

                overlaps = (pred_masks * mask_for_overlap.unsqueeze(1)).sum(0)
                best_q = overlaps.argmax().item()

                # Get this query's embedding
                emb = pred_emb[best_q:best_q+1]  # (1, 768)

                # Cosine sim to all protos
                sims = (emb @ all_protos.T).squeeze(0)  # (16+N,)

                gt_sim = sims[cls].item()

                # Nearest non-GT among 16 base classes
                base_sims = sims[:16].clone()
                base_sims[cls] = -1.0
                nearest_base = base_sims.max().item()
                margin = gt_sim - nearest_base
                margins_gt_vs_nearest.append(margin)
                per_class_margins[cls].append(margin)

                # Nearest among extra concepts
                extra_sims = sims[16:]
                nearest_extra = extra_sims.max().item()
                margins_gt_vs_novel.append(gt_sim - nearest_extra)

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{n_samples}] avg_margin={np.mean(margins_gt_vs_nearest):.4f}")

    # Report
    print("\n" + "="*70)
    print("VOXEL MARGIN CHECK RESULTS")
    print("="*70)

    print(f"\n--- RAW BACKBONE (mask-averaged voxel features) ---")
    if backbone_margins:
        print(f"  Mean margin:   {np.mean(backbone_margins):.4f}")
        print(f"  Median margin: {np.median(backbone_margins):.4f}")
        print(f"  Correct (>0):  {100*np.mean(np.array(backbone_margins)>0):.1f}%")
        print(f"  Per-class:")
        for cls in range(16):
            if backbone_per_class[cls]:
                m = np.mean(backbone_per_class[cls])
                n = len(backbone_per_class[cls])
                tag = " [NOVEL]" if cls in {2, 6, 14, 15} else ""
                print(f"    {NUSCENES_CLASS_NAMES[cls]:20s}: margin={m:.4f} (n={n}){tag}")

    print(f"\n--- DECODER QUERY EMBEDDINGS (clip_embed_head output) ---")
    print(f"  Checkpoint: {args.checkpoint}")
    if not margins_gt_vs_nearest:
        print(f"  [SKIPPED] No pred_embed in outputs — model has no clip_embed_head")
    else:
        print(f"\n  Overall GT vs nearest base confuser:")
        print(f"    Mean margin:   {np.mean(margins_gt_vs_nearest):.4f}")
        print(f"    Median margin: {np.median(margins_gt_vs_nearest):.4f}")
        print(f"    Min margin:    {np.min(margins_gt_vs_nearest):.4f}")
        print(f"    Std:           {np.std(margins_gt_vs_nearest):.4f}")
        print(f"    Correct (>0):  {100*np.mean(np.array(margins_gt_vs_nearest)>0):.1f}%")

        print(f"\n  GT vs nearest novel concept (ambulance, police van, etc.):")
        print(f"    Mean margin:   {np.mean(margins_gt_vs_novel):.4f}")
        print(f"    Correct (>0):  {100*np.mean(np.array(margins_gt_vs_novel)>0):.1f}%")

        print(f"\n  Per-class margins (GT vs nearest base confuser):")
        for cls in range(16):
            if per_class_margins[cls]:
                m = np.mean(per_class_margins[cls])
                n = len(per_class_margins[cls])
                tag = " [NOVEL]" if cls in {2, 6, 14, 15} else ""
                print(f"    {NUSCENES_CLASS_NAMES[cls]:20s}: margin={m:.4f} (n={n}){tag}")

    # Cleanup hooks
    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
