"""Comprehensive panoptic evaluation: 5 modes in a single forward pass.

Modes:
  A. Linear         – pred_logits softmax → 16 known classes
  B. Token-16       – pred_tokens cosine sim → 16 known classes
  C. Ensemble       – alpha sweep blending A+B → 16 known classes
  D. Token-OpenVocab – pred_tokens cosine sim → 22 classes (16 known + 6 novel)
  E. Backbone-Pool  – mean-pool encoder feats per mask → 22 classes (baseline)

Usage:
    python -u panoptic/scripts/evaluate_comprehensive.py \
        --config panoptic/configs/full_nuscenes_token_v2.yaml \
        --checkpoint panoptic/checkpoints_token_full_v2/best.pth \
        --version v1.0-mini
"""
import argparse
import sys
import os
import json
import io
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import MinkowskiEngine as ME
import yaml

sys.path.insert(0, "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene")

from models.disnet import DisNet, state_dict_remove_moudle
from panoptic.models.mask_decoder import PanopticDecoder
from panoptic.eval.eval_pq import PanopticEval
from panoptic.dataset.nuscenes_panoptic import (
    NuScenesPanoptic, NUM_CLASSES, NUSCENES_CLASS_NAMES,
)

# ── Constants ──────────────────────────────────────────────────────────────
NOVEL_CLASS_NAMES = ["pole", "bench", "trash can", "fire hydrant", "tree trunk", "curb"]
ALL_22_CLASS_NAMES = list(NUSCENES_CLASS_NAMES) + NOVEL_CLASS_NAMES
THING_CLASSES_16 = set(range(10))  # 0-9 are things in nuScenes-16

ALL_MODES = ["linear", "token", "ensemble", "token_openvocab", "backbone_openvocab"]


# ── TeeLogger ──────────────────────────────────────────────────────────────
class TeeLogger:
    """Duplicate stdout to a log file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w")

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# ── Model loading ──────────────────────────────────────────────────────────
def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_backbone(ckpt_path, device):
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


def extract_backbone_features(backbone, batch, device):
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
        output = backbone(sinput)
    feats = output.F if hasattr(output, "F") else output
    coords_xyz = vox_feats.to(device)
    return feats.unsqueeze(0), coords_xyz.unsqueeze(0)


def load_decoder_flexible(cfg, ckpt_path, device):
    """Load decoder, auto-detect token_head from checkpoint keys."""
    mcfg = cfg["model"]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    decoder_state = ckpt["decoder"]
    has_token_head = any("token_head" in k for k in decoder_state.keys())

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
        use_token_head=has_token_head,
        token_dim=mcfg.get("token_dim", 768),
    ).to(device)
    decoder.load_state_dict(decoder_state, strict=True)
    decoder.eval()
    return decoder, has_token_head


# ── Prototype loading/generation ───────────────────────────────────────────
def load_or_generate_22_prototypes(known_proto_path, novel_proto_path, device):
    """Load 16-class protos, extend to 22 with novel classes."""
    known_data = torch.load(known_proto_path, map_location="cpu")
    known_emb = known_data["embeddings"].float()  # (16, 768)

    if novel_proto_path and os.path.exists(novel_proto_path):
        novel_data = torch.load(novel_proto_path, map_location="cpu")
        all_emb = novel_data["embeddings"].float()  # (22, 768)
        print(f"Loaded 22-class prototypes from {novel_proto_path}", flush=True)
        return known_emb.to(device), all_emb.to(device)

    # Generate novel prototypes on the fly
    print("Generating novel class prototypes with CLIP...", flush=True)
    try:
        import clip
        model, _ = clip.load("ViT-L/14@336px", device=device, jit=False)
        model.eval()
        template = known_data.get("prompt_template", "a {} in a scene")
        prompts = [template.format(name) for name in NOVEL_CLASS_NAMES]
        tokens = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            novel_feats = model.encode_text(tokens)
            novel_feats = novel_feats / novel_feats.norm(dim=-1, keepdim=True)
        novel_feats = novel_feats.float().cpu()
        all_emb = torch.cat([known_emb, novel_feats], dim=0)  # (22, 768)
        # Save for next time
        save_path = novel_proto_path or known_proto_path.replace(".pt", "_22class.pt")
        torch.save({
            "embeddings": all_emb,
            "class_names": ALL_22_CLASS_NAMES,
            "prompt_template": template,
        }, save_path)
        print(f"Saved 22-class prototypes to {save_path}", flush=True)
        del model
        torch.cuda.empty_cache()
        return known_emb.to(device), all_emb.to(device)
    except ImportError:
        print("ERROR: CLIP not installed. Cannot generate novel prototypes.", flush=True)
        print("Install with: pip install git+https://github.com/openai/CLIP.git", flush=True)
        return known_emb.to(device), None


# ── Inference functions ────────────────────────────────────────────────────
def _greedy_assign(scores, labels, masks, V, overlap_thr, thing_set=None):
    """Greedy panoptic assignment. thing_set defaults to 0-9 for nuScenes-16."""
    if thing_set is None:
        thing_set = THING_CLASSES_16
    sorted_idx = scores.argsort(descending=True)
    sem_pred = np.full(V, 0, dtype=np.int64)
    inst_pred = np.zeros(V, dtype=np.int64)
    inst_id = 1
    stuff_assigned = {}

    for idx in sorted_idx:
        label = labels[idx].item()
        mask_prob = masks[:, idx]
        binary = mask_prob >= 0.5
        if binary.sum() == 0:
            continue
        orig_area = binary.sum().item()
        assigned = torch.from_numpy(inst_pred).to(mask_prob.device) > 0
        final_mask = binary & ~assigned
        if final_mask.sum().item() / max(orig_area, 1) < overlap_thr:
            continue
        mask_np = final_mask.cpu().numpy().astype(bool)
        if label in thing_set:
            sem_pred[mask_np] = label
            inst_pred[mask_np] = inst_id
            inst_id += 1
        else:
            if label in stuff_assigned:
                sem_pred[mask_np] = label
                inst_pred[mask_np] = stuff_assigned[label]
            else:
                sem_pred[mask_np] = label
                inst_pred[mask_np] = inst_id
                stuff_assigned[label] = inst_id
                inst_id += 1
    return sem_pred, inst_pred


def panoptic_inference_linear(pred_logits, pred_masks, num_classes=16, overlap_thr=0.8):
    scores, labels = pred_logits.softmax(-1).max(-1)
    masks = pred_masks.sigmoid()
    V = masks.shape[0]
    keep = labels != num_classes
    if keep.sum() == 0:
        return np.zeros(V, dtype=np.int64), np.zeros(V, dtype=np.int64)
    return _greedy_assign(scores[keep], labels[keep], masks[:, keep], V, overlap_thr)


def panoptic_inference_token(pred_tokens, pred_masks, text_prototypes, overlap_thr=0.8):
    """Token cosine sim against prototypes (works for 16 or 22 classes)."""
    cos_sim = torch.mm(pred_tokens, text_prototypes.t())  # (Q, C)
    scores, labels = cos_sim.max(-1)
    masks = pred_masks.sigmoid()
    V = masks.shape[0]
    keep = scores > 0.1
    if keep.sum() == 0:
        return np.zeros(V, dtype=np.int64), np.zeros(V, dtype=np.int64)
    # For 22-class: novel classes (16-21) treated as things
    num_protos = text_prototypes.shape[0]
    thing_set = THING_CLASSES_16 | set(range(16, num_protos)) if num_protos > 16 else THING_CLASSES_16
    return _greedy_assign(scores[keep], labels[keep], masks[:, keep], V, overlap_thr, thing_set)


def panoptic_inference_ensemble(pred_logits, pred_tokens, pred_masks, text_prototypes,
                                 num_classes=16, overlap_thr=0.8, alpha=0.5):
    linear_probs = pred_logits[:, :num_classes].softmax(-1)
    cos_sim = torch.mm(pred_tokens, text_prototypes.t())
    cos_sim_norm = (cos_sim + 1) / 2
    combined = alpha * linear_probs + (1 - alpha) * cos_sim_norm
    scores, labels = combined.max(-1)
    masks = pred_masks.sigmoid()
    V = masks.shape[0]
    keep = scores > 0.1
    if keep.sum() == 0:
        return np.zeros(V, dtype=np.int64), np.zeros(V, dtype=np.int64)
    return _greedy_assign(scores[keep], labels[keep], masks[:, keep], V, overlap_thr)


def panoptic_inference_backbone_pool(backbone_feats, pred_masks, text_protos_22, overlap_thr=0.8):
    """Mean-pool raw backbone features per mask, cosine sim to 22 prototypes."""
    masks = pred_masks.sigmoid()  # (V, Q)
    V, Q = masks.shape
    pooled = []
    valid_queries = []
    for q in range(Q):
        binary = masks[:, q] >= 0.5
        if binary.sum() < 10:
            continue
        feat_pool = backbone_feats[binary].mean(dim=0)  # (768,)
        feat_pool = F.normalize(feat_pool, dim=0)
        pooled.append(feat_pool)
        valid_queries.append(q)

    if len(pooled) == 0:
        return np.zeros(V, dtype=np.int64), np.zeros(V, dtype=np.int64)

    pooled = torch.stack(pooled)  # (Qv, 768)
    cos_sim = torch.mm(pooled, text_protos_22.t())  # (Qv, 22)
    scores, labels = cos_sim.max(-1)

    # Rebuild masks for valid queries only
    valid_masks = masks[:, valid_queries]  # (V, Qv)
    num_protos = text_protos_22.shape[0]
    thing_set = THING_CLASSES_16 | set(range(16, num_protos))
    return _greedy_assign(scores, labels, valid_masks, V, overlap_thr, thing_set)


# ── Novel class stats ──────────────────────────────────────────────────────
def count_novel_detections(sem_pred, num_known=16):
    """Count voxels and instances predicted as novel classes."""
    stats = {}
    for i, name in enumerate(NOVEL_CLASS_NAMES):
        cls_id = num_known + i
        mask = sem_pred == cls_id
        stats[name] = {"voxels": int(mask.sum()), "class_id": cls_id}
    return stats


def remap_novel_to_ignore(sem_pred, inst_pred, num_known=16):
    """For known-class PQ eval, set novel predictions to 0 (background/ignore)."""
    sem_out = sem_pred.copy()
    inst_out = inst_pred.copy()
    novel_mask = sem_out >= num_known
    sem_out[novel_mask] = 0
    inst_out[novel_mask] = 0
    return sem_out, inst_out


# ── Table formatting ───────────────────────────────────────────────────────
def format_table(pq_all, sq_all, rq_all, iou_all, class_names):
    lines = []
    lines.append(f"  {'Class':<25s} {'PQ':>8s} {'SQ':>8s} {'RQ':>8s} {'IoU':>8s}")
    lines.append(f"  {'-'*57}")
    for c in range(len(class_names)):
        pq = pq_all[c] if c < len(pq_all) else 0.0
        sq = sq_all[c] if c < len(sq_all) else 0.0
        rq = rq_all[c] if c < len(rq_all) else 0.0
        iou = iou_all[c] if c < len(iou_all) else 0.0
        lines.append(f"  {class_names[c]:<25s} {pq:8.4f} {sq:8.4f} {rq:8.4f} {iou:8.4f}")
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Comprehensive panoptic evaluation (5 modes)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--version", default=None, help="Override data version (v1.0-mini or v1.0-trainval)")
    parser.add_argument("--dataroot", default=None, help="Override data root path")
    parser.add_argument("--output_dir", default="panoptic/eval_results/comprehensive/")
    parser.add_argument("--modes", default=",".join(ALL_MODES),
                        help="Comma-separated modes: linear,token,ensemble,token_openvocab,backbone_openvocab")
    parser.add_argument("--prototype_path", default=None, help="16-class prototype path (default from config)")
    parser.add_argument("--novel_prototype_path", default=None, help="22-class prototype path (auto-generated if missing)")
    parser.add_argument("--ensemble_alphas", default="0.0,0.3,0.5,0.7,1.0")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modes = [m.strip() for m in args.modes.split(",")]
    alphas = [float(a) for a in args.ensemble_alphas.split(",")]

    # Determine version and dataroot
    version = args.version or cfg["data"]["version"]
    if args.dataroot:
        cfg["data"]["dataroot"] = args.dataroot

    # Create output directory
    ckpt_stem = Path(args.checkpoint).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_tag = version.replace("v1.0-", "")
    run_name = f"{ckpt_stem}_{version_tag}_{timestamp}"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Start logging
    tee = TeeLogger(str(out_dir / "eval_log.txt"))
    sys.stdout = tee

    print(f"{'='*70}", flush=True)
    print(f"  Comprehensive Panoptic Evaluation", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Config:     {args.config}", flush=True)
    print(f"  Checkpoint: {args.checkpoint}", flush=True)
    print(f"  Version:    {version}", flush=True)
    print(f"  Modes:      {modes}", flush=True)
    print(f"  Output:     {out_dir}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # ── Load data ──
    dcfg = cfg["data"]
    from nuscenes.nuscenes import NuScenes as NuSc
    print("Loading NuScenes database...", flush=True)
    nusc = NuSc(version=version, dataroot=dcfg["dataroot"], verbose=False)
    val_ds = NuScenesPanoptic(
        dcfg["dataroot"], version, split="val",
        voxel_size=dcfg["voxel_size"],
        point_range=np.array(dcfg["point_range"]),
        augment=False, aug_cfg=None, nusc=nusc,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
        collate_fn=lambda x: x[0],
    )
    print(f"Val: {len(val_ds)} samples\n", flush=True)

    # ── Load models ──
    backbone = load_backbone(cfg["model"]["backbone_ckpt"], device)
    decoder, has_token_head = load_decoder_flexible(cfg, args.checkpoint, device)
    print(f"Token head detected: {has_token_head}\n", flush=True)

    # ── Filter modes based on capabilities ──
    token_modes = {"token", "ensemble", "token_openvocab"}
    if not has_token_head:
        skipped = [m for m in modes if m in token_modes]
        if skipped:
            print(f"WARNING: No token_head — skipping modes: {skipped}\n", flush=True)
        modes = [m for m in modes if m not in token_modes]

    # ── Load prototypes ──
    token_cfg = cfg.get("token", {})
    proto_path = args.prototype_path or token_cfg.get(
        "prototype_path", "panoptic/clip_prototypes/text_prototypes_vitl14_336.pt"
    )
    text_protos_16 = None
    text_protos_22 = None

    needs_16 = any(m in modes for m in ["token", "ensemble"])
    needs_22 = any(m in modes for m in ["token_openvocab", "backbone_openvocab"])

    if needs_16 or needs_22:
        if needs_22:
            text_protos_16, text_protos_22 = load_or_generate_22_prototypes(
                proto_path, args.novel_prototype_path, device
            )
            if text_protos_22 is None:
                print("WARNING: Could not load/generate 22-class prototypes, skipping open-vocab modes\n", flush=True)
                modes = [m for m in modes if m not in {"token_openvocab", "backbone_openvocab"}]
        else:
            proto_data = torch.load(proto_path, map_location="cpu")
            text_protos_16 = proto_data["embeddings"].float().to(device)

        if text_protos_16 is not None:
            print(f"16-class prototypes: {text_protos_16.shape}", flush=True)
        if text_protos_22 is not None:
            print(f"22-class prototypes: {text_protos_22.shape}", flush=True)
        print(flush=True)

    # ── Setup evaluators ──
    num_classes = cfg["model"]["num_classes"]
    overlap_thr = cfg["inference"]["overlap_thr"]
    min_points = cfg["inference"].get("min_points", 50)

    evaluators = {}
    if "linear" in modes:
        evaluators["linear"] = PanopticEval(n_classes=num_classes, min_points=min_points)
    if "token" in modes:
        evaluators["token"] = PanopticEval(n_classes=num_classes, min_points=min_points)
    if "ensemble" in modes:
        for a in alphas:
            evaluators[f"ensemble_a{a:.1f}"] = PanopticEval(n_classes=num_classes, min_points=min_points)
    if "token_openvocab" in modes:
        evaluators["token_openvocab_known"] = PanopticEval(n_classes=num_classes, min_points=min_points)
    if "backbone_openvocab" in modes:
        evaluators["backbone_openvocab_known"] = PanopticEval(n_classes=num_classes, min_points=min_points)

    novel_stats_agg = {
        "token_openvocab": {name: {"voxels": 0} for name in NOVEL_CLASS_NAMES},
        "backbone_openvocab": {name: {"voxels": 0} for name in NOVEL_CLASS_NAMES},
    }

    # ── Evaluation loop ──
    print(f"Evaluating {len(val_ds)} samples across modes: {modes}\n", flush=True)
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            feats, coords = extract_backbone_features(backbone, batch, device)
            outputs = decoder(feats, coords)

            pred_logits = outputs["pred_logits"][0]   # (Q, C+1)
            pred_masks = outputs["pred_masks"][0]     # (V, Q)
            pred_tokens = outputs.get("pred_tokens", [None])[0]  # (Q, 768) or None

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

            # Move to CPU for inference
            pl = pred_logits.cpu()
            pm = pred_masks.cpu()
            pt = pred_tokens.cpu() if pred_tokens is not None else None

            # A: Linear
            if "linear" in modes:
                sem_a, inst_a = panoptic_inference_linear(pl, pm, num_classes, overlap_thr)
                evaluators["linear"].addBatch(sem_a[valid], inst_a[valid], gt_sem[valid], gt_inst[valid])

            # B: Token-16
            if "token" in modes and pt is not None:
                sem_b, inst_b = panoptic_inference_token(pt, pm, text_protos_16.cpu(), overlap_thr)
                evaluators["token"].addBatch(sem_b[valid], inst_b[valid], gt_sem[valid], gt_inst[valid])

            # C: Ensemble alpha sweep
            if "ensemble" in modes and pt is not None:
                for a in alphas:
                    sem_c, inst_c = panoptic_inference_ensemble(
                        pl, pt, pm, text_protos_16.cpu(), num_classes, overlap_thr, alpha=a
                    )
                    evaluators[f"ensemble_a{a:.1f}"].addBatch(
                        sem_c[valid], inst_c[valid], gt_sem[valid], gt_inst[valid]
                    )

            # D: Token-OpenVocab (22 classes) — MAIN OPEN-VOCAB TEST
            if "token_openvocab" in modes and pt is not None:
                sem_d, inst_d = panoptic_inference_token(pt, pm, text_protos_22.cpu(), overlap_thr)
                # Track novel detections
                stats_d = count_novel_detections(sem_d)
                for name, s in stats_d.items():
                    novel_stats_agg["token_openvocab"][name]["voxels"] += s["voxels"]
                # Remap novel → ignore for known-class PQ
                sem_d_known, inst_d_known = remap_novel_to_ignore(sem_d, inst_d, num_classes)
                evaluators["token_openvocab_known"].addBatch(
                    sem_d_known[valid], inst_d_known[valid], gt_sem[valid], gt_inst[valid]
                )

            # E: Backbone pooling (22 classes) — baseline
            if "backbone_openvocab" in modes:
                backbone_feats_v = feats[0]  # (V, 768) raw backbone
                sem_e, inst_e = panoptic_inference_backbone_pool(
                    backbone_feats_v, pm.to(device), text_protos_22, overlap_thr
                )
                stats_e = count_novel_detections(sem_e)
                for name, s in stats_e.items():
                    novel_stats_agg["backbone_openvocab"][name]["voxels"] += s["voxels"]
                sem_e_known, inst_e_known = remap_novel_to_ignore(sem_e, inst_e, num_classes)
                evaluators["backbone_openvocab_known"].addBatch(
                    sem_e_known[valid], inst_e_known[valid], gt_sem[valid], gt_inst[valid]
                )

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(val_ds)}]", flush=True)

    # ── Collect results ──
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
            "per_class_sq": [float(x) for x in sq_all],
            "per_class_rq": [float(x) for x in rq_all],
            "per_class_iou": [float(x) for x in iou_all],
        }

        # Determine novel voxels for summary
        novel_total = "N/A"
        if "openvocab" in name:
            mode_key = name.replace("_known", "")
            if mode_key in novel_stats_agg:
                novel_total = sum(s["voxels"] for s in novel_stats_agg[mode_key].values())

        summary_rows.append((name, PQ, SQ, RQ, iou_mean, novel_total))

        # Print per-class table
        print(f"  Mode: {name}", flush=True)
        print(f"  PQ={PQ:.4f}  SQ={SQ:.4f}  RQ={RQ:.4f}  mIoU={iou_mean:.4f}", flush=True)
        table = format_table(pq_all, sq_all, rq_all, iou_all, NUSCENES_CLASS_NAMES)
        print(table, flush=True)
        print(flush=True)

        # Save per-class table file
        table_name = name.replace(".", "_")
        with open(out_dir / f"per_class_{table_name}.txt", "w") as f:
            f.write(f"Mode: {name}\n")
            f.write(f"PQ={PQ:.4f}  SQ={SQ:.4f}  RQ={RQ:.4f}  mIoU={iou_mean:.4f}\n\n")
            f.write(table + "\n")

    # ── Summary table ──
    print(f"\n{'='*70}", flush=True)
    print(f"  SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  {'Mode':<30s} {'PQ':>8s} {'SQ':>8s} {'RQ':>8s} {'mIoU':>8s} {'Novel Vox':>12s}", flush=True)
    print(f"  {'-'*74}", flush=True)
    for name, pq, sq, rq, iou, novel in summary_rows:
        nv_str = f"{novel:>12,d}" if isinstance(novel, int) else f"{'N/A':>12s}"
        print(f"  {name:<30s} {pq:8.4f} {sq:8.4f} {rq:8.4f} {iou:8.4f} {nv_str}", flush=True)

    # Find best ensemble alpha
    if "ensemble" in modes:
        ensemble_keys = [k for k in results if k.startswith("ensemble_a")]
        if ensemble_keys:
            best_ens = max(ensemble_keys, key=lambda k: results[k]["PQ"])
            print(f"\n  Best ensemble: {best_ens} (PQ={results[best_ens]['PQ']:.4f})", flush=True)
            results["best_ensemble"] = {"name": best_ens, **results[best_ens]}

    # ── Novel detection comparison (D vs E) ──
    if any(m in modes for m in ["token_openvocab", "backbone_openvocab"]):
        print(f"\n{'='*70}", flush=True)
        print(f"  NOVEL CLASS DETECTIONS (D: Token-OpenVocab vs E: Backbone-Pool)", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"  {'Novel Class':<20s} {'Token-OV':>12s} {'Backbone-Pool':>14s} {'Winner':>10s}", flush=True)
        print(f"  {'-'*60}", flush=True)
        for name in NOVEL_CLASS_NAMES:
            tv = novel_stats_agg["token_openvocab"].get(name, {}).get("voxels", 0) if "token_openvocab" in modes else 0
            bv = novel_stats_agg["backbone_openvocab"].get(name, {}).get("voxels", 0) if "backbone_openvocab" in modes else 0
            winner = "Token" if tv > bv else ("Backbone" if bv > tv else "Tie")
            print(f"  {name:<20s} {tv:>12,d} {bv:>14,d} {winner:>10s}", flush=True)

    # ── Save results ──
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results.json to {out_dir}", flush=True)

    # Save novel detections
    if any(m in modes for m in ["token_openvocab", "backbone_openvocab"]):
        with open(out_dir / "novel_detections.json", "w") as f:
            json.dump(novel_stats_agg, f, indent=2)
        print(f"Saved novel_detections.json to {out_dir}", flush=True)

    print(f"\nDone. All outputs in: {out_dir}", flush=True)

    # Restore stdout
    sys.stdout = tee.terminal
    tee.close()


if __name__ == "__main__":
    main()
