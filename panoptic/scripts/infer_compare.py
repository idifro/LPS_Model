#!/usr/bin/env python3
"""
Side-by-side inference for two LPS panoptic checkpoints.
  Run A: checkpoints_e2e_mini_all  (flat 1e-6, all BB, best PQ=0.3512)
  Run B: checkpoints_e2e_mini      (cosine 1e-4, partial BB, best PQ=0.3202)

Usage:
    python panoptic/scripts/infer_compare.py --ckpt_type best --split val
    python panoptic/scripts/infer_compare.py --ckpt_type best --max_samples 5 --vis
"""

import argparse
import os
import sys
import time
import numpy as np
import torch

# ── project root (same as training script) ────────────────────────────────────
PROJECT_ROOT = "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model"
sys.path.insert(0, PROJECT_ROOT)

import MinkowskiEngine as ME
from models.disnet import DisNet, state_dict_remove_moudle
from panoptic.models.mask_decoder import PanopticDecoder
from panoptic.eval.eval_pq import PanopticEval
from panoptic.dataset.nuscenes_panoptic import (
    NuScenesPanoptic, NUM_CLASSES, THING_IDS, STUFF_IDS,
)

# ── checkpoint paths ──────────────────────────────────────────────────────────
CKPT_ROOT = os.path.join(PROJECT_ROOT, "panoptic")

RUNS = {
    "all_bb": {
        "dir":   os.path.join(CKPT_ROOT, "checkpoints_e2e_mini_all"),
        "label": "E2E-AllBB  (flat 1e-6,    PQ=0.3512)",
    },
    "dec_only": {
        "dir":   os.path.join(CKPT_ROOT, "checkpoints_e2e_mini"),
        "label": "E2E-DecOnly (cosine 1e-4, PQ=0.3202)",
    },
}

PALETTE = np.random.RandomState(42).randint(50, 255, (300, 3)).astype(np.uint8)


# ── backbone config (identical to training) ───────────────────────────────────
class Cfg:
    feature_2d_extractor = "openseg"
    arch_3d              = "MinkUNet18A"


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_checkpoint(ckpt_path, device):
    print(f"  loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return ckpt


def build_backbone(ckpt, device):
    backbone = DisNet(Cfg()).to(device)
    sd = ckpt.get("backbone")
    if sd is None:
        print("  [!] 'backbone' key missing in checkpoint")
        return backbone.eval()
    backbone.load_state_dict(state_dict_remove_moudle(sd), strict=True)
    return backbone.eval()


def build_decoder(ckpt, device, cfg_path=None):
    """Build PanopticDecoder using dims stored inside the checkpoint config."""
    ckpt_cfg = ckpt.get("config", {})
    mcfg = ckpt_cfg.get("model", {})

    # fall back to training defaults if config not in ckpt
    decoder = PanopticDecoder(
        backbone_dim      = mcfg.get("backbone_dim",        768),
        hidden_dim        = mcfg.get("hidden_dim",          256),
        num_classes       = mcfg.get("num_classes",          16),
        num_thing_queries = mcfg.get("num_thing_queries",   100),
        num_stuff_classes = mcfg.get("num_stuff_classes",     6),
        nheads            = mcfg.get("nheads",                8),
        dim_feedforward   = mcfg.get("dim_feedforward",    1024),
        num_blocks        = mcfg.get("num_blocks",            3),
        num_levels        = mcfg.get("num_levels",            3),
        dropout           = mcfg.get("dropout",             0.0),
    ).to(device)

    sd = ckpt.get("decoder")
    if sd is None:
        print("  [!] 'decoder' key missing in checkpoint")
        return decoder.eval()
    decoder.load_state_dict(state_dict_remove_moudle(sd), strict=True)
    return decoder.eval()


def load_run(run_key, ckpt_type, device):
    info  = RUNS[run_key]
    path  = os.path.join(info["dir"], f"{ckpt_type}.pth")
    assert os.path.isfile(path), f"Not found: {path}"

    print(f"\n[{run_key}] {info['label']}")
    ckpt     = load_checkpoint(path, device)
    backbone = build_backbone(ckpt, device)
    decoder  = build_decoder(ckpt, device)

    epoch   = ckpt.get("epoch", "?")
    best_pq = ckpt.get("best_pq", float("nan"))
    print(f"  epoch={epoch}  best_pq={best_pq:.4f}")
    print(f"  BB params : {sum(p.numel() for p in backbone.parameters()):,}")
    print(f"  Dec params: {sum(p.numel() for p in decoder.parameters()):,}")

    return backbone, decoder, info["label"]


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction (exact same as training's extract_backbone_features_e2e)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_features(backbone, decoder, batch, device):
    vox_coords = batch["vox_coords"]
    vox_feats  = batch["vox_feats"]
    V = vox_coords.shape[0]

    batch_coords = torch.cat([
        torch.zeros(V, 1, dtype=torch.int32), vox_coords
    ], dim=1)

    sinput = ME.SparseTensor(
        vox_feats.float(),
        coordinates=batch_coords.int(),
        device=device,
    )

    bb_out = backbone(sinput)
    feats  = bb_out.F if hasattr(bb_out, "F") else bb_out   # (V, 768)
    coords = vox_feats.to(device)                            # (V, 4)

    feats_in  = feats.unsqueeze(0)    # (1, V, 768)
    coords_in = coords.unsqueeze(0)   # (1, V, 4)

    outputs = decoder(feats_in, coords_in)

    return {
        "bb_feats":    feats.cpu().numpy(),                  # (V, 768)
        "pred_logits": outputs["pred_logits"][0].cpu(),      # (Q, C+1)
        "pred_masks":  outputs["pred_masks"][0].cpu(),       # (V, Q)
        "vox_coords":  vox_coords.numpy(),                   # (V, 3)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Panoptic inference (same logic as training)
# ─────────────────────────────────────────────────────────────────────────────
def panoptic_inference(pred_logits, pred_masks, num_classes=16, overlap_thr=0.8):
    scores, labels = pred_logits.softmax(-1).max(-1)
    masks = pred_masks.sigmoid()
    V = masks.shape[0]

    keep = labels != num_classes
    if keep.sum() == 0:
        return np.zeros(V, dtype=np.int64), np.zeros(V, dtype=np.int64)

    scores = scores[keep]
    labels = labels[keep]
    masks  = masks[:, keep]

    sorted_idx   = scores.argsort(descending=True)
    sem_pred     = np.zeros(V, dtype=np.int64)
    inst_pred    = np.zeros(V, dtype=np.int64)
    inst_id      = 1
    stuff_assigned = {}

    for idx in sorted_idx:
        label     = labels[idx].item()
        mask_prob = masks[:, idx]
        binary    = mask_prob >= 0.5
        if binary.sum() == 0:
            continue
        orig_area = binary.sum().item()
        assigned  = torch.from_numpy(inst_pred).to(mask_prob.device) > 0
        final_mask = binary & ~assigned
        if final_mask.sum().item() / max(orig_area, 1) < overlap_thr:
            continue
        mask_np = final_mask.cpu().numpy().astype(bool)
        if label in THING_IDS:
            sem_pred[mask_np]  = label
            inst_pred[mask_np] = inst_id
            inst_id += 1
        else:
            if label in stuff_assigned:
                sem_pred[mask_np]  = label
                inst_pred[mask_np] = stuff_assigned[label]
            else:
                sem_pred[mask_np]  = label
                inst_pred[mask_np] = inst_id
                stuff_assigned[label] = inst_id
                inst_id += 1

    return sem_pred, inst_pred


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def run_eval(backbone, decoder, val_loader, device,
             num_classes=16, overlap_thr=0.8, min_points=15):
    evaluator = PanopticEval(n_classes=num_classes, min_points=min_points)

    for i, batch in enumerate(val_loader):
        out = extract_features(backbone, decoder, batch, device)
        sem_pred, inst_pred = panoptic_inference(
            out["pred_logits"], out["pred_masks"],
            num_classes=num_classes, overlap_thr=overlap_thr,
        )

        gt_sem    = batch["gt_sem"].numpy()
        gt_masks  = batch["gt_masks"].numpy()
        gt_classes = batch["gt_classes"].numpy()
        V = gt_sem.shape[0]
        gt_inst = np.zeros(V, dtype=np.int64)
        for m in range(len(gt_classes)):
            gt_inst[gt_masks[m]] = m + 1

        valid = gt_sem != 255
        if valid.sum() == 0:
            continue
        evaluator.addBatch(
            sem_pred[valid], inst_pred[valid],
            gt_sem[valid],   gt_inst[valid],
        )
        print(f"  [{i+1}/{len(val_loader)}] done", end="\r")

    PQ, SQ, RQ, _, _, _ = evaluator.getPQ()
    mIoU, _              = evaluator.getSemIoU()
    return {"PQ": PQ, "SQ": SQ, "RQ": RQ, "mIoU": mIoU}


# ─────────────────────────────────────────────────────────────────────────────
# Saving per-sample outputs
# ─────────────────────────────────────────────────────────────────────────────
def save_sample(out, sem_pred, inst_pred, out_dir, tag, idx, token):
    d = os.path.join(out_dir, tag)
    os.makedirs(d, exist_ok=True)
    base = os.path.join(d, f"{idx:04d}_{token}")
    np.save(f"{base}_coords.npy",    out["vox_coords"])
    np.save(f"{base}_bb_feats.npy",  out["bb_feats"])
    np.save(f"{base}_sem_pred.npy",  sem_pred)
    np.save(f"{base}_inst_pred.npy", inst_pred)


def feat_cosine_sim(fa, fb):
    n = min(len(fa), len(fb))
    fa = fa[:n] / (np.linalg.norm(fa[:n], axis=1, keepdims=True) + 1e-8)
    fb = fb[:n] / (np.linalg.norm(fb[:n], axis=1, keepdims=True) + 1e-8)
    return (fa * fb).sum(axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation (Open3D, optional)
# ─────────────────────────────────────────────────────────────────────────────
def vis_side_by_side(out_a, sem_a, out_b, sem_b, label_a, label_b, idx):
    try:
        import open3d as o3d
    except ImportError:
        print("  [!] open3d not installed, skipping vis")
        return

    def make_pcd(out, sem, offset_x):
        pts = out["vox_coords"].astype(np.float64).copy()
        pts[:, 0] += offset_x
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        cols = PALETTE[sem % len(PALETTE)] / 255.0
        pcd.colors = o3d.utility.Vector3dVector(cols)
        return pcd

    x_range = out_a["vox_coords"][:, 0].ptp() * 1.2
    pcd_a = make_pcd(out_a, sem_a, offset_x=0.0)
    pcd_b = make_pcd(out_b, sem_b, offset_x=x_range)

    o3d.visualization.draw_geometries(
        [pcd_a, pcd_b],
        window_name=f"[{idx}] L:{label_a[:35]}  |  R:{label_b[:35]}",
        width=1600, height=800,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_type",   default="best",
                        choices=["best", "latest"])
    parser.add_argument("--split",       default="val",
                        choices=["train", "val"])
    parser.add_argument("--dataroot",    default="/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes")
    parser.add_argument("--version",     default="v1.0-mini")
    parser.add_argument("--out_dir",     default="infer_compare_out")
    parser.add_argument("--vis",         action="store_true")
    parser.add_argument("--eval",        action="store_true",
                        help="Run full PQ evaluation on val set")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--overlap_thr", type=float, default=0.8)
    parser.add_argument("--min_points",  type=int,   default=15)
    parser.add_argument("--num_classes", type=int,   default=16)
    parser.add_argument("--device",      default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # ── load both runs ────────────────────────────────────────────────────────
    bb_a, dec_a, label_a = load_run("all_bb",   args.ckpt_type, device)
    bb_b, dec_b, label_b = load_run("dec_only", args.ckpt_type, device)

    # ── dataset ───────────────────────────────────────────────────────────────
    from nuscenes.nuscenes import NuScenes as NuScenesDB
    nusc = NuScenesDB(version=args.version, dataroot=args.dataroot, verbose=False)

    ds = NuScenesPanoptic(
        args.dataroot, args.version, split=args.split,
        augment=False, nusc=nusc,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False,
        num_workers=2, collate_fn=lambda x: x[0],
    )
    print(f"\n[dataset] {args.split}: {len(ds)} samples")

    # ── optional full eval ────────────────────────────────────────────────────
    if args.eval:
        print(f"\n{'='*60}")
        print(f"  Full PQ eval on {args.split}")
        print(f"{'='*60}")

        print(f"\n[A] {label_a}")
        t0 = time.time()
        m_a = run_eval(bb_a, dec_a, loader, device,
                       args.num_classes, args.overlap_thr, args.min_points)
        print(f"  PQ={m_a['PQ']:.4f}  SQ={m_a['SQ']:.4f}  "
              f"RQ={m_a['RQ']:.4f}  mIoU={m_a['mIoU']:.4f}  "
              f"({time.time()-t0:.1f}s)")

        print(f"\n[B] {label_b}")
        t0 = time.time()
        m_b = run_eval(bb_b, dec_b, loader, device,
                       args.num_classes, args.overlap_thr, args.min_points)
        print(f"  PQ={m_b['PQ']:.4f}  SQ={m_b['SQ']:.4f}  "
              f"RQ={m_b['RQ']:.4f}  mIoU={m_b['mIoU']:.4f}  "
              f"({time.time()-t0:.1f}s)")

        print(f"\n{'─'*60}")
        print(f"  Delta PQ  : {m_a['PQ']-m_b['PQ']:+.4f}  (A - B)")
        print(f"  Delta SQ  : {m_a['SQ']-m_b['SQ']:+.4f}")
        print(f"  Delta RQ  : {m_a['RQ']-m_b['RQ']:+.4f}")
        print(f"  Delta mIoU: {m_a['mIoU']-m_b['mIoU']:+.4f}")
        print(f"{'─'*60}")
        return

    # ── per-sample inference + save ───────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    cos_sims = []

    for idx, batch in enumerate(loader):
        if args.max_samples and idx >= args.max_samples:
            break

        token = batch.get("scene_token", f"sample_{idx}")

        out_a = extract_features(bb_a, dec_a, batch, device)
        out_b = extract_features(bb_b, dec_b, batch, device)

        sem_a, inst_a = panoptic_inference(
            out_a["pred_logits"], out_a["pred_masks"],
            args.num_classes, args.overlap_thr,
        )
        sem_b, inst_b = panoptic_inference(
            out_b["pred_logits"], out_b["pred_masks"],
            args.num_classes, args.overlap_thr,
        )

        # cosine sim between BB features of the two runs
        cos = feat_cosine_sim(out_a["bb_feats"], out_b["bb_feats"])
        cos_sims.append(cos.mean())
        np.save(os.path.join(args.out_dir, f"{idx:04d}_feat_cos_sim.npy"), cos)

        save_sample(out_a, sem_a, inst_a, args.out_dir, "all_bb",   idx, token)
        save_sample(out_b, sem_b, inst_b, args.out_dir, "dec_only", idx, token)

        print(f"[{idx+1:3d}] token={token}  pts={out_a['vox_coords'].shape[0]}  "
              f"feat_cos={cos.mean():.4f}  "
              f"sem_agree={( sem_a == sem_b).mean()*100:.1f}%")

        if args.vis:
            vis_side_by_side(out_a, sem_a, out_b, sem_b, label_a, label_b, idx)

    if cos_sims:
        print(f"\n[summary]  mean feat cosine sim across samples: "
              f"{np.mean(cos_sims):.4f}")
    print(f"[✓] outputs in {args.out_dir}/all_bb/  and  {args.out_dir}/dec_only/")


if __name__ == "__main__":
    main()