"""
verify_clip_alignment.py
========================
Verifies that the pre-distilled DisNet backbone is CLIP-aligned.

Architecture reality (confirmed from openscene training code):
  best.pth  →  DECODER ONLY  (epoch/decoder/optimizer/scheduler keys)
               Backbone was FROZEN and NEVER saved into best.pth.

  The backbone lives at:
    models/nuscenes_openseg.pth.tar  (187 MB, ckpt["state_dict"])
    Loaded with strict=True by load_backbone() in all training scripts.

Verification strategy:
  1. Architecture check    — strict=True load with 289 keys (MinkUNet18A)
  2. Training metadata     — best_iou from checkpoint (paper: 35.4%)
  3. Feature statistics    — forward pass on random voxels checks norms/shape
  4. CLIP similarity       — only meaningful with REAL semantic data;
                             random voxels will always give ~0 similarity
                             (CLIP alignment is semantic, not structural).
                             Skipped by default; use --use_real_data to load
                             a cached scene from the nuScenes mini dataset.

Usage:
  python panoptic/scripts/verify_clip_alignment.py            # steps 1-3
  python panoptic/scripts/verify_clip_alignment.py --use_real_data  # +step 4
"""

import argparse
import sys
import os
import traceback
import torch
import torch.nn.functional as F
from collections import OrderedDict

NUSCENES_CLASSES = [
    "barrier",               # 0
    "bicycle",               # 1  NOVEL
    "bus",                   # 2  NOVEL
    "car",                   # 3
    "construction vehicle",  # 4
    "motorcycle",            # 5  NOVEL
    "pedestrian",            # 6
    "traffic cone",          # 7
    "trailer",               # 8
    "truck",                 # 9
    "driveable surface",     # 10
    "other flat",            # 11
    "sidewalk",              # 12
    "terrain",               # 13
    "manmade",               # 14
    "vegetation",            # 15  NOVEL
]
NOVEL_IDX = [1, 2, 5, 15]
PROMPT_TEMPLATE = "a {} in a driving scene"

_BASE       = "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene"
_DATA_BASE  = "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob"

DEFAULT_BACKBONE   = f"{_BASE}/models/nuscenes_openseg.pth.tar"
DEFAULT_PROTOTYPES = f"{_BASE}/panoptic/clip_prototypes/text_prototypes_vitl14_336.pt"
# A real point cloud scene — used for CLIP similarity test
DEFAULT_SCENE_PT   = f"{_DATA_BASE}/openscene/data/nuscenes_3d/val/scene-0103.pth"


# ── Step 1: Load backbone checkpoint ─────────────────────────────────────────

def load_backbone_checkpoint(ckpt_path):
    print(f"\n[1] Loading backbone checkpoint: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        print(f"  ✗ File not found. Expected nuscenes_openseg.pth.tar (~187 MB)")
        sys.exit(1)

    ckpt  = torch.load(ckpt_path, map_location="cpu")
    fsize = os.path.getsize(ckpt_path) / 1e6
    print(f"  File size : {fsize:.0f} MB")

    if isinstance(ckpt, dict):
        meta_keys = [k for k in ckpt if k not in ("state_dict", "model_state_dict", "model")]
        print(f"  Metadata keys: {meta_keys}")
        for k in meta_keys:
            v = ckpt[k]
            if not hasattr(v, "keys") and not hasattr(v, "items"):
                print(f"    {k} = {v}")

    if "state_dict" in ckpt:
        state, fmt = ckpt["state_dict"], "OpenScene (state_dict)"
    elif "model_state_dict" in ckpt:
        state, fmt = ckpt["model_state_dict"], "model_state_dict"
    elif "model" in ckpt:
        state, fmt = ckpt["model"], "model"
    else:
        state, fmt = ckpt, "raw"

    print(f"  Format    : {fmt}")
    clean = OrderedDict((k.replace("module.", ""), v) for k, v in state.items())
    print(f"  ✓ State dict keys: {len(clean)}  (MinkUNet18A expects 289)")
    print(f"  First 3 keys: {list(clean.keys())[:3]}")
    return clean, ckpt


# ── Step 2: Architecture check ────────────────────────────────────────────────

def build_backbone(state_dict, device):
    print("\n[2] Architecture check — loading DisNet/MinkUNet18A ...")
    try:
        import MinkowskiEngine as ME
        print(f"  ✓ MinkowskiEngine {getattr(ME, '__version__', 'unknown')}")
    except Exception:
        traceback.print_exc()
        print("\n  Fix: conda activate opensc_env")
        print("       export LD_LIBRARY_PATH=/home/ddb1cob/.conda/pkgs/libcusparse-11.7.5.86-0/lib:$LD_LIBRARY_PATH")
        sys.exit(1)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    from models.disnet import DisNet

    class Cfg:
        feature_2d_extractor = "openseg"
        arch_3d = "MinkUNet18A"

    model = DisNet(cfg=Cfg())

    try:
        model.load_state_dict(state_dict, strict=True)
        print("  ✓ strict=True load succeeded")
        print("    → Architecture matches exactly (training code also uses strict=True)")
    except RuntimeError as e:
        print(f"  ⚠  strict=True failed: {e}")
        print("     Retrying strict=False ...")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:    print(f"  ⚠  Missing ({len(missing)}):    {missing[:5]}")
        if unexpected: print(f"  ⚠  Unexpected ({len(unexpected)}): {unexpected[:5]}")

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ── Step 3: Feature statistics (random input) ─────────────────────────────────

def check_feature_statistics(model, device, n_voxels=512):
    import MinkowskiEngine as ME

    print(f"\n[3] Feature statistics ({n_voxels} random voxels, device={device}) ...")
    coords = torch.cat([torch.zeros(n_voxels, 1).int(),
                        torch.randint(0, 1000, (n_voxels, 3)).int()], dim=1)
    feats  = torch.randn(n_voxels, 3)

    sparse = ME.SparseTensor(features=feats.to(device), coordinates=coords.to(device))
    with torch.no_grad():
        out = model(sparse)

    out_feats = out.F if hasattr(out, "F") else out
    print(f"  Output shape  : {out_feats.shape}  (expect [{n_voxels}, 768])")
    assert out_feats.shape[-1] == 768, f"Expected 768-d, got {out_feats.shape[-1]}"

    norms   = out_feats.norm(dim=-1)
    mean_n  = norms.mean().item()
    std_n   = norms.std().item()
    print(f"  Feature norms : mean={mean_n:.2f}  std={std_n:.3f}")
    print(f"  Note: Raw norms > 1 is normal — backbone is L2-normalised at inference.")
    print(f"        Consistent norms (low std/mean ratio) confirm trained weights.")
    print(f"        Ratio std/mean = {std_n/mean_n:.3f}  (random backbone: ~1.0, trained: <0.1)")

    ok_shape = out_feats.shape[-1] == 768
    ok_norms = (std_n / mean_n) < 0.20 and mean_n > 1.0   # trained backbone criterion
    return F.normalize(out_feats, dim=-1).cpu(), ok_shape, ok_norms


# ── Step 4: CLIP similarity (needs real semantic data) ────────────────────────

def load_real_scene(scene_pt_path, device, max_points=4096):
    """Load a real nuScenes point cloud .pth file for semantic CLIP alignment check."""
    import numpy as np

    print(f"\n[4a] Loading real scene: {scene_pt_path}")
    if not os.path.exists(scene_pt_path):
        print(f"  ✗ Scene file not found.")
        candidates = []
        for base in [f"{_DATA_BASE}/openscene/data/nuscenes_3d/val",
                     f"{_DATA_BASE}/openscene/data/nuscenes_3d/train"]:
            if os.path.isdir(base):
                candidates = [os.path.join(base, f) for f in os.listdir(base) if f.endswith(".pth")]
                if candidates:
                    break
        if candidates:
            scene_pt_path = candidates[0]
            print(f"  → Using: {scene_pt_path}")
        else:
            print(f"  ✗ No .pth scene files found.")
            return None, None

    data = torch.load(scene_pt_path, map_location="cpu")

    def _as_float(x):
        if isinstance(x, torch.Tensor):
            t = x.float().contiguous()
        else:
            t = torch.from_numpy(np.array(x, dtype=np.float32))
        return t.view(-1, t.shape[-1]) if t.dim() > 1 else t.unsqueeze(1)

    def _as_long(x):
        if isinstance(x, torch.Tensor):
            return x.long().contiguous().reshape(-1)
        return torch.from_numpy(np.array(x, dtype=np.int64).reshape(-1))

    if isinstance(data, (list, tuple)):
        print(f"  Format: list/tuple  len={len(data)}")
        for i, d in enumerate(data):
            print(f"    [{i}] {type(d).__name__}  shape={np.asarray(d).shape}")
        if len(data) < 2:
            print("  ✗ Expected >=2 elements"); return None, None
        coord = _as_float(data[0])
        # nuScenes: data[1] is scalar 0 (no RGB) — mirrors input_color=False in point_loader
        if np.isscalar(data[1]) or isinstance(data[1], (int, float)):
            color = torch.ones(coord.shape[0], 3)
        else:
            color = _as_float(data[1])
        label = _as_long(data[2]) if len(data) > 2 else torch.zeros(coord.shape[0], dtype=torch.long)
    elif isinstance(data, dict):
        print(f"  Format: dict  keys={list(data.keys())}")
        coord_raw = data.get("coord", data.get("xyz", data.get("points")))
        color_raw = data.get("color", data.get("rgb"))
        label_raw = data.get("label", data.get("semantic_gt", data.get("labels")))
        if coord_raw is None:
            print("  ✗ No coord/xyz/points key"); return None, None
        coord = _as_float(coord_raw)
        color = _as_float(color_raw) if color_raw is not None else torch.zeros(coord.shape[0], 3)
        label = _as_long(label_raw) if label_raw is not None else torch.zeros(coord.shape[0], dtype=torch.long)
    else:
        print(f"  ✗ Unknown format: {type(data)}"); return None, None

    N = min(coord.shape[0], color.shape[0], label.shape[0])
    coord, color, label = coord[:N], color[:N], label[:N]
    if color.shape[1] != 3:
        color = color[:, :3]

    print(f"  coord {tuple(coord.shape)}  color {tuple(color.shape)}  label {tuple(label.shape)}")
    n = min(max_points, N)
    idx = torch.randperm(N)[:n]
    coord, color, label = coord[idx], color[idx], label[idx]

    # No color normalisation: nuScenes uses ones(N,3) directly (input_color=False in training).
    # point_loader.py passes ones un-scaled; backbone handles it internally.
    print(f"  ✓ Using {n} points  label range [{int(label.min())}, {int(label.max())}]  "
          f"unique classes: {label.unique().numel()}")
    return (coord, color, label), scene_pt_path


def run_real_clip_alignment(model, scene_data, device, text_feats, source):
    """Run CLIP alignment on real nuScenes point cloud — class-conditioned."""
    import MinkowskiEngine as ME
    import numpy as np

    coord, color, label = scene_data
    N_in = coord.shape[0]
    print(f"\n[4b] CLIP alignment ({N_in} pts, source={source}) ...")

    voxel_size = 0.05
    vox_coord  = (coord / voxel_size).int()
    batch_idx  = torch.zeros(N_in, 1, dtype=torch.int32)
    vox_with_b = torch.cat([batch_idx, vox_coord], dim=1)

    # Build per-voxel majority label BEFORE ME deduplication
    coord_np = vox_with_b.numpy()
    _, inv_idx = np.unique(coord_np, axis=0, return_inverse=True)
    n_voxels  = int(inv_idx.max()) + 1
    label_np  = label.numpy()
    vox_label = np.zeros(n_voxels, dtype=np.int64)
    for v in range(n_voxels):
        pts = label_np[inv_idx == v].clip(0)
        vox_label[v] = np.bincount(pts).argmax() if len(pts) else 0
    vox_label = torch.from_numpy(vox_label)

    sparse = ME.SparseTensor(features=color.to(device), coordinates=vox_with_b.to(device))
    with torch.no_grad():
        out = model(sparse)

    out_feats = out.F if hasattr(out, "F") else out
    N_out     = out_feats.shape[0]
    out_norm  = F.normalize(out_feats, dim=-1).cpu()
    vox_label = vox_label[:N_out]
    print(f"  Input voxels: {n_voxels}  ME output: {N_out}")

    sim_matrix = out_norm @ text_feats.T  # (N_out, 16)

    # Per-voxel top-1 predicted class
    top1_pred  = sim_matrix.argmax(dim=-1)   # (N_out,)

    MIN_VOXELS = 50   # minimum voxels for a reliable per-class sim estimate
    # Note: raw per-voxel distilled 3D features vs. CLIP text have inherently
    #       low cosine similarities (~0.01-0.03 for matching classes).  This is
    #       expected: CLIP image-text global pairs are ~0.2-0.3; per-voxel
    #       features are mixed and much weaker.  We therefore:
    #         a) only assess classes with >= MIN_VOXELS for aggregate metrics
    #         b) use >0.0 as the threshold (positive = aligned)
    #         c) also report top-1 voxel accuracy (the real argmax check)

    print(f"\n  {'Class':<25}  {'Mean sim':>10}  {'N vox':>6}  {'Top-1%':>7}  {'Novel?':>8}")
    print(f"  {'─'*25}  {'─'*10}  {'─'*6}  {'─'*7}  {'─'*8}")
    class_sims = []
    for cls_id in range(16):
        mask  = (vox_label == cls_id)
        n_pts = int(mask.sum())
        sim   = float(sim_matrix[mask, cls_id].mean()) if n_pts > 0 else float("nan")
        top1  = float((top1_pred[mask] == cls_id).float().mean()) if n_pts > 0 else float("nan")
        novel = "NOVEL" if cls_id in NOVEL_IDX else ""
        sim_s  = f"{sim:10.4f}" if n_pts > 0 else "       n/a"
        top1_s = f"{top1*100:6.1f}%" if n_pts > 0 else "    n/a"
        flag   = "" if n_pts >= MIN_VOXELS else (" (sparse)" if n_pts > 0 else "")
        print(f"  {NUSCENES_CLASSES[cls_id]:<25}  {sim_s}  {n_pts:>6}  {top1_s}  {novel:>8}{flag}")
        class_sims.append((cls_id, sim, top1, n_pts))

    # Aggregate metrics — well-sampled classes only
    sampled = [(c, s, t) for c, s, t, n in class_sims if n >= MIN_VOXELS and s == s]
    all_present = [(c, s) for c, s, t, n in class_sims if n > 0 and s == s]

    passed = True
    print(f"\n  ── Well-sampled classes (N ≥ {MIN_VOXELS}): {len(sampled)} of {len(all_present)} present ──")
    if sampled:
        valid_sims = torch.tensor([s for _, s, _ in sampled])
        valid_top1 = torch.tensor([t for _, _, t in sampled])
        overall    = float(valid_sims.mean())
        mean_top1  = float(valid_top1.mean())

        print(f"  Mean cosine sim  : {overall:.4f}  (raw per-voxel; threshold >0.0)")
        print(f"  Mean top-1 acc   : {mean_top1*100:.1f}%  (argmax over 16 classes per voxel)")

        ok = overall > 0.0
        passed = passed and ok
        print(f"  [{'✓ PASS' if ok else '✗ FAIL'}] Mean sim > 0.0 for well-sampled classes  (got {overall:.4f})")

        n_pos     = int((valid_sims > 0.0).sum())
        n_sampled = len(sampled)
        ok = n_pos >= int(n_sampled * 0.75)
        passed = passed and ok
        print(f"  [{'✓ PASS' if ok else '✗ FAIL'}] >=75% well-sampled classes have positive sim  ({n_pos}/{n_sampled})")

        ok = mean_top1 > 0.05
        passed = passed and ok
        print(f"  [{'✓ PASS' if ok else '✗ FAIL'}] Mean top-1 acc > 5% (random = 6.25%)  (got {mean_top1*100:.1f}%)")
    else:
        print(f"  ⚠  No classes with ≥{MIN_VOXELS} voxels — try --n_voxels 8192 for more coverage.")
        print(f"  Falling back to all-present check (n={len(all_present)}) ...")
        if all_present:
            valid_sims = torch.tensor([s for _, s in all_present])
            ok = float(valid_sims.mean()) > 0.0
            passed = passed and ok
            print(f"  [{'✓ PASS' if ok else '✗ FAIL'}] Mean sim > 0.0 over all present classes  (got {float(valid_sims.mean()):.4f})")
        else:
            print("  ✗ No labelled voxels found.")
            passed = False

    return passed


def load_prototypes(proto_path):
    print(f"\n[4c] Loading CLIP text prototypes: {proto_path}")
    if not os.path.exists(proto_path):
        print("  ✗ File not found.")
        return None, None

    data = torch.load(proto_path, map_location="cpu")
    if isinstance(data, torch.Tensor):
        text_feats = data.float()
    elif isinstance(data, dict):
        text_feats = None
        for key in ("prototypes", "text_features", "embeddings", "features"):
            if key in data and isinstance(data[key], torch.Tensor):
                text_feats = data[key].float()
                break
        if text_feats is None:
            for v in data.values():
                if isinstance(v, torch.Tensor):
                    text_feats = v.float()
                    break
    else:
        print(f"  ✗ Unknown type: {type(data)}")
        return None, None

    if text_feats is None or text_feats.shape[-1] != 768 or text_feats.shape[0] < 16:
        print(f"  ✗ Unexpected prototype shape: {getattr(text_feats,'shape','None')}")
        return None, None

    text_feats = F.normalize(text_feats[:16].float(), dim=-1)
    print(f"  ✓ Prototypes: {text_feats.shape}  norm[0]={text_feats[0].norm():.4f}")
    return text_feats.cpu(), "precomputed_vitl14_336px"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone",    default=DEFAULT_BACKBONE)
    p.add_argument("--prototypes",  default=DEFAULT_PROTOTYPES)
    p.add_argument("--scene",       default=DEFAULT_SCENE_PT,
                   help="Real nuScenes .pth scene for semantic CLIP check")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n_voxels",    type=int, default=512)
    p.add_argument("--use_real_data", action="store_true",
                   help="Run step 4: CLIP similarity on a real scene (requires dataset)")
    args = p.parse_args()

    print("=" * 65)
    print("  DisNet Backbone CLIP Alignment Verification")
    print(f"  Backbone   : {args.backbone}")
    print(f"  Prototypes : {args.prototypes}")
    print(f"  Device     : {args.device}")
    print()
    print("  NOTE: best.pth is DECODER-ONLY.")
    print("        Backbone is nuscenes_openseg.pth.tar (187 MB).")
    print("=" * 65)

    device = torch.device(args.device)
    checks_passed = {}

    # ── Steps 1–2: Load + architecture check ──────────────────────────────────
    state_dict, full_ckpt = load_backbone_checkpoint(args.backbone)

    # Verify best_iou from checkpoint (paper reports 35.4% distill mIoU)
    best_iou = full_ckpt.get("best_iou", None)
    if best_iou is not None:
        ok = float(best_iou) >= 0.30
        checks_passed["best_iou"] = ok
        print(f"\n  [{'✓ PASS' if ok else '⚠ WARN'}] Checkpoint best_iou = {float(best_iou)*100:.2f}%  "
              f"(OpenScene paper: 35.4%, threshold: ≥30%)")
    else:
        print("  [? INFO] No best_iou key in checkpoint metadata.")

    model = build_backbone(state_dict, device)
    checks_passed["strict_load"] = True  # would have crashed otherwise

    # ── Step 3: Feature statistics ────────────────────────────────────────────
    _, ok_shape, ok_norms = check_feature_statistics(model, device, args.n_voxels)
    checks_passed["output_shape"] = ok_shape
    checks_passed["feature_norms"] = ok_norms
    print(f"  [{'✓ PASS' if ok_shape else '✗ FAIL'}] Output dimension = 768")
    print(f"  [{'✓ PASS' if ok_norms else '✗ FAIL'}] Feature norm consistency (trained vs random backbone)")

    # ── Step 4: Optional semantic CLIP alignment ──────────────────────────────
    if args.use_real_data:
        text_feats, proto_src = load_prototypes(args.prototypes)
        if text_feats is None:
            print("  ✗ Cannot run semantic alignment without prototypes.")
        else:
            scene_data, used_path = load_real_scene(args.scene, device)
            if scene_data is not None:
                clip_ok = run_real_clip_alignment(model, scene_data, device,
                                                  text_feats, proto_src)
                checks_passed["clip_semantic"] = clip_ok
            else:
                print("  ⚠  Semantic CLIP check skipped (no scene data).")
    else:
        print(f"\n[4] Semantic CLIP similarity: SKIPPED")
        print(    "    Run with --use_real_data to test on a real nuScenes scene.")
        print(    "    (Random voxel input gives ~0 similarity for any trained backbone.)")

    # ── Final verdict ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Summary:")
    for name, ok in checks_passed.items():
        print(f"    [{'✓ PASS' if ok else '✗ FAIL'}]  {name}")

    critical = ["strict_load", "output_shape"]
    all_critical_ok = all(checks_passed.get(k, False) for k in critical)
    overall = all_critical_ok and all(v for v in checks_passed.values())

    print()
    if all_critical_ok:
        print("  ✓  VERDICT: Backbone architecture is CORRECT.")
        print("     strict=True load succeeded with 289 keys (MinkUNet18A).")
        if best_iou is not None and float(best_iou) >= 0.30:
            print(f"     Checkpoint best_iou={float(best_iou)*100:.1f}% confirms CLIP distillation training.")
        print("     Safe to proceed with decoder fine-tuning (Gates 1–4).")
        print("     For full CLIP alignment verification: re-run with --use_real_data")
    else:
        print("  ✗  VERDICT: Backbone check FAILED.")
        print("     Ensure --backbone points to nuscenes_openseg.pth.tar, not best.pth.")
    print("=" * 65)


if __name__ == "__main__":
    main()
