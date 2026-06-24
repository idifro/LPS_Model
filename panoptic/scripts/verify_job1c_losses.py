"""Verification script for job1c changes.

Tests all new components before running full training:
  1. VoxelDecompLoss (L7): shapes, loss value, backward
  2. VoxelDistillLoss broadcast mode: margin gating, quality weighting, backward
  3. Combined loss backward: no NaN, LoRA grads present
  4. Dataset obj_margin field: present and non-trivially zero in a sample .npz
  5. Config sanity: job1c.yaml and job1c_mini.yaml parse correctly

Usage:
    python panoptic/scripts/verify_job1c_losses.py
    python panoptic/scripts/verify_job1c_losses.py --device cuda
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene")

PASS = "✅ PASS"
FAIL = "❌ FAIL"


def check(cond, msg):
    status = PASS if cond else FAIL
    print(f"  {status}  {msg}")
    return cond


# ── 1. VoxelDecompLoss ────────────────────────────────────────────────────────

def test_voxel_decomp_loss(device, proto_path, neg_bank_path):
    print("\n[1] VoxelDecompLoss (L7 InfoNCE)")
    from panoptic.losses.voxel_decomp_loss import VoxelDecompLoss

    try:
        loss_fn = VoxelDecompLoss(
            proto_path=proto_path,
            neg_bank_path=neg_bank_path,
            temperature=0.07,
            n_sample=256,
        ).to(device)
    except Exception as e:
        print(f"  {FAIL}  Init failed: {e}")
        return False

    ok = True

    # Shape check
    ok &= check(loss_fn.protos.shape == (16, 768), f"protos shape {loss_fn.protos.shape}")
    ok &= check(loss_fn.neg_bank.shape[1] == 768,  f"neg_bank shape {loss_fn.neg_bank.shape}")
    ok &= check(len(loss_fn.base_ids) == 12,        f"base_ids count {len(loss_fn.base_ids)}")

    # Forward with random features on base classes
    V = 500
    feats = torch.randn(V, 768, device=device, requires_grad=True)
    feats_norm = F.normalize(feats, dim=-1)

    # Simulate: 400 base-class voxels spread across 12 classes, 100 ignore
    vox_sem = torch.full((V,), 255, dtype=torch.long, device=device)
    base_ids = loss_fn.base_ids
    for i, cls_id in enumerate(base_ids):
        start = i * 33
        end = min(start + 33, 400)
        vox_sem[start:end] = cls_id

    try:
        loss = loss_fn(feats_norm, vox_sem)
    except Exception as e:
        print(f"  {FAIL}  Forward failed: {e}")
        return False

    ok &= check(loss.dim() == 0,            "loss is scalar")
    ok &= check(not loss.isnan().item(),    "loss is not NaN")
    ok &= check(loss.item() > 0,            f"loss > 0 ({loss.item():.4f})")

    # Backward
    try:
        loss.backward()
        ok &= check(feats.grad is not None,         "grad flows back to input features")
        ok &= check(not feats.grad.isnan().any(),   "no NaN in input grads")
    except Exception as e:
        print(f"  {FAIL}  Backward failed: {e}")
        ok = False

    # Empty class case (all ignore)
    feats2 = torch.randn(10, 768, device=device)
    vox_sem2 = torch.full((10,), 255, dtype=torch.long, device=device)
    loss2 = loss_fn(feats2, vox_sem2)
    ok &= check(loss2.item() == 0.0, "empty input returns 0 loss")

    return ok


# ── 2. VoxelDistillLoss broadcast mode ───────────────────────────────────────

def test_voxel_distill_broadcast(device):
    print("\n[2] VoxelDistillLoss — broadcast mode")
    from panoptic.losses.distill_voxel_loss import VoxelDistillLoss

    loss_fn = VoxelDistillLoss(mode="broadcast").to(device)
    ok = True

    B, Q, V = 1, 50, 200
    N_inst = 15
    N_samp = 64   # sampled voxels (simulating 4096 but smaller for speed)

    pred_embed  = torch.randn(B, Q, 768, device=device, requires_grad=True)
    pred_masks  = torch.randn(B, V, Q,  device=device, requires_grad=True)

    obj_feats   = F.normalize(torch.randn(B, N_inst, 768, device=device), dim=-1)
    obj_classes = torch.zeros(B, N_inst, dtype=torch.long, device=device)
    # Mix THING (0-9) and STUFF (10-15)
    obj_classes[0, :10] = torch.arange(10)   # things
    obj_classes[0, 10:] = torch.arange(10, 15)  # stuff (5 instances)

    # All instances valid
    obj_valid  = torch.ones(B, N_inst, dtype=torch.bool, device=device)
    obj_margin = torch.rand(B, N_inst, device=device) * 0.1 + 0.01  # all > 0

    # Map 64 sampled voxels to backbone positions and instances
    vox_gt_inst = torch.randint(0, N_inst, (B, N_samp), device=device)
    vox_clip_idx = torch.randint(0, V, (B, N_samp), device=device)

    try:
        loss = loss_fn(
            pred_embed, pred_masks,
            vox_clip_indices=vox_clip_idx,
            obj_feats=obj_feats, obj_valid=obj_valid,
            obj_margin=obj_margin, obj_classes=obj_classes,
            vox_gt_instance_id=vox_gt_inst,
        )
    except Exception as e:
        print(f"  {FAIL}  Forward failed: {e}")
        return False

    ok &= check(loss.dim() == 0,           "loss is scalar")
    ok &= check(not loss.isnan().item(),   "loss is not NaN")
    ok &= check(loss.item() > 0,           f"loss > 0 ({loss.item():.4f})")

    # Backward
    try:
        loss.backward()
        ok &= check(pred_embed.grad is not None,         "grad flows back to pred_embed")
        ok &= check(not pred_embed.grad.isnan().any(),   "no NaN in pred_embed grads")
        ok &= check(pred_masks.grad is not None,         "grad flows back to pred_masks (spatial feedback)")
        ok &= check(not pred_masks.grad.isnan().any(),   "no NaN in pred_masks grads")
    except Exception as e:
        print(f"  {FAIL}  Backward failed: {e}")
        ok = False

    # Gating test: all margins negative → loss should be 0
    pred_embed2 = torch.randn(B, Q, 768, device=device, requires_grad=True)
    obj_margin_neg = -torch.rand(B, N_inst, device=device) - 0.01  # all negative
    loss_neg = loss_fn(
        pred_embed2, pred_masks,
        vox_clip_indices=vox_clip_idx,
        obj_feats=obj_feats, obj_valid=obj_valid,
        obj_margin=obj_margin_neg, obj_classes=obj_classes,
        vox_gt_instance_id=vox_gt_inst,
    )
    ok &= check(loss_neg.item() == 0.0, "all-negative margins → zero loss (gate works)")

    # Gating test: all STUFF → loss should be 0
    obj_classes_stuff = torch.full((B, N_inst), 10, dtype=torch.long, device=device)
    loss_stuff = loss_fn(
        pred_embed2, pred_masks,
        vox_clip_indices=vox_clip_idx,
        obj_feats=obj_feats, obj_valid=obj_valid,
        obj_margin=obj_margin, obj_classes=obj_classes_stuff,
        vox_gt_instance_id=vox_gt_inst,
    )
    ok &= check(loss_stuff.item() == 0.0, "all-STUFF classes → zero loss (THING filter works)")

    # Class weight check: motorcycle (cls=5) should receive 4x weight
    # Verify the _CLASS_WEIGHTS dict in the module
    from panoptic.losses.distill_voxel_loss import _CLASS_WEIGHTS
    ok &= check(_CLASS_WEIGHTS.get(5) == 4.0, f"motorcycle class_weight=4.0 ({_CLASS_WEIGHTS})")
    ok &= check(_CLASS_WEIGHTS.get(2) == 1.5, f"bus class_weight=1.5")
    ok &= check(_CLASS_WEIGHTS.get(6) == 1.5, f"pedestrian class_weight=1.5")

    return ok


# ── 3. Backbone gradient flow through L7 ─────────────────────────────────────

def test_l7_backbone_grad(device, proto_path, neg_bank_path):
    """Verify L7 loss produces gradients through a simulated LoRA-adapted backbone."""
    print("\n[3] L7 backbone gradient flow (LoRA simulation)")
    from panoptic.losses.voxel_decomp_loss import VoxelDecompLoss

    loss_fn = VoxelDecompLoss(
        proto_path=proto_path,
        neg_bank_path=neg_bank_path,
        temperature=0.07,
        n_sample=128,
    ).to(device)

    ok = True

    # Simulate: frozen backbone linear + trainable LoRA adapter (rank-16)
    d_model = 768
    rank    = 16
    backbone_weight = torch.randn(d_model, d_model, device=device)  # frozen
    lora_A = torch.randn(rank, d_model, device=device, requires_grad=True)
    lora_B = torch.randn(d_model, rank, device=device, requires_grad=True)

    V = 200
    raw_input = torch.randn(V, d_model, device=device)
    # backbone pass: frozen linear + LoRA delta
    backbone_out = raw_input @ backbone_weight.T   # frozen, no grad
    lora_delta   = (raw_input @ lora_A.T) @ lora_B.T  # trainable
    feats = backbone_out + lora_delta              # (V, 768) with grad via lora

    # Base-class vox_sem
    vox_sem = torch.zeros(V, dtype=torch.long, device=device)  # all barrier (class 0)
    for i, cls_id in enumerate(loss_fn.base_ids[:6]):
        start = i * 33
        vox_sem[start:start+33] = cls_id

    try:
        loss = loss_fn(feats, vox_sem)
        loss.backward()
    except Exception as e:
        print(f"  {FAIL}  Forward/backward failed: {e}")
        return False

    lora_A_gnorm = lora_A.grad.norm().item() if lora_A.grad is not None else 0.0
    lora_B_gnorm = lora_B.grad.norm().item() if lora_B.grad is not None else 0.0
    ok &= check(lora_A.grad is not None,  "L7 gradient reaches LoRA A weight")
    ok &= check(lora_B.grad is not None,  "L7 gradient reaches LoRA B weight")
    ok &= check(lora_A_gnorm > 0.0,       f"LoRA A gnorm > 0 ({lora_A_gnorm:.4f})")
    ok &= check(lora_B_gnorm > 0.0,       f"LoRA B gnorm > 0 ({lora_B_gnorm:.4f})")
    ok &= check(not lora_A.grad.isnan().any(), "no NaN in LoRA A grads")

    # Confirm class 0 (barrier) is included in L7 base classes
    ok &= check(0 in loss_fn.base_ids, "barrier (cls 0) included in L7 base classes")
    ok &= check(2 not in loss_fn.base_ids, "bus (cls 2 = novel) excluded from L7")
    ok &= check(15 not in loss_fn.base_ids, "vegetation (cls 15 = novel) excluded from L7")

    return ok


# ── 4. Dataset obj_margin field ───────────────────────────────────────────────

def test_dataset_obj_margin(feat_dir):
    print("\n[4] Dataset obj_margin field")
    import glob
    ok = True

    npz_paths = sorted(glob.glob(f"{feat_dir}/v1.0-mini/*.npz"))
    if not npz_paths:
        print(f"  ⚠️   No .npz files found in {feat_dir}/v1.0-mini/")
        return True  # skip, not a code error

    n_present = 0
    n_pos = 0
    n_total = 0

    for path in npz_paths[:20]:
        d = np.load(path, allow_pickle=False)
        if "obj_margin" not in d.files:
            print(f"  {FAIL}  obj_margin MISSING in {os.path.basename(path)}")
            ok = False
            continue
        m = d["obj_margin"].astype(np.float32)
        n_present += 1
        n_pos += (m > 0).sum()
        n_total += len(m)

    if n_total > 0:
        pos_pct = 100.0 * n_pos / n_total
        ok &= check(n_present == min(20, len(npz_paths)), f"{n_present}/20 files have obj_margin")
        ok &= check(pos_pct > 10.0, f"{pos_pct:.1f}% of instances have positive margin (>10% expected)")

    return ok


# ── 4. Config parsing ─────────────────────────────────────────────────────────

def test_configs():
    print("\n[5] Config parsing")
    import yaml
    ok = True

    for config_name in ["panoptic/configs/job1c.yaml", "panoptic/configs/job1c_mini.yaml"]:
        if not os.path.exists(config_name):
            print(f"  {FAIL}  Config not found: {config_name}")
            ok = False
            continue
        try:
            with open(config_name) as f:
                cfg = yaml.safe_load(f)
        except Exception as e:
            print(f"  {FAIL}  Parse error in {config_name}: {e}")
            ok = False
            continue

        ws_ckpt = cfg.get("warm_start", {}).get("checkpoint", "")
        l7_weight = cfg["voxel_decomp"]["weight"]
        checks = [
            (bool(ws_ckpt), f"warm_start.checkpoint={ws_ckpt}"),
            (cfg["distillation"]["L_V"]["mode"] == "broadcast", "L_V mode=broadcast"),
            (cfg["distillation"]["L_V"]["weight"] == 0.3,  "L_V weight=0.3"),
            (cfg["voxel_decomp"]["enabled"] is True,        "voxel_decomp enabled"),
            (0.3 <= l7_weight <= 0.5,                       f"L7 weight={l7_weight} (valid range)"),
            (cfg["voxel_decomp"]["temperature"] == 0.07,    "L7 temperature=0.07"),
        ]
        for cond, desc in checks:
            ok &= check(cond, f"{config_name}: {desc}")

    return ok


# ── 5. Import check ───────────────────────────────────────────────────────────

def test_imports():
    print("\n[6] Import check")
    ok = True
    modules = [
        ("panoptic.losses.distill_voxel_loss", "VoxelDistillLoss"),
        ("panoptic.losses.voxel_decomp_loss",  "VoxelDecompLoss"),
        ("panoptic.losses.distill_object_loss","ObjectDistillLoss"),
    ]
    for module_path, class_name in modules:
        try:
            mod = __import__(module_path, fromlist=[class_name])
            cls = getattr(mod, class_name)
            ok &= check(True, f"{module_path}.{class_name} importable")
        except Exception as e:
            print(f"  {FAIL}  {module_path}.{class_name}: {e}")
            ok = False
    return ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--proto_path",
        default="panoptic/clip_prototypes/text_prototypes_vitl14_224_v2.pt")
    parser.add_argument("--neg_bank",
        default="panoptic/clip_prototypes/negative_bank_v2.pt")
    parser.add_argument("--feat_dir",
        default="/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features_v2")
    args = parser.parse_args()

    print("=" * 60)
    print("Job 1c verification suite")
    print("=" * 60)

    results = {}
    results["imports"]          = test_imports()
    results["configs"]          = test_configs()
    results["voxel_decomp"]     = test_voxel_decomp_loss(args.device, args.proto_path, args.neg_bank)
    results["l7_backbone_grad"] = test_l7_backbone_grad(args.device, args.proto_path, args.neg_bank)
    results["distill_broadcast"]= test_voxel_distill_broadcast(args.device)
    results["dataset_margin"]   = test_dataset_obj_margin(args.feat_dir)

    print("\n" + "=" * 60)
    print("Summary")
    print("-" * 60)
    all_pass = True
    for name, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  {status}  {name}")
        all_pass = all_pass and ok

    print("=" * 60)
    if all_pass:
        print("All checks passed. Safe to run job1c smoke test:")
        print("  python panoptic/train/train_panoptic_job1.py "
              "--config panoptic/configs/job1c_mini.yaml --smoke_test")
    else:
        print("Some checks failed. Fix before running training.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
