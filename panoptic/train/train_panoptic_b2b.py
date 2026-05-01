"""Phase 5a: Class-agnostic panoptic decoder training (B2b).

Same architecture as Phase 4, but trained with mask + dice loss only
(weight_ce=0.0, cost_class=0.0). At inference, masks are classified via
backbone feature pooling against 22 CLIP text prototypes (16 known + 6 novel).

Features beyond train_panoptic.py:
- Warm-start from Phase 4 checkpoint
- Class-agnostic evaluation via backbone feature pooling
- 22-class prototype loading for open-vocab inference
- Novel class detection logging

Usage:
    python -u panoptic/train/train_panoptic_b2b.py \
        --config panoptic/configs/full_nuscenes_b2b.yaml
"""
import argparse
import math
import os
import signal
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import MinkowskiEngine as ME
import yaml

sys.path.insert(0, "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene")

from models.disnet import DisNet, state_dict_remove_moudle
from panoptic.models.mask_decoder import PanopticDecoder
from panoptic.losses.panoptic_loss import PanopticLoss
from panoptic.eval.eval_pq import PanopticEval
from panoptic.dataset.nuscenes_panoptic import (
    NuScenesPanoptic, NUM_CLASSES, NUSCENES_CLASS_NAMES, THING_IDS, STUFF_IDS,
)

# Reuse helpers from base training script
from panoptic.train.train_panoptic import (
    load_config, set_seed, setup_distributed, load_backbone,
    extract_backbone_features, CosineWarmupScheduler, build_scheduler,
    GracefulKiller, save_checkpoint, find_resume_path,
    init_wandb, log_wandb, log_wandb_artifact, panoptic_inference,
)

# ── WandB import (optional) ─────────────────────────────────────────────────
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ══════════════════════════════════════════════════════════════════════════════
# 22-class prototype loading
# ══════════════════════════════════════════════════════════════════════════════

NOVEL_CLASS_NAMES = ["pole", "bench", "trash can", "fire hydrant", "tree trunk", "curb"]
ALL_22_CLASS_NAMES = list(NUSCENES_CLASS_NAMES) + NOVEL_CLASS_NAMES


def load_text_prototypes_22(path, device):
    """Load 22-class CLIP text prototypes (16 known + 6 novel)."""
    if os.path.isfile(path):
        protos = torch.load(path, map_location=device)
        print(f"[Prototypes] Loaded 22-class from {path}, shape={protos.shape}")
        return protos

    # Generate if not found
    print(f"[Prototypes] {path} not found, generating...")
    try:
        import clip
    except ImportError:
        raise RuntimeError("CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git")

    model, _ = clip.load("ViT-L/14@336px", device=device, jit=False)
    model.eval()

    template = "a point cloud of a {}"
    texts = [template.format(name) for name in ALL_22_CLASS_NAMES]
    tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_feats = model.encode_text(tokens).float()
        text_feats = F.normalize(text_feats, dim=-1)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(text_feats, path)
    print(f"[Prototypes] Generated and saved to {path}, shape={text_feats.shape}")
    return text_feats


# ══════════════════════════════════════════════════════════════════════════════
# Class-agnostic panoptic inference
# ══════════════════════════════════════════════════════════════════════════════

def panoptic_inference_class_agnostic(
    backbone_feats, pred_masks, text_protos,
    overlap_thr=0.8, obj_thr=0.3, min_mask_pts=10,
    num_known=16,
):
    """Class-agnostic inference: objectness from mask confidence, class from backbone pooling.

    Args:
        backbone_feats: (V, 768) raw backbone features
        pred_masks: (V, Q) mask logits
        text_protos: (C, 768) CLIP text prototypes (C = 16 or 22)
        overlap_thr: minimum remaining area ratio after greedy assignment
        obj_thr: minimum objectness (max sigmoid) to keep a query
        min_mask_pts: minimum points in binary mask
        num_known: number of known classes (16)

    Returns:
        sem_pred: (V,) int64 class labels (0-indexed, 0-21)
        inst_pred: (V,) int64 instance IDs
    """
    masks = pred_masks.sigmoid()  # (V, Q)
    V, Q = masks.shape

    # Objectness = max sigmoid value per query
    objectness = masks.max(dim=0).values  # (Q,)

    # Pool backbone features per predicted mask
    pooled = []
    valid_queries = []
    for q in range(Q):
        if objectness[q] < obj_thr:
            continue
        binary = masks[:, q] >= 0.5
        if binary.sum() < min_mask_pts:
            continue
        feat = backbone_feats[binary].mean(dim=0)  # (768,)
        feat = F.normalize(feat, dim=0)
        pooled.append(feat)
        valid_queries.append(q)

    if len(pooled) == 0:
        return np.zeros(V, dtype=np.int64), np.zeros(V, dtype=np.int64)

    pooled = torch.stack(pooled)  # (Qv, 768)
    cos_sim = torch.mm(pooled, text_protos.t())  # (Qv, C)
    scores, labels = cos_sim.max(-1)

    # Final score = objectness * cosine similarity
    obj_scores = objectness[valid_queries]
    final_scores = obj_scores * scores

    # Greedy assignment (same logic as panoptic_inference)
    # Things: 0-9 (known) + 16-21 (novel), Stuff: 10-15 (known)
    num_total = text_protos.shape[0]
    thing_set = set(range(10))
    if num_total > num_known:
        thing_set |= set(range(num_known, num_total))  # novel classes as things

    sorted_idx = final_scores.argsort(descending=True)

    sem_pred = np.zeros(V, dtype=np.int64)
    inst_pred = np.zeros(V, dtype=np.int64)
    inst_id = 1
    stuff_assigned = {}

    for idx in sorted_idx:
        idx = idx.item()
        label = labels[idx].item()
        q = valid_queries[idx]
        mask_prob = masks[:, q]

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


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation (class-agnostic with backbone pooling)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_class_agnostic(decoder, backbone, dataloader, device, cfg, text_protos_22):
    """Evaluate using backbone pooling for classification."""
    num_classes = cfg["model"]["num_classes"]
    num_total = text_protos_22.shape[0]  # 22
    overlap_thr = cfg["inference"]["overlap_thr"]
    ov_cfg = cfg.get("open_vocab", {})
    obj_thr = ov_cfg.get("objectness_thr", 0.3)
    min_mask_pts = ov_cfg.get("min_mask_pts", 10)

    decoder.eval()

    # Evaluator for known 16 classes (novel remapped to ignore)
    evaluator_known = PanopticEval(n_classes=num_classes, min_points=cfg["inference"]["min_points"])

    novel_voxel_counts = np.zeros(num_total - num_classes, dtype=np.int64)

    with torch.no_grad():
        for batch in dataloader:
            feats, coords = extract_backbone_features(backbone, batch, device)
            outputs = decoder(feats, coords)
            pred_masks = outputs["pred_masks"][0].cpu()
            backbone_feats = feats[0].cpu()  # (V, 768)

            sem_pred, inst_pred = panoptic_inference_class_agnostic(
                backbone_feats, pred_masks, text_protos_22.cpu(),
                overlap_thr=overlap_thr, obj_thr=obj_thr,
                min_mask_pts=min_mask_pts, num_known=num_classes,
            )

            # Count novel voxels
            for c in range(num_classes, num_total):
                novel_voxel_counts[c - num_classes] += (sem_pred == c).sum()

            # For known-class PQ: remap novel (>=16) to 0 (unlabeled)
            sem_known = sem_pred.copy()
            sem_known[sem_pred >= num_classes] = 0
            inst_known = inst_pred.copy()
            inst_known[sem_pred >= num_classes] = 0

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
            evaluator_known.addBatch(
                sem_known[valid], inst_known[valid],
                gt_sem[valid], gt_inst[valid],
            )

    PQ, SQ, RQ, pq_all, sq_all, rq_all = evaluator_known.getPQ()
    iou_mean, iou_all = evaluator_known.getSemIoU()

    novel_total = int(novel_voxel_counts.sum())
    novel_per_class = {
        NOVEL_CLASS_NAMES[i]: int(novel_voxel_counts[i])
        for i in range(len(NOVEL_CLASS_NAMES))
    }

    decoder.train()
    return {
        "PQ": PQ, "SQ": SQ, "RQ": RQ, "mIoU": iou_mean,
        "novel_voxels": novel_total,
        "novel_per_class": novel_per_class,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Phase 5a: Class-agnostic panoptic training (B2b)")
    p.add_argument("--config", default="panoptic/configs/full_nuscenes_b2b.yaml")
    p.add_argument("--resume", default=None, help="Override resume checkpoint path")
    args, unknown = p.parse_known_args()
    return args, unknown


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args, cli_overrides = parse_args()
    cfg = load_config(args.config, cli_overrides)

    # ── Distributed setup ──
    local_rank, rank, world_size, is_distributed, device = setup_distributed()
    is_main = (rank == 0)

    if is_main:
        print(f"Device: {device}, world_size: {world_size}")
        print(f"[B2b] Class-agnostic training: weight_ce={cfg['loss']['weight_ce']}, "
              f"cost_class={cfg['loss']['cost_class']}")

    seed = cfg["training"]["seed"] + rank
    set_seed(seed)

    killer = GracefulKiller()

    save_dir = cfg["checkpoint"]["save_dir"]
    if is_main:
        os.makedirs(save_dir, exist_ok=True)

    wandb_active = init_wandb(cfg) if is_main else False

    # ── Load 22-class text prototypes ──
    ov_cfg = cfg.get("open_vocab", {})
    proto_path = ov_cfg.get("prototypes_path", "panoptic/embeddings/text_prototypes_22.pt")
    text_protos_22 = load_text_prototypes_22(proto_path, device)

    # ── Data ──
    dcfg = cfg["data"]
    acfg = cfg["augmentation"]
    from nuscenes.nuscenes import NuScenes as NuScenesDB
    nusc = NuScenesDB(version=dcfg["version"], dataroot=dcfg["dataroot"], verbose=False)
    train_ds = NuScenesPanoptic(
        dcfg["dataroot"], dcfg["version"], split="train",
        voxel_size=dcfg["voxel_size"],
        point_range=np.array(dcfg["point_range"]),
        augment=acfg["enabled"],
        aug_cfg=acfg,
        nusc=nusc,
    )
    val_ds = NuScenesPanoptic(
        dcfg["dataroot"], dcfg["version"], split="val",
        voxel_size=dcfg["voxel_size"],
        point_range=np.array(dcfg["point_range"]),
        augment=False,
        nusc=nusc,
    )
    tcfg = cfg["training"]

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True
    ) if is_distributed else None

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=1,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=tcfg["num_workers"], collate_fn=lambda x: x[0],
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=tcfg["num_workers"], collate_fn=lambda x: x[0],
    )
    if is_main:
        print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # ── Models ──
    backbone = load_backbone(cfg["model"]["backbone_ckpt"], device)
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
        use_token_head=mcfg.get("use_token_head", False),
        token_dim=mcfg.get("token_dim", 768),
    ).to(device)

    if is_main:
        n_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        print(f"[Decoder] {n_params:,} trainable parameters")

    # ── Warm-start from Phase 4 ──
    ws_cfg = cfg.get("warm_start", {})
    ws_ckpt = ws_cfg.get("checkpoint")
    if ws_ckpt and os.path.isfile(ws_ckpt):
        ws_strict = ws_cfg.get("strict", True)
        ckpt = torch.load(ws_ckpt, map_location="cpu")
        missing, unexpected = decoder.load_state_dict(ckpt["decoder"], strict=ws_strict)
        if is_main:
            print(f"[Warm-start] Loaded from {ws_ckpt} (epoch={ckpt.get('epoch', '?')}, "
                  f"PQ={ckpt.get('best_pq', '?')}, strict={ws_strict})")
            if missing:
                print(f"  Missing keys: {missing}")
            if unexpected:
                print(f"  Unexpected keys: {unexpected}")
    elif ws_ckpt and is_main:
        print(f"[Warm-start] Checkpoint not found: {ws_ckpt}")

    # ── Loss ──
    lcfg = cfg["loss"]
    criterion = PanopticLoss(
        num_classes=mcfg["num_classes"],
        eos_coef=lcfg["eos_coef"],
        cost_class=lcfg["cost_class"],
        cost_mask=lcfg["cost_mask"],
        cost_dice=lcfg["cost_dice"],
        weight_ce=lcfg["weight_ce"],
        weight_mask=lcfg["weight_mask"],
        weight_dice=lcfg["weight_dice"],
        focal_gamma=lcfg["focal_gamma"],
        focal_alpha=lcfg["focal_alpha"],
        num_points=lcfg["num_points"],
        num_mask_pts=lcfg["num_mask_pts"],
        p_ratio=lcfg["p_ratio"],
    )

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )

    # ── Scheduler ──
    steps_per_epoch = len(train_loader) // tcfg["grad_accum"]
    scheduler, sched_step_per = build_scheduler(optimizer, cfg, steps_per_epoch)

    # ── Mixed precision ──
    use_amp = tcfg["mixed_precision"] and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # ── Resume ──
    start_epoch = 0
    best_pq = 0.0
    global_step = 0
    resume_path = find_resume_path(cfg, args.resume)
    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu")
        decoder.load_state_dict(ckpt["decoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_pq = ckpt.get("best_pq", 0.0)
        global_step = start_epoch * steps_per_epoch
        if is_main:
            print(f"[Resume] From {resume_path}, epoch {start_epoch}, best_pq={best_pq:.4f}")

    # ── DDP wrap ──
    if is_distributed:
        decoder = DDP(decoder, device_ids=[local_rank], output_device=local_rank,
                      find_unused_parameters=False)

    # ── Log config as wandb artifact ──
    if is_main and wandb_active:
        config_path = os.path.join(save_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        log_wandb_artifact(config_path, name="training-config", artifact_type="config")

    # ── Training ──
    if is_main:
        print(f"\n{'='*60}")
        print(f"B2b Training: epochs {start_epoch}→{tcfg['epochs']-1}, "
              f"lr={tcfg['lr']}, weight_ce={lcfg['weight_ce']}, cost_class={lcfg['cost_class']}")
        print(f"{'='*60}\n")

    log_every = cfg["logging"]["log_every"]
    eval_every = cfg["logging"]["eval_every"]
    save_every = cfg["checkpoint"]["save_every"]

    decoder.train()
    for epoch in range(start_epoch, tcfg["epochs"]):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_loss_ce = 0.0
        epoch_loss_mask = 0.0
        epoch_loss_dice = 0.0
        epoch_steps = 0
        optimizer.zero_grad()
        t_epoch = time.time()

        for step, batch in enumerate(train_loader):
            if killer.kill_now:
                if is_main:
                    save_checkpoint(
                        os.path.join(save_dir, "interrupted.pth"),
                        epoch, decoder, optimizer, scheduler, scaler, best_pq, cfg,
                    )
                    if wandb_active:
                        wandb.finish()
                if is_distributed:
                    dist.barrier()
                sys.exit(0)

            feats, coords = extract_backbone_features(backbone, batch, device)

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = decoder(feats, coords)
                    targets = {
                        "classes": [batch["gt_classes"].to(device)],
                        "masks": [batch["gt_masks"].float().to(device)],
                    }
                    losses, _ = criterion(outputs, targets)
                    total_loss = sum(losses.values()) / tcfg["grad_accum"]
                scaler.scale(total_loss).backward()
            else:
                outputs = decoder(feats, coords)
                targets = {
                    "classes": [batch["gt_classes"].to(device)],
                    "masks": [batch["gt_masks"].float().to(device)],
                }
                losses, _ = criterion(outputs, targets)
                total_loss = sum(losses.values()) / tcfg["grad_accum"]
                total_loss.backward()

            epoch_loss += total_loss.item() * tcfg["grad_accum"]
            # Track individual losses (final layer only, same scale)
            epoch_loss_ce += losses.get("loss_ce", torch.tensor(0.0)).item()
            epoch_loss_mask += losses.get("loss_mask", torch.tensor(0.0)).item()
            epoch_loss_dice += losses.get("loss_dice", torch.tensor(0.0)).item()
            epoch_steps += 1
            global_step += 1

            if (step + 1) % tcfg["grad_accum"] == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), tcfg["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), tcfg["grad_clip"])
                    optimizer.step()
                optimizer.zero_grad()

                if sched_step_per == "step":
                    scheduler.step()

            if is_main and (step + 1) % log_every == 0:
                avg = epoch_loss / epoch_steps
                lr_now = optimizer.param_groups[0]["lr"]
                loss_str = " ".join(
                    f"{k}={v.item():.4f}" for k, v in losses.items()
                    if not k[0].isdigit()
                )
                print(f"  [E{epoch} S{step+1}/{len(train_loader)}] "
                      f"loss={avg:.4f} lr={lr_now:.2e} | {loss_str}")

                log_data = {"train/loss": avg, "train/lr": lr_now}
                for k, v in losses.items():
                    if not k[0].isdigit():
                        log_data[f"train/{k}"] = v.item()
                log_wandb(log_data, step=global_step)

        # Flush remaining gradients
        if epoch_steps % tcfg["grad_accum"] != 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), tcfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), tcfg["grad_clip"])
                optimizer.step()
            optimizer.zero_grad()

        if sched_step_per == "epoch":
            scheduler.step()

        # Epoch summary
        epoch_time = time.time() - t_epoch
        if is_distributed:
            loss_tensor = torch.tensor([epoch_loss, float(epoch_steps)], device=device, dtype=torch.float64)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            epoch_loss = loss_tensor[0].item()
            epoch_steps_total = int(loss_tensor[1].item())
        else:
            epoch_steps_total = epoch_steps

        if is_main:
            # avg_loss includes all auxiliary layers; per-component are final-layer only
            avg_loss = epoch_loss / max(epoch_steps_total, 1)
            avg_ce = epoch_loss_ce / max(epoch_steps, 1)
            avg_mask = epoch_loss_mask / max(epoch_steps, 1)
            avg_dice = epoch_loss_dice / max(epoch_steps, 1)
            # Note: avg_loss ≈ num_levels × (avg_ce + avg_mask + avg_dice)
            lr_now = optimizer.param_groups[0]["lr"]

            gpu_mem = 0
            if device.type == "cuda":
                gpu_mem = torch.cuda.max_memory_allocated() / 1024**2
                torch.cuda.reset_peak_memory_stats()

            print(f"Epoch {epoch}: loss={avg_loss:.4f} (ce={avg_ce:.6f} mask={avg_mask:.4f} "
                  f"dice={avg_dice:.4f}), lr={lr_now:.2e}, time={epoch_time:.1f}s, gpu={gpu_mem:.0f}MB")
            log_wandb({
                "epoch/loss": avg_loss, "epoch/lr": lr_now,
                "epoch/loss_ce": avg_ce, "epoch/loss_mask": avg_mask,
                "epoch/loss_dice": avg_dice,
                "epoch/time_s": epoch_time, "epoch/gpu_peak_mb": gpu_mem,
            }, step=global_step)

        # ── Eval ──
        if (epoch + 1) % eval_every == 0 or epoch == tcfg["epochs"] - 1:
            if is_main:
                print("  Evaluating (class-agnostic backbone pooling)...")
                decoder_raw = decoder.module if is_distributed else decoder
                metrics = evaluate_class_agnostic(
                    decoder_raw, backbone, val_loader, device, cfg, text_protos_22,
                )
                print(f"  PQ_known={metrics['PQ']:.4f} SQ={metrics['SQ']:.4f} "
                      f"RQ={metrics['RQ']:.4f} mIoU={metrics['mIoU']:.4f} "
                      f"novel_voxels={metrics['novel_voxels']}")
                for cls_name, cnt in metrics["novel_per_class"].items():
                    if cnt > 0:
                        print(f"    {cls_name}: {cnt:,} voxels")

                log_wandb({
                    "val/PQ": metrics["PQ"], "val/SQ": metrics["SQ"],
                    "val/RQ": metrics["RQ"], "val/mIoU": metrics["mIoU"],
                    "val/novel_voxels": metrics["novel_voxels"],
                }, step=global_step)
                # Log per-novel-class counts
                for cls_name, cnt in metrics["novel_per_class"].items():
                    log_wandb({f"val/novel_{cls_name}": cnt}, step=global_step)

                if metrics["PQ"] > best_pq:
                    best_pq = metrics["PQ"]
                    save_checkpoint(
                        os.path.join(save_dir, "best.pth"),
                        epoch, decoder, optimizer, scheduler, scaler, best_pq, cfg, metrics,
                    )
                    print(f"  Saved best (PQ_known={best_pq:.4f})")
                    log_wandb_artifact(
                        os.path.join(save_dir, "best.pth"),
                        name="best-checkpoint",
                        metadata={"epoch": epoch, "PQ": float(best_pq),
                                  "novel_voxels": metrics["novel_voxels"]},
                    )

            if is_distributed:
                dist.barrier()
            decoder.train()

        # ── Save latest ──
        if (epoch + 1) % save_every == 0 or epoch == tcfg["epochs"] - 1:
            if is_main:
                save_checkpoint(
                    os.path.join(save_dir, "latest.pth"),
                    epoch, decoder, optimizer, scheduler, scaler, best_pq, cfg,
                )
                log_wandb_artifact(
                    os.path.join(save_dir, "latest.pth"),
                    name="latest-checkpoint",
                    metadata={"epoch": epoch, "best_pq": float(best_pq)},
                )
            if is_distributed:
                dist.barrier()

    if is_main:
        print(f"\nB2b training complete. Best PQ_known: {best_pq:.4f}")
        if wandb_active:
            wandb.finish()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
