"""Panoptic decoder training loop with production best practices.

Features:
- YAML config (all hyperparams in one file)
- WandB logging (falls back to stdout if unavailable)
- Mixed precision (AMP)
- Cosine warmup / CosineAnnealingLR / StepLR scheduler (configurable)
- Checkpoint resume (auto-detect latest.pth)
- Signal handler for graceful HPC shutdown (SIGTERM)
- Seed for reproducibility
- GPU memory monitoring
- Config saved in checkpoint
- Multi-GPU DDP support (backward-compatible with single-GPU)

Usage:
    # Single GPU:
    python -u panoptic/train/train_panoptic.py --config panoptic/configs/full_nuscenes.yaml
    # Multi-GPU (4 GPUs, single node):
    torchrun --standalone --nproc_per_node=4 panoptic/train/train_panoptic.py --config panoptic/configs/full_nuscenes_4gpu.yaml
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
    NuScenesPanoptic, NUM_CLASSES, THING_IDS, STUFF_IDS,
)

# ── WandB import (optional) ─────────────────────────────────────────────────
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("[WARN] wandb not installed. Logging to stdout only.")


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

def load_config(config_path, cli_overrides=None):
    """Load YAML config, apply CLI overrides."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    # Apply flat CLI overrides like --training.lr=3e-4
    if cli_overrides:
        for kv in cli_overrides:
            if "=" not in kv:
                continue
            key, val = kv.split("=", 1)
            keys = key.lstrip("-").split(".")
            d = cfg
            for k in keys[:-1]:
                d = d[k]
            # Auto-cast types
            old_val = d.get(keys[-1])
            if isinstance(old_val, bool):
                val = val.lower() in ("true", "1", "yes")
            elif isinstance(old_val, int):
                val = int(val)
            elif isinstance(old_val, float):
                val = float(val)
            d[keys[-1]] = val
    return cfg


def parse_args():
    p = argparse.ArgumentParser(description="Panoptic decoder training")
    p.add_argument("--config", default="panoptic/configs/default.yaml")
    p.add_argument("--resume", default=None, help="Override resume checkpoint path")
    args, unknown = p.parse_known_args()
    return args, unknown


# ══════════════════════════════════════════════════════════════════════════════
# Seed
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        pass  # don't force deterministic — ME doesn't support it


# ══════════════════════════════════════════════════════════════════════════════
# Distributed setup
# ══════════════════════════════════════════════════════════════════════════════

def setup_distributed():
    """Auto-detect DDP from torchrun env vars. Backward-compatible with single-GPU."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        return local_rank, rank, world_size, True, device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 0, 1, False, device


# ══════════════════════════════════════════════════════════════════════════════
# Backbone
# ══════════════════════════════════════════════════════════════════════════════

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
    print(f"[Backbone] Loaded DisNet from {ckpt_path}, frozen.")
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


# ══════════════════════════════════════════════════════════════════════════════
# Scheduler builders
# ══════════════════════════════════════════════════════════════════════════════

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup then cosine decay to min_lr."""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            # Linear warmup
            scale = step / max(self.warmup_steps, 1)
            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            progress = min(progress, 1.0)
            cos_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cos_scale
                for base_lr in self.base_lrs
            ]


def build_scheduler(optimizer, cfg, steps_per_epoch):
    sched_type = cfg["training"]["scheduler"]
    if sched_type == "cosine_warmup":
        warmup_steps = cfg["training"]["warmup_epochs"] * steps_per_epoch
        total_steps = cfg["training"]["epochs"] * steps_per_epoch
        scheduler = CosineWarmupScheduler(
            optimizer, warmup_steps, total_steps,
            min_lr=cfg["training"]["min_lr"],
        )
        step_per = "step"
    elif sched_type == "cosine_annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["training"].get("cosine_T_max", cfg["training"]["epochs"]),
            eta_min=cfg["training"].get("cosine_eta_min", 1e-6),
        )
        step_per = "epoch"
    elif sched_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg["training"]["step_size"],
            gamma=cfg["training"]["step_gamma"],
        )
        step_per = "epoch"
    else:
        raise ValueError(f"Unknown scheduler: {sched_type}")
    print(f"[Scheduler] {sched_type}, step_per={step_per}")
    return scheduler, step_per


# ══════════════════════════════════════════════════════════════════════════════
# Panoptic inference
# ══════════════════════════════════════════════════════════════════════════════

def panoptic_inference(pred_logits, pred_masks, num_classes=16, overlap_thr=0.8):
    scores, labels = pred_logits.softmax(-1).max(-1)
    masks = pred_masks.sigmoid()
    V = masks.shape[0]

    keep = labels != num_classes
    if keep.sum() == 0:
        return np.zeros(V, dtype=np.int64), np.zeros(V, dtype=np.int64)

    scores = scores[keep]
    labels = labels[keep]
    masks = masks[:, keep]

    sorted_idx = scores.argsort(descending=True)

    sem_pred = np.full(V, 0, dtype=np.int64)
    inst_pred = np.zeros(V, dtype=np.int64)
    inst_id = 1
    thing_set = set(range(10))
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


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(decoder, backbone, dataloader, device, cfg):
    num_classes = cfg["model"]["num_classes"]
    overlap_thr = cfg["inference"]["overlap_thr"]
    decoder.eval()
    evaluator = PanopticEval(n_classes=num_classes, min_points=cfg["inference"]["min_points"])

    with torch.no_grad():
        for batch in dataloader:
            feats, coords = extract_backbone_features(backbone, batch, device)
            outputs = decoder(feats, coords)
            pred_logits = outputs["pred_logits"][0].cpu()
            pred_masks = outputs["pred_masks"][0].cpu()
            sem_pred, inst_pred = panoptic_inference(pred_logits, pred_masks, num_classes, overlap_thr)

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
            evaluator.addBatch(
                sem_pred[valid], inst_pred[valid],
                gt_sem[valid], gt_inst[valid],
            )

    PQ, SQ, RQ, pq_all, sq_all, rq_all = evaluator.getPQ()
    iou_mean, iou_all = evaluator.getSemIoU()
    decoder.train()
    return {"PQ": PQ, "SQ": SQ, "RQ": RQ, "mIoU": iou_mean}


# ══════════════════════════════════════════════════════════════════════════════
# Signal handler for graceful HPC shutdown
# ══════════════════════════════════════════════════════════════════════════════

class GracefulKiller:
    """Catches SIGTERM (from HPC scheduler) and sets a flag for graceful exit."""

    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGTERM, self._handler)
        signal.signal(signal.SIGUSR1, self._handler)  # some HPC schedulers use USR1

    def _handler(self, signum, frame):
        print(f"\n[SIGNAL] Received {signal.Signals(signum).name}. Saving checkpoint and exiting...")
        self.kill_now = True


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(path, epoch, decoder, optimizer, scheduler, scaler, best_pq, cfg, metrics=None):
    state = {
        "epoch": epoch,
        "decoder": decoder.module.state_dict() if hasattr(decoder, "module") else decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_pq": best_pq,
        "config": cfg,
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if metrics is not None:
        state["metrics"] = metrics
    torch.save(state, path)


def find_resume_path(cfg, cli_resume):
    """Determine checkpoint path to resume from."""
    if cli_resume is not None:
        return cli_resume if os.path.isfile(cli_resume) else None
    if cfg["checkpoint"]["resume"] is not None:
        p = cfg["checkpoint"]["resume"]
        return p if os.path.isfile(p) else None
    if cfg["checkpoint"]["auto_resume"]:
        p = os.path.join(cfg["checkpoint"]["save_dir"], "latest.pth")
        return p if os.path.isfile(p) else None
    return None


# ══════════════════════════════════════════════════════════════════════════════
# WandB helpers
# ══════════════════════════════════════════════════════════════════════════════

def init_wandb(cfg):
    if not HAS_WANDB or not cfg["logging"]["wandb_enabled"]:
        return False
    wandb.init(
        project=cfg["logging"]["wandb_project"],
        entity=cfg["logging"]["wandb_entity"],
        name=cfg["logging"]["wandb_run_name"],
        config=cfg,
        resume="allow",
    )
    return True


def log_wandb(data, step=None):
    if HAS_WANDB and wandb.run is not None:
        wandb.log(data, step=step)


def log_wandb_artifact(path, name, artifact_type="model", metadata=None):
    """Log a checkpoint file as a wandb artifact."""
    if not HAS_WANDB or wandb.run is None:
        return
    artifact = wandb.Artifact(name, type=artifact_type, metadata=metadata or {})
    artifact.add_file(path)
    wandb.log_artifact(artifact)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args, cli_overrides = parse_args()
    cfg = load_config(args.config, cli_overrides)

    # ── Distributed setup (auto-detect from torchrun env) ──
    local_rank, rank, world_size, is_distributed, device = setup_distributed()
    is_main = (rank == 0)

    if is_main:
        print(f"Device: {device}, world_size: {world_size}")

    # Seed (offset by rank for different augmentation per GPU)
    seed = cfg["training"]["seed"] + rank
    set_seed(seed)
    if is_main:
        print(f"[Seed] {seed} (rank {rank})")

    # Signal handler
    killer = GracefulKiller()

    # Directories (rank 0 only)
    save_dir = cfg["checkpoint"]["save_dir"]
    if is_main:
        os.makedirs(save_dir, exist_ok=True)

    # WandB (rank 0 only)
    wandb_active = init_wandb(cfg) if is_main else False
    if is_main:
        print(f"[WandB] {'Active' if wandb_active else 'Disabled'}")

    # ── Data (share NuScenes object between train/val to save ~2GB RAM) ──
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

    # DistributedSampler for training (splits data across GPUs)
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
        print(f"Train: {len(train_ds)} samples ({len(train_loader)} batches/GPU), Val: {len(val_ds)} samples")

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
    ).to(device)

    if is_main:
        n_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        print(f"[Decoder] {n_params:,} trainable parameters")

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

    # ── Optimizer (create before DDP wrap — references same param objects) ──
    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )

    # ── Scheduler ──
    steps_per_epoch = len(train_loader) // tcfg["grad_accum"]
    scheduler, sched_step_per = build_scheduler(optimizer, cfg, steps_per_epoch)

    # ── Mixed precision ──
    use_amp = tcfg["mixed_precision"] and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if is_main:
        print(f"[AMP] {'Enabled' if use_amp else 'Disabled'}")

    # ── Resume (load into unwrapped decoder, before DDP wrap) ──
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

    # ── Wrap decoder in DDP (after resume load) ──
    if is_distributed:
        decoder = DDP(decoder, device_ids=[local_rank], output_device=local_rank,
                      find_unused_parameters=False)
        if is_main:
            print(f"[DDP] Wrapped decoder, {world_size} GPUs")

    # ── Training ──
    if is_main:
        print(f"\n{'='*60}")
        print(f"Training: epochs {start_epoch}→{tcfg['epochs']-1}, "
              f"scheduler={tcfg['scheduler']}, lr={tcfg['lr']}, "
              f"grad_accum={tcfg['grad_accum']}, world_size={world_size}")
        print(f"{'='*60}\n")

    log_every = cfg["logging"]["log_every"]
    eval_every = cfg["logging"]["eval_every"]
    save_every = cfg["checkpoint"]["save_every"]

    decoder.train()
    for epoch in range(start_epoch, tcfg["epochs"]):
        # Set epoch on sampler for correct shuffling
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_steps = 0
        optimizer.zero_grad()
        t_epoch = time.time()

        for step, batch in enumerate(train_loader):
            # Check for HPC kill signal
            if killer.kill_now:
                if is_main:
                    print("[SIGNAL] Saving interrupted checkpoint...")
                    save_checkpoint(
                        os.path.join(save_dir, "interrupted.pth"),
                        epoch, decoder, optimizer, scheduler, scaler, best_pq, cfg,
                    )
                    print(f"[SIGNAL] Saved interrupted.pth at epoch {epoch} step {step}")
                    if wandb_active:
                        wandb.finish()
                if is_distributed:
                    dist.barrier()
                sys.exit(0)

            feats, coords = extract_backbone_features(backbone, batch, device)

            # Forward + loss with optional AMP
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
            epoch_steps += 1
            global_step += 1

            # Optimizer step with grad accumulation
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

            # Logging (rank 0 only)
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

        # Epoch summary — all-reduce loss across ranks for accurate reporting
        epoch_time = time.time() - t_epoch
        if is_distributed:
            loss_tensor = torch.tensor([epoch_loss, float(epoch_steps)], device=device, dtype=torch.float64)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            epoch_loss = loss_tensor[0].item()
            epoch_steps_total = int(loss_tensor[1].item())
        else:
            epoch_steps_total = epoch_steps

        if is_main:
            avg_loss = epoch_loss / max(epoch_steps_total, 1)
            lr_now = optimizer.param_groups[0]["lr"]

            gpu_mem = 0
            if device.type == "cuda":
                gpu_mem = torch.cuda.max_memory_allocated() / 1024**2
                torch.cuda.reset_peak_memory_stats()

            print(f"Epoch {epoch}: loss={avg_loss:.4f}, lr={lr_now:.2e}, "
                  f"time={epoch_time:.1f}s, gpu_peak={gpu_mem:.0f}MB")
            log_wandb({
                "epoch/loss": avg_loss, "epoch/lr": lr_now,
                "epoch/time_s": epoch_time, "epoch/gpu_peak_mb": gpu_mem,
            }, step=global_step)

        # ── Eval (rank 0 only, others wait at barrier) ──
        if (epoch + 1) % eval_every == 0 or epoch == tcfg["epochs"] - 1:
            if is_main:
                print("  Evaluating...")
                decoder_raw = decoder.module if is_distributed else decoder
                metrics = evaluate(decoder_raw, backbone, val_loader, device, cfg)
                print(f"  PQ={metrics['PQ']:.4f} SQ={metrics['SQ']:.4f} "
                      f"RQ={metrics['RQ']:.4f} mIoU={metrics['mIoU']:.4f}")
                log_wandb({
                    "val/PQ": metrics["PQ"], "val/SQ": metrics["SQ"],
                    "val/RQ": metrics["RQ"], "val/mIoU": metrics["mIoU"],
                }, step=global_step)

                if metrics["PQ"] > best_pq:
                    best_pq = metrics["PQ"]
                    save_checkpoint(
                        os.path.join(save_dir, "best.pth"),
                        epoch, decoder, optimizer, scheduler, scaler, best_pq, cfg, metrics,
                    )
                    print(f"  Saved best (PQ={best_pq:.4f})")
                    log_wandb_artifact(
                        os.path.join(save_dir, "best.pth"),
                        name="best-checkpoint",
                        metadata={"epoch": epoch, "PQ": float(best_pq)},
                    )

            if is_distributed:
                dist.barrier()
            # Ensure decoder is back in train mode on all ranks
            decoder.train()

        # ── Save latest (rank 0 only) ──
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
        print(f"\nTraining complete. Best PQ: {best_pq:.4f}")
        if wandb_active:
            wandb.finish()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
