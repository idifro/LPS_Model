"""End-to-End panoptic training with unfrozen backbone encoder.

This script extends the frozen-backbone panoptic training by allowing gradients
to flow through the DisNet backbone (fully or partially), enabling joint
optimization of 3D features and the panoptic decoder.

Key differences from train_panoptic.py:
- Backbone is NOT frozen (or partially unfrozen based on config)
- Differential learning rates: backbone gets lower LR than decoder
- Optional delayed unfreezing: backbone stays frozen for N epochs then unfreezes
- Warm-starts from a frozen-backbone checkpoint for stability
- Gradient checkpointing option for memory efficiency

Usage:
    python -u panoptic/train/train_panoptic_e2e.py --config panoptic/configs/mini_e2e_encoder.yaml
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

sys.path.insert(0, "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/vandana_experiments/LPS_Model")

from models.disnet import DisNet, state_dict_remove_moudle
from panoptic.models.mask_decoder import PanopticDecoder
from panoptic.losses.panoptic_loss import PanopticLoss
from panoptic.eval.eval_pq import PanopticEval
from panoptic.dataset.nuscenes_panoptic import (
    NuScenesPanoptic, NUM_CLASSES, THING_IDS, STUFF_IDS,
)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("[WARN] wandb not installed. Logging to stdout only.")


# ══════════════════════════════════════════════════════════════════════════════
# Config & Setup (reused from train_panoptic.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_config(config_path, cli_overrides=None):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if cli_overrides:
        for kv in cli_overrides:
            if "=" not in kv:
                continue
            key, val = kv.split("=", 1)
            keys = key.lstrip("-").split(".")
            d = cfg
            for k in keys[:-1]:
                d = d[k]
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
    p = argparse.ArgumentParser(description="E2E panoptic training (unfrozen backbone)")
    p.add_argument("--config", default="panoptic/configs/mini_e2e_encoder.yaml")
    p.add_argument("--resume", default=None)
    args, unknown = p.parse_known_args()
    return args, unknown


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def setup_distributed():
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
# Backbone with E2E support
# ══════════════════════════════════════════════════════════════════════════════

def load_backbone_e2e(ckpt_path, device, e2e_cfg):
    """Load DisNet backbone with configurable freezing/unfreezing.
    
    Args:
        ckpt_path: Path to pretrained checkpoint
        device: Target device
        e2e_cfg: dict with keys: enabled, unfreeze_mode, unfreeze_last_n
    
    Returns:
        (backbone, frozen_params_names, trainable_params_names)
    """
    class Cfg:
        feature_2d_extractor = "openseg"
        arch_3d = "MinkUNet18A"
    
    model = DisNet(Cfg()).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = state_dict_remove_moudle(ckpt["state_dict"])
    model.load_state_dict(state, strict=True)
    
    if not e2e_cfg["enabled"]:
        # Fully frozen (same as original)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        print(f"[Backbone] Fully FROZEN (e2e disabled)")
        return model, [], []
    
    # Identify layers for unfreezing
    mode = e2e_cfg["unfreeze_mode"]
    frozen_names = []
    trainable_names = []
    
    if mode == "all":
        # Unfreeze everything
        for name, p in model.named_parameters():
            p.requires_grad = True
            trainable_names.append(name)
        model.train()
        print(f"[Backbone] ALL layers UNFROZEN ({len(trainable_names)} params)")
        
    elif mode == "decoder_only":
        # Freeze encoder stages, unfreeze decoder stages + final head
        # MinkUNet18A structure: convN/bnN = encoder, convtrN/bntrN = decoder, final = head
        for name, p in model.named_parameters():
            is_decoder = any(k in name for k in [
                "convtr", "bntr", "final",  # transposed conv layers (decoder)
            ])
            if is_decoder:
                p.requires_grad = True
                trainable_names.append(name)
            else:
                p.requires_grad = False
                frozen_names.append(name)
        model.train()
        print(f"[Backbone] DECODER_ONLY unfrozen: {len(trainable_names)} trainable, "
              f"{len(frozen_names)} frozen")
              
    elif mode == "last_n_stages":
        # Unfreeze the last N encoder stages + full decoder
        n = e2e_cfg["unfreeze_last_n"]
        # MinkUNet18A has 4 encoder stages (conv1-conv4) and 4 decoder stages
        # Stage numbering: conv1(stem), conv2(enc1), conv3(enc2), conv4(enc3), conv4b(enc4)
        # Decoder: convtr4, convtr3, convtr2, convtr1, final
        encoder_stages = ["conv1", "conv2", "conv3", "conv4"]
        freeze_stages = set(encoder_stages[:max(0, 4 - n)])
        
        for name, p in model.named_parameters():
            # Check if this param belongs to a frozen encoder stage
            should_freeze = False
            for stage in freeze_stages:
                if stage in name and "convtr" not in name:
                    should_freeze = True
                    break
            
            if should_freeze:
                p.requires_grad = False
                frozen_names.append(name)
            else:
                p.requires_grad = True
                trainable_names.append(name)
        model.train()
        print(f"[Backbone] LAST_{n}_STAGES unfrozen: {len(trainable_names)} trainable, "
              f"{len(frozen_names)} frozen")
    else:
        raise ValueError(f"Unknown unfreeze_mode: {mode}")
    
    return model, frozen_names, trainable_names


def freeze_backbone(backbone):
    """Freeze all backbone parameters (used during delayed unfreeze warmup)."""
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()


def unfreeze_backbone(backbone, e2e_cfg):
    """Apply the configured unfreeze policy to backbone."""
    mode = e2e_cfg["unfreeze_mode"]
    trainable_names = []
    
    if mode == "all":
        for name, p in backbone.named_parameters():
            p.requires_grad = True
            trainable_names.append(name)
    elif mode == "decoder_only":
        for name, p in backbone.named_parameters():
            is_decoder = any(k in name for k in ["convtr", "bntr", "final"])
            if is_decoder:
                p.requires_grad = True
                trainable_names.append(name)
    elif mode == "last_n_stages":
        n = e2e_cfg["unfreeze_last_n"]
        encoder_stages = ["conv1", "conv2", "conv3", "conv4"]
        freeze_stages = set(encoder_stages[:max(0, 4 - n)])
        for name, p in backbone.named_parameters():
            should_freeze = False
            for stage in freeze_stages:
                if stage in name and "convtr" not in name:
                    should_freeze = True
                    break
            if not should_freeze:
                p.requires_grad = True
                trainable_names.append(name)
    
    backbone.train()
    return trainable_names


def extract_backbone_features_e2e(backbone, batch, device, is_frozen=False):
    """Extract backbone features — with or without gradients.
    
    When backbone is unfrozen, gradients flow through for E2E training.
    When frozen (during warmup phase), uses torch.no_grad().
    """
    vox_coords = batch["vox_coords"]
    vox_feats = batch["vox_feats"]
    V = vox_coords.shape[0]
    batch_coords = torch.cat([
        torch.zeros(V, 1, dtype=torch.int32), vox_coords
    ], dim=1)
    sinput = ME.SparseTensor(
        vox_feats.float(), coordinates=batch_coords.int(), device=device
    )
    
    if is_frozen:
        with torch.no_grad():
            output = backbone(sinput)
    else:
        output = backbone(sinput)
    
    feats = output.F if hasattr(output, "F") else output
    coords_xyz = vox_feats.to(device)
    return feats.unsqueeze(0), coords_xyz.unsqueeze(0)


# ══════════════════════════════════════════════════════════════════════════════
# Scheduler
# ══════════════════════════════════════════════════════════════════════════════

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            scale = step / max(self.warmup_steps, 1)
            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            progress = min(progress, 1.0)
            cos_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cos_scale for base_lr in self.base_lrs]


# ══════════════════════════════════════════════════════════════════════════════
# Panoptic inference & evaluation (same as train_panoptic.py)
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


def evaluate(decoder, backbone, dataloader, device, cfg):
    num_classes = cfg["model"]["num_classes"]
    overlap_thr = cfg["inference"]["overlap_thr"]
    decoder.eval()
    backbone.eval()
    evaluator = PanopticEval(n_classes=num_classes, min_points=cfg["inference"]["min_points"])

    with torch.no_grad():
        for batch in dataloader:
            feats, coords = extract_backbone_features_e2e(backbone, batch, device, is_frozen=True)
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
            evaluator.addBatch(sem_pred[valid], inst_pred[valid], gt_sem[valid], gt_inst[valid])

    PQ, SQ, RQ, pq_all, sq_all, rq_all = evaluator.getPQ()
    iou_mean, iou_all = evaluator.getSemIoU()
    decoder.train()
    return {"PQ": PQ, "SQ": SQ, "RQ": RQ, "mIoU": iou_mean}


# ══════════════════════════════════════════════════════════════════════════════
# Signal handler & Checkpointing
# ══════════════════════════════════════════════════════════════════════════════

class GracefulKiller:
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGTERM, self._handler)
        signal.signal(signal.SIGUSR1, self._handler)

    def _handler(self, signum, frame):
        print(f"\n[SIGNAL] Received {signal.Signals(signum).name}. Saving and exiting...")
        self.kill_now = True


def save_checkpoint(path, epoch, decoder, backbone, optimizer, scheduler, scaler, best_pq, cfg, metrics=None):
    """Save full state including backbone (since it's trainable now)."""
    state = {
        "epoch": epoch,
        "decoder": decoder.module.state_dict() if hasattr(decoder, "module") else decoder.state_dict(),
        "backbone": backbone.module.state_dict() if hasattr(backbone, "module") else backbone.state_dict(),
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
        config=cfg, resume="allow",
    )
    return True


def log_wandb(data, step=None):
    if HAS_WANDB and wandb.run is not None:
        wandb.log(data, step=step)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args, cli_overrides = parse_args()
    cfg = load_config(args.config, cli_overrides)

    local_rank, rank, world_size, is_distributed, device = setup_distributed()
    is_main = (rank == 0)

    if is_main:
        print(f"{'='*70}")
        print(f"  E2E ENCODER TRAINING — Unfrozen Backbone")
        print(f"{'='*70}")
        print(f"Device: {device}, world_size: {world_size}")

    seed = cfg["training"]["seed"] + rank
    set_seed(seed)
    killer = GracefulKiller()

    save_dir = cfg["checkpoint"]["save_dir"]
    if is_main:
        os.makedirs(save_dir, exist_ok=True)

    wandb_active = init_wandb(cfg) if is_main else False

    # ── Data ──
    dcfg = cfg["data"]
    acfg = cfg["augmentation"]
    from nuscenes.nuscenes import NuScenes as NuScenesDB
    nusc = NuScenesDB(version=dcfg["version"], dataroot=dcfg["dataroot"], verbose=False)
    train_ds = NuScenesPanoptic(
        dcfg["dataroot"], dcfg["version"], split="train",
        voxel_size=dcfg["voxel_size"],
        point_range=np.array(dcfg["point_range"]),
        augment=acfg["enabled"], aug_cfg=acfg, nusc=nusc,
    )
    val_ds = NuScenesPanoptic(
        dcfg["dataroot"], dcfg["version"], split="val",
        voxel_size=dcfg["voxel_size"],
        point_range=np.array(dcfg["point_range"]),
        augment=False, nusc=nusc,
    )
    tcfg = cfg["training"]

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True
    ) if is_distributed else None

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=1, shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=tcfg["num_workers"], collate_fn=lambda x: x[0],
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=tcfg["num_workers"], collate_fn=lambda x: x[0],
    )
    if is_main:
        print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # ── Backbone (E2E) ──
    e2e_cfg = cfg.get("e2e", {"enabled": False})
    backbone, frozen_names, trainable_names = load_backbone_e2e(
        cfg["model"]["backbone_ckpt"], device, e2e_cfg
    )

    # Handle delayed unfreezing: start frozen, unfreeze after N epochs
    unfreeze_after = e2e_cfg.get("unfreeze_after_epoch", 0)
    backbone_is_frozen = (unfreeze_after > 0)
    if backbone_is_frozen and e2e_cfg["enabled"]:
        freeze_backbone(backbone)
        if is_main:
            print(f"[E2E] Backbone frozen for first {unfreeze_after} epochs (delayed unfreeze)")

    # ── Decoder ──
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

    # ── Warm-start from frozen-backbone checkpoint ──
    warmstart_path = e2e_cfg.get("warmstart_ckpt")
    if warmstart_path and os.path.isfile(warmstart_path):
        ws_ckpt = torch.load(warmstart_path, map_location="cpu")
        decoder.load_state_dict(ws_ckpt["decoder"], strict=False)
        if is_main:
            print(f"[Warmstart] Loaded decoder from {warmstart_path}")
    elif warmstart_path and is_main:
        print(f"[Warmstart] WARNING: {warmstart_path} not found, training from scratch")

    if is_main:
        n_decoder = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        n_backbone = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        n_total = n_decoder + n_backbone
        print(f"[Params] Decoder: {n_decoder:,}, Backbone: {n_backbone:,}, Total: {n_total:,}")

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

    # ── Optimizer with differential LR ──
    backbone_lr = e2e_cfg.get("backbone_lr",1e-6)
    param_groups = [
        {"params": [p for p in decoder.parameters() if p.requires_grad],
         "lr": tcfg["lr"], "name": "decoder"},
        {"params": [p for p in backbone.parameters() if p.requires_grad],
         "lr": backbone_lr, "name": "backbone"},
    ]
    # Filter empty groups
    param_groups = [g for g in param_groups if len(g["params"]) > 0]
    
    optimizer = torch.optim.AdamW(
        param_groups, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )
    if is_main:
        for g in optimizer.param_groups:
            print(f"  Param group '{g.get('name', '?')}': {len(g['params'])} tensors, lr={g['lr']:.1e}")

    # ── Scheduler ──
    steps_per_epoch = len(train_loader) // tcfg["grad_accum"]
    warmup_steps = cfg["training"]["warmup_epochs"] * steps_per_epoch
    total_steps = cfg["training"]["epochs"] * steps_per_epoch
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_steps, total_steps, min_lr=cfg["training"]["min_lr"]
    )

    # ── AMP ──
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
        if "backbone" in ckpt:
            backbone.load_state_dict(ckpt["backbone"])
        # optimizer.load_state_dict(ckpt["optimizer"])
        ckpt_opt = ckpt.get("optimizer")
        if ckpt_opt and len(ckpt_opt["param_groups"]) == len(optimizer.param_groups):
            optimizer.load_state_dict(ckpt_opt)
        else:
            if is_main:
                print(f"[Resume] Skipping optimizer state — "
                      f"group count mismatch: ckpt={len(ckpt_opt['param_groups']) if ckpt_opt else 0} "
                      f"vs current={len(optimizer.param_groups)}")
        scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_pq = ckpt.get("best_pq", 0.0)
        global_step = start_epoch * steps_per_epoch
        if is_main:
            print(f"[Resume] epoch {start_epoch}, best_pq={best_pq:.4f}")

    # ── DDP wrap ──
    if is_distributed:
        decoder = DDP(decoder, device_ids=[local_rank], find_unused_parameters=False)
        # Only wrap backbone in DDP if it has trainable params
        if any(p.requires_grad for p in backbone.parameters()):
            backbone = DDP(backbone, device_ids=[local_rank], find_unused_parameters=True)

    # ── Training loop ──
    if is_main:
        print(f"\n{'='*70}")
        print(f"  Starting E2E training: epochs {start_epoch}→{tcfg['epochs']-1}")
        print(f"  Backbone LR: {backbone_lr:.1e}, Decoder LR: {tcfg['lr']:.1e}")
        print(f"  Unfreeze after epoch: {unfreeze_after}")
        print(f"{'='*70}\n")

    log_every = cfg["logging"]["log_every"]
    eval_every = cfg["logging"]["eval_every"]
    save_every = cfg["checkpoint"]["save_every"]

    decoder.train()
    if not backbone_is_frozen:
        backbone.train()

    for epoch in range(start_epoch, tcfg["epochs"]):
        # ── Delayed backbone unfreezing ──
        if backbone_is_frozen and epoch >= unfreeze_after and e2e_cfg["enabled"]:
            backbone_raw = backbone.module if hasattr(backbone, "module") else backbone
            trainable_names = unfreeze_backbone(backbone_raw, e2e_cfg)
            backbone_is_frozen = False
            
            # Add backbone params to optimizer
            new_params = [p for p in backbone_raw.parameters() if p.requires_grad]
            if new_params:
                optimizer.add_param_group({
                    "params": new_params,
                    "lr": backbone_lr,
                    "name": "backbone_unfrozen",
                })
            if is_main:
                n_new = sum(p.numel() for p in new_params)
                print(f"\n[E2E] *** BACKBONE UNFROZEN at epoch {epoch} ***")
                print(f"[E2E] Added {n_new:,} trainable backbone params (lr={backbone_lr:.1e})\n")

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_steps = 0
        optimizer.zero_grad()
        t_epoch = time.time()

        for step, batch in enumerate(train_loader):
            if killer.kill_now:
                if is_main:
                    save_checkpoint(
                        os.path.join(save_dir, "interrupted.pth"),
                        epoch, decoder, backbone, optimizer, scheduler, scaler, best_pq, cfg,
                    )
                    print(f"[SIGNAL] Saved interrupted.pth")
                    if wandb_active:
                        wandb.finish()
                if is_distributed:
                    dist.barrier()
                sys.exit(0)

            # Extract features (with or without backbone gradients)
            feats, coords = extract_backbone_features_e2e(
                backbone, batch, device, is_frozen=backbone_is_frozen
            )

            # Forward + loss
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

            # Optimizer step
            if (step + 1) % tcfg["grad_accum"] == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    # Clip both decoder and backbone gradients
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), tcfg["grad_clip"])
                    if not backbone_is_frozen:
                        backbone_params = [p for p in backbone.parameters() if p.requires_grad]
                        if backbone_params:
                            torch.nn.utils.clip_grad_norm_(backbone_params, tcfg["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), tcfg["grad_clip"])
                    if not backbone_is_frozen:
                        backbone_params = [p for p in backbone.parameters() if p.requires_grad]
                        if backbone_params:
                            torch.nn.utils.clip_grad_norm_(backbone_params, tcfg["grad_clip"])
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            # Logging
            if is_main and (step + 1) % log_every == 0:
                avg = epoch_loss / epoch_steps
                lr_dec = optimizer.param_groups[0]["lr"]
                lr_bb = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0
                frozen_str = " [BB FROZEN]" if backbone_is_frozen else ""
                print(f"  [E{epoch} S{step+1}/{len(train_loader)}] "
                      f"loss={avg:.4f} lr_dec={lr_dec:.2e} lr_bb={lr_bb:.2e}{frozen_str}")
                log_wandb({
                    "train/loss": avg, "train/lr_decoder": lr_dec,
                    "train/lr_backbone": lr_bb,
                    "train/backbone_frozen": int(backbone_is_frozen),
                }, step=global_step)

        # Epoch end
        epoch_time = time.time() - t_epoch
        if is_main:
            avg_loss = epoch_loss / max(epoch_steps, 1)
            gpu_mem = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0
            torch.cuda.reset_peak_memory_stats()
            print(f"Epoch {epoch}: loss={avg_loss:.4f}, time={epoch_time:.1f}s, gpu_peak={gpu_mem:.0f}MB")
            log_wandb({"epoch/loss": avg_loss, "epoch/time_s": epoch_time, "epoch/gpu_peak_mb": gpu_mem}, step=global_step)

        # ── Eval ──
        if (epoch + 1) % eval_every == 0 or epoch == tcfg["epochs"] - 1:
            if is_main:
                print("  Evaluating...")
                decoder_raw = decoder.module if is_distributed else decoder
                backbone_raw = backbone.module if hasattr(backbone, "module") else backbone
                metrics = evaluate(decoder_raw, backbone_raw, val_loader, device, cfg)
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
                        epoch, decoder, backbone, optimizer, scheduler, scaler, best_pq, cfg, metrics,
                    )
                    print(f"  ★ New best PQ={best_pq:.4f}")

            if is_distributed:
                dist.barrier()
            decoder.train()
            if not backbone_is_frozen:
                backbone_raw = backbone.module if hasattr(backbone, "module") else backbone
                backbone_raw.train()

        # ── Save latest ──
        if (epoch + 1) % save_every == 0 or epoch == tcfg["epochs"] - 1:
            if is_main:
                save_checkpoint(
                    os.path.join(save_dir, "latest.pth"),
                    epoch, decoder, backbone, optimizer, scheduler, scaler, best_pq, cfg,
                )
            if is_distributed:
                dist.barrier()

    if is_main:
        print(f"\n{'='*70}")
        print(f"  E2E Training complete. Best PQ: {best_pq:.4f}")
        print(f"{'='*70}")
        if wandb_active:
            wandb.finish()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
