"""Panoptic decoder training with Dense Camera-CLIP Supervision (Job 1).

Extends train_panoptic_token.py with:
- LoRA sparse conv adapters on backbone (B1)
- Single CLIP embedding head (A4) replacing dual head
- 216+ negative bank contrastive loss (L5/C1)
- Multi-prompt template augmentation (T4)
- [Job 1b] Camera CLIP distillation losses (L_O, L_V)
- [Job 1c] CASD context-aware stuff targets + L7 voxel decomposition

Usage:
    python -u panoptic/train/train_panoptic_job1.py --config panoptic/configs/job1a.yaml
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import MinkowskiEngine as ME
import yaml

sys.path.insert(0, "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene")

from models.disnet import DisNet, state_dict_remove_moudle
from panoptic.models.mask_decoder import PanopticDecoder
from panoptic.models.lora_backbone import insert_lora_adapters
from panoptic.losses.panoptic_loss import PanopticLoss
from panoptic.losses.token_loss_c1 import TokenLossC1
from panoptic.losses.matcher import HungarianMatcher
from panoptic.eval.eval_pq import PanopticEval
from panoptic.dataset.nuscenes_panoptic import (
    NuScenesPanoptic, NUM_CLASSES, THING_IDS, STUFF_IDS,
)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

def load_config(config_path):
    """Load YAML config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_nested(cfg, key, default=None):
    """Get nested config value via dot-separated key."""
    parts = key.split(".")
    val = cfg
    for p in parts:
        if isinstance(val, dict) and p in val:
            val = val[p]
        else:
            return default
    return val


# ══════════════════════════════════════════════════════════════════════════════
# Backbone
# ══════════════════════════════════════════════════════════════════════════════

def load_backbone(ckpt_path, device):
    """Load frozen DisNet backbone."""
    class Cfg:
        feature_2d_extractor = "openseg"
        arch_3d = "MinkUNet18A"

    model = DisNet(Cfg()).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = state_dict_remove_moudle(ckpt["state_dict"])
    model.load_state_dict(state, strict=True)

    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Backbone loaded: {n_params:,} params (frozen)")
    return model


def extract_backbone_features(backbone, batch, device, lora_adapters=None):
    """Extract backbone features with optional LoRA adapters.

    When LoRA adapters are installed via hooks, the backbone forward pass
    automatically applies them — no special handling needed here.

    Returns:
        feats: (1, V, 768) backbone features
        coords: (1, V, 3) xyz coordinates
    """
    vox_coords = batch["vox_coords"].to(device)  # (V, 3)
    vox_feats = batch["vox_feats"].to(device)    # (V, 3)
    V = vox_coords.shape[0]

    # Prepend batch index for ME
    batch_coords = torch.cat([
        torch.zeros(V, 1, dtype=torch.int32, device=device),
        vox_coords.int()
    ], dim=1)

    sinput = ME.SparseTensor(
        features=vox_feats.float(),
        coordinates=batch_coords,
        device=device,
    )

    # If LoRA adapters are active, they fire via forward hooks
    # We need grad for L7 (voxel decomposition) which trains through LoRA
    if lora_adapters is not None:
        output = backbone(sinput)
    else:
        with torch.no_grad():
            output = backbone(sinput)

    feats = output.F if hasattr(output, "F") else output  # (V, 768)
    coords_xyz = vox_feats[:, :3]  # (V, 3) — raw XYZ

    return feats.unsqueeze(0), coords_xyz.unsqueeze(0)


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def build_optimizer(decoder, lora_adapters, cfg):
    """Build optimizer with differential LR for decoder, head, and LoRA."""
    lr_decoder = get_nested(cfg, "training.lr_decoder", 5e-5)
    lr_head = get_nested(cfg, "training.lr_token_head", 1e-4)
    lr_lora = get_nested(cfg, "training.lr_lora", 1e-4)
    weight_decay = get_nested(cfg, "training.weight_decay", 0.01)

    # Separate param groups
    head_names = ["clip_embed_head", "objectness_head", "log_temperature"]
    head_params = []
    decoder_params = []

    for name, p in decoder.named_parameters():
        if not p.requires_grad:
            continue
        if any(hn in name for hn in head_names):
            head_params.append(p)
        else:
            decoder_params.append(p)

    param_groups = [
        {"params": decoder_params, "lr": lr_decoder, "name": "decoder"},
        {"params": head_params, "lr": lr_head, "name": "head"},
    ]

    if lora_adapters is not None:
        lora_params = [p for p in lora_adapters.parameters() if p.requires_grad]
        param_groups.append({"params": lora_params, "lr": lr_lora, "name": "lora"})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    return optimizer


def get_loss_weight(loss_name, epoch, cfg):
    """Linear warmup for distillation losses."""
    phase1_epochs = get_nested(cfg, "training.phase1_epochs", 2)
    phase2_epoch = max(0, epoch - phase1_epochs)

    schedules = {
        "L_O": (get_nested(cfg, "distillation.L_O.warmup_epochs", 2),
                get_nested(cfg, "distillation.L_O.weight", 1.0)),
        "L_V": (get_nested(cfg, "distillation.L_V.warmup_epochs", 5),
                get_nested(cfg, "distillation.L_V.weight", 0.1)),
        "L7": (get_nested(cfg, "voxel_decomp.warmup_epochs", 5),
               get_nested(cfg, "voxel_decomp.weight", 0.5)),
    }

    if loss_name not in schedules:
        return 1.0

    warmup_epochs, target_weight = schedules[loss_name]
    if warmup_epochs <= 0:
        return target_weight

    ramp = min(1.0, phase2_epoch / warmup_epochs)
    return target_weight * ramp


def save_checkpoint(save_dir, epoch, decoder, lora_adapters, optimizer, scheduler,
                    scaler, best_pq, cfg, global_step=0, token_criterion=None):
    """Save training checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_pq": best_pq,
        "config": cfg,
    }
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if lora_adapters is not None:
        state["lora_adapters"] = lora_adapters.state_dict()
    if token_criterion is not None:
        state["token_criterion"] = token_criterion.state_dict()

    path = os.path.join(save_dir, f"epoch_{epoch:03d}.pth")
    torch.save(state, path)

    # Also save as latest
    latest_path = os.path.join(save_dir, "latest.pth")
    torch.save(state, latest_path)

    return path


def panoptic_inference(pred_logits, pred_masks, overlap_thr=0.8, min_pts=50):
    """Greedy panoptic assignment from decoder outputs."""
    Q = pred_logits.shape[0]
    V = pred_masks.shape[0]

    scores, labels = pred_logits.max(dim=-1)  # (Q,)
    mask_probs = pred_masks.sigmoid()  # (V, Q)

    # Sort by score descending
    sorted_idx = scores.argsort(descending=True)

    dev = mask_probs.device
    pan_seg = torch.zeros(V, dtype=torch.long, device=dev)
    instance_id = 0
    used = torch.zeros(V, dtype=torch.bool, device=dev)

    for idx in sorted_idx:
        label = labels[idx].item()
        if label == NUM_CLASSES:  # no-object
            continue

        mask = mask_probs[:, idx] > 0.5
        mask = mask & ~used

        if mask.sum() < min_pts:
            continue

        # Check overlap with existing assignments
        if used.any():
            overlap = (mask & used).sum().float() / mask.sum().float()
            if overlap > overlap_thr:
                continue

        instance_id += 1
        pan_seg[mask] = label * 1000 + instance_id
        if label < 10:  # thing classes use instance IDs
            used[mask] = True
        else:  # stuff classes can overlap (same class merges)
            used[mask] = True

    return pan_seg


def panoptic_inference_clip(pred_embed, pred_masks, text_protos, temperature,
                            objectness=None, overlap_thr=0.8, min_pts=50):
    """Panoptic assignment using CLIP embeddings (single-head mode)."""
    Q = pred_embed.shape[0]
    V = pred_masks.shape[0]

    # Classify via cosine similarity
    sim = pred_embed @ text_protos.T / temperature  # (Q, C)
    scores, labels = sim.max(dim=-1)  # (Q,)

    # Apply objectness if available
    if objectness is not None:
        scores = scores * objectness.squeeze(-1).sigmoid()

    mask_probs = pred_masks.sigmoid()  # (V, Q)

    # Sort by score descending
    sorted_idx = scores.argsort(descending=True)

    dev = mask_probs.device
    pan_seg = torch.zeros(V, dtype=torch.long, device=dev)
    instance_id = 0
    used = torch.zeros(V, dtype=torch.bool, device=dev)

    for idx in sorted_idx:
        label = labels[idx].item()
        mask = mask_probs[:, idx] > 0.5
        mask = mask & ~used

        if mask.sum() < min_pts:
            continue

        instance_id += 1
        pan_seg[mask] = label * 1000 + instance_id
        used[mask] = True

    return pan_seg


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Data ──
    dataroot = get_nested(cfg, "data.dataroot")
    version = get_nested(cfg, "data.version", "v1.0-trainval")
    voxel_size = get_nested(cfg, "data.voxel_size", 0.05)
    clip_feat_dir = get_nested(cfg, "distillation.clip_feat_dir", None)

    train_dataset = NuScenesPanoptic(
        dataroot=dataroot, version=version, split="train",
        voxel_size=voxel_size, augment=True,
        aug_cfg=cfg.get("augmentation", {}),
        clip_feat_dir=clip_feat_dir,
    )
    val_dataset = NuScenesPanoptic(
        dataroot=dataroot, version=version, split="val",
        voxel_size=voxel_size, augment=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        num_workers=get_nested(cfg, "training.num_workers", 0),
        collate_fn=lambda x: x[0],  # single sample per batch
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=lambda x: x[0],
    )

    # ── Backbone + LoRA ──
    backbone_ckpt = get_nested(cfg, "model.backbone_ckpt", "models/nuscenes_openseg.pth.tar")
    backbone = load_backbone(backbone_ckpt, device)

    lora_adapters = None
    lora_hooks = None
    if get_nested(cfg, "lora.enabled", False):
        rank = get_nested(cfg, "lora.rank", 16)
        dropout = get_nested(cfg, "lora.dropout", 0.05)
        lora_adapters, lora_hooks = insert_lora_adapters(
            backbone, rank=rank, dropout=dropout
        )
        lora_adapters = lora_adapters.to(device)

    # ── Decoder ──
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
        use_token_head=True,  # keep for warm-start compatibility
        token_dim=get_nested(cfg, "model.clip_embed_dim", 768),
    ).to(device)

    # Enable single CLIP head if configured
    head_type = get_nested(cfg, "model.head_type", "dual")
    if head_type == "single_clip":
        decoder.enable_single_clip_head(
            clip_dim=get_nested(cfg, "model.clip_embed_dim", 768)
        )
        decoder = decoder.to(device)

    # ── Warm-start ──
    warm_start = get_nested(cfg, "warm_start.checkpoint")
    warm_start_ckpt = None
    if warm_start and os.path.exists(warm_start):
        print(f"Warm-starting from {warm_start}")
        warm_start_ckpt = torch.load(warm_start, map_location="cpu")

        # Load decoder state
        state = warm_start_ckpt.get("decoder", warm_start_ckpt.get("model", warm_start_ckpt))
        # strict=False: allows missing clip_embed_head / objectness_head keys
        missing, unexpected = decoder.load_state_dict(state, strict=False)
        if missing:
            print(f"  Missing keys (expected for new heads): {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")

        # Load LoRA adapters if available
        if lora_adapters is not None and "lora_state" in warm_start_ckpt:
            lora_adapters.load_state_dict(warm_start_ckpt["lora_state"], strict=False)
            print(f"  LoRA adapters restored ({sum(p.numel() for p in lora_adapters.parameters())} params)")

    # ── Losses ──
    # Panoptic loss (mask + dice + CE)
    cost_class = 0.0 if head_type == "single_clip" else get_nested(cfg, "loss.cost_class", 2.0)
    criterion = PanopticLoss(
        num_classes=get_nested(cfg, "model.num_classes", 16),
        eos_coef=get_nested(cfg, "loss.eos_coef", 0.1),
        weight_ce=get_nested(cfg, "loss.weight_ce", 2.0) if head_type != "single_clip" else 0.0,
        weight_mask=get_nested(cfg, "loss.weight_mask", 5.0),
        weight_dice=get_nested(cfg, "loss.weight_dice", 5.0),
        cost_class=cost_class,
        cost_mask=get_nested(cfg, "loss.cost_mask", 5.0),
        cost_dice=get_nested(cfg, "loss.cost_dice", 5.0),
        p_ratio=get_nested(cfg, "loss.p_ratio", 0.75),
    ).to(device)

    # Token loss C1 (216+ negatives)
    token_criterion = None
    base_protos_path = get_nested(cfg, "token.prototype_path",
                                  "panoptic/clip_prototypes/text_prototypes_vitl14_224.pt")
    neg_bank_path = get_nested(cfg, "token.negative_bank")
    if neg_bank_path and os.path.exists(neg_bank_path):
        multiprompt_path = get_nested(cfg, "token.multiprompt_path")
        token_criterion = TokenLossC1(
            base_protos_path=base_protos_path,
            negative_bank_path=neg_bank_path,
            multiprompt_path=multiprompt_path,
            learnable_temperature=get_nested(cfg, "token.learnable_temperature", True),
            temp_max=get_nested(cfg, "token.temp_max", 5.0),
            margin=get_nested(cfg, "token.margin", 0.1),
            exclude_novel_from_denom=get_nested(cfg, "token.exclude_novel_from_denom", True),
        ).to(device)

        # Restore token_criterion from warm-start checkpoint if available
        if warm_start_ckpt is not None and "token_criterion" in warm_start_ckpt:
            if get_nested(cfg, "token.restore_from_warm_start", True):
                token_criterion.load_state_dict(warm_start_ckpt["token_criterion"])
                temp_val = token_criterion.temperature.item()
                print(f"  Token criterion restored (temperature={temp_val:.4f})")
            else:
                print(f"  Token criterion NOT restored from warm-start (token.restore_from_warm_start=false)")
                print(f"  Temperature initialized at {token_criterion.temperature.item():.4f}")

    if token_criterion is not None:
        print(f"  Token loss active — temperature={token_criterion.temperature.item():.4f}, "
              f"n_total={token_criterion.n_total}, n_base_train={token_criterion.n_base_train}")


    distill_obj_loss = None
    distill_vox_loss = None
    if get_nested(cfg, "distillation.enabled", False):
        from panoptic.losses.distill_object_loss import ObjectDistillLoss
        from panoptic.losses.distill_voxel_loss import VoxelDistillLoss
        distill_obj_loss = ObjectDistillLoss(
            loss_type=get_nested(cfg, "distillation.L_O.loss_type", "cosine"),
        ).to(device)
        distill_vox_loss = VoxelDistillLoss(
            loss_type=get_nested(cfg, "distillation.L_V.loss_type", "cosine"),
        ).to(device)
        print(f"Distillation enabled: L_O + L_V")

    # ── Optimizer ──
    optimizer = build_optimizer(decoder, lora_adapters, cfg)

    # Add token_criterion's learnable temperature to optimizer
    if token_criterion is not None and hasattr(token_criterion, "log_temperature"):
        optimizer.add_param_group({
            "params": [token_criterion.log_temperature],
            "lr": get_nested(cfg, "training.lr_token_head", 1e-4),
            "name": "token_temperature",
        })

    # ── Scheduler ──
    epochs = get_nested(cfg, "training.epochs", 20)
    warmup_epochs = get_nested(cfg, "training.warmup_epochs", 2)
    min_lr = get_nested(cfg, "training.min_lr", 1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=min_lr
    )

    # ── AMP scaler ──
    use_amp = get_nested(cfg, "training.mixed_precision", True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Training config ──
    grad_accum = get_nested(cfg, "training.grad_accum", 8)
    grad_clip = get_nested(cfg, "training.grad_clip", 0.5)
    log_every = get_nested(cfg, "logging.log_every", 50)
    eval_every = get_nested(cfg, "logging.eval_every", 2)
    save_dir = get_nested(cfg, "checkpoint.save_dir", "panoptic/checkpoints_job1")
    phase1_epochs = get_nested(cfg, "training.phase1_epochs", 2)
    token_weight = get_nested(cfg, "token.weight", 1.0)

    # Load text prototypes for inference evaluation
    text_protos = None
    if os.path.exists(base_protos_path):
        proto_data = torch.load(base_protos_path, map_location=device)
        if isinstance(proto_data, dict):
            text_protos = proto_data["embeddings"].float().to(device)
        else:
            text_protos = proto_data.float().to(device)
        text_protos = F.normalize(text_protos, dim=-1)

    # ── WandB ──
    use_wandb = HAS_WANDB and get_nested(cfg, "logging.wandb_enabled", False)
    if use_wandb:
        wandb.init(
            project=get_nested(cfg, "logging.wandb_project", "openscene-panoptic"),
            name=get_nested(cfg, "logging.wandb_run_name", "job1"),
            config=cfg,
            resume="allow",
        )

    # ══════════════════════════════════════════════════════════════════════
    # Training loop
    # ══════════════════════════════════════════════════════════════════════
    best_pq = 0.0
    global_step = 0
    last_lora_gnorm = 0.0
    start_epoch = 0

    # ── Resume from checkpoint ──
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location=device)
        # Decoder state
        dec_state = resume_ckpt.get("decoder", resume_ckpt.get("decoder_state", None))
        if dec_state is not None:
            decoder.load_state_dict(dec_state, strict=False)
        # LoRA state
        lora_state = resume_ckpt.get("lora_adapters", resume_ckpt.get("lora_state", None))
        if lora_adapters is not None and lora_state is not None:
            lora_adapters.load_state_dict(lora_state, strict=False)
        # Token criterion (has learnable temperature)
        if token_criterion is not None and "token_criterion" in resume_ckpt:
            token_criterion.load_state_dict(resume_ckpt["token_criterion"])
        # Optimizer
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        # Scaler
        if "scaler" in resume_ckpt:
            scaler.load_state_dict(resume_ckpt["scaler"])
        # Epoch / step
        if "epoch" in resume_ckpt:
            start_epoch = resume_ckpt["epoch"] + 1
        if "global_step" in resume_ckpt:
            global_step = resume_ckpt["global_step"]
        if "best_pq" in resume_ckpt:
            best_pq = resume_ckpt["best_pq"]
        print(f"  Resumed at epoch {start_epoch}, step {global_step}, best_pq={best_pq:.4f}")

    for epoch in range(start_epoch, epochs):
        is_phase1 = (epoch < phase1_epochs)

        # Phase 1: freeze decoder base, train only LoRA + new heads
        if is_phase1:
            for name, p in decoder.named_parameters():
                if any(x in name for x in ["clip_embed_head", "objectness_head"]):
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        else:
            for p in decoder.parameters():
                p.requires_grad = True

        decoder.train()
        if lora_adapters is not None:
            lora_adapters.train()

        epoch_losses = []
        t_start = time.time()

        for step, batch in enumerate(train_loader):
            # ── Forward ──
            with torch.cuda.amp.autocast(enabled=use_amp):
                feats, coords = extract_backbone_features(
                    backbone, batch, device, lora_adapters
                )

                outputs = decoder(feats, coords)

                # Build targets (B12/N4: exclude novel classes from supervision)
                NOVEL_IDS_TRAIN = {2, 5, 6, 15}  # bus, motorcycle, pedestrian, vegetation
                gt_classes_all = batch["gt_classes"].to(device)
                gt_masks_all = batch["gt_masks"].float().to(device)

                # Filter to base classes only
                base_mask = torch.tensor(
                    [c.item() not in NOVEL_IDS_TRAIN for c in gt_classes_all],
                    dtype=torch.bool, device=device
                )
                targets = {
                    "classes": [gt_classes_all[base_mask]],
                    "masks": [gt_masks_all[base_mask]],
                }

                # Panoptic loss (mask + dice, CE disabled in single_clip mode)
                panoptic_losses, indices = criterion(outputs, targets)

                total_loss = sum(panoptic_losses.values())
                loss_mask_val = panoptic_losses.get("loss_mask", torch.tensor(0.0)).item()
                loss_dice_val = panoptic_losses.get("loss_dice", torch.tensor(0.0)).item()

                # Objectness loss (single_clip mode)
                # Trained class-agnostically on ALL GT masks (base + novel) so that
                # novel-attending queries are not forced to have objectness=0.
                loss_obj_val = 0.0
                if head_type == "single_clip" and "pred_obj" in outputs:
                    obj_scores = outputs["pred_obj"][0].squeeze(-1)  # (Q,)
                    obj_target = torch.zeros_like(obj_scores)

                    # Run matcher over full GT (base + novel) for objectness only
                    all_targets_obj = {
                        "classes": [gt_classes_all],
                        "masks":   [gt_masks_all.float()],
                    }
                    _, indices_all = criterion(outputs, all_targets_obj)
                    pred_idx_all = indices_all[0][0]
                    if len(pred_idx_all) > 0:
                        obj_target[pred_idx_all] = 1.0

                    loss_obj = F.binary_cross_entropy_with_logits(obj_scores, obj_target)
                    loss_obj_val = loss_obj.item()
                    total_loss = total_loss + get_nested(cfg, "loss.weight_obj", 2.0) * loss_obj

                # Token loss C1
                loss_token_val = 0.0
                if token_criterion is not None and "pred_embed" in outputs:
                    loss_token = token_criterion(outputs["pred_embed"], targets, indices)
                    loss_token_val = loss_token.item()
                    total_loss = total_loss + token_weight * loss_token

                # Distillation losses (Job 1b)
                loss_lo_val = 0.0
                loss_lv_val = 0.0
                if distill_obj_loss is not None and "obj_clip_feats" in batch:
                    obj_clip = batch["obj_clip_feats"].to(device).unsqueeze(0)
                    obj_cls = batch["obj_clip_classes"].to(device).unsqueeze(0)
                    vox_clip = batch["vox_clip_feats"].to(device).unsqueeze(0)
                    vox_idx_clip = batch["vox_clip_indices"].to(device).unsqueeze(0)

                    w_lo = get_loss_weight("L_O", epoch, cfg)
                    if w_lo > 0 and obj_clip.shape[1] > 0:
                        loss_lo = distill_obj_loss(
                            outputs["pred_embed"], obj_clip, obj_cls, indices, targets
                        )
                        loss_lo_val = loss_lo.item()
                        total_loss = total_loss + w_lo * loss_lo

                    w_lv = get_loss_weight("L_V", epoch, cfg)
                    if w_lv > 0:
                        loss_lv = distill_vox_loss(
                            outputs["pred_embed"], outputs["pred_masks"],
                            vox_clip, vox_idx_clip
                        )
                        loss_lv_val = loss_lv.item()
                        total_loss = total_loss + w_lv * loss_lv

                total_loss = total_loss / grad_accum

            # ── Backward ──
            scaler.scale(total_loss).backward()

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                # Clip all param groups
                all_params = []
                for group in optimizer.param_groups:
                    all_params.extend([p for p in group["params"] if p.grad is not None])
                if all_params:
                    torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
                # Capture lora gnorm after unscale, before zero_grad
                if lora_adapters is not None:
                    last_lora_gnorm = (sum(
                        p.grad.norm().item()**2 for p in lora_adapters.parameters()
                        if p.grad is not None
                    ))**0.5
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            global_step += 1
            epoch_losses.append(total_loss.item() * grad_accum)

            # Logging
            if step % log_every == 0:
                avg_loss = np.mean(epoch_losses[-log_every:]) if epoch_losses else 0
                lora_gnorm = last_lora_gnorm if "last_lora_gnorm" in dir() else 0.0
                # Learnable temperature
                temp_val = 0.0
                if token_criterion is not None and hasattr(token_criterion, "temperature"):
                    temp_val = token_criterion.temperature.item()

                print(f"[E{epoch:02d} S{step:04d}] loss={avg_loss:.4f} "
                      f"mask={loss_mask_val:.3f} dice={loss_dice_val:.3f} "
                      f"obj={loss_obj_val:.3f} token={loss_token_val:.3f} "
                      f"lo={loss_lo_val:.3f} lv={loss_lv_val:.3f} "
                      f"temp={temp_val:.3f} lora_gnorm={lora_gnorm:.4e} phase={'1' if is_phase1 else '2'}")

                if use_wandb:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/loss_mask": loss_mask_val,
                        "train/loss_dice": loss_dice_val,
                        "train/loss_obj": loss_obj_val,
                        "train/loss_token": loss_token_val,
                        "train/loss_lo": loss_lo_val,
                        "train/loss_lv": loss_lv_val,
                        "train/token_weight": token_weight,
                        "train/temperature": temp_val,
                        "train/lora_gnorm": lora_gnorm,
                        "train/phase": 1 if is_phase1 else 2,
                        "train/lr_decoder": optimizer.param_groups[0]["lr"],
                        "train/lr_head": optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0,
                        "train/lr_lora": optimizer.param_groups[2]["lr"] if len(optimizer.param_groups) > 2 else 0,
                        "train/epoch": epoch,
                    }, step=global_step)

            # Smoke test early exit
            if args.smoke_test and step >= 50:
                break

        scheduler.step()
        elapsed = time.time() - t_start
        avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0
        print(f"Epoch {epoch} done in {elapsed:.0f}s, avg_loss={avg_epoch_loss:.4f}")

        # ── Evaluation ──
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            decoder.eval()
            evaluator = PanopticEval(NUM_CLASSES, min_points=50)

            # B12/N4 split (0-indexed class IDs)
            NOVEL_IDS = [2, 5, 6, 15]  # bus, motorcycle, pedestrian, vegetation
            BASE_IDS = [i for i in range(NUM_CLASSES) if i not in NOVEL_IDS]

            # Zero-shot novel concepts (not in nuScenes label set)
            ZEROSHOT_CONCEPTS = ["ambulance", "trash can", "fire hydrant", "police van"]
            if text_protos is not None:
                import clip as _clip
                zs_tokens = _clip.tokenize([f"a {c} in a scene" for c in ZEROSHOT_CONCEPTS]).to(device)
                # Reuse the CLIP model isn't loaded here, so encode from scratch
                # Instead, precompute once and cache
                if not hasattr(main, '_zs_protos'):
                    # Load CLIP for zero-shot eval (one-time)
                    _clip_model, _ = _clip.load("ViT-L/14", device=device, jit=False)
                    _clip_model.eval()
                    with torch.no_grad():
                        zs_emb = _clip_model.encode_text(zs_tokens).float()
                        zs_emb = zs_emb / zs_emb.norm(dim=-1, keepdim=True)
                    main._zs_protos = zs_emb  # (4, 768)
                    del _clip_model
                    torch.cuda.empty_cache()
                zs_protos = main._zs_protos

            # Voxel-level zero-shot counters
            zs_voxel_counts = [0] * len(ZEROSHOT_CONCEPTS)
            zs_voxel_confidence = [0.0] * len(ZEROSHOT_CONCEPTS)
            total_voxels = 0

            with torch.no_grad():
                for val_batch in val_loader:
                    feats, coords = extract_backbone_features(
                        backbone, val_batch, device, lora_adapters
                    )
                    outputs = decoder(feats, coords)

                    # Inference
                    if head_type == "single_clip" and text_protos is not None:
                        pred_embed = outputs["pred_embed"][0]
                        pred_masks = outputs["pred_masks"][0]
                        obj_scores = outputs.get("pred_obj", [None])[0]
                        temp = token_criterion.temperature.item() if token_criterion is not None else decoder.clip_embed_head.temperature.item()
                        pan_pred = panoptic_inference_clip(
                            pred_embed, pred_masks, text_protos, temp,
                            obj_scores
                        )

                        # Zero-shot novel voxel check
                        # Per-voxel embed from mask-weighted queries
                        mask_probs = pred_masks.sigmoid()  # (V, Q)
                        # Weighted sum of query embeds per voxel
                        voxel_embeds = mask_probs @ pred_embed  # (V, 768)
                        voxel_embeds = F.normalize(voxel_embeds, dim=-1)
                        # Similarity to zero-shot concepts
                        zs_sim = voxel_embeds @ zs_protos.T  # (V, 4)
                        zs_sim = zs_sim * temp
                        # Count voxels where zero-shot concept is top match
                        # (compare against all 16 base protos too)
                        all_sim = voxel_embeds @ text_protos.T  # (V, 16)
                        max_base_sim, _ = all_sim.max(dim=-1)  # (V,)
                        for zi in range(len(ZEROSHOT_CONCEPTS)):
                            above_base = zs_sim[:, zi] > max_base_sim
                            zs_voxel_counts[zi] += above_base.sum().item()
                            if above_base.sum() > 0:
                                zs_voxel_confidence[zi] += zs_sim[above_base, zi].mean().item()
                        total_voxels += voxel_embeds.shape[0]
                    else:
                        pan_pred = panoptic_inference(
                            outputs["pred_logits"][0],
                            outputs["pred_masks"][0],
                        )

                    # GT
                    gt_sem = val_batch["gt_sem"].numpy()
                    gt_masks = val_batch["gt_masks"].numpy()
                    gt_classes = val_batch["gt_classes"].numpy()
                    V = len(gt_sem)

                    # Build GT panoptic
                    gt_pan = np.zeros(V, dtype=np.int64)
                    for m_idx in range(len(gt_classes)):
                        cls = gt_classes[m_idx]
                        mask = gt_masks[m_idx]
                        gt_pan[mask] = cls * 1000 + (m_idx + 1)

                    pred_sem = pan_pred.cpu().numpy() // 1000
                    pred_inst = pan_pred.cpu().numpy() % 1000
                    gt_sem_eval = gt_pan // 1000
                    gt_inst_eval = gt_pan % 1000

                    evaluator.addBatch(pred_sem.astype(np.int32),
                                       pred_inst.astype(np.int32),
                                       gt_sem_eval.astype(np.int32),
                                       gt_inst_eval.astype(np.int32))

            pq_mean, sq_mean, rq_mean, pq_per, sq_per, rq_per = evaluator.getPQ()

            # Compute base/novel splits
            pq_base = np.mean([pq_per[i] for i in BASE_IDS if i < len(pq_per)])
            pq_novel = np.mean([pq_per[i] for i in NOVEL_IDS if i < len(pq_per)])

            print(f"  Val PQ_all={pq_mean:.4f} PQ_base={pq_base:.4f} PQ_novel={pq_novel:.4f}")
            print(f"      SQ={sq_mean:.4f} RQ={rq_mean:.4f}")

            # Zero-shot report
            if text_protos is not None and total_voxels > 0:
                print(f"  Zero-shot novel voxels (out of {total_voxels} total):")
                for zi, concept in enumerate(ZEROSHOT_CONCEPTS):
                    pct = 100.0 * zs_voxel_counts[zi] / total_voxels
                    print(f"    {concept}: {zs_voxel_counts[zi]} voxels ({pct:.2f}%)")

            if use_wandb:
                log_dict = {
                    "val/PQ_all": pq_mean,
                    "val/PQ_base": pq_base,
                    "val/PQ_novel": pq_novel,
                    "val/SQ": sq_mean,
                    "val/RQ": rq_mean,
                }
                # Per-class PQ
                class_names_short = ["barrier", "bicycle", "bus", "car",
                    "constr_veh", "motorcycle", "pedestrian", "cone",
                    "trailer", "truck", "drive_surf", "other_flat",
                    "sidewalk", "terrain", "manmade", "vegetation"]
                for i, name in enumerate(class_names_short):
                    if i < len(pq_per):
                        log_dict[f"val_pq/{name}"] = pq_per[i]
                # Zero-shot voxel counts
                if total_voxels > 0:
                    for zi, concept in enumerate(ZEROSHOT_CONCEPTS):
                        log_dict[f"val_zeroshot/{concept.replace(' ', '_')}_voxels"] = zs_voxel_counts[zi]
                        log_dict[f"val_zeroshot/{concept.replace(' ', '_')}_pct"] = 100.0 * zs_voxel_counts[zi] / total_voxels
                wandb.log(log_dict, step=global_step)

            if pq_mean > best_pq:
                best_pq = pq_mean
                path = save_checkpoint(save_dir, epoch, decoder, lora_adapters,
                                       optimizer, scheduler, scaler, best_pq, cfg,
                                       global_step=global_step, token_criterion=token_criterion)
                # Save as best
                best_path = os.path.join(save_dir, "best.pth")
                torch.save(torch.load(path, map_location="cpu"), best_path)
                print(f"  New best PQ={best_pq:.4f}, saved to {best_path}")

        # Save periodically
        save_every = get_nested(cfg, "checkpoint.save_every", 5)
        if (epoch + 1) % save_every == 0:
            save_checkpoint(save_dir, epoch, decoder, lora_adapters,
                            optimizer, scheduler, scaler, best_pq, cfg,
                            global_step=global_step, token_criterion=token_criterion)

    print(f"\nTraining complete. Best PQ={best_pq:.4f}")
    print(f"Checkpoints in: {save_dir}")

    if use_wandb:
        best_path = os.path.join(save_dir, "best.pth")
        if os.path.exists(best_path):
            artifact = wandb.Artifact("best-checkpoint", type="model",
                                      metadata={"best_pq": float(best_pq)})
            artifact.add_file(best_path)
            wandb.log_artifact(artifact)
        wandb.finish()


if __name__ == "__main__":
    main()
