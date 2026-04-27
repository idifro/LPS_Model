"""Panoptic decoder training with CLIP token head (Phase 5).

Extends train_panoptic.py with:
- CLIP token head on decoder queries
- Token loss (cosine similarity to text prototypes)
- Warm-start from Phase 4 best.pth
- Token loss warmup schedule

Usage:
    python -u panoptic/train/train_panoptic_token.py --config panoptic/configs/mini_token.yaml
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
from panoptic.losses.token_loss import CLIPTokenLoss, token_warmup_weight
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
# Reuse utilities from train_panoptic.py
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
    p = argparse.ArgumentParser(description="Panoptic token head training")
    p.add_argument("--config", default="panoptic/configs/mini_token.yaml")
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
    print(f"[Backbone] Loaded DisNet from {ckpt_path}, frozen.", flush=True)
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
    else:
        raise ValueError(f"Unknown scheduler: {sched_type}")
    print(f"[Scheduler] {sched_type}, step_per={step_per}", flush=True)
    return scheduler, step_per


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


def panoptic_inference_token(pred_tokens, pred_masks, text_prototypes, num_classes=16, overlap_thr=0.8):
    """Token-based panoptic inference: classify by cosine similarity to text prototypes."""
    # pred_tokens: (Q, token_dim), pred_masks: (V, Q), text_prototypes: (num_classes, token_dim)
    cos_sim = torch.mm(pred_tokens, text_prototypes.t())  # (Q, num_classes)
    scores, labels = cos_sim.max(-1)  # (Q,)

    masks = pred_masks.sigmoid()
    V = masks.shape[0]

    # No "no-object" class in token space — use score threshold
    keep = scores > 0.1
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


def evaluate(decoder, backbone, dataloader, device, cfg, text_prototypes=None):
    num_classes = cfg["model"]["num_classes"]
    overlap_thr = cfg["inference"]["overlap_thr"]
    decoder.eval()

    evaluator_linear = PanopticEval(n_classes=num_classes, min_points=cfg["inference"]["min_points"])
    evaluator_token = PanopticEval(n_classes=num_classes, min_points=cfg["inference"]["min_points"]) if text_prototypes is not None else None

    with torch.no_grad():
        for batch in dataloader:
            feats, coords = extract_backbone_features(backbone, batch, device)
            outputs = decoder(feats, coords)
            pred_logits = outputs["pred_logits"][0].cpu()
            pred_masks = outputs["pred_masks"][0].cpu()

            # Linear head inference
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
            evaluator_linear.addBatch(
                sem_pred[valid], inst_pred[valid],
                gt_sem[valid], gt_inst[valid],
            )

            # Token-based inference
            if evaluator_token is not None and "pred_tokens" in outputs:
                pred_tokens = outputs["pred_tokens"][0].cpu()
                sem_pred_t, inst_pred_t = panoptic_inference_token(
                    pred_tokens, pred_masks, text_prototypes.cpu(), num_classes, overlap_thr
                )
                evaluator_token.addBatch(
                    sem_pred_t[valid], inst_pred_t[valid],
                    gt_sem[valid], gt_inst[valid],
                )

    PQ, SQ, RQ, _, _, _ = evaluator_linear.getPQ()
    iou_mean, _ = evaluator_linear.getSemIoU()
    results = {"PQ": PQ, "SQ": SQ, "RQ": RQ, "mIoU": iou_mean}

    if evaluator_token is not None:
        PQ_t, SQ_t, RQ_t, _, _, _ = evaluator_token.getPQ()
        iou_mean_t, _ = evaluator_token.getSemIoU()
        results["PQ_token"] = PQ_t
        results["SQ_token"] = SQ_t
        results["RQ_token"] = RQ_t
        results["mIoU_token"] = iou_mean_t

    decoder.train()
    return results


class GracefulKiller:
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGTERM, self._handler)
        signal.signal(signal.SIGUSR1, self._handler)

    def _handler(self, signum, frame):
        print(f"\n[SIGNAL] Received {signal.Signals(signum).name}. Saving checkpoint and exiting...", flush=True)
        self.kill_now = True


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
    if cli_resume is not None:
        return cli_resume if os.path.isfile(cli_resume) else None
    if cfg["checkpoint"]["resume"] is not None:
        p = cfg["checkpoint"]["resume"]
        return p if os.path.isfile(p) else None
    if cfg["checkpoint"]["auto_resume"]:
        p = os.path.join(cfg["checkpoint"]["save_dir"], "latest.pth")
        return p if os.path.isfile(p) else None
    return None


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

    local_rank, rank, world_size, is_distributed, device = setup_distributed()
    is_main = (rank == 0)

    if is_main:
        print(f"Device: {device}, world_size: {world_size}", flush=True)

    seed = cfg["training"]["seed"] + rank
    set_seed(seed)

    killer = GracefulKiller()

    save_dir = cfg["checkpoint"]["save_dir"]
    if is_main:
        os.makedirs(save_dir, exist_ok=True)

    wandb_active = init_wandb(cfg) if is_main else False

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
        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}", flush=True)

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

    # ── Warm-start from best.pth ──
    warm_cfg = cfg.get("warm_start", {})
    warm_ckpt_path = warm_cfg.get("checkpoint")
    if warm_ckpt_path and os.path.isfile(warm_ckpt_path):
        ckpt = torch.load(warm_ckpt_path, map_location="cpu")
        decoder_state = ckpt["decoder"]
        # strict=False: allows missing token_head keys
        missing, unexpected = decoder.load_state_dict(decoder_state, strict=False)
        if is_main:
            print(f"[Warm-start] Loaded from {warm_ckpt_path}", flush=True)
            if missing:
                print(f"  Missing keys (expected for new token_head): {missing}", flush=True)
            if unexpected:
                print(f"  Unexpected keys: {unexpected}", flush=True)
    elif warm_ckpt_path and is_main:
        print(f"[WARN] Warm-start checkpoint not found: {warm_ckpt_path}", flush=True)

    if is_main:
        n_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        n_token = sum(p.numel() for p in decoder.token_head.parameters()) if decoder.token_head else 0
        print(f"[Decoder] {n_params:,} trainable params (token_head: {n_token:,})", flush=True)

    # ── Load text prototypes ──
    token_cfg = cfg.get("token", {})
    proto_path = token_cfg.get("prototype_path", "panoptic/clip_prototypes/text_prototypes_vitl14_336.pt")
    proto_data = torch.load(proto_path, map_location="cpu")
    text_prototypes = proto_data["embeddings"].float().to(device)  # (16, 768)
    if is_main:
        print(f"[Prototypes] Loaded {text_prototypes.shape} from {proto_path}", flush=True)

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

    token_criterion = CLIPTokenLoss(
        text_prototypes, num_classes=mcfg["num_classes"],
        loss_type=token_cfg.get("loss_type", "cosine"),
        init_temperature=token_cfg.get("temperature", 0.07),
        learnable_temperature=token_cfg.get("learnable_temperature", True),
    ).to(device)
    token_weight_max = token_cfg.get("weight", 0.5)
    token_warmup_steps = token_cfg.get("warmup_steps", 100)
    aux_token_loss = token_cfg.get("aux_token_loss", False)
    print(f"[Token] loss_type={token_cfg.get('loss_type', 'cosine')}, weight={token_weight_max}, "
          f"warmup={token_warmup_steps}, aux={aux_token_loss}", flush=True)
    if hasattr(token_criterion, "log_temperature"):
        print(f"[Token] learnable temperature, init={token_criterion.log_temperature.exp().item():.4f}", flush=True)

    # Collect learnable temperature parameter from token_criterion if present
    token_loss_params = list(token_criterion.parameters())  # learnable log_temperature

    # ── Optimizer (fresh — even for warm-start, we use new optimizer) ──
    # Support differential LR: lower LR for pretrained decoder, higher for fresh token head
    token_head_lr = tcfg.get("token_head_lr", None)  # None = use same LR for all
    if token_head_lr is not None and decoder.token_head is not None:
        token_head_params = set(id(p) for p in decoder.token_head.parameters())
        base_params = [p for p in decoder.parameters() if id(p) not in token_head_params]
        param_groups = [
            {"params": base_params, "lr": tcfg["lr"]},
            {"params": list(decoder.token_head.parameters()) + token_loss_params, "lr": token_head_lr},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=tcfg["weight_decay"])
        print(f"[Optimizer] Differential LR: decoder={tcfg['lr']:.1e}, token_head={token_head_lr:.1e}", flush=True)
    else:
        all_params = list(decoder.parameters()) + token_loss_params
        optimizer = torch.optim.AdamW(
            all_params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
        )

    # ── Scheduler ──
    steps_per_epoch = len(train_loader) // tcfg["grad_accum"]
    scheduler, sched_step_per = build_scheduler(optimizer, cfg, steps_per_epoch)

    # ── Mixed precision ──
    use_amp = tcfg["mixed_precision"] and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # ── Resume (from token training checkpoint, not warm-start) ──
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
            print(f"[Resume] From {resume_path}, epoch {start_epoch}, best_pq={best_pq:.4f}", flush=True)

    # ── DDP ──
    if is_distributed:
        decoder = DDP(decoder, device_ids=[local_rank], output_device=local_rank,
                      find_unused_parameters=False)

    # ── Training ──
    if is_main:
        print(f"\n{'='*60}", flush=True)
        print(f"Token Training: epochs {start_epoch}→{tcfg['epochs']-1}, "
              f"lr={tcfg['lr']}, token_weight={token_weight_max}, "
              f"warmup_steps={token_warmup_steps}", flush=True)
        print(f"{'='*60}\n", flush=True)

    log_every = cfg["logging"]["log_every"]
    eval_every = cfg["logging"]["eval_every"]
    save_every = cfg["checkpoint"]["save_every"]

    decoder.train()
    for epoch in range(start_epoch, tcfg["epochs"]):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_token_loss = 0.0
        epoch_loss_components = {}
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
                    panoptic_losses, indices = criterion(outputs, targets)

                    # Token loss (final layer)
                    tw = token_warmup_weight(global_step, token_warmup_steps) * token_weight_max
                    token_losses = token_criterion(outputs["pred_tokens"], targets, indices)
                    token_loss_val = token_losses["loss_token"] * tw

                    # Aux token loss (intermediate decoder layers)
                    if aux_token_loss and "aux_outputs" in outputs:
                        for i, aux in enumerate(outputs["aux_outputs"]):
                            if "pred_tokens" in aux:
                                aux_indices = criterion.matcher(aux, targets)
                                aux_tl = token_criterion(aux["pred_tokens"], targets, aux_indices)
                                token_loss_val = token_loss_val + aux_tl["loss_token"] * tw

                    total_loss = (sum(panoptic_losses.values()) + token_loss_val) / tcfg["grad_accum"]
                scaler.scale(total_loss).backward()
            else:
                outputs = decoder(feats, coords)
                targets = {
                    "classes": [batch["gt_classes"].to(device)],
                    "masks": [batch["gt_masks"].float().to(device)],
                }
                panoptic_losses, indices = criterion(outputs, targets)

                tw = token_warmup_weight(global_step, token_warmup_steps) * token_weight_max
                token_losses = token_criterion(outputs["pred_tokens"], targets, indices)
                token_loss_val = token_losses["loss_token"] * tw

                # Aux token loss (intermediate decoder layers)
                if aux_token_loss and "aux_outputs" in outputs:
                    for i, aux in enumerate(outputs["aux_outputs"]):
                        if "pred_tokens" in aux:
                            aux_indices = criterion.matcher(aux, targets)
                            aux_tl = token_criterion(aux["pred_tokens"], targets, aux_indices)
                            token_loss_val = token_loss_val + aux_tl["loss_token"] * tw

                total_loss = (sum(panoptic_losses.values()) + token_loss_val) / tcfg["grad_accum"]
                total_loss.backward()

            epoch_loss += total_loss.item() * tcfg["grad_accum"]
            epoch_token_loss += token_losses["loss_token"].item()
            # Track individual loss components
            for lk, lv in panoptic_losses.items():
                if lk not in epoch_loss_components:
                    epoch_loss_components[lk] = 0.0
                epoch_loss_components[lk] += lv.item()
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
                avg_token = epoch_token_loss / epoch_steps
                lr_now = optimizer.param_groups[0]["lr"]
                lr_token = optimizer.param_groups[-1]["lr"] if len(optimizer.param_groups) > 1 else lr_now
                print(f"  [E{epoch} S{step+1}/{len(train_loader)}] "
                      f"loss={avg:.4f} token={avg_token:.4f} tw={tw:.3f} lr={lr_now:.2e}", flush=True)
                log_dict = {
                    "train/loss": avg, "train/token_loss": avg_token,
                    "train/token_weight": tw, "train/lr": lr_now,
                    "train/lr_token": lr_token,
                }
                # Log individual loss components (final layer only, aux summed)
                aux_total = 0.0
                for lk, lv in epoch_loss_components.items():
                    avg_lv = lv / epoch_steps
                    if lk.startswith(("loss_ce", "loss_mask", "loss_dice")):
                        log_dict[f"train/{lk}"] = avg_lv
                    else:
                        aux_total += avg_lv
                log_dict["train/aux_loss"] = aux_total
                # Log learnable temperature if contrastive loss
                if hasattr(token_criterion, "log_temperature"):
                    log_dict["train/temperature"] = token_criterion.log_temperature.exp().item()
                log_wandb(log_dict, step=global_step)

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

        epoch_time = time.time() - t_epoch
        if is_main:
            avg_loss = epoch_loss / max(epoch_steps, 1)
            avg_token = epoch_token_loss / max(epoch_steps, 1)
            lr_now = optimizer.param_groups[0]["lr"]
            gpu_mem = 0
            if device.type == "cuda":
                gpu_mem = torch.cuda.max_memory_allocated() / 1024**2
                torch.cuda.reset_peak_memory_stats()
            print(f"Epoch {epoch}: loss={avg_loss:.4f}, token_loss={avg_token:.4f}, "
                  f"lr={lr_now:.2e}, time={epoch_time:.1f}s, gpu={gpu_mem:.0f}MB", flush=True)

        # ── Eval ──
        if (epoch + 1) % eval_every == 0 or epoch == tcfg["epochs"] - 1:
            if is_main:
                print("  Evaluating...", flush=True)
                decoder_raw = decoder.module if is_distributed else decoder
                metrics = evaluate(decoder_raw, backbone, val_loader, device, cfg, text_prototypes)
                print(f"  Linear: PQ={metrics['PQ']:.4f} SQ={metrics['SQ']:.4f} "
                      f"RQ={metrics['RQ']:.4f} mIoU={metrics['mIoU']:.4f}", flush=True)
                if "PQ_token" in metrics:
                    print(f"  Token:  PQ={metrics['PQ_token']:.4f} SQ={metrics['SQ_token']:.4f} "
                          f"RQ={metrics['RQ_token']:.4f} mIoU={metrics['mIoU_token']:.4f}", flush=True)
                log_wandb({
                    "val/PQ": metrics["PQ"], "val/SQ": metrics["SQ"],
                    "val/RQ": metrics["RQ"], "val/mIoU": metrics["mIoU"],
                    "val/PQ_token": metrics.get("PQ_token", 0),
                    "val/mIoU_token": metrics.get("mIoU_token", 0),
                }, step=global_step)

                if metrics["PQ"] > best_pq:
                    best_pq = metrics["PQ"]
                    save_checkpoint(
                        os.path.join(save_dir, "best.pth"),
                        epoch, decoder, optimizer, scheduler, scaler, best_pq, cfg, metrics,
                    )
                    print(f"  Saved best (PQ={best_pq:.4f})", flush=True)
                    log_wandb_artifact(
                        os.path.join(save_dir, "best.pth"),
                        name="best-checkpoint",
                        metadata={"epoch": epoch, "PQ": float(best_pq)},
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
        print(f"\nTraining complete. Best PQ: {best_pq:.4f}", flush=True)
        if wandb_active:
            wandb.finish()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
