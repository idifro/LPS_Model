"""Open-Vocabulary E2E Panoptic Training — stable v2.

Changes from v1:
  - Hardened bad-loss guard (threshold 150, Adam momentum decay on skip)
  - scaler.update() always called even on skipped steps (prevents AMP lockup)
  - Grad norm logged every log_every steps
  - Inf grad norm causes optimizer step skip without zeroing Adam state
  - Resume resets Adam exp_avg/exp_avg_sq to avoid loading corrupted momentum
  - Per-component loss clamping before sum (handles both Tensor and float values)
  - warmup_epochs respected on resume (scheduler already advanced)
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
from panoptic.models.open_vocab_head import OpenVocabPanopticHead
from panoptic.models.context_encoder import (
    ContextAwareBackboneWrapper, SceneContextEncoder,
)
from panoptic.losses.open_vocab_loss import OpenVocabPanopticLoss, NUSCENES_B12N4_NOVEL
from panoptic.eval.eval_pq import PanopticEval
from panoptic.dataset.nuscenes_panoptic import (
    NuScenesPanoptic, NUM_CLASSES, THING_IDS, STUFF_IDS,
    NUSCENES_CLASS_NAMES as NUSCENES_16_NAMES,
)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from panoptic.train.train_panoptic_e2e import (
    load_config, set_seed, setup_distributed,
    load_backbone_e2e, freeze_backbone, unfreeze_backbone,
    extract_backbone_features_e2e,
    CosineWarmupScheduler, GracefulKiller,
    find_resume_path, init_wandb, log_wandb,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def clamp_loss_dict(losses, max_val, device):
    """Clamp each loss component to max_val.
    Handles both torch.Tensor and plain float values."""
    clamped = {}
    for k, v in losses.items():
        if isinstance(v, torch.Tensor):
            clamped[k] = torch.clamp(v, max=max_val)
        else:
            clamped[k] = torch.tensor(
                min(float(v), max_val), device=device, dtype=torch.float32
            )
    return clamped


def sum_loss_dict(losses):
    """Sum all values in loss dict, handling mixed Tensor / float."""
    total = None
    for v in losses.values():
        if not isinstance(v, torch.Tensor):
            continue
        total = v if total is None else total + v
    if total is None:
        # fallback: should not happen in normal operation
        total = torch.tensor(0.0)
    return total


# ══════════════════════════════════════════════════════════════════════════════
# Open-Vocab Decoder
# ══════════════════════════════════════════════════════════════════════════════

class OpenVocabPanopticDecoder(PanopticDecoder):

    def __init__(self, open_vocab_cfg, **decoder_kwargs):
        super().__init__(**decoder_kwargs)
        self.open_vocab_head = OpenVocabPanopticHead(
            hidden_dim=decoder_kwargs.get("hidden_dim", 256),
            clip_dim=open_vocab_cfg.get("token_dim", 768),
            num_mlp_layers=open_vocab_cfg.get("token_mlp_layers", 3),
            objectness_mode=open_vocab_cfg.get("objectness_mode", "ternary"),
            pool_mode=open_vocab_cfg.get("region_pool_mode", "weighted_mean"),
        )
        self.use_open_vocab = True

    def forward(self, backbone_feats, coords, pad_mask=None):
        out = super().forward(backbone_feats, coords, pad_mask)
        pred_masks = out["pred_masks"]  # (B, V, Q)

        ov_out = self.open_vocab_head(
            query_features=self._last_query_features,
            backbone_feats=backbone_feats,
            pred_masks=pred_masks.detach(),
        )

        out["objectness"]  = ov_out["objectness"]
        out["query_clip"]  = ov_out["query_clip"]
        out["region_clip"] = ov_out["region_clip"]
        out["temperature"] = self.open_vocab_head.temperature

        if "aux_outputs" in out:
            for aux in out["aux_outputs"]:
                aux_ov = self.open_vocab_head(
                    query_features=self._last_query_features,
                    backbone_feats=None,
                    pred_masks=None,
                )
                aux["objectness"] = aux_ov["objectness"]
                aux["query_clip"]  = aux_ov["query_clip"]

        return out

    def pred_heads(self, output, mask_features, pad_mask=None):
        decoder_output = self.decoder_norm(output)
        self._last_query_features = decoder_output

        outputs_class = self.class_embed(decoder_output)
        mask_embed    = self.mask_embed(decoder_output)
        outputs_mask  = torch.einsum("bqc,bpc->bpq", mask_embed, mask_features)

        attn_mask = (outputs_mask.sigmoid() < 0.5).detach().bool()
        if pad_mask is not None:
            attn_mask[pad_mask.unsqueeze(-1).expand_as(attn_mask)] = True
        attn_mask = (
            attn_mask.unsqueeze(1)
            .repeat(1, self.nheads, 1, 1)
            .flatten(0, 1)
            .permute(0, 2, 1)
        )

        return outputs_class, outputs_mask, attn_mask


# ══════════════════════════════════════════════════════════════════════════════
# Text prototypes
# ══════════════════════════════════════════════════════════════════════════════

def load_or_generate_prototypes(cfg, device):
    proto_path = "panoptic/clip_prototypes/text_prototypes_diverse_16class.pt"
    if os.path.exists(proto_path):
        prototypes = torch.load(proto_path, map_location=device)
        print(f"[Prototypes] Loaded from {proto_path}: {prototypes.shape}")
        return prototypes

    std_path = "panoptic/clip_prototypes/text_prototypes_vitl14_336_16class.pt"
    if os.path.exists(std_path):
        prototypes = torch.load(std_path, map_location=device)
        print(f"[Prototypes] Loaded from {std_path}: {prototypes.shape}")
        return prototypes

    raise RuntimeError(
        "No prototypes found. Run first:\n"
        "  python panoptic/generate_prototypes.py"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Open-vocab inference
# ══════════════════════════════════════════════════════════════════════════════

def openvocab_panoptic_inference(objectness, query_clip, pred_masks,
                                 text_embeddings, overlap_thr=0.8,
                                 clip_weight=0.7, obj_weight=0.3,
                                 objectness_mode="ternary"):
    obj_probs = objectness.softmax(-1)
    if objectness_mode == "ternary":
        obj_score = obj_probs[:, 0] + obj_probs[:, 1]
        is_thing  = obj_probs[:, 0] > obj_probs[:, 1]
    else:
        obj_score = obj_probs[:, 0]
        is_thing  = torch.ones(obj_probs.shape[0], dtype=torch.bool)

    clip_sim             = query_clip @ text_embeddings.T
    clip_scores, clip_labels = clip_sim.max(dim=-1)
    combined_score       = obj_weight * obj_score + clip_weight * clip_scores

    V     = pred_masks.shape[0]
    masks = pred_masks.sigmoid()

    keep = obj_score > 0.3
    if keep.sum() == 0:
        return np.zeros(V, dtype=np.int64), np.zeros(V, dtype=np.int64)

    scores     = combined_score[keep]
    labels     = clip_labels[keep]
    thing_mask = is_thing[keep]
    masks_kept = masks[:, keep]
    sorted_idx = scores.argsort(descending=True)

    sem_pred       = np.zeros(V, dtype=np.int64)
    inst_pred      = np.zeros(V, dtype=np.int64)
    inst_id        = 1
    stuff_assigned = {}

    for idx in sorted_idx:
        label          = labels[idx].item()
        is_thing_query = thing_mask[idx].item()
        mask_prob      = masks_kept[:, idx]

        binary = mask_prob >= 0.5
        if binary.sum() == 0:
            continue
        orig_area = binary.sum().item()

        assigned   = torch.from_numpy(inst_pred).to(mask_prob.device) > 0
        final_mask = binary & ~assigned

        if final_mask.sum().item() / max(orig_area, 1) < overlap_thr:
            continue

        mask_np = final_mask.cpu().numpy().astype(bool)

        if is_thing_query:
            sem_pred[mask_np]  = label
            inst_pred[mask_np] = inst_id
            inst_id += 1
        else:
            if label in stuff_assigned:
                sem_pred[mask_np]  = label
                inst_pred[mask_np] = stuff_assigned[label]
            else:
                sem_pred[mask_np]       = label
                inst_pred[mask_np]      = inst_id
                stuff_assigned[label]   = inst_id
                inst_id += 1

    return sem_pred, inst_pred


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(decoder, backbone, dataloader, device, cfg, text_prototypes):
    num_classes = cfg["model"]["num_classes"]
    ov_cfg      = cfg.get("open_vocab", {})
    inf_cfg     = cfg["inference"]

    decoder.eval()
    backbone.eval()
    evaluator = PanopticEval(
        n_classes=num_classes, min_points=inf_cfg["min_points"]
    )

    with torch.no_grad():
        for batch in dataloader:
            feats, coords = extract_backbone_features_e2e(
                backbone, batch, device, is_frozen=True
            )
            outputs = decoder(feats, coords)

            objectness = outputs["objectness"][0].cpu()
            query_clip = outputs["query_clip"][0].cpu()
            pred_masks = outputs["pred_masks"][0].cpu()

            sem_pred, inst_pred = openvocab_panoptic_inference(
                objectness, query_clip, pred_masks,
                text_prototypes.cpu(),
                overlap_thr=inf_cfg["overlap_thr"],
                clip_weight=inf_cfg.get("clip_score_weight", 0.7),
                obj_weight=inf_cfg.get("objectness_weight", 0.3),
                objectness_mode=ov_cfg.get("objectness_mode", "ternary"),
            )

            gt_sem     = batch["gt_sem"].numpy()
            gt_masks   = batch["gt_masks"].numpy()
            gt_classes = batch["gt_classes"].numpy()
            V          = gt_sem.shape[0]
            gt_inst    = np.zeros(V, dtype=np.int64)
            for m_idx in range(len(gt_classes)):
                gt_inst[gt_masks[m_idx]] = m_idx + 1

            valid = gt_sem != 255
            if valid.sum() == 0:
                continue
            evaluator.addBatch(
                sem_pred[valid], inst_pred[valid],
                gt_sem[valid],   gt_inst[valid],
            )

    PQ, SQ, RQ, _, _, _ = evaluator.getPQ()
    iou_mean, _          = evaluator.getSemIoU()
    decoder.train()
    return {"PQ": PQ, "SQ": SQ, "RQ": RQ, "mIoU": iou_mean}


# ══════════════════════════════════════════════════════════════════════════════
# Checkpointing
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(path, epoch, decoder, backbone, optimizer, scheduler,
                    scaler, best_pq, cfg, metrics=None):
    state = {
        "epoch":     epoch,
        "decoder":   (decoder.module.state_dict()
                      if hasattr(decoder, "module") else decoder.state_dict()),
        "backbone":  (backbone.module.state_dict()
                      if hasattr(backbone, "module") else backbone.state_dict()),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_pq":   best_pq,
        "config":    cfg,
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if metrics is not None:
        state["metrics"] = metrics
    torch.save(state, path)
    print(f"  [Checkpoint] Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Adam state helpers
# ══════════════════════════════════════════════════════════════════════════════

def reset_adam_state(optimizer):
    """Zero Adam exp_avg and exp_avg_sq. Used after loading a checkpoint
    that may have corrupted momentum."""
    count = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p, {})
            if "exp_avg" in state:
                state["exp_avg"].zero_()
                state["exp_avg_sq"].zero_()
                count += 1
    return count


def decay_adam_state(optimizer, factor=0.9):
    """Decay (not zero) Adam momentum on a bad batch.
    Gentler than full reset — keeps direction, reduces magnitude."""
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p, {})
            if "exp_avg" in state:
                state["exp_avg"].mul_(factor)
                state["exp_avg_sq"].mul_(factor)


# ══════════════════════════════════════════════════════════════════════════════
# Args
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Open-vocab E2E panoptic training v2"
    )
    p.add_argument(
        "--config",
        default="panoptic/configs/mini_openvocab_e2e_decoder_only.yaml",
    )
    p.add_argument("--resume", default=None)
    p.add_argument(
        "--reset-adam", action="store_true",
        help="Reset Adam momentum on resume (legacy flag — now always reset)",
    )
    args, unknown = p.parse_known_args()
    return args, unknown


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
        print(f"  OPEN-VOCABULARY E2E PANOPTIC TRAINING  (stable v2)")
        print(f"  Class-agnostic masks + CLIP region features")
        print(f"{'='*70}")

    seed = cfg["training"]["seed"] + rank
    set_seed(seed)
    killer = GracefulKiller()

    save_dir = cfg["checkpoint"]["save_dir"]
    if is_main:
        os.makedirs(save_dir, exist_ok=True)

    wandb_active = init_wandb(cfg) if is_main else False

    # ── Training hyper-params ──
    tcfg               = cfg["training"]
    bad_loss_threshold = tcfg.get("bad_loss_threshold", 150.0)
    reset_on_skip      = tcfg.get("reset_momentum_on_skip", True)
    loss_component_max = cfg["loss"].get("loss_component_max", 50.0)

    # ── Data ──
    dcfg = cfg["data"]
    acfg = cfg["augmentation"]
    from nuscenes.nuscenes import NuScenes as NuScenesDB
    nusc = NuScenesDB(
        version=dcfg["version"], dataroot=dcfg["dataroot"], verbose=False
    )
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

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True
    ) if is_distributed else None

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=1, shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=tcfg["num_workers"],
        collate_fn=lambda x: x[0],
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=tcfg["num_workers"],
        collate_fn=lambda x: x[0],
    )
    if is_main:
        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # ── Text prototypes ──
    text_prototypes = load_or_generate_prototypes(cfg, device)

    # ── Backbone ──
    e2e_cfg = cfg.get("e2e", {"enabled": False})
    ctx_cfg = cfg.get("context_encoder", {})

    raw_backbone, _, _ = load_backbone_e2e(
        cfg["model"]["backbone_ckpt"], device, e2e_cfg
    )

    backbone = ContextAwareBackboneWrapper(
        raw_backbone,
        context_dim=ctx_cfg.get("dim", 256),
        context_heads=ctx_cfg.get("num_heads", 4),
        context_depth=ctx_cfg.get("depth", 1),
        use_pos_enc=ctx_cfg.get("use_pos_enc", True),
    ).to(device)

    backbone.freeze_backbone_encoder()
    if is_main:
        n_trainable = sum(
            1 for p in backbone.parameters() if p.requires_grad
        )
        n_frozen = sum(
            1 for p in backbone.parameters() if not p.requires_grad
        )
        n_ctx = sum(
            p.numel() for p in backbone.context_encoder.parameters()
        )
        n_bb_trainable = sum(
            p.numel() for p in backbone.parameters() if p.requires_grad
        )
        print(f"[Backbone] DECODER_ONLY unfrozen: "
              f"{n_trainable} trainable, {n_frozen} frozen")
        print(f"[Backbone] Context encoder: {n_ctx:,} params")
        print(f"[Backbone] Trainable total: {n_bb_trainable:,} params")

    unfreeze_after          = e2e_cfg.get("unfreeze_after_epoch", 0)
    backbone_decoder_frozen = (unfreeze_after > 0)
    if backbone_decoder_frozen and e2e_cfg.get("enabled", False):
        for name, param in backbone.backbone.named_parameters():
            decoder_names = (
                'convtr', 'bntr', 'block5', 'block6',
                'block7', 'block8', 'final',
            )
            if any(name.startswith(d) for d in decoder_names):
                param.requires_grad = False
        if is_main:
            print(
                f"[E2E] Backbone decoder frozen for first "
                f"{unfreeze_after} epochs"
            )

    # ── Decoder ──
    mcfg   = cfg["model"]
    ov_cfg = cfg["open_vocab"]

    decoder = OpenVocabPanopticDecoder(
        open_vocab_cfg=ov_cfg,
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

    # ── Warm-start ──
    warmstart_path = e2e_cfg.get("warmstart_ckpt")
    if (warmstart_path and warmstart_path != "null"
            and os.path.isfile(warmstart_path)):
        ws_ckpt = torch.load(warmstart_path, map_location="cpu")
        missing, unexpected = decoder.load_state_dict(
            ws_ckpt["decoder"], strict=False
        )
        if is_main:
            print(f"[Warmstart] Loaded from {warmstart_path}")
            print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    if is_main:
        n_dec = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        n_bb  = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        print(f"[Params] Decoder: {n_dec:,}, Backbone: {n_bb:,}")

    # ── Loss ──
    lcfg = cfg["loss"]
    criterion = OpenVocabPanopticLoss(
        num_classes=mcfg["num_classes"],
        eos_coef=lcfg["eos_coef"],
        cost_class=lcfg["cost_class"],
        cost_mask=lcfg["cost_mask"],
        cost_dice=lcfg["cost_dice"],
        weight_cls=lcfg.get("weight_cls", 1.0),
        weight_mask=lcfg.get("weight_mask", 1.0),
        weight_dice=lcfg.get("weight_dice", 1.0),
        weight_object_distill=lcfg.get("weight_object_distill", 1.0),
        weight_voxel_distill=lcfg.get("weight_voxel_distill", 0.1),
        novel_class_idx=cfg.get("open_vocab", {}).get(
            "novel_class_idx", NUSCENES_B12N4_NOVEL
        ),
        focal_gamma=lcfg["focal_gamma"],
        focal_alpha=lcfg["focal_alpha"],
        num_points=lcfg["num_points"],
        num_mask_pts=lcfg["num_mask_pts"],
        p_ratio=lcfg["p_ratio"],
    ).to(device)

    # ── Optimizer ──
    backbone_lr = e2e_cfg.get("backbone_lr", 5e-6)
    ctx_lr      = ctx_cfg.get("lr", tcfg["lr"] * 0.5)

    ctx_params    = backbone.get_context_encoder_params()
    bb_dec_params = backbone.get_backbone_decoder_params()
    dec_params    = [p for p in decoder.parameters() if p.requires_grad]

    param_groups = [
        {"params": dec_params,    "lr": tcfg["lr"],  "name": "decoder"},
        {"params": ctx_params,    "lr": ctx_lr,       "name": "ctx_encoder"},
        {"params": bb_dec_params, "lr": backbone_lr,  "name": "bb_decoder"},
    ]
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    if is_main:
        for g in param_groups:
            n = sum(p.numel() for p in g["params"])
            print(f"[Optim] {g['name']}: {n:,} params @ lr={g['lr']}")

    optimizer = torch.optim.AdamW(
        param_groups, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )

    # ── Scheduler ──
    steps_per_epoch = len(train_loader) // tcfg["grad_accum"]
    warmup_steps    = tcfg["warmup_epochs"] * steps_per_epoch
    total_steps     = tcfg["epochs"] * steps_per_epoch
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_steps, total_steps, min_lr=tcfg["min_lr"]
    )

    # ── AMP ──
    use_amp = tcfg["mixed_precision"] and device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    # ── Resume ──
    start_epoch = 0
    best_pq     = 0.0
    global_step = 0
    resume_path = find_resume_path(cfg, args.resume)

    if resume_path:
        if is_main:
            print(f"[Resume] Loading from {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        decoder.load_state_dict(ckpt["decoder"])
        if "backbone" in ckpt:
            backbone.load_state_dict(ckpt["backbone"])

        optimizer.load_state_dict(ckpt["optimizer"])

        # Always reset Adam momentum on resume to clear any corrupted state
        n_reset = reset_adam_state(optimizer)
        if is_main:
            print(f"[Resume] Adam momentum reset for {n_reset} param tensors")

        scheduler.load_state_dict(ckpt["scheduler"])
        if scaler and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])

        start_epoch = ckpt["epoch"] + 1
        best_pq     = ckpt.get("best_pq", 0.0)
        global_step = start_epoch * steps_per_epoch
        if is_main:
            print(
                f"[Resume] epoch {start_epoch}, best_pq={best_pq:.4f}, "
                f"Adam state reset"
            )

    # ── Freeze tracking ──
    unfreeze_encoder_flag = e2e_cfg.get("unfreeze_encoder", False)
    encoder_lr            = e2e_cfg.get("encoder_lr", 1e-7)
    encoder_is_frozen     = True
    backbone_is_frozen    = backbone_decoder_frozen

    if is_main:
        print(
            f"[E2E] unfreeze_encoder={unfreeze_encoder_flag}, "
            f"backbone decoder unfreeze at epoch {unfreeze_after}"
        )

    # ── DDP ──
    if is_distributed:
        decoder = DDP(
            decoder, device_ids=[local_rank], find_unused_parameters=True
        )
        if any(p.requires_grad for p in backbone.parameters()):
            backbone = DDP(
                backbone, device_ids=[local_rank], find_unused_parameters=True
            )

    # ── Training loop ──
    if is_main:
        print(f"\n{'='*70}")
        print(
            f"  Open-Vocab E2E: epochs {start_epoch}→{tcfg['epochs']-1}"
        )
        print(
            f"  Decoder LR: {tcfg['lr']:.1e}, "
            f"Backbone LR: {backbone_lr:.1e}"
        )
        print(
            f"  grad_clip={tcfg['grad_clip']}, "
            f"bad_loss_thr={bad_loss_threshold}, "
            f"loss_clamp={loss_component_max}"
        )
        print(f"{'='*70}\n")

    log_every  = cfg["logging"]["log_every"]
    eval_every = cfg["logging"]["eval_every"]
    save_every = cfg["checkpoint"]["save_every"]

    MAX_CONSECUTIVE_SKIPS = 50

    decoder.train()

    for epoch in range(start_epoch, tcfg["epochs"]):
        backbone_raw = backbone.module if hasattr(backbone, "module") else backbone

        # ── Unfreeze backbone DECODER ──
        if (backbone_is_frozen and epoch >= unfreeze_after
                and e2e_cfg["enabled"]):
            backbone_raw.unfreeze_backbone_decoder()
            backbone_is_frozen = False
            new_dec_params = [
                p for p in backbone_raw.get_backbone_decoder_params()
                if p.requires_grad
            ]
            if new_dec_params:
                optimizer.add_param_group({
                    "params": new_dec_params,
                    "lr":     backbone_lr,
                    "name":   "bb_decoder_unfrozen",
                })
            if is_main:
                print(
                    f"\n[E2E] *** BACKBONE DECODER UNFROZEN "
                    f"at epoch {epoch} ***\n"
                )

        # ── Unfreeze backbone ENCODER (optional) ──
        if (unfreeze_encoder_flag and encoder_is_frozen
                and epoch >= (unfreeze_after + 2) and e2e_cfg["enabled"]):
            enc_params, _ = backbone_raw.unfreeze_encoder_stages(
                stages=(
                    'conv3', 'bn3', 'block3',
                    'conv4', 'bn4', 'block4',
                ),
                lr_override=encoder_lr,
            )
            encoder_is_frozen = False
            if enc_params:
                optimizer.add_param_group({
                    "params": enc_params,
                    "lr":     encoder_lr,
                    "name":   "encoder_unfrozen",
                })
            if is_main:
                print(
                    f"\n[E2E] *** ENCODER UNFROZEN at epoch {epoch} "
                    f"@ lr={encoder_lr:.1e} ***\n"
                )

        if train_sampler:
            train_sampler.set_epoch(epoch)

        epoch_loss        = 0.0
        epoch_steps       = 0
        epoch_skips       = 0
        consecutive_skips = 0
        optimizer.zero_grad()
        t_epoch = time.time()

        for step, batch in enumerate(train_loader):

            # ── Graceful kill ──
            if killer.kill_now:
                if is_main:
                    save_checkpoint(
                        os.path.join(save_dir, "interrupted.pth"),
                        epoch, decoder, backbone, optimizer,
                        scheduler, scaler, best_pq, cfg,
                    )
                sys.exit(0)

            # ── Forward + loss ──
            feats, coords = extract_backbone_features_e2e(
                backbone, batch, device, is_frozen=backbone_is_frozen
            )

            skip_step  = False
            total_loss = None
            losses     = {}

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = decoder(feats, coords)
                    targets = {
                        "classes": [batch["gt_classes"].to(device)],
                        "masks":   [batch["gt_masks"].float().to(device)],
                    }
                    losses, _ = criterion(
                        outputs, targets, text_prototypes, feats
                    )
                    # Clamp each component — handles Tensor and float
                    losses     = clamp_loss_dict(losses, loss_component_max, device)
                    total_loss = sum_loss_dict(losses) / tcfg["grad_accum"]

                loss_val = total_loss.item() * tcfg["grad_accum"]
                if not math.isfinite(loss_val) or loss_val > bad_loss_threshold:
                    skip_step = True

            else:
                outputs = decoder(feats, coords)
                targets = {
                    "classes": [batch["gt_classes"].to(device)],
                    "masks":   [batch["gt_masks"].float().to(device)],
                }
                losses, _ = criterion(
                    outputs, targets, text_prototypes, feats
                )
                # Clamp each component — handles Tensor and float
                losses     = clamp_loss_dict(losses, loss_component_max, device)
                total_loss = sum_loss_dict(losses) / tcfg["grad_accum"]

                loss_val = total_loss.item() * tcfg["grad_accum"]
                if not math.isfinite(loss_val) or loss_val > bad_loss_threshold:
                    skip_step = True

            # ── Handle bad loss ──
            if skip_step:
                if is_main:
                    print(
                        f"  [WARN] Bad loss={loss_val:.4f} at step {step}, "
                        f"skipping"
                    )
                optimizer.zero_grad()
                if reset_on_skip:
                    decay_adam_state(optimizer, factor=0.9)
                # Must call scaler.update() even on skip to prevent AMP lockup
                if use_amp:
                    scaler.update()
                epoch_skips       += 1
                consecutive_skips += 1
                if consecutive_skips >= MAX_CONSECUTIVE_SKIPS:
                    if is_main:
                        print(
                            f"  [ERROR] {MAX_CONSECUTIVE_SKIPS} consecutive "
                            f"skips at epoch {epoch} step {step} — "
                            f"saving and aborting epoch"
                        )
                        save_checkpoint(
                            os.path.join(save_dir, "interrupted.pth"),
                            epoch, decoder, backbone, optimizer,
                            scheduler, scaler, best_pq, cfg,
                        )
                    break
                continue

            consecutive_skips = 0

            # ── Backward ──
            if use_amp:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            epoch_loss  += loss_val
            epoch_steps += 1
            global_step += 1

            # ── Optimizer step every grad_accum steps ──
            if (step + 1) % tcfg["grad_accum"] == 0:
                if use_amp:
                    scaler.unscale_(optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(decoder.parameters()), tcfg["grad_clip"]
                )

                # Skip optimizer step if gradient is non-finite
                if not math.isfinite(grad_norm.item()):
                    if is_main:
                        print(
                            f"  [WARN] Non-finite grad norm={grad_norm:.3f} "
                            f"at step {step}, skipping optimizer step"
                        )
                    optimizer.zero_grad()
                    if use_amp:
                        scaler.update()
                    continue

                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()

            # ── Logging ──
            if is_main and (step + 1) % log_every == 0:
                avg    = epoch_loss / max(epoch_steps, 1)
                lr_now = optimizer.param_groups[0]["lr"]

                # Only log Tensor loss components by name
                loss_parts = " ".join(
                    f"{k}={v.item():.3f}"
                    if isinstance(v, torch.Tensor)
                    else f"{k}={v:.3f}"
                    for k, v in losses.items()
                    if not k[0].isdigit()
                )

                # Grad norm for monitoring (no extra backward)
                with torch.no_grad():
                    mon_norm = sum(
                        p.grad.norm().item() ** 2
                        for p in decoder.parameters()
                        if p.grad is not None
                    ) ** 0.5

                print(
                    f"  [E{epoch} S{step+1}/{len(train_loader)}] "
                    f"loss={avg:.4f} lr={lr_now:.2e} "
                    f"gnorm={mon_norm:.3f} skips={epoch_skips}"
                )
                print(f"    {loss_parts}")
                log_wandb({
                    "train/loss":      avg,
                    "train/lr":        lr_now,
                    "train/grad_norm": mon_norm,
                    "train/skips":     epoch_skips,
                }, step=global_step)

        # ── End of epoch ──
        epoch_time = time.time() - t_epoch
        if is_main:
            avg_loss = epoch_loss / max(epoch_steps, 1)
            gpu_mem  = (
                torch.cuda.max_memory_allocated() / 1024**2
                if device.type == "cuda" else 0
            )
            torch.cuda.reset_peak_memory_stats()
            print(
                f"Epoch {epoch}: loss={avg_loss:.4f}, "
                f"time={epoch_time:.1f}s, gpu={gpu_mem:.0f}MB, "
                f"skips={epoch_skips}/{len(train_loader)}"
            )
            log_wandb({
                "epoch/loss":   avg_loss,
                "epoch/time_s": epoch_time,
                "epoch/skips":  epoch_skips,
            }, step=global_step)

        # ── Evaluation ──
        if (epoch + 1) % eval_every == 0 or epoch == tcfg["epochs"] - 1:
            if is_main:
                decoder_raw  = decoder.module if is_distributed else decoder
                backbone_raw = (
                    backbone.module
                    if hasattr(backbone, "module") else backbone
                )
                metrics = evaluate(
                    decoder_raw, backbone_raw, val_loader,
                    device, cfg, text_prototypes,
                )
                print(
                    f"  PQ={metrics['PQ']:.4f} SQ={metrics['SQ']:.4f} "
                    f"RQ={metrics['RQ']:.4f} mIoU={metrics['mIoU']:.4f}"
                )
                log_wandb({
                    "val/PQ":   metrics["PQ"],
                    "val/SQ":   metrics["SQ"],
                    "val/RQ":   metrics["RQ"],
                    "val/mIoU": metrics["mIoU"],
                }, step=global_step)

                if metrics["PQ"] > best_pq:
                    best_pq = metrics["PQ"]
                    save_checkpoint(
                        os.path.join(save_dir, "best.pth"),
                        epoch, decoder, backbone, optimizer,
                        scheduler, scaler, best_pq, cfg, metrics,
                    )
                    print(f"  ★ New best PQ={best_pq:.4f}")

            if is_distributed:
                dist.barrier()
            decoder.train()

        # ── Periodic checkpoint ──
        if (epoch + 1) % save_every == 0 or epoch == tcfg["epochs"] - 1:
            if is_main:
                save_checkpoint(
                    os.path.join(save_dir, "latest.pth"),
                    epoch, decoder, backbone, optimizer,
                    scheduler, scaler, best_pq, cfg,
                )
            if is_distributed:
                dist.barrier()

    if is_main:
        print(f"\n{'='*70}")
        print(f"  Open-Vocab E2E complete. Best PQ: {best_pq:.4f}")
        print(f"{'='*70}")
        if wandb_active:
            wandb.finish()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()