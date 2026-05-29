"""Open-Vocabulary E2E Panoptic Training.

Architecture change from closed-set:
  BEFORE: query → 16-class CE → argmax → fixed label
  AFTER:  query → objectness (thing/stuff/nothing) + CLIP feature
          inference: CLIP_feature @ text("any query").T → score

This enables arbitrary text queries at inference:
  - "car" → detects cars
  - "red SUV near a building" → detects that specific thing
  - "person crossing the road" → detects that

Key components:
  1. Objectness head (class-agnostic detection)
  2. Region-CLIP head (query → CLIP space)
  3. Region pooler (mask-weighted backbone features → CLIP descriptor)
  4. Diverse text prototypes (train against varied prompts, not 1 per class)
  5. Unfrozen backbone (E2E) for feature adaptation

Usage:
    python -u panoptic/train/train_openvocab_e2e.py --config panoptic/configs/mini_openvocab_e2e.yaml
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
)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Reuse utilities from e2e script
from panoptic.train.train_panoptic_e2e import (
    load_config, set_seed, setup_distributed, 
    load_backbone_e2e, freeze_backbone, unfreeze_backbone,
    extract_backbone_features_e2e,
    CosineWarmupScheduler, GracefulKiller,
    find_resume_path, init_wandb, log_wandb,
)


# ══════════════════════════════════════════════════════════════════════════════
# Open-Vocab Decoder (extends PanopticDecoder with open-vocab head)
# ══════════════════════════════════════════════════════════════════════════════

class OpenVocabPanopticDecoder(PanopticDecoder):
    """PanopticDecoder + OpenVocabPanopticHead replacing the class_embed head.
    
    Key difference: instead of self.class_embed (Linear → 16+1 classes),
    we use objectness (thing/stuff/no-obj) + region-CLIP alignment.
    """
    
    def __init__(self, open_vocab_cfg, **decoder_kwargs):
        # Initialize parent (creates class_embed which we'll bypass)
        super().__init__(**decoder_kwargs)
        
        # Add open-vocab head
        self.open_vocab_head = OpenVocabPanopticHead(
            hidden_dim=decoder_kwargs.get("hidden_dim", 256),
            clip_dim=open_vocab_cfg.get("token_dim", 768),
            num_mlp_layers=open_vocab_cfg.get("token_mlp_layers", 3),
            objectness_mode=open_vocab_cfg.get("objectness_mode", "ternary"),
            pool_mode=open_vocab_cfg.get("region_pool_mode", "weighted_mean"),
        )
        
        self.use_open_vocab = True
    
    def forward(self, backbone_feats, coords, pad_mask=None):
        """Forward pass with open-vocab outputs.
        
        Returns standard mask predictions + open-vocab head outputs.
        """
        # Run standard decoder forward (gets pred_logits, pred_masks, aux)
        out = super().forward(backbone_feats, coords, pad_mask)
        
        # Now compute open-vocab outputs from the last decoder layer's queries
        # We need access to the raw query features before class_embed
        # Re-run the decoder's final norm on the queries
        # The parent's forward already computed predictions — we need the query states
        
        # Get the final query features by running through the decoder again?
        # Better: modify parent to also return raw queries. For now, use a workaround:
        # Recompute from the mask_embed features (they share the same source)
        
        # Actually, let's just hook into the parent's computation.
        # The pred_logits come from self.class_embed(decoder_norm(output))
        # We can get the pre-class_embed features from pred_logits inversely,
        # but that's hacky. Instead, we'll override pred_heads in forward.
        
        # Simpler approach: the open_vocab_head needs query features.
        # We stored them during parent forward via pred_logits computation.
        # Let's just compute objectness and clip features from the logits dimension.
        
        # CLEAN APPROACH: We override and compute from the mask predictions
        pred_masks = out["pred_masks"]  # (B, V, Q)
        
        # Compute open-vocab outputs using a second forward through our head
        # We need the decoder queries — extract them by reversing class_embed
        # OR we simply re-extract from backbone features
        
        # Best: store decoder output in parent. Let's use the pred_logits shape
        # to identify the decoder hidden state. Since class_embed is Linear(hidden→17),
        # we can get hidden = class_embed.weight.pinv() @ logits... too hacky.
        
        # PRAGMATIC: Run open_vocab_head on the same normalized output that class_embed uses.
        # We need to store it. Let's add it during the mask_embed computation.
        # The mask_embed MLP takes the same input: decoder_norm(output).
        
        # For now: use the backbone_feats + region pooling for open-vocab classification
        ov_out = self.open_vocab_head(
            query_features=self._last_query_features,  # set during forward
            backbone_feats=backbone_feats,
            pred_masks=pred_masks.detach(),  # detach to avoid double backward through masks
        )
        
        out["objectness"] = ov_out["objectness"]
        out["query_clip"] = ov_out["query_clip"]
        out["region_clip"] = ov_out["region_clip"]
        out["temperature"] = self.open_vocab_head.temperature
        
        # Also add to aux_outputs
        if "aux_outputs" in out:
            for aux in out["aux_outputs"]:
                # Aux outputs only get objectness and query_clip (no region pooling)
                aux_ov = self.open_vocab_head(
                    query_features=self._last_query_features,
                    backbone_feats=None,
                    pred_masks=None,
                )
                aux["objectness"] = aux_ov["objectness"]
                aux["query_clip"] = aux_ov["query_clip"]
        
        return out
    
    def pred_heads(self, output, mask_features, pad_mask=None):
        """Override to store query features for open-vocab head."""
        decoder_output = self.decoder_norm(output)
        
        # Store for open-vocab head access
        self._last_query_features = decoder_output
        
        # Standard predictions (still compute class_embed for compatibility)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bpc->bpq", mask_embed, mask_features)
        
        # Attention mask
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
# Text prototypes with diversity
# ══════════════════════════════════════════════════════════════════════════════

def load_or_generate_prototypes(cfg, device):
    """Load precomputed prototypes or generate diverse ones with CLIP."""
    from panoptic.dataset.nuscenes_panoptic import NUSCENES_16_NAMES
    
    proto_path = "panoptic/clip_prototypes/text_prototypes_diverse_16class.pt"
    
    if os.path.exists(proto_path):
        prototypes = torch.load(proto_path, map_location=device)
        print(f"[Prototypes] Loaded diverse prototypes from {proto_path}: {prototypes.shape}")
        return prototypes
    
    # Generate using CLIP
    print("[Prototypes] Generating diverse prototypes with CLIP...")
    try:
        import clip
        model, _ = clip.load("ViT-L/14@336px", device=device)
        model.eval()
        
        from panoptic.models.diverse_prompts import DiversePromptGenerator
        generator = DiversePromptGenerator(
            NUSCENES_16_NAMES, num_templates=7,
            compositional=True, fine_grained=True,
        )
        
        all_prototypes = []
        with torch.no_grad():
            for idx in range(len(NUSCENES_16_NAMES)):
                prompts = generator.prompt_pools[idx]
                tokens = clip.tokenize(prompts).to(device)
                # Process in batches (some classes have 30+ prompts)
                features_list = []
                for i in range(0, len(tokens), 32):
                    batch_tokens = tokens[i:i+32]
                    feats = model.encode_text(batch_tokens).float()
                    feats = F.normalize(feats, dim=-1)
                    features_list.append(feats)
                features = torch.cat(features_list, dim=0)
                avg = F.normalize(features.mean(dim=0, keepdim=True), dim=-1)
                all_prototypes.append(avg.squeeze(0))
        
        prototypes = torch.stack(all_prototypes)
        os.makedirs(os.path.dirname(proto_path), exist_ok=True)
        torch.save(prototypes.cpu(), proto_path)
        print(f"[Prototypes] Saved diverse prototypes: {prototypes.shape}")
        del model
        torch.cuda.empty_cache()
        return prototypes.to(device)
        
    except Exception as e:
        print(f"[Prototypes] CLIP generation failed: {e}")
        print("[Prototypes] Falling back to standard prototypes...")
        std_path = "panoptic/clip_prototypes/text_prototypes_vitl14_336_16class.pt"
        if os.path.exists(std_path):
            return torch.load(std_path, map_location=device)
        raise RuntimeError("No prototypes available. Run generate_text_prototypes.py first.")


# ══════════════════════════════════════════════════════════════════════════════
# Open-vocab inference
# ══════════════════════════════════════════════════════════════════════════════

def openvocab_panoptic_inference(objectness, query_clip, pred_masks, text_embeddings,
                                 overlap_thr=0.8, clip_weight=0.7, obj_weight=0.3,
                                 objectness_mode="ternary"):
    """Open-vocab panoptic inference using CLIP similarity for classification.
    
    Instead of argmax over 16 class logits:
      score = obj_weight * objectness_prob + clip_weight * max_clip_sim
      class = argmax(query_clip @ text_embeddings.T)
    
    This allows ANY text embeddings at inference (not just 16 classes).
    """
    # Objectness scores
    obj_probs = objectness.softmax(-1)  # (Q, 2or3)
    if objectness_mode == "ternary":
        # P(object) = P(thing) + P(stuff)
        obj_score = obj_probs[:, 0] + obj_probs[:, 1]  # (Q,)
        is_thing = obj_probs[:, 0] > obj_probs[:, 1]    # thing vs stuff
    else:
        obj_score = obj_probs[:, 0]
        is_thing = torch.ones(obj_probs.shape[0], dtype=torch.bool)
    
    # CLIP classification: query_clip @ text_embeddings.T
    clip_sim = query_clip @ text_embeddings.T  # (Q, num_text_classes)
    clip_scores, clip_labels = clip_sim.max(dim=-1)  # (Q,), (Q,)
    
    # Combined score
    combined_score = obj_weight * obj_score + clip_weight * clip_scores
    
    # Filter: keep queries with high combined score
    V = pred_masks.shape[0]
    masks = pred_masks.sigmoid()  # (V, Q)
    
    keep = obj_score > 0.3  # minimum objectness threshold
    if keep.sum() == 0:
        return np.zeros(V, dtype=np.int64), np.zeros(V, dtype=np.int64)
    
    scores = combined_score[keep]
    labels = clip_labels[keep]
    thing_mask = is_thing[keep]
    masks_kept = masks[:, keep]
    
    # Greedy assignment (same as closed-set)
    sorted_idx = scores.argsort(descending=True)
    
    sem_pred = np.zeros(V, dtype=np.int64)
    inst_pred = np.zeros(V, dtype=np.int64)
    inst_id = 1
    stuff_assigned = {}
    
    for idx in sorted_idx:
        label = labels[idx].item()
        is_thing_query = thing_mask[idx].item()
        mask_prob = masks_kept[:, idx]
        
        binary = mask_prob >= 0.5
        if binary.sum() == 0:
            continue
        orig_area = binary.sum().item()
        
        assigned = torch.from_numpy(inst_pred).to(mask_prob.device) > 0
        final_mask = binary & ~assigned
        
        if final_mask.sum().item() / max(orig_area, 1) < overlap_thr:
            continue
        
        mask_np = final_mask.cpu().numpy().astype(bool)
        
        if is_thing_query:
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

def evaluate(decoder, backbone, dataloader, device, cfg, text_prototypes):
    """Evaluate using open-vocab inference."""
    num_classes = cfg["model"]["num_classes"]
    ov_cfg = cfg.get("open_vocab", {})
    inf_cfg = cfg["inference"]
    
    decoder.eval()
    backbone.eval()
    evaluator = PanopticEval(n_classes=num_classes, min_points=inf_cfg["min_points"])
    
    with torch.no_grad():
        for batch in dataloader:
            feats, coords = extract_backbone_features_e2e(backbone, batch, device, is_frozen=True)
            outputs = decoder(feats, coords)
            
            # Open-vocab inference
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
            
            # GT
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
    
    PQ, SQ, RQ, _, _, _ = evaluator.getPQ()
    iou_mean, _ = evaluator.getSemIoU()
    decoder.train()
    return {"PQ": PQ, "SQ": SQ, "RQ": RQ, "mIoU": iou_mean}


# ══════════════════════════════════════════════════════════════════════════════
# Checkpointing
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(path, epoch, decoder, backbone, optimizer, scheduler, scaler, best_pq, cfg, metrics=None):
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


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Open-vocab E2E panoptic training")
    p.add_argument("--config", default="panoptic/configs/mini_openvocab_e2e.yaml")
    p.add_argument("--resume", default=None)
    args, unknown = p.parse_known_args()
    return args, unknown


def main():
    args, cli_overrides = parse_args()
    cfg = load_config(args.config, cli_overrides)
    
    local_rank, rank, world_size, is_distributed, device = setup_distributed()
    is_main = (rank == 0)
    
    if is_main:
        print(f"{'='*70}")
        print(f"  OPEN-VOCABULARY E2E PANOPTIC TRAINING")
        print(f"  Class-agnostic masks + CLIP region features")
        print(f"{'='*70}")
    
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
        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # ── Text prototypes (diverse) ──
    text_prototypes = load_or_generate_prototypes(cfg, device)
    
    # ── Backbone (E2E) with Scene Context Encoder ──
    e2e_cfg = cfg.get("e2e", {"enabled": False})
    ctx_cfg = cfg.get("context_encoder", {})

    raw_backbone, _, _ = load_backbone_e2e(cfg["model"]["backbone_ckpt"], device, e2e_cfg)

    # Wrap backbone with SceneContextEncoder inserted at bottleneck
    ctx_dim   = ctx_cfg.get("dim", 256)
    ctx_heads = ctx_cfg.get("num_heads", 4)
    ctx_depth = ctx_cfg.get("depth", 1)
    use_pos_enc = ctx_cfg.get("use_pos_enc", True)

    backbone = ContextAwareBackboneWrapper(
        raw_backbone,
        context_dim=ctx_dim,
        context_heads=ctx_heads,
        context_depth=ctx_depth,
        use_pos_enc=use_pos_enc,
    ).to(device)

    # Freeze backbone encoder stages; context encoder + decoder start trainable
    backbone.freeze_backbone_encoder()
    if is_main:
        n_ctx = sum(p.numel() for p in backbone.context_encoder.parameters())
        n_bb_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        print(f"[Backbone] Context encoder: {n_ctx:,} params")
        print(f"[Backbone] Trainable total: {n_bb_trainable:,} params")

    unfreeze_after = e2e_cfg.get("unfreeze_after_epoch", 0)
    backbone_decoder_frozen = (unfreeze_after > 0)
    if backbone_decoder_frozen and e2e_cfg.get("enabled", False):
        # Also freeze backbone decoder initially; unfreeze at epoch `unfreeze_after`
        for name, param in backbone.backbone.named_parameters():
            decoder_names = ('convtr', 'bntr', 'block5', 'block6', 'block7', 'block8', 'final')
            if any(name.startswith(d) for d in decoder_names):
                param.requires_grad = False
        if is_main:
            print(f"[E2E] Backbone decoder frozen for first {unfreeze_after} epochs")
    
    # ── Open-Vocab Decoder ──
    mcfg = cfg["model"]
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
    
    # ── Warm-start decoder from closed-set checkpoint (optional) ──
    warmstart_path = e2e_cfg.get("warmstart_ckpt")
    if warmstart_path and warmstart_path != "null" and os.path.isfile(warmstart_path):
        ws_ckpt = torch.load(warmstart_path, map_location="cpu")
        # Load what matches (decoder layers, mask heads), skip new open_vocab_head
        missing, unexpected = decoder.load_state_dict(ws_ckpt["decoder"], strict=False)
        if is_main:
            print(f"[Warmstart] Loaded from {warmstart_path}")
            print(f"  Missing (new): {len(missing)} keys")
            print(f"  Unexpected (dropped): {len(unexpected)} keys")
    
    if is_main:
        n_dec = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        n_bb = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        print(f"[Params] Decoder: {n_dec:,}, Backbone: {n_bb:,}")
    
    # ── Loss (paper-aligned: L_cls + L_mask + L_O + L_V) ──
    lcfg = cfg["loss"]
    criterion = OpenVocabPanopticLoss(
        num_classes=mcfg["num_classes"],
        eos_coef=lcfg["eos_coef"],
        cost_class=lcfg["cost_class"],
        cost_mask=lcfg["cost_mask"],
        cost_dice=lcfg["cost_dice"],
        # Paper-aligned weights
        weight_cls=lcfg.get("weight_cls", 1.0),
        weight_mask=lcfg.get("weight_mask", 1.0),
        weight_dice=lcfg.get("weight_dice", 1.0),
        weight_object_distill=lcfg.get("weight_object_distill", 1.0),   # L_O
        weight_voxel_distill=lcfg.get("weight_voxel_distill", 0.1),     # L_V
        # B12/N4 split: novel classes excluded from cls loss at train time
        novel_class_idx=cfg.get("open_vocab", {}).get(
            "novel_class_idx", NUSCENES_B12N4_NOVEL),
        focal_gamma=lcfg["focal_gamma"],
        focal_alpha=lcfg["focal_alpha"],
        num_points=lcfg["num_points"],
        num_mask_pts=lcfg["num_mask_pts"],
        p_ratio=lcfg["p_ratio"],
    ).to(device)
    
    # ── Optimizer (3-way differential LR) ──
    # decoder  : full lr  (1e-4)         — new layers, need to train fast
    # ctx_enc  : medium lr (5e-5)        — new but must align with frozen encoder
    # bb_dec   : low lr   (backbone_lr)  — pre-trained, careful fine-tuning
    backbone_lr = e2e_cfg.get("backbone_lr", 5e-6)
    ctx_lr      = ctx_cfg.get("lr", tcfg["lr"] * 0.5)   # default: half decoder lr

    ctx_params = backbone.get_context_encoder_params()
    bb_dec_params = backbone.get_backbone_decoder_params()
    dec_params = [p for p in decoder.parameters() if p.requires_grad]

    param_groups = [
        {"params": dec_params,     "lr": tcfg["lr"],  "name": "decoder"},
        {"params": ctx_params,     "lr": ctx_lr,       "name": "ctx_encoder"},
        {"params": bb_dec_params,  "lr": backbone_lr,  "name": "bb_decoder"},
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
    warmup_steps = tcfg["warmup_epochs"] * steps_per_epoch
    total_steps = tcfg["epochs"] * steps_per_epoch
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps, min_lr=tcfg["min_lr"])
    
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
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if scaler and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_pq = ckpt.get("best_pq", 0.0)
        global_step = start_epoch * steps_per_epoch
        if is_main:
            print(f"[Resume] epoch {start_epoch}, best_pq={best_pq:.4f}")
    
    # ── Frozen-state tracking ──
    # backbone_decoder_frozen: convtr* layers frozen until unfreeze_after_epoch
    # encoder_is_frozen: encoder conv1-conv4 frozen; optionally unfrozen later
    unfreeze_encoder_flag = e2e_cfg.get("unfreeze_encoder", False)
    encoder_lr = e2e_cfg.get("encoder_lr", 1e-7)
    # backbone_decoder_frozen already set above
    encoder_is_frozen = True  # always starts frozen; may unfreeze at unfreeze_after_epoch+2
    backbone_is_frozen = backbone_decoder_frozen  # alias used in loop below

    if is_main:
        print(f"[E2E] unfreeze_encoder={unfreeze_encoder_flag}, "
              f"encoder_lr={encoder_lr:.1e}, "
              f"backbone decoder unfreeze at epoch {unfreeze_after}")

    # ── DDP ──
    if is_distributed:
        decoder = DDP(decoder, device_ids=[local_rank], find_unused_parameters=True)
        if any(p.requires_grad for p in backbone.parameters()):
            backbone = DDP(backbone, device_ids=[local_rank], find_unused_parameters=True)

    # ── Training ──
    if is_main:
        print(f"\n{'='*70}")
        print(f"  Open-Vocab E2E Training: {start_epoch}→{tcfg['epochs']-1}")
        print(f"  Decoder LR: {tcfg['lr']:.1e}, Backbone LR: {backbone_lr:.1e}")
        if unfreeze_encoder_flag:
            print(f"  Encoder will unfreeze at epoch {unfreeze_after + 2} @ lr={encoder_lr:.1e}")
        print(f"{'='*70}\n")
    
    log_every = cfg["logging"]["log_every"]
    eval_every = cfg["logging"]["eval_every"]
    save_every = cfg["checkpoint"]["save_every"]
    
    decoder.train()
    
    for epoch in range(start_epoch, tcfg["epochs"]):
        backbone_raw = backbone.module if hasattr(backbone, "module") else backbone

        # Step 1: Unfreeze backbone DECODER at unfreeze_after_epoch
        if backbone_is_frozen and epoch >= unfreeze_after and e2e_cfg["enabled"]:
            backbone_raw.unfreeze_backbone_decoder()
            backbone_is_frozen = False
            new_dec_params = [p for p in backbone_raw.get_backbone_decoder_params() if p.requires_grad]
            if new_dec_params:
                optimizer.add_param_group({"params": new_dec_params, "lr": backbone_lr, "name": "bb_decoder_unfrozen"})
            if is_main:
                print(f"\n[E2E] *** BACKBONE DECODER UNFROZEN at epoch {epoch} ***\n")

        # Step 2: Unfreeze backbone ENCODER 2 epochs after decoder unfreeze (if requested)
        if unfreeze_encoder_flag and encoder_is_frozen and epoch >= (unfreeze_after + 2) and e2e_cfg["enabled"]:
            enc_params, _ = backbone_raw.unfreeze_encoder_stages(
                stages=('conv3', 'bn3', 'block3', 'conv4', 'bn4', 'block4'),
                lr_override=encoder_lr,
            )
            encoder_is_frozen = False
            if enc_params:
                optimizer.add_param_group({"params": enc_params, "lr": encoder_lr, "name": "encoder_unfrozen"})
            if is_main:
                n_enc = sum(p.numel() for p in enc_params)
                print(f"\n[E2E] *** ENCODER (conv3-conv4) UNFROZEN at epoch {epoch} — {n_enc:,} params @ lr={encoder_lr:.1e} ***\n")
        
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        epoch_steps = 0
        optimizer.zero_grad()
        t_epoch = time.time()
        
        for step, batch in enumerate(train_loader):
            if killer.kill_now:
                if is_main:
                    save_checkpoint(os.path.join(save_dir, "interrupted.pth"),
                                    epoch, decoder, backbone, optimizer, scheduler, scaler, best_pq, cfg)
                sys.exit(0)
            
            feats, coords = extract_backbone_features_e2e(
                backbone, batch, device, is_frozen=backbone_is_frozen
            )
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = decoder(feats, coords)
                    targets = {
                        "classes": [batch["gt_classes"].to(device)],
                        "masks": [batch["gt_masks"].float().to(device)],
                    }
                    losses, _ = criterion(outputs, targets, text_prototypes, feats)
                    total_loss = sum(losses.values()) / tcfg["grad_accum"]
                scaler.scale(total_loss).backward()
            else:
                outputs = decoder(feats, coords)
                targets = {
                    "classes": [batch["gt_classes"].to(device)],
                    "masks": [batch["gt_masks"].float().to(device)],
                }
                losses, _ = criterion(outputs, targets, text_prototypes, feats)
                total_loss = sum(losses.values()) / tcfg["grad_accum"]
                total_loss.backward()
            
            epoch_loss += total_loss.item() * tcfg["grad_accum"]
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
                scheduler.step()
            
            if is_main and (step + 1) % log_every == 0:
                avg = epoch_loss / epoch_steps
                lr_now = optimizer.param_groups[0]["lr"]
                loss_parts = " ".join(f"{k}={v.item():.3f}" for k, v in losses.items() if not k[0].isdigit())
                print(f"  [E{epoch} S{step+1}/{len(train_loader)}] loss={avg:.4f} lr={lr_now:.2e}")
                print(f"    {loss_parts}")
                log_wandb({"train/loss": avg, "train/lr": lr_now}, step=global_step)
        
        # Epoch end
        epoch_time = time.time() - t_epoch
        if is_main:
            avg_loss = epoch_loss / max(epoch_steps, 1)
            gpu_mem = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0
            torch.cuda.reset_peak_memory_stats()
            print(f"Epoch {epoch}: loss={avg_loss:.4f}, time={epoch_time:.1f}s, gpu={gpu_mem:.0f}MB")
            log_wandb({"epoch/loss": avg_loss, "epoch/time_s": epoch_time}, step=global_step)
        
        # Eval
        if (epoch + 1) % eval_every == 0 or epoch == tcfg["epochs"] - 1:
            if is_main:
                decoder_raw = decoder.module if is_distributed else decoder
                backbone_raw = backbone.module if hasattr(backbone, "module") else backbone
                metrics = evaluate(decoder_raw, backbone_raw, val_loader, device, cfg, text_prototypes)
                print(f"  PQ={metrics['PQ']:.4f} SQ={metrics['SQ']:.4f} RQ={metrics['RQ']:.4f} mIoU={metrics['mIoU']:.4f}")
                log_wandb({"val/PQ": metrics["PQ"], "val/SQ": metrics["SQ"],
                           "val/RQ": metrics["RQ"], "val/mIoU": metrics["mIoU"]}, step=global_step)
                
                if metrics["PQ"] > best_pq:
                    best_pq = metrics["PQ"]
                    save_checkpoint(os.path.join(save_dir, "best.pth"),
                                    epoch, decoder, backbone, optimizer, scheduler, scaler, best_pq, cfg, metrics)
                    print(f"  ★ New best PQ={best_pq:.4f}")
            
            if is_distributed:
                dist.barrier()
            decoder.train()
        
        # Save
        if (epoch + 1) % save_every == 0 or epoch == tcfg["epochs"] - 1:
            if is_main:
                save_checkpoint(os.path.join(save_dir, "latest.pth"),
                                epoch, decoder, backbone, optimizer, scheduler, scaler, best_pq, cfg)
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
