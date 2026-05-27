"""Comprehensive panoptic evaluation: 8 modes in a single forward pass.

Modes:
  A. Linear         – pred_logits softmax → 16 known classes
  B. Token-16       – pred_tokens cosine sim → 16 known classes
  C. Ensemble       – alpha sweep blending A+B → 16 known classes
  D. Token-OpenVocab – pred_tokens cosine sim → 22 classes (16 known + 6 novel)
  E. Backbone-Pool  – mean-pool encoder feats per mask → 22 classes (legacy, buggy)
  F. Backbone-Pool-v2 – normalized soft-pool encoder feats → N classes (bug-fixed)
  G. Geometric-Ensemble – linear^alpha * clip_pool^(1-alpha) → N classes (Strategy E)
  H. Geo-Ensemble-v2 – G + multi-prompt prototypes + per-class temperature (improved)

Usage:
    python -u panoptic/scripts/evaluate_comprehensive.py \
        --config panoptic/configs/full_nuscenes_token_v2.yaml \
        --checkpoint panoptic/checkpoints_token_full_v2/best.pth \
        --version v1.0-mini \
        --novel-classes "police car,ambulance,trash cans,pole,bench"
"""
import argparse
import sys
import os
import json
import io
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import MinkowskiEngine as ME
import yaml

sys.path.insert(0, "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene")

from models.disnet import DisNet, state_dict_remove_moudle
from panoptic.models.mask_decoder import PanopticDecoder
from panoptic.eval.eval_pq import PanopticEval
from panoptic.dataset.nuscenes_panoptic import (
    NuScenesPanoptic, NUM_CLASSES, NUSCENES_CLASS_NAMES,
)

# ── Constants ──────────────────────────────────────────────────────────────
DEFAULT_NOVEL_CLASS_NAMES = ["police car", "ambulance", "trash cans", "pole", "bench"]
LEGACY_NOVEL_CLASS_NAMES = ["pole", "bench", "trash can", "fire hydrant", "tree trunk", "curb"]
THING_CLASSES_16 = set(range(10))  # 0-9 are things in nuScenes-16

ALL_MODES = ["linear", "token", "backbone_pool_v2", "geometric_ensemble", "geo_ensemble_v2"]

# ── Multi-prompt templates for novel classes ──────────────────────────────
# Each novel class has 5-8 descriptions emphasizing distinguishing features.
# Averaged embeddings are more distinctive than single generic prompts.
MULTI_PROMPT_TEMPLATES = {
    "police car": [
        "a police car with flashing lights on the road",
        "a law enforcement vehicle with sirens and markings",
        "a black and white police cruiser parked on the street",
        "a police patrol car with emergency lights on top",
        "a marked police sedan driving on the highway",
        "a police vehicle with decals and a light bar",
    ],
    "ambulance": [
        "a white ambulance with red cross markings",
        "an emergency ambulance with flashing lights and sirens",
        "a paramedic ambulance van parked at a hospital",
        "a large white emergency medical vehicle on the road",
        "an ambulance with a red stripe and emergency markings",
        "a boxy white ambulance driving with lights flashing",
    ],
    "trash cans": [
        "a plastic trash can on the sidewalk",
        "a green garbage bin at the curb for waste collection",
        "a wheeled trash container next to a building",
        "a large rectangular waste bin on the street",
        "a dumpster or trash receptacle near a building",
        "a cylindrical garbage can on the pavement",
    ],
    "pole": [
        "a thin vertical street light pole on the sidewalk",
        "a tall metal lamp post with a light fixture on top",
        "a narrow sign pole standing vertically on the curb",
        "a thin cylindrical traffic signal pole at an intersection",
        "a slender metal utility pole along the road",
    ],
    "bench": [
        "a park bench with a backrest on the sidewalk",
        "a wooden bench seat near a bus stop",
        "a metal bench with armrests on the pavement",
        "a public sitting bench by the roadside",
        "a flat seating bench at a pedestrian area",
        "a horizontal bench with legs on the ground",
    ],
}

# ── Per-class temperature (higher = flatter/less dominant) ────────────────
# DEPRECATED: Hardcoded per-class temperatures are not generalizable.
# Replaced by automatic prototype bias correction (see compute_prototype_bias
# and panoptic_inference_geometric_ensemble_v2). Kept for reference only.
PER_CLASS_TEMPERATURE_NOVEL = {
    "police car": 0.07,
    "ambulance": 0.07,
    "trash cans": 0.1,
    "pole": 0.4,
    "bench": 0.1,
}


# ── TeeLogger ──────────────────────────────────────────────────────────────
class TeeLogger:
    """Duplicate stdout to both the terminal and a log file simultaneously.

    Used in the comprehensive evaluation script to capture all printed output
    into a persistent log file while still displaying it in real time on the
    console.

    Attributes:
        terminal: The original sys.stdout stream.
        log: File handle opened for writing at the specified log_path.
    """
    def __init__(self, log_path):
        """Initialize TeeLogger by opening the log file for writing.

        Creates the dual-output logger that will mirror all writes to both
        the original stdout and the specified log file on disk.

        Steps:
            1. Stores a reference to the current sys.stdout as self.terminal.
            2. Opens log_path in write mode as self.log.

        Args:
            log_path (str): Absolute path where the log file will be created.
        """
        self.terminal = sys.stdout
        self.log = open(log_path, "w")

    def write(self, msg):
        """Write a message to both the terminal and the log file.

        Called implicitly by print() when sys.stdout is replaced with this
        TeeLogger instance, ensuring output appears in both destinations.

        Steps:
            1. Writes msg to the original terminal stream.
            2. Writes msg to the opened log file.

        Args:
            msg (str): The string to write.
        """
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self):
        """Flush both the terminal and log file output buffers.

        Ensures that all buffered output is immediately written to the
        underlying streams, preventing data loss on crash or early exit.

        Steps:
            1. Flushes the terminal stream buffer.
            2. Flushes the log file buffer.
        """
        self.terminal.flush()
        self.log.flush()

    def close(self):
        """Close the log file handle.

        Should be called when evaluation is complete to release the file
        descriptor. The terminal stream is not closed as it belongs to
        the process.

        Steps:
            1. Closes the log file handle.
        """
        self.log.close()


# ── Model loading ──────────────────────────────────────────────────────────
def load_config(config_path):
    """Load a YAML configuration file and return it as a dictionary.

    Used at the start of comprehensive evaluation to read all model, data,
    inference, and token hyperparameters from a single config file.

    Steps:
        1. Opens the file at config_path in read mode.
        2. Parses the YAML content with yaml.safe_load.
        3. Returns the resulting nested dictionary.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_backbone(ckpt_path, device):
    """Load the DisNet 3D backbone from checkpoint and freeze all parameters.

    Used in the comprehensive evaluation pipeline to provide per-voxel
    features that are consumed by the panoptic decoder and optionally
    pooled for the backbone open-vocabulary baseline (Mode E).

    Steps:
        1. Instantiates DisNet with openseg/MinkUNet18A configuration.
        2. Loads and cleans the state dict from the checkpoint file.
        3. Sets the model to eval mode and disables gradient computation.

    Args:
        ckpt_path (str): Path to the backbone .pth.tar checkpoint.
        device (torch.device): Target device (cpu or cuda).

    Returns:
        DisNet: The loaded and frozen backbone model.
    """
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
    return model


def extract_backbone_features(backbone, batch, device):
    """Extract per-voxel features from the 3D backbone for a single sample.

    Called once per validation sample to produce the feature tensor that
    feeds into both the panoptic decoder and the backbone pooling baseline.

    Steps:
        1. Prepends a batch-index column of zeros to voxel coordinates.
        2. Constructs a MinkowskiEngine SparseTensor on the target device.
        3. Runs the backbone in no-grad mode.
        4. Returns features and coordinates each with a leading batch dim.

    Args:
        backbone (DisNet): The frozen 3D backbone model.
        batch (dict): Dataset sample containing 'vox_coords' and 'vox_feats'.
        device (torch.device): Target device for computation.

    Returns:
        tuple: (feats, coords_xyz) where feats is (1, V, D) and
            coords_xyz is (1, V, 3).
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
    with torch.no_grad():
        output = backbone(sinput)
    feats = output.F if hasattr(output, "F") else output
    coords_xyz = vox_coords.to(device).float()
    return feats.unsqueeze(0), coords_xyz.unsqueeze(0)


def load_decoder_flexible(cfg, ckpt_path, device):
    """Load decoder, auto-detect token_head from checkpoint keys."""
    mcfg = cfg["model"]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    decoder_state = ckpt["decoder"]
    has_token_head = any("token_head" in k for k in decoder_state.keys())

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
        use_token_head=has_token_head,
        token_dim=mcfg.get("token_dim", 768),
    ).to(device)
    decoder.load_state_dict(decoder_state, strict=True)
    decoder.eval()
    return decoder, has_token_head


# ── Prototype loading/generation ───────────────────────────────────────────
def load_or_generate_22_prototypes(known_proto_path, novel_proto_path, device,
                                    novel_class_names=None):
    """Load 16-class protos, extend with novel classes.

    Args:
        known_proto_path: Path to 16-class prototype .pt file.
        novel_proto_path: Path to pre-computed extended prototype file (optional).
        device: Target device.
        novel_class_names: List of novel class name strings. Defaults to DEFAULT_NOVEL_CLASS_NAMES.

    Returns:
        tuple: (known_emb on device, all_emb on device or None if generation fails).
    """
    if novel_class_names is None:
        novel_class_names = DEFAULT_NOVEL_CLASS_NAMES

    known_data = torch.load(known_proto_path, map_location="cpu")
    known_emb = known_data["embeddings"].float()  # (16, 768)

    if novel_proto_path and os.path.exists(novel_proto_path):
        novel_data = torch.load(novel_proto_path, map_location="cpu")
        all_emb = novel_data["embeddings"].float()
        print(f"Loaded extended prototypes from {novel_proto_path} ({all_emb.shape[0]} classes)", flush=True)
        return known_emb.to(device), all_emb.to(device)

    # Generate novel prototypes on the fly
    num_novel = len(novel_class_names)
    print(f"Generating {num_novel} novel class prototypes with CLIP: {novel_class_names}", flush=True)
    # Check if cached file exists
    auto_save_path = novel_proto_path or known_proto_path.replace(".pt", f"_{16+num_novel}class.pt")
    if os.path.exists(auto_save_path):
        cached = torch.load(auto_save_path, map_location="cpu")
        all_emb = cached["embeddings"].float()
        print(f"Loaded cached {all_emb.shape[0]}-class prototypes from {auto_save_path}", flush=True)
        return known_emb.to(device), all_emb.to(device)

    try:
        import clip
        # CLIP runs on GPU here — panoptic model not loaded yet
        model, _ = clip.load("ViT-L/14@336px", device=device, jit=False)
        model.eval()
        template = known_data.get("prompt_template", "a {} in a scene")
        prompts = [template.format(name) for name in novel_class_names]
        tokens = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            novel_feats = model.encode_text(tokens)
            novel_feats = novel_feats / novel_feats.norm(dim=-1, keepdim=True)
        novel_feats = novel_feats.float().cpu()
        all_emb = torch.cat([known_emb, novel_feats], dim=0)  # (16+N, 768)
        # Save for next time
        save_path = auto_save_path
        all_class_names = list(NUSCENES_CLASS_NAMES) + novel_class_names
        torch.save({
            "embeddings": all_emb,
            "class_names": all_class_names,
            "prompt_template": template,
        }, save_path)
        print(f"Saved {all_emb.shape[0]}-class prototypes to {save_path}", flush=True)
        del model
        torch.cuda.empty_cache()
        return known_emb.to(device), all_emb.to(device)
    except ImportError:
        print("ERROR: CLIP not installed. Cannot generate novel prototypes.", flush=True)
        print("Install with: pip install git+https://github.com/openai/CLIP.git", flush=True)
        return known_emb.to(device), None


def generate_multi_prompt_prototypes(known_proto_path, device, novel_class_names=None):
    """Generate novel class prototypes using multi-prompt ensembling.

    Instead of a single "a {class} in a scene" template, each novel class is
    represented by the average of 5-8 diverse descriptions that emphasize
    distinguishing features. This produces more distinctive embeddings that
    separate novel classes from similar known classes.

    Args:
        known_proto_path: Path to 16-class prototype .pt file.
        device: Target device.
        novel_class_names: List of novel class names.

    Returns:
        tuple: (known_emb, all_emb) on device, or (known_emb, None) on failure.
    """
    if novel_class_names is None:
        novel_class_names = DEFAULT_NOVEL_CLASS_NAMES

    known_data = torch.load(known_proto_path, map_location="cpu")
    known_emb = known_data["embeddings"].float()  # (16, 768)

    print(f"Generating multi-prompt ensembled prototypes for {novel_class_names}", flush=True)
    # Check if cached multi-prompt file exists
    save_path = known_proto_path.replace(".pt", f"_multiprompt_{16 + len(novel_class_names)}class.pt")
    if os.path.exists(save_path):
        cached = torch.load(save_path, map_location="cpu")
        all_emb = cached["embeddings"].float().to(device)
        print(f"Loaded cached multi-prompt prototypes from {save_path} ({all_emb.shape[0]} classes)", flush=True)
        return known_emb.to(device), all_emb

    try:
        import clip
        # CLIP runs on GPU here — panoptic model not loaded yet
        model, _ = clip.load("ViT-L/14@336px", device=device, jit=False)
        model.eval()

        novel_embeddings = []
        for name in novel_class_names:
            # Use multi-prompt templates if available, else fallback
            if name in MULTI_PROMPT_TEMPLATES:
                prompts = MULTI_PROMPT_TEMPLATES[name]
            else:
                prompts = [
                    f"a {name} in a scene",
                    f"a {name} on the road",
                    f"a {name} in an urban environment",
                    f"a {name} near the street",
                    f"a photo of a {name}",
                ]

            tokens = clip.tokenize(prompts).to(device)
            with torch.no_grad():
                feats = model.encode_text(tokens)  # (N_prompts, 768)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                avg_feat = feats.mean(dim=0)
                avg_feat = avg_feat / avg_feat.norm()

            novel_embeddings.append(avg_feat.float().cpu())
            print(f"    {name}: {len(prompts)} prompts averaged", flush=True)

        del model
        torch.cuda.empty_cache()

        novel_emb = torch.stack(novel_embeddings)  # (N_novel, 768)
        all_emb = torch.cat([known_emb, novel_emb], dim=0)

        # Save for future runs (avoids reloading CLIP)
        torch.save({
            "embeddings": all_emb,
            "class_names": list(NUSCENES_CLASS_NAMES) + novel_class_names,
            "multi_prompt": True,
        }, save_path)
        print(f"Saved multi-prompt {all_emb.shape[0]}-class prototypes to {save_path}", flush=True)

        return known_emb.to(device), all_emb.to(device)

    except ImportError:
        print("ERROR: CLIP not installed.", flush=True)
        return known_emb.to(device), None


def compute_prototype_bias(backbone, val_loader, text_protos, device, num_calibration=50):
    """Compute per-prototype background affinity bias from calibration samples.

    For each text prototype, computes the average cosine similarity to
    randomly pooled backbone features. Prototypes with high average similarity
    to "anything" (like "pole") get a high bias value that will be subtracted
    at inference, suppressing false positives.

    Args:
        backbone: Frozen DisNet backbone model.
        val_loader: DataLoader for calibration (uses first num_calibration samples).
        text_protos: (C, 768) all text prototypes (known + novel).
        device: Target device.
        num_calibration: Number of samples to use for bias estimation.

    Returns:
        Tensor: (C,) per-prototype bias (average background similarity).
    """
    num_classes = text_protos.shape[0]
    sim_accum = torch.zeros(num_classes, device=device)
    total_queries = 0

    print(f"Computing prototype bias from {num_calibration} calibration samples...", flush=True)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_calibration:
                break

            feats, _ = extract_backbone_features(backbone, batch, device)
            feats_v = feats[0]  # (V, 768)
            V = feats_v.shape[0]

            # Normalize
            feats_normed = F.normalize(feats_v.float(), dim=-1)

            # Random "queries": groups of ~100 voxels averaged (simulates mask-pooling)
            num_random_queries = min(30, V // 100)
            for _ in range(num_random_queries):
                size = torch.randint(50, min(200, V), (1,)).item()
                idx = torch.randperm(V, device=device)[:size]
                pooled = F.normalize(feats_normed[idx].mean(dim=0, keepdim=True), dim=-1)  # (1, 768)
                sim = (pooled @ text_protos.T).squeeze(0)  # (C,)
                sim_accum += sim
                total_queries += 1

            if (i + 1) % 25 == 0:
                print(f"    [{i+1}/{num_calibration}]", flush=True)

    bias = sim_accum / max(total_queries, 1)  # (C,) average similarity per prototype
    print(f"  Prototype bias range: [{bias.min():.4f}, {bias.max():.4f}]", flush=True)
    all_names = list(NUSCENES_CLASS_NAMES)
    for c in range(num_classes):
        name = all_names[c] if c < len(all_names) else f"novel_{c-len(all_names)}"
        print(f"    {name:<25s}: {bias[c]:.4f}", flush=True)

    return bias


# ── Inference functions ────────────────────────────────────────────────────
def _greedy_assign(scores, labels, masks, V, overlap_thr, thing_set=None):
    """Greedy panoptic assignment. thing_set defaults to 0-9 for nuScenes-16."""
    if thing_set is None:
        thing_set = THING_CLASSES_16
    sorted_idx = scores.argsort(descending=True)
    sem_pred = np.full(V, 0, dtype=np.int64)
    inst_pred = np.zeros(V, dtype=np.int64)
    inst_id = 1
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


def panoptic_inference_linear(pred_logits, pred_masks, num_classes=16, overlap_thr=0.8):
    """Perform panoptic inference using the linear classification head (Mode A).

    Applies softmax over predicted logits to obtain class scores, filters
    out the no-object class, and delegates to greedy mask assignment.

    Steps:
        1. Softmax on pred_logits, take max score and argmax label per query.
        2. Sigmoid on pred_masks to get per-voxel mask probabilities.
        3. Filters queries whose predicted class is the no-object class.
        4. Calls _greedy_assign for final panoptic assignment.

    Args:
        pred_logits (Tensor): Query logits of shape (Q, C+1).
        pred_masks (Tensor): Mask logits of shape (V, Q).
        num_classes (int): Number of valid classes (default 16).
        overlap_thr (float): Minimum overlap ratio for mask acceptance.

    Returns:
        tuple: (sem_pred, inst_pred) numpy arrays of shape (V,).
    """
    scores, labels = pred_logits.softmax(-1).max(-1)
    masks = pred_masks.sigmoid()
    V = masks.shape[0]
    keep = labels != num_classes
    if keep.sum() == 0:
        return np.zeros(V, dtype=np.int64), np.zeros(V, dtype=np.int64)
    return _greedy_assign(scores[keep], labels[keep], masks[:, keep], V, overlap_thr)


def panoptic_inference_token(pred_tokens, pred_masks, text_prototypes, overlap_thr=0.8):
    """Token cosine sim against prototypes (works for 16 or 22 classes)."""
    cos_sim = torch.mm(pred_tokens, text_prototypes.t())  # (Q, C)
    scores, labels = cos_sim.max(-1)
    masks = pred_masks.sigmoid()
    V = masks.shape[0]
    keep = scores > 0.1
    if keep.sum() == 0:
        return np.zeros(V, dtype=np.int64), np.zeros(V, dtype=np.int64)
    # For 22-class: novel classes (16-21) treated as things
    num_protos = text_prototypes.shape[0]
    thing_set = THING_CLASSES_16 | set(range(16, num_protos)) if num_protos > 16 else THING_CLASSES_16
    return _greedy_assign(scores[keep], labels[keep], masks[:, keep], V, overlap_thr, thing_set)


def panoptic_inference_backbone_pool_v2(backbone_feats, pred_masks, text_protos,
                                         overlap_thr=0.8, min_mask_pts=50):
    """Fixed backbone-pool: normalize BEFORE pooling, soft mask-weighted averaging.

    This fixes the PQ=0.209 bug in the legacy backbone_pool by:
    1. L2-normalizing backbone features before pooling (prevents large-norm dominance)
    2. Using soft sigmoid mask weights instead of binary thresholding
    3. Per-query weight normalization for proper averaging
    4. Filtering by mask area as an objectness proxy

    Args:
        backbone_feats (Tensor): Raw backbone output, shape (V, 768).
        pred_masks (Tensor): Mask logits from decoder, shape (V, Q).
        text_protos (Tensor): Text prototypes, shape (C, 768) for C classes.
        overlap_thr (float): Overlap threshold for greedy assignment.
        min_mask_pts (float): Minimum soft mask area to consider a query valid.

    Returns:
        tuple: (sem_pred, inst_pred) numpy arrays of shape (V,).
    """
    V, Q = pred_masks.shape
    num_protos = text_protos.shape[0]

    # 1. Normalize backbone features BEFORE pooling (critical fix)
    backbone_normed = F.normalize(backbone_feats.float(), dim=-1)  # (V, 768) unit sphere

    # 2. Soft mask-weighted pooling
    mask_probs = pred_masks.sigmoid()  # (V, Q) soft [0,1]
    mask_area = mask_probs.sum(dim=0)  # (Q,) total activation per query

    # 3. Per-query normalized weights for proper averaging
    weights = mask_probs / mask_area.clamp(min=1e-6).unsqueeze(0)  # (V, Q)

    # 4. Weighted average pooling + re-normalize
    pooled = weights.T @ backbone_normed  # (Q, 768)
    pooled = F.normalize(pooled, dim=-1)

    # 5. Classify via cosine similarity
    sim = pooled @ text_protos.T  # (Q, C)
    scores, labels = sim.max(dim=-1)

    # 6. Filter by mask area (objectness proxy)
    scores[mask_area < min_mask_pts] = -1.0

    # 7. Greedy panoptic assignment
    thing_set = THING_CLASSES_16 | set(range(16, num_protos))
    return _greedy_assign(scores, labels, mask_probs, V, overlap_thr, thing_set)


def panoptic_inference_geometric_ensemble(pred_logits, pred_masks, backbone_feats,
                                           text_protos_all, num_known=16,
                                           overlap_thr=0.8, alpha=0.5,
                                           temperature=0.1, min_mask_pts=50):
    """Geometric ensemble: linear head (known) + backbone pool (known+novel).

    For known classes (0..num_known-1): combined = linear_prob^alpha * clip_prob^(1-alpha)
    For novel classes (num_known..C-1): combined = clip_prob only

    This enables open-vocabulary detection by leveraging the linear head's strength
    on known classes while allowing the frozen backbone's CLIP features to detect
    novel objects via mask-pooled cosine similarity.

    Args:
        pred_logits (Tensor): Query logits, shape (Q, num_known+1).
        pred_masks (Tensor): Mask logits, shape (V, Q).
        backbone_feats (Tensor): Raw backbone features, shape (V, 768).
        text_protos_all (Tensor): All text prototypes, shape (C, 768) where C = known + novel.
        num_known (int): Number of known classes (default 16).
        overlap_thr (float): Overlap threshold for greedy assignment.
        alpha (float): Blend weight. 1.0 = pure linear, 0.0 = pure CLIP.
        temperature (float): Softmax temperature for CLIP logits.
        min_mask_pts (float): Minimum soft mask area for valid queries.

    Returns:
        tuple: (sem_pred, inst_pred) numpy arrays of shape (V,).
    """
    V, Q = pred_masks.shape
    num_total = text_protos_all.shape[0]

    # ── Linear head probs (known classes only) ──
    linear_probs = pred_logits[:, :num_known].softmax(dim=-1)  # (Q, num_known)

    # ── CLIP backbone-pool probs (all classes) ──
    backbone_normed = F.normalize(backbone_feats.float(), dim=-1)  # (V, 768)
    mask_probs = pred_masks.sigmoid()  # (V, Q)
    mask_area = mask_probs.sum(dim=0)  # (Q,)
    weights = mask_probs / mask_area.clamp(min=1e-6).unsqueeze(0)  # (V, Q)
    pooled = F.normalize(weights.T @ backbone_normed, dim=-1)  # (Q, 768)
    clip_logits = pooled @ text_protos_all.T / temperature  # (Q, C_all)
    clip_probs = clip_logits.softmax(dim=-1)  # (Q, C_all)

    # ── Geometric ensemble ──
    combined = torch.zeros(Q, num_total, device=pred_logits.device)
    # Known classes: geometric mean
    combined[:, :num_known] = (linear_probs ** alpha) * (clip_probs[:, :num_known] ** (1 - alpha))
    # Novel classes: CLIP only
    if num_total > num_known:
        combined[:, num_known:] = clip_probs[:, num_known:]
    # Renormalize
    combined = combined / combined.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    scores, labels = combined.max(dim=-1)

    # ── Objectness from linear head's no-object score ──
    no_obj_prob = pred_logits.softmax(dim=-1)[:, -1]  # P(no-object)
    objectness = 1.0 - no_obj_prob
    scores = scores * objectness

    # Filter tiny masks
    scores[mask_area < min_mask_pts] = -1.0

    # Greedy assign (novel classes treated as things)
    thing_set = THING_CLASSES_16 | set(range(num_known, num_total))
    return _greedy_assign(scores, labels, mask_probs, V, overlap_thr, thing_set)


def panoptic_inference_geometric_ensemble_v2(pred_logits, pred_masks, backbone_feats,
                                              text_protos_all, proto_bias,
                                              num_known=16, overlap_thr=0.8, alpha=0.5,
                                              temperature=0.1, min_mask_pts=50):
    """Geometric ensemble v2: multi-prompt prototypes + automatic bias correction.

    Key improvements over v1:
    1. Multi-prompt ensembled prototypes (more distinctive embeddings).
    2. Prototype bias correction: subtracts each prototype's "background affinity"
       (average similarity to random features) before softmax. Prototypes that are
       generically similar to everything (like "pole") get suppressed automatically.
       This is fully generalized — works for any novel class without manual tuning.

    The corrected logit is: logit_c = (cos_sim_c - bias_c) / temperature
    A prototype with high bias (high average similarity to random pooled features)
    needs a HIGHER actual similarity to fire — effectively raising its threshold.

    Args:
        pred_logits (Tensor): Query logits, shape (Q, num_known+1).
        pred_masks (Tensor): Mask logits, shape (V, Q).
        backbone_feats (Tensor): Raw backbone features, shape (V, 768).
        text_protos_all (Tensor): All text prototypes, shape (C, 768).
        proto_bias (Tensor): Per-prototype background affinity, shape (C,).
        num_known (int): Number of known classes (default 16).
        overlap_thr (float): Overlap threshold for greedy assignment.
        alpha (float): Blend weight (1.0=pure linear, 0.0=pure CLIP).
        temperature (float): Global softmax temperature.
        min_mask_pts (float): Minimum soft mask area for valid queries.

    Returns:
        tuple: (sem_pred, inst_pred) numpy arrays of shape (V,).
    """
    V, Q = pred_masks.shape
    num_total = text_protos_all.shape[0]

    # ── Linear head probs (known classes only) ──
    linear_probs = pred_logits[:, :num_known].softmax(dim=-1)  # (Q, num_known)

    # ── CLIP backbone-pool with BIAS CORRECTION ──
    backbone_normed = F.normalize(backbone_feats.float(), dim=-1)  # (V, 768)
    mask_probs = pred_masks.sigmoid()  # (V, Q)
    mask_area = mask_probs.sum(dim=0)  # (Q,)
    weights = mask_probs / mask_area.clamp(min=1e-6).unsqueeze(0)  # (V, Q)
    pooled = F.normalize(weights.T @ backbone_normed, dim=-1)  # (Q, 768)

    # Raw cosine similarity
    cos_sim = pooled @ text_protos_all.T  # (Q, C_all) values in [-1, 1]

    # Bias correction: subtract each prototype's background affinity
    # This makes "pole" (bias≈0.3) need cos_sim≈0.3+ to even register,
    # while "police car" (bias≈0.15) fires at cos_sim≈0.2
    corrected_logits = (cos_sim - proto_bias.unsqueeze(0)) / temperature  # (Q, C_all)
    clip_probs = corrected_logits.softmax(dim=-1)  # (Q, C_all)

    # ── Geometric ensemble ──
    combined = torch.zeros(Q, num_total, device=pred_logits.device)
    combined[:, :num_known] = (linear_probs ** alpha) * (clip_probs[:, :num_known] ** (1 - alpha))
    if num_total > num_known:
        combined[:, num_known:] = clip_probs[:, num_known:]
    # Renormalize
    combined = combined / combined.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    scores, labels = combined.max(dim=-1)

    # ── Objectness from linear head's no-object score ──
    no_obj_prob = pred_logits.softmax(dim=-1)[:, -1]
    objectness = 1.0 - no_obj_prob
    scores = scores * objectness

    # Filter tiny masks
    scores[mask_area < min_mask_pts] = -1.0

    # Greedy assign
    thing_set = THING_CLASSES_16 | set(range(num_known, num_total))
    return _greedy_assign(scores, labels, mask_probs, V, overlap_thr, thing_set)


# ── Novel class stats ──────────────────────────────────────────────────────
def count_novel_detections(sem_pred, num_known=16, novel_class_names=None):
    """Count voxels and instances predicted as novel classes."""
    if novel_class_names is None:
        novel_class_names = DEFAULT_NOVEL_CLASS_NAMES
    stats = {}
    for i, name in enumerate(novel_class_names):
        cls_id = num_known + i
        mask = sem_pred == cls_id
        stats[name] = {"voxels": int(mask.sum()), "class_id": cls_id}
    return stats


def remap_novel_to_ignore(sem_pred, inst_pred, num_known=16):
    """For known-class PQ eval, set novel predictions to 0 (background/ignore)."""
    sem_out = sem_pred.copy()
    inst_out = inst_pred.copy()
    novel_mask = sem_out >= num_known
    sem_out[novel_mask] = 0
    inst_out[novel_mask] = 0
    return sem_out, inst_out


# ── Table formatting ───────────────────────────────────────────────────────
def format_table(pq_all, sq_all, rq_all, iou_all, class_names):
    """Format per-class PQ/SQ/RQ/IoU metrics as an aligned text table.

    Used after evaluation to produce human-readable output for both
    console display and saving to per-mode text files.

    Steps:
        1. Creates a header row with column names.
        2. Iterates over class_names, formatting each metric to 4 decimals.
        3. Joins all lines with newlines and returns the result.

    Args:
        pq_all (array-like): Per-class PQ values.
        sq_all (array-like): Per-class SQ values.
        rq_all (array-like): Per-class RQ values.
        iou_all (array-like): Per-class IoU values.
        class_names (list[str]): List of class name strings.

    Returns:
        str: Multi-line formatted table string.
    """
    lines = []
    lines.append(f"  {'Class':<25s} {'PQ':>8s} {'SQ':>8s} {'RQ':>8s} {'IoU':>8s}")
    lines.append(f"  {'-'*57}")
    for c in range(len(class_names)):
        pq = pq_all[c] if c < len(pq_all) else 0.0
        sq = sq_all[c] if c < len(sq_all) else 0.0
        rq = rq_all[c] if c < len(rq_all) else 0.0
        iou = iou_all[c] if c < len(iou_all) else 0.0
        lines.append(f"  {class_names[c]:<25s} {pq:8.4f} {sq:8.4f} {rq:8.4f} {iou:8.4f}")
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    """Run the comprehensive 5-mode panoptic evaluation pipeline.

    This is the top-level entry point that evaluates a panoptic model across
    linear, token, ensemble (alpha sweep), token open-vocabulary, and backbone
    pooling modes in a single pass over the validation set.

    Steps:
        1. Parses CLI arguments, loads config, and sets up output directory with logging.
        2. Loads dataset, backbone, decoder (auto-detecting token head), and prototypes.
        3. Filters requested modes based on model capabilities.
        4. Iterates over all validation samples, running applicable inference modes.
        5. Collects and prints per-mode results tables and summary comparison.
        6. Reports novel class detection statistics for open-vocabulary modes.
        7. Saves results.json and novel_detections.json to the output directory.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Comprehensive panoptic evaluation (5 modes)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--version", default=None, help="Override data version (v1.0-mini or v1.0-trainval)")
    parser.add_argument("--dataroot", default=None, help="Override data root path")
    parser.add_argument("--output_dir", default="panoptic/eval_results/comprehensive/")
    parser.add_argument("--modes", default=",".join(ALL_MODES),
                        help="Comma-separated modes: " + ",".join(ALL_MODES))
    parser.add_argument("--prototype_path", default=None, help="16-class prototype path (default from config)")
    parser.add_argument("--novel_prototype_path", default=None, help="22-class prototype path (auto-generated if missing)")
    parser.add_argument("--novel-classes", default=None,
                        help="Comma-separated novel class names (default: police car,ambulance,trash cans,pole,bench)")
    parser.add_argument("--geo_alphas", default="0.3,0.5,0.7", help="Alpha values for geometric ensemble sweep")
    parser.add_argument("--geo_temperature", type=float, default=0.1, help="Temperature for geometric ensemble CLIP logits")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modes = [m.strip() for m in args.modes.split(",")]
    geo_alphas = [float(a) for a in args.geo_alphas.split(",")]

    # Resolve novel class names (CLI > config > default)
    if args.novel_classes:
        novel_class_names = [n.strip() for n in args.novel_classes.split(",")]
    else:
        novel_class_names = cfg.get("novel_classes", DEFAULT_NOVEL_CLASS_NAMES)

    # Determine version and dataroot
    version = args.version or cfg["data"]["version"]
    if args.dataroot:
        cfg["data"]["dataroot"] = args.dataroot

    # Create output directory
    ckpt_stem = Path(args.checkpoint).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_tag = version.replace("v1.0-", "")
    run_name = f"{ckpt_stem}_{version_tag}_{timestamp}"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Start logging
    tee = TeeLogger(str(out_dir / "eval_log.txt"))
    sys.stdout = tee

    print(f"{'='*70}", flush=True)
    print(f"  Comprehensive Panoptic Evaluation", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Config:     {args.config}", flush=True)
    print(f"  Checkpoint: {args.checkpoint}", flush=True)
    print(f"  Version:    {version}", flush=True)
    print(f"  Modes:      {modes}", flush=True)
    print(f"  Output:     {out_dir}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # ── Load data ──
    dcfg = cfg["data"]
    from nuscenes.nuscenes import NuScenes as NuSc
    print(f"Loading NuScenes database (version={version}, dataroot={dcfg['dataroot']})...", flush=True)
    nusc = NuSc(version=version, dataroot=dcfg["dataroot"], verbose=False)
    print(f"  NuScenes loaded: {len(nusc.scene)} scenes", flush=True)
    print(f"Creating NuScenesPanoptic dataset (split=val)...", flush=True)
    val_ds = NuScenesPanoptic(
        dcfg["dataroot"], version, split="val",
        voxel_size=dcfg["voxel_size"],
        point_range=np.array(dcfg["point_range"]),
        augment=False, aug_cfg=None, nusc=nusc,
    )
    print(f"  Dataset created: {len(val_ds)} samples", flush=True)
    print(f"Creating DataLoader (num_workers={args.num_workers})...", flush=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
        collate_fn=lambda x: x[0],
    )
    print(f"[NuScenesPanoptic] val: {len(val_ds)} samples from {version}\n", flush=True)

    # ── Generate prototypes (CLIP on GPU first, before loading panoptic model) ──
    token_cfg = cfg.get("token", {})
    proto_path = args.prototype_path or token_cfg.get(
        "prototype_path", "panoptic/clip_prototypes/text_prototypes_vitl14_336.pt"
    )
    text_protos_16 = None
    text_protos_22 = None
    text_protos_mp = None
    proto_bias = None

    needs_16 = "token" in modes
    needs_22 = any(m in modes for m in ["backbone_pool_v2", "geometric_ensemble",
                                         "geo_ensemble_v2"])

    if needs_16 or needs_22:
        if needs_22:
            text_protos_16, text_protos_22 = load_or_generate_22_prototypes(
                proto_path, args.novel_prototype_path, device, novel_class_names
            )
            if text_protos_22 is None:
                print("WARNING: Could not load/generate extended prototypes, skipping open-vocab modes\n", flush=True)
                modes = [m for m in modes if m not in {"backbone_pool_v2", "geometric_ensemble",
                                                        "geo_ensemble_v2"}]
            # Generate multi-prompt prototypes for v2
            if "geo_ensemble_v2" in modes:
                _, text_protos_mp = generate_multi_prompt_prototypes(
                    proto_path, device, novel_class_names
                )
                if text_protos_mp is None:
                    print("WARNING: Could not generate multi-prompt prototypes, skipping geo_ensemble_v2\n", flush=True)
                    modes = [m for m in modes if m != "geo_ensemble_v2"]
        else:
            proto_data = torch.load(proto_path, map_location="cpu")
            text_protos_16 = proto_data["embeddings"].float().to(device)

        if text_protos_16 is not None:
            print(f"16-class prototypes: {text_protos_16.shape}", flush=True)
        if text_protos_22 is not None:
            print(f"22-class prototypes: {text_protos_22.shape}", flush=True)
        if text_protos_mp is not None:
            print(f"Multi-prompt prototypes: {text_protos_mp.shape}", flush=True)
        print(flush=True)

    # Clear any CLIP remnants from GPU before loading panoptic models
    torch.cuda.empty_cache()
    print("CLIP unloaded, GPU cache cleared.", flush=True)

    # ── Load panoptic models ──
    print(f"Loading backbone from: {cfg['model']['backbone_ckpt']}", flush=True)
    backbone = load_backbone(cfg["model"]["backbone_ckpt"], device)
    print("  Backbone loaded.", flush=True)
    print(f"Loading decoder from: {args.checkpoint}", flush=True)
    decoder, has_token_head = load_decoder_flexible(cfg, args.checkpoint, device)
    print(f"  Decoder loaded. Token head detected: {has_token_head}\n", flush=True)

    # ── Filter modes based on capabilities ──
    token_modes = {"token"}
    if not has_token_head:
        skipped = [m for m in modes if m in token_modes]
        if skipped:
            print(f"WARNING: No token_head — skipping modes: {skipped}\n", flush=True)
        modes = [m for m in modes if m not in token_modes]

    # ── Compute prototype bias for geo_ensemble_v2 ──
    if "geo_ensemble_v2" in modes and text_protos_mp is not None:
        print("Computing prototype bias (calibration on 50 samples)...", flush=True)
        proto_bias = compute_prototype_bias(backbone, val_loader, text_protos_mp, device)
        print(f"  Proto bias: {proto_bias}", flush=True)

    # ── Setup evaluators ──
    num_classes = cfg["model"]["num_classes"]
    overlap_thr = cfg["inference"]["overlap_thr"]
    min_points = cfg["inference"].get("min_points", 50)

    evaluators = {}
    if "linear" in modes:
        evaluators["linear"] = PanopticEval(n_classes=num_classes, min_points=min_points)
    if "token" in modes:
        evaluators["token"] = PanopticEval(n_classes=num_classes, min_points=min_points)
    if "backbone_pool_v2" in modes:
        evaluators["backbone_pool_v2_known"] = PanopticEval(n_classes=num_classes, min_points=min_points)
    if "geometric_ensemble" in modes:
        for ga in geo_alphas:
            evaluators[f"geo_ensemble_a{ga:.1f}"] = PanopticEval(n_classes=num_classes, min_points=min_points)
    if "geo_ensemble_v2" in modes:
        for ga in geo_alphas:
            evaluators[f"geo_v2_a{ga:.1f}"] = PanopticEval(n_classes=num_classes, min_points=min_points)

    novel_stats_agg = {
        "backbone_pool_v2": {name: {"voxels": 0} for name in novel_class_names},
        "geometric_ensemble": {name: {"voxels": 0} for name in novel_class_names},
        "geo_ensemble_v2": {name: {"voxels": 0} for name in novel_class_names},
    }

    # ── Evaluation loop ──
    print(f"Evaluating {len(val_ds)} samples across modes: {modes}\n", flush=True)
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            feats, coords = extract_backbone_features(backbone, batch, device)
            outputs = decoder(feats, coords)

            pred_logits = outputs["pred_logits"][0]   # (Q, C+1)
            pred_masks = outputs["pred_masks"][0]     # (V, Q)
            pred_tokens = outputs.get("pred_tokens", [None])[0]  # (Q, 768) or None

            gt_sem = batch["gt_sem"].numpy()
            gt_masks_arr = batch["gt_masks"].numpy()
            gt_classes = batch["gt_classes"].numpy()
            V = gt_sem.shape[0]
            gt_inst = np.zeros(V, dtype=np.int64)
            for m_idx in range(len(gt_classes)):
                gt_inst[gt_masks_arr[m_idx]] = m_idx + 1
            valid = gt_sem != 255
            if valid.sum() == 0:
                continue

            # Move to CPU for inference
            pl = pred_logits.cpu()
            pm = pred_masks.cpu()
            pt = pred_tokens.cpu() if pred_tokens is not None else None

            # A: Linear
            if "linear" in modes:
                sem_a, inst_a = panoptic_inference_linear(pl, pm, num_classes, overlap_thr)
                evaluators["linear"].addBatch(sem_a[valid], inst_a[valid], gt_sem[valid], gt_inst[valid])

            # B: Token-16
            if "token" in modes and pt is not None:
                sem_b, inst_b = panoptic_inference_token(pt, pm, text_protos_16.cpu(), overlap_thr)
                evaluators["token"].addBatch(sem_b[valid], inst_b[valid], gt_sem[valid], gt_inst[valid])

            # F: Backbone-Pool v2 (fixed) — normalized soft-pool
            if "backbone_pool_v2" in modes:
                backbone_feats_v = feats[0]  # (V, 768)
                sem_f, inst_f = panoptic_inference_backbone_pool_v2(
                    backbone_feats_v, pm.to(device), text_protos_22, overlap_thr, min_points
                )
                stats_f = count_novel_detections(sem_f, num_classes, novel_class_names)
                for name, s in stats_f.items():
                    novel_stats_agg["backbone_pool_v2"][name]["voxels"] += s["voxels"]
                sem_f_known, inst_f_known = remap_novel_to_ignore(sem_f, inst_f, num_classes)
                evaluators["backbone_pool_v2_known"].addBatch(
                    sem_f_known[valid], inst_f_known[valid], gt_sem[valid], gt_inst[valid]
                )

            # G: Geometric Ensemble (Strategy E) — alpha sweep
            if "geometric_ensemble" in modes:
                backbone_feats_v = feats[0]  # (V, 768)
                for ga in geo_alphas:
                    sem_g, inst_g = panoptic_inference_geometric_ensemble(
                        pl.to(device), pm.to(device), backbone_feats_v, text_protos_22,
                        num_known=num_classes, overlap_thr=overlap_thr,
                        alpha=ga, temperature=args.geo_temperature, min_mask_pts=min_points,
                    )
                    # Track novel for best alpha (use first alpha for aggregation)
                    if ga == geo_alphas[0]:
                        stats_g = count_novel_detections(sem_g, num_classes, novel_class_names)
                        for name, s in stats_g.items():
                            novel_stats_agg["geometric_ensemble"][name]["voxels"] += s["voxels"]
                    sem_g_known, inst_g_known = remap_novel_to_ignore(sem_g, inst_g, num_classes)
                    evaluators[f"geo_ensemble_a{ga:.1f}"].addBatch(
                        sem_g_known[valid], inst_g_known[valid], gt_sem[valid], gt_inst[valid]
                    )

            # H: Geometric Ensemble v2 (multi-prompt + per-class temperature)
            if "geo_ensemble_v2" in modes:
                backbone_feats_v = feats[0]
                for ga in geo_alphas:
                    sem_h, inst_h = panoptic_inference_geometric_ensemble_v2(
                        pl.to(device), pm.to(device), backbone_feats_v,
                        text_protos_mp, proto_bias,
                        num_known=num_classes, overlap_thr=overlap_thr,
                        alpha=ga, temperature=args.geo_temperature, min_mask_pts=min_points,
                    )
                    if ga == geo_alphas[0]:
                        stats_h = count_novel_detections(sem_h, num_classes, novel_class_names)
                        for name, s in stats_h.items():
                            novel_stats_agg["geo_ensemble_v2"][name]["voxels"] += s["voxels"]
                    sem_h_known, inst_h_known = remap_novel_to_ignore(sem_h, inst_h, num_classes)
                    evaluators[f"geo_v2_a{ga:.1f}"].addBatch(
                        sem_h_known[valid], inst_h_known[valid], gt_sem[valid], gt_inst[valid]
                    )

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(val_ds)}]", flush=True)

    # ── Collect results ──
    print(f"\n{'='*70}", flush=True)
    print(f"  RESULTS", flush=True)
    print(f"{'='*70}\n", flush=True)

    results = {}
    summary_rows = []

    for name, ev in evaluators.items():
        PQ, SQ, RQ, pq_all, sq_all, rq_all = ev.getPQ()
        iou_mean, iou_all = ev.getSemIoU()
        results[name] = {
            "PQ": float(PQ), "SQ": float(SQ), "RQ": float(RQ), "mIoU": float(iou_mean),
            "per_class_pq": [float(x) for x in pq_all],
            "per_class_sq": [float(x) for x in sq_all],
            "per_class_rq": [float(x) for x in rq_all],
            "per_class_iou": [float(x) for x in iou_all],
        }

        # Determine novel voxels for summary
        novel_total = "N/A"
        # Match evaluator key to novel_stats_agg key
        mode_key = name.replace("_known", "")
        for agg_key in novel_stats_agg:
            if mode_key.startswith(agg_key) or mode_key.startswith(agg_key.replace("_ensemble", "")):
                novel_total = sum(s["voxels"] for s in novel_stats_agg[agg_key].values())
                break

        summary_rows.append((name, PQ, SQ, RQ, iou_mean, novel_total))

        # Print per-class table
        print(f"  Mode: {name}", flush=True)
        print(f"  PQ={PQ:.4f}  SQ={SQ:.4f}  RQ={RQ:.4f}  mIoU={iou_mean:.4f}", flush=True)
        table = format_table(pq_all, sq_all, rq_all, iou_all, NUSCENES_CLASS_NAMES)
        print(table, flush=True)
        print(flush=True)

        # Save per-class table file
        table_name = name.replace(".", "_")
        with open(out_dir / f"per_class_{table_name}.txt", "w") as f:
            f.write(f"Mode: {name}\n")
            f.write(f"PQ={PQ:.4f}  SQ={SQ:.4f}  RQ={RQ:.4f}  mIoU={iou_mean:.4f}\n\n")
            f.write(table + "\n")

    # ── Summary table ──
    print(f"\n{'='*70}", flush=True)
    print(f"  SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  {'Mode':<30s} {'PQ':>8s} {'SQ':>8s} {'RQ':>8s} {'mIoU':>8s} {'Novel Vox':>12s}", flush=True)
    print(f"  {'-'*74}", flush=True)
    for name, pq, sq, rq, iou, novel in summary_rows:
        nv_str = f"{novel:>12,d}" if isinstance(novel, int) else f"{'N/A':>12s}"
        print(f"  {name:<30s} {pq:8.4f} {sq:8.4f} {rq:8.4f} {iou:8.4f} {nv_str}", flush=True)

    # Find best geometric ensemble alpha
    if "geometric_ensemble" in modes:
        geo_keys = [k for k in results if k.startswith("geo_ensemble_a")]
        if geo_keys:
            best_geo = max(geo_keys, key=lambda k: results[k]["PQ"])
            print(f"  Best geo ensemble: {best_geo} (PQ={results[best_geo]['PQ']:.4f})", flush=True)
            results["best_geo_ensemble"] = {"name": best_geo, **results[best_geo]}

    # Find best geo ensemble v2 alpha
    if "geo_ensemble_v2" in modes:
        geo_v2_keys = [k for k in results if k.startswith("geo_v2_a")]
        if geo_v2_keys:
            best_geo_v2 = max(geo_v2_keys, key=lambda k: results[k]["PQ"])
            print(f"  Best geo v2 (multi-prompt): {best_geo_v2} (PQ={results[best_geo_v2]['PQ']:.4f})", flush=True)
            results["best_geo_v2"] = {"name": best_geo_v2, **results[best_geo_v2]}

    # ── Novel detection comparison ──
    openvocab_modes_active = [m for m in ["backbone_pool_v2", "geometric_ensemble",
                                           "geo_ensemble_v2"] if m in modes]
    if openvocab_modes_active:
        print(f"\n{'='*70}", flush=True)
        print(f"  NOVEL CLASS DETECTIONS", flush=True)
        print(f"{'='*70}", flush=True)
        header_modes = [m[:12] for m in openvocab_modes_active]
        print(f"  {'Novel Class':<20s} " + " ".join(f"{h:>14s}" for h in header_modes), flush=True)
        print(f"  {'-'*70}", flush=True)
        for name in novel_class_names:
            vals = []
            for m in openvocab_modes_active:
                v = novel_stats_agg.get(m, {}).get(name, {}).get("voxels", 0)
                vals.append(v)
            print(f"  {name:<20s} " + " ".join(f"{v:>14,d}" for v in vals), flush=True)

    # ── Save results ──
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results.json to {out_dir}", flush=True)

    # Save novel detections
    if openvocab_modes_active:
        with open(out_dir / "novel_detections.json", "w") as f:
            json.dump(novel_stats_agg, f, indent=2)
        print(f"Saved novel_detections.json to {out_dir}", flush=True)

    print(f"\nDone. All outputs in: {out_dir}", flush=True)

    # Restore stdout
    sys.stdout = tee.terminal
    tee.close()


if __name__ == "__main__":
    main()
