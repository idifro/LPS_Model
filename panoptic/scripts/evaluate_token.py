"""Evaluate panoptic model with CLIP token head (Phase 5).

Reports PQ/SQ/RQ/mIoU for three classification modes:
  (A) Linear head only (existing class_embed)
  (B) Token-based (cosine similarity to text prototypes)
  (C) Ensemble (weighted sum of linear logits + cosine similarities)

Usage:
    python -u panoptic/scripts/evaluate_token.py --config panoptic/configs/mini_token.yaml \
        --checkpoint panoptic/checkpoints_token/best.pth
"""
import argparse
import sys
import os
import json

import numpy as np
import torch
import MinkowskiEngine as ME
import yaml

sys.path.insert(0, "/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene")

from models.disnet import DisNet, state_dict_remove_moudle
from panoptic.models.mask_decoder import PanopticDecoder
from panoptic.eval.eval_pq import PanopticEval
from panoptic.dataset.nuscenes_panoptic import (
    NuScenesPanoptic, NUM_CLASSES, NUSCENES_CLASS_NAMES,
)


def load_config(config_path):
    """Load a YAML configuration file and return it as a dictionary.

    Used at the start of evaluation to read model, data, and inference
    hyperparameters from disk.

    Steps:
        1. Opens the file at config_path in read mode.
        2. Parses the YAML content using yaml.safe_load.
        3. Returns the resulting dictionary.

    Args:
        config_path (str): Absolute or relative path to the YAML config file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_backbone(ckpt_path, device):
    """Load the DisNet 3D backbone from a checkpoint and freeze all parameters.

    Used in the evaluation pipeline to provide per-voxel features that feed
    into the panoptic decoder.

    Steps:
        1. Instantiates a DisNet model with openseg/MinkUNet18A config.
        2. Loads and cleans the state dict from the checkpoint.
        3. Sets the model to eval mode and disables gradients.

    Args:
        ckpt_path (str): Path to the backbone .pth.tar checkpoint.
        device (torch.device): Target device (cpu or cuda).

    Returns:
        DisNet: The loaded and frozen backbone model in eval mode.
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

    Called per-batch during evaluation to convert raw voxel coordinates and
    features into the dense feature representation consumed by the decoder.

    Steps:
        1. Prepends a batch-index column of zeros to voxel coordinates.
        2. Constructs a MinkowskiEngine SparseTensor on the target device.
        3. Runs the backbone forward pass (no grad).
        4. Returns features and coordinates each with a batch dimension.

    Args:
        backbone (DisNet): The frozen 3D backbone model.
        batch (dict): Dataset sample dict containing 'vox_coords' and 'vox_feats'.
        device (torch.device): Target device for computation.

    Returns:
        tuple: (feats, coords_xyz) where feats is (1, V, D) backbone output
            and coords_xyz is (1, V, 3) spatial coordinates.
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
    coords_xyz = vox_feats.to(device)
    return feats.unsqueeze(0), coords_xyz.unsqueeze(0)


def panoptic_inference_linear(pred_logits, pred_masks, num_classes=16, overlap_thr=0.8):
    """Perform panoptic inference using the linear classification head.

    Uses softmax over the predicted logits to determine query class scores,
    then applies greedy mask assignment to produce per-voxel semantic and
    instance labels.

    Steps:
        1. Applies softmax to pred_logits and takes argmax for class labels.
        2. Sigmoids the predicted masks to get per-voxel probabilities.
        3. Filters out queries assigned to the no-object class.
        4. Delegates to _greedy_assign for panoptic label assignment.

    Args:
        pred_logits (Tensor): Query classification logits of shape (Q, C+1).
        pred_masks (Tensor): Per-voxel mask logits of shape (V, Q).
        num_classes (int): Number of valid semantic classes (default 16).
        overlap_thr (float): Minimum overlap ratio to accept a mask.

    Returns:
        tuple: (sem_pred, inst_pred) numpy arrays of shape (V,) with
            semantic class IDs and instance IDs respectively.
    """
    scores, labels = pred_logits.softmax(-1).max(-1)
    masks = pred_masks.sigmoid()
    V = masks.shape[0]
    keep = labels != num_classes
    if keep.sum() == 0:
        return np.zeros(V, dtype=np.int64), np.zeros(V, dtype=np.int64)
    return _greedy_assign(scores[keep], labels[keep], masks[:, keep], V, overlap_thr)


def panoptic_inference_token(pred_tokens, pred_masks, text_prototypes, num_classes=16, overlap_thr=0.8):
    """Perform panoptic inference using token cosine similarity to text prototypes.

    Classifies each query by computing cosine similarity between its predicted
    token embedding and the CLIP text prototypes, then applies greedy assignment.

    Steps:
        1. Computes cosine similarity matrix between tokens and prototypes.
        2. Takes max similarity as score and argmax as class label per query.
        3. Filters queries with score below 0.1 threshold.
        4. Delegates to _greedy_assign for panoptic label assignment.

    Args:
        pred_tokens (Tensor): Query token embeddings of shape (Q, D).
        pred_masks (Tensor): Per-voxel mask logits of shape (V, Q).
        text_prototypes (Tensor): CLIP text prototype embeddings of shape (C, D).
        num_classes (int): Number of semantic classes (default 16).
        overlap_thr (float): Minimum overlap ratio to accept a mask.

    Returns:
        tuple: (sem_pred, inst_pred) numpy arrays of shape (V,).
    """
    cos_sim = torch.mm(pred_tokens, text_prototypes.t())  # (Q, C)
    scores, labels = cos_sim.max(-1)
    masks = pred_masks.sigmoid()
    V = masks.shape[0]
    keep = scores > 0.1
    if keep.sum() == 0:
        return np.zeros(V, dtype=np.int64), np.zeros(V, dtype=np.int64)
    return _greedy_assign(scores[keep], labels[keep], masks[:, keep], V, overlap_thr)


def panoptic_inference_ensemble(pred_logits, pred_tokens, pred_masks, text_prototypes,
                                 num_classes=16, overlap_thr=0.8, alpha=0.5):
    """Ensemble: alpha * linear_softmax + (1-alpha) * token_cosine."""
    linear_probs = pred_logits[:, :num_classes].softmax(-1)  # (Q, C) exclude no-obj
    cos_sim = torch.mm(pred_tokens, text_prototypes.t())  # (Q, C)
    # Normalize cosine similarities to [0,1] per query
    cos_sim_norm = (cos_sim + 1) / 2
    combined = alpha * linear_probs + (1 - alpha) * cos_sim_norm
    scores, labels = combined.max(-1)
    masks = pred_masks.sigmoid()
    V = masks.shape[0]
    keep = scores > 0.1
    if keep.sum() == 0:
        return np.zeros(V, dtype=np.int64), np.zeros(V, dtype=np.int64)
    return _greedy_assign(scores[keep], labels[keep], masks[:, keep], V, overlap_thr)


def _greedy_assign(scores, labels, masks, V, overlap_thr):
    """Greedily assign panoptic labels to voxels using sorted mask confidence.

    Processes queries in descending score order, assigning each mask's voxels
    to the corresponding semantic/instance label only if sufficient unoccupied
    overlap remains. Thing classes get unique instance IDs; stuff classes share
    a single instance ID per class.

    Steps:
        1. Sorts queries by score in descending order.
        2. For each query, binarizes the mask at 0.5 threshold.
        3. Removes already-assigned voxels and checks overlap ratio.
        4. Assigns semantic and instance labels based on thing/stuff distinction.

    Args:
        scores (Tensor): Confidence scores of shape (K,) for kept queries.
        labels (Tensor): Class labels of shape (K,) for kept queries.
        masks (Tensor): Sigmoid mask probabilities of shape (V, K).
        V (int): Total number of voxels.
        overlap_thr (float): Minimum fraction of original mask area that
            must remain unoccupied for the query to be accepted.

    Returns:
        tuple: (sem_pred, inst_pred) numpy arrays of shape (V,) with int64 dtype.
    """
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


def main():
    """Run panoptic evaluation across linear, token, and ensemble modes.

    This is the top-level entry point for the Phase 5 token head evaluation.
    It loads the model with token head support, runs inference on the val set,
    and reports PQ/SQ/RQ/mIoU for each classification mode.

    Steps:
        1. Parses CLI arguments and loads config/checkpoint.
        2. Initializes the dataset, backbone, decoder, and text prototypes.
        3. Creates separate PanopticEval evaluators for each mode.
        4. Iterates over validation samples, running all three inference paths.
        5. Prints per-class results tables and optionally saves to JSON.

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--ensemble_alpha", type=float, default=0.5)
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    dcfg = cfg["data"]
    val_ds = NuScenesPanoptic(
        dcfg["dataroot"], dcfg["version"], split="val",
        voxel_size=dcfg["voxel_size"],
        point_range=np.array(dcfg["point_range"]),
        augment=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x[0],
    )
    print(f"Val: {len(val_ds)} samples", flush=True)

    # Load models
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

    # Load checkpoint
    ckpt_path = args.checkpoint or os.path.join(cfg["checkpoint"]["save_dir"], "best.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    decoder.load_state_dict(ckpt["decoder"], strict=True)
    decoder.eval()
    print(f"Loaded checkpoint: {ckpt_path}", flush=True)

    # Load text prototypes
    token_cfg = cfg.get("token", {})
    proto_path = token_cfg.get("prototype_path", "panoptic/clip_prototypes/text_prototypes_vitl14_336.pt")
    proto_data = torch.load(proto_path, map_location="cpu")
    text_prototypes = proto_data["embeddings"].float().to(device)
    print(f"Loaded prototypes: {text_prototypes.shape}", flush=True)

    num_classes = mcfg["num_classes"]
    overlap_thr = cfg["inference"]["overlap_thr"]
    min_points = cfg["inference"]["min_points"]

    evaluator_linear = PanopticEval(n_classes=num_classes, min_points=min_points)
    evaluator_token = PanopticEval(n_classes=num_classes, min_points=min_points)
    evaluator_ensemble = PanopticEval(n_classes=num_classes, min_points=min_points)

    print(f"\nEvaluating ({len(val_ds)} samples)...", flush=True)
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            feats, coords = extract_backbone_features(backbone, batch, device)
            outputs = decoder(feats, coords)

            pred_logits = outputs["pred_logits"][0].cpu()
            pred_masks = outputs["pred_masks"][0].cpu()
            pred_tokens = outputs.get("pred_tokens", [None])[0]
            if pred_tokens is not None:
                pred_tokens = pred_tokens.cpu()

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

            # (A) Linear
            sem_a, inst_a = panoptic_inference_linear(pred_logits, pred_masks, num_classes, overlap_thr)
            evaluator_linear.addBatch(sem_a[valid], inst_a[valid], gt_sem[valid], gt_inst[valid])

            # (B) Token
            if pred_tokens is not None:
                sem_b, inst_b = panoptic_inference_token(pred_tokens, pred_masks, text_prototypes.cpu(), num_classes, overlap_thr)
                evaluator_token.addBatch(sem_b[valid], inst_b[valid], gt_sem[valid], gt_inst[valid])

                # (C) Ensemble
                sem_c, inst_c = panoptic_inference_ensemble(
                    pred_logits, pred_tokens, pred_masks, text_prototypes.cpu(),
                    num_classes, overlap_thr, alpha=args.ensemble_alpha,
                )
                evaluator_ensemble.addBatch(sem_c[valid], inst_c[valid], gt_sem[valid], gt_inst[valid])

            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(val_ds)}", flush=True)

    # ── Results ──
    results = {}
    for name, ev in [("linear", evaluator_linear), ("token", evaluator_token), ("ensemble", evaluator_ensemble)]:
        PQ, SQ, RQ, pq_all, sq_all, rq_all = ev.getPQ()
        iou_mean, iou_all = ev.getSemIoU()
        results[name] = {"PQ": PQ, "SQ": SQ, "RQ": RQ, "mIoU": iou_mean}

        print(f"\n{'='*60}", flush=True)
        print(f"  Mode: {name.upper()}", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"  PQ={PQ:.4f}  SQ={SQ:.4f}  RQ={RQ:.4f}  mIoU={iou_mean:.4f}", flush=True)
        print(f"\n  {'Class':<25s} {'PQ':>8s} {'SQ':>8s} {'RQ':>8s} {'IoU':>8s}", flush=True)
        print(f"  {'-'*57}", flush=True)
        for c in range(num_classes):
            print(f"  {NUSCENES_CLASS_NAMES[c]:<25s} {pq_all[c]:8.4f} {sq_all[c]:8.4f} "
                  f"{rq_all[c]:8.4f} {iou_all[c]:8.4f}", flush=True)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
