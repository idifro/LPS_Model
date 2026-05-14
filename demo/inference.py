"""Inference pipeline: backbone features → decoder → panoptic predictions.

Functions adapted from panoptic/scripts/evaluate_token.py.
The public entry point is run_full_inference().
"""

import numpy as np
import torch
import MinkowskiEngine as ME


# ── Backbone feature extraction ───────────────────────────────────────────────

def extract_backbone_features(backbone, batch, device):
    """Build a MinkowskiEngine SparseTensor and run the frozen DisNet backbone.

    Args:
        backbone: frozen DisNet model (on device)
        batch:    dict with:
                    vox_coords (V, 3) int32   — voxel grid coordinates
                    vox_feats  (V, 3) float32 — raw XYZ (LIDAR frame)
        device:   torch.device

    Returns:
        feats:  (1, V, 768) backbone feature tensor on device
        coords: (1, V, 3)   xyz coordinate tensor on device (for decoder PE)
    """
    vox_coords = batch["vox_coords"]   # (V, 3) int32
    vox_feats  = batch["vox_feats"]    # (V, 3) float32
    V = vox_coords.shape[0]

    # Prepend batch index column (all zeros — single sample)
    batch_coords = torch.cat(
        [torch.zeros(V, 1, dtype=torch.int32), vox_coords], dim=1
    )  # (V, 4)

    sinput = ME.SparseTensor(
        vox_feats.float(),
        coordinates=batch_coords.int(),
        device=device,
    )

    with torch.no_grad():
        output = backbone(sinput)

    feats = output.F if hasattr(output, "F") else output   # (V, 768)
    coords_xyz = vox_feats.to(device)                       # (V, 3)

    return feats.unsqueeze(0), coords_xyz.unsqueeze(0)      # (1, V, 768), (1, V, 3)


# ── Decoder forward ───────────────────────────────────────────────────────────

def run_decoder(decoder, feats, coords):
    """Run PanopticDecoder in no_grad mode.

    Returns:
        dict with at minimum:
          pred_logits (1, Q, 17)
          pred_masks  (1, V, Q)
          pred_tokens (1, Q, 768)   — when use_token_head=True
    """
    with torch.no_grad():
        return decoder(feats, coords)


# ── Greedy panoptic mask assignment ──────────────────────────────────────────

def _greedy_assign(scores, labels, masks, V, overlap_thr):
    """Assign panoptic IDs greedily from highest-scoring query to lowest.

    Adapted from evaluate_token.py / MaskPLS.

    Args:
        scores:     (K,)    confidence score per kept query
        labels:     (K,)    predicted class index (0-indexed) per kept query
        masks:      (V, K)  sigmoid mask probs
        V:          number of voxels
        overlap_thr: minimum fraction of unoccluded area to assign a mask

    Returns:
        sem_pred  (V,) int64 — 0-indexed class label (0 = background/unassigned)
        inst_pred (V,) int64 — instance ID (0 = background)
    """
    sorted_idx = scores.argsort(descending=True)
    sem_pred  = np.zeros(V, dtype=np.int64)
    inst_pred = np.zeros(V, dtype=np.int64)
    inst_id   = 1
    thing_set = set(range(10))     # classes 0-9 are things (0-indexed)
    stuff_assigned = {}

    for idx in sorted_idx:
        label     = labels[idx].item()
        mask_prob = masks[:, idx]
        binary    = mask_prob >= 0.5

        if binary.sum() == 0:
            continue

        orig_area  = binary.sum().item()
        assigned   = torch.from_numpy(inst_pred).to(mask_prob.device) > 0
        final_mask = binary & ~assigned

        if final_mask.sum().item() / max(orig_area, 1) < overlap_thr:
            continue

        mask_np = final_mask.cpu().numpy().astype(bool)

        if label in thing_set:
            sem_pred[mask_np]  = label
            inst_pred[mask_np] = inst_id
            inst_id += 1
        else:
            if label in stuff_assigned:
                sem_pred[mask_np]  = label
                inst_pred[mask_np] = stuff_assigned[label]
            else:
                sem_pred[mask_np]  = label
                inst_pred[mask_np] = inst_id
                stuff_assigned[label] = inst_id
                inst_id += 1

    return sem_pred, inst_pred


# ── Ensemble panoptic inference ───────────────────────────────────────────────

def panoptic_inference_ensemble(
    pred_logits, pred_tokens, pred_masks, text_prototypes,
    num_classes=16, overlap_thr=0.8, alpha=0.5,
):
    """Ensemble: alpha * linear_softmax + (1-alpha) * token_cosine.

    Args:
        pred_logits:     (Q, 17)   — class logits incl. no-object (CPU tensor)
        pred_tokens:     (Q, 768)  — L2-normalised token embeddings (CPU tensor)
        pred_masks:      (V, Q)    — raw mask logits (CPU tensor)
        text_prototypes: (16, 768) — prototype embeddings (CPU tensor)
        num_classes:     16
        overlap_thr:     greedy-assign overlap threshold
        alpha:           weight for linear head (0.5 = equal blend)

    Returns:
        sem_pred  (V,) int64
        inst_pred (V,) int64
    """
    linear_probs  = pred_logits[:, :num_classes].softmax(-1)          # (Q, 16)
    cos_sim       = torch.mm(pred_tokens, text_prototypes.t())         # (Q, 16)
    cos_sim_norm  = (cos_sim + 1.0) / 2.0                             # map [-1,1]→[0,1]

    combined      = alpha * linear_probs + (1.0 - alpha) * cos_sim_norm
    scores, labels = combined.max(-1)

    masks = pred_masks.sigmoid()   # (V, Q)
    V     = masks.shape[0]
    keep  = scores > 0.1

    if keep.sum() == 0:
        return np.zeros(V, dtype=np.int64), np.zeros(V, dtype=np.int64)

    return _greedy_assign(
        scores[keep], labels[keep], masks[:, keep], V, overlap_thr
    )


# ── Public entry point ────────────────────────────────────────────────────────

def run_full_inference(backbone, decoder, batch, text_protos, device,
                       alpha=0.5, overlap_thr=0.8):
    """Run the complete one-sample inference pipeline.

    Steps:
      1. Build ME SparseTensor → backbone features
      2. Decoder forward → pred_logits, pred_masks, pred_tokens
      3. Ensemble panoptic inference → sem_pred, inst_pred

    Args:
        backbone:    frozen DisNet on device
        decoder:     PanopticDecoder on device
        batch:       dict from data_utils.load_sample_for_inference
        text_protos: (16, 768) prototype tensor on device
        device:      torch.device
        alpha:       ensemble blend weight
        overlap_thr: greedy assign threshold

    Returns:
        sem_pred:    (V,) int64 np.ndarray  — 0-indexed class per voxel
        inst_pred:   (V,) int64 np.ndarray  — instance ID per voxel
        pred_masks:  (V, Q) float32 CPU tensor — raw mask logits (for text query)
        pred_tokens: (Q, 768) float32 CPU tensor — CLIP query tokens (for text query)
        pred_logits: (Q, 17) float32 CPU tensor  — class logits
    """
    feats, coords = extract_backbone_features(backbone, batch, device)
    outputs       = run_decoder(decoder, feats, coords)

    pred_logits  = outputs["pred_logits"][0].cpu()   # (Q, 17)
    pred_masks   = outputs["pred_masks"][0].cpu()    # (V, Q)
    pred_tokens  = outputs["pred_tokens"][0].cpu()   # (Q, 768)

    text_protos_cpu = text_protos.cpu()

    sem_pred, inst_pred = panoptic_inference_ensemble(
        pred_logits, pred_tokens, pred_masks, text_protos_cpu,
        num_classes=16, overlap_thr=overlap_thr, alpha=alpha,
    )

    return sem_pred, inst_pred, pred_masks, pred_tokens, pred_logits
