"""Text-prompt encoding and threshold-gated query matching.

Key design decision: matching is THRESHOLD-GATED, not top-k.
A scene with 8 cars should return all 8 matched queries, not just the top 3.

Pipeline for find_matching_queries:
  Step 1 — Inactive query filter:
    mask_area[q] = sigmoid(pred_masks[:, q]).sum()
    Drop any query where mask_area < min_mask_area  (dead / empty slot)
  Step 2 — Cosine similarity (only active queries):
    cos_sim = active_tokens @ text_feat.T
  Step 3 — Threshold gate:
    Return ALL active queries where cos_sim >= threshold
    (sorted descending by score — unbounded count)
"""

import numpy as np
import torch
import torch.nn.functional as F


def encode_text_prompt(clip_model, text, device):
    """Encode a text string with CLIP to a 768-d L2-normalised embedding.

    Uses the same CLIP model (ViT-L/14@336px) that the backbone was distilled
    from, ensuring the embedding lives in the same space as pred_tokens.

    Args:
        clip_model: loaded CLIP model (on device)
        text:       prompt string, e.g. "a car in the scene"
        device:     torch.device

    Returns:
        feat: (1, 768) float32 tensor on device, L2-normalised
    """
    import clip

    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        feat = clip_model.encode_text(tokens).float()
        feat = F.normalize(feat, dim=-1)
    return feat  # (1, 768)


def find_matching_queries(
    text_feat, pred_tokens, pred_masks,
    threshold=0.20, min_mask_area=50,
):
    """Find all decoder queries whose CLIP token matches the text prompt above threshold.

    This is the core text-query API. It is THRESHOLD-GATED — every query that
    passes both the active-query filter and the cosine similarity threshold is
    returned, regardless of count. A scene with 8 cars will return up to 8 queries.

    Args:
        text_feat:     (1, 768) float32  L2-normalised text embedding (any device)
        pred_tokens:   (Q, 768) float32  L2-normalised query tokens (CPU tensor)
        pred_masks:    (V, Q)   float32  raw mask logits (CPU tensor)
        threshold:     float    cosine similarity threshold, e.g. 0.20
        min_mask_area: int      minimum expected voxel count to treat a query as active

    Returns:
        List of tuples (query_idx: int, cos_sim_score: float, binary_mask: np.ndarray)
          - query_idx:    index into the Q=106 queries
          - cos_sim_score: cosine similarity to the text prompt (higher = better match)
          - binary_mask:  (V,) bool np.ndarray — True where sigmoid(pred_mask) >= 0.5
        Sorted descending by cos_sim_score.
        Returns an empty list if no queries pass.
    """
    text_feat_cpu = text_feat.cpu()

    # ── Step 1: Active query filter ─────────────────────────────────────────
    # A "dead" query slot has near-zero mask probability everywhere.
    # sigmoid(logit).sum() gives the expected number of voxels in that mask.
    mask_probs = pred_masks.sigmoid()           # (V, Q)
    mask_area  = mask_probs.sum(dim=0)          # (Q,) — expected voxel count per query

    # nonzero() returns (N, 1); squeeze to (N,) for indexing
    active_indices = (mask_area >= min_mask_area).nonzero(as_tuple=False).squeeze(-1)

    if active_indices.numel() == 0:
        return []

    # ── Step 2: Cosine similarity (active queries only) ─────────────────────
    active_tokens = pred_tokens[active_indices]              # (N_active, 768)
    cos_sim = (active_tokens @ text_feat_cpu.T).squeeze(-1)  # (N_active,)

    # ── Step 3: Threshold gate — ALL passing queries, no count cap ──────────
    pass_local = (cos_sim >= threshold).nonzero(as_tuple=False).squeeze(-1)

    if pass_local.numel() == 0:
        return []

    # Handle scalar case (single match returns 0-d tensor after squeeze)
    if pass_local.dim() == 0:
        pass_local = pass_local.unsqueeze(0)

    matched = []
    for local_i in pass_local.tolist():
        orig_q      = active_indices[local_i].item()
        score       = cos_sim[local_i].item()
        binary_mask = (mask_probs[:, orig_q] >= 0.5).numpy()  # (V,) bool
        matched.append((orig_q, score, binary_mask))

    matched.sort(key=lambda x: x[1], reverse=True)
    return matched


def get_all_active_query_scores(text_feat, pred_tokens, pred_masks, min_mask_area=50):
    """Return cosine similarities for ALL active queries (used in the score chart tab).

    Returns scores for every non-dead query, whether or not they pass threshold.
    This allows the user to understand where the threshold is relative to the
    full distribution of queries.

    Args:
        text_feat:     (1, 768) float32  L2-normalised text embedding (any device)
        pred_tokens:   (Q, 768) float32  CPU tensor
        pred_masks:    (V, Q)   float32  CPU tensor
        min_mask_area: int

    Returns:
        active_q_indices: list of int   — original query indices (Q-space)
        cos_sims:         list of float — cosine similarity for each active query
    """
    text_feat_cpu = text_feat.cpu()
    mask_probs    = pred_masks.sigmoid()
    mask_area     = mask_probs.sum(dim=0)

    active_indices = (mask_area >= min_mask_area).nonzero(as_tuple=False).squeeze(-1)

    if active_indices.numel() == 0:
        return [], []

    if active_indices.dim() == 0:
        active_indices = active_indices.unsqueeze(0)

    active_tokens   = pred_tokens[active_indices]              # (N_active, 768)
    cos_sim         = (active_tokens @ text_feat_cpu.T).squeeze(-1)  # (N_active,)

    return active_indices.tolist(), cos_sim.tolist()
