"""L_O: Object-level CLIP distillation loss.

For each matched query, align its predicted CLIP embedding with the
precomputed crop-CLS CLIP feature of the GT instance.

Indexing: the precomputed cache (obj_feats, obj_classes, obj_valid) is built
by process_sample_obj() in the same GT order as NuScenesPanoptic.__getitem__
(THING_IDS 1-10 first, then STUFF_IDS 11-16).  indices_all gt_idx values are
therefore valid direct indices into the cache — no class-queue needed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Stuff class crop-CLS features are too noisy (random 224×224 patches capture
# global scene context, not the target class). Only apply L_O to THING instances.
_THING_CLASS_IDS = frozenset(range(10))  # 0-indexed classes 0–9


class ObjectDistillLoss(nn.Module):
    """Object-level distillation: pred_embed → crop-CLS CLIP per GT instance."""

    def __init__(self, loss_type="cosine"):
        """
        Args:
            loss_type: "cosine" (1 - cos_sim) or "smooth_l1"
        """
        super().__init__()
        self.loss_type = loss_type

    def forward(self, pred_embed, obj_clip_feats, obj_clip_classes,
                obj_clip_valid, indices, targets):
        """
        Args:
            pred_embed:       (B, Q, 768) predicted query CLIP embeddings
            obj_clip_feats:   (B, N_inst, 768) precomputed crop-CLS per GT instance
                              N_inst = all GT instances (incl. novel), same order as GT masks
            obj_clip_classes: (B, N_inst) GT class per instance (0-indexed, matches cache)
            obj_clip_valid:   (B, N_inst) bool — True only for instances with valid crop-CLS
            indices:          list of (pred_idx, gt_idx) from all-GT Hungarian matching
                              gt_idx directly indexes into obj_clip_feats/obj_clip_valid
            targets:          dict with "classes" key → all GT classes per batch item

        Returns:
            loss: scalar tensor
        """
        device = pred_embed.device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        n_valid = 0

        B = pred_embed.shape[0]
        for b in range(B):
            pred_idx, gt_idx = indices[b] if isinstance(indices, list) else indices

            if len(pred_idx) == 0:
                continue

            gt_classes_b  = targets["classes"][b]   # (N_gt_all,) all GT classes
            obj_feats_b   = obj_clip_feats[b]        # (N_inst, 768)
            obj_valid_b   = obj_clip_valid[b]        # (N_inst,) bool
            obj_classes_b = obj_clip_classes[b]      # (N_inst,) for consistency check

            for i, gi in enumerate(gt_idx):
                gi_i = int(gi.item())
                gt_cls = int(gt_classes_b[gi_i].item())

                # Only apply to THING instances — stuff crop-CLS captures global context
                if gt_cls not in _THING_CLASS_IDS:
                    continue

                # Bounds check (cache and GT must have same length)
                if gi_i >= obj_feats_b.shape[0]:
                    continue

                # Use the explicit valid flag from the precompute pipeline
                if not obj_valid_b[gi_i]:
                    continue

                # Temporary class-consistency assertion: catches GT-order mismatch
                # between build_gt_masks(), all_targets_obj, and the .npz cache.
                # Remove after Phase 8 smoke test validates ordering is correct.
                obj_cls = int(obj_classes_b[gi_i].item())
                if obj_cls != gt_cls:
                    print(f"WARNING L_O class mismatch: gt={gt_cls} obj={obj_cls} "
                          f"gi={gi_i} b={b} — skipping pair")
                    continue

                target_feat = obj_feats_b[gi_i]         # (768,)
                pred_feat   = pred_embed[b, pred_idx[i]] # (768,)

                if self.loss_type == "cosine":
                    loss_i = 1.0 - F.cosine_similarity(
                        pred_feat.unsqueeze(0), target_feat.unsqueeze(0)
                    )
                else:
                    loss_i = F.smooth_l1_loss(pred_feat, target_feat)

                total_loss = total_loss + loss_i.squeeze()
                n_valid   += 1

        if n_valid > 0:
            total_loss = total_loss / n_valid

        return total_loss
