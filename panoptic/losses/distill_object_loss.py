"""L_O: Object-level CLIP distillation loss.

For each matched query, align its predicted CLIP embedding with the
precomputed mask-pooled camera CLIP feature of the GT instance.

Indexing note: the precomputed cache (obj_feats, obj_classes) spans ALL GT
instances including novel and skips those with no camera-visible voxels.
The Hungarian matching indices (gt_idx) reference the BASE-FILTERED target
list, NOT the full GT list.  We resolve this by looking up obj_clip_feats by
(class, occurrence-order) instead of by raw index.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectDistillLoss(nn.Module):
    """Object-level distillation: pred_embed → camera CLIP pooled per instance."""

    def __init__(self, loss_type="cosine"):
        """
        Args:
            loss_type: "cosine" (1 - cos_sim) or "smooth_l1"
        """
        super().__init__()
        self.loss_type = loss_type

    def forward(self, pred_embed, obj_clip_feats, obj_clip_classes, indices, targets):
        """
        Args:
            pred_embed:      (B, Q, 768) predicted query CLIP embeddings
            obj_clip_feats:  (B, N_inst, 768) precomputed mask-pooled CLIP per GT instance
                             N_inst = all GT instances with ≥1 camera-visible voxel (incl. novel)
            obj_clip_classes:(B, N_inst) GT class per instance (0-indexed, matches cache order)
            indices:         list of (pred_idx, gt_idx) from Hungarian matching
                             gt_idx references the BASE-FILTERED target list
            targets:         dict with "classes" key → base-filtered GT classes per batch item

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

            matched_pred  = pred_embed[b, pred_idx]   # (M, 768)
            gt_classes_b  = targets["classes"][b]      # (M_base,) base-filtered classes
            obj_classes_b = obj_clip_classes[b]        # (N_inst,) cache classes (all GT)
            obj_feats_b   = obj_clip_feats[b]          # (N_inst, 768)

            # Build per-class FIFO queue of cache indices so multiple instances of the
            # same class are consumed in order, matching the base-filtered target order.
            from collections import defaultdict
            class_queue: dict = defaultdict(list)
            for ci, cls_val in enumerate(obj_classes_b.tolist()):
                class_queue[int(cls_val)].append(ci)
            # Track how many times each class has been consumed so far this batch
            class_consumed: dict = defaultdict(int)

            for i, gi in enumerate(gt_idx):
                gt_cls = int(gt_classes_b[gi].item())

                queue      = class_queue.get(gt_cls, [])
                consumed   = class_consumed[gt_cls]
                if consumed >= len(queue):
                    # No more cache entries for this class
                    continue

                cache_idx   = queue[consumed]
                class_consumed[gt_cls] += 1
                target_feat = obj_feats_b[cache_idx]   # (768,)

                # Skip if target is zero (no valid CLIP projection)
                if target_feat.norm() < 0.1:
                    continue

                pred_feat = matched_pred[i]   # (768,)

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
