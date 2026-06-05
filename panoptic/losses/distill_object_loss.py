"""L_O: Object-level CLIP distillation loss.

For each matched query, align its predicted CLIP embedding with the
precomputed mask-pooled camera CLIP feature of the GT instance.

This provides open-vocabulary signal: camera CLIP sees ALL classes
(including novel), so the decoder learns to produce embeddings that
align with novel-class CLIP features even without class labels.
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
            pred_embed: (B, Q, 768) predicted query CLIP embeddings
            obj_clip_feats: (B, N_inst, 768) precomputed mask-pooled CLIP per GT instance
            obj_clip_classes: (B, N_inst) GT class per instance (for matching)
            indices: list of (pred_idx, gt_idx) tuples from Hungarian matching
            targets: dict with "classes" key for GT labels

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

            # Get matched query embeddings
            matched_pred = pred_embed[b, pred_idx]  # (M, 768)

            # For each matched GT instance, find its corresponding precomputed CLIP feature
            # gt_idx indexes into the targets["classes"] list for this batch
            # We need to map gt_idx to obj_clip_feats indices
            gt_classes_b = targets["classes"][b]  # (M_gt,)
            obj_classes_b = obj_clip_classes[b]   # (N_inst,)
            obj_feats_b = obj_clip_feats[b]       # (N_inst, 768)

            for i, gi in enumerate(gt_idx):
                gt_cls = gt_classes_b[gi].item()

                # Find matching instance in precomputed features
                # Match by class and order (precomputed follows same GT build order)
                if gi.item() < obj_feats_b.shape[0]:
                    target_feat = obj_feats_b[gi.item()]  # (768,)
                else:
                    continue

                # Skip if target is zero (no valid CLIP projection for this instance)
                if target_feat.norm() < 0.1:
                    continue

                pred_feat = matched_pred[i]  # (768,)

                if self.loss_type == "cosine":
                    loss_i = 1.0 - F.cosine_similarity(
                        pred_feat.unsqueeze(0), target_feat.unsqueeze(0)
                    )
                else:
                    loss_i = F.smooth_l1_loss(pred_feat, target_feat)

                total_loss = total_loss + loss_i.squeeze()
                n_valid += 1

        if n_valid > 0:
            total_loss = total_loss / n_valid

        return total_loss
