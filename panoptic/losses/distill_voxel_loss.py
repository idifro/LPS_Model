"""L_V: Voxel-level CLIP distillation loss.

Reconstructs per-voxel embeddings from the decoder's mask predictions and
query embeddings, then aligns them with precomputed camera CLIP features.

This provides dense, per-voxel open-vocabulary supervision — every voxel
with a valid camera projection gets pushed toward its camera CLIP feature,
regardless of class (including novel classes).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VoxelDistillLoss(nn.Module):
    """Voxel-level distillation: reconstructed per-voxel features → camera CLIP."""

    def __init__(self, loss_type="cosine"):
        """
        Args:
            loss_type: "cosine" (1 - cos_sim) or "smooth_l1"
        """
        super().__init__()
        self.loss_type = loss_type

    def forward(self, pred_embed, pred_masks, vox_clip_feats, vox_clip_indices):
        """
        Args:
            pred_embed: (B, Q, 768) query embeddings from decoder
            pred_masks: (B, V, Q) mask logits
            vox_clip_feats: (B, 4096, 768) precomputed CLIP features at sampled voxels
            vox_clip_indices: (B, 4096) voxel indices into the V-dim

        Returns:
            loss: scalar tensor
        """
        device = pred_embed.device
        B = pred_embed.shape[0]
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        for b in range(B):
            indices = vox_clip_indices[b].long()  # (4096,)
            clip_feats = vox_clip_feats[b]         # (4096, 768)

            # Skip if all zeros (no valid projections)
            if clip_feats.norm(dim=1).mean() < 0.1:
                continue

            # Get mask weights at sampled voxel locations.
            # softmax over queries so each voxel's weight sums to 1 — the dominant
            # query takes most of the weight instead of blending all queries equally.
            V = pred_masks.shape[1]
            indices = indices.clamp(0, V - 1)
            weights = pred_masks[b, indices, :].softmax(dim=-1)  # (4096, Q)

            # Reconstruct per-voxel features: softmax-weighted sum of query embeddings
            # (4096, Q) @ (Q, 768) → (4096, 768)
            voxel_feats = weights @ pred_embed[b]   # (4096, 768)
            voxel_feats = F.normalize(voxel_feats, dim=-1)

            # Filter out zero-norm targets (padding from samples with <4096 valid)
            valid = clip_feats.norm(dim=1) > 0.1  # (4096,)
            if valid.sum() == 0:
                continue

            voxel_feats_valid = voxel_feats[valid]
            clip_feats_valid = clip_feats[valid]

            if self.loss_type == "cosine":
                cos_sim = F.cosine_similarity(voxel_feats_valid, clip_feats_valid, dim=-1)
                loss_b = (1.0 - cos_sim).mean()
            else:
                loss_b = F.smooth_l1_loss(voxel_feats_valid, clip_feats_valid)

            total_loss = total_loss + loss_b

        total_loss = total_loss / max(B, 1)
        return total_loss
