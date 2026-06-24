"""L_V: Voxel-level CLIP distillation loss.

Two modes
---------
patch:     (v1) Align reconstructed voxel features with precomputed patch-pooled CLIP
           features from the camera images. Operates on all valid voxels.
broadcast: (v2) Broadcast per-instance crop-CLS CLIP features (obj_feats) to all
           voxels belonging to that GT THING instance. Margin-gated (skip instances
           where the teacher signal itself is not discriminative) and quality-weighted
           (scale by how strongly the teacher aligns with the correct class prototype).
           Base and novel THING classes are included; STUFF is skipped (unreliable
           broadcast due to spatial non-locality).

In both modes the predicted per-voxel feature is reconstructed from the decoder:
  pred_vox[v] = softmax(pred_masks[v, :]) @ pred_embed   # (768,)
This lets gradients flow through both pred_masks (spatial) and pred_embed (semantic).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

_THING_IDS = frozenset(range(10))  # 0-indexed: 0–9 are THING classes
_CLASS_WEIGHTS = {2: 1.5, 5: 4.0, 6: 1.5}  # bus, motorcycle, pedestrian (0-indexed)


class VoxelDistillLoss(nn.Module):

    def __init__(self, mode="patch", loss_type="cosine",
                 margin_gate_threshold=0.0, class_weights=None):
        """
        Args:
            mode:                   "patch" (v1) or "broadcast" (v2)
            loss_type:              "cosine" or "smooth_l1"
            margin_gate_threshold:  skip instances with margin <= this (default 0.0)
            class_weights:          dict {cls_id: weight} override; None = use defaults
        """
        super().__init__()
        self.mode = mode
        self.loss_type = loss_type
        self.margin_gate_threshold = margin_gate_threshold
        self._class_weights = class_weights if class_weights is not None else _CLASS_WEIGHTS

    def forward(self, pred_embed, pred_masks,
                vox_clip_feats=None, vox_clip_indices=None,
                obj_feats=None, obj_valid=None, obj_margin=None,
                obj_classes=None, vox_gt_instance_id=None):
        """
        Args (patch mode):
            pred_embed:       (B, Q, 768)
            pred_masks:       (B, V, Q)
            vox_clip_feats:   (B, 4096, 768) patch-pooled CLIP targets
            vox_clip_indices: (B, 4096)      backbone V-dim positions for each sample

        Args (broadcast mode, additionally):
            obj_feats:          (B, N_inst, 768) crop-CLS CLIP per GT instance
            obj_valid:          (B, N_inst)      bool — instance has a valid crop
            obj_margin:         (B, N_inst)      float — sim_own - sim_best_base
            obj_classes:        (B, N_inst)      int   — 0-indexed class
            vox_gt_instance_id: (B, 4096)        int   — GT instance index per voxel (-1=none)
            vox_clip_indices:   (B, 4096)              — backbone positions (same as patch)

        Returns:
            loss: scalar tensor
        """
        if self.mode == "broadcast":
            return self._forward_broadcast(
                pred_embed, pred_masks, vox_clip_indices,
                obj_feats, obj_valid, obj_margin, obj_classes, vox_gt_instance_id,
            )
        return self._forward_patch(pred_embed, pred_masks, vox_clip_feats, vox_clip_indices)

    # ── patch mode (v1, unchanged) ────────────────────────────────────────────

    def _forward_patch(self, pred_embed, pred_masks, vox_clip_feats, vox_clip_indices):
        device = pred_embed.device
        B = pred_embed.shape[0]
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        for b in range(B):
            indices = vox_clip_indices[b].long()
            clip_feats = vox_clip_feats[b]

            if clip_feats.norm(dim=1).mean() < 0.1:
                continue

            V = pred_masks.shape[1]
            indices = indices.clamp(0, V - 1)
            weights = pred_masks[b, indices, :].softmax(dim=-1)   # (4096, Q)
            voxel_feats = weights @ pred_embed[b]                  # (4096, 768)
            voxel_feats = F.normalize(voxel_feats, dim=-1)

            valid = clip_feats.norm(dim=1) > 0.1
            if valid.sum() == 0:
                continue

            vf = voxel_feats[valid]
            cf = clip_feats[valid]

            if self.loss_type == "cosine":
                loss_b = (1.0 - F.cosine_similarity(vf, cf, dim=-1)).mean()
            else:
                loss_b = F.smooth_l1_loss(vf, cf)

            total_loss = total_loss + loss_b

        return total_loss / max(B, 1)

    # ── broadcast mode (v2) ───────────────────────────────────────────────────

    def _forward_broadcast(self, pred_embed, pred_masks, vox_clip_indices,
                           obj_feats, obj_valid, obj_margin, obj_classes,
                           vox_gt_instance_id):
        device = pred_embed.device
        B = pred_embed.shape[0]
        losses = []

        for b in range(B):
            gi_all  = vox_gt_instance_id[b]      # (4096,) int64
            vox_idx = vox_clip_indices[b].long()  # (4096,) backbone V-dim positions
            of      = obj_feats[b].float()        # (N_inst, 768)
            ov      = obj_valid[b]                # (N_inst,) bool
            om      = obj_margin[b].float()       # (N_inst,) float
            oc      = obj_classes[b]              # (N_inst,) int
            N_inst  = of.shape[0]

            if N_inst == 0:
                continue

            # Vectorised gate over all 4096 sampled voxels
            gi_safe   = gi_all.clamp(0, N_inst - 1)
            valid_gi  = (gi_all >= 0) & (gi_all < N_inst)        # assigned voxels
            is_thing  = (oc[gi_safe] < 10)                        # THING class
            is_valid  = ov[gi_safe]                                # valid crop
            is_pos    = (om[gi_safe] > self.margin_gate_threshold)

            gate = valid_gi & is_thing & is_valid & is_pos
            n_active = int(gate.sum().item())
            if n_active == 0:
                continue

            av    = gate.nonzero(as_tuple=True)[0]  # active voxel sample positions
            a_gi  = gi_safe[av]                      # active instance indices
            a_om  = om[a_gi]                         # active margins
            a_cls = oc[a_gi]                         # active class IDs

            # Quality weight: clamp(margin / 0.05, 0.1, 1.0)
            q_w = (a_om / 0.05).clamp(0.1, 1.0)

            # Class weight
            c_w = torch.ones(n_active, dtype=torch.float32, device=device)
            for cls_id_k, w_k in self._class_weights.items():
                c_w = torch.where(a_cls == cls_id_k,
                                  c_w.new_full(c_w.shape, w_k), c_w)

            per_vox_w = q_w * c_w
            per_vox_w = per_vox_w / (per_vox_w.sum() + 1e-8)  # normalise

            # Broadcast targets: L2-normalised obj_feats[instance]
            targets = F.normalize(of[a_gi], dim=-1)  # (n_active, 768)

            # Reconstruct predicted voxel features
            V_dim   = pred_masks.shape[1]
            bb_idx  = vox_idx[av].clamp(0, V_dim - 1)            # backbone positions
            weights = pred_masks[b, bb_idx, :].softmax(dim=-1)   # (n_active, Q)
            pred_vox = F.normalize(weights @ pred_embed[b], dim=-1)  # (n_active, 768)

            # Weighted cosine loss
            cos_sim  = (pred_vox * targets).sum(dim=-1)           # (n_active,)
            loss_b   = ((1.0 - cos_sim) * per_vox_w).sum()
            losses.append(loss_b)

        if not losses:
            return pred_embed.new_zeros(())

        return sum(losses) / len(losses)
