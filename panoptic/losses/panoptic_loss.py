"""Panoptic loss: FocalLoss(class) + BCE(mask) + Dice(mask) with Hungarian matching.

Design decisions:
- FocalLoss (gamma=4.0, alpha=0.25) for classification (handles class imbalance)
- Sigmoid BCE + Dice for masks (well-tested from MaskPLS)
- Auxiliary losses on all intermediate decoder outputs
"""
import torch
import torch.nn.functional as F
from torch import nn

from panoptic.losses.matcher import HungarianMatcher


class PanopticLoss(nn.Module):
    """Combined panoptic segmentation loss with Hungarian matching.

    Aggregates three loss terms for each decoder layer output:
        1. Cross-entropy classification loss (with eos_coef down-weighting
           the no-object class to handle query surplus).
        2. Sigmoid binary cross-entropy mask loss on matched masks.
        3. Dice mask loss on matched masks.

    The module first runs Hungarian matching to pair predicted queries with
    ground-truth instances, then computes losses only on matched pairs.
    Auxiliary losses from intermediate decoder layers are included with the
    same weights (prefixed by layer index).

    Args:
        num_classes (int): Number of foreground semantic classes.
        eos_coef (float): Weight for the no-object (background) class in CE loss.
        cost_class (float): Classification cost weight for matcher.
        cost_mask (float): Mask BCE cost weight for matcher.
        cost_dice (float): Dice cost weight for matcher.
        weight_ce (float): Final multiplier for classification loss.
        weight_mask (float): Final multiplier for mask BCE loss.
        weight_dice (float): Final multiplier for Dice loss.
        focal_gamma (float): Focal loss gamma (unused in current CE variant).
        focal_alpha (float): Focal loss alpha (unused in current CE variant).
        num_points (int): Max points sampled for mask loss computation.
        num_mask_pts (int): Additional mask point budget (reserved).
        p_ratio (float): Point sampling ratio passed to matcher.

    Returns (in forward):
        tuple[dict[str, Tensor], list]: Weighted loss dict and match indices.
    """

    def __init__(
        self,
        num_classes=16,
        eos_coef=0.1,
        cost_class=2.0,
        cost_mask=5.0,
        cost_dice=5.0,
        weight_ce=2.0,
        weight_mask=5.0,
        weight_dice=5.0,
        focal_gamma=4.0,
        focal_alpha=0.25,
        num_points=50000,
        num_mask_pts=500,
        p_ratio=0.4,
    ):
        """Initialize PanopticLoss with matcher, weight dictionary, and hyperparams.

        Creates the HungarianMatcher instance, stores per-loss weights in a
        dictionary used to scale raw losses before back-propagation, and
        records hyperparameters controlling focal loss and point sampling.

        Args:
            num_classes (int): Foreground class count; no-object is class index
                num_classes. Default: 16.
            eos_coef (float): CE weight for the no-object class. Default: 0.1.
            cost_class (float): Matcher classification cost weight. Default: 2.0.
            cost_mask (float): Matcher mask BCE cost weight. Default: 5.0.
            cost_dice (float): Matcher Dice cost weight. Default: 5.0.
            weight_ce (float): Loss multiplier for cross-entropy. Default: 2.0.
            weight_mask (float): Loss multiplier for mask BCE. Default: 5.0.
            weight_dice (float): Loss multiplier for Dice. Default: 5.0.
            focal_gamma (float): Focal loss focusing parameter. Default: 4.0.
            focal_alpha (float): Focal loss balancing factor. Default: 0.25.
            num_points (int): Max voxel points for mask loss sampling. Default: 50000.
            num_mask_pts (int): Reserved mask point budget. Default: 500.
            p_ratio (float): Point ratio for matcher mask costs. Default: 0.4.
        """
        super().__init__()
        self.num_classes = num_classes
        self.eos_coef = eos_coef
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        self.matcher = HungarianMatcher(cost_class, cost_mask, cost_dice, p_ratio)

        self.weight_dict = {
            "loss_ce": weight_ce,
            "loss_mask": weight_mask,
            "loss_dice": weight_dice,
        }

        self.num_points = num_points
        self.num_mask_pts = num_mask_pts

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with 'pred_logits', 'pred_masks', 'aux_outputs'
            targets: dict with 'classes' list[Tensor], 'masks' list[Tensor]
        Returns:
            dict of weighted losses
        """
        losses = {}
        num_masks = sum(len(t) for t in targets["classes"])
        num_masks = max(num_masks, 1.0)

        outputs_no_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Final layer loss
        indices = self.matcher(outputs_no_aux, targets)
        losses.update(self._get_losses(outputs_no_aux, targets, indices, num_masks))

        # Auxiliary losses
        if "aux_outputs" in outputs:
            for i, aux in enumerate(outputs["aux_outputs"]):
                aux_indices = self.matcher(aux, targets)
                aux_losses = self._get_losses(aux, targets, aux_indices, num_masks)
                losses.update({f"{i}_{k}": v for k, v in aux_losses.items()})

        # Apply weights
        weighted = {}
        for loss_name, loss_val in losses.items():
            for w_key, w_val in self.weight_dict.items():
                if w_key in loss_name:
                    weighted[loss_name] = loss_val * w_val
                    break
        return weighted, indices

    def _get_losses(self, outputs, targets, indices, num_masks):
        """Aggregate classification and mask losses for one set of matched pairs.

        Called once for the final decoder output and once per auxiliary layer.
        Combines the cross-entropy classification loss and the mask losses
        (sigmoid BCE + Dice) into a single dictionary.

        Args:
            outputs (dict): Model predictions with keys 'pred_logits' (B, Q, C+1)
                and 'pred_masks' (B, V, Q).
            targets (dict): Ground-truth with 'classes' list[Tensor(M,)] and
                'masks' list[Tensor(M, V)].
            indices (list[tuple[Tensor, Tensor]]): Hungarian matching result,
                one (pred_idx, tgt_idx) pair per batch element.
            num_masks (float): Total number of GT masks across the batch,
                used to normalize mask losses.

        Returns:
            dict[str, Tensor]: Dictionary with keys 'loss_ce', 'loss_mask',
                'loss_dice' containing scalar loss tensors.
        """
        losses = {}
        losses.update(self._loss_class(outputs, targets, indices))
        losses.update(self._loss_masks(outputs, targets, indices, num_masks))
        return losses

    def _loss_class(self, outputs, targets, indices):
        """Cross-entropy classification loss with eos_coef weighting (matching MaskPLS)."""
        pred_logits = outputs["pred_logits"].float()  # (B, Q, C+1)
        B, Q, C_plus_1 = pred_logits.shape

        idx = self._get_pred_idx(indices)
        target_classes_o = torch.cat(
            [t[J] for t, (_, J) in zip(targets["classes"], indices)]
        ).to(pred_logits.device)

        # Default: all queries predict no-object (class = num_classes)
        target_classes = torch.full(
            (B, Q), self.num_classes, dtype=torch.int64, device=pred_logits.device
        )
        target_classes[idx] = target_classes_o

        # Per-class weights: real classes=1.0, no-object=eos_coef
        weights = torch.ones(C_plus_1, device=pred_logits.device)
        weights[-1] = self.eos_coef

        # Standard cross-entropy (same as MaskPLS)
        loss = F.cross_entropy(
            pred_logits.transpose(1, 2),  # (B, C+1, Q)
            target_classes,                # (B, Q)
            weight=weights,
        )

        return {"loss_ce": loss}

    def _loss_masks(self, outputs, targets, indices, num_masks):
        """Sigmoid BCE + Dice loss on matched masks."""
        pred_masks = outputs["pred_masks"]  # (B, V, Q)
        B = pred_masks.shape[0]

        pred_idx = self._get_pred_idx(indices)

        # Gather matched predictions: (N_matched, V)
        pred_matched = pred_masks[pred_idx[0], :, pred_idx[1]]

        # Gather matched targets: (N_matched, V)
        tgt_list = []
        for b_targets, (_, tgt_j) in zip(targets["masks"], indices):
            if len(tgt_j) > 0:
                tgt_list.append(b_targets[tgt_j].to(pred_matched))
        if len(tgt_list) == 0:
            return {"loss_mask": pred_masks.sum() * 0.0, "loss_dice": pred_masks.sum() * 0.0}

        tgt_matched = torch.cat(tgt_list, dim=0)  # (N_matched, V)

        # Point sampling for efficiency (without replacement to reduce noise)
        N_matched, V = pred_matched.shape
        n_sample = min(self.num_points, V)
        if n_sample >= V:
            # Use all points — no sampling needed
            pred_sampled = pred_matched
            tgt_sampled = tgt_matched
        else:
            pt_idx = torch.randperm(V, device=pred_matched.device)[:n_sample]
            pred_sampled = pred_matched[:, pt_idx]
            tgt_sampled = tgt_matched[:, pt_idx]

        # Sigmoid BCE
        loss_mask = F.binary_cross_entropy_with_logits(
            pred_sampled, tgt_sampled, reduction="none"
        ).mean(1).sum() / num_masks

        # Dice
        pred_sig = pred_sampled.sigmoid()
        numerator = 2 * (pred_sig * tgt_sampled).sum(-1)
        denominator = pred_sig.sum(-1) + tgt_sampled.sum(-1)
        loss_dice = (1 - (numerator + 1) / (denominator + 1)).sum() / num_masks

        return {"loss_mask": loss_mask, "loss_dice": loss_dice}

    def _get_pred_idx(self, indices):
        """Convert list of per-sample index tuples to batch and source index tensors.

        Flattens the Hungarian matching output into two 1-D tensors that can
        be used for advanced indexing into (B, Q, ...) prediction tensors.
        The batch_idx tensor holds the sample index for each matched query,
        and src_idx holds the query index within that sample.

        Args:
            indices (list[tuple[Tensor, Tensor]]): Per-batch matched pairs
                where each tuple contains (predicted_indices, target_indices).

        Returns:
            tuple[Tensor, Tensor]: (batch_idx, src_idx) both of shape
                (N_total_matched,) suitable for indexing pred_logits or pred_masks.
        """
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
