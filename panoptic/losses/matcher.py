"""Hungarian matcher for panoptic segmentation.

Adapted from MaskPLS / Mask2Former. Computes optimal query-GT assignment
using classification cost + sigmoid BCE cost + dice cost.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment as lsa
from torch import nn
from torch.cuda.amp import autocast


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=2.0, cost_mask=5.0, cost_dice=5.0, p_ratio=0.4):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.p_ratio = p_ratio

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with 'pred_logits' (B, Q, C+1) and 'pred_masks' (B, V, Q)
            targets: dict with 'classes' list[Tensor(M,)] and 'masks' list[Tensor(M, V)]
        Returns:
            list of (pred_idx, tgt_idx) tuples per batch element
        """
        bs, num_queries, num_classes = outputs["pred_logits"].shape
        indices = []
        for b in range(bs):
            out_prob = outputs["pred_logits"][b].softmax(-1)
            tgt_ids = targets["classes"][b].long()
            if len(tgt_ids) == 0:
                indices.append((torch.tensor([], dtype=torch.int64),
                                torch.tensor([], dtype=torch.int64)))
                continue

            cost_class = -out_prob[:, tgt_ids]  # (Q, M)

            out_mask = outputs["pred_masks"][b].permute(1, 0)  # (Q, V)
            tgt_mask = targets["masks"][b].to(out_mask)        # (M, V)
            n_pts = tgt_mask.shape[1]

            # Subsample points for efficiency (without replacement)
            n_sample = int(self.p_ratio * n_pts)
            if n_sample < 1:
                n_sample = n_pts
            if n_sample >= n_pts:
                pt_idx = torch.arange(n_pts)
            else:
                pt_idx = torch.randperm(n_pts)[:n_sample]
            tgt_mask_s = tgt_mask[:, pt_idx]
            out_mask_s = out_mask[:, pt_idx]

            with autocast(enabled=False):
                out_mask_s = out_mask_s.float()
                tgt_mask_s = tgt_mask_s.float()
                cost_mask_val = _batch_sigmoid_ce_cost(out_mask_s, tgt_mask_s)
                cost_dice_val = _batch_dice_cost(out_mask_s, tgt_mask_s)

            C = (
                self.cost_class * cost_class
                + self.cost_mask * cost_mask_val
                + self.cost_dice * cost_dice_val
            )
            C = C.reshape(num_queries, -1).cpu()
            indices.append(lsa(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64),
             torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


def _batch_dice_cost(inputs, targets):
    """Dice cost between predictions and targets. inputs: (Q, P), targets: (M, P)."""
    inputs = inputs.sigmoid()
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    return 1 - (numerator + 1) / (denominator + 1)


def _batch_sigmoid_ce_cost(inputs, targets):
    """Sigmoid CE cost. inputs: (Q, P), targets: (M, P)."""
    hw = inputs.shape[1]
    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )
    return loss / hw
