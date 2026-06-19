"""Open-vocabulary panoptic loss.

Aligned with Xiao et al. ECCV 2024 (arXiv:2401.02402) loss formulation:

    L = w_α·L_cls + w_β·L_mask + w_λ·L_O + w_γ·L_V + w_δ·L_bg

Where:
    L_cls  — focal/CE classification loss on BASE classes only (novel masked out)
    L_mask — per-voxel mask BCE + Dice (unchanged from closed-set)
    L_O    — Object-level distillation: 1 - cos(v_q, w_q) for matched queries
             Forces query CLIP embed to match mask-pooled backbone CLIP features
             → directly teaches novel THINGS detection
    L_V    — Voxel-level distillation: L1(M_Q^T @ F_Qemb, F_backbone)
             Reconstructs per-voxel CLIP from all query embeddings weighted by masks
             Applied to ALL queries (no GT matching needed)
             → the key loss for novel STUFF detection
             Optionally density-stratified: sparse novel things get higher weight
    L_bg   — Background CLIP anchor loss for UNMATCHED queries (our addition):
             1 - cos(v_q_unmatched, mask_pool(F_backbone, attn_unmatched))
             Unmatched queries fire on novel class regions at test time —
             this loss pulls them toward the CLIP centroid of those voxels
             even without GT labels. Directly improves novel THINGS detection.

    Paper weights: w_α=1, w_β=1, w_λ=1, w_γ=0.1
    Our additions: w_δ=0.3 (L_bg)

B12/N4 Training Protocol:
    - Novel class indices (bicycle=1, bus=2, motorcycle=5, vegetation=15) are
      EXCLUDED from L_cls during training (set to no-object in matching).
    - At inference, all 16 text embeddings are passed. Novel classes are discovered
      via the CLIP similarity alone (no training supervision needed).
    - L_V uses density stratification: sparse voxels (novel things like bicycles
      with ~80 pts) get higher per-voxel weight than dense stuff (vegetation
      with ~50k pts), ensuring novel things get adequate gradient signal.

How this enables novel class detection:
    The backbone was pre-distilled from OpenSeg/CLIP on millions of frames:
        Loss = 1 - cos(backbone_output_voxel, openseg_2d_feature)
    So F_backbone is already in CLIP space. L_V forces ALL query embeddings —
    including those covering novel class regions — to reconstruct F_backbone.
    These queries stay in CLIP space even though they have zero class supervision.
    At test time, cos(query_embed, text["bicycle"]) is high for those queries.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from panoptic.losses.matcher import HungarianMatcher

# B12/N4 main split: novel class 0-indexed indices
# bicycle=1, bus=2, motorcycle=5, vegetation=15
NUSCENES_B12N4_NOVEL = [1, 2, 5, 15]
NUSCENES_B12N4_NOVEL_THINGS = [1, 2, 5]   # bicycle, bus, motorcycle
NUSCENES_B12N4_NOVEL_STUFF  = [15]         # vegetation


class OpenVocabPanopticLoss(nn.Module):
    """Loss for open-vocabulary panoptic segmentation (paper-aligned).

    Components:
        L_cls   : CE/focal on BASE classes only  (novel classes → no-object)
        L_mask  : mask BCE + Dice  (class-agnostic, same as closed-set)
        L_O     : object-level distillation  (matched queries only)
        L_V     : voxel-level distillation   (ALL queries)
    """

    def __init__(
        self,
        num_classes=16,
        eos_coef=0.1,
        # Matcher costs
        cost_class=2.0,
        cost_mask=5.0,
        cost_dice=5.0,
        # Loss weights (paper defaults)
        weight_cls=1.0,
        weight_mask=1.0,
        weight_dice=1.0,
        weight_object_distill=1.0,    # w_λ  (L_O)
        weight_voxel_distill=0.1,     # w_γ  (L_V) — small, prevents training instability
        weight_background_distill=0.3, # w_δ  (L_bg) — our addition, unmatched query anchor
        # Legacy aliases (keep backward compat)
        weight_objectness=None,
        weight_region_clip=None,
        weight_aux_semantic=None,
        # Params
        focal_gamma=4.0,
        focal_alpha=0.25,
        num_points=50000,
        num_mask_pts=500,
        p_ratio=0.75,
        # B12/N4 split — novel classes excluded from cls loss
        novel_class_idx=None,
        # Density stratification for L_V: upweight sparse voxels (novel things)
        use_density_stratified_lv=True,
        # Thing/stuff for no-object weight
        thing_ids=None,
        stuff_ids=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.eos_coef = eos_coef
        self.weight_cls = weight_cls
        self.weight_mask = weight_mask
        self.weight_dice = weight_dice
        self.weight_object_distill = weight_object_distill
        self.weight_voxel_distill = weight_voxel_distill
        self.weight_background_distill = weight_background_distill
        self.use_density_stratified_lv = use_density_stratified_lv
        self.num_points = num_points
        self.num_mask_pts = num_mask_pts
        self.p_ratio = p_ratio

        # Novel classes: excluded from L_cls during training
        self.novel_class_idx = set(novel_class_idx or NUSCENES_B12N4_NOVEL)

        self.thing_ids = set(thing_ids or range(10))
        self.stuff_ids = set(stuff_ids or range(10, 16))

        # no-object class weight
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

        self.matcher = HungarianMatcher(
            cost_class=cost_class,
            cost_mask=cost_mask,
            cost_dice=cost_dice,
            p_ratio=p_ratio,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Main forward
    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, outputs, targets, text_prototypes=None, backbone_feats=None):
        """
        Args:
            outputs: dict with keys:
                - pred_logits: (B, Q, C+1) class logits  [or None if using query_clip]
                - pred_masks:  (B, V, Q) mask logits
                - query_clip:  (B, Q, 768) predicted CLIP features  [for L_O]
                - region_clip: (B, Q, 768) mask-pooled backbone CLIP [for L_O target]
            targets: dict with:
                - classes: list of (M,) GT class labels  (0-indexed, base classes only)
                - masks:   list of (M, V) or (V, M) GT binary masks
            text_prototypes: (C, 768) CLIP text embeddings for base classes
            backbone_feats: (B, V, 768) raw backbone CLIP features  [for L_V]
        Returns:
            (losses_dict, indices)
        """
        # ── Hungarian matching ────────────────────────────────────────────
        # Mask novel classes from matching (treat as no-object during training)
        match_targets = self._mask_novel_from_targets(targets)
        indices = self.matcher(outputs, match_targets)

        losses = {}

        # ── L_cls: classification on base classes only ────────────────────
        if "pred_logits" in outputs and outputs["pred_logits"] is not None:
            loss_cls = self._cls_loss(outputs, match_targets, indices)
            losses["cls"] = loss_cls * self.weight_cls

        # ── L_mask: BCE + Dice ────────────────────────────────────────────
        loss_bce, loss_dice = self._mask_loss(outputs, targets, indices)
        losses["mask_bce"] = loss_bce * self.weight_mask
        losses["mask_dice"] = loss_dice * self.weight_dice

        # ── L_O: object-level distillation (matched queries only) ─────────
        if "query_clip" in outputs and "region_clip" in outputs:
            loss_obj_distill = self._object_distill_loss(outputs, indices)
            losses["object_distill"] = loss_obj_distill * self.weight_object_distill

        # ── L_V: voxel-level distillation (ALL queries, no GT needed) ─────
        if backbone_feats is not None and "query_clip" in outputs:
            loss_vox_distill = self._voxel_distill_loss(outputs, backbone_feats)
            losses["voxel_distill"] = loss_vox_distill * self.weight_voxel_distill

        # ── L_bg: background CLIP anchor for UNMATCHED queries (our addition) ─
        # Matched queries are supervised by L_O. Unmatched queries — which fire
        # on novel class regions at test time — get zero object-level supervision
        # from L_O. L_bg pulls them toward the CLIP centroid of the voxels they
        # attend to, keeping them in CLIP space even without GT labels.
        if backbone_feats is not None and "query_clip" in outputs and "pred_masks" in outputs:
            loss_bg = self._background_distill_loss(outputs, backbone_feats, indices)
            losses["background_distill"] = loss_bg * self.weight_background_distill

        # ── Auxiliary losses from intermediate decoder layers ─────────────
        if "aux_outputs" in outputs:
            for i, aux in enumerate(outputs["aux_outputs"]):
                aux_indices = self.matcher(aux, match_targets)

                if "pred_logits" in aux and aux["pred_logits"] is not None:
                    losses[f"{i}_cls"] = self._cls_loss(aux, match_targets, aux_indices) * self.weight_cls

                l_bce, l_dice = self._mask_loss(aux, targets, aux_indices)
                losses[f"{i}_mask_bce"] = l_bce * self.weight_mask
                losses[f"{i}_mask_dice"] = l_dice * self.weight_dice

                if "query_clip" in aux and "region_clip" in aux:
                    losses[f"{i}_object_distill"] = (
                        self._object_distill_loss(aux, aux_indices) * self.weight_object_distill
                    )

        return losses, indices

    # ──────────────────────────────────────────────────────────────────────────
    # B12/N4: mask novel classes from classification supervision
    # ──────────────────────────────────────────────────────────────────────────

    def _mask_novel_from_targets(self, targets):
        """Return targets with novel-class instances removed from GT.

        Novel class voxels still exist in the scene — masks are kept for L_V
        (voxel distillation doesn't need GT class labels). But for the
        classification and Hungarian matching, we treat novel GT instances
        as if they don't exist.
        """
        masked_targets = {"classes": [], "masks": []}
        for b in range(len(targets["classes"])):
            cls = targets["classes"][b]           # (M,) int tensor
            msk = targets["masks"][b]             # (M, V) or (V, M)

            # Keep only base-class instances
            keep = torch.tensor(
                [i for i, c in enumerate(cls.tolist()) if c not in self.novel_class_idx],
                dtype=torch.long, device=cls.device
            )
            if len(keep) == 0:
                masked_targets["classes"].append(cls[:0])
                masked_targets["masks"].append(msk[:0] if msk.shape[0] == len(cls) else msk[:, :0])
            else:
                masked_targets["classes"].append(cls[keep])
                if msk.shape[0] == len(cls):
                    masked_targets["masks"].append(msk[keep])
                else:
                    masked_targets["masks"].append(msk[:, keep])

        return masked_targets

    # ──────────────────────────────────────────────────────────────────────────
    # L_cls — classification loss (base classes only)
    # ──────────────────────────────────────────────────────────────────────────

    def _cls_loss(self, outputs, targets, indices):
        """CE loss between predicted class logits and GT base-class labels."""
        pred_logits = outputs["pred_logits"]  # (B, Q, C+1)
        B, Q, _ = pred_logits.shape

        # Build target label tensor: default = no-object (C)
        tgt_classes = torch.full((B, Q), self.num_classes,
                                 dtype=torch.long, device=pred_logits.device)

        for b_idx, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            tgt_classes[b_idx, pred_idx] = targets["classes"][b_idx][gt_idx]

        loss = F.cross_entropy(
            pred_logits.reshape(-1, self.num_classes + 1),
            tgt_classes.reshape(-1),
            weight=self.empty_weight,
        )
        return loss

    # ──────────────────────────────────────────────────────────────────────────
    # L_mask — BCE + Dice
    # ──────────────────────────────────────────────────────────────────────────

    def _mask_loss(self, outputs, targets, indices):
        pred_masks = outputs["pred_masks"]  # (B, V, Q)
        total_bce = 0.0
        total_dice = 0.0
        n_matched = 0

        for b_idx, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue

            gt_masks = targets["masks"][b_idx]
            V = pred_masks.shape[1]

            pred_m = pred_masks[b_idx, :, pred_idx]   # (V, matched)
            gt_m = gt_masks[gt_idx].float()

            # Ensure shape (V, matched)
            if gt_m.dim() == 2 and gt_m.shape[0] != V:
                gt_m = gt_m.T
            if gt_m.dim() == 2 and gt_m.shape[0] != V:
                gt_m = gt_m.T

            n_pts = min(self.num_mask_pts, V)
            sample_idx = torch.randperm(V, device=pred_m.device)[:n_pts]
            pred_s = pred_m[sample_idx]
            gt_s   = gt_m[sample_idx]

            bce = F.binary_cross_entropy_with_logits(pred_s, gt_s, reduction="mean")
            total_bce += bce

            pred_sig = pred_s.sigmoid()
            inter = (pred_sig * gt_s).sum(0)
            union = pred_sig.sum(0) + gt_s.sum(0)
            dice = 1.0 - (2 * inter + 1) / (union + 1)
            total_dice += dice.mean()
            n_matched += 1

        n = max(n_matched, 1)
        return total_bce / n, total_dice / n

    # ──────────────────────────────────────────────────────────────────────────
    # L_O — Object-level distillation (Eq. 4 in paper)
    # ──────────────────────────────────────────────────────────────────────────

    def _object_distill_loss(self, outputs, indices):
        """L_O = mean over matched queries of (1 - cos(v_q, w_q))

        v_q = predicted class embedding (from RegionCLIPHead)
        w_q = mask-pooled backbone CLIP features (from RegionPooler / ContextualRegionPooler)

        Applied ONLY to matched queries (unmatched = novel regions, no GT label,
        but backbone features are still reliable supervision).
        """
        query_clip  = outputs["query_clip"]   # (B, Q, 768)  v_q
        region_clip = outputs["region_clip"]  # (B, Q, 768)  w_q

        total_loss = 0.0
        n_matched = 0

        for b_idx, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue

            v_q = query_clip[b_idx, pred_idx]   # (M, 768)
            w_q = region_clip[b_idx, pred_idx]  # (M, 768)

            # Cosine distance: 1 - cos_sim
            if torch.isnan(v_q).any() or torch.isnan(w_q).any():
                continue
            v_q = F.normalize(v_q, dim=-1)
            w_q = F.normalize(w_q, dim=-1)
            cos = F.cosine_similarity(v_q, w_q, dim=-1).clamp(-1, 1)
            loss = (1.0 - cos).mean()
            if torch.isnan(loss):
                continue
            total_loss += loss
            n_matched += 1

        return total_loss / max(n_matched, 1)

    # ──────────────────────────────────────────────────────────────────────────
    # L_V — Voxel-level distillation (Eq. 5-6 in paper)
    # ──────────────────────────────────────────────────────────────────────────

    def _voxel_distill_loss(self, outputs, backbone_feats):
        """L_V = L1(F_rec, F_backbone)

        F_rec = M_Q^T @ F_Qemb
            M_Q    : (B, Q, V) predicted mask probabilities (all queries)
            F_Qemb : (B, Q, D) predicted class embeddings
        F_backbone: (B, V, D) raw backbone CLIP features (detached — fixed target)

        This loss does NOT require GT labels — applied to ALL queries.
        Unmatched queries (covering novel-class regions) learn to reconstruct
        backbone CLIP features in novel regions → enabling novel class detection.

        This is the KEY loss for novel STUFF classes (vegetation etc.).
        Paper ablation: removing L_V drops vegetation PQ from 35.2 → 0.2.

        Optional density stratification (use_density_stratified_lv=True):
            Weight each voxel by 1 / local_point_density.
            Sparse novel things (bicycle ~80 pts) get 10-100× higher weight
            than dense stuff (vegetation ~50k pts). Without this, L_V gradient
            is dominated by large stuff regions and bicycle/motorcycle barely train.
        """
        query_clip = outputs["query_clip"]   # (B, Q, 768)  F_Qemb
        pred_masks = outputs["pred_masks"]   # (B, V, Q)    used as M_Q

        B, V, Q = pred_masks.shape

        # M_Q: (B, Q, V) — mask probabilities
        M_Q = pred_masks.sigmoid().permute(0, 2, 1)  # (B, Q, V)

        # F_rec = M_Q^T @ F_Qemb: (B, V, Q) @ (B, Q, D) → (B, V, D)
        M_Q_T = M_Q.permute(0, 2, 1)                            # (B, V, Q)
        F_rec = torch.bmm(M_Q_T, query_clip)                    # (B, V, D)

        if backbone_feats.shape != F_rec.shape:
            min_v = min(backbone_feats.shape[1], F_rec.shape[1])
            backbone_feats = backbone_feats[:, :min_v, :]
            F_rec = F_rec[:, :min_v, :]

        if self.use_density_stratified_lv and "voxel_coords" in outputs:
            # Density-stratified weighting: upweight sparse novel things
            # local_density approximated from nearest-neighbour count in voxel grid
            coords = outputs["voxel_coords"]  # (B, V, 3)
            density_weight = self._compute_density_weight(coords)  # (B, V)
            per_voxel_loss = F.l1_loss(F_rec, backbone_feats.detach(), reduction="none")  # (B, V, D)
            loss = (per_voxel_loss.mean(dim=-1) * density_weight).mean()
        else:
            loss = F.l1_loss(F_rec, backbone_feats.detach())

        if torch.isnan(loss):
            return torch.tensor(0.0, device=loss.device if hasattr(loss, "device") else "cpu", requires_grad=True)
        return loss

    def _compute_density_weight(self, coords, radius=0.5):
        """Compute inverse local density weight for density-stratified L_V.

        Args:
            coords: (B, V, 3) voxel coordinates
            radius: neighbourhood radius in voxel units (default 0.5 ~ 1 voxel)

        Returns:
            density_weight: (B, V) normalised inverse density weights.
            Sparse voxels (novel things: bicycle ~80 pts) get high weight.
            Dense voxels (vegetation ~50k pts) get low weight.
        """
        B, V, _ = coords.shape
        weights = []
        for b in range(B):
            c = coords[b].float()  # (V, 3)
            # Approximate local density: count voxels within radius
            # Use a fast approximation: pairwise L-inf distance
            if V > 4096:
                # Subsample for speed; apply weight to all using nearest-subsample
                idx = torch.randperm(V, device=c.device)[:4096]
                c_sub = c[idx]
                dist = torch.cdist(c, c_sub, p=float('inf'))  # (V, 4096)
            else:
                dist = torch.cdist(c, c, p=float('inf'))  # (V, V)

            local_count = (dist < radius).float().sum(dim=-1)  # (V,)
            w = 1.0 / (local_count + 1.0)
            w = w / w.mean()  # normalise so mean weight = 1
            weights.append(w)
        return torch.stack(weights, dim=0)  # (B, V)

    def _background_distill_loss(self, outputs, backbone_feats, indices):
        """L_bg: background CLIP anchor loss for UNMATCHED queries (our addition).

        Problem: L_O only supervises matched queries (those with a GT instance).
        Unmatched queries get zero object-level supervision — only L_V touches them
        indirectly. But unmatched queries are exactly the ones that will fire on
        novel class regions at test time (bicycle, bus, vegetation).

        Fix: Pull each unmatched query toward the CLIP centroid of the voxels
        it attends to (weighted by its predicted mask), even without GT labels.

            w_q_unmatched = mask_pool(F_backbone, attn_unmatched)
            L_bg = mean( 1 - cos(v_q_unmatched, w_q_unmatched) )

        This directly improves novel THINGS detection (bicycle, motorcycle) because:
        - Their voxels are already near "bicycle" in CLIP space (from pre-distillation)
        - L_bg anchors the unmatched query to that CLIP centroid at training time
        - At test time, cos(query, text["bicycle"]) is high

        Cost: one extra mask-pool operation per batch. No new architecture needed.
        """
        query_clip = outputs["query_clip"]   # (B, Q, 768)
        pred_masks = outputs["pred_masks"]   # (B, V, Q)

        B, V, Q = pred_masks.shape
        total_loss = 0.0
        n_items = 0

        for b_idx, (pred_idx, _) in enumerate(indices):
            all_query_idx = torch.arange(Q, device=query_clip.device)
            matched_set = set(pred_idx.tolist()) if len(pred_idx) > 0 else set()
            unmatched_mask = torch.tensor(
                [i not in matched_set for i in range(Q)],
                dtype=torch.bool, device=query_clip.device
            )

            if unmatched_mask.sum() == 0:
                continue

            v_q_unmatched = query_clip[b_idx][unmatched_mask]           # (U, 768)
            attn_weights   = pred_masks[b_idx, :, unmatched_mask].sigmoid()  # (V, U)

            # mask_pool: weighted average of backbone CLIP features per query
            # w_q = (attn^T @ F_backbone) / attn.sum  → (U, 768)
            attn_norm = attn_weights.sum(dim=0, keepdim=True).clamp(min=1e-2)  # (1, U)
            # skip queries with negligible attention (avoid NaN)
            valid = attn_norm.squeeze(0) > 1e-2  # (U,)
            if valid.sum() == 0:
                continue
            F_bb = backbone_feats[b_idx]  # (V, 768)
            w_q = (attn_weights[:, valid].T @ F_bb) / attn_norm[:, valid].T  # (U', 768)
            v_q_valid = v_q_unmatched[valid]  # (U', 768)
            # NaN guard
            if torch.isnan(w_q).any() or torch.isnan(v_q_valid).any():
                continue
            cos = F.cosine_similarity(v_q_valid, w_q.detach(), dim=-1)
            loss = (1.0 - cos).mean()
            if torch.isnan(loss):
                continue
            total_loss += loss
            n_items += 1

        return total_loss / max(n_items, 1)



class OpenVocabPanopticLossV2(nn.Module):
    """Loss for open-vocabulary panoptic segmentation.
    
    Components:
        1. Objectness CE: thing/stuff/no-object (class-agnostic detection)
        2. Mask BCE + Dice: same as closed-set
        3. Region-CLIP alignment: contrastive loss between query CLIP features
           and GT class text prototypes
        4. Region consistency: L2/cosine between query_clip and region_pooled_clip
    """
    
    def __init__(
        self,
        num_classes=16,
        eos_coef=0.1,
        # Matcher costs (class cost uses objectness similarity)
        cost_class=2.0,
        cost_mask=5.0,
        cost_dice=5.0,
        # Loss weights
        weight_objectness=2.0,
        weight_mask=5.0,
        weight_dice=5.0,
        weight_region_clip=4.0,
        weight_aux_semantic=1.0,
        # Standard params
        focal_gamma=4.0,
        focal_alpha=0.25,
        num_points=50000,
        num_mask_pts=500,
        p_ratio=0.75,
        # Open-vocab specific
        objectness_mode="ternary",
        thing_ids=None,
        stuff_ids=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.eos_coef = eos_coef
        self.weight_objectness = weight_objectness
        self.weight_mask = weight_mask
        self.weight_dice = weight_dice
        self.weight_region_clip = weight_region_clip
        self.weight_aux_semantic = weight_aux_semantic
        self.num_points = num_points
        self.num_mask_pts = num_mask_pts
        self.p_ratio = p_ratio
        self.objectness_mode = objectness_mode
        
        # Thing class IDs (0-9) and stuff class IDs (10-15) for nuScenes
        self.thing_ids = set(thing_ids or range(10))
        self.stuff_ids = set(stuff_ids or range(10, 16))
        
        # Objectness weight: lower weight for no-object class
        if objectness_mode == "ternary":
            obj_weight = torch.ones(3)
            obj_weight[2] = eos_coef  # no-object class
        else:
            obj_weight = torch.ones(2)
            obj_weight[1] = eos_coef
        self.register_buffer("obj_weight", obj_weight)
        
        # Matcher (uses objectness + mask costs)
        self.matcher = HungarianMatcher(
            cost_class=cost_class,
            cost_mask=cost_mask,
            cost_dice=cost_dice,
            p_ratio=p_ratio,
        )
    
    def forward(self, outputs, targets, text_prototypes=None, backbone_feats=None):
        """
        Args:
            outputs: dict with keys:
                - pred_masks: (B, V, Q) mask logits
                - objectness: (B, Q, 2or3) objectness logits
                - query_clip: (B, Q, 768) predicted CLIP features
                - region_clip: (B, Q, 768) pooled backbone CLIP features
                - aux_outputs: list of dicts (same keys, from intermediate layers)
            targets: dict with keys:
                - classes: list of (M,) GT class labels per sample
                - masks: list of (M, V) GT binary masks per sample
            text_prototypes: (num_classes, 768) CLIP text embeddings
            backbone_feats: (B, V, 768) raw backbone features (for aux alignment)
        
        Returns:
            (losses_dict, indices)
        """
        # Compute matching using objectness-based class cost
        # Convert to format matcher expects: fake "class logits" from objectness
        match_outputs = self._prepare_matcher_inputs(outputs, targets)
        indices = self.matcher(match_outputs, targets)
        
        losses = {}
        
        # 1. Objectness loss
        loss_obj = self._objectness_loss(outputs, targets, indices)
        losses["objectness"] = loss_obj * self.weight_objectness
        
        # 2. Mask losses (BCE + Dice)
        loss_bce, loss_dice = self._mask_loss(outputs, targets, indices)
        losses["mask_bce"] = loss_bce * self.weight_mask
        losses["mask_dice"] = loss_dice * self.weight_dice
        
        # 3. Region-CLIP contrastive loss
        if text_prototypes is not None and "query_clip" in outputs:
            loss_clip = self._region_clip_loss(outputs, targets, indices, text_prototypes)
            losses["region_clip"] = loss_clip * self.weight_region_clip
        
        # 4. Region consistency loss (query_clip should match region_pooled_clip)
        if "region_clip" in outputs and "query_clip" in outputs:
            loss_consist = self._consistency_loss(outputs, indices)
            losses["region_consist"] = loss_consist * self.weight_aux_semantic
        
        # Auxiliary losses from intermediate layers
        if "aux_outputs" in outputs:
            for i, aux in enumerate(outputs["aux_outputs"]):
                aux_match = self._prepare_matcher_inputs(aux, targets)
                aux_indices = self.matcher(aux_match, targets)
                
                l_obj = self._objectness_loss(aux, targets, aux_indices)
                l_bce, l_dice = self._mask_loss(aux, targets, aux_indices)
                losses[f"{i}_objectness"] = l_obj * self.weight_objectness
                losses[f"{i}_mask_bce"] = l_bce * self.weight_mask
                losses[f"{i}_mask_dice"] = l_dice * self.weight_dice
                
                if text_prototypes is not None and "query_clip" in aux:
                    l_clip = self._region_clip_loss(aux, targets, aux_indices, text_prototypes)
                    losses[f"{i}_region_clip"] = l_clip * self.weight_region_clip
        
        return losses, indices
    
    def _prepare_matcher_inputs(self, outputs, targets):
        """Convert objectness to fake class logits for the matcher."""
        # The matcher expects pred_logits of shape (B, Q, C+1)
        # We create fake logits: matched class cost = 1 - objectness_prob[thing/stuff]
        objectness = outputs["objectness"]  # (B, Q, 2or3)
        B, Q, _ = objectness.shape
        
        # Create pseudo-logits: (B, Q, num_classes+1)
        # All GT classes get the same "object" probability
        obj_probs = objectness.softmax(-1)
        
        if self.objectness_mode == "ternary":
            thing_prob = obj_probs[..., 0:1]  # (B, Q, 1)
            stuff_prob = obj_probs[..., 1:2]
            no_obj_prob = obj_probs[..., 2:3]
        else:
            thing_prob = obj_probs[..., 0:1]
            no_obj_prob = obj_probs[..., 1:2]
            stuff_prob = thing_prob  # same as thing for binary
        
        # Expand: thing classes get thing_prob, stuff classes get stuff_prob
        fake_logits = torch.zeros(B, Q, self.num_classes + 1, device=objectness.device)
        for c in self.thing_ids:
            fake_logits[..., c] = thing_prob.squeeze(-1)
        for c in self.stuff_ids:
            fake_logits[..., c] = stuff_prob.squeeze(-1)
        fake_logits[..., self.num_classes] = no_obj_prob.squeeze(-1)
        
        return {
            "pred_logits": fake_logits,
            "pred_masks": outputs["pred_masks"],
        }
    
    def _objectness_loss(self, outputs, targets, indices):
        """CE loss for objectness prediction."""
        objectness = outputs["objectness"]  # (B, Q, 2or3)
        B, Q, _ = objectness.shape
        
        # Build objectness targets
        if self.objectness_mode == "ternary":
            # 0=thing, 1=stuff, 2=no-object
            obj_targets = torch.full((B, Q), 2, dtype=torch.long, device=objectness.device)
        else:
            obj_targets = torch.full((B, Q), 1, dtype=torch.long, device=objectness.device)
        
        for b_idx, (pred_idx, gt_idx) in enumerate(indices):
            gt_classes = targets["classes"][b_idx]
            for p, g in zip(pred_idx, gt_idx):
                cls = gt_classes[g].item()
                if self.objectness_mode == "ternary":
                    obj_targets[b_idx, p] = 0 if cls in self.thing_ids else 1
                else:
                    obj_targets[b_idx, p] = 0  # object
        
        loss = F.cross_entropy(
            objectness.reshape(-1, objectness.shape[-1]),
            obj_targets.reshape(-1),
            weight=self.obj_weight,
        )
        return loss
    
    def _mask_loss(self, outputs, targets, indices):
        """BCE + Dice mask losses (same as closed-set)."""
        pred_masks = outputs["pred_masks"]  # (B, V, Q)
        
        total_bce = 0.0
        total_dice = 0.0
        n_matched = 0
        
        for b_idx, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            
            gt_masks = targets["masks"][b_idx]  # (M, V) or (V, M)
            V = pred_masks.shape[1]
            
            pred_m = pred_masks[b_idx, :, pred_idx]  # (V, matched)
            gt_m = gt_masks[gt_idx].float()  # (matched, V)
            
            if gt_m.shape[-1] != V:
                gt_m = gt_m.T  # ensure (V, matched)
            if gt_m.shape[0] != V:
                gt_m = gt_m.T
            
            # Point sampling for efficiency
            n_pts = min(self.num_mask_pts, V)
            sample_idx = torch.randperm(V, device=pred_m.device)[:n_pts]
            
            pred_sampled = pred_m[sample_idx]  # (n_pts, matched)
            gt_sampled = gt_m[sample_idx]      # (n_pts, matched)
            
            # BCE
            bce = F.binary_cross_entropy_with_logits(pred_sampled, gt_sampled, reduction="mean")
            total_bce += bce
            
            # Dice
            pred_sig = pred_sampled.sigmoid()
            intersection = (pred_sig * gt_sampled).sum(dim=0)
            union = pred_sig.sum(dim=0) + gt_sampled.sum(dim=0)
            dice = 1.0 - (2 * intersection + 1) / (union + 1)
            total_dice += dice.mean()
            
            n_matched += 1
        
        n_matched = max(n_matched, 1)
        return total_bce / n_matched, total_dice / n_matched
    
    def _region_clip_loss(self, outputs, targets, indices, text_prototypes):
        """Contrastive loss: query_clip features should align with GT class prototypes.
        
        This is the KEY loss for open-vocab. Instead of CE over 16 classes,
        we use InfoNCE: positive = GT class prototype, negatives = all other prototypes.
        
        This teaches the head to produce general CLIP-space features that respond
        to ANY text query, not just 16 fixed labels.
        """
        query_clip = outputs["query_clip"]  # (B, Q, 768)
        temperature = outputs.get("temperature", 14.3)
        
        # If temperature is from the model's learnable param
        if hasattr(self, 'temperature'):
            temperature = 14.3  # fallback; real temp comes from model forward
        
        total_loss = 0.0
        n_matched = 0
        
        for b_idx, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            
            gt_classes = targets["classes"][b_idx]
            matched_clip = query_clip[b_idx, pred_idx]  # (matched, 768)
            
            # Get GT prototypes for matched queries
            gt_cls_indices = gt_classes[gt_idx]  # (matched,)
            gt_protos = text_prototypes[gt_cls_indices]  # (matched, 768)
            
            # InfoNCE: similarity to all prototypes, maximize GT similarity
            # logits: (matched, num_prototypes)
            logits = matched_clip @ text_prototypes.T * temperature  # (matched, C)
            labels = gt_cls_indices  # correct prototype index
            
            loss = F.cross_entropy(logits, labels)
            total_loss += loss
            n_matched += 1
        
        return total_loss / max(n_matched, 1)
    
    def _consistency_loss(self, outputs, indices):
        """Ensure query_clip ≈ region_pooled_clip for matched queries.
        
        This regularizes the query head to be consistent with what the
        backbone actually sees in the predicted region.
        """
        query_clip = outputs["query_clip"]    # (B, Q, 768)
        region_clip = outputs["region_clip"]  # (B, Q, 768)
        
        total_loss = 0.0
        n_matched = 0
        
        for b_idx, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            
            q_feat = query_clip[b_idx, pred_idx]   # (matched, 768)
            r_feat = region_clip[b_idx, pred_idx]  # (matched, 768)
            
            # Cosine similarity loss: 1 - cos_sim
            cos_sim = F.cosine_similarity(q_feat, r_feat, dim=-1)
            loss = (1.0 - cos_sim).mean()
            total_loss += loss
            n_matched += 1
        
        return total_loss / max(n_matched, 1)
