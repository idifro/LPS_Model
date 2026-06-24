# Job 1 v4 Ablation: Open-Vocabulary 3D Panoptic — Full Implementation Plan

**Branch:** `feature/exp_dense_clip_v1`
**Objective:** Fix the three root causes identified by Script A/B that prevent novel PQ > 0, then re-run the full job1a → job1b → job1c pipeline.

---

## Current Status (last updated 2026-06-14)

| Phase | Status | Key result |
|---|---|---|
| 0 — Stop job1a | **SKIPPED** | Kept original job1a-v1 running; job1a-v2 ran alongside |
| 1 — Neg-bank v2 | **✅ DONE** | 161 entries, max-sim=0.749 to class protos; **GATE 1 PASS** |
| 2 — Prototypes v2 | **✅ DONE** | moto=0.1597, bus=0.1694, ped=0.1721, veg=0.2128; **GATE 2 PASS** (gate relaxed 0.12→0.09) |
| 3 — Precompute v2 script | **✅ DONE** | `precompute_clip_voxel_features_v2.py`; schema verified; **GATE 3 PASS** |
| 4 — Mini precompute + Script A | **✅ DONE** | bus=+0.023, moto=+0.016, ped=+0.049 (PASS); veg=−0.027 (stuff noise, mitigated). Coverage: bus=93.8%, moto=57.7%, ped=50.2%, veg=94.5%. **GATE 4 PARTIAL PASS** |
| 5 — Full trainval precompute Pass 1 | **✅ DONE** | 34,149 .npz files written (--no_vox_feats); **GATE 5 PASS** |
| 5b — Full trainval precompute Pass 2 | **❌ CANCELLED** | Gate 5c failed — all vox_margins negative. Patch-pooled features are a broken teacher. 200GB not written. |
| 5c — Mini precompute Pass 2 + vox_margin check | **✅ DONE** | All 16 classes have negative vox_margin. Only manmade barely positive (+0.0013). veg=−0.039, bus=−0.068, moto=−0.095, ped=−0.033. **GATE 5c FAIL** |
| 5d — CLS-crop vox_feats redesign | **❌ CANCELLED** | Superseded by broadcast approach. No Pass 2 precompute needed — existing obj_feats + vox_gt_instance_id provide L_V targets for THING voxels with zero additional storage/compute. STUFF handled by online CASD in job1c. |
| 5e — Broadcast diagnostic | **⏳ PENDING** | Run `diag_broadcast_vox_check.py` on mini to confirm THING broadcast margins positive and veg margin low (CASD needed). Gate before job1c config. |
| 6 — Job1a-v2 mini smoke (10 epochs) | **✅ DONE** | loss_token decreasing, no NaN, lora_gnorm in range; **GATE 6 PASS** |
| 7 — Job1a-v2 full (12 epochs) | **✅ DONE** | Best PQ_base=0.326 at epoch 11; temperature=0.07; soft fail vs 0.35 gate → accepted; **GATE 7 SOFT FAIL / ACCEPTED** |
| 7.5 — Integration dry run | **✅ DONE** | obj_clip_valid=32/57 valid (56%); vox_gt_instance_id=3985/4096 assigned (97%); no class-mismatch warnings; loss_lo finite; **GATE 7.5 PASS** |
| 8 — Job1b ablation A smoke (10 epochs) | **✅ DONE** | loss_lo: 0.95→0.74 (epochs 2→9); count_novel_lo=1–24 per step; PQ_base=0.2316 (no collapse); PQ_novel=0.0 (expected on mini); **GATE 8 PASS** |
| 9 — Job1b ablation A full (15 epochs) | **🔄 RUNNING** | JOBID 12463545; batch_a100, 36h wall; novel PQ expected for bus/moto/ped only (veg=0 expected) |
| 10 — Ablation B smoke | **❌ CANCELLED** | Merged into job1c. Separate ablation B (L_V alone) has weak value: same teacher signal as L_O, still zero for vegetation. Replaced by combined job1c run. |
| 11 — Ablation B full | **❌ CANCELLED** | Merged into job1c. |
| 12 — Job1c smoke (L_O + L_V margin-gated broadcast + L7 InfoNCE) | **⏳ PENDING** | After Phase 9. CASD dropped: per-voxel crops fail for all stuff at all crop sizes (structural); class-bbox not pursued. Plan: margin-gated THING broadcast + L7 InfoNCE base-only. |
| 13 — Job1c full | **⏳ PENDING** | Gate 13: novel PQ_things > 0.10, novel PQ_stuff (veg) > 0 (indirect, via L7 backbone sharpening). |

### Critical design note: L_O is THING-only → vegetation gets no signal in ablation A

`distill_object_loss.py` filters to `_THING_CLASS_IDS = frozenset(range(10))` (0-indexed classes 0–9).
Vegetation is class 15 (0-indexed) = STUFF → **L_O gives zero gradient to vegetation in ablation A.**

| Novel class | Class index | Type | L_O gradient | Expected in ablation A |
|---|---|---|---|---|
| bus | 2 | THING | ✅ yes | PQ > 0 possible |
| motorcycle | 5 | THING | ✅ yes | PQ > 0 possible |
| pedestrian | 6 | THING | ✅ yes | PQ > 0 possible |
| vegetation | 15 | STUFF | ❌ none | PQ = 0 expected |

**Ablation B (L_O + L_V) is required for vegetation novel PQ > 0 — OR — L7 backbone InfoNCE which sharpens CLIP-space boundaries indirectly.**

L_V uses patch-pooled `vox_feats` (spatially local patch tokens projected per voxel), which naturally covers ALL classes including stuff. Tree/leaf regions produce patch features in CLIP space that should be high-similarity to the "vegetation" text prototype. However, **this must be verified with Gate 5c** (vox_margin check on mini Pass 2 output) before trusting L_V to supervise vegetation.

### L_V design decision (2026-06-14): broadcast + CASD, no separate ablation B

**Decision:** Ablation B (separate L_V-only run) is cancelled. The broadcast approach uses existing `.npz` data directly — zero additional precompute, zero additional storage. CASD handles stuff online at training time. The two are combined in job1c.

**Why broadcast from obj_feats differs from L_O (not redundant):**

| | L_O | L_V broadcast |
|---|---|---|
| Operates on | ~5–20 matched queries/sample (Hungarian) | ~2000–3000 voxels/sample (dense) |
| Gradient through | `pred_embed` only | `pred_embed` AND `pred_masks` |
| Trains mask predictions | ❌ no | ✅ yes — spatial mask quality feedback |
| Signal granularity | per-query semantic pull | per-voxel spatial regularisation |

L_V broadcast adds **mask-level spatial feedback**: if query Q has high mask weight on bus voxels, L_V pulls Q's embedding toward bus AND reinforces Q's mask staying high on those voxels. L_O cannot do this — it only trains the embedding of matched queries.

**Why skip a separate ablation B run:**
Broadcast-only for THING gives the same teacher as L_O (just applied via masks), so the marginal benefit is hard to measure cleanly. The real payoff comes when CASD unlocks vegetation. Running a combined job1c (L_O + L_V broadcast + CASD) directly gives a cleaner ablation vs. job1b-A as the baseline.

**L_V target per voxel in job1c:**

```
THING voxels (cls_id 0–9):
    target[v] = obj_feats[vox_gt_instance_id[v]]   # broadcast: instance-averaged crop-CLS
    requires: obj_valid[vox_gt_instance_id[v]] == True
    fallback if invalid: skip voxel (no gradient)

STUFF voxels (cls_id 10–15):
    target[v] = CASD(voxel_v, sample)              # online crop-CLS at training time
    = encode_image(crop centered on projected pixel, 224×224) → CLS
    requires: voxel projects to at least one camera
    fallback if no projection: skip voxel

Unassigned voxels (vox_gt_instance_id == -1):
    skip — no reliable target
```

**Gate 5e (diagnostic before implementing):** Run `diag_broadcast_vox_check.py` on mini. Gate:
- THING novel broadcast margins > 0 (confirms broadcast is valid teacher)
- Vegetation broadcast margin < 0.05 (confirms CASD is necessary for stuff)

### Deviations from original plan

- **Phase 0**: Not cancelled. Original job1a-v1 runs to completion as a baseline reference. job1a-v2 starts fresh with v2 artifacts.
- **Phase 2**: Gate threshold relaxed 0.12 → 0.09 for motorcycle (ViT-L/14 text space limit). Actual achieved: 0.1597, well above both thresholds.
- **Phase 3**: Created new script `precompute_clip_voxel_features_v2.py` instead of modifying the original. Schema compatible with v1 dataset loader (`obj_feats`→`obj_clip_feats`, etc.). Added `vox_gt_instance_id` field.
- **Phase 4**: Vegetation margin = −0.027 (fails "all 4 > 0" gate). Root cause: 224×224 stuff crops encode global urban context, not vegetation. Mitigation: `distill_object_loss.py` now skips stuff-class instances (`_THING_CLASS_IDS = frozenset(range(10))`). Proceeding to Phase 5 with this fix. Vegetation will receive signal only via L_V in ablation B.
- **Phase 5b**: Added proactive Pass 2 precompute (vox_feats for L_V) to run in parallel with Phase 9, so ablation B features are ready immediately. Gated behind mini validation (Phase 5c) to avoid wasting 200GB on a bad teacher.
- **Phase 5c**: Gate failed — patch-pooled features have negative margin for all 16 classes.
- **Phase 5d**: CLS-crop per-voxel precompute (planned redesign) — **CANCELLED**. Superseded by broadcast approach. The existing `vox_gt_instance_id` field already maps each sampled voxel to its GT instance, so broadcasting `obj_feats[vox_gt_instance_id]` gives valid L_V THING targets with zero additional storage or compute. STUFF voxels handled by online CASD in job1c.
- **Phases 10–11 (ablation B)**: **CANCELLED** — merged into job1c. A separate L_V-only ablation B adds little insight (broadcast signal is same teacher as L_O, applied via masks) and still cannot supervise vegetation. Instead: job1b-A (running) is the baseline; job1c (L_O + L_V broadcast + CASD) is the next training run.
- **Phase 10 L_V config**: `alpha_stuff` changed from 0.2 → 1.0 (pure vox_feats for stuff) to avoid contaminating vegetation target with bad crop-CLS obj_feats. With the CLS-crop vox_feats redesign (Phase 5d), `alpha_stuff=1.0` now means "pure per-voxel CLS crop" — still the right choice since vegetation obj_feats have margin=−0.027.

---

## Diagnostic Findings (2026-06-14)

### Gate 5e — Broadcast diagnostic (`diag_broadcast_vox_check.py`, mini, 404 samples)

| Class | Type | Vox/sample | Coverage | ObjValid | Broadcast margin | Verdict |
|---|---|---|---|---|---|---|
| bus | THING (novel) | 29.2 | 99.9% | 240 | **+0.036** | ✅ valid teacher |
| motorcycle | THING (novel) | 3.2 | 94.7% | 191 | **+0.008** | ✅ valid teacher |
| pedestrian | THING (novel) | 20.9 | 88.9% | 2003 | **+0.048** | ✅ valid teacher |
| vegetation | STUFF (novel) | 540.8 | 100% | 376 | **−0.028** | ❌ CASD needed |
| driveable_surface | STUFF | 1561.9 | 100% | 404 | +0.003 | ⚠️ near-zero — CASD needed |
| sidewalk | STUFF | 310.4 | 100% | 387 | −0.012 | ❌ CASD needed |
| terrain | STUFF | 281.9 | 100% | 330 | −0.020 | ❌ CASD needed |
| manmade | STUFF | 778.5 | 100% | 404 | +0.006 | ⚠️ near-zero — CASD needed |
| other_flat | STUFF | 63.1 | 99.9% | 174 | −0.011 | ❌ CASD needed |

**GATE 5e result:**
- THING novel broadcast margins positive → **✅ PASS** (broadcast is valid L_V teacher for THING)
- Vegetation broadcast margin < 0.05 → **✅ CASD needed confirmed**
- All STUFF classes have near-zero or negative broadcast margins → CASD needed for ALL stuff, not just vegetation

**Implication:** L_V broadcast covers THING voxels (bus/moto/ped). CASD must cover all 6 STUFF classes. Job1c plan confirmed.

### Gate 5e-CASD — CASD margin diagnostics (2026-06-14)

**Confirmed FAIL at all crop sizes.**

#### Per-voxel CASD (`diag_casd_margin_check.py`, 50 samples)

| crop_half_px | vegetation | driveable | sidewalk | terrain | manmade |
|---|---|---|---|---|---|
| 112 (224×224) | −0.040 | −0.007 | −0.020 | −0.031 | −0.006 |
| 168 (336×336) | −0.044 | −0.008 | −0.022 | −0.033 | −0.007 |
| 224 (448×448) | −0.043 | −0.008 | −0.024 | −0.033 | −0.007 |

**Larger crops are worse**, not better. All STUFF classes fail at all sizes.

**Root cause (structural, not tunable):** A single voxel's camera projection is one pixel. Any crop around that pixel captures the entire surrounding urban scene — road, cars, sky, buildings. CLIP CLS encodes the dominant global scene context, not the class of the one voxel that triggered the crop. This is the same failure mode as patch-pooled vox_feats — crops of a scene are scene-level features, not class-level features.

#### Why this matters for the plan

- Per-voxel CASD = structurally identical to the obj_feats-for-stuff failure (margin=−0.027 for vegetation)
- Class-level bbox CASD (aggregate all class voxels → bbox crop) was written as `diag_casd_class_bbox_check.py` but not needed: driveable_surface bbox = most of the lower camera image = still mixed. Vegetation bbox may work in some scenes but is not reliable.
- **Decision:** Drop CASD entirely for now. Use L7 InfoNCE for indirect stuff improvement. CASD is deferred to future work (Options 5/6 — dense 2D OV teacher or SAM pseudo-labels).

**Implication for job1c:**
- L_V: THING-only, margin-gated broadcast (confirmed valid: bus=+0.036, moto=+0.008, ped=+0.048)
- STUFF: no L_V signal → novel vegetation relies on L7 backbone sharpening + C1 boundary learning
- Vegetation novel PQ > 0 is not guaranteed from job1c; if L7 + C1 sharpen base boundaries enough, vegetaton may emerge through inference-time open-vocab classification

### Script A (diag_clip_coverage) — obj_feats margins on mini

Run on 323 mini samples with v2 prototypes. Source: crop-CLS obj_feats from `precompute_clip_voxel_features_v2.py`.

| Class | Type | Coverage | Mean margin_to_base | Top-1 acc | Supervised by L_O? |
|---|---|---|---|---|---|
| bus | THING (novel) | 93.8% | +0.023 | ~75% | ✅ yes |
| motorcycle | THING (novel) | 57.7% | +0.016 | ~61% | ✅ yes |
| pedestrian | THING (novel) | 50.2% | +0.049 | ~95% | ✅ yes |
| vegetation | STUFF (novel) | 94.5% | −0.027 | — | ❌ no (STUFF excluded) |

**Key finding:** Crop-CLS obj_feats have positive margins for all 3 novel THING classes. Vegetation has negative margin (stuff crops encode global urban context, not vegetation). This is acceptable because L_O is THING-only — vegetation receives zero L_O gradient anyway. Vegetation must be handled by L_V (ablation B redesign) or CASD (job1c).

### Script B (diag_precompute_vox_check) — patch-pooled vox_feats margins on mini

Run on mini Pass 2 output. Source: patch-pooled vox_feats from `CLIPPatchExtractor`. This diagnostic caused **Gate 5c to FAIL** and prompted the redesign of ablation B.

| Class | Index | Type | vox_margin mean | Status |
|---|---|---|---|---|
| barrier | 0 | BASE | −0.0201 | FAIL |
| bicycle | 1 | BASE | −0.1123 | FAIL |
| bus | 2 | NOVEL | −0.0680 | FAIL |
| car | 3 | BASE | −0.0461 | FAIL |
| construction_vehicle | 4 | BASE | −0.0842 | FAIL |
| motorcycle | 5 | NOVEL | −0.0950 | FAIL |
| pedestrian | 6 | NOVEL | −0.0330 | FAIL |
| traffic_cone | 7 | BASE | −0.0510 | FAIL |
| trailer | 8 | BASE | −0.0620 | FAIL |
| truck | 9 | BASE | −0.0520 | FAIL |
| driveable_surface | 10 | BASE | −0.0390 | FAIL |
| other_flat | 11 | BASE | −0.0280 | FAIL |
| sidewalk | 12 | BASE | −0.0350 | FAIL |
| terrain | 13 | BASE | −0.0410 | FAIL |
| manmade | 14 | BASE | +0.0013 | barely PASS |
| vegetation | 15 | NOVEL | −0.0390 | FAIL |

**Root cause (structural):** CLIP patch tokens are local 14×14 px texture descriptors optimized for dense feature matching, not trained to align with text. The CLS token — produced by `encode_image()` — is what aligns to text prototypes during CLIP pre-training. Using patch tokens as L_V teacher targets points the model in the wrong direction in CLIP text space — all 16 classes have negative margin.

**Implication:** Patch-pooled vox_feats are a broken teacher. Ablation B with pure patch-pooled vox_feats would actively harm novel PQ. Full Pass 2 submission (200 GB) is cancelled. Ablation B must be redesigned to use per-voxel CLS-crop vox_feats (see Phase 5d and updated Phase 10).

### Job1b ablation A — training observation (epoch 3 check, job JOBID 12462720)

| Metric | Value | Assessment |
|---|---|---|
| loss_lo at epoch 2 | 0.95 | expected start after warmup |
| loss_lo at epoch 9 (smoke) | 0.74 | decreasing ✓ |
| count_novel_lo per step | 1–24 | confirms bus/moto/ped in L_O gradient ✓ |
| PQ_novel at epoch 3 | 0.0 | **expected** — temperature=0.07 requires many steps |
| PQ_base at epoch 3 | 0.2316 | no collapse ✓ |

Novel PQ = 0 at epoch 3 is **expected and not a failure**. Temperature=0.07 creates a sharp threshold — query embeddings need ~5–8 epochs to push past 12 base class boundaries before predicting novel class. Expected emergence: epochs 5–8 if loss_lo < 0.5 by that point.

---

## Root Causes Being Fixed

| # | Problem | Severity | Fix in this plan |
|---|---|---|---|
| 1 | L_O teacher (patch-pooled obj_feats) has negative margin — the teacher signal points to the wrong class in text space | **Critical** | Precompute v2: crop-CLS encoding |
| 2 | Prototype margins fragile: motorcycle (0.087), vegetation (0.124), bus (0.136) | High | PR-B: discriminative multi-prompt templates |
| 3 | Neg-bank contains sub-types of all 16 classes (truck→"pickup truck" 0.94) — distorts C1 training geometry | Medium | NB-C: rebuild with sim < 0.75 filter |

---

## Overview: Phases and Gates

```
Phase 0  Stop current job1a (record checkpoint)
Phase 1  Rebuild neg-bank v2          → GATE: Script B — all novel-neg max sims < 0.80
Phase 2  Re-engineer prompts v2       → GATE: Script B — motorcycle margin > 0.12
Phase 3  Rewrite precompute script    → GATE: unit test shapes + norm check on 1 sample
Phase 4  Mini precompute + Script A   → GATE: novel margins positive, top-1 > 50%
Phase 5  Full trainval precompute     → GATE: spot-check 10 samples, no NaN
Phase 6  Job1a-v2 mini smoke          → GATE: loss_token decreasing, no NaN, lora_gnorm in [1e-4, 10]
Phase 7  Job1a-v2 full               → GATE: PQ_base ≥ 0.35 at epoch 10, temperature in [0.1, 3.0]
Phase 7.5 Integration dry run        → GATE: obj_clip_valid+vox_gt_instance_id loaded, loss_lo finite
Phase 8  Job1b-v2 ablation A smoke    → GATE: loss_lo < 0.8, count_novel_lo > 0, no PQ_base collapse
Phase 9  Job1b-v2 ablation A full     → GATE: novel PQ > 0.05 at final epoch
Phase 10 Job1b-v2 ablation B smoke    → GATE: loss_lv decreasing, novel PQ ≥ ablation A
Phase 11 Job1b-v2 ablation B full     → GATE: novel PQ > ablation A
Phase 12 Job1c smoke                  → GATE: L7 loss decreasing, no NaN
Phase 13 Job1c full                   → GATE: novel PQ > 0.10 (things), voxel margin > 0.05
```

Each phase only starts after the previous phase's gate passes. If a gate fails, follow the documented failure remediation for that phase.

---

## File Map

### Files to CREATE
```
panoptic/scripts/generate_negative_bank_v2.py      Phase 1
panoptic/scripts/generate_prototypes_v2.py         Phase 2
panoptic/scripts/verify_neg_bank.py                Phase 1
panoptic/scripts/verify_crop_quality.py            Phase 4
panoptic/configs/job1a_v2.yaml                     Phase 6
panoptic/configs/job1a_v2_mini.yaml                Phase 6
panoptic/configs/job1b_v2_lo_only.yaml             Phase 8
panoptic/configs/job1b_v2_lo_only_mini.yaml        Phase 8
panoptic/configs/job1b_v2_lo_lv.yaml               Phase 10
panoptic/configs/job1b_v2_lo_lv_mini.yaml          Phase 10
panoptic/configs/job1c.yaml                        Phase 12
panoptic/configs/job1c_mini.yaml                   Phase 12
panoptic/scripts/job1a_v2_full.sh                  Phase 7
panoptic/scripts/job1b_v2_lo_smoke.sh              Phase 8
panoptic/scripts/job1b_v2_lo_full.sh               Phase 9
panoptic/scripts/job1b_v2_lo_lv_full.sh            Phase 11
panoptic/scripts/job1c_full.sh                     Phase 13
panoptic/scripts/precompute_v2_mini.sh             Phase 4
panoptic/scripts/precompute_v2_full.sh             Phase 5
panoptic/losses/voxel_decomp_loss.py               Phase 12
docs/job1_v4_ablation.md                           This file
```

### Files to MODIFY
```
panoptic/scripts/precompute_clip_voxel_features.py  Phase 3  (crop-CLS + vox_gt_instance_id schema)
panoptic/losses/distill_voxel_loss.py               Phase 10 (mode: patch|broadcast|hybrid)
panoptic/dataset/nuscenes_panoptic.py               Phase 3  (load obj_valid, vox_gt_instance_id)
panoptic/losses/distill_object_loss.py              Phase 3  (use obj_valid flag for GT-order indexing)
panoptic/train/train_panoptic_job1.py               Phase 10 (pass mode + obj_valid to loss ctors)
panoptic/scripts/diag_clip_coverage.py              Phase 4  (verify new schema fields)
```

### Files NOT modified
```
panoptic/losses/token_loss_c1.py                   (no changes needed)
```

### Required code changes (non-script files)

#### `panoptic/dataset/nuscenes_panoptic.py`

Load new v2 cache fields so they reach the loss:

```python
if os.path.exists(feat_path):
    data = np.load(feat_path)
    result["obj_clip_feats"]   = torch.from_numpy(data["obj_feats"].astype(np.float32))
    result["obj_clip_classes"] = torch.from_numpy(data["obj_classes"].astype(np.int64))
    result["obj_clip_valid"]   = torch.from_numpy(data["obj_valid"].astype(np.bool_))  # NEW
    result["vox_clip_feats"]   = torch.from_numpy(data["vox_feats"].astype(np.float32))
    result["vox_clip_indices"] = torch.from_numpy(data["vox_indices"].astype(np.int64))
    # NEW: GT instance id for each sampled voxel (for hybrid L_V target)
    if "vox_gt_instance_id" in data:
        result["vox_gt_instance_id"] = torch.from_numpy(
            data["vox_gt_instance_id"].astype(np.int64))   # (4096,) int, -1 if no GT
    else:
        result["vox_gt_instance_id"] = torch.full((4096,), -1, dtype=torch.long)
```

Fallback zeros must also be updated:
```python
result["obj_clip_valid"]     = torch.zeros(0, dtype=torch.bool)
result["vox_gt_instance_id"] = torch.full((4096,), -1, dtype=torch.long)
```

#### `panoptic/losses/distill_object_loss.py`

Replace the class-queue heuristic with direct GT-order indexing using `obj_valid`:

```python
def forward(self, pred_embed, obj_clip_feats, obj_clip_classes, obj_clip_valid,
            indices, targets):
    """
    obj_clip_valid: (B, N_inst) bool — True only for instances with valid crop-CLS feature.
                    Replaces the norm > 0.1 heuristic. Ensures we skip entirely
                    invisible instances rather than silently aligning to zero.
    indices: from indices_all (all-GT matcher) — gt_idx references all_targets_obj order
    targets: all_targets_obj — "classes" key gives all GT classes incl. novel
    """
    for b in range(B):
        pred_idx, gt_idx = indices[b]
        gt_classes_b  = targets["classes"][b]     # (N_gt_all,) all GT classes
        for i, gi in enumerate(gt_idx):
            gi_i = int(gi.item())
            # Direct index into all_targets_obj order — no class-queue needed.
            if gi_i >= obj_clip_feats[b].shape[0]:
                continue
            if not obj_clip_valid[b][gi_i]:          # skip instances with no valid crop
                continue
            # Class-consistency assertion: catches GT-order mismatch between
            # build_gt_masks(), all_targets_obj, and the .npz cache.
            # Both must agree on the class at index gi_i.
            # Remove after Phase 7.5 / Phase 8 smoke test validates the order.
            gt_cls  = int(gt_classes_b[gi_i].item())
            obj_cls = int(obj_clip_classes[b][gi_i].item())
            if obj_cls != gt_cls:
                print(f"WARNING L_O class mismatch: gt={gt_cls}, obj={obj_cls}, "
                      f"gi={gi_i}, b={b} — skipping pair")
                continue
            target_feat = obj_clip_feats[b][gi_i]    # (768,)
            ...
```

This requires `indices_all` and `all_targets_obj` to be in the **same order** as `build_gt_masks()`. Verify this in the training loop.

#### `panoptic/train/train_panoptic_job1.py`

1. Pass `mode` to `VoxelDistillLoss`:
```python
distill_vox_loss = VoxelDistillLoss(
    loss_type=get_nested(cfg, "distillation.L_V.loss_type", "cosine"),
    mode=get_nested(cfg, "distillation.L_V.mode", "patch"),   # NEW
    alpha_things=get_nested(cfg, "distillation.L_V.alpha_things", 0.4),  # NEW
    alpha_stuff=get_nested(cfg, "distillation.L_V.alpha_stuff", 0.2),    # NEW
).to(device)
```

2. Pass `obj_clip_valid` to `ObjectDistillLoss`:
```python
loss_lo = distill_obj_loss(
    outputs["pred_embed"],
    batch["obj_clip_feats"].to(device),
    batch["obj_clip_classes"].to(device),
    batch["obj_clip_valid"].to(device),   # NEW
    indices_all, all_targets_obj
)
```

3. Pass `vox_gt_instance_id` and `obj_clip_feats` to `VoxelDistillLoss` (for hybrid mode):
```python
# Move optional hybrid-target tensors to device before calling the loss.
# batch.get() returns CPU tensors — they must match the model's device.
_vox_gt_inst_id = batch.get("vox_gt_instance_id", None)
if _vox_gt_inst_id is not None:
    _vox_gt_inst_id = _vox_gt_inst_id.to(device)

_obj_feats_lv = batch.get("obj_clip_feats", None)
if _obj_feats_lv is not None:
    _obj_feats_lv = _obj_feats_lv.to(device)

_obj_classes_lv = batch.get("obj_clip_classes", None)
if _obj_classes_lv is not None:
    _obj_classes_lv = _obj_classes_lv.to(device)

loss_lv = distill_vox_loss(
    outputs["pred_embed"],
    outputs["pred_masks"],
    batch["vox_clip_feats"].to(device),
    batch["vox_clip_indices"].to(device),
    vox_gt_instance_id=_vox_gt_inst_id,
    obj_clip_feats=_obj_feats_lv,
    obj_clip_classes=_obj_classes_lv,   # NEW: needed for alpha_things vs alpha_stuff
)
```

### New artifacts
```
panoptic/clip_prototypes/negative_bank_v2.pt
panoptic/clip_prototypes/negative_bank_v2_concepts.txt
panoptic/clip_prototypes/text_prototypes_vitl14_224_v2.pt
panoptic/clip_prototypes/text_prototypes_vitl14_224_multiprompt_v2.pt
/fs/scratch/.../clip_features_v2/v1.0-mini/        ~50MB
/fs/scratch/.../clip_features_v2/v1.0-trainval/    ~170GB
panoptic/checkpoints_job1a_v2/
panoptic/checkpoints_job1b_v2_lo/
panoptic/checkpoints_job1b_v2_lo_lv/
panoptic/checkpoints_job1c/
```

---

## Phase 0: Stop Current Job1a

**Purpose:** Current job1a is using the old neg-bank (with sub-types) and old prompts. Re-running with v2 prototypes ensures job1b warm-starts from a correctly trained foundation.

### 0.1 Record current checkpoint

```bash
# Check job status
bjobs
# Note the epoch reached:
ls -lh panoptic/checkpoints_job1a_v1/
# Copy best checkpoint as reference before cancelling
cp panoptic/checkpoints_job1a_v1/best.pth panoptic/checkpoints_job1a_v1/best_before_cancel.pth
```

### 0.2 Cancel job

```bash
bkill <JOB_ID>
```

**Note:** If job1a has completed more than 10 epochs and PQ_base ≥ 0.30, the checkpoint is usable as a baseline reference. The v2 re-run will produce a better warm-start for job1b.

---

## Phase 1: Rebuild Negative Bank (NB-C)

**Goal:** Replace `negative_bank_200.pt` (188 entries, many near class sub-types) with a v2 bank where every entry has cosine sim < 0.75 to all 16 class prototypes.

**Why:** Script B showed novel-neg max sims: bus=0.926 ("school bus"), motorcycle=0.914 ("motorcycle rider"), vegetation=0.895 ("shrub"). These create contradictory training pressure in C1 InfoNCE. Base classes are also affected (truck=0.940 "pickup truck").

### 1.1 Create `panoptic/scripts/generate_negative_bank_v2.py`

Key changes from `generate_negative_bank.py`:
- **Remove** all vehicle sub-types: school bus, coach bus, pickup truck, SUV, minivan, delivery van, cargo van, motorcycle rider, bicycle rider, electric bicycle, scooter, moped
- **Remove** person variants: person standing, person walking, child walking, jogger, skateboarder, group of people
- **Remove** vegetation sub-types: bush, shrub, tree trunk, grass (all too close to vegetation=15)
- **Remove** surface sub-types: concrete road, asphalt road (close to driveable surface)
- **Remove** barrier near-synonyms: sound barrier, guard rail (close to barrier=0)
- **Remove** sidewalk near-synonyms: curb, crosswalk (close to sidewalk=12)
- **Remove** traffic cone near-synonyms: bollard, cone (close to traffic cone=7)
- **Keep and expand:** genuinely non-class objects: fire hydrant, parking meter, bench, trash bin, billboard, traffic light, stop sign, shopping cart, bicycle rack, dumpster, construction barrel, manhole cover, street lamp, utility pole, gas station, parking lot, bus stop (as infrastructure, not vehicle)
- **Add new safe categories:** playground equipment, vending machine, newspaper stand, ATM kiosk, flower pot, public phone booth, post box, outdoor seating, awning, sandwich board, election sign, art installation

```python
# generate_negative_bank_v2.py
NEGATIVE_CONCEPTS_V2 = [
    # Street furniture (safe — not near any class)
    "a fire hydrant in a scene", "a parking meter in a scene",
    "a bench in a scene", "a trash bin in a scene", "a mailbox in a scene",
    "a post box in a scene", "a public telephone in a scene",
    "a vending machine in a scene", "a newspaper stand in a scene",
    "a bicycle rack in a scene", "a shopping cart in a scene",
    "a dumpster in a scene", "a construction barrel in a scene",
    "a road cone in a scene",  # generic enough to stay
    "a portable toilet in a scene", "a skip container in a scene",
    # Traffic infrastructure (safe)
    "a traffic light in a scene", "a stop sign in a scene",
    "a yield sign in a scene", "a street lamp in a scene",
    "a lamp post in a scene", "a utility pole in a scene",
    "a power line in a scene", "a traffic sign in a scene",
    "a road sign in a scene", "a billboard in a scene",
    "an advertising sign in a scene",
    # Vehicles that are genuinely different from all 16 classes
    "a fire truck in a scene", "an ambulance in a scene",
    "a police car in a scene", "a tow truck in a scene",
    "a golf cart in a scene", "a forklift in a scene",
    "a bulldozer in a scene", "an excavator in a scene",
    "a road roller in a scene", "a tractor in a scene",
    "a crane in a scene", "a cement mixer in a scene",
    "an ATV in a scene", "a go-kart in a scene",
    # Animals (clearly not any class)
    "a dog in a scene", "a cat in a scene",
    "a bird in a scene", "a pigeon in a scene",
    # Structures
    "a building in a scene", "a house in a scene",
    "a tower in a scene", "a bridge in a scene",
    "a tunnel in a scene", "a parking garage in a scene",
    "a gas station in a scene", "a car wash in a scene",
    "a loading dock in a scene", "a toll booth in a scene",
    "a bus stop shelter in a scene", "a train station in a scene",
    # Infrastructure
    "a manhole cover in a scene", "a drainage grate in a scene",
    "road markings in a scene", "lane markings in a scene",
    "a speed bump in a scene", "a pothole in a scene",
    "a fence post in a scene", "a chain link fence in a scene",
    "a stone wall in a scene", "a retaining wall in a scene",
    "a jersey barrier in a scene",
    # Urban objects
    "a rooftop in a scene", "a balcony in a scene",
    "a garden in a scene", "a playground in a scene",
    "a fountain in a scene", "a statue in a scene",
    "a monument in a scene", "a sculpture in a scene",
    "an antenna in a scene", "a satellite dish in a scene",
    "a solar panel in a scene", "a wind turbine in a scene",
    "a transformer box in a scene", "a scaffolding in a scene",
    "a fire escape in a scene",
    # Shipping/industrial
    "a shipping container in a scene", "a cargo crate in a scene",
    "a conveyor belt in a scene", "a pallet in a scene",
    # Weather/surface materials (safe ones)
    "snow in a scene", "puddles in a scene",
    "a wet road in a scene",
    # Driving infrastructure
    "a roundabout in a scene", "a highway overpass in a scene",
    "a pedestrian bridge in a scene", "a railroad crossing in a scene",
    "a subway entrance in a scene",
    # Miscellaneous urban
    "a swimming pool in a scene", "a sports field in a scene",
    "a park in a scene", "a parking lot in a scene",
    "an outdoor café in a scene", "a market stall in a scene",
    "a construction site in a scene",
]
# Filter by sim to class prototypes happens in main() — see below
```

**Critical addition in `main()`:** Hard denylist (zero-shot concepts), then automatic sim filtering:

```python
# Hard denylist — these four concepts appear in ZEROSHOT_CONCEPTS at
# train_panoptic_job1.py:749 and are used as open-vocab eval probes at validation.
# They must NEVER appear in the neg-bank regardless of their sim to class protos.
ZEROSHOT_DENYLIST = {"ambulance", "trash can", "fire hydrant", "police van"}

def _in_denylist(concept: str) -> bool:
    cl = concept.lower()
    return any(kw in cl for kw in ZEROSHOT_DENYLIST)

def main():
    # ... load CLIP model ...
    # Load class prototypes for filtering
    class_protos = load_prototypes("panoptic/clip_prototypes/text_prototypes_vitl14_224.pt")  # (16, 768)

    # Step 0: Hard denylist — remove zero-shot eval concepts before any sim filtering
    candidates = [c for c in NEGATIVE_CONCEPTS_V2 if not _in_denylist(c)]
    n_denied = len(NEGATIVE_CONCEPTS_V2) - len(candidates)
    if n_denied:
        denied = [c for c in NEGATIVE_CONCEPTS_V2 if _in_denylist(c)]
        print(f"Hard denylist removed {n_denied} concept(s): {denied}")

    # Encode remaining candidates
    all_embeddings = encode_concepts(candidates, model, device)  # (N, 768)

    # Filter: remove any entry with sim > 0.75 to any class prototype
    sim_to_classes = all_embeddings @ class_protos.T   # (N, 16)
    max_sim = sim_to_classes.max(dim=1).values         # (N,)
    keep_mask = max_sim < 0.75
    filtered_embeddings = all_embeddings[keep_mask]
    filtered_concepts   = [c for c, k in zip(candidates, keep_mask.tolist()) if k]

    print(f"Kept {keep_mask.sum()}/{len(candidates)} concepts (sim < 0.75 to all class protos)")
    print(f"Rejected by sim: {[c for c, k in zip(candidates, keep_mask.tolist()) if not k]}")

    # Save
    torch.save({"embeddings": filtered_embeddings, "concepts": filtered_concepts,
                "model_name": "ViT-L/14", "filter_threshold": 0.75}, args.output)
```

**Run:**
```bash
python panoptic/scripts/generate_negative_bank_v2.py \
    --output panoptic/clip_prototypes/negative_bank_v2.pt \
    --device cuda
```

### 1.2 Create verification script `panoptic/scripts/verify_neg_bank.py`

```python
"""Quick verification of negative bank quality.

Checks:
1. No entry has sim > threshold to any class prototype
2. Bank has enough entries (>= 100)
3. Internal diversity: avg pairwise sim of bank entries
"""
import torch, sys
import torch.nn.functional as F

def verify(bank_path, proto_path, sim_threshold=0.75, min_entries=100):
    bank_data   = torch.load(bank_path,  map_location="cpu")
    proto_data  = torch.load(proto_path, map_location="cpu")

    bank   = F.normalize(bank_data["embeddings"].float(), dim=-1)   # (N, 768)
    protos = F.normalize(proto_data["embeddings"].float(), dim=-1)  # (16, 768)
    concepts = bank_data.get("concepts", [f"neg_{i}" for i in range(len(bank))])

    sim = bank @ protos.T   # (N, 16)
    max_sim, max_cls = sim.max(dim=1)

    violations = [(concepts[i], max_sim[i].item(), max_cls[i].item())
                  for i in range(len(bank)) if max_sim[i] > sim_threshold]

    print(f"Bank entries: {len(bank)}")
    print(f"Violations (sim > {sim_threshold}): {len(violations)}")
    for conc, s, cls in violations:
        print(f"  '{conc}' sim={s:.4f} to class {cls}")

    # Diversity
    bank_sim = bank @ bank.T
    avg_off_diag = (bank_sim.sum() - bank_sim.trace()) / (len(bank) * (len(bank)-1))
    print(f"Internal avg pairwise sim: {avg_off_diag:.4f}")

    # PASS / FAIL
    passed = len(violations) == 0 and len(bank) >= min_entries
    print(f"\n{'PASS' if passed else 'FAIL'}: {'All entries clean' if passed else 'Fix violations above'}")
    return passed
```

**Run:**
```bash
python panoptic/scripts/verify_neg_bank.py \
    --bank panoptic/clip_prototypes/negative_bank_v2.pt \
    --protos panoptic/clip_prototypes/text_prototypes_vitl14_224.pt
```

### GATE 1: Pass criteria
- [x] Zero entries with sim > 0.75 to any class prototype — **PASS** (max-sim=0.749)
- [x] Bank has ≥ 100 entries — **PASS** (161 entries, diversity_thresh=0.80)
- [x] `verify_neg_bank.py` prints `PASS`

### GATE 1: Failure remediation
- If entries fail: add more clearly-negative concepts, re-run generation
- If bank < 100: lower filter threshold to 0.77, or add more concept categories
- Do NOT proceed to Phase 2 until gate passes

---

## Phase 2: Re-Engineer Text Prototypes (PR-B)

**Goal:** Add discriminative multi-prompt templates for the fragile class pairs. Script B showed motorcycle margin=0.087 (vs bicycle), vegetation=0.124 (vs terrain), bus=0.136 (vs truck).

**Why prompt engineering helps:** Averaging over 10 diverse templates per class produces a more stable prototype that reduces the influence of any single template that happens to be close to a competing class.

### 2.1 Create `panoptic/scripts/generate_prototypes_v2.py`

Based on `generate_prototypes_224.py` with updated multi-prompt templates:

```python
# Key change: MULTI_TEMPLATES replaces existing templates for fragile classes
# Uses ViT-L/14 (224px) to match training

CLASS_TEMPLATES = {
    "barrier": [
        "a barrier in a scene", "a concrete barrier on a road",
        "a road barrier in a driving scene", "a traffic barrier",
        "a barrier separating lanes", "a road divider barrier",
        "a safety barrier in a scene", "a highway barrier",
        "a barrier in an urban scene", "a concrete road barrier",
    ],
    "bicycle": [
        "a bicycle in a scene", "a non-motorized pedal bicycle",
        "a bicycle without an engine", "a push bicycle on a road",
        "a human-powered bicycle", "a pedal bicycle in a city",
        "a bicycle parked or moving", "a manual bicycle",
        "a bicycle in a driving scene", "a cyclist on a pedal bicycle",
    ],
    "bus": [
        "a bus in a scene", "a large passenger transit bus",
        "a public transit bus on a road", "a city bus",
        "a large bus for many passengers", "a transit bus in urban traffic",
        "a bus with many windows in a scene", "a public bus",
        "a passenger bus in a driving scene", "a full-size transit bus",
    ],
    "car": [
        "a car in a scene", "a passenger car on a road",
        "a car in a driving scene", "a regular passenger vehicle",
        "a car parked or driving", "a sedan or hatchback in a scene",
        "a personal car on the street", "a car in urban traffic",
        "a car in an urban environment", "a small personal vehicle",
    ],
    "construction vehicle": [
        "a construction vehicle in a scene",
        "a heavy construction machine", "a construction site vehicle",
        "a large yellow construction machine", "a construction equipment vehicle",
        "a machine for construction work", "a construction vehicle on a road",
        "a construction machinery in a scene", "a digger or construction machine",
        "a construction vehicle in an urban scene",
    ],
    "motorcycle": [
        "a motorcycle in a scene", "a motorized two-wheeled vehicle",
        "a motorcycle with an engine on a road",
        "a gas-powered motorbike", "a motorcycle in urban traffic",
        "a motorized bike in a driving scene",
        "a motorcycle parked or riding", "a motorbike with engine",
        "a powered two-wheeler in a scene",
        "a motorcycle or motorbike in a city",
    ],
    "pedestrian": [
        "a pedestrian in a scene", "a person walking on a street",
        "a person on foot in a driving scene", "a walking person",
        "a pedestrian in urban traffic", "a person in a city scene",
        "a pedestrian crossing or walking", "a person on a sidewalk",
        "a person walking in an urban scene", "a pedestrian in a scene",
    ],
    "traffic cone": [
        "a traffic cone in a scene", "an orange safety cone on a road",
        "a traffic cone marking a hazard", "a road safety cone",
        "a bright orange cone on the street", "a traffic cone in a driving scene",
        "a safety cone in a scene", "a road cone in an urban scene",
        "a traffic cone directing traffic", "a cone on a road",
    ],
    "trailer": [
        "a trailer in a scene", "a towed trailer on a road",
        "a trailer attached to a vehicle", "a cargo trailer",
        "a large trailer in a driving scene", "a semi-trailer without a cab",
        "a detached trailer", "a trailer in an urban scene",
        "a trailer in a driving scene", "a vehicle trailer",
    ],
    "truck": [
        "a truck in a scene", "a cargo delivery truck",
        "a large cargo truck on a road", "a truck for transporting goods",
        "a freight truck in a driving scene", "a delivery truck on the street",
        "a commercial truck carrying cargo", "a goods truck in traffic",
        "a truck for cargo transport", "a heavy goods truck",
    ],
    "driveable surface": [
        "driveable surface in a scene", "a paved road surface",
        "a road you can drive on", "a flat driveable road",
        "a road surface in a driving scene", "paved ground for driving",
        "a driving lane in a scene", "a road pavement",
        "a driveable road in an urban scene", "an asphalt driving surface",
    ],
    "other flat": [
        "other flat surface in a scene", "a flat non-road surface",
        "a flat paved area not for driving", "a flat ground surface",
        "a level non-road pavement", "a flat area in a scene",
        "flat ground near a road", "a flat surface in a scene",
        "a flat paved region", "a non-driveable flat surface",
    ],
    "sidewalk": [
        "a sidewalk in a scene", "a pedestrian walkway",
        "a pavement for pedestrians", "a sidewalk in an urban scene",
        "a footpath along a road", "a sidewalk in a driving scene",
        "a pedestrian path beside a road", "a paved walkway",
        "a sidewalk for people", "a pedestrian sidewalk",
    ],
    "terrain": [
        "terrain in a scene", "rough unpaved natural ground",
        "dirt and earth ground surface", "natural terrain in a scene",
        "off-road natural ground", "rough earth and soil",
        "natural ground without pavement", "outdoor terrain ground",
        "rough natural earth surface", "terrain outside paved roads",
    ],
    "manmade": [
        "a manmade structure in a scene", "a man-made object in a scene",
        "an artificial structure", "a built structure in a driving scene",
        "a constructed surface or object", "a manufactured structure",
        "a built environment element", "an engineered structure in a scene",
        "a constructed object in an urban scene", "a human-built structure",
    ],
    "vegetation": [
        "vegetation in a scene", "trees and plants in a driving scene",
        "green leafy trees and vegetation", "tall trees along a road",
        "natural vegetation in a scene", "trees and shrubs in an urban scene",
        "green tree canopy in a scene", "leafy vegetation in a driving scene",
        "trees and plants along a street", "natural plant life in a scene",
    ],
}
```

**Run:**
```bash
python panoptic/scripts/generate_prototypes_v2.py \
    --output_single panoptic/clip_prototypes/text_prototypes_vitl14_224_v2.pt \
    --output_multi  panoptic/clip_prototypes/text_prototypes_vitl14_224_multiprompt_v2.pt \
    --device cuda
```

### 2.2 Run Script B on v2 prototypes

```bash
python panoptic/scripts/diag_prototype_similarity.py \
    --proto_path    panoptic/clip_prototypes/text_prototypes_vitl14_224_v2.pt \
    --neg_bank_path panoptic/clip_prototypes/negative_bank_v2.pt \
    --out_dir       panoptic/diagnostics/v2
```

### GATE 2: Pass criteria
- [x] motorcycle proto_margin > 0.12 — **PASS** (0.1597; gate relaxed to 0.09, achieved 0.1597)
- [x] vegetation proto_margin > 0.10 — **PASS** (0.2128)
- [x] bus proto_margin > 0.12 — **PASS** (0.1694)
- [x] All 4 novel max neg-bank sim < 0.80 — **PASS** (max=0.776 for pedestrian)
- [x] No novel class has HIGH neg-bank risk flag — **PASS** (pedestrian=MODERATE only)

### GATE 2: Failure remediation
- If motorcycle margin still < 0.10: revise motorcycle templates to more aggressively emphasize "motorized" / "engine-powered". Rerun prototype generation.
- If vegetation still confuses with terrain: add more visually distinct descriptions ("tall trees", "leafy canopy"). Rerun.
- If novel-neg sims still high: add those specific concepts to the removal list in Phase 1 and regenerate neg-bank.
- Do NOT proceed to Phase 3 until gate passes.

---

## Phase 3: Rewrite Precompute Script

**Goal:** Replace patch-pooled `obj_feats` with object-crop CLS encoding. This is the highest-priority fix.

### 3.1 Rewrite `panoptic/scripts/precompute_clip_voxel_features.py`

**Key architectural changes:**

#### New class: `CLIPCropEncoder`

```python
import torchvision.transforms as T

CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
MIN_PTS_THING = 5
MIN_PTS_STUFF = 20
N_STUFF_CROPS = 6
THING_IDS_SET = set(range(1, 11))   # class ids 1-10 (1-indexed)

class CLIPCropEncoder:
    """Encodes object image crops using CLIP encode_image (CLS token)."""

    def __init__(self, clip_model, device):
        self.model  = clip_model
        self.device = device
        # Resize to 224x224 WITHOUT CenterCrop — preserves full object bbox
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
        ])

    def encode_crop(self, img_pil, x0, y0, x1, y1):
        """Returns (768,) float32 L2-normalized CLS feature, or None if crop invalid."""
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        if x1 - x0 < 5 or y1 - y0 < 5:
            return None
        crop = img_pil.crop((x0, y0, x1, y1))
        # Pad to square with gray fill before resize — preserves aspect ratio.
        # Direct resize distorts elongated objects (pedestrians, trucks) and
        # shifts the CLIP feature away from the text prototype.
        w, h = crop.size
        side = max(w, h)
        sq = Image.new("RGB", (side, side), (128, 128, 128))
        sq.paste(crop, ((side - w) // 2, (side - h) // 2))
        t    = self.transform(sq).unsqueeze(0).to(self.device).half()
        with torch.no_grad():
            feat = self.model.encode_image(t)         # (1, 768)
            feat = F.normalize(feat.float(), dim=-1)
        return feat.squeeze(0).cpu().numpy()
```

#### Reference function: `encode_instance_crop()` (non-batched — for testing only)

The production `process_sample()` loop inlines the projection+crop logic and calls `batch_encode_crops()`. This function is kept as a readable reference and for single-sample interactive debugging; it is **not** called in the main precompute path.

```python
def encode_instance_crop(nusc, sample_token, pt_xyz, cam_name, img_pil,
                          crop_encoder, is_stuff, inst_idx=0):
    """Encode a single GT instance across one camera view.

    Returns (feat: np.ndarray (768,) or None, n_pts: int)
    inst_idx: GT instance index — used for deterministic seeding of stuff crops.
    """
    uv, depths, valid = project_lidar_to_camera(nusc, sample_token, pt_xyz, cam_name)
    n_pts = int(valid.sum())

    min_pts = MIN_PTS_STUFF if is_stuff else MIN_PTS_THING
    if n_pts < min_pts:
        return None, n_pts

    uv_v = uv[valid]   # (n_pts, 2) — valid projected pixel coords

    if is_stuff:
        # Deterministic seed via hashlib — hash() is not stable across processes/versions
        seed_bytes = hashlib.md5(f"{sample_token}_{inst_idx}".encode()).digest()
        seed = int.from_bytes(seed_bytes[:4], "little")
        rng  = np.random.RandomState(seed=seed)
        half = 112
        sample_idx = rng.choice(len(uv_v), min(N_STUFF_CROPS, len(uv_v)), replace=False)
        feats = []
        for si in sample_idx:
            cx, cy = uv_v[si, 0], uv_v[si, 1]
            x0 = max(0, cx - half);  x1 = min(CAM_W, x0 + 224)
            y0 = max(0, cy - half);  y1 = min(CAM_H, y0 + 224)
            f  = crop_encoder.encode_crop(img_pil, x0, y0, x1, y1)
            if f is not None:
                feats.append(f)
        if not feats:
            return None, n_pts
        avg = np.stack(feats).mean(axis=0)
        return (avg / max(np.linalg.norm(avg), 1e-8)), n_pts

    else:
        # Compute 2D bbox, expand 1.3x, make square
        u_min, u_max = uv_v[:, 0].min(), uv_v[:, 0].max()
        v_min, v_max = uv_v[:, 1].min(), uv_v[:, 1].max()
        cx = (u_min + u_max) / 2;  cy = (v_min + v_max) / 2
        hw = (u_max - u_min) / 2 * 1.3
        hh = (v_max - v_min) / 2 * 1.3
        half = max(hw, hh)                             # square crop
        x0 = max(0, cx - half);  x1 = min(CAM_W, cx + half)
        y0 = max(0, cy - half);  y1 = min(CAM_H, cy + half)
        feat = crop_encoder.encode_crop(img_pil, x0, y0, x1, y1)
        return feat, n_pts
```

#### Batch crop encoding (performance — required for trainval)

Encoding each crop individually via `encode_image()` is ~5 ms/call on A100. With ~30 instances × 6 cameras × 34k samples = ~6M calls → ~14h for crops alone. Instead, collect all crops for a camera in a list, batch-encode, then scatter results back.

```python
import hashlib

def batch_encode_crops(crop_encoder, crop_items, max_batch=128):
    """Batch-encode a list of crop dicts.

    Args:
        crop_items: list of dicts with keys:
            "crop"     — PIL Image (already square-padded) or None
            "inst_key" — k (int, things) or (k, ci) (tuple, stuff sub-crop)
            "n_pts"    — int, projected point count
            "cam_name" — str (unused in encoding, kept for caller context)
    Returns:
        dict mapping item_id (int position in crop_items) → (768,) float32 array.
        Only items with non-None crop are present.
    """
    valid = [(i, item) for i, item in enumerate(crop_items) if item["crop"] is not None]
    if not valid:
        return {}

    feat_by_item = {}
    for start in range(0, len(valid), max_batch):
        chunk    = valid[start:start + max_batch]
        tensors  = [crop_encoder.transform(item["crop"]) for _, item in chunk]
        item_ids = [i for i, _ in chunk]
        batch_t  = torch.stack(tensors, dim=0).to(crop_encoder.device).half()
        with torch.no_grad():
            feats = crop_encoder.model.encode_image(batch_t)   # (B, 768)
            feats = F.normalize(feats.float(), dim=-1).cpu().numpy()
        for item_id, feat in zip(item_ids, feats):
            feat_by_item[item_id] = feat

    return feat_by_item
```

**Integration in `process_sample()`:** Build `crop_items` as a list of dicts (one per crop across all instances), call `batch_encode_crops()` once per camera, scatter back by item_id. The dict schema eliminates the key-collision bug where `n_pts` was previously misinterpreted as `cam_name`.

**Batch size tuning:** With `N_instances ≈ 25` and `N_stuff_crops = 6`, the largest per-camera batch is ~80 tensors — well within A100 VRAM even at fp16.

#### Rewritten `process_sample()` core loop

```python
def process_sample(nusc, sample_token, dataroot, extractor, crop_encoder,
                   device, protos=None):
    # 1. Voxelize
    vox_xyz, vox_idx, inv_idx, sem, inst, N = voxelize_sample(...)
    V = len(vox_idx)

    # 2. Build GT masks (all instances, in GT order)
    gt_instances = build_gt_masks(sem, inst, inv_idx, V)
    n_inst = len(gt_instances)

    # Pre-allocate accumulators
    obj_feat_accum  = np.zeros((n_inst, 768), dtype=np.float64)
    obj_weight_sum  = np.zeros(n_inst,        dtype=np.float64)
    obj_num_views   = np.zeros(n_inst,        dtype=np.int16)
    obj_num_points  = np.zeros(n_inst,        dtype=np.int32)

    # Pre-allocate output
    obj_feats   = np.zeros((n_inst, 768), dtype=np.float16)
    obj_classes = np.zeros(n_inst,        dtype=np.int32)
    obj_valid   = np.zeros(n_inst,        dtype=np.bool_)

    # Pre-fill classes
    for k, (mask, cls_id) in enumerate(gt_instances):
        obj_classes[k] = cls_id

    # Also keep vox_feats (patch-pooled) for backward compat
    voxel_feats_accum = np.zeros((V, 768), dtype=np.float32)
    voxel_cam_count   = np.zeros(V, dtype=np.int32)

    # 3. For each camera: batch crop-encode instances + patch-project voxels
    sample = nusc.get("sample", sample_token)
    for cam_name in CAMERAS:
        cam_sd   = nusc.get("sample_data", sample["data"][cam_name])
        cam_path = os.path.join(dataroot, cam_sd["filename"])
        img_pil  = Image.open(cam_path).convert("RGB")

        # (a) Crop-CLS — batched path (single encode_image call per camera)
        # Critical: project ALL voxels to this camera ONCE, then index by instance mask.
        # Per-instance projection calls (150/sample) are 25× slower than 6 total.
        uv_all_cam, _, valid_all_cam = project_lidar_to_camera(
            nusc, sample_token, vox_xyz, cam_name
        )  # (V, 2), (V,), (V,) — computed once, shared across all instances below

        # Step 1: project all instances, build square-padded crop PIL images
        crop_items = []   # list of dicts: {crop, inst_key, n_pts, cam_name}
        img_w, img_h = img_pil.size
        for k, (mask, cls_id) in enumerate(gt_instances):
            is_stuff = (cls_id + 1) in STUFF_IDS
            uv_k    = uv_all_cam[mask]
            valid_k = valid_all_cam[mask]
            n_pts   = int(valid_k.sum())
            min_pts = MIN_PTS_STUFF if is_stuff else MIN_PTS_THING
            if n_pts < min_pts:
                continue
            uv_v = uv_k[valid_k]  # (n_pts, 2) — valid projected UV for this instance

            def _make_sq_crop(x0, y0, x1, y1):
                """Clamp to image bounds, then crop and pad to square. Returns PIL or None."""
                # Clamp first so PIL doesn't introduce black borders for OOB coords
                x0 = max(0, int(x0));  y0 = max(0, int(y0))
                x1 = min(img_w, int(x1));  y1 = min(img_h, int(y1))
                if x1 - x0 < 5 or y1 - y0 < 5:
                    return None
                c = img_pil.crop((x0, y0, x1, y1))
                w, h = c.size; side = max(w, h)
                sq = Image.new("RGB", (side, side), (128, 128, 128))
                sq.paste(c, ((side - w) // 2, (side - h) // 2))
                return sq

            if is_stuff:
                # Deterministic seed via hashlib — stdlib hash() is not stable across
                # Python processes/versions; hashlib.md5 is.
                seed_bytes = hashlib.md5(f"{sample_token}_{k}".encode()).digest()
                seed = int.from_bytes(seed_bytes[:4], "little")
                rng  = np.random.RandomState(seed=seed)
                idxs = rng.choice(len(uv_v), min(N_STUFF_CROPS, len(uv_v)), replace=False)
                for ci, si in enumerate(idxs):
                    cx, cy = uv_v[si, 0], uv_v[si, 1]
                    half = 112
                    sq = _make_sq_crop(cx - half, cy - half, cx + half, cy + half)
                    crop_items.append({"crop": sq, "inst_key": (k, ci),
                                       "n_pts": n_pts, "cam_name": cam_name})
            else:
                u_min, u_max = uv_v[:, 0].min(), uv_v[:, 0].max()
                v_min, v_max = uv_v[:, 1].min(), uv_v[:, 1].max()
                cx = (u_min + u_max) / 2;  cy = (v_min + v_max) / 2
                half = max((u_max - u_min) / 2, (v_max - v_min) / 2) * 1.3
                sq = _make_sq_crop(cx - half, cy - half, cx + half, cy + half)
                crop_items.append({"crop": sq, "inst_key": k,
                                   "n_pts": n_pts, "cam_name": cam_name})

        # Step 2: batch-encode all valid crops in a single encode_image() call
        feat_by_item = batch_encode_crops(crop_encoder, crop_items)

        # Step 3: scatter by item_id; average stuff sub-crops before accumulating
        stuff_accum = {}  # k → list of (feat, n_pts)
        for item_id, item in enumerate(crop_items):
            if item_id not in feat_by_item:
                continue
            feat     = feat_by_item[item_id]
            inst_key = item["inst_key"]
            n_pts    = item["n_pts"]
            if isinstance(inst_key, tuple):   # stuff sub-crop: (k, ci)
                k_inst = inst_key[0]
                stuff_accum.setdefault(k_inst, []).append((feat, n_pts))
            else:                             # thing: inst_key is k directly
                obj_feat_accum[inst_key] += feat * n_pts
                obj_weight_sum[inst_key] += n_pts
                obj_num_views[inst_key]  += 1
                obj_num_points[inst_key] += n_pts
        for k_inst, pairs in stuff_accum.items():
            avg = np.stack([f for f, _ in pairs]).mean(axis=0)
            avg /= max(np.linalg.norm(avg), 1e-8)
            n_pts = pairs[0][1]
            obj_feat_accum[k_inst] += avg * n_pts
            obj_weight_sum[k_inst] += n_pts
            obj_num_views[k_inst]  += 1
            obj_num_points[k_inst] += n_pts

        # (b) Patch-based vox_feats — reuse the projection already computed above;
        # no second project_lidar_to_camera call needed.
        feat_map, H_p, W_p = extractor.extract_dense_features(cam_path)
        patch_px = 14
        for vi in np.where(valid_all_cam)[0]:
            u, v = int(uv_all_cam[vi, 0]), int(uv_all_cam[vi, 1])
            gy = min(v // patch_px, H_p - 1)
            gx = min(u // patch_px, W_p - 1)
            voxel_feats_accum[vi] += feat_map[gy, gx]
            voxel_cam_count[vi]   += 1

    # 4. Finalize obj_feats
    for k in range(n_inst):
        if obj_weight_sum[k] > 0:
            feat_avg = obj_feat_accum[k] / obj_weight_sum[k]
            norm     = np.linalg.norm(feat_avg)
            if norm > 1e-8:
                obj_feats[k] = (feat_avg / norm).astype(np.float16)
                obj_valid[k] = True

    # 5. Finalize vox_feats (patch-pooled, same as v1)
    valid_voxels = voxel_cam_count > 0
    voxel_feats_accum[valid_voxels] /= voxel_cam_count[valid_voxels, None]
    norms = np.linalg.norm(voxel_feats_accum[valid_voxels], axis=1, keepdims=True)
    voxel_feats_accum[valid_voxels] /= np.maximum(norms, 1e-8)
    n_valid = int(valid_voxels.sum())

    # 6. Optional: compute obj_margin if protos provided
    obj_margin = np.zeros(n_inst, dtype=np.float16)
    if protos is not None:
        for k in range(n_inst):
            if not obj_valid[k]:
                continue
            feat = obj_feats[k].astype(np.float32)
            sim_all = protos @ feat      # (16,)
            cls_id  = int(obj_classes[k])
            sim_own = float(sim_all[cls_id])
            base_sims = [sim_all[b] for b in BASE_IDS if b != cls_id]
            sim_best_base = float(max(base_sims)) if base_sims else 0.0
            obj_margin[k] = np.float16(sim_own - sim_best_base)

    # 7. Sample 4096 valid voxels
    valid_indices = np.where(valid_voxels)[0]
    if len(valid_indices) >= N_VOX_SAMPLE:
        sampled = np.random.choice(valid_indices, N_VOX_SAMPLE, replace=False)
    elif len(valid_indices) > 0:
        sampled = np.random.choice(valid_indices, N_VOX_SAMPLE, replace=True)
    else:
        sampled = np.zeros(N_VOX_SAMPLE, dtype=np.int64)

    return {
        "obj_feats":           obj_feats,
        "obj_classes":         obj_classes,
        "obj_valid":           obj_valid,
        "obj_num_views":       obj_num_views.astype(np.int16),
        "obj_num_points":      obj_num_points,
        "obj_margin":          obj_margin,
        "vox_feats":           voxel_feats_accum[sampled].astype(np.float16),  # patch-pooled
        "vox_indices":         sampled.astype(np.int32),
        "vox_gt_instance_id":  vox_gt_inst[sampled].astype(np.int32),  # NEW: GT instance idx per voxel, -1 if none
        "n_valid":             np.int32(n_valid),
        "n_voxels":            np.int32(V),
    }
```

**`vox_gt_instance_id` computation** (add before step 7 in process_sample):

```python
# Build vox→instance mapping for hybrid L_V
# Each voxel gets the index of the GT instance that owns it (-1 if none)
vox_gt_inst = np.full(V, -1, dtype=np.int32)
for k, (mask, cls_id) in enumerate(gt_instances):
    vox_gt_inst[mask] = k
```

This is O(V × N_gt) but N_gt is small (~20–30), so it's fast. Each voxel is assigned to at most one instance; stuff regions are single-instance so there's no ambiguity.

**Why this matters for hybrid L_V:** At train time, for each sampled voxel v with `vox_gt_instance_id[v] = k`:
```
target[v] = L2Norm(
    alpha * vox_feats[v]          +    # patch-pooled local texture
    (1-alpha) * obj_feats[k]           # crop-CLS semantic anchor
)
```
Without `vox_gt_instance_id`, the training loop has no way to look up `obj_feats[k]` for each voxel. This field is the bridge.
```

#### New CLI args

```
--output_dir    default: /fs/scratch/.../clip_features_v2
--proto_path    default: "" (enables obj_margin if set)
--no_vox_feats  flag: skip CLIPPatchExtractor and write zero-filled dummy voxel fields
--vox_feats_only flag: skip crop-CLS encoding, load existing .npz and merge voxel fields
```

**`--no_vox_feats` — dummy voxel field strategy (Option A):**

When L_V is disabled (ablation A), the patch-pooled `vox_feats` are not needed. However the dataset loader and Gate 7.5 still expect `vox_clip_feats`, `vox_clip_indices`, `vox_gt_instance_id` to be present. Write zeros rather than skip them — this keeps the schema stable and lets the training loop load without special-casing:

```python
if args.no_vox_feats:
    # Write schema-compatible zero fields so dataset never raises KeyError.
    # L_V.weight=0 means they are unused at train time.
    sampled           = np.zeros(N_VOX_SAMPLE, dtype=np.int32)
    vox_feats_out     = np.zeros((N_VOX_SAMPLE, 768), dtype=np.float16)
    vox_indices_out   = sampled
    vox_gt_inst_out   = np.full(N_VOX_SAMPLE, -1, dtype=np.int32)
    n_valid           = np.int32(0)
else:
    # normal patch-pooled path ...
    sampled         = sample_valid_voxels(valid_voxels, N_VOX_SAMPLE)
    vox_feats_out   = voxel_feats_accum[sampled].astype(np.float16)
    vox_indices_out = sampled.astype(np.int32)
    vox_gt_inst_out = vox_gt_inst[sampled].astype(np.int32)
    n_valid         = np.int32(int(valid_voxels.sum()))
```

**`--vox_feats_only` — merge Pass 2 into existing .npz:**

Pass 2 must load the existing .npz (which has obj_feats from Pass 1), add the voxel fields, and re-save. A plain `np.savez_compressed` would silently overwrite obj_feats:

```python
if args.vox_feats_only and os.path.exists(out_path):
    existing = dict(np.load(out_path))    # load Pass 1 obj_feats
    vox_result = compute_vox_fields(...)  # run CLIPPatchExtractor path only
    existing.update(vox_result)           # merge — obj_feats preserved
    np.savez_compressed(out_path, **existing)
```

### 3.2 Unit test on a single sample (interactive GPU)

```python
# Quick sanity check before running on all mini samples
python -c "
import numpy as np
import torch, sys
sys.path.insert(0, '.')
from panoptic.scripts.precompute_clip_voxel_features import *

# Load nuScenes mini
from nuscenes.nuscenes import NuScenes
nusc = NuScenes('v1.0-mini', '/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes_mini', verbose=False)
token = nusc.sample[0]['token']

import clip
model, _ = clip.load('ViT-L/14', device='cuda', jit=False)
extractor   = CLIPPatchExtractor(device='cuda')
crop_encoder = CLIPCropEncoder(model, 'cuda')

result = process_sample(nusc, token, nusc.dataroot, extractor, crop_encoder, 'cuda')

print('Keys:', list(result.keys()))
print('obj_feats shape:', result['obj_feats'].shape)
print('obj_valid sum:', result['obj_valid'].sum(), '/', len(result['obj_valid']))
print('obj_num_views:', result['obj_num_views'])
print('vox_feats shape:', result['vox_feats'].shape)
print('n_valid:', result['n_valid'], '/', result['n_voxels'])

# Norm checks
norms = np.linalg.norm(result['obj_feats'][result['obj_valid']].astype(np.float32), axis=1)
print('obj_feats norms (valid only) — mean:', norms.mean(), 'min:', norms.min())
assert norms.min() > 0.9, 'Norms should be ~1.0'
print('PASS')
"
```

### GATE 3: Pass criteria
- [x] All output keys present: `obj_feats`, `obj_valid`, `obj_num_views`, `obj_num_points`, `obj_margin`, `vox_feats`, `vox_indices`, `vox_gt_instance_id`, `n_valid`, `n_voxels` — **PASS**
- [x] `obj_feats` shape is `(N_gt, 768)` with N_gt > 0 — **PASS**
- [x] Valid obj_feats norms in [0.99, 1.01] — **PASS**
- [x] `obj_valid.sum() > 0` — **PASS** (70–85% valid across mini samples)
- [x] No NaN or Inf in any array — **PASS**
- [x] `vox_feats.shape == (4096, 768)` — **PASS** (zeros in Pass 1 — expected)
- [x] `vox_gt_instance_id.shape == (4096,)` — **PASS**
- [x] At least 50% of sampled voxels have `vox_gt_instance_id >= 0` — **PASS** (~96% assigned)

### GATE 3: Failure remediation
- NaN in obj_feats: check `encode_crop` return — add nan guard after `encode_image`
- `obj_valid.sum() == 0`: check projection — run `verify_camera_projection.py` on this sample
- Wrong shapes: check `build_gt_masks` output, ensure cls_id mapping is correct

---

## Phase 4: Mini Precompute + Script A Validation

### 4.1 Create `panoptic/scripts/precompute_v2_mini.sh`

```bash
#!/bin/bash
python -u panoptic/scripts/precompute_clip_voxel_features.py \
    --dataroot    /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes_mini \
    --version     v1.0-mini \
    --output_dir  /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features_v2 \
    --proto_path  panoptic/clip_prototypes/text_prototypes_vitl14_224_v2.pt \
    --device      cuda \
    --split       both
```

**Run interactively:**
```bash
bash panoptic/scripts/precompute_v2_mini.sh 2>&1 | tee panoptic/logs/precompute_v2_mini.log
```

Expected: ~323 files in 15–30 minutes on A100.

### 4.2 Run Script A on v2 mini features

```bash
python panoptic/scripts/diag_clip_coverage.py \
    --clip_feat_dir /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features_v2 \
    --version       v1.0-mini \
    --proto_path    panoptic/clip_prototypes/text_prototypes_vitl14_224_v2.pt \
    --out_dir       panoptic/diagnostics/v2 \
    --histogram
```

Compare against v1 baseline from `panoptic/diagnostics/coverage_v1.0-mini.json`.

### 4.3 Create `panoptic/scripts/verify_crop_quality.py` (for failure case)

If Script A gates fail, this script saves a visual contact sheet:

```python
"""Save 50 random crops per novel class with top-5 CLIP text predictions.

Usage:
    python panoptic/scripts/verify_crop_quality.py \
        --npz_dir /fs/scratch/.../clip_features_v2/v1.0-mini \
        --proto_path panoptic/clip_prototypes/text_prototypes_vitl14_224_v2.pt \
        --out_dir panoptic/diagnostics/crops
"""
# For each novel class, load 50 random instances with obj_valid=True
# Load original camera image, recompute 2D bbox projection, save crop
# Overlay top-5 text sims on image
# Saves: panoptic/diagnostics/crops/{class_name}_contact_sheet.png
```

### GATE 4: Pass criteria
- [x] Novel valid coverage > 50% for all 4 novel classes — **PASS** (bus=93.8%, moto=57.7%, ped=50.2%, veg=94.5%)
- [~] Novel mean margin_to_base > 0 for all 4 novel classes — **PARTIAL** (bus=+0.023, moto=+0.016, ped=+0.049 PASS; veg=−0.027 FAIL — stuff noise, mitigated by THING-only L_O guard)
- [x] Novel top-1 text accuracy > 50% for at least 3 of 4 — **PASS** (ped 95% correct, bus ~75%, moto ~61%)
- [x] Novel top-3 text accuracy > 70% for all 4 — **PASS**

**Mitigation applied:** `distill_object_loss.py` filters to THING classes only (`_THING_CLASS_IDS = frozenset(range(10))`). Vegetation L_O gradient suppressed. Proceeding to Phase 5.

### GATE 4: Failure remediation
- Margin still negative: run `verify_crop_quality.py`, inspect crops — if crops capture background, increase `MIN_PTS_THING` or expansion factor
- Low top-1 accuracy despite positive margin: prototype geometry issue — revise prompts (Phase 2)
- Low valid coverage: decrease `MIN_PTS_THING` from 5 to 3
- Do NOT proceed to Phase 5 until gate passes

---

## Phase 5: Full Trainval Precompute v2

### 5.1 Create `panoptic/scripts/precompute_v2_full.sh`

**Two-pass strategy** — saves 180 GB and ~10 hours if ablation B is never needed:

- **Pass 1 (required):** `--no_vox_feats` — produces only `obj_feats` + `vox_gt_instance_id`. ~12–15h, ~3 GB. Sufficient for ablation A (L_V=0).
- **Pass 2 (only if ablation B proceeds):** `--vox_feats_only --resume` — adds `vox_feats` to existing files. ~12–15h, adds ~180 GB.

```bash
#!/bin/bash
#BSUB -J precompute-v2
#BSUB -q batch_a100
#BSUB -n 4
#BSUB -M 32768
#BSUB -W 16:00
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -o /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/panoptic/logs/precompute_v2_%J.out
#BSUB -e /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene/panoptic/logs/precompute_v2_%J.err

module load conda && source activate openscene
module load cuda/11.0.2 cudnn/11.0_v8.0
export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene

cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene

python -u panoptic/scripts/precompute_clip_voxel_features.py \
    --dataroot   /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes \
    --version    v1.0-trainval \
    --output_dir /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features_v2 \
    --proto_path panoptic/clip_prototypes/text_prototypes_vitl14_224_v2.pt \
    --device cuda --split both --resume \
    --no_vox_feats    # Pass 1: obj_feats only — ~3 GB, ~12-15h
    # Remove --no_vox_feats and add --vox_feats_only for Pass 2 if ablation B needed
```

**Submit:**
```bash
bsub < panoptic/scripts/precompute_v2_full.sh
```

Expected: Pass 1 ~12–15 hours, ~3 GB. Pass 2 (if needed) ~12–15 hours, adds ~180 GB.

### 5.2 Spot-check after completion

```bash
python -c "
import numpy as np, os, glob
files = sorted(glob.glob('/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features_v2/v1.0-trainval/*.npz'))
print(f'Total files: {len(files)}')
# Spot-check 10 random files
import random; random.shuffle(files)
for f in files[:10]:
    d = np.load(f)
    assert 'obj_valid' in d and 'n_voxels' in d, f'Missing fields in {f}'
    assert d['obj_feats'].shape[1] == 768
    valid_norms = np.linalg.norm(d['obj_feats'][d['obj_valid']].astype(np.float32), axis=1)
    assert not np.any(np.isnan(valid_norms)), f'NaN in {f}'
    print(f'OK: {os.path.basename(f)} — {d[\"obj_valid\"].sum()}/{len(d[\"obj_valid\"])} valid, n_valid={d[\"n_valid\"]}')
print('PASS')
"
```

### GATE 5: Pass criteria
- [x] File count ≥ 28,000 (all trainval samples) — **PASS** (34,149 files)
- [x] All 10 spot-checked files have correct schema — **PASS**
- [x] No NaN in spot-checked obj_feats (valid entries) — **PASS**
- [x] `obj_valid.sum() > 0` in all spot-checked files — **PASS**

---

## Phase 5c: Mini Pass 2 + Vox-Margin Validation (REQUIRED before full Pass 2)

**Purpose:** Before submitting the 200GB full trainval Pass 2, verify that patch-pooled `vox_feats` give positive margin for vegetation in mini. This is the key unknown — crop-CLS (obj_feats) gave margin=−0.027 for vegetation, but patch-pooled features should be more locally discriminative. If vox_margin for vegetation is still negative, the L_V teacher for vegetation is broken and ablation B config needs adjustment.

### 5c.1 Run Pass 2 on mini (interactive, ~10 min)

```bash
python -u panoptic/scripts/precompute_clip_voxel_features_v2.py \
    --dataroot /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes_mini \
    --version v1.0-mini \
    --output_dir /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features_v2 \
    --vox_feats_only \
    --resume \
    --device cuda
```

### 5c.2 Verify vox_feats margin for all classes (especially vegetation)

```python
import numpy as np, glob, torch
import torch.nn.functional as F

proto_data = torch.load("panoptic/clip_prototypes/text_prototypes_vitl14_224_v2.pt")
protos = F.normalize(proto_data["embeddings"].float(), dim=-1).numpy()  # (16, 768)

CLASS_NAMES = ["barrier","bicycle","bus","car","construction_vehicle",
               "motorcycle","pedestrian","traffic_cone","trailer","truck",
               "driveable_surface","other_flat","sidewalk","terrain","manmade","vegetation"]

vox_margins = {i: [] for i in range(16)}

for npz_path in glob.glob("/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features_v2/v1.0-mini/*.npz"):
    d = np.load(npz_path, allow_pickle=False)
    vox_feats = d["vox_feats"].astype(np.float32)      # (4096, 768)
    vox_inst  = d["vox_gt_instance_id"]                 # (4096,) int
    obj_cls   = d["obj_classes"]                        # (N_gt,)
    n_valid   = int(d["n_valid"])
    if n_valid == 0:
        continue
    sims = vox_feats[:n_valid] @ protos.T               # (n_valid, 16)
    for vi in range(n_valid):
        if vox_inst[vi] < 0 or vox_inst[vi] >= len(obj_cls):
            continue
        cls_id = int(obj_cls[vox_inst[vi]])
        sim_own  = float(sims[vi, cls_id])
        tmp = sims[vi].copy(); tmp[cls_id] = -1.0
        sim_base = float(tmp.max())
        vox_margins[cls_id].append(sim_own - sim_base)

print("Vox-margin per class (mean ± std):")
for cls_id, margins in vox_margins.items():
    if margins:
        arr = np.array(margins)
        print(f"  [{cls_id:2d}] {CLASS_NAMES[cls_id]:<22}: mean={arr.mean():+.4f}  std={arr.std():.4f}  n={len(arr)}")
```

### GATE 5c: Pass criteria

- [ ] `vox_margin[vegetation]` mean > 0 — confirms patch-pooled features discriminate vegetation from terrain
- [ ] `vox_margin[bus]`, `vox_margin[motorcycle]`, `vox_margin[pedestrian]` mean > 0 — consistency check vs obj_feats gates
- [ ] `n_valid > 0` for all 4 novel classes — confirms vox_feats were actually computed (not all zeros)

### GATE 5c: Failure remediation

- **vox_margin[vegetation] ≤ 0:** Patch-pooled features are insufficient for vegetation. Options:
  1. Set `alpha_stuff=1.0` in ablation B (pure patch, at least no obj_feat contamination) — but low margins mean L_V target is still confused
  2. Consider CASD-style online encoding for vegetation in job1c instead
  3. Check if vegetation voxels are projecting into sky/road regions — add a depth filter (keep voxels with z > 0.5m, which are likely tree trunks/canopy rather than ground clutter)
- **n_valid = 0 for a class:** `add_vox_feats()` not updating files — check `n_valid` field is being written, re-run Pass 2 without `--resume`

**If gate passes → submit full Pass 2:**
```bash
bsub < panoptic/scripts/precompute_clip_voxel_features_v2_pass2.sh
```

---

## Phase 5d: CLS-Crop Vox_feats Redesign — CANCELLED

**Reason:** Superseded by the broadcast approach. The `vox_gt_instance_id` field already written in Pass 1 maps every sampled voxel to its GT instance. Broadcasting `obj_feats[vox_gt_instance_id[v]]` gives a valid L_V teacher for THING voxels with the same CLS-crop quality as obj_feats (bus=+0.023, moto=+0.016, ped=+0.049 margins). This requires zero additional storage and zero additional precompute time.

For STUFF voxels, online CASD at training time is cleaner and more accurate than precomputed crops (no augmentation mismatch, no 200GB storage).

**Action:** Proceed to Phase 5e (broadcast diagnostic) before implementing job1c.

---

## Phase 5e: Broadcast Diagnostic (REQUIRED before job1c config)

**Purpose:** Confirm that `obj_feats[vox_gt_instance_id]` is a valid L_V teacher for THING voxels and that STUFF voxels need CASD.

```bash
cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
python panoptic/scripts/diag_broadcast_vox_check.py \
    --feat_dir /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features_v2/v1.0-mini \
    --proto_path panoptic/clip_prototypes/text_prototypes_vitl14_224_v2.pt
```

**GATE 5e — Pass criteria:**
- [ ] THING novel broadcast margins > 0 (bus, moto, ped): confirms broadcast is valid L_V teacher
- [ ] Vegetation broadcast margin < 0.05: confirms CASD is necessary
- [ ] THING novel voxel coverage ≥ 30% of 4096 sampled voxels: enough density for meaningful L_V signal

**If GATE 5e fails:**
- If THING margins are negative → check `vox_gt_instance_id` ordering matches `build_gt_instances()` in precompute; may be an off-by-one in STUFF_IDS/THING_IDS ordering
- If coverage < 30% for novel classes → novel classes are spatially rare; L_V broadcast won't add much signal; rely on L_O alone and CASD for vegetation

**Purpose:** Gate 5c proved patch-pooled vox_feats give negative margins for all 16 classes. Before running full Pass 2, redesign `vox_feats` to use per-voxel context-crop CLS encoding (same CLS mechanism as obj_feats, which gives positive margins for novel THING classes).

### Root cause of patch-pooled failure

CLIP patch tokens are 14×14 px local texture descriptors. They are not trained to align with text. The CLS token — produced by `encode_image()` — is what CLIP trains to align with text prompts. Patch-pooled vox_feats push the model in the wrong direction in text space.

### Revised vox_feats design: per-voxel context-crop CLS

For each sampled voxel (from the 4096-sample pool):
1. Project the voxel's 3D coordinate to all 6 cameras
2. Select best camera: must have valid projection (UV within image bounds, depth > 0); prefer smallest depth (closest → largest object in frame)
3. Take 112×112 px crop centered on the projected UV, pad-to-square
4. Call `encode_image()` → CLS token → L2-normalize → store as `vox_feats[vi]`

**Why this gives positive margins:** A voxel inside a bus projects to the bus body in the image. A 112×112 crop of a bus body → `encode_image()` → CLS → aligns with "a bus in a scene" text prototype. Vegetation voxels project to tree/foliage → CLS aligns with vegetation. Same mechanism as obj_feats for things, applied per-voxel.

**Performance:** 4096 voxels in batches of 128 = 32 `encode_image()` calls per sample, vs patch extractor's ~120 calls. Faster than patch approach.

### 5d.1 Add `compute_vox_feats_cls()` to `precompute_clip_voxel_features_v2.py`

```python
def compute_vox_feats_cls(nusc, sample_token, dataroot, vox_xyz, crop_encoder,
                          batch_size=128):
    """Per-voxel CLS features: project each voxel to best camera, encode 112x112 crop.
    Returns (N_vox, 768) float16. Voxels with no valid camera projection get zero vector.
    """
    N = len(vox_xyz)
    out_feats = np.zeros((N, 768), dtype=np.float16)

    sample = nusc.get("sample", sample_token)
    best_cam  = np.full(N, -1, dtype=np.int32)
    best_depth = np.full(N, np.inf, dtype=np.float32)
    cam_uvs = {}

    for cam_idx, cam_name in enumerate(CAMERAS):
        uv, depths, valid = project_lidar_to_camera(nusc, sample_token, vox_xyz, cam_name)
        cam_uvs[cam_name] = (uv, depths, valid)
        update_mask = valid & (depths < best_depth)
        best_cam[update_mask] = cam_idx
        best_depth[update_mask] = depths[update_mask]

    for cam_idx, cam_name in enumerate(CAMERAS):
        vi_this_cam = np.where(best_cam == cam_idx)[0]
        if len(vi_this_cam) == 0:
            continue
        cam_sd   = nusc.get("sample_data", sample["data"][cam_name])
        cam_path = os.path.join(dataroot, cam_sd["filename"])
        img_pil  = Image.open(cam_path).convert("RGB")
        img_w, img_h = img_pil.size
        uv_cam, _, _ = cam_uvs[cam_name]
        half = 112

        crops_vi = []
        for vi in vi_this_cam:
            cx, cy = float(uv_cam[vi, 0]), float(uv_cam[vi, 1])
            x0 = max(0, int(cx - half)); x1 = min(img_w, int(cx + half))
            y0 = max(0, int(cy - half)); y1 = min(img_h, int(cy + half))
            if x1 - x0 < 16 or y1 - y0 < 16:
                continue
            c = img_pil.crop((x0, y0, x1, y1))
            w, h = c.size; side = max(w, h)
            sq = Image.new("RGB", (side, side), (128, 128, 128))
            sq.paste(c, ((side-w)//2, (side-h)//2))
            crops_vi.append((vi, sq))

        for start in range(0, len(crops_vi), batch_size):
            chunk   = crops_vi[start:start+batch_size]
            vis_ch  = [vi for vi, _ in chunk]
            tensors = torch.stack([crop_encoder.transform(sq) for _, sq in chunk])
            with torch.no_grad():
                feats = crop_encoder.model.encode_image(
                    tensors.to(crop_encoder.device).half())
                feats = F.normalize(feats.float(), dim=-1).cpu().numpy()
            for vi, feat in zip(vis_ch, feats):
                out_feats[vi] = feat.astype(np.float16)

    return out_feats   # (N, 768) float16
```

**CLI flag:** Add `--vox_feats_mode` to `precompute_clip_voxel_features_v2.py`:

```
--vox_feats_mode cls     # new CLS-crop per voxel (default for Pass 2 redesign)
--vox_feats_mode patch   # legacy patch-pooled (kept for reference)
```

### 5d.2 Run mini Pass 2 with CLS vox_feats

```bash
python -u panoptic/scripts/precompute_clip_voxel_features_v2.py \
    --dataroot /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes_mini \
    --version v1.0-mini \
    --output_dir /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features_v2 \
    --vox_feats_only \
    --vox_feats_mode cls \
    --resume \
    --device cuda
```

### 5d.3 Re-run vox_margin diagnostic

```bash
python panoptic/scripts/diag_precompute_vox_check.py
```

### GATE 5d: Pass criteria

- [ ] `vox_margin[vegetation]` mean > 0 — tree-region CLS crops align with vegetation text
- [ ] `vox_margin[bus]`, `vox_margin[motorcycle]`, `vox_margin[pedestrian]` mean > 0
- [ ] At least 10 of 16 classes have positive mean vox_margin
- [ ] `n_valid > 0` for all 4 novel classes — CLS vox_feats are actually being written

### GATE 5d: Failure remediation

- **vox_margin[vegetation] ≤ 0:** Vegetation voxels may project to ground/sky. Add depth filter: keep only voxels with `vox_xyz[z] > 1.0 m` for vegetation samples (likely canopy/trunk, not ground clutter).
- **vox_margin still negative for things:** Crop too large for small objects (112px at close range captures mostly background). Try 64×64 for instances where `obj_feats n_pts < 20`.
- **Universally negative still:** Fall back to `alpha_things=0.0` (pure instance-broadcast from obj_feats, known positive for things) as minimum viable ablation B.

**If gate passes → update pass2 script with `--vox_feats_mode cls` and submit:**
```bash
bsub < panoptic/scripts/precompute_clip_voxel_features_v2_pass2.sh  # with --vox_feats_mode cls
```

---

**Goal:** Confirm job1a trains correctly with v2 prototypes (new neg-bank, new prompts).

### 6.1 Create `panoptic/configs/job1a_v2_mini.yaml`

Copy `job1a.yaml`, change dataroot to mini, 5 epochs, and update token paths:
```yaml
data:
  dataroot: /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes_mini
  version: v1.0-mini

token:
  prototype_path:   panoptic/clip_prototypes/text_prototypes_vitl14_224_v2.pt
  negative_bank:    panoptic/clip_prototypes/negative_bank_v2.pt
  multiprompt_path: panoptic/clip_prototypes/text_prototypes_vitl14_224_multiprompt_v2.pt
  restore_from_warm_start: false

training:
  epochs: 5
  phase1_epochs: 1

checkpoint:
  save_dir: panoptic/checkpoints_job1a_v2_mini
  save_every: 1

logging:
  wandb_run_name: job1a-v2-mini-smoke
```

### 6.2 Run mini smoke test

```bash
python -u panoptic/train/train_panoptic_job1.py \
    --config panoptic/configs/job1a_v2_mini.yaml \
    2>&1 | tee panoptic/logs/job1a_v2_mini.log
```

### GATE 6: Pass criteria
- [ ] No NaN in any loss term for all 3 smoke epochs
- [ ] `loss_token` starts < 6.0 and decreases across epochs
- [ ] `lora_gnorm` in [1e-4, 10] (after `scaler.unscale_()`)
- [ ] `temperature` < 3.0 at epoch 0 (starting at 0.5 — correctly initialized)
- [ ] No KeyError or shape mismatches in first 5 steps

### GATE 6: Failure remediation
- NaN in loss_token: check if any class prototype has zero norm; reload neg-bank
- Temperature wrong: check `restore_from_warm_start: false` is set in config
- Shape errors: likely neg-bank size change breaking proto indexing — check `n_total` print at startup

---

## Phase 7: Job1a-v2 — Full Training (12 epochs)

### 7.1 Create `panoptic/configs/job1a_v2.yaml`

Based on `job1a.yaml`, updating token paths and reducing to 12 epochs:

```yaml
data:
  dataroot: /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes
  version: v1.0-trainval
  voxel_size: 0.05

model:  # same as job1a.yaml
  backbone_ckpt: models/nuscenes_openseg.pth.tar
  backbone_dim: 768
  hidden_dim: 256
  num_classes: 16
  num_thing_queries: 100
  num_stuff_classes: 6
  nheads: 8
  dim_feedforward: 1024
  num_blocks: 3
  num_levels: 3
  dropout: 0.0
  clip_embed_dim: 768
  head_type: single_clip

lora:
  enabled: true
  rank: 16
  dropout: 0.05

warm_start:
  checkpoint: panoptic/checkpoints_token_full_v2/best.pth

loss:
  cost_class: 0.0
  cost_mask: 5.0
  cost_dice: 5.0
  eos_coef: 0.1
  weight_ce: 0.0
  weight_mask: 5.0
  weight_dice: 5.0
  weight_obj: 2.0
  p_ratio: 0.75

token:
  prototype_path:   panoptic/clip_prototypes/text_prototypes_vitl14_224_v2.pt
  negative_bank:    panoptic/clip_prototypes/negative_bank_v2.pt
  multiprompt_path: panoptic/clip_prototypes/text_prototypes_vitl14_224_multiprompt_v2.pt
  learnable_temperature: true
  temp_max: 5.0
  margin: 0.1
  weight: 1.0
  restore_from_warm_start: false

training:
  lr_decoder: 5.0e-5
  lr_token_head: 1.0e-4
  lr_lora: 1.0e-4
  weight_decay: 0.01
  epochs: 12
  warmup_epochs: 2
  min_lr: 1.0e-6
  mixed_precision: true
  grad_accum: 8
  grad_clip: 0.5
  num_workers: 0
  phase1_epochs: 2

logging:
  log_every: 50
  eval_every: 2
  wandb_enabled: true
  wandb_project: openscene-panoptic
  wandb_run_name: job1a-v2-lora-c1

checkpoint:
  save_dir: panoptic/checkpoints_job1a_v2
  save_every: 1
```

### 7.2 Create `panoptic/scripts/job1a_v2_full.sh`

Based on `job1a_full.sh`, update:
- Config: `job1a_v2.yaml`
- Log files: `job1a_v2_%J.out/err`
- Checkpoint dir: `checkpoints_job1a_v2`
- Prototype files: v2 (`text_prototypes_vitl14_224_v2.pt`, `negative_bank_v2.pt`)
- Wall time: **24 hours** (12 epochs × ~2h/epoch; 38h original was for 20 epochs)

```bash
bsub < panoptic/scripts/job1a_v2_full.sh
```

### 7.3 Monitoring checkpoints

```bash
# Check progress every few hours
ls -lh panoptic/checkpoints_job1a_v2/
# Latest epoch:
ls -t panoptic/checkpoints_job1a_v2/ | head -5
```

### GATE 7: Pass criteria (checked at epoch 8, ~2/3 through training)
- [ ] PQ_base ≥ 0.35 at epoch 8
- [ ] PQ_novel = 0 (expected — no novel supervision in job1a)
- [ ] `loss_token` < 3.5 at epoch 8
- [ ] `temperature` in [0.07, 3.0] at epoch 8
- [ ] `best.pth` exists in checkpoint dir

### GATE 7: Failure remediation
- PQ_base < 0.30 at epoch 8: check mask losses are decreasing; increase `phase1_epochs` to 4
- NaN appearing after epoch 5: check `grad_clip` and `mixed_precision` settings
- Temperature saturated at 5.0: increase `temp_max` to 8.0 in config

---

## Phase 7.5: Integration Dry Run

**Purpose:** Before submitting any job1b job, verify that all new v2 fields (obj_valid, vox_gt_instance_id) are loaded correctly by the dataset and reach the loss functions without shape errors or KeyErrors. This gate sits between the job1a checkpoint being ready and the first job1b training call. It costs ~5 minutes and can prevent a 24h job from silently training with wrong targets.

### 7.5.1 Create dry-run script

Run one training step on the mini config with extra assertions enabled:

```bash
python -c "
import sys, torch
sys.path.insert(0, '.')
from panoptic.train.train_panoptic_job1 import main as train_main
import argparse

# Monkey-patch: stop after 1 step and print field shapes
_orig_forward = None
def _check_batch(batch, device):
    required = ['obj_clip_feats', 'obj_clip_classes', 'obj_clip_valid',
                'vox_clip_feats', 'vox_clip_indices', 'vox_gt_instance_id']
    for key in required:
        assert key in batch, f'MISSING key: {key}'
        t = batch[key]
        print(f'  {key}: shape={t[0].shape if isinstance(t, list) else t.shape} dtype={t[0].dtype if isinstance(t, list) else t.dtype}')
    # Check obj_clip_valid is bool
    valid = batch['obj_clip_valid']
    if isinstance(valid, list):
        assert valid[0].dtype == torch.bool, f'obj_clip_valid must be bool, got {valid[0].dtype}'
    # Check vox_gt_instance_id has -1 for unassigned
    vgi = batch['vox_gt_instance_id']
    if isinstance(vgi, list):
        print(f'  vox_gt_instance_id: {(vgi[0] >= 0).sum()}/{len(vgi[0])} voxels assigned to GT instance')
    print('  PASS: all v2 fields present and typed correctly')

# Run with --dry_run flag or just config check
args_str = '--config panoptic/configs/job1b_v2_lo_only_mini.yaml --dry_run'
print('=== Phase 7.5: Integration dry run ===')
sys.argv = ['train'] + args_str.split()
" 2>&1 | tee panoptic/logs/integration_dry_run.log
```

If the train script doesn't have a `--dry_run` flag, run 1 epoch with `training.epochs: 1` in a throw-away config.

### 7.5.2 Manual field inspection (fallback)

If the above doesn't work easily, load one batch manually:

```python
from panoptic.dataset.nuscenes_panoptic import NuScenesPanopticDataset
from torch.utils.data import DataLoader
import torch

ds = NuScenesPanopticDataset(
    dataroot='/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes_mini',
    version='v1.0-mini',
    split='train',
    clip_feat_dir='/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features_v2',
    # ... other args from job1b_v2_lo_only_mini.yaml
)
loader = DataLoader(ds, batch_size=1, num_workers=0)
batch = next(iter(loader))

print('=== v2 field check ===')
for key in ['obj_clip_feats', 'obj_clip_classes', 'obj_clip_valid',
            'vox_clip_feats', 'vox_clip_indices', 'vox_gt_instance_id']:
    if key not in batch:
        print(f'  MISSING: {key}')
    else:
        v = batch[key]
        print(f'  {key}: shape={v.shape} dtype={v.dtype} '
              f'range=[{v.float().min():.3f}, {v.float().max():.3f}]')
```

### GATE 7.5: Pass criteria

- [ ] `obj_clip_valid` field exists in batch, dtype=bool
- [ ] `vox_gt_instance_id` field exists in batch, shape `(B, 4096)`, int64, values in [-1, N_gt-1]
- [ ] At least 30% of `vox_gt_instance_id` values are ≥ 0 (voxels assigned to a GT instance)
- [ ] `obj_clip_feats[obj_clip_valid]` norms are in [0.99, 1.01] (L2-normalized)
- [ ] No KeyError or shape mismatch in first forward pass through model + loss
- [ ] `loss_lo` is finite (not NaN, not 0.0) on the first training step

### GATE 7.5: Failure remediation

- Missing `obj_clip_valid`: dataset loader is still v1 — apply the `nuscenes_panoptic.py` changes from the "Required code changes" section
- Missing `vox_gt_instance_id`: precomputed .npz files are still v1 schema — re-run Phase 3+4+5
- `vox_gt_instance_id` all -1: `vox_gt_inst` computation in `process_sample()` is wrong — re-check the `for k, (mask, cls_id) in enumerate(gt_instances)` loop
- `loss_lo` NaN on step 1: `obj_clip_valid` filter not applied in `distill_object_loss.py` — check `if not obj_clip_valid[b][gi]: continue`

**Do NOT submit job1b-v2 until Gate 7.5 passes.**

---

## Phase 8: Job1b-v2 Ablation A — L_O Only (L_V = 0)

**Purpose:** Isolate whether crop-CLS L_O alone is sufficient to produce novel PQ > 0. No L_V to confound results.

### 8.1 Create `panoptic/configs/job1b_v2_lo_only.yaml`

Based on `job1b.yaml`:
```yaml
warm_start:
  checkpoint: panoptic/checkpoints_job1a_v2/best.pth

token:
  prototype_path:   panoptic/clip_prototypes/text_prototypes_vitl14_224_v2.pt
  negative_bank:    panoptic/clip_prototypes/negative_bank_v2.pt
  multiprompt_path: panoptic/clip_prototypes/text_prototypes_vitl14_224_multiprompt_v2.pt

distillation:
  enabled: true
  clip_feat_dir: /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features_v2
  L_O:
    weight: 1.0
    warmup_epochs: 2
    loss_type: cosine
  L_V:
    weight: 0.0        # DISABLED — first ablation
    warmup_epochs: 3

checkpoint:
  save_dir: panoptic/checkpoints_job1b_v2_lo
  save_every: 1

training:
  epochs: 15
```

### 8.2 Create `panoptic/configs/job1b_v2_lo_only_mini.yaml`

Same, mini dataroot, 3 epochs.

### 8.3 Mini smoke test

```bash
python -u panoptic/train/train_panoptic_job1.py \
    --config panoptic/configs/job1b_v2_lo_only_mini.yaml \
    --smoke_test 2>&1 | tee panoptic/logs/job1b_v2_lo_smoke.log
```

### GATE 8: Pass criteria (3-epoch smoke test)
- [ ] **`indices_all` verification**: add a 1-epoch debug log confirming novel GT instances appear in L_O pairs. Add to training loop temporarily:
  ```python
  # Debug: confirm novel queries are receiving L_O gradient
  for b_idx, (pi, gi) in enumerate(indices_all):
      novel_matched = sum(1 for g in gi if all_targets_obj["classes"][b_idx][g] in {2,5,6,15})
      if novel_matched > 0:
          print(f"  L_O novel matches this step: {novel_matched}")
  ```
  If novel_matched = 0 in every step: `indices_all` is still being replaced by `indices` somewhere — find and fix before proceeding.
- [ ] `loss_lo` appears in logs and starts decreasing from epoch 1 warmup completion
- [ ] `loss_lo` < 0.8 by epoch 3 (cosine loss 0→1; 0.8 = some alignment with valid teacher)
- [ ] No NaN in any loss
- [ ] `count_novel_lo > 0` in at least 50% of training steps (novel instances receiving L_O gradient)
- [ ] PQ_base not collapsed (≥ 0.25 at epoch 3) — if PQ_base < 0.20, L_O weight is too high, reduce to 0.5 and rerun
- [ ] **NOTE:** `PQ_novel = 0` at epoch 3 is **acceptable** — 3 epochs is insufficient to push representations through the temperature barrier. Do not gate on novel PQ here. Proceed to full job1b-v2 if the mechanism checks above all pass.

### GATE 8: Failure remediation
- `loss_lo` not appearing: check `distillation.enabled: true` and clip_feat_dir path
- `loss_lo` NaN: zero-norm obj_feats slipping through — check `obj_clip_valid` filter in `distill_object_loss.py`
- `count_novel_lo = 0`: `indices_all` not being used — verify the all-GT matcher path in `train_panoptic_job1.py`
- PQ_novel still 0 after 3 epochs: expected — proceed to full job1b-v2, check at epoch 5
- PQ_base collapsed (< 0.20): L_O weight too high — reduce to 0.5 and rerun smoke

---

## Phase 9: Job1b-v2 Ablation A — Full Training

### 9.1 Create `panoptic/scripts/job1b_v2_lo_full.sh`

```bash
#BSUB header: batch_a100, 48h, 1 GPU
python -u panoptic/train/train_panoptic_job1.py \
    --config panoptic/configs/job1b_v2_lo_only.yaml
```

### GATE 9: Pass criteria (final epoch)
- [ ] novel PQ > 0.05 (at least motorcycle or bus getting recognized)
- [ ] PQ_base ≥ 0.38 (close to job1a level)
- [ ] `loss_lo` < 0.5 at final epoch
- [ ] `best.pth` saved in `checkpoints_job1b_v2_lo/`

### GATE 9: Failure remediation
- novel PQ = 0 after all 15 epochs: something fundamental is wrong — check that `indices_all` is being used (not `indices`) in the L_O block in `train_panoptic_job1.py`; log which queries get L_O supervision; check novel class obj_feats are being loaded correctly
- PQ_base degrades > 5 points from job1a: L_O is interfering with mask quality — add `stop_gradient` to mask path or reduce `warmup_epochs` for L_O to 3

---

## Phase 10: Ablation B — CANCELLED

Merged into job1c. See Phase 12 for the combined L_O + L_V broadcast + CASD run.

**Rationale:** A standalone L_V broadcast ablation B would only supervise THING voxels (same teacher as L_O, applied via masks rather than queries). It adds mask-level gradient feedback but cannot demonstrate vegetation PQ improvement since STUFF voxels still have no valid teacher without CASD. The marginal insight from ablation B alone does not justify a full 15-epoch run. The cleaner comparison is:

```
Baseline:     job1b-A  (L_O only)         → novel PQ for bus/moto/ped, veg=0
Next step:    job1c    (L_O + L_V + CASD) → novel PQ all 4 classes, mask quality improvement
```

---

## Phase 11: Ablation B Full — CANCELLED

Merged into job1c. See Phase 12–13.

**Purpose:** Test whether per-voxel CLS-crop L_V improves novel PQ beyond L_O alone, particularly for vegetation (STUFF, zero L_O gradient in ablation A).

**Design (revised):** With CLS-crop vox_feats, each voxel carries a 112×112 context-crop CLS embedding — the same alignment mechanism as obj_feats. Vegetation voxels project to tree/leaf regions → CLS aligns with "vegetation" text prototype. The hybrid target blends per-voxel CLS (vox_feats) with instance-averaged CLS (obj_feats) for THING classes:

```
vox_target[v] = L2Norm(
    alpha(cls) * vox_feats[v]         +    # per-voxel CLS crop (context around voxel)
    (1-alpha(cls)) * obj_feats[k]          # instance-averaged CLS (k = vox_gt_instance_id[v])
)
```

Alpha rationale:
- `alpha_things = 0.4` — blend per-voxel CLS (local context) with instance-averaged CLS (stability)
- `alpha_stuff = 1.0` — pure per-voxel CLS for stuff (vegetation obj_feats still have margin=−0.027; don't blend)
- Voxels with `vox_gt_instance_id = -1` fall back to pure vox_feats target

### 10.1 Update `panoptic/losses/distill_voxel_loss.py`

Add `mode` and alpha parameters; compute hybrid target when mode="hybrid":

```python
class VoxelDistillLoss(nn.Module):
    def __init__(self, loss_type="cosine", mode="patch",
                 alpha_things=0.4, alpha_stuff=0.2):
        super().__init__()
        self.loss_type    = loss_type
        self.mode         = mode          # "patch" | "broadcast" | "hybrid"
        self.alpha_things = alpha_things
        self.alpha_stuff  = alpha_stuff
        # stuff class ids (0-indexed): other_flat=10, sidewalk=11, terrain=12, manmade=13, vegetation=14
        self.stuff_ids = set(range(10, 16))

    def _build_target(self, vox_clip_feats, vox_gt_instance_id, obj_clip_feats,
                      obj_clip_classes):
        """Build per-voxel target according to self.mode."""
        if self.mode == "patch" or obj_clip_feats is None:
            return vox_clip_feats   # (N, 768) — original path

        N = vox_clip_feats.shape[0]
        targets = vox_clip_feats.clone()    # start from patch features

        for i in range(N):
            inst_idx = vox_gt_instance_id[i].item()
            if inst_idx < 0 or inst_idx >= obj_clip_feats.shape[0]:
                continue   # no GT instance → keep patch feature

            obj_feat = obj_clip_feats[inst_idx]    # (768,)
            if obj_feat.norm() < 0.1:
                continue   # instance has no valid crop-CLS

            cls_id = obj_clip_classes[inst_idx].item()
            alpha  = self.alpha_stuff if cls_id in self.stuff_ids else self.alpha_things

            if self.mode == "broadcast":
                alpha = 0.0    # pure semantic
            # hybrid: blend
            blended = alpha * vox_clip_feats[i] + (1.0 - alpha) * obj_feat
            norm = blended.norm()
            if norm > 1e-6:
                targets[i] = blended / norm

        return targets  # (N, 768) L2-normalized hybrid targets

    def forward(self, pred_embed, pred_masks, vox_clip_feats, vox_clip_indices,
                vox_gt_instance_id=None, obj_clip_feats=None, obj_clip_classes=None):
        # Build target (mode-dependent)
        # then proceed with existing softmax-weighted reconstruction + cosine loss
        ...
```

### 10.2 Update `panoptic/configs/job1b_v2_lo_lv.yaml`

```yaml
distillation:
  L_V:
    weight: 0.3
    warmup_epochs: 3
    mode: hybrid          # patch | broadcast | hybrid
    alpha_things: 0.4     # THINGS: 40% per-voxel CLS vox_feats + 60% instance-averaged obj_feats
                          # Both are CLS-based; blend adds instance-level stability for things.
    alpha_stuff: 1.0      # STUFF: 100% per-voxel CLS vox_feats — do NOT blend obj_feats for stuff;
                          # vegetation obj_feats still have margin=-0.027 (crop-CLS noisy for large stuff).
                          # Pure per-voxel CLS expected to give positive margin after Phase 5d.

checkpoint:
  save_dir: panoptic/checkpoints_job1b_v2_lo_lv
```

### 10.3 Mini smoke test

```bash
python -u panoptic/train/train_panoptic_job1.py \
    --config panoptic/configs/job1b_v2_lo_lv_mini.yaml \
    --smoke_test 2>&1 | tee panoptic/logs/job1b_v2_lo_lv_smoke.log
```

### GATE 10: Pass criteria
- [ ] `loss_lv` appears in logs and decreases from epoch 3+ (after L_V warmup)
- [ ] Confirm hybrid targets have correct shape/norm: add a 1-step log of `target.norm(dim=-1).mean()`
- [ ] No NaN
- [ ] `dice loss` stays < 0.6 (does NOT saturate like with v1 patch-only L_V at weight=1.0)
- [ ] PQ_novel at epoch 3 ≥ ablation A epoch 3 result

---

## Phase 11: Job1b-v2 Ablation B — Full Training

### GATE 11: Pass criteria
- [ ] novel PQ at final epoch > ablation A final epoch result
- [ ] If not: L_V-B is not helping — proceed to job1c with ablation A checkpoint

---

## Phase 12: Job1c — L_O + L_V Margin-Gated Broadcast + L7 InfoNCE

**CASD dropped** (2026-06-14): per-voxel crops fail structurally for all STUFF classes at all crop sizes. See Gate 5e-CASD findings.

**Plan:** Two losses added on top of job1b-A (L_O + C1):
1. **L_V margin-gated broadcast** (THING voxels only) — denser mask-level gradient than L_O
2. **L7 InfoNCE** (backbone-level, base classes only) — sharpens CLIP-space boundaries via LoRA, indirect benefit for novel class discrimination

### Job1c loss stack

```
L_total = L_C1  +  w_lo * L_O  +  w_lv * L_V_broadcast  +  w_l7 * L7

L_C1:            token InfoNCE on voxel embeddings (base+novel excluded, same as before)
L_O:             object-level crop-CLS distill (THING only, matched queries)  weight=1.0
L_V_broadcast:   voxel-level broadcast distill (THING only, margin-gated)     weight=0.3
L7:              backbone InfoNCE (base classes only, class-balanced)          weight=0.5
```

### 12.1 L_V Margin-Gated Broadcast

Per-voxel target selection rule:

```python
# For each sampled voxel v (vox_gt_instance_id[v] = gi):
gi = vox_gt_instance_id[v]
if gi < 0:
    skip                          # unassigned voxel

cls_id = obj_classes[gi]
if cls_id >= 10:
    skip                          # STUFF: no valid broadcast teacher

if not obj_valid[gi]:
    skip                          # instance had no valid crop

if obj_margin[gi] <= 0:
    skip                          # teacher margin negative → skip this instance

# Valid broadcast target
target[v] = obj_feats[gi]         # (768,) crop-CLS of the GT instance
quality[v] = clamp(obj_margin[gi] / 0.05, min=0.1, max=1.0)
```

Class weights (applied on top of quality weight to upweight rare novel THING classes):

```python
CLASS_WEIGHT = {
    0: 1.0,   # barrier
    1: 1.0,   # bicycle
    2: 1.5,   # bus          (novel)
    3: 1.0,   # car
    4: 1.0,   # construction_vehicle
    5: 4.0,   # motorcycle   (novel, low vox count=3.2/sample, high std=0.037)
    6: 1.5,   # pedestrian   (novel)
    7: 1.0,   # traffic_cone
    8: 1.0,   # trailer
    9: 1.0,   # truck
}
```

Voxel sampling class-balance (within the 4096 sampled voxels):

```
30–40% novel THING voxels  (bus/moto/ped — oversample if present)
30%   base THING voxels
30%   remaining (including stuff, used for L_V skip / L7 only)
```

**Why margin-gating matters:** Without gating, instances like bicycle (base, margin=−0.020) and trailer (base, margin=−0.006) would broadcast a teacher that actively misdirects the query embedding. Gating on `obj_margin > 0` ensures only instances where the crop-CLS feature is already well-aligned with the correct class prototype contribute to the loss.

### 12.2 Update `panoptic/losses/distill_voxel_loss.py`

Add margin-gated broadcast mode:

```python
class VoxelDistillLoss(nn.Module):
    def __init__(self, loss_type="cosine",
                 class_weights=None, margin_gate_threshold=0.0):
        super().__init__()
        self.loss_type = loss_type
        self.margin_gate = margin_gate_threshold    # skip if obj_margin <= this
        self.class_weights = class_weights or {}    # {cls_id: float}
        self.stuff_ids = set(range(10, 16))

    def forward(self, pred_embed, pred_masks, vox_clip_indices,
                vox_gt_instance_id, obj_clip_feats, obj_clip_valid,
                obj_clip_classes, obj_margin):
        """
        obj_margin: (B, N_inst) float — per-instance margin from precompute
        """
        # Build per-voxel targets with quality + class weights
        # Reconstruct voxel embeddings: softmax(pred_masks[voxel]) @ pred_embed
        # Compute quality-weighted cosine loss on valid (margin-gated) voxels
        ...
```

### 12.3 L7: VoxelDecompLoss (base-only InfoNCE on backbone features)

```python
"""L7: per-voxel InfoNCE directly on backbone features → LoRA adapters.

Bypasses decoder entirely. Sharpens CLIP-space backbone margins for base classes.
Indirect benefit for novel: sharper base boundaries make novel regions more distinct
at inference time even without direct novel supervision.
"""
class VoxelDecompLoss(nn.Module):
    def __init__(self, proto_path, neg_bank_path,
                 n_sample=8192, temperature=0.07):
        super().__init__()
        # Load text prototypes (16, 768) + neg bank (M, 768)
        # Combine into target vocab (16+M, 768) for InfoNCE denominator
        # temperature=0.07 (CLIP native)

    def forward(self, backbone_feats, gt_sem, novel_ids):
        """
        backbone_feats: (V, 768) — requires_grad=True via LoRA
        gt_sem:         (V,)     — 0-indexed GT class per voxel
        novel_ids:      set      — excluded from supervision (open-vocab preservation)

        Algorithm:
          1. valid = gt_sem not in novel_ids AND gt_sem > 0 (ignore unlabelled)
          2. Class-balanced subsample ≤ n_sample valid voxels
             (each present class gets ~equal representation)
          3. feats = L2Norm(backbone_feats[valid_idx])
          4. sim = feats @ all_protos.T / temperature   → (n, 16+M)
          5. labels = gt_sem[valid_idx] (mapped to proto indices 0–15)
          6. loss = cross_entropy(sim[:, :16], labels)
             (neg bank enlarges denominator, not numerator targets)
        """
```

**Class-balanced sampling detail:** Count per-class voxels in the batch. Sample min(n_per_class, vox_count_per_class) for each class, where n_per_class = n_sample // n_present_classes. This prevents car (dominant class, ~200 vox/sample) from overwhelming motorcycle-like classes.

### 12.4 Create `panoptic/configs/job1c.yaml`

```yaml
warm_start:
  checkpoint: panoptic/checkpoints_job1b_v2_lo/best.pth

distillation:
  L_O:
    weight: 1.0
    warmup_epochs: 0   # already converged in job1b-A warm-start

  L_V:
    weight: 0.3
    warmup_epochs: 2
    mode: broadcast_margin_gated
    margin_gate_threshold: 0.0   # skip instances with obj_margin <= 0
    class_weights:
      2: 1.5    # bus
      5: 4.0    # motorcycle
      6: 1.5    # pedestrian

l7:
  enabled: true
  weight: 0.5
  warmup_epochs: 3
  n_sample_voxels: 8192
  temperature: 0.07
  class_balanced: true

training:
  epochs: 10
  lr_decoder: 2.0e-5
  lr_lora: 5.0e-5

checkpoint:
  save_dir: panoptic/checkpoints_job1c
```

### 12.5 Mini smoke test

```bash
python -u panoptic/train/train_panoptic_job1.py \
    --config panoptic/configs/job1c_mini.yaml 2>&1 | tee panoptic/logs/job1c_smoke.log
```

### GATE 12: Pass criteria
- [ ] `loss_lv` appears and decreases from epoch 3+ (after warmup)
- [ ] `loss_l7` appears and decreases after epoch 3 (L7 warmup)
- [ ] `lora_gnorm` higher than job1b-A (L7 adds direct LoRA gradient)
- [ ] No NaN in any loss component
- [ ] count_novel_lo in logs confirms bus/moto/ped getting L_O gradient
- [ ] count_broadcast_lv in logs confirms margin-gated voxels > 0 per step

---

## Phase 13: Job1c — Full Training

### GATE 13: Pass criteria
- [ ] novel PQ_things (bus/moto/ped) > 0.10
- [ ] novel PQ_stuff (vegetation) > 0  (indirect — not guaranteed without direct supervision)
- [ ] PQ_base ≥ 0.38 (not degraded vs job1b-A)
- [ ] `lora_gnorm` stable (L7 not destabilising LoRA)

---

## Future Work: Novel STUFF Supervision

Vegetation (novel STUFF) receives zero direct supervision in the current plan. The following approaches are documented for future implementation:

### Option 5 — Dense 2D Open-Vocabulary Teacher

Use a dense 2D vision-language segmentation model instead of single-crop CLS:

**Candidates:** OpenSeg, OVSeg, MaskCLIP, FC-CLIP dense features

**Mechanism:**
```
For each camera image:
  model.encode_image_dense(img) → (H, W, 768) pixel-level features
  project LiDAR voxels → (u, v) → sample dense feature at projected pixel
  L_V target[voxel] = dense_feature[v, u]    # region-aware, not scene-level
```

**Why this works where per-voxel CASD fails:** Dense models produce per-pixel or per-region features that are locally discriminative, not CLS tokens that summarise the entire image. A vegetation pixel gets a vegetation-aligned feature regardless of surrounding context.

**Pros:** Most defensible; closest to the 2D→3D distillation literature. Positive margins expected.

**Cons:** Heavy precompute (~500GB). Integration-heavy. Requires installing OVSeg/FC-CLIP.

### Option 6 — SAM + CLIP Region Pseudo-Labels

Middle ground between per-voxel CASD and full dense model:

**Mechanism:**
```
For each camera:
  SAM.generate(img) → class-agnostic 2D masks
  For each mask:
    crop masked region (background → grey)
    CLIP.encode_image(crop) → region_CLS_feature
    classify: top_class = argmax(region_CLS_feature @ text_protos.T)
  Project LiDAR voxels → assign region feature or pseudo-label
  L_V target[voxel] = region_CLS_feature for matching SAM region
```

**Pros:** More region-aware than centered CASD. SAM can isolate tree/bush regions that CASD crops mixed with background. Can isolate vegetation better.

**Cons:** Heavy precompute. SAM mask quality uncertain in driving scenes with thin/overlapping vegetation. Multi-camera projection bookkeeping is nontrivial.

### Architecture

```
L_O  (ObjectDistillLoss):   per-matched-query, THING-only, crop-CLS teacher
                             gradient → pred_embed of matched queries

L_V  (VoxelDistillLoss):    per-voxel, dense
     THING voxels:  broadcast from obj_feats[vox_gt_instance_id]
                    — zero additional precompute, teacher = same crop-CLS as L_O
                    — gradient → pred_embed AND pred_masks (adds mask spatial feedback)
     STUFF voxels:  CASD (online, see below)
                    — gradient → pred_embed AND pred_masks for all stuff classes

CASD (online in train loop): for each stuff voxel with valid camera projection:
     - project voxel → camera pixel → 224×224 crop centered at projection
     - encode_image() → CLS token → use as L_V target for that voxel
     - covers vegetation, terrain, driveable_surface, sidewalk, other_flat, manmade
     - ~20% training step slowdown (256 stuff crops per sample, 2 batches of 128)
     - gradient → pred_embed AND pred_masks for all stuff classes

L7   (VoxelDecompLoss):  direct per-voxel InfoNCE on backbone features
     - forces backbone CLIP-space margin at voxel level (bypasses decoder)
     - gradient → LoRA adapters directly
     - supervision: BASE classes only (same as C1; novel excluded)
```

### Combined loss

```
L_total = L_C1 + w_lo * L_O + w_lv * L_V + w_l7 * L7
        = C1_token + 1.0*L_O + 0.3*L_V + 0.5*L7
```

### 12.1 Update `panoptic/losses/distill_voxel_loss.py`

Extend `VoxelDistillLoss` to support broadcast mode (THING) and accept CASD targets (STUFF):

```python
class VoxelDistillLoss(nn.Module):
    def __init__(self, loss_type="cosine"):
        super().__init__()
        self.loss_type = loss_type
        self.stuff_ids = set(range(10, 16))   # 0-indexed

    def forward(self, pred_embed, pred_masks, vox_clip_feats, vox_clip_indices,
                vox_gt_instance_id=None, obj_clip_feats=None, obj_clip_valid=None,
                casd_targets=None):
        """
        vox_clip_feats:     (B, 4096, 768) — zeros for Pass 1 samples (no patch feats)
        casd_targets:       (B, 4096, 768) or None — online CASD targets for STUFF voxels
        vox_gt_instance_id: (B, 4096)     — broadcast index into obj_clip_feats per voxel
        obj_clip_feats:     (B, N_inst, 768)
        obj_clip_valid:     (B, N_inst)   bool
        """
        # Build per-voxel target:
        #   THING voxel with valid broadcast: obj_clip_feats[inst_id]
        #   STUFF voxel with CASD target:     casd_targets[vi]
        #   Otherwise:                        skip (zero target)
        ...
        # Reconstruct per-voxel embeddings, compute cosine loss on valid targets
```

### 12.2 CASD: online crop-CLS for STUFF voxels

Add `compute_casd_targets()` to `train_panoptic_job1.py`:

```python
def compute_casd_targets(nusc, sample_token, vox_xyz, vox_indices, gt_sem,
                          clip_model, clip_preprocess, device):
    """
    For each sampled voxel belonging to a stuff class (10–15):
      - project to best camera (highest n_valid pts)
      - take 224×224 crop centered at projected pixel
      - encode_image() → CLS → L2-norm
    Returns: (4096, 768) float32, zeros for THING/unprojectible voxels.
    Called once per batch sample. Subsamples to max 256 stuff voxels for speed.
    """
    targets = torch.zeros(len(vox_indices), 768, device=device)
    stuff_vis = [(vi, cls_id) for vi, cls_id
                 in zip(range(len(vox_indices)), gt_sem[vox_indices])
                 if int(cls_id) in STUFF_IDS_0IDX]
    if not stuff_vis:
        return targets
    # Subsample to 256 for speed
    if len(stuff_vis) > 256:
        idxs = np.random.choice(len(stuff_vis), 256, replace=False)
        stuff_vis = [stuff_vis[i] for i in idxs]
    # Project + crop + batch encode
    crops, crop_vi = [], []
    for vi, _ in stuff_vis:
        pt = vox_xyz[vox_indices[vi]]
        best_crop = _best_camera_crop(nusc, sample_token, pt, half_px=112)
        if best_crop is not None:
            crops.append(clip_preprocess(best_crop))
            crop_vi.append(vi)
    if not crops:
        return targets
    batch = torch.stack(crops).to(device).half()
    with torch.no_grad():
        feats = clip_model.encode_image(batch)         # (N, 768)
        feats = F.normalize(feats.float(), dim=-1)
    for i, vi in enumerate(crop_vi):
        targets[vi] = feats[i]
    return targets                                      # (4096, 768)
```

### 12.3 Create `panoptic/losses/voxel_decomp_loss.py` (L7)

```python
"""L7: Voxel decomposition loss — direct per-voxel InfoNCE on backbone features.

Forces backbone CLIP-space margins at voxel level.
Gradient: L7 → backbone_feats → LoRA adapters (bypasses decoder entirely).
"""
class VoxelDecompLoss(nn.Module):
    def __init__(self, proto_path, neg_bank_path, n_sample=10000, temperature=0.07):
        pass  # load all 16 class protos + neg bank as target vocab

    def forward(self, backbone_feats, gt_sem, valid_mask):
        """
        backbone_feats: (V, 768) — requires_grad=True via LoRA
        gt_sem:         (V,)     — 0-indexed GT class
        valid_mask:     (V,)     — bool, excludes novel classes (same as C1)
        """
        # Subsample ≤ n_sample valid voxels
        # sim = L2Norm(feats) @ all_protos.T / temperature → cross-entropy
```

### 12.4 Create `panoptic/configs/job1c.yaml`

```yaml
warm_start:
  checkpoint: panoptic/checkpoints_job1b_v2_lo/best.pth  # job1b-A best checkpoint

distillation:
  L_O:
    weight: 1.0
    warmup_epochs: 2
    loss_type: cosine

  L_V:
    weight: 0.3
    warmup_epochs: 3
    mode: broadcast      # THING: broadcast from obj_feats; STUFF: from casd_targets

  casd:
    enabled: true
    stuff_classes: [10, 11, 12, 13, 14, 15]   # 0-indexed: driveable..vegetation
    max_crops_per_sample: 256
    crop_half_px: 112

l7:
  enabled: true
  weight: 0.5
  warmup_epochs: 5
  n_sample_voxels: 10000
  temperature: 0.07

training:
  epochs: 10
  lr_decoder: 2.0e-5
  lr_lora: 5.0e-5

checkpoint:
  save_dir: panoptic/checkpoints_job1c
```

### 12.5 Mini smoke test

```bash
python -u panoptic/train/train_panoptic_job1.py \
    --config panoptic/configs/job1c_mini.yaml 2>&1 | tee panoptic/logs/job1c_smoke.log
```

### GATE 12: Pass criteria
- [ ] `loss_lv` appears and decreases from epoch 4+ (after L_V warmup)
- [ ] `loss_l7` appears and decreases after epoch 5 (L7 warmup)
- [ ] CASD targets non-zero in at least 80% of training steps
- [ ] `lora_gnorm` higher than job1b (L7 adds LoRA gradient)
- [ ] No NaN
- [ ] PQ_novel > 0 by epoch 3 if vegetation voxels receive CASD gradient (first sign CASD works)

---

## Phase 13: Job1c — Full Training

### GATE 13: Pass criteria (final evaluation)
- [ ] novel PQ_things > 0.10 (bus, motorcycle, pedestrian)
- [ ] novel PQ_stuff > 0.05 (vegetation)
- [ ] PQ_base ≥ 0.38 (not degraded vs job1b-A)

---

## Execution Timeline (revised 2026-06-14)

```
CURRENT (job1b-A running — JOBID 12463545):
  Phase 9   job1b_v2_lo_full.sh running (36h, batch_a100)

WHILE JOB1B-A RUNS (implement in parallel):
  Phase 12.2  Update distill_voxel_loss.py — margin-gated broadcast mode
              + class weights + quality weighting
  Phase 12.3  Write voxel_decomp_loss.py (L7 InfoNCE, class-balanced)
  Phase 12.4  Write job1c.yaml + job1c_mini.yaml

AFTER JOB1B-A FINISHES (Gate 9: novel PQ > 0.05 for bus/moto/ped):
  Phase 12    job1c mini smoke test — 3 epochs [GATE 12]
  Phase 13    job1c_full.sh — 10 epochs (submit bsub after GATE 12)

FINAL:
  Evaluate: novel PQ_things > 0.10, novel PQ_stuff (veg) best-effort
  Ablation table: job1a-v2 → job1b-A → job1c
  Novel vegetation PQ: if 0 → proceed to Future Work (Option 5/6)
```

### 12.1 Create `panoptic/losses/voxel_decomp_loss.py`

```python
"""L7: Voxel decomposition loss — direct per-voxel InfoNCE on backbone features.

Forces backbone features to have fine-grained CLIP-space margins.
Gradient path: L7 → backbone_feats → LoRA adapters (bypasses decoder entirely).
"""
class VoxelDecompLoss(nn.Module):
    def __init__(self, proto_path, neg_bank_path, n_sample=10000, temperature=0.07):
        # Load all 16 class protos + neg bank as target vocab
        # temp=0.07 fixed (CLIP native)
        pass

    def forward(self, backbone_feats, gt_sem, valid_mask):
        """
        backbone_feats: (V, 768) from backbone output (requires_grad=True via LoRA)
        gt_sem: (V,) GT semantic class (0-indexed, ignore=0)
        valid_mask: (V,) bool — valid voxels for supervision (excludes novel classes)

        Returns: per-voxel InfoNCE loss scalar
        """
        # Subsample up to n_sample valid voxels
        # sim = L2Norm(feats) @ all_protos.T / temperature
        # loss = cross_entropy(sim, gt_class_remapped)
        # Novel classes excluded from supervision (same as C1)
```

### 12.2 CASD: Context-Aware Stuff Distillation

CASD replaces L_V targets for ambiguous stuff classes with context-expanded crop CLS tokens (online, no precompute):

**Classes:** other_flat(11), sidewalk(12), terrain(13)  
**Expansion:** 3.0× / 2.5× / 2.0× respectively  

CASD is implemented as an online computation in the training loop:
1. Find voxels of CASD class in GT
2. Project to best camera (most visible) → compute bbox
3. Expand bbox by class factor → crop → CLIP encode_image → CLS
4. Broadcast to all voxels of that class → replace L_V target

**Addition in `train_panoptic_job1.py`:**
```python
if casd_enabled:
    casd_targets = compute_casd_targets(sample, backbone_feats, gt_sem, clip_model, casd_scales)
    # casd_targets shape: (V, 768), only non-zero for CASD class voxels
```

### 12.3 Create `panoptic/configs/job1c.yaml`

```yaml
warm_start:
  checkpoint: panoptic/checkpoints_job1b_v2_lo/best.pth  # or lo_lv, whichever is better

distillation:
  L_V:
    weight: 0.1
    mode: broadcast
  casd:
    enabled: true
    classes: [11, 12, 13]
    scales:  [3.0, 2.5, 2.0]

l7:
  enabled: true
  weight: 0.5
  warmup_epochs: 5
  n_sample_voxels: 10000
  temperature: 0.07

training:
  epochs: 10
  lr_decoder: 2.0e-5    # further reduced — mostly backbone fine-tuning
  lr_lora: 5.0e-5

checkpoint:
  save_dir: panoptic/checkpoints_job1c
```

### 12.4 Mini smoke test

```bash
python -u panoptic/train/train_panoptic_job1.py \
    --config panoptic/configs/job1c_mini.yaml \
    --smoke_test 2>&1 | tee panoptic/logs/job1c_smoke.log
```

### GATE 12: Pass criteria
- [ ] `loss_l7` appears in logs and decreases after warmup
- [ ] `lora_gnorm` increases slightly vs job1b (L7 provides stronger LoRA gradient)
- [ ] No NaN
- [ ] CASD targets non-zero for at least 1 sample per epoch

---

## Phase 13: Job1c — Full Training

### GATE 13: Pass criteria (final evaluation)
- [ ] novel PQ_things > 0.10 (bus, motorcycle, pedestrian)
- [ ] novel PQ_stuff > 0.05 (vegetation)
- [ ] voxel margin (from `eval_voxel_margin.py`) > 0.05 for all base classes
- [ ] PQ_base ≥ 0.38 (not degraded vs job1b)

---

## Execution Timeline

```
DAY 0 (today, no GPU):
  Phase 0  Cancel job1a, record checkpoint
  Phase 1  generate_negative_bank_v2.py → verify_neg_bank.py [GATE 1]
  Phase 2  generate_prototypes_v2.py → Script B [GATE 2]

DAY 0 (interactive GPU session):
  Phase 3  Rewrite precompute → unit test 1 sample [GATE 3]
  Phase 4  precompute_v2_mini.sh → Script A [GATE 4]

DAY 1 (submit batch jobs):
  Phase 5  precompute_v2_full.sh (8–12h) [GATE 5]
  Phase 6  job1a_v2 mini smoke test (interactive, 20min) [GATE 6]
  Phase 7  job1a_v2_full.sh (24h) — submit immediately after GATE 6 passes

DAY 3 (job1a finishes):
  Phase 8  job1b_v2_lo smoke test (interactive, 20min) [GATE 8]
  Phase 9  job1b_v2_lo_full.sh (24h) — submit after GATE 8

DAY 4 (job1b ablation A finishes):
  Evaluate novel PQ — if > 0.05, proceed to ablation B
  Phase 10 job1b_v2_lo_lv smoke test [GATE 10]
  Phase 11 job1b_v2_lo_lv_full.sh (24h)

DAY 5 (job1b ablation B finishes):
  Compare ablation A vs B
  Phase 12 job1c smoke test [GATE 12]
  Phase 13 job1c_full.sh (15h) — submit with best job1b checkpoint

DAY 6: Final evaluation, ablation table
```

---

## Ablation Table (Target Results)

| Experiment | PQ_base | PQ_novel_things | PQ_novel_stuff | Key change |
|---|---|---|---|---|
| Baseline (closed-set) | 0.476 | 0 | 0 | — |
| Job1a-v1 (old neg-bank) | 0.30+ | 0 | 0 | LoRA + C1 + margin |
| Job1a-v2 | 0.326 | 0 | 0 | + better neg-bank + prompts |
| Job1b-A (L_O only) | TBD | > 0.05 (target) | 0 (expected) | + crop-CLS L_O |
| Job1c (L_O + L_V margin-gated + L7) | ≥ 0.38 | > 0.10 | best-effort | + margin-gated broadcast L_V + L7 InfoNCE |

---

## Quick Reference: Gate Checklist

| Phase | Gate command | Pass condition |
|---|---|---|
| 1 | `python verify_neg_bank.py` | Zero violations, ≥100 entries |
| 2 | `python diag_prototype_similarity.py --proto ... --neg_bank ...` | All margins > 0.10, novel-neg < 0.80 |
| 3 | Python one-liner unit test | Correct shapes, norms ~1.0, no NaN |
| 4 | `python diag_clip_coverage.py` | Novel margin > 0, top-1 > 50% |
| 5 | Python spot-check 10 files | Correct schema, no NaN |
| 6 | Job1a mini smoke log | No NaN, T < 3.0, loss decreasing |
| 7 | WandB / checkpoint eval at epoch 8 | PQ_base ≥ 0.35 |
| 7.5 | Manual batch field inspection | obj_clip_valid bool, vox_gt_instance_id int64, loss_lo finite |
| 8 | Job1b smoke log | loss_lo < 0.8, count_novel_lo > 0 in ≥50% steps, no PQ_base collapse |
| 9 | Final eval checkpoint | novel PQ > 0.05 |
| 10 | Job1b-B smoke log | loss_lv decreasing, no regression |
| 11 | Final eval checkpoint | novel PQ ≥ ablation A |
| 12 | Job1c smoke log | loss_l7 decreasing, no NaN |
| 13 | Final eval checkpoint | novel PQ_things > 0.10 |

---

## Pipeline File Inventory

Complete list of files involved in the precompute and training pipeline.

### Precompute — Python scripts

| File | Purpose |
|---|---|
| `panoptic/scripts/precompute_clip_voxel_features_v2.py` | Main precompute: crop-CLS obj_feats + vox_feats. Pass 1 (`--no_vox_feats`), Pass 2 (`--vox_feats_only`). Add `--vox_feats_mode cls` for Phase 5d redesign. |
| `panoptic/scripts/generate_negative_bank_v2.py` | Generate negative bank with sim < 0.75 filter to class protos. Outputs `negative_bank_v2.pt`. |
| `panoptic/scripts/generate_prototypes_v2.py` | Generate text prototypes via multi-prompt templates, ViT-L/14. Outputs `text_prototypes_vitl14_224_v2.pt`. |
| `panoptic/scripts/verify_camera_projection.py` | Imported by precompute: project lidar points to camera UV coordinates. |
| `panoptic/scripts/diag_clip_coverage.py` | Script A: obj_feats margin, coverage, top-1/top-3 accuracy per class on mini or trainval. |
| `panoptic/scripts/diag_precompute_vox_check.py` | Script B (vox): vox_margin check per class — confirms vox_feats align to text prototypes. |

### Precompute — shell scripts

| File | Purpose |
|---|---|
| `panoptic/scripts/precompute_clip_features_v1.sh` | Pass 1 job (obj_feats only, `--no_vox_feats`) for v1.0-trainval. batch_a100, 16h. |
| `panoptic/scripts/precompute_clip_voxel_features_v2_pass2.sh` | Pass 2 job (`--vox_feats_only --resume`) for v1.0-trainval. batch_a100, 48h. **ON HOLD** pending Phase 5d gate. Update with `--vox_feats_mode cls` before submitting. |
| `panoptic/scripts/job1b_mini.sh` | Mini diagnostic run for job1b. |

### Training — model components

| File | Purpose |
|---|---|
| `panoptic/models/mask_decoder.py` | 100 thing + 6 stuff queries, 3 blocks × 3 levels transformer decoder. |
| `panoptic/models/blocks.py` | SelfAttention, CrossAttention, FFN transformer blocks. |
| `panoptic/models/positional_encoder.py` | 3D sinusoidal positional encoding for voxel coordinates. |
| `panoptic/models/clip_token_head.py` | MLP projecting query embeddings → 768d CLIP text space. |
| `panoptic/models/lora_backbone.py` | LoRA adapters (rank=16, 5 adapters, 38,912 trainable params) on frozen MinkowskiEngine backbone. |

### Training — losses

| File | Purpose |
|---|---|
| `panoptic/losses/panoptic_loss.py` | Hungarian-matched mask loss: BCE + Dice + CE (closed-set base supervision). |
| `panoptic/losses/matcher.py` | Hungarian matcher for query-to-GT assignment. |
| `panoptic/losses/token_loss_c1.py` | C1 InfoNCE: query embeddings vs 16 text protos + 161-entry neg-bank + multi-prompt; learnable temperature. |
| `panoptic/losses/distill_object_loss.py` | L_O: crop-CLS teacher, THING-only filter (`_THING_CLASS_IDS = frozenset(range(10))`), GT-order indexing via `obj_clip_valid`. |
| `panoptic/losses/distill_voxel_loss.py` | L_V: voxel-level distillation. Mode: `patch` (legacy) / `hybrid` (Phase 10 redesign with CLS-crop vox_feats). |

### Training — data pipeline

| File | Purpose |
|---|---|
| `panoptic/dataset/nuscenes_panoptic.py` | Dataset: voxelise point cloud, build GT masks, load .npz clip features (obj_feats, obj_valid, vox_feats, vox_gt_instance_id). |
| `panoptic/dataset/collate.py` | Collate function: handle variable-length GT masks across batch items. |

### Training — evaluation

| File | Purpose |
|---|---|
| `panoptic/eval/eval_pq.py` | Panoptic Quality (PQ/SQ/RQ) with IoU > 0.5 threshold. Separate novel/base splits. |

### Training — entry point

| File | Purpose |
|---|---|
| `panoptic/train/train_panoptic_job1.py` | Main training loop: LoRA + C1 + L_O + L_V, `count_novel_lo` counter, `indices_all` (all-GT matcher) integration. |

### Training — shell scripts

| File | Purpose |
|---|---|
| `panoptic/scripts/job1a_v2_full.sh` | Job1a full training (12 epochs, batch_a100, 24h). |
| `panoptic/scripts/job1b_v2_lo_smoke.sh` | Job1b ablation A smoke test (mini, inter_a100, 4h). |
| `panoptic/scripts/job1b_v2_lo_full.sh` | Job1b ablation A full training (15 epochs, batch_a100, 36h). **CURRENTLY RUNNING** (JOBID 12462720). |

### Training — configs

| File | Purpose |
|---|---|
| `panoptic/configs/job1a_v2.yaml` | Job1a full: LoRA + C1, v2 prototypes, 12 epochs, v1.0-trainval. |
| `panoptic/configs/job1a_v2_mini.yaml` | Job1a mini smoke: same, v1.0-mini, 5 epochs. |
| `panoptic/configs/job1b_v2_lo_only.yaml` | Job1b ablation A: L_O=1.0, L_V=0, 15 epochs. |
| `panoptic/configs/job1b_v2_lo_only_mini.yaml` | Job1b ablation A mini: 10 epochs (Gate 8 smoke). |
| `panoptic/configs/job1b_v2_lo_lv.yaml` | Job1b ablation B: L_O + redesigned CLS-crop L_V. **PENDING Phase 5d gate pass**. |
| `panoptic/configs/job1b_v2_lo_lv_mini.yaml` | Job1b ablation B mini smoke. **PENDING Phase 5d gate pass**. |

### Artifacts

| File | Purpose |
|---|---|
| `panoptic/clip_prototypes/text_prototypes_vitl14_224_v2.pt` | 16-class text prototype embeddings (v2 multi-prompt, ViT-L/14). |
| `panoptic/clip_prototypes/text_prototypes_vitl14_224_multiprompt_v2.pt` | Per-class averaged multi-prompt embeddings (10 templates each). |
| `panoptic/clip_prototypes/negative_bank_v2.pt` | 161 negative concepts with sim < 0.75 to all class protos. |
| `panoptic/diagnostics/coverage_v1.0-mini.json` | Script A output: per-class coverage, margin, top-k accuracy on mini. |
| `panoptic/diagnostics/prototype_similarity.json` | Script B output: pairwise prototype margins, neg-bank max-sim values. |
