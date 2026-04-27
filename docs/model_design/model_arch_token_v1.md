# Model Architecture v2.0 — CLIP Token Head (Phase 5)

## 1. Overview

Phase 5 extends the Phase 4 panoptic decoder (PQ=0.473) with a CLIP token head that predicts CLIP-space embeddings per query. This enables text-based mask classification alongside the existing linear head.

**Key addition**: Each decoder query now produces both:
- A class logit vector (17-dim, via `class_embed` — existing)
- A CLIP token embedding (768-dim, via `CLIPTokenHead` — new)

## 2. Architecture Diagram

```
Frozen DisNet Backbone (768-d)
         │
    ┌────┴────┐
    │backbone_proj (768→256)  mask_feat_proj (768→256)
    │         │                      │
    ▼         │                      │
  src (B,V,256)                mask_features (B,V,256)
         │
    ┌────┴────────────── Transformer Decoder (9 layers) ──┐
    │  cross-attn → self-attn → FFN (per layer)           │
    └──────────────────────────────────────────────────────┘
         │
    decoder_output = decoder_norm(output)  (B, Q, 256)
         │
    ┌────┼───────────────┐
    │    │                │
    ▼    ▼                ▼
class_embed    mask_embed     CLIPTokenHead  ← NEW
(256→17)       MLP(256→256)   MLP(256→512→768) + L2-norm
    │              │                │
pred_logits   pred_masks      pred_tokens (B, Q, 768)
(B,Q,17)      (B,V,Q)
```

## 3. CLIPTokenHead Module

```python
CLIPTokenHead(hidden_dim=256, token_dim=768, num_layers=3)
  └── MLP: Linear(256,512) → ReLU → Linear(512,768)
  └── F.normalize(dim=-1)  # L2 normalization
```

- **Input**: `decoder_output` (B, Q, 256) — same as class_embed and mask_embed
- **Output**: (B, Q, 768) — L2-normalized, directly comparable to CLIP text embeddings
- **Parameters**: 256×512 + 512×768 = 524,288 (~0.5M params, ~5% of decoder)

### Why 768-dim?
The frozen DisNet backbone was distilled from CLIP ViT-L/14@336px, which produces 768-dim embeddings. Text prototypes are also 768-dim. The token head must output in the same space for cosine similarity to be meaningful.

### Why L2 normalization?
CLIP embeddings are L2-normalized by design. Cosine similarity between normalized vectors equals their dot product, making the loss well-conditioned and the similarity scores interpretable.

## 4. Text Prototypes

Pre-computed using the same CLIP model (ViT-L/14@336px) and prompt template as the backbone distillation:

```python
prompt = "a {label} in a scene"  # matches util/util.py:extract_text_feature()
```

16 class prototypes (768-dim each), one per nuScenes panoptic class:
- barrier, bicycle, bus, car, construction vehicle, motorcycle, pedestrian,
  traffic cone, trailer, truck, driveable surface, other flat, sidewalk,
  terrain, manmade, vegetation

Stored at: `panoptic/clip_prototypes/text_prototypes_vitl14_336.pt`

## 5. Token Loss

### 5a. Cosine Similarity Loss (v1 — default)

```
L_token = mean(1 - cos_sim(pred_token_i, text_prototype[gt_class_i]))
```

- Applied only to Hungarian-matched queries (same matching as panoptic loss)
- Unmatched queries (no-object) are excluded — no supervision for no-object in token space
- **Linear warmup**: weight scales from 0 to `token_weight` over `warmup_steps` steps

**Pros**: Simple, directly optimizes inference metric (cosine similarity), smooth gradients.
**Cons**: Only pulls toward GT prototype, doesn't push away from wrong classes. Token loss magnitude (~0.02) is negligible compared to panoptic loss (~34), so gradients barely influence decoder queries.

### 5b. Contrastive Loss / InfoNCE (v2 — `token.loss_type: contrastive`)

```
L_contrastive = -log( exp(sim(pred, gt_proto) / τ) / Σ_c exp(sim(pred, proto_c) / τ) )
```

This is cross-entropy over cosine similarities to ALL class prototypes, with learnable temperature τ.

- **Positive**: GT class prototype (pull toward)
- **Negatives**: All other 15 class prototypes (push away)
- **Temperature**: Learnable `log_temperature` parameter (initialized from τ=0.07, CLIP default). Clamped at max=100 for stability. Higher temperature → sharper distribution → stronger gradients.
- **Applied to**: Hungarian-matched queries only (same as cosine loss)

**Why this is better:**
1. Every wrong class contributes a repulsive gradient — much stronger signal than cosine loss
2. Cross-entropy loss has natural magnitude scaling (CE ~ 2-3 for 16 classes) — no longer negligible vs panoptic losses
3. Learnable temperature adapts the sharpness during training
4. Well-studied in CLIP literature — this IS how CLIP itself is trained

**Config**: `token.loss_type: contrastive` (default: `cosine` for backward compatibility)

### 5c. Auxiliary Token Loss (v2 — `token.aux_token_loss: true`)

Applies token loss (cosine or contrastive) to **all 9 intermediate decoder layers** in addition to the final layer. This means the token gradient flows into the decoder at every level, not just the last one.

- Each aux layer has its own `pred_tokens` (computed by `token_head(decoder_norm(output_i))`)
- Each aux layer gets its own Hungarian matching (already done for panoptic aux losses)
- Total token loss = final_layer_token_loss + Σ aux_layer_token_losses

**Why this matters:** Without aux token loss, only the final decoder layer receives token gradients. The earlier layers learn purely from panoptic losses, so their representations are optimized for mask prediction, not CLIP alignment. Aux token loss ensures ALL decoder layers produce CLIP-friendly representations.

**Config**: `token.aux_token_loss: true` (default: `false` for backward compatibility)

### How GT is obtained
No new annotations. For matched query → GT class `c`:
1. Hungarian matching gives `(query_idx, gt_mask_idx)` pairs
2. GT class index from `targets["classes"][gt_mask_idx]`
3. GT prototype = `text_prototypes[class_idx]` (pre-computed, frozen)

## 6. Training

### Warm-start
- Load decoder weights from `panoptic/checkpoints_full/best.pth` (PQ=0.473)
- `strict=False`: missing `token_head.*` keys are expected (initialized randomly)
- Fresh optimizer (new learning rate, new momentum buffers)

### Loss function
```
L_total = L_panoptic + w(t) * L_token

where:
  L_panoptic = existing CE + BCE + Dice losses (weighted)
  w(t) = min(1, t/warmup_steps) * token_weight_max
```

### Hyperparameters (mini)
| Parameter | Value |
|---|---|
| lr | 5e-5 |
| epochs | 10 |
| token_weight | 0.5 |
| warmup_steps | 100 |
| grad_accum | 4 |

### Hyperparameters (full v1 — warm-start from mini checkpoint)
| Parameter | Value |
|---|---|
| lr | 5e-5 (uniform) |
| epochs | 20 |
| token_weight | 0.5 |
| warmup_steps | 500 |
| grad_accum | 8 |
| warm_start | `checkpoints/best.pth` (mini, PQ=0.24) |

### Hyperparameters (full v2 — warm-start from full checkpoint)
| Parameter | Value | Rationale |
|---|---|---|
| lr (decoder) | 1e-5 | Lower to preserve PQ=0.473 mask quality |
| lr (token_head) | 5e-5 | Higher for fresh token head params |
| epochs | 20 | |
| token_weight | 1.0 | Aggressive CLIP alignment |
| loss_type | contrastive | InfoNCE — stronger gradients from all-class repulsion |
| temperature | 0.07 (learnable) | CLIP default, adapts during training |
| aux_token_loss | true | Token loss on all 9 aux decoder layers |
| warmup_steps | 500 | |
| grad_accum | 8 | |
| warm_start | `checkpoints_full/best.pth` (full, PQ=0.473) | Proper starting point |

**Differential LR**: Controlled by `training.token_head_lr` in YAML config. When set, creates two optimizer param groups. When absent, falls back to uniform LR (backward compatible).

**Learnable temperature**: The `log_temperature` parameter is included in the token_head optimizer param group. It is trained alongside the token head MLP.

## 7. Inference & Evaluation Modes

Five evaluation modes, all computed from a single forward pass:

```
backbone → feats (V, 768) → decoder → pred_logits, pred_masks, pred_tokens
                │                              │              │            │
                │                   ┌──────────┼──────────────┼────────────┘
                │                   │          │              │
                ▼                   ▼          ▼              ▼
       [E] Backbone pool    [A] Linear   [B] Token-16  [C] Ensemble
       feats per mask       softmax      cos sim×16    alpha sweep
       cos sim × 22         → 16 cls     → 16 cls      → 16 cls
       → 22 cls                    │
                                   ▼
                            [D] Token-OpenVocab
                            cos sim × 22
                            → 22 cls (16 known + 6 novel)
```

### (A) Linear Head Only — Closed-Set Baseline
`argmax(class_embed(decoder_output).softmax(-1))` — same as Phase 4. 16 known classes only.

### (B) Token-16 — Token Head on Known Classes
`argmax(cos_sim(pred_tokens, text_prototypes_16))` — classify by CLIP similarity to 16 known prototypes. No explicit no-object class; low-similarity queries filtered by score threshold (>0.1).

### (C) Ensemble — Best of Both
```
score = α * linear_softmax[:C] + (1-α) * normalize(cos_sim_16)
label = argmax(score)
```
Alpha sweep: `[0.0, 0.3, 0.5, 0.7, 1.0]`. Reports best alpha.

### (D) Token-OpenVocab — Main Open-Vocabulary Test ⭐
```
cos_sim = pred_tokens @ text_prototypes_22.T    # (Q, 22)
scores, labels = cos_sim.max(-1)                # labels ∈ {0..21}
```
This is the **primary open-vocab evaluation**. The token head learned to produce CLIP-space embeddings per query during training on 16 classes. At inference, we compare against 22 prototypes (16 known + 6 novel). If the token head truly learned CLIP alignment (not just a 16-class classifier), queries overlapping novel objects should get high similarity to novel prototypes.

**Novel classes**: `["pole", "bench", "trash can", "fire hydrant", "tree trunk", "curb"]` — 6 driving-relevant objects not in nuScenes 16.

For known-class PQ comparison: predictions with label ≥16 are remapped to ignore (0).

### (E) Backbone-Pooling — Open-Vocab Lower Bound
For each predicted mask, mean-pool raw backbone features (768-dim, frozen CLIP) within the mask region, L2-normalize, then cosine sim to 22 prototypes.

This requires **no learned token head** — it tests whether the backbone's frozen CLIP features alone can identify classes. Serves as a lower bound: if Mode D novel detections >> Mode E novel detections, the token head learned generalizable CLIP alignment beyond what the backbone provides for free.

### Summary Table (example output)
```
Mode                    PQ      SQ      RQ      mIoU    Novel Voxels
─────────────────────────────────────────────────────────────────────
A. Linear (16-cls)      0.473   ...     ...     ...     N/A
B. Token-16             0.467   ...     ...     ...     N/A
C. Ensemble (α=0.5)     0.478   ...     ...     ...     N/A
D. Token-OpenVocab(22)  0.465   ...     ...     ...     12,345
E. Backbone-Pool (22)   0.320   ...     ...     ...     8,901
```

**Evaluation script**: `panoptic/scripts/evaluate_comprehensive.py` (see §10 File Map)

## 8. WandB Logging

### Training Metrics (logged every `log_every` steps)
| Key | Description |
|---|---|
| `train/loss` | Total training loss (panoptic + token) |
| `train/loss_ce` | Cross-entropy classification loss |
| `train/loss_mask` | Binary mask loss |
| `train/loss_dice` | Dice loss |
| `train/loss_token` | Token loss (cosine or contrastive) |
| `train/aux_loss` | Sum of all auxiliary decoder layer losses |
| `train/temperature` | Learnable temperature τ (contrastive only) |
| `train/lr` | Decoder learning rate |
| `train/lr_token` | Token head learning rate (differential LR only) |

### Validation Metrics (logged every `eval_every` epochs)
| Key | Description |
|---|---|
| `val/PQ`, `val/SQ`, `val/RQ`, `val/mIoU` | Linear head metrics |
| `val/PQ_token`, `val/mIoU_token` | Token head metrics |

### Artifacts
Both training scripts (`train_panoptic.py`, `train_panoptic_token.py`) log checkpoints as wandb artifacts:
- `best-checkpoint` — saved when val PQ improves, with metadata `{epoch, PQ}`
- `latest-checkpoint` — saved every `save_every` epochs, with metadata `{epoch, best_pq}`

## 9. Expected Improvements

| Aspect | Phase 4 | Phase 5 (expected) |
|---|---|---|
| Classification | Linear head only | Linear + CLIP-aligned tokens |
| mIoU gap | 0.433 (vs 0.71 MaskPLS) | Should improve — CLIP features are more discriminative |
| Rare classes | barrier PQ=0.227 | May improve via CLIP's pre-trained class knowledge |
| Open-vocab potential | None | Token head enables text-based open-vocabulary (Mode D) |

### Key hypothesis
mIoU (0.433) was the primary bottleneck in Phase 4. The linear head had to learn class boundaries from scratch on 16 classes. The token head leverages CLIP's pre-trained semantic understanding, which should improve classification accuracy especially for underperforming classes.

## 10. Potential Improvements

1. **Multi-prompt prototypes**: Average embeddings from multiple prompts per class (e.g., "a photo of a {}", "a {} on the road", "a {}")
2. **Learnable prototype refinement**: Fine-tune text prototypes during training (small LR)
3. ~~**Temperature scaling**~~: ✅ Implemented in v2 — learnable log-temperature (CLIP-style)
4. ~~**Auxiliary token losses**~~: ✅ Implemented in v2 — token loss on all 9 aux decoder layers
5. ~~**Contrastive loss**~~: ✅ Implemented in v2 — InfoNCE replaces cosine loss
6. **Open-vocabulary backbone pooling**: Pool raw 768-dim backbone features per predicted mask for zero-shot classification (see `docs/open_vocab_ablation.md`)
7. **Class-agnostic training (Option B)**: Remove CE loss, keep mask+dice only, for unbiased mask proposals

## 11. File Map

| File | Purpose |
|---|---|
| `panoptic/models/clip_token_head.py` | CLIPTokenHead module |
| `panoptic/losses/token_loss.py` | Token loss (cosine or contrastive, with aux support) |
| `panoptic/scripts/generate_text_prototypes.py` | Text prototype generation (16 known classes) |
| `panoptic/train/train_panoptic.py` | Base training script (linear head only, wandb artifacts) |
| `panoptic/train/train_panoptic_token.py` | Token training script (differential LR, aux token loss, wandb artifacts) |
| `panoptic/scripts/evaluate_token.py` | Quick evaluation (3 modes: linear, token, ensemble) |
| `panoptic/scripts/evaluate_comprehensive.py` | Full evaluation (5 modes incl. open-vocab, per-class tables, JSON output) |
| `panoptic/scripts/push_to_hf.py` | Push checkpoints, configs, docs to HuggingFace Hub |
| `panoptic/configs/mini_token.yaml` | Mini config |
| `panoptic/configs/full_nuscenes_token.yaml` | Full config v1 (cosine, uniform LR) |
| `panoptic/configs/full_nuscenes_token_v2.yaml` | Full config v2 (contrastive, differential LR, aux) |
| `panoptic/scripts/train_token_job.sh` | bsub job script |
| `panoptic/clip_prototypes/` | Cached text prototypes (16-class) |
| `panoptic/checkpoints_token/` | Mini token checkpoints |
| `panoptic/checkpoints_token_full_v1/` | Full v1 token checkpoints |
| `panoptic/checkpoints_token_full_v2/` | Full v2 token checkpoints |
| `docs/model_design/` | Architecture documentation |
| `docs/open_vocab_ablation.md` | Open-vocabulary evaluation design (Option A + B) |

## 12. Experimental Results

### v1 — Full nuScenes (warm-start from mini checkpoint PQ=0.24)

**Note**: This run used the mini checkpoint as warm-start because the full checkpoint was lost. Results are artificially low — the decoder had to re-learn mask quality AND learn token alignment simultaneously.

| Metric | Phase 4 (linear only) | v1 Linear Head | v1 Token Head |
|---|---|---|---|
| PQ | 0.473 | 0.429 | 0.410 |
| SQ | 0.793 | 0.772 | — |
| RQ | 0.580 | 0.534 | — |
| mIoU | 0.433 | 0.405 | 0.384 |

**Observations:**
- Linear PQ regressed from 0.473 → 0.429 (weak warm-start + uniform LR disrupted decoder)
- Token PQ=0.410 trails linear by ~0.02 — CLIP classification works but slightly weaker
- Token loss converged very early (~0.02 by 100k steps) — mapping to CLIP space is learnable
- SQ=0.772 (mask quality) still strong, regression mainly in RQ (classification accuracy)

### v2 — Full nuScenes (warm-start from full checkpoint PQ=0.473) — COMPLETE

**Training config**: `full_nuscenes_token_v2.yaml` (contrastive InfoNCE, differential LR, aux token loss, 20 epochs)

#### Comprehensive Evaluation (all 5 modes, v1.0-trainval val set)

| Mode | PQ | SQ | RQ | mIoU | Novel Voxels |
|---|---|---|---|---|---|
| A. Linear | 0.4759 | 0.7912 | 0.5841 | 0.4311 | N/A |
| B. Token-16 | 0.4684 | 0.7939 | 0.5751 | 0.4190 | N/A |
| C. Ensemble (α=0.0 best) | 0.4684 | 0.7939 | 0.5751 | 0.4190 | N/A |
| D. Token-OpenVocab (22) | 0.4684 | 0.7939 | 0.5751 | 0.4190 | 0 |
| E. Backbone-Pool (22) | 0.2090 | 0.6655 | 0.2479 | 0.1688 | 25,325,178 |

#### Per-class comparison (Linear vs Token)

| Class | Linear PQ | Token PQ | Winner | Linear IoU | Token IoU | Winner |
|---|---|---|---|---|---|---|
| barrier | 0.2303 | **0.2623** | Token | 0.0604 | **0.0636** | Token |
| bicycle | **0.3183** | 0.2095 | Linear | **0.0377** | 0.0237 | Linear |
| bus | 0.4587 | **0.4640** | Token | **0.6643** | 0.6513 | Linear |
| car | **0.8485** | 0.8475 | ~Tie | **0.7726** | 0.7446 | Linear |
| construction_vehicle | 0.3079 | **0.3330** | Token | 0.2352 | **0.2423** | Token |
| motorcycle | **0.6391** | 0.5484 | Linear | **0.4422** | 0.3282 | Linear |
| pedestrian | **0.7394** | 0.7217 | Linear | **0.3820** | 0.3176 | Linear |
| traffic_cone | **0.5691** | 0.5636 | ~Tie | **0.1843** | 0.1678 | Linear |
| trailer | 0.2695 | **0.3240** | Token | 0.2621 | **0.2989** | Token |
| truck | **0.4858** | 0.4762 | Linear | **0.5816** | 0.5514 | Linear |
| driveable_surface | **0.8566** | 0.8530 | ~Tie | **0.8620** | 0.8588 | ~Tie |
| other_flat | 0.0694 | **0.1352** | Token | 0.1269 | **0.2282** | Token |
| sidewalk | 0.2111 | 0.2098 | ~Tie | **0.3100** | **0.3891** | Token |
| terrain | **0.2531** | 0.1944 | Linear | **0.4654** | 0.3573 | Linear |
| manmade | **0.6739** | 0.6628 | Linear | **0.7662** | 0.7318 | Linear |
| vegetation | 0.6843 | **0.6890** | Token | 0.7444 | **0.7496** | Token |

#### Key Observations

1. **Linear PQ=0.476 vs Phase 4 PQ=0.473** — minimal regression (+0.003), differential LR preserved decoder quality ✅
2. **Token PQ=0.468** — competitive, only 0.008 below linear. Token classification works.
3. **Token wins on rare/hard classes**: barrier (+0.032), construction_vehicle (+0.025), trailer (+0.055), other_flat (+0.066). CLIP prototypes help where linear head has few training samples.
4. **Linear wins on common classes**: car, pedestrian, motorcycle, truck. Linear head has enough training data to outperform CLIP similarity.
5. **Best ensemble at α=0.0** — pure token mode. Blending hurts because linear scores and cosine scores are on different scales.
6. **Token novel=0** — contrastive loss collapsed token head to 16-class classifier. Not truly open-vocabulary.
7. **Backbone-pool PQ=0.209** — raw backbone features too noisy for classification (25M voxels as "pole").
8. **SQ slightly higher for token (0.794 vs 0.791)** — token head doesn't disturb mask quality.

#### v1 vs v2 comparison

| Metric | v1 Linear | v1 Token | v2 Linear | v2 Token |
|---|---|---|---|---|
| PQ | 0.429 | 0.410 | **0.476** | **0.468** |
| SQ | 0.772 | — | **0.791** | **0.794** |
| RQ | 0.534 | — | **0.584** | **0.575** |
| mIoU | 0.405 | 0.384 | **0.431** | **0.419** |

v2 improvements worked: proper warm-start + differential LR + contrastive loss → +0.047 PQ linear, +0.058 PQ token.

### Config files
| Config | Purpose |
|---|---|
| `full_nuscenes_token.yaml` | v1 config (uniform LR, token_weight=0.5) |
| `full_nuscenes_token_v2.yaml` | v2 config (differential LR, token_weight=1.0) |
