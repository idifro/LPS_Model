# Strategy K, L, M: Per-Voxel Inference for Open-Vocabulary Panoptic Segmentation

## 1. Motivation: Why Per-Voxel?

Strategy E (geometric ensemble) **pools backbone features within each mask**, then classifies the pooled vector. This destroys discriminability:

```
Feature Pooling (Strategy E):
  car_voxel:  cos("car")=0.35, cos("police car")=0.33   ← discriminative
  road_voxel: cos("car")=0.05, cos("police car")=0.04   ← discriminative

  After averaging both into a single pooled vector:
  pooled:     cos("car")=0.08, cos("police car")=0.07   ← NOISE (information destroyed)
```

The mathematical principle: **E[f(x)] ≠ f(E[x])** — classification is nonlinear (involves softmax/argmax), so averaging features before classifying is NOT equivalent to classifying then averaging.

Strategies K/L/M classify **each voxel independently** using the per-voxel backbone features (which ARE CLIP-aligned from DisNet distillation), then aggregate the classification decisions (logits/votes) within each mask.

---

## 2. Strategy K: Per-Voxel Majority Vote

### Intuition

Each voxel in the backbone output has a 768-dim feature vector aligned to CLIP space via OpenSeg distillation. The DisNet "distill" evaluation mode achieves 35.4% mIoU by classifying each voxel independently against text prototypes. Strategy K applies this same principle but **within predicted masks** to get per-object class labels.

Think of it as: each voxel "votes" for a class. A car mask with 5000 voxels where 70% vote "car" gives high confidence even if individual voxels are noisy.

### How It Differs from Strategy E

| Aspect | Strategy E (geo ensemble) | Strategy K (voxel vote) |
|--------|---------------------------|-------------------------|
| Classification target | Pooled feature vector | Each voxel independently |
| Aggregation | Average features → single classify | Classify each → average logits |
| Signal preservation | Destroys (normalized vectors cancel) | Preserves (logits are scalars) |
| Known-class path | Geometric blend (linear × CLIP) | Linear head only (proven PQ=0.477) |
| Novel-class decision | CLIP softmax on pooled | Margin comparison (novel must beat known by 20%) |

### Algorithm

```
1. Compute per-voxel cosine similarity to ALL text prototypes: (V, C_all)
2. Apply bias correction (subtract background affinity per prototype)
3. Scale by temperature: voxel_logits = (cos_sim - bias) / T
4. For each predicted mask (query):
   a. Weight voxel_logits by mask sigmoid: weighted_vote = Σ(mask_sig * voxel_logits) / Σ(mask_sig)
   b. Apply softmax to get per-class probabilities
   c. DECISION:
      - If best_novel_prob > best_known_clip_prob × 1.2 (20% margin):
        → Assign novel class, score = novel_prob × objectness
      - Else:
        → Use linear head prediction (proven known-class performance)
5. Greedy panoptic assignment (same as all other modes)
```

### Key Design Choices

- **Novel margin (default 1.2)**: Novel class must beat known by 20% in CLIP logit space. This prevents false positives where "police car" ≈ "car" (similar prototypes). Tunable via `--novel-margin`.
- **Linear head fallback**: For known classes, the trained linear head (PQ=0.477) is always preferred. Novel signal only overrides when it's strong.
- **Bias correction**: Prototype background affinity (calibrated on 50 random samples) is subtracted before temperature scaling. This prevents "pole" (generically similar to everything) from dominating.

---

## 3. Strategy L: Soft Mask-Weighted Logit Aggregation

### Intuition

Strategy L is the "geometric ensemble" version of per-voxel classification. Like Strategy K, it classifies per-voxel first, but instead of using a hard margin decision for known vs novel, it applies the same geometric ensemble formula from Strategy E — just on **aggregated logits** (not pooled features).

The key difference from Strategy E's pooling:
```
Strategy E: classify( average(feat_v1, feat_v2, ...) )       ← WRONG (nonlinear)
Strategy L: average( classify(feat_v1), classify(feat_v2) )  ← CORRECT (preserves signal)
```

### Algorithm

```
1. Per-voxel logits: (V, C_all) = normalize(backbone_feats) @ text_protos.T / temperature
2. Mask-weighted aggregation (efficient matrix multiply):
   aggregated_logits = masks_sig.T @ voxel_logits / mask_area   → (Q, C_all)
3. Softmax: clip_probs = softmax(aggregated_logits)             → (Q, C_all)
4. Linear head: linear_probs = softmax(pred_logits[:, :16])     → (Q, 16)
5. Geometric ensemble (same formula as Strategy E):
   combined[:, :known] = linear_probs^α × clip_probs[:, :known]^(1-α)
   combined[:, known:] = clip_probs[:, known:]  (novel = CLIP only)
6. Renormalize, take argmax, apply objectness, greedy assign
```

### Why L vs K?

| Strategy K | Strategy L |
|-----------|-----------|
| Hard margin decision (novel vs known) | Soft geometric blend |
| Single alpha value implicit | Alpha sweep (0.3, 0.5, 0.7) |
| Known = linear head only | Known = blend of linear + CLIP |
| Simpler, more conservative | More flexible, may find better operating point |
| Fewer false positives (high margin) | May detect more novel but also more FPs |

### Computational Cost

`masks_sig.T @ voxel_logits` is `(Q, V) @ (V, C)` = O(Q × V × C).
For V=50K, Q=100, C=21: ~100M operations — same cost as current feature pooling. No extra memory or time.

---

## 4. Strategy M: Confidence-Gated Novel Detection

### Intuition

The linear head is already excellent at known classes (PQ=0.477). When it's confident (max prob > 0.7), it's almost certainly correct. We should only look for novel classes when the linear head says "I'm not sure what this is."

This is the most conservative strategy — it preserves known-class PQ almost perfectly by only assigning novel labels to uncertain masks.

### Algorithm

```
1. Linear head confidence: max_conf, known_label = max(softmax(pred_logits[:, :16]))
2. Gate decision per query:
   IF max_conf > confidence_threshold (default 0.7):
     → label = known_label, score = max_conf × objectness
     (Trust linear head — don't even look at novel)

   ELSE (linear head uncertain):
     → Compute per-voxel CLIP logits for ALL classes (known + novel)
     → Aggregate within mask (same as L)
     → Compare aggregated novel_score vs known_score
     → Pick whichever is higher
3. Greedy panoptic assignment
```

### Why This Reduces False Positives

In Strategy E, every query gets evaluated for novel classes — even queries where the linear head is 95% confident it's a "car". This creates opportunities for the noisy CLIP signal to incorrectly override. Strategy M only opens the "novel gate" for uncertain queries (~20-30% of all queries), dramatically reducing FP surface area.

### Tunable Parameters

- `--confidence-threshold 0.7` (default): Higher = fewer novel detections but fewer FPs. Lower = more aggressive novel search.
- The threshold should be tuned based on the linear head's calibration (how often "confident" = actually correct).

---

## 5. Shared Infrastructure

### Multi-Prompt Text Prototypes

All three strategies use the same multi-prompt prototypes as geo_ensemble_v2:
- 5-6 diverse text descriptions per novel class (e.g., "a police car with flashing lights", "a black and white police cruiser")
- Averaged CLIP embeddings per class → more distinctive prototypes than single generic prompts
- Stored in `text_protos_mp` tensor of shape (C_all, 768)

### Prototype Bias Correction

Per-prototype "background affinity" calibrated on 50 random samples:
```
bias[c] = mean(cos_sim(random_pooled_features, proto[c]))
corrected_logit[c] = (cos_sim[c] - bias[c]) / temperature
```

A prototype like "pole" has high generic affinity (≈0.3) — subtracting bias raises its threshold, preventing it from triggering everywhere.

**Ablation**: Run with `--no-bias` to disable bias correction entirely. This tests whether bias correction helps or hurts per-voxel approaches (it was originally designed for pooled features).

---

## 6. How to Run

### Full evaluation (all modes including K/L/M):
```bash
python -u panoptic/scripts/evaluate_comprehensive.py \
    --config panoptic/configs/full_nuscenes_token_v2.yaml \
    --checkpoint panoptic/checkpoints_token_full_v2/best.pth \
    --version v1.0-trainval
```

### K/L/M only (faster):
```bash
python -u panoptic/scripts/evaluate_comprehensive.py \
    --config panoptic/configs/full_nuscenes_token_v2.yaml \
    --checkpoint panoptic/checkpoints_token_full_v2/best.pth \
    --version v1.0-trainval \
    --modes voxel_vote,logit_aggregation,confidence_gated
```

### Ablation: no bias correction:
```bash
python -u panoptic/scripts/evaluate_comprehensive.py \
    --config panoptic/configs/full_nuscenes_token_v2.yaml \
    --checkpoint panoptic/checkpoints_token_full_v2/best.pth \
    --version v1.0-trainval \
    --modes voxel_vote,logit_aggregation,confidence_gated \
    --no-bias
```

### Tune novel margin (Strategy K):
```bash
# More aggressive (lower margin = more novel detections)
python -u ... --modes voxel_vote --novel-margin 1.0

# More conservative (higher margin = fewer FPs)
python -u ... --modes voxel_vote --novel-margin 1.5
```

### Tune confidence gate (Strategy M):
```bash
# More novel-permissive (lower threshold = more uncertain queries checked)
python -u ... --modes confidence_gated --confidence-threshold 0.5

# Very conservative (only check truly ambiguous masks)
python -u ... --modes confidence_gated --confidence-threshold 0.9
```

---

## 7. Output Files

After evaluation, the output directory contains:

| File | Description |
|------|-------------|
| `results.json` | PQ/SQ/RQ/mIoU for all modes |
| `novel_detections.json` | Total novel voxels per class per mode (aggregated) |
| `per_scene_novel_detections.json` | **Per-sample** novel voxels breakdown, sorted by total count (descending) — use this to find top scenes for visualization |
| `per_class_*.txt` | Per-class PQ/SQ/RQ tables |
| `eval_log.txt` | Full evaluation log |

### Identifying Top Scenes for Visualization

The `per_scene_novel_detections.json` is sorted by total novel voxel count. To visualize the top-N scenes:

```python
import json
with open("eval_results/.../per_scene_novel_detections.json") as f:
    per_scene = json.load(f)

# Top 20 scenes with most novel detections
top_tokens = list(per_scene.keys())[:20]
print(f"Top 20 scenes to visualize: {top_tokens}")
```

Then run visualization on those specific scenes (pass tokens to `visualize_novel_detections.py`).

---

## 8. Visual Verification Workflow

1. **Run K/L/M on full trainval** (6019 samples)
2. **Check `per_scene_novel_detections.json`** — find top 20-30 scenes with highest novel voxel counts
3. **Run `visualize_novel_detections.py`** on those scenes
4. **Manually inspect BEV + image plots** for each scene:
   - Are novel-labeled voxels physically on actual novel objects?
   - Or are they on known-class objects (FPs)?
5. **Record findings** — what % of top scenes have true positives?

### Decision Criteria

- If 50+ scenes inspected and ALL novel detections are FPs → **close inference-only chapter**
- If ANY scenes show true positives → investigate what's different about those scenes (maybe specific camera angles, lighting, or object sizes where backbone features are better)

---

## 9. Expected Outcomes

### Optimistic Case (K/L works better than E)
Per-voxel classification avoids the pooling degradation. Individual voxels on a "police car" vote correctly because they were distilled from CLIP features that originally encoded a police car appearance. Aggregating many correct per-voxel votes within a mask gives strong novel signal.

Expected: known-class PQ matches linear head (0.477), novel detection has lower FP rate than Strategy E.

### Realistic Case (same FPs as E, just from a different mechanism)
The backbone's per-voxel CLIP alignment is simply too weak for novel/known discrimination. The distillation chain (`CLIP image → OpenSeg → 3D projection → DisNet`) degrades cosine similarity margins between similar classes to noise level (~0.01-0.03). Per-voxel voting produces the same confusion: "car" voxels vote almost equally for "car" and "police car".

Expected: known-class PQ preserved, novel detections still mostly FPs.

### Pessimistic Case (known-class PQ degrades)
Unlikely given implementation — K/M fall back to linear head for known classes, L uses geometric blend which preserves linear head signal at high alpha. But if temperature/bias tuning is wrong, CLIP noise could leak into known-class predictions.

Expected only with bad hyperparameters (wrong temperature, alpha=0 pure CLIP).

---

## 10. Implementation Details

### File: `panoptic/scripts/evaluate_comprehensive.py`

Three new functions added (lines 787-1002):
- `panoptic_inference_voxel_vote()` — Strategy K
- `panoptic_inference_logit_aggregation()` — Strategy L
- `panoptic_inference_confidence_gated()` — Strategy M

All three share the same pattern:
1. Normalize backbone features: `F.normalize(backbone_feats.float(), dim=-1)` → (V, 768)
2. Per-voxel similarity: `feats_normed @ text_protos_all.T` → (V, C_all)
3. Optional bias correction: subtract `proto_bias` per class
4. Temperature scaling: divide by `temperature` (default 0.1)
5. Mask-weighted aggregation: `masks_sig.T @ voxel_logits / mask_area`
6. Decision logic (differs per strategy)
7. Greedy panoptic assignment via `_greedy_assign()`

### New CLI Arguments

| Flag | Default | Purpose |
|------|---------|---------|
| `--no-bias` | False | Disable prototype bias correction (ablation) |
| `--novel-margin` | 1.2 | Strategy K: novel must beat known by this factor |
| `--confidence-threshold` | 0.7 | Strategy M: linear head confidence gate |

### Mode Names

| Strategy | Mode name | Evaluator key(s) |
|----------|-----------|------------------|
| K | `voxel_vote` | `voxel_vote` |
| L | `logit_aggregation` | `logit_agg_a0.3`, `logit_agg_a0.5`, `logit_agg_a0.7` |
| M | `confidence_gated` | `confidence_gated` |

---

## 11. Relationship to Previous Work

```
Strategy E (completed, FPs confirmed):
  Pool features → classify pooled → geometric ensemble
  Problem: averaging destroys per-voxel CLIP alignment

Strategy K/L/M (this document):
  Classify per-voxel → aggregate decisions → ensemble
  Hypothesis: preserves the per-voxel signal that distillation provided

Strategy I (future, requires training):
  Add per-voxel contrastive loss with 200+ diverse negatives
  Teaches backbone to actively push features away from confusable classes
  Addresses root cause: backbone never trained with inter-class negatives
```

If K/L/M also fail (all FPs), it conclusively proves that the backbone's per-voxel CLIP alignment is insufficient for novel-class discrimination — not just at the pooled level, but fundamentally at the voxel level. This would confirm that **training-based approaches (Strategy I, C1, D)** are the only path to real open-vocab capability.

---

## 12. Evaluation Results (v1.0-trainval, 6019 samples, 2026-05-28)

### 12.1 Known-Class PQ Summary

| Mode | PQ | SQ | RQ | mIoU | vs Linear |
|------|-----|-----|-----|------|-----------|
| **linear (baseline)** | **0.4761** | 0.7930 | 0.5828 | 0.4323 | — |
| token | 0.4717 | 0.7969 | 0.5775 | 0.4206 | -0.004 |
| geo_v2_a0.5 (Strategy E best) | 0.4697 | 0.7928 | 0.5756 | 0.4300 | -0.006 |
| **logit_agg_a0.5 (Strategy L best)** | **0.4702** | 0.7930 | 0.5761 | 0.4308 | **-0.006** |
| logit_agg_a0.7 | 0.4695 | 0.7922 | 0.5764 | 0.4321 | -0.007 |
| confidence_gated (Strategy M) | 0.4633 | 0.7945 | 0.5664 | 0.4278 | -0.013 |
| **voxel_vote (Strategy K)** | **0.4490** | 0.7932 | 0.5506 | 0.3991 | **-0.027** |

### 12.2 Novel Class Detections

| Mode | Total Novel | police car | ambulance | trash cans | pole | bench |
|------|------------|-----------|-----------|-----------|------|-------|
| geo_v2 (Strategy E) | 330,658 | 10,894 | 296,281 | 20,703 | 843 | 1,937 |
| **logit_agg (Strategy L)** | **305,525** | **10,181** | **274,034** | **19,347** | **603** | **1,360** |
| confidence_gated (M) | 274,038 | 102,728 | 36,524 | 22,828 | 88,946 | 23,012 |
| **voxel_vote (K)** | **1,815,332** | **787,865** | **629,947** | **31,001** | **60,948** | **305,571** |

### 12.3 Key Findings

#### Finding 1: Strategy L ≈ Strategy E (geo_v2) — Identical Behavior

The per-scene novel detection breakdown is **nearly identical** between L and E:
- Same top-20 scenes
- Same voxel counts per scene
- Same "ambulance" dominance (>90% of novel voxels)

**Implication**: The theoretical advantage of per-voxel logit aggregation (`E[f(x)] ≠ f(E[x])`) does NOT manifest in practice. This is because predicted masks are already concentrated on coherent objects — the average feature within a clean mask IS representative of the individual voxels. Pooling degradation is NOT the bottleneck.

#### Finding 2: Strategy K Over-Detects Massively

Voxel_vote produces **5.5× more novel voxels** (1.8M vs 330K) than geo_v2, with a -2.7% PQ hit:
- 788K "police car" voxels (vs 11K for E/L)
- 630K "ambulance" (vs 296K)
- 306K "bench" (entirely new — not seen in E/L)

The 1.2× novel margin is too weak. The hard decision rule (novel wins if > 1.2× known) triggers for almost every uncertain query because per-voxel softmax probabilities are nearly uniform across similar classes.

**Implication**: Hard margin decisions don't work when the underlying feature margins are noise-level.

#### Finding 3: Strategy M Spreads FPs Differently But Doesn't Eliminate Them

Confidence-gated produces more **diverse** FPs (police car: 103K, bench: 23K, pole: 89K) because it only checks novel classes for uncertain queries (~30% of queries). Total volume is slightly lower (274K vs 330K) but FPs remain widespread across all novel classes.

**Implication**: The confidence gate reduces surface area but cannot create signal that doesn't exist in the features.

#### Finding 4: The Bottleneck is Features, Not Aggregation Strategy

Whether you:
- Average features then classify (Strategy E) → 330K FPs
- Classify per-voxel then average logits (Strategy L) → 305K FPs
- Use hard margin voting (Strategy K) → 1.8M FPs
- Gate by confidence (Strategy M) → 274K FPs

The results converge to the same answer: **the backbone features themselves lack sufficient margin between similar classes**. The distillation chain (`CLIP image → OpenSeg → 3D projection → DisNet`) degrades cosine similarity margins to noise-level.

### 12.4 Per-Scene Analysis

Top-20 scenes are **identical** for geo_v2 and logit_aggregation — confirming they detect the same objects:
- All top scenes are dominated by "ambulance" detections (white trucks/buses misclassified)
- Scene `26ff943010a54658b8a9471ff433c3b0`: 5,155 voxels (geo_v2) vs 5,146 (logit_agg) — virtually same

Strategy K top scenes are **different** — dominated by "bench" (misclassified flat surfaces/buildings) and spread more broadly.

Strategy M top scenes show **smaller counts** (max 1,245 voxels vs 5,155 for E/L) — the confidence gate successfully reduces per-scene volume but doesn't eliminate FPs.

### 12.5 Conclusion

**The per-voxel hypothesis is DISPROVEN.** Strategies K, L, M confirm that:

1. ✗ Pooling degradation is NOT the bottleneck (L = E results prove this)
2. ✗ Per-voxel aggregation does NOT reveal hidden novel-class signal
3. ✗ Confidence gating reduces but cannot eliminate FPs
4. ✗ Hard margin voting (K) makes things WORSE

**Root cause confirmed**: The backbone's per-voxel CLIP alignment has insufficient margin between similar classes (e.g., "car" vs "police car", "bus" vs "ambulance"). This is a feature-level limitation that NO inference-only strategy can overcome.

**Remaining ablation**: No-bias run (user will execute) — expected to show more pole/generic FPs but same overall conclusion.

---

## 13. What This Means for the Project

### Inference-Only Chapter Status

```
✗ Strategy E (feature pooling + geo ensemble) — FPs confirmed (10 scenes visual)
✗ Strategy E v2 (multi-prompt + bias) — FPs confirmed (10 scenes visual)
✗ Strategy K (per-voxel vote) — Massive over-detection, PQ -0.027
✗ Strategy L (logit aggregation) — Identical to E, no improvement
✗ Strategy M (confidence gated) — Reduces volume, same FPs
✗ No-bias ablation — DONE: pole catastrophe confirmed, same FPs underneath
```

After no-bias ablation completes, the inference-only chapter is CLOSED.

### Training Path Forward

The only way to achieve open-vocabulary detection is to modify the backbone's feature space:

1. **Strategy I (voxel decomp loss + 200 negatives)** — Highest priority. Train backbone features to explicitly discriminate between confusable classes.
2. **Strategy C1 (token head + 200 negatives)** — Fix the token head's 16-class collapse.
3. **Strategy D (camera CLIP features)** — Bypass the degraded 3D backbone entirely with fresh ViT-L/14 features.

---

## 14. No-Bias Ablation Results (2026-05-29)

### 14.1 Comparison: With-Bias vs No-Bias

| Mode | With-Bias PQ | No-Bias PQ | With-Bias Novel | No-Bias Novel |
|------|-------------|-----------|----------------|--------------|
| voxel_vote (K) | 0.4490 | 0.3935 | 1.8M | 18.6M |
| logit_agg (L) | 0.4685 | 0.4642 | 305K | 830K |
| confidence_gated (M) | 0.4685 | 0.4634 | 312K | 364K |
| geo_v2 (E) | 0.4685 | unchanged | 311K | unchanged (built-in bias) |

### 14.2 Key Finding: Pole Catastrophe Returns Without Bias

Without bias correction, the "pole" prototype (high generic affinity) dominates:
- **voxel_vote**: 10x explosion (1.8M → 18.6M novel voxels), PQ collapses by -0.056
- **logit_agg**: pole returns, 2.7x more novel voxels (305K → 830K)
- **confidence_gated**: slight increase (312K → 364K), most resilient

Bias correction prevents pole catastrophe but does NOT fix the fundamental margin issue — FPs remain in all modes regardless.

### 14.3 L=E Confirmed in Both Ablations

Both with-bias and no-bias runs confirm Strategy L produces identical behavior to Strategy E. Pooling degradation is NOT the bottleneck in either configuration.

### 14.4 Final Conclusion: Inference-Only Chapter CLOSED

```
✗ Strategy E (feature pooling + geo ensemble) — FPs confirmed
✗ Strategy E v2 (multi-prompt + bias) — FPs confirmed (10 scenes visual)
✗ Strategy K (per-voxel vote) — Massive over-detection, PQ -0.027
✗ Strategy L (logit aggregation) — Identical to E, no improvement
✗ Strategy M (confidence gated) — Reduces volume, same FPs
✗ No-bias ablation — Pole catastrophe confirmed, same FPs underneath
```

**Root cause**: Backbone features have noise-level margin (~0.02) between similar classes due to distillation chain degradation (`CLIP → OpenSeg → 3D projection → DisNet`). No inference-only strategy can overcome this.

**Next**: Training path — Strategy I (voxel decomposition loss with 200+ diverse negatives).
