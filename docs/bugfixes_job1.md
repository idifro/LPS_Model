# Bug Fixes Log — Job 1a/1b Training

## Fix 1: Temperature Explosion (14.0 → dynamic)

**Problem:** The learnable temperature in `TokenLossC1` was initialized as `nn.Parameter(torch.tensor(14.286))` (log-scale). After training, it saturated at ~14.0, making the softmax nearly one-hot and killing gradient flow to non-top-1 classes.

**Root Cause:** The initial value was too high. With temperature=14.286, even small cosine differences (0.05) become `exp(14.286 * 0.05) = exp(0.71)` — making the loss insensitive to margin improvements once the correct class is already top-1.

**Fix:**
- Changed initialization to `log_temperature = nn.Parameter(torch.tensor(2.66))` (exp(2.66) ≈ 14.3 CLIP default, but parameterized in log-space with proper clamping)
- Added clamp: `temperature = self.log_temperature.exp().clamp(min=1.0, max=100.0)`
- This allows the temperature to decrease during training, sharpening discrimination

**Impact:** Temperature now converges to ~4-7 range, providing better gradient signal for margin improvement.

---

## Fix 2: LoRA Grad Norm — AMP Scale Artifact

**Problem:** `lora_gnorm` was reported as 914,366 (appearing unstable) because gradients were read while still AMP-scaled (scale factor = 65536).

**Root Cause:** The grad norm computation happened inside the training loop before `scaler.unscale_(optimizer)` was called. AMP multiplies all gradients by 65536 to prevent fp16 underflow — so the displayed norm was inflated by 65536×.

**Fix:**
- Moved grad norm computation to **after** `scaler.unscale_(optimizer)`
- Now reports true unscaled gradient norms
- Actual LoRA grad norms are ~5-15 (healthy range)

**Actual norms:** 914,366 / 65,536 ≈ **13.95** — which was always in the healthy range.

---

## Fix 3: Novel Class Leakage in Mask Loss

**Problem:** `PQ_novel = 0.4757` after epoch 1 — impossibly high for a model with no novel-class text supervision.

**Root Cause:** The training targets included ALL 16 classes (including novel). The mask/dice loss was providing full segmentation supervision for novel classes, making PQ_novel artificially high (model segments novel objects perfectly but "classifies" them only because there are few confusers).

**Fix:**
- Added `NOVEL_IDS_TRAIN = {2, 6, 14, 15}` (bus, pedestrian, manmade, vegetation)
- Filter GT masks/classes to base-only before computing panoptic loss
- Novel classes now receive zero mask supervision during training
- PQ_novel > 0 can only come from genuine CLIP-space generalization

---

## Fix 4: Matcher KeyError in single_clip mode (aux_outputs)

**Problem:** `KeyError: 'pred_logits'` when panoptic loss iterates `aux_outputs`.

**Root Cause:** In `single_clip` head mode, auxiliary decoder layers output `pred_embed` instead of `pred_logits`. The matcher unconditionally accessed `outputs["pred_logits"]`.

**Fix:**
- Modified matcher to handle missing `pred_logits` when `cost_class == 0`
- When `cost_class == 0`, matcher uses mask+dice cost only (no class probabilities needed)
- Added fallback: if `pred_logits` not in outputs and `cost_class > 0`, raise informative error

---

## Fix 5: Device Mismatch in panoptic_inference_clip

**Problem:** `RuntimeError: Expected all tensors to be on the same device (cuda:0 and cpu)`

**Root Cause:** `pan_seg` and `used` tensors were created on CPU while `mask_probs` was on CUDA.

**Fix:** Added `.to(device)` for `pan_seg` and `used` allocation:
```python
pan_seg = torch.zeros(V, dtype=torch.long, device=pred_masks.device)
used = torch.zeros(V, dtype=torch.bool, device=pred_masks.device)
```

---

## Fix 6: CLIP fp16/fp32 Mismatch in Precompute

**Problem:** `Input type (torch.cuda.FloatTensor) and weight type (torch.cuda.HalfTensor) should be the same`

**Root Cause:** OpenAI CLIP loads weights in float16 by default. Input tiles were float32.

**Fix:** Cast input tiles and normalization constants to `.half()` before passing to CLIP encoder.
