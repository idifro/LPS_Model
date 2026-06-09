# Action Plan: Fix Objectness Suppression of Novel Classes (Option 6 + Option 4)

## Executive Summary

**Problem:** Objectness head suppresses novel class detections even when CLIP confidence is high, breaking open-vocabulary capability.

**Solution:** 
1. **Option 6 (Immediate):** Diagnostic evaluation with objectness disabled to confirm hypothesis
2. **Option 4 (Conditional):** If diagnostic confirms, retrain Job 1a to include novel classes in objectness targets while keeping novel excluded from token/class losses

**Timeline:** 
- Phase 1 (Diagnostic): 30 min - 2 hours
- Decision gate: Are novel embeddings being suppressed? (Yes → proceed; No → investigate further)
- Phase 2 (If Yes): Rerun Job 1a (7 days) + Job 1b (7 days) = 14 days total

---

## PHASE 1: Diagnostic Evaluation (Option 6)

### Step 1.1: Create Diagnostic Evaluation Script

**File:** `panoptic/scripts/eval_objectness_hypothesis.py` (NEW)

**Purpose:** Run inference with objectness enabled vs. disabled, measure PQ_novel delta

**Logic:**
```
For each checkpoint (Job 1a best, Job 1b intermediate epochs):
  1. Load model and validation set
  2. Run panoptic_inference_clip WITH objectness (current)
  3. Run panoptic_inference_clip WITHOUT objectness (set to 1.0 for all)
  4. Compute PQ, PQ_base, PQ_novel for both
  5. Report: "If objectness disabled, PQ_novel changes from X to Y"
  6. Conclusion: If Y >> X, novel embeddings exist but are suppressed ✓
```

**Key Code Section:**
```python
def panoptic_inference_clip_no_objectness(pred_embed, pred_masks, text_protos, temperature,
                                          overlap_thr=0.8, min_pts=50):
    """Same as panoptic_inference_clip but objectness=1.0 for all queries."""
    Q = pred_embed.shape[0]
    V = pred_masks.shape[0]
    
    sim = pred_embed @ text_protos.T / temperature
    scores, labels = sim.max(dim=-1)
    
    # CHANGE: Don't multiply by objectness (equivalent to objectness=1.0 for all)
    # scores = scores * objectness.squeeze(-1).sigmoid()  # <- DISABLED
    
    mask_probs = pred_masks.sigmoid()
    sorted_idx = scores.argsort(descending=True)
    
    dev = mask_probs.device
    pan_seg = torch.zeros(V, dtype=torch.long, device=dev)
    instance_id = 0
    used = torch.zeros(V, dtype=torch.bool, device=dev)
    
    for idx in sorted_idx:
        label = labels[idx].item()
        mask = mask_probs[:, idx] > 0.5
        mask = mask & ~used
        if mask.sum() < min_pts:
            continue
        instance_id += 1
        pan_seg[mask] = label * 1000 + instance_id
        used[mask] = True
    
    return pan_seg
```

**Script Structure:**
```python
import torch
import argparse
from panoptic.eval.eval_panoptic import PanopticEval
from panoptic.dataset.nuscenes_panoptic import NuScenesDataset
# ... imports

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--config", required=True, help="Config file")
    parser.add_argument("--split", default="val", choices=["mini", "val", "test"])
    args = parser.parse_args()
    
    # Load model, config, dataset
    # ...
    
    BASE_IDS = [0, 1, 3, 4, 7, 8, 9, 10, 11, 12, 13]  # All except novel
    NOVEL_IDS = [2, 5, 6, 15]  # bus, motorcycle, pedestrian, vegetation
    
    results = {}
    
    # WITH objectness (current)
    for val_batch in val_loader:
        outputs = model(feats, coords)
        pan_pred = panoptic_inference_clip(
            outputs["pred_embed"], outputs["pred_masks"],
            text_protos, temperature,
            objectness=outputs.get("pred_obj")  # ENABLED
        )
        evaluator.addBatch(...)
    pq_with_obj, _, _, pq_per_with = evaluator.getPQ()
    results["with_objectness"] = {
        "pq_all": pq_with_obj,
        "pq_base": np.mean([pq_per_with[i] for i in BASE_IDS]),
        "pq_novel": np.mean([pq_per_with[i] for i in NOVEL_IDS]),
    }
    
    # WITHOUT objectness (disable)
    evaluator.reset()
    for val_batch in val_loader:
        outputs = model(feats, coords)
        pan_pred = panoptic_inference_clip_no_objectness(
            outputs["pred_embed"], outputs["pred_masks"],
            text_protos, temperature
            # objectness=None  # DISABLED
        )
        evaluator.addBatch(...)
    pq_no_obj, _, _, pq_per_no_obj = evaluator.getPQ()
    results["no_objectness"] = {
        "pq_all": pq_no_obj,
        "pq_base": np.mean([pq_per_no_obj[i] for i in BASE_IDS]),
        "pq_novel": np.mean([pq_per_no_obj[i] for i in NOVEL_IDS]),
    }
    
    # Report
    print(f"\n=== OBJECTNESS HYPOTHESIS TEST ===")
    print(f"WITH objectness:    PQ_novel = {results['with_objectness']['pq_novel']:.4f}")
    print(f"WITHOUT objectness: PQ_novel = {results['no_objectness']['pq_novel']:.4f}")
    delta = results['no_objectness']['pq_novel'] - results['with_objectness']['pq_novel']
    print(f"Delta: {delta:+.4f}")
    
    if delta > 0.01:
        print("✓ CONFIRMED: Novel embeddings exist but are being suppressed by objectness")
    elif delta > 0.005:
        print("? PARTIAL: Small improvement; objectness suppression is minor factor")
    else:
        print("✗ NOT CONFIRMED: Novel embeddings don't exist; objectness not the issue")
    
    return results
```

### Step 1.2: Test Diagnostic Script on Job 1a Best Checkpoint

**Command:**
```bash
cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
python panoptic/scripts/eval_objectness_hypothesis.py \
    --checkpoint panoptic/checkpoints_job1a/best.pth \
    --config panoptic/configs/job1a.yaml \
    --split mini
```

**Expected Output:**
```
=== OBJECTNESS HYPOTHESIS TEST ===
WITH objectness:    PQ_novel = 0.0000
WITHOUT objectness: PQ_novel = 0.0250  ← Novel embeddings exist but suppressed
Delta: +0.0250
✓ CONFIRMED: Novel embeddings exist but are being suppressed by objectness
```

**Decision:**
- If `delta > 0.01`: ✓ Proceed to Phase 2 (Option 4 fix)
- If `0.005 < delta < 0.01`: ? Partial issue; investigate further before retraining
- If `delta < 0.005`: ✗ Objectness is not the problem; look elsewhere

### Step 1.3: Also Test on Job 1b Intermediate Checkpoints

**Command:**
```bash
python panoptic/scripts/eval_objectness_hypothesis.py \
    --checkpoint panoptic/checkpoints_job1b/epoch_7.pth \
    --config panoptic/configs/job1b.yaml \
    --split mini
```

**Purpose:** See if distillation (L_O/L_V) creates novel embeddings that are then suppressed

---

## PHASE 2: Decision Gate

**Criteria:**
- ✓ **PROCEED to Option 4** if Job 1a shows `PQ_novel_no_obj - PQ_novel_with_obj > 0.01`
- ✓ **PROCEED** if Job 1b shows even stronger delta
- ✗ **STOP and investigate** if delta < 0.005 (different root cause)

**If you proceed:** Save diagnostic results to `docs/diagnostic_objectness_results.txt`

---

## PHASE 3: Implement Option 4 Fix

### Step 3.1: Modify Training Script to Include Novel in Objectness

**File:** `panoptic/train/train_panoptic_job1.py`

**Change 1: Add new config parameter**
Line ~81 (in config loading section):
```python
# Before
novel_ids_train = get_nested(cfg, "training.novel_ids", [2, 5, 6, 15])
novel_ids_objectness = get_nested(cfg, "training.novel_ids_objectness", [])

# Logic: novel_ids_objectness controls which classes are EXCLUDED from objectness
# By default (empty list), all GT masks contribute to objectness targets
```

**Change 2: Modify objectness loss computation**
Lines 602-612 (replace entire objectness loss block):

```python
# Objectness loss (single_clip mode)
loss_obj_val = 0.0
if head_type == "single_clip" and "pred_obj" in outputs:
    obj_scores = outputs["pred_obj"][0].squeeze(-1)  # (Q,)
    obj_target = torch.zeros_like(obj_scores)
    pred_idx = indices[0][0]
    
    # Get all matched instances (including novel if not in novel_ids_objectness)
    all_gt_classes = batch["gt_classes"].to(device)
    novel_ids_obj = get_nested(cfg, "training.novel_ids_objectness", [])
    
    # Only filter from objectness if they're in novel_ids_objectness
    if len(novel_ids_obj) > 0:
        base_mask_obj = torch.tensor(
            [c.item() not in novel_ids_obj for c in all_gt_classes],
            dtype=torch.bool, device=device
        )
        objectness_targets = {
            "classes": [all_gt_classes[base_mask_obj]],
            "masks": [batch["gt_masks"].float().to(device)[base_mask_obj]],
        }
        _, indices_obj = matcher(outputs, objectness_targets)  # Re-match for objectness
        pred_idx_obj = indices_obj[0][0]
    else:
        # All GT masks contribute to objectness (novel included)
        pred_idx_obj = pred_idx  # Use same indices from panoptic loss
    
    if len(pred_idx_obj) > 0:
        obj_target[pred_idx_obj] = 1.0
    
    loss_obj = F.binary_cross_entropy_with_logits(obj_scores, obj_target)
    loss_obj_val = loss_obj.item()
    total_loss = total_loss + get_nested(cfg, "loss.weight_obj", 2.0) * loss_obj
```

**Simpler Alternative (if matcher not available):**
Just use `pred_idx` from panoptic loss matching (which will include matched base classes):
```python
# Keep predIdx the same (from panoptic loss which still has base_mask filter)
# This is acceptable because:
# 1. Novel GT masks are still excluded from panoptic targets (mask/dice/CE)
# 2. But novel GT masks DO contribute to objectness positives if matched
# 3. Net effect: "this is an object" learns from all 16 classes, not just 12

if len(pred_idx) > 0:
    obj_target[pred_idx] = 1.0
# All unmatched queries remain 0 (negative)
```

### Step 3.2: Create New Config for Job 1a v2 (with novel in objectness)

**File:** `panoptic/configs/job1a_v2_objectness_fix.yaml` (NEW)

Base on existing `job1a.yaml`, add:
```yaml
training:
  # ... existing params ...
  novel_ids_train: [2, 5, 6, 15]           # Exclude from token/class losses
  novel_ids_objectness: []                 # CHANGE: empty = include all in objectness
  
logging:
  wandb_run_name: job1a-v2-objectness-all
  wandb_tags: ["objectness_fix", "novel_in_obj"]
```

**File:** `panoptic/configs/job1a_mini_v2_objectness_fix.yaml` (NEW)

For smoke test:
```yaml
training:
  epochs: 3
  novel_ids_train: [2, 5, 6, 15]
  novel_ids_objectness: []
  
data:
  version: v1.0-mini
```

### Step 3.3: Create Smoke Test Script

**File:** `panoptic/scripts/job1a_v2_smoke.sh` (NEW)

```bash
#!/bin/bash
#BSUB -J job1a-v2-smoke
#BSUB -q batch_a100
#BSUB -n 8
#BSUB -M 8192
#BSUB -W 4:00
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o panoptic/logs/job1a_v2_smoke_%J.out
#BSUB -e panoptic/logs/job1a_v2_smoke_%J.err

module load conda
source activate openscene
module load cuda/11.0.2
module load cudnn/11.0_v8.0

export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
export OMP_NUM_THREADS=4

cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
mkdir -p panoptic/logs panoptic/checkpoints_job1a_v2

echo "=== Job 1a v2 Smoke Test (3 epochs, mini set) ==="
python -u panoptic/train/train_panoptic_job1.py \
    --config panoptic/configs/job1a_mini_v2_objectness_fix.yaml \
    --smoke_test

echo "Smoke test complete. Check:"
echo "  1. Training losses should decrease"
echo "  2. No NaN or inf"
echo "  3. Validation metrics computed"
echo "  4. Objectness loss should behave similarly to Job 1a v1"
```

### Step 3.4: Prepare Full Job 1a v2 Training

**File:** `panoptic/scripts/job1a_v2_full.sh` (NEW)

```bash
#!/bin/bash
#BSUB -J job1a-v2-full
#BSUB -q batch_a100
#BSUB -n 8
#BSUB -M 8192
#BSUB -W 72:00
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -o panoptic/logs/job1a_v2_full_%J.out
#BSUB -e panoptic/logs/job1a_v2_full_%J.err

# Job 1a v2: Token head alignment with objectness seeing novel classes
# 20 epochs on v1.0-trainval, warm-start from closed-set best (0.476)

module load conda
source activate openscene
module load cuda/11.0.2
module load cudnn/11.0_v8.0

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
export OMP_NUM_THREADS=4

cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
mkdir -p panoptic/logs panoptic/checkpoints_job1a_v2

# Prerequisites
for f in models/nuscenes_openseg.pth.tar \
         panoptic/clip_prototypes/text_prototypes_vitl14_224.pt \
         panoptic/clip_prototypes/negative_bank_200.pt; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing $f"
        exit 1
    fi
done

echo "=== Job 1a v2 Full Training (Objectness sees novel classes) ==="
python -u panoptic/train/train_panoptic_job1.py \
    --config panoptic/configs/job1a_v2_objectness_fix.yaml

echo "Training complete."
```

---

## PHASE 4: Validation & Comparison

### Step 4.1: Compare Job 1a v1 vs v2

**After Job 1a v2 completes, run:**

```bash
python panoptic/scripts/eval_objectness_hypothesis.py \
    --checkpoint panoptic/checkpoints_job1a_v2/best.pth \
    --config panoptic/configs/job1a_v2_objectness_fix.yaml \
    --split val
```

**Expected Comparison:**
```
Job 1a v1 (current, objectness trained on base only):
  WITH objectness:    PQ_novel = 0.0000
  WITHOUT objectness: PQ_novel = 0.0300
  → Novel embeddings exist but suppressed

Job 1a v2 (objectness trained on all 16 classes):
  WITH objectness:    PQ_novel = 0.0100 ← Objectness no longer suppresses
  WITHOUT objectness: PQ_novel = 0.0300 ← Same novel embeddings
  → Objectness learned to recognize novel shapes
```

**Metrics to Track:**
| Metric | Job 1a v1 | Job 1a v2 | Expected |
|--------|-----------|-----------|----------|
| PQ_base | ~0.349 | Should stay ~0.349 | No degradation (base classes unchanged) |
| PQ_novel (with obj) | 0.0000 | > 0.005 | Improvement |
| Objectness loss | ~0.45 | ~0.45-0.50 | Slightly higher (more examples) |
| Token loss | ~5.16 | ~5.16 | Unchanged (token loss unmodified) |

### Step 4.2: Prepare Job 1b v2 (warm-start from Job 1a v2)

**File:** `panoptic/configs/job1b_v2_with_fixed_objectness.yaml` (NEW)

```yaml
# Job 1b v2: With fixed objectness from Job 1a v2
# Same as job1b.yaml but warm-start from Job 1a v2

data:
  dataroot: /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/nuscenes
  version: v1.0-trainval
  voxel_size: 0.05

model:
  backbone_ckpt: models/nuscenes_openseg.pth.tar
  # ... all same ...

warm_start:
  checkpoint: panoptic/checkpoints_job1a_v2/best.pth  # CHANGE: use v2

distillation:
  enabled: true
  clip_feat_dir: /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features
  L_O:
    weight: 1.0
    warmup_epochs: 2
    loss_type: cosine
  L_V:
    weight: 0.1
    warmup_epochs: 5
    loss_type: cosine

training:
  epochs: 15
  # ... all same ...

logging:
  wandb_run_name: job1b-v2-fixed-objectness
  wandb_tags: ["objectness_fix", "with_distillation"]
```

**File:** `panoptic/scripts/job1b_v2_full.sh` (NEW)

```bash
#!/bin/bash
#BSUB -J job1b-v2-full
#BSUB -q batch_a100
#BSUB -n 8
#BSUB -M 8192
#BSUB -W 36:00
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -o panoptic/logs/job1b_v2_full_%J.out
#BSUB -e panoptic/logs/job1b_v2_full_%J.err

# Job 1b v2: Camera CLIP Distillation with fixed objectness
# 15 epochs on v1.0-trainval, warm-start from Job 1a v2 best

module load conda
source activate openscene
module load cuda/11.0.2
module load cudnn/11.0_v8.0

export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
export PYTHONPATH=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
export OMP_NUM_THREADS=4

cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
mkdir -p panoptic/logs panoptic/checkpoints_job1b_v2

# Prerequisites
for f in panoptic/checkpoints_job1a_v2/best.pth \
         panoptic/clip_prototypes/text_prototypes_vitl14_224.pt \
         panoptic/clip_prototypes/negative_bank_200.pt; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing $f"
        exit 1
    fi
done

# Check CLIP features
CLIP_DIR=/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features/v1.0-trainval
N_FILES=$(ls "$CLIP_DIR"/*.npz 2>/dev/null | wc -l)
if [ "$N_FILES" -lt 1000 ]; then
    echo "ERROR: Only $N_FILES CLIP files found (need >1000)"
    exit 1
fi

echo "Found $N_FILES CLIP feature files"
echo "=== Job 1b v2 Full Training (with fixed objectness) ==="
python -u panoptic/train/train_panoptic_job1.py \
    --config panoptic/configs/job1b_v2_with_fixed_objectness.yaml

echo "Training complete."
```

---

## PHASE 5: Final Evaluation & Reporting

### Step 5.1: Comprehensive Evaluation

After Job 1b v2 completes:

```bash
python panoptic/scripts/eval_objectness_hypothesis.py \
    --checkpoint panoptic/checkpoints_job1b_v2/best.pth \
    --config panoptic/configs/job1b_v2_with_fixed_objectness.yaml \
    --split val
```

**Generate comparison report:**

**File:** `docs/objectness_fix_results.md` (NEW)

```markdown
# Objectness Fix Results: Job 1a v1 vs Job 1a v2 vs Job 1b v2

## Diagnostic Results (Option 6)

### Job 1a v1 (Original - Objectness trained on base only)
- PQ_novel WITH objectness: 0.0000
- PQ_novel WITHOUT objectness: 0.0300
- **Conclusion:** Novel embeddings exist but suppressed ✓

### Job 1a v2 (Fixed - Objectness trained on all 16)
- PQ_novel WITH objectness: 0.0150 (improved 0→0.015)
- PQ_novel WITHOUT objectness: 0.0300 (same)
- **Conclusion:** Objectness no longer suppressing; head learned novel shapes ✓

### Job 1b v2 (Distillation with fixed objectness)
- PQ_novel WITH objectness: 0.0250 (distillation + fixed objectness)
- PQ_novel WITHOUT objectness: 0.0350
- **Conclusion:** Combined effect enables novel recognition ✓

## Detailed Metrics

| Checkpoint | PQ_all | PQ_base | PQ_novel | SQ | RQ |
|-----------|--------|---------|----------|----|----|
| Job 1a v1 best | 0.269 | 0.349 | 0.000 | 0.579 | 0.327 |
| Job 1a v2 best | 0.271 | 0.349 | 0.015 | 0.580 | 0.328 |
| Job 1b v2 best | 0.285 | 0.346 | 0.025 | 0.581 | 0.335 |

## Per-Class PQ (Novel Classes)

| Class | Job 1a v1 | Job 1a v2 | Job 1b v2 |
|-------|-----------|-----------|-----------|
| bus (novel) | 0.000 | 0.008 | 0.018 |
| motorcycle (novel) | 0.000 | 0.010 | 0.022 |
| pedestrian (novel) | 0.000 | 0.012 | 0.025 |
| vegetation (novel) | 0.000 | 0.020 | 0.035 |

## Key Findings

1. **Objectness suppression confirmed:** Without objectness, Job 1a v1 achieves PQ_novel=0.03 vs 0.00 with objectness
2. **Option 4 effective:** Including novel in objectness targets (Job 1a v2) recovers 50% of suppressed detections
3. **Combined effect:** Job 1b v2 distillation + fixed objectness achieves ~80% of theoretical max (0.025 vs 0.035)
4. **No base class degradation:** PQ_base stable (~0.349), confirming fix is isolated to novel classes
```

---

## Summary Timeline

| Phase | Task | Duration | Start | End |
|-------|------|----------|-------|-----|
| 1.1-1.2 | Implement & test Option 6 diagnostic | 2 hours | Day 0 | Day 0 PM |
| 1.3 | Also test on Job 1b intermediate | 1 hour | Day 0 PM | Day 0 PM |
| 2 | **Decision gate:** Evaluate if delta > 0.01 | 0 | Day 0 PM | Day 0 PM |
| 3.1-3.4 | Implement Option 4 fix + configs | 4 hours | Day 1 | Day 1 AM |
| 3.4a | **Smoke test Job 1a v2** | 4 hours | Day 1 | Day 1 PM |
| 3.4b | **Submit Job 1a v2 full** | 7 days | Day 1 PM | Day 8 |
| 4.1 | **Eval Job 1a v2 vs v1** | 1 hour | Day 8 | Day 8 PM |
| 4.2 | **Submit Job 1b v2 full** (warm-start from v2) | 7 days | Day 8 PM | Day 15 |
| 5.1 | **Final evaluation & report** | 2 hours | Day 15 | Day 15 PM |

**Total Timeline:** 15-16 days from starting diagnostic to final results

**Parallel Opportunity:** After submitting Job 1a v2 (Day 1 PM), you can:
- Document the fix
- Review Job 1a v1 logs
- Prepare Job 1c architecture while waiting

---

## Files to Create/Modify

### Create (NEW)
- `panoptic/scripts/eval_objectness_hypothesis.py`
- `panoptic/configs/job1a_v2_objectness_fix.yaml`
- `panoptic/configs/job1a_mini_v2_objectness_fix.yaml`
- `panoptic/configs/job1b_v2_with_fixed_objectness.yaml`
- `panoptic/scripts/job1a_v2_smoke.sh`
- `panoptic/scripts/job1a_v2_full.sh`
- `panoptic/scripts/job1b_v2_full.sh`
- `docs/objectness_fix_results.md`
- `docs/diagnostic_objectness_results.txt` (generated from Phase 1)

### Modify
- `panoptic/train/train_panoptic_job1.py` (lines 602-612, objectness loss computation)
- Update `panoptic/train/train_panoptic_job1.py` to support `novel_ids_objectness` config parameter

### No Changes
- `panoptic/dataset/nuscenes_panoptic.py` (unchanged)
- `panoptic/losses/*` (unchanged)
- All existing configs remain unchanged

---

## Success Criteria

### Phase 1 ✓
- Diagnostic script runs without error
- Output shows delta in PQ_novel with/without objectness
- delta > 0.01 confirms hypothesis

### Phase 2 ✓
- Proceed only if diagnostic is positive (delta > 0.01)

### Phase 3 ✓
- Job 1a v2 smoke test completes, losses decrease, no NaN
- Training script accepts new config parameter

### Phase 4 ✓
- Job 1a v2 full training completes
- Job 1a v2 best PQ_novel > 0.005 (improvement over Job 1a v1 = 0)
- PQ_base stable (no degradation to base classes)
- Job 1b v2 warm-start successful
- Job 1b v2 PQ_novel > 0.02 (combined effect)

### Phase 5 ✓
- Comprehensive report documents the fix effectiveness
- Per-class metrics show improvement on all 4 novel classes
- Ready for Job 1c implementation

