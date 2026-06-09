# PHASE 1: DIAGNOSTIC EVALUATION (Revised)

## Quick Start

### Option A: Quick Mini Test (1-2 hours)
```bash
cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
bsub < panoptic/scripts/diagnostic_objectness.sh
```

This runs on v1.0-mini (10 samples) for quick feedback.

### Option B: Full Validation Test (4-6 hours)
```bash
python panoptic/scripts/eval_objectness_hypothesis.py \
    --ckpt_job1a panoptic/checkpoints_job1a/best.pth \
    --cfg_job1a panoptic/configs/job1a.yaml \
    --ckpt_job1b panoptic/checkpoints_job1b/best.pth \
    --cfg_job1b panoptic/configs/job1b.yaml \
    --device cuda \
    --split val
```

This runs on full validation set (more reliable).

---

## What the Diagnostic Tests

### Structure
For BOTH Job 1a and Job 1b checkpoints:
1. Run panoptic inference WITH objectness enabled (current production behavior)
2. Run panoptic inference WITHOUT objectness (set to 1.0 for all queries)
3. Compare PQ_novel: if delta > 0.01, novel embeddings are being suppressed

### Expected Results

| Checkpoint | WITH objectness | WITHOUT objectness | Delta | Interpretation |
|-----------|-----------------|-------------------|-------|-----------------|
| **Job 1a best** | 0.0000 | 0.0000 | ~0 | ✓ No embeddings yet (no distillation) |
| **Job 1b best** | 0.0000 | **0.025+** | **0.025+** | ✓ Novel embeddings exist but suppressed! |

**Key:** If Job 1b delta > 0.01, this CONFIRMS the hypothesis and validates Option 4 fix.

---

## Output Interpretation

The script will produce a report like this:

```
================================================================================
OBJECTNESS SUPPRESSION HYPOTHESIS TEST
================================================================================

================================================================================
JOB 1a: Token Head Alignment (No Distillation)
================================================================================
Expected: Novel embeddings = 0 (no distillation yet)

[Job 1a] WITH objectness...
[Job 1a] WITHOUT objectness...

Job 1a Results:
  WITH objectness:    PQ_all=0.2690  PQ_base=0.3490  PQ_novel=0.0000
  WITHOUT objectness: PQ_all=0.2690  PQ_base=0.3490  PQ_novel=0.0000
  Delta (PQ_novel):   +0.0000
  ✓ Job 1a has no novel embeddings (expected - no distillation yet)

================================================================================
JOB 1b: Camera CLIP Distillation (L_O + L_V)
================================================================================
Expected: Novel embeddings > 0 (distillation should create them)

[Job 1b] WITH objectness...
[Job 1b] WITHOUT objectness...

Job 1b Results:
  WITH objectness:    PQ_all=0.2810  PQ_base=0.3460  PQ_novel=0.0000
  WITHOUT objectness: PQ_all=0.3050  PQ_base=0.3460  PQ_novel=0.0280
  Delta (PQ_novel):   +0.0280
  ✓ Job 1b has novel embeddings BUT objectness suppresses them!

================================================================================
DIAGNOSIS & RECOMMENDATION
================================================================================
✓ CONFIRMED: Novel embeddings exist (from distillation) but are suppressed by objectness

  ROOT CAUSE: Objectness head trained only on base class matches
  → Novel queries never learn P(object)=high
  → At inference: high CLIP similarity × low objectness = suppressed detection

  SOLUTION: Option 4 Fix
  1. Modify Job 1a training: include novel GT masks in objectness targets
  2. Retrain Job 1a v2 (7 days)
  3. Retrain Job 1b v2 warm-starting from Job 1a v2 (7 days)

================================================================================
JOB 1b: Per-Class PQ (Novel Classes)
================================================================================
Class                WITH Obj     WITHOUT Obj  Delta       
----------------------------------------------------
bus                  0.0000       0.0150       +0.0150
motorcycle           0.0000       0.0200       +0.0200
pedestrian           0.0000       0.0350       +0.0350
vegetation           0.0000       0.0400       +0.0400

================================================================================
END OF REPORT
================================================================================
```

---

## Decision Tree

```
Run diagnostic (2-4 hours)
    ↓
IF Job 1b delta > 0.01
    ✓ CONFIRMED: Novel embeddings suppressed
       ↓
       PROCEED TO OPTION 4 FIX
       ├─ Modify train_panoptic_job1.py
       ├─ Create job1a_v2 configs
       ├─ Rerun Job 1a v2 (7 days)
       ├─ Rerun Job 1b v2 (7 days)
       └─ Evaluate improvements

ELSE IF Job 1b delta < 0.005
    ✗ NOT CONFIRMED: No novel embeddings detected
       ↓
       DO NOT PROCEED WITH OPTION 4
       ├─ Problem is not objectness suppression
       ├─ Investigate:
       │  ├─ L_O/L_V weights too low?
       │  ├─ CLIP features quality?
       │  ├─ Voxel projection issues?
       │  └─ Model version mismatch?
       └─ Consider alternative fixes (increase distillation weights, etc.)

ELSE (delta 0.005-0.01)
    ? PARTIAL: Weak suppression effect
       ↓
       INVESTIGATE BEFORE PROCEEDING
       ├─ Check per-class breakdown
       ├─ Manually inspect predictions
       └─ Decision: proceed or investigate further
```

---

## Files Created

1. **`panoptic/scripts/eval_objectness_hypothesis.py`** — Main diagnostic script
2. **`panoptic/scripts/diagnostic_objectness.sh`** — BSUB submission script

---

## Next Steps (After Diagnostic)

**If CONFIRMED (delta > 0.01):**
→ Proceed to PHASE 2: Implement Option 4 Fix
  - Modify `train_panoptic_job1.py` lines 602-612
  - Create v2 configs
  - Submit Job 1a v2 smoke test
  - Submit Job 1a v2 full
  - Submit Job 1b v2 full

**If NOT CONFIRMED (delta < 0.005):**
→ Investigate alternative root causes
  - Check distillation loss values in Job 1b logs
  - Verify CLIP feature quality/coverage
  - Run voxel_margin diagnostic on Job 1b checkpoint
  - Consider increasing L_O/L_V weights

---

## Monitoring

To check job status:
```bash
bjobs -l <job_id>
```

To watch log output:
```bash
bpeek <job_id>
```

Or tail the output file:
```bash
tail -50 panoptic/logs/eval_objectness_hypothesis_*.out
```
