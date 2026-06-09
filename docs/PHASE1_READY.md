# PHASE 1: DIAGNOSTIC - Ready to Execute

## Summary

**Problem:** Novel classes show PQ=0 in Job 1b despite distillation. Is this because:
1. Novel embeddings don't exist? (distillation failed)
2. Novel embeddings exist but are suppressed by objectness? (architectural bug)

**Solution:** Diagnostic script tests both Job 1a and Job 1b with/without objectness to isolate the cause.

---

## What's Ready

### Diagnostic Script
- **File:** `panoptic/scripts/eval_objectness_hypothesis.py` ✓ Created
- **Status:** Ready to run
- **Tests:** 
  - Job 1a best checkpoint (with/without objectness)
  - Job 1b best checkpoint (with/without objectness)
- **Metrics:** PQ_all, PQ_base, PQ_novel, per-class breakdown

### BSUB Submission Script
- **File:** `panoptic/scripts/diagnostic_objectness.sh` ✓ Created
- **Status:** Ready to submit
- **Runtime:** ~1-2 hours (mini split) or 4-6 hours (val split)

### Documentation
- **File:** `docs/phase1_diagnostic_guide.md` ✓ Created
- **Contains:** Quick start, expected results, decision tree, next steps

---

## How to Run

### Option 1: Submit to Cluster (Recommended)
```bash
cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
bsub < panoptic/scripts/diagnostic_objectness.sh
```

Then monitor:
```bash
bjobs -l <job_id>
# or
bpeek <job_id>
# or tail the log
tail -f panoptic/logs/eval_objectness_hypothesis_*.out
```

### Option 2: Run Locally (if in conda environment)
```bash
cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
source activate openscene

# Quick test (mini split, ~1 hour)
python panoptic/scripts/eval_objectness_hypothesis.py \
    --ckpt_job1a panoptic/checkpoints_job1a/best.pth \
    --cfg_job1a panoptic/configs/job1a.yaml \
    --ckpt_job1b panoptic/checkpoints_job1b/best.pth \
    --cfg_job1b panoptic/configs/job1b.yaml \
    --device cuda \
    --split mini

# Full validation (val split, ~4-6 hours)
python panoptic/scripts/eval_objectness_hypothesis.py \
    --ckpt_job1a panoptic/checkpoints_job1a/best.pth \
    --cfg_job1a panoptic/configs/job1a.yaml \
    --ckpt_job1b panoptic/checkpoints_job1b/best.pth \
    --cfg_job1b panoptic/configs/job1b.yaml \
    --device cuda \
    --split val
```

---

## Expected Output

### Scenario 1: ✓ CONFIRMED (Proceed with Option 4 fix)
```
Job 1a Results:
  WITH objectness:    PQ_novel = 0.0000
  WITHOUT objectness: PQ_novel = 0.0000
  Delta: +0.0000
  ✓ Job 1a has no novel embeddings (expected)

Job 1b Results:
  WITH objectness:    PQ_novel = 0.0000
  WITHOUT objectness: PQ_novel = 0.0250+
  Delta: +0.0250+
  ✓ Job 1b has novel embeddings BUT objectness suppresses them!

DIAGNOSIS & RECOMMENDATION:
✓ CONFIRMED: Novel embeddings exist but are suppressed by objectness
  SOLUTION: Option 4 Fix (retrain Job 1a v2, then Job 1b v2)
```

→ **Decision: PROCEED TO PHASE 2 (Implement Option 4 fix)**

### Scenario 2: ✗ NOT CONFIRMED (Investigate alternatives)
```
Job 1b Results:
  WITH objectness:    PQ_novel = 0.0000
  WITHOUT objectness: PQ_novel = 0.0020
  Delta: +0.0020
  ✗ Job 1b distillation did not create detectable novel embeddings

DIAGNOSIS & RECOMMENDATION:
✗ NOT CONFIRMED: Job 1b distillation did NOT create novel embeddings
  ALTERNATIVE CAUSES:
  1. L_O/L_V weights too low
  2. CLIP features not well-aligned
  3. Camera CLIP model version mismatch
  RECOMMENDATION: Do NOT proceed with Option 4
```

→ **Decision: DO NOT PROCEED; investigate distillation quality instead**

### Scenario 3: ? PARTIAL (Investigate further)
```
Job 1b Results:
  Delta: +0.0075 (between 0.005 and 0.01)
  ? PARTIAL: Small improvement; objectness suppression is minor factor
```

→ **Decision: INVESTIGATE before proceeding (check per-class breakdown)**

---

## Timeline from Here

**Assuming CONFIRMED (most likely):**
1. **Now:** Run diagnostic (2-4 hours)
2. **Day 1:** Review results, decide to proceed
3. **Day 1-2:** Implement Option 4 fix (4 hours work)
4. **Day 2:** Submit Job 1a v2 smoke test (4 hours runtime)
5. **Day 3-10:** Job 1a v2 full training (7 days)
6. **Day 10-17:** Job 1b v2 full training (7 days)
7. **Day 17:** Final evaluation and reporting

**Total: ~17 days from now to final results**

---

## What Happens After Diagnostic

### If CONFIRMED ✓
Proceed to PHASE 2 with full Option 4 implementation:
- File: `docs/action_plan_objectness_fix.md` (14 pages, already written)
- Checklist: `docs/implementation_checklist_objectness_fix.md` (all tasks listed)
- Configs: job1a_v2, job1b_v2
- Scripts: job1a_v2_smoke.sh, job1a_v2_full.sh, job1b_v2_full.sh

### If NOT CONFIRMED ✗
Investigate distillation quality:
1. Check L_O/L_V loss values in Job 1b logs
2. Run voxel_margin diagnostic on Job 1b checkpoint
3. Verify CLIP feature quality and coverage
4. Consider alternative fixes (increase weights, etc.)

---

## Ready to Proceed?

The diagnostic script is **ready to run immediately**. 

**To start:**
```bash
bsub < panoptic/scripts/diagnostic_objectness.sh
```

**Expected completion:** 2-4 hours (mini split)  
**Then:** Review output and proceed to Phase 2 or investigate alternatives

Would you like me to:
1. ✓ Submit the diagnostic now?
2. Review the diagnostic script code first?
3. Modify any parameters (e.g., use val split instead of mini)?
