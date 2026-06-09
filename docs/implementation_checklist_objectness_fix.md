# Implementation Checklist: Objectness Fix (Option 6 + Option 4)

## PHASE 1: Diagnostic Evaluation (30 min - 2 hours)

### Task 1.1: Create eval_objectness_hypothesis.py script
- [ ] Create file: `panoptic/scripts/eval_objectness_hypothesis.py`
- [ ] Implement `panoptic_inference_clip_no_objectness()` function (objectness disabled)
- [ ] Load checkpoint, config, dataset
- [ ] Run panoptic_inference_clip WITH objectness
- [ ] Run panoptic_inference_clip WITHOUT objectness
- [ ] Compute PQ, PQ_base, PQ_novel for both
- [ ] Print comparison table
- [ ] Decision: If delta > 0.01, mark "✓ PROCEED TO OPTION 4"

### Task 1.2: Test on Job 1a best checkpoint
```bash
python panoptic/scripts/eval_objectness_hypothesis.py \
    --checkpoint panoptic/checkpoints_job1a/best.pth \
    --config panoptic/configs/job1a.yaml \
    --split mini
```
- [ ] Script runs without error
- [ ] Output shows: "PQ_novel WITH objectness = X, WITHOUT objectness = Y"
- [ ] Calculate delta = Y - X
- [ ] Record result in `docs/diagnostic_objectness_results.txt`

### Task 1.3: Test on Job 1b current checkpoint (if available)
```bash
python panoptic/scripts/eval_objectness_hypothesis.py \
    --checkpoint panoptic/checkpoints_job1b/latest.pth \
    --config panoptic/configs/job1b.yaml \
    --split mini
```
- [ ] Optional if Job 1b still training
- [ ] Record delta values

### Task 1.4: Decision Gate
- [ ] If delta > 0.01: **✓ APPROVE OPTION 4 FIX**
- [ ] If delta <= 0.005: **✗ INVESTIGATE FURTHER** (novel embeddings don't exist)
- [ ] Document decision in `docs/diagnostic_objectness_results.txt`

**→ Only proceed to Phase 2 if decision is ✓ APPROVE**

---

## PHASE 2: Implement Option 4 Fix (4 hours)

### Task 2.1: Modify train_panoptic_job1.py - Add config parameter
**File:** `panoptic/train/train_panoptic_job1.py`

- [ ] Find line ~81 (config loading section)
- [ ] Add: `novel_ids_objectness = get_nested(cfg, "training.novel_ids_objectness", [])`
- [ ] Verify parameter loaded correctly (can print during init)

### Task 2.2: Modify objectness loss computation
**File:** `panoptic/train/train_panoptic_job1.py` Lines 602-612

Replace entire objectness block with:
```python
# Objectness loss (single_clip mode)
loss_obj_val = 0.0
if head_type == "single_clip" and "pred_obj" in outputs:
    obj_scores = outputs["pred_obj"][0].squeeze(-1)  # (Q,)
    obj_target = torch.zeros_like(obj_scores)
    pred_idx = indices[0][0]  # Use panoptic matching indices
    
    if len(pred_idx) > 0:
        obj_target[pred_idx] = 1.0  # All matched queries (including novel)
    
    loss_obj = F.binary_cross_entropy_with_logits(obj_scores, obj_target)
    loss_obj_val = loss_obj.item()
    total_loss = total_loss + get_nested(cfg, "loss.weight_obj", 2.0) * loss_obj
```

- [ ] Code modified
- [ ] Syntax check: `python -m py_compile panoptic/train/train_panoptic_job1.py`

### Task 2.3: Create job1a_v2_objectness_fix.yaml
**File:** `panoptic/configs/job1a_v2_objectness_fix.yaml`

Copy from `job1a.yaml` and add:
```yaml
training:
  novel_ids_objectness: []  # CHANGE: empty = all classes included in objectness

logging:
  wandb_run_name: job1a-v2-objectness-all
  wandb_tags: ["objectness_fix", "novel_in_obj"]
```

- [ ] File created
- [ ] Verify syntax: `python -c "import yaml; yaml.safe_load(open('panoptic/configs/job1a_v2_objectness_fix.yaml'))"`

### Task 2.4: Create job1a_mini_v2_objectness_fix.yaml
**File:** `panoptic/configs/job1a_mini_v2_objectness_fix.yaml`

- [ ] Copy from existing mini config
- [ ] Set `data.version: v1.0-mini`
- [ ] Set `training.epochs: 3`
- [ ] Set `training.novel_ids_objectness: []`

### Task 2.5: Create job1a_v2_smoke.sh
**File:** `panoptic/scripts/job1a_v2_smoke.sh`

- [ ] File created (use template from action plan)
- [ ] Make executable: `chmod +x panoptic/scripts/job1a_v2_smoke.sh`
- [ ] Test submission format (don't submit yet)

### Task 2.6: Create job1a_v2_full.sh
**File:** `panoptic/scripts/job1a_v2_full.sh`

- [ ] File created
- [ ] Make executable: `chmod +x panoptic/scripts/job1a_v2_full.sh`
- [ ] Verify checkpoint dependencies check

### Task 2.7: Verify all configs valid
- [ ] `python panoptic/train/train_panoptic_job1.py --config panoptic/configs/job1a_v2_objectness_fix.yaml --help` (should not error)
- [ ] `python panoptic/train/train_panoptic_job1.py --config panoptic/configs/job1a_mini_v2_objectness_fix.yaml --help` (should not error)

---

## PHASE 3: Smoke Test Job 1a v2 (4 hours)

### Task 3.1: Submit smoke test job
```bash
cd /fs/scratch/SX_ETL3_GenAI_Team/mha2cob/openscene
bsub < panoptic/scripts/job1a_v2_smoke.sh
```

- [ ] Job submitted successfully
- [ ] Job ID recorded: `_________________`
- [ ] Monitor: `bpeek <job_id>`

### Task 3.2: Monitor smoke test
- [ ] Training starts: epoch 0
- [ ] Loss values printed
- [ ] No NaN or inf
- [ ] Validation metrics computed
- [ ] Smoke test finishes: epoch 2 or 50 steps (whichever first)

### Task 3.3: Check smoke test results
- [ ] Open log: `tail -100 panoptic/logs/job1a_v2_smoke_*.out`
- [ ] Verify training loss decreasing: `grep "Epoch.*avg_loss" panoptic/logs/job1a_v2_smoke_*.out`
- [ ] Verify validation runs: `grep "Val PQ" panoptic/logs/job1a_v2_smoke_*.out`
- [ ] Verify objectness loss computing: `grep "train/loss_obj" panoptic/logs/job1a_v2_smoke_*.out`

### Task 3.4: Checkpoint created
- [ ] `panoptic/checkpoints_job1a_v2/` directory exists
- [ ] At least one checkpoint saved (latest.pth or epoch_*.pth)

**→ If all pass, proceed to PHASE 4. If any fail, debug before proceeding.**

---

## PHASE 4: Full Job 1a v2 Training (7 days - submit and wait)

### Task 4.1: Submit full Job 1a v2
```bash
bsub < panoptic/scripts/job1a_v2_full.sh
```

- [ ] Job submitted
- [ ] Job ID recorded: `_________________`
- [ ] Set calendar reminder: check results in ~7 days

### Task 4.2: Monitor job progress
- [ ] (Day 1-7) Check periodically: `bjobs -l <job_id>`
- [ ] (Day 1-7) Watch logs: `tail -50 panoptic/logs/job1a_v2_full_*.out`
- [ ] Expected: 20 epochs, ~6-8 hours per epoch on A100

### Task 4.3: After completion
- [ ] Job completed successfully (status DONE)
- [ ] Check for best checkpoint: `ls panoptic/checkpoints_job1a_v2/best.pth`
- [ ] Save best PQ: `grep "Val PQ_all" panoptic/logs/job1a_v2_full_*.out | tail -1`
- [ ] Record in comparison sheet

---

## PHASE 5: Evaluate Job 1a v2 vs Job 1a v1

### Task 5.1: Run diagnostic on Job 1a v2 best
```bash
python panoptic/scripts/eval_objectness_hypothesis.py \
    --checkpoint panoptic/checkpoints_job1a_v2/best.pth \
    --config panoptic/configs/job1a_v2_objectness_fix.yaml \
    --split val
```

- [ ] Script runs
- [ ] Output shows delta (should be 0.01-0.03)
- [ ] Compare vs Job 1a v1 delta

### Task 5.2: Prepare comparison report
**File:** `docs/comparison_job1a_v1_vs_v2.md`

- [ ] Record metrics: PQ_all, PQ_base, PQ_novel for both
- [ ] Per-class PQ for all 16 classes
- [ ] Objectness loss values
- [ ] Temperature progression
- [ ] Conclusions

- [ ] Create

### Task 5.3: Verify no base class degradation
- [ ] PQ_base v2 ≥ PQ_base v1 - 0.005 (small margin acceptable)
- [ ] If degradation > 0.005, investigate

---

## PHASE 6: Prepare & Submit Job 1b v2 (7 days)

### Task 6.1: Create job1b_v2 config
**File:** `panoptic/configs/job1b_v2_with_fixed_objectness.yaml`

Copy from `job1b.yaml` and change:
```yaml
warm_start:
  checkpoint: panoptic/checkpoints_job1a_v2/best.pth  # Use v2 not v1

logging:
  wandb_run_name: job1b-v2-fixed-objectness
```

- [ ] Config created
- [ ] Verify syntax

### Task 6.2: Create job1b_v2_full.sh
**File:** `panoptic/scripts/job1b_v2_full.sh`

- [ ] File created (use template from action plan)
- [ ] Make executable: `chmod +x panoptic/scripts/job1b_v2_full.sh`

### Task 6.3: Submit Job 1b v2
```bash
bsub < panoptic/scripts/job1b_v2_full.sh
```

- [ ] Job submitted
- [ ] Job ID recorded: `_________________`
- [ ] Set reminder for ~7 days

### Task 6.4: Monitor Job 1b v2
- [ ] (Day 1-7) Check periodically
- [ ] Expected: 15 epochs, ~6-8 hours per epoch

---

## PHASE 7: Final Evaluation (2 hours)

### Task 7.1: Run diagnostic on Job 1b v2 best
```bash
python panoptic/scripts/eval_objectness_hypothesis.py \
    --checkpoint panoptic/checkpoints_job1b_v2/best.pth \
    --config panoptic/configs/job1b_v2_with_fixed_objectness.yaml \
    --split val
```

- [ ] Script runs
- [ ] Output shows combined effect (should be delta ~0.02)

### Task 7.2: Create final report
**File:** `docs/objectness_fix_final_results.md`

Include:
- [ ] Job 1a v1 metrics (current baseline)
- [ ] Job 1a v2 metrics (objectness fix alone)
- [ ] Job 1b v2 metrics (objectness fix + distillation)
- [ ] Per-class breakdown
- [ ] Delta comparisons
- [ ] Conclusions

### Task 7.3: Archive results
- [ ] Save configs to `docs/configs_v2/`
- [ ] Save logs summary to `docs/logs_summary/`
- [ ] Save comparison plots/tables

---

## Rollback Plan (if needed)

If Job 1a v2 or Job 1b v2 show degradation:

### Option A: Issue with implementation
- [ ] Revert `panoptic/train/train_panoptic_job1.py` to original
- [ ] Revert novel_ids_objectness logic
- [ ] Resubmit Job 1a with original code

### Option B: Issue is real but needs tuning
- [ ] Check if novel GT masks are low quality
- [ ] Experiment with `novel_ids_objectness` filtering specific classes
- [ ] Resubmit with modified config

### Option C: Hypothesis was wrong
- [ ] Diagnostic showed delta < 0.005 (novel embeddings don't exist)
- [ ] Focus on Job 1c or increasing distillation weights instead

---

## Notes & Risks

**Risks to watch:**
1. Base class PQ degrades (novel supervision leaks to base)
   - Mitigation: Keep token loss base-only, only objectness sees novel
   
2. Novel GT masks are low quality, corrupt objectness
   - Mitigation: Inspect Job 1a v2 training loss; if objectness_loss > 1.0, investigate
   
3. Job 1b v2 diverges from Job 1b v1 due to different warm-start
   - Expected: Not directly comparable; need Job 1a v2 + Job 1b v2 as new baseline

**Assumptions:**
- CLIP features precomputed and available at `/fs/scratch/SX_ETL3_GenAI_Team/mha2cob/clip_features/v1.0-trainval/`
- Job 1a v1 best checkpoint exists at `panoptic/checkpoints_job1a/best.pth`
- Matcher function available in training loop

---

## Success Criteria Checklist

### Diagnostic Pass (Phase 1)
- [ ] `delta_job1a > 0.01` (confirm hypothesis)
- [ ] Decision: PROCEED TO OPTION 4

### Smoke Test Pass (Phase 3)
- [ ] No NaN/inf
- [ ] Losses decrease
- [ ] Validation runs
- [ ] Objectness loss computes

### Job 1a v2 Complete (Phase 4)
- [ ] Training completes all 20 epochs
- [ ] `PQ_novel_v2 > 0.005` (improvement over v1 = 0)
- [ ] `PQ_base_v2 >= PQ_base_v1 - 0.005` (no degradation)
- [ ] Best checkpoint saved

### Comparison Pass (Phase 5)
- [ ] Job 1a v2 diagnostic shows improvement
- [ ] No base class degradation

### Job 1b v2 Complete (Phase 6)
- [ ] Training completes all 15 epochs
- [ ] `PQ_novel_v2_job1b > PQ_novel_v1_job1b`
- [ ] Combined effect validates fix

### Final Report Complete (Phase 7)
- [ ] All metrics documented
- [ ] Conclusions drawn
- [ ] Ready for Job 1c or production deployment

