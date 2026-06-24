Train Parameters and Learning Rates

### **Training Parameters & Learning Rates**

| Metric               | Start Value | Middle Value | End Value  | Remarks on Behavior |
| :------------------- | :---------- | :----------- | :--------- | :--------------------------------------------------------------------------------------------- |
| **train/token_weight**| 1.0         | 1.0          | 1.0        | Constant throughout the entire training run.                                                   |
| **train/temperature** | 0.5         | ~0.12        | ~0.07      | Follows a smooth exponential decay curve, plateauing towards the end.                          |
| **train/phase** | 1.0         | 2.0          | 2.0        | Step function. The model shifts from Phase 1 to Phase 2 at roughly the 55k–60k step mark.      |
| **train/lr_lora** | 0.0001      | ~0.00006     | ~0.00001   | Follows a strict step decay schedule, dropping incrementally over the epochs.                  |
| **train/lr_head** | 0.0001      | ~0.00006     | ~0.00001   | Identical step decay schedule to the LoRA learning rate.                                       |
| **train/lr_decoder** | 0.00005     | ~0.00003     | ~0.000005  | Step decay schedule, starting at exactly half the learning rate of the head/LoRA components.   |
| **train/epoch** | 0           | 5            | 10         | Linear staircase progression; 10 full epochs completed by 300k steps.                          |
| **train/lora_gnorm** | ~0.5        | ~0.1         | ~0.1       | Drops sharply right at the phase shift (~55k steps), with a massive outlier spike near 280k.   |

### **Training Losses**

| Metric               | Start Value | Middle Value | End Value  | Remarks on Behavior |
| :------------------- | :---------- | :----------- | :--------- | :--------------------------------------------------------------------------------------------- |
| **train/loss_token** | ~4.5        | ~2.5         | ~2.0       | Consistent, noisy downward trend. Shows standard, healthy convergence.                         |
| **train/loss_obj** | ~0.4        | ~0.4         | ~0.4       | Highly noisy with no clear downward trend; fluctuates heavily between 0.1 and 0.8.             |
| **train/loss_mask** | ~0.25       | ~0.25        | ~0.25      | Noisy but relatively flat average. Occasional upward spikes.                                   |
| **train/loss_lv** | 0.0         | 0.0          | 0.0        | Zeroed out/inactive for this run.                                                              |
| **train/loss_lo** | 0.0         | 0.0          | 0.0        | Zeroed out/inactive for this run.                                                              |
| **train/loss_dice** | ~1.5        | ~1.5         | ~1.5       | Extremely noisy band oscillating between 0.5 and 2.5; average remains relatively flat.         |
| **train/loss** | ~27.0       | ~22.0        | ~20.0      | Tracks well with `loss_token`. Steady, noisy decline indicating overall learning.              |

### **Validation Metrics (Panoptic & Segmentation)**
*(Note: Validation logging appears to start at 60k steps, shortly after the Phase 2 shift).*

| Metric               | Start Value (60k) | Middle Value (170k) | End Value (280k) | Remarks on Behavior |
| :------------------- | :---------------- | :------------------ | :--------------- | :------------------------------------------------------------------------------------------- |
| **val/SQ** | ~0.563            | ~0.567              | ~0.567           | Dips initially at the 110k mark, then recovers sharply and plateaus.                         |
| **val/RQ** | ~0.297            | ~0.296              | ~0.308           | Forms a distinct "V" shape, dipping at 110k before climbing steadily through the end.        |
| **val/PQ_base** | ~0.314            | ~0.314              | ~0.326           | Mirrors RQ exactly. Initial degradation followed by a strong, steady upward trend.           |
| **val/PQ_novel** | 0.0               | 0.0                 | 0.0              | Constant at zero; either no novel classes are in the val set or the metric is inactive.      |
| **val/PQ_all** | ~0.236            | ~0.236              | ~0.244           | Reflects the average of base and novel. Shows the same 110k step dip and subsequent recovery.|
