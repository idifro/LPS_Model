architecture doc says Job 1a is the foundation stage: LoRA + A4 single CLIP head + C1/token contrastive + T4, mainly targeting PQ_base and margin, while true novel improvement is expected later from Job 1b/1c with camera distillation and L7/CASD. It also says B12/N4 excludes novel GT masks from losses, so PQ_novel can only improve through CLIP-space generalization.

Aggregate validation metrics

| Metric       |   Start |     Mid |         End | Read                      |
| ------------ | ------: | ------: | ----------: | ------------------------- |
| val/SQ       | ~0.5638 | ~0.5725 | **0.57903** | Improving                 |
| val/RQ       |  ~0.290 |  ~0.319 | **0.32669** | Improving                 |
| val/PQ_base  |  ~0.315 |  ~0.338 | **0.34939** | Improving, but low        |
| val/PQ_all   |  ~0.235 |  ~0.254 | **0.26204** | Low because novel is zero |
| val/PQ_novel |       0 |       0 |       **0** | No novel transfer yet     |


Training metrics

| Metric           | Start |   Mid |         End | Read                                |
| ---------------- | ----: | ----: | ----------: | ----------------------------------- |
| train/loss_token | ~5.25 | ~5.02 | **5.16342** | Slight early drop, then noisy/flat  |
| train/loss_obj   | ~0.50 | ~0.30 | **0.45674** | Very noisy                          |
| train/loss_mask  | ~0.35 | ~0.20 | **0.12758** | Good downward signal                |
| train/loss_dice  |  ~2.4 |  ~2.1 | **2.21003** | Noisy, modest improvement           |
| train/loss_lv    |     0 |     0 |       **0** | Expected for Job 1a if L_V disabled |
| train/loss_lo    |     0 |     0 |       **0** | Expected for Job 1a if L_O disabled |


Per-class validation PQ

| Metric            |  Start |        Mid |          End | Read                                   |
| ----------------- | -----: | ---------: | -----------: | -------------------------------------- |
| val_pq/drive_surf | ~0.775 |     ~0.859 |  **0.86793** | Strong                                 |
| val_pq/car        | ~0.768 |     ~0.764 |  **0.78157** | Strong                                 |
| val_pq/manmade    | ~0.558 |     ~0.700 |  **0.70248** | Strong, but see note below             |
| val_pq/truck      | ~0.368 |     ~0.365 |  **0.37067** | Moderate, stable                       |
| val_pq/other_flat | ~0.135 |     ~0.208 |  **0.24740** | Good improvement                       |
| val_pq/trailer    | ~0.186 |     ~0.231 |  **0.24562** | Improving                              |
| val_pq/terrain    | ~0.223 |     ~0.233 |  **0.24082** | Slight improvement                     |
| val_pq/constr_veh |  ~0.19 | ~0.19–0.23 |  **0.21150** | Noisy/moderate-low                     |
| val_pq/cone       | ~0.135 |     ~0.156 |  **0.18999** | Improving but low                      |
| val_pq/sidewalk   | ~0.210 |     ~0.154 |  **0.17978** | Recovered, still below start           |
| val_pq/barrier    | ~0.119 |     ~0.123 |  **0.13393** | Low                                    |
| val_pq/bicycle    | ~0.034 |     ~0.020 | **0.020934** | Poor/degraded                          |
| val_pq/motorcycle |      0 |          0 |        **0** | Problematic because it is a base class |
| val_pq/bus        |      0 |          0 |        **0** | Expected if novel                      |
| val_pq/pedestrian |      0 |          0 |        **0** | Expected if novel                      |
| val_pq/vegetation |      0 |          0 |        **0** | Expected if novel                      |

Zero-shot / novel probe metrics

| Metric                           |    Start |  Mid |              End | Read                                 |
| -------------------------------- | -------: | ---: | ---------------: | ------------------------------------ |
| val_zeroshot/trash_can_voxels    |        0 |    0 |            **0** | No detection                         |
| val_zeroshot/trash_can_pct       |        0 |    0 |            **0** | No detection                         |
| val_zeroshot/police_van_voxels   | ~700–750 | ~0–5 |            **9** | Collapsed from early false positives |
| val_zeroshot/police_van_pct      | ~0.00063 |   ~0 | **0.0000075308** | Near zero                            |
| val_zeroshot/fire_hydrant_voxels |        0 |    0 |            **0** | No detection                         |
| val_zeroshot/fire_hydrant_pct    |        0 |    0 |            **0** | No detection                         |

