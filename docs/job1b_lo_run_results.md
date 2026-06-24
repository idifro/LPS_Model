Graph Name,Start Value,Mid Value,End Value,Remark
val_pq/vegetation,0.0,0.0,0.0,Completely flat at 0. (Expected per the design doc since vegetation receives no L_O gradient).
val_pq/truck,~0.400,~0.435,~0.453,"Steady increase early on, gradually saturating toward the end."
val_pq/trailer,~0.260,~0.280,~0.288,"Increases to a peak of ~0.305 around 225k steps, then dips and stabilizes."
val_pq/terrain,~0.306,~0.332,~0.337,"Consistent, steady climb followed by saturation in the second half."
val_pq/sidewalk,~0.238,~0.310,~0.315,"Steep initial climb, then clearly saturates and flattens out."
val_pq/pedestrian,0.0,~0.150,~0.170,"Starts at 0, experiences a steep climb until 175k steps, then settles into a slow, steady increase."

Graph Name,Start Value,Mid Value,End Value,Remark
val_pq/other_flat,~0.200,~0.300,~0.310,"Strong initial upward trend, saturating by the midpoint."
val_pq/motorcycle,0.0,0.0,~0.001,"Flatlines at 0 for almost the entire run, with a sharp, transient late-stage spike up to ~0.007 before dropping back down."
val_pq/manmade,~0.723,~0.732,~0.737,"Smooth, gradual, and steady upward climb."
val_pq/drive_surf,~0.880,~0.893,~0.898,"Consistent increase, gently converging at the end."
val_pq/constr_veh,~0.239,~0.250,~0.258,"Highly volatile; dips initially, spikes to ~0.268 mid-training, and continues to fluctuate."
val_pq/cone,~0.200,~0.260,~0.290,"Steady climb that peaks around ~0.330 late in training, followed by a noticeable drop."

Graph Name,Start Value,Mid Value,End Value,Remark
val_pq/car,~0.798,~0.833,~0.835,"Steep initial climb, followed by saturation with very minor fluctuations."
val_pq/bus,0.0,~0.320,~0.350,"Remains at 0 initially, shoots up sharply after 120k steps, then saturates."
val_pq/bicycle,~0.019,~0.038,~0.028,"Highly volatile. Peaks early at ~0.046, drops sharply, and ends with a slow, slight recovery."
val_pq/barrier,~0.194,~0.202,~0.202,Very volatile with multiple peaks and valleys; shows no clear convergence trend.

Graph Name,Start Value,Mid Value,End Value,Remark
train/token_weight,1.0,1.0,1.0,Absolute flatline.
train/temperature,0.07,0.07,0.07,Flat baseline at 0.07 with negligible micro-spikes (maxing out at barely ~0.07000003).
train/phase,1.0,2.0,2.0,Step function; jumps from phase 1 to 2 around 60k steps and stays there.
train/lr_lora,5.0e-5,3.0e-5,~0.2e-5,"Follows a strict, steady step-decay schedule."
train/lr_head,5.0e-5,3.0e-5,~0.2e-5,"Follows a strict, steady step-decay schedule."
train/lr_decoder,3.0e-5,2.0e-5,~0.2e-5,"Follows a strict, steady step-decay schedule."


Graph Name,Start Value,Mid Value,End Value,Remark
train/loss_token,~2.0,~2.0,~2.0,Highly noisy with a relatively flat overall average throughout.
train/loss_obj,~0.4,~0.4,~0.35,"Very noisy, showing only a very slight downward drift over time."
train/loss_mask,~0.3,~0.3,~0.2,"Noisy, but exhibits a visible and steady downward trend."
train/loss_lv,0.0,0.0,0.0,Absolute flatline at 0 (reflecting ablation A where weight: 0.0).
train/loss_lo,0.0,~0.6,~0.5,"0 during warmup, abruptly spikes to ~0.9 at ~80k steps, then follows a noisy, steady decay."
train/loss_dice,~1.5,~1.5,~1.4,"Highly noisy with an extremely slow, marginal decrease."

Graph Name,Start Value,Mid Value,End Value,Remark
train/loss,~20.0,~18.0,~16.5,"Noisy but displays a clear, steady downward trajectory."
train/lora_gnorm,~0.5,~0.2,~0.2,"Generally flat near zero, interrupted by infrequent but sharp upward spikes."
train/epoch,0,7.5,14,"Perfectly linear staircase, as expected for epoch tracking."
train/count_novel_lo,0,~15 (avg),~15 (avg),"0 during warmup, then highly volatile (ranging from 1 to 30) for the remainder of training."

Graph Name,Start Value,Mid Value,End Value,Remark
val/SQ,~0.580,~0.695,~0.700,"Steady initial climb to a plateau, a late spike to ~0.750, followed by a final drop."
val/RQ,~0.350,~0.410,~0.415,"Climbs smoothly early on, then thoroughly saturates."
val/PQ_novel,0.0,~0.120,~0.130,"Sharp, aggressive early climb starting from 0, transitioning into a slow, steady crawl."
val/PQ_base,~0.380,~0.410,~0.410,"Quick initial climb, followed by a slight dip and a solid flatline to the end."
val/PQ_all,~0.280,~0.340,~0.340,"Smooth, steady progression upward, saturating late in the run."