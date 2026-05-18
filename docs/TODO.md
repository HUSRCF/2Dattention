# TODO: 2Dattention Research Roadmap

## Current P0: Proposal Consumption + Frozen Rank Calibration

Goal: move from classification/probe models toward a real detection model family that can eventually be compared against RF-DETR-style detectors. Based on the current evidence, the project should no longer be framed as early prefill or memory-first. The active research route is:

```text
local-state backbone
+ anchor/proposal query coupling
+ low-interference frozen quality ranking
```

Boundary:

- Current supported signal: clean no-prefill online anchor/region interaction improves dense bbox-mask localization.
- Current supported detector signal: residual-anchor queries plus two-stage frozen quality ranking improve real-mini AP under held-out alpha calibration. Fixed `q^2` remains a pre-registered baseline, while `eval_ap50_q_fixed` can be calibration-selected when `--calibration-frac` is enabled.
- Current unsupported claims: early prefill, stale history pools, full prefill-lattice memory, graph-prefill, side-only DenseMaskAux, quality loss on class logits, and old memory-first routing.
- RF-DETR-level performance requires a real detector, not just classification or bbox-mask probes.

Active collision-avoidance route:

- Avoid claiming "new DETR query initialization"; frame the detector path as layerwise proposal refresh / `reinject` plus ranking calibration.
- Do not frame persistent proposal state as the active architecture. The standard stability matrix did not rescue predicted persistent; keep it only as an oracle/proposal-quality diagnostic branch.
- Avoid claiming "new quality score"; frame the quality path as a frozen, post-detector, held-out calibrated ranking head.
- Avoid claiming "new local-global backbone"; if this route returns, frame it as ambiguity-triggered interaction scheduling.
- See `docs/research_route_collision_avoidance.md` for the detailed roadmap, stage gates, and IP/literature risk framing.

Immediate stage gates:

- P0 calibration gate: frozen ranking should produce meaningful held-out AP75/ECE or score-IoU correlation gains before expanding calibration.
- P1 proposal-consumption gate: layerwise proposal refresh / `reinject` is the active predicted-proposal path. Persistent state can only return to mainline if a future proposal-quality/oracle-gap run beats `reinject` on both geometry and AP without post-hoc tuning.
- Current update: held-out alpha calibration now supports the frozen quality route; query-mask moment refinement and longer proposal refresh did not.
- Protocol correction: existing historical real-mini artifacts used legacy `--label-map-source all`; future strict held-out detector runs should use `--label-map-source train` to avoid held-out label-frequency leakage when selecting top classes.
- Reporting tool: use `scripts/summarize_det_real_results.py` to summarize final-step CSV metrics and paired deltas. Its default columns include both pre-registered `eval_ap50_q2` and calibration/fixed-score `eval_ap50_q_fixed` plus `eval_ap50_q_fixed_alpha`.
- Strict protocol check: `results/det_real_quality_calib_trainlabels_400step_3seed.csv` confirms the quality-ranking signal survives `--label-map-source train` with the same key paired gains as the legacy label-map run.
- Larger strict check: `results/det_real_quality_calib_trainlabels_500img_500step_3seed.csv` strengthens the same conclusion. Quality ranking gives AP50 `+0.046` (`3/3`), fixed AP50 `+0.082` (`3/3`), and combined ECE50/ECE75 drops from `0.291/0.320` to `0.196/0.104`, while final IoU only moves `+0.006`.
- Quality-head generalization: `results/det_real_quality_generalization_trainlabels_400step_3seed.csv` shows the two-stage frozen quality head also improves `local_learned` strongly (AP50 `+0.091`, q2 AP50 `+0.095`, fixed AP50 `+0.082`, all `3/3` vs `local_learned`). Treat quality ranking as a general post-detector fix, not only a residual-anchor-specific trick.
- Larger learned-query check: `results/det_real_learned_quality_trainlabels_500img_500step_3seed.csv` keeps the quality-weighted scoring and ECE benefit, but the base AP50 drops (`-0.040`) and fixed AP50 gain is weaker (`+0.037`, `2/3`). The best 500-image recipe remains residual-anchor base plus two-stage quality ranking.
- 1000-image strict check: `results/det_real_quality_calib_trainlabels_1000img_500step_3seed.csv` is now the strongest detector-side evidence. Residual-anchor + two-stage quality gives AP50 `+0.080` (`3/3`), q2 AP50 `+0.127` (`3/3`), fixed AP50 `+0.126` (`3/3`), and combined ECE50/ECE75 drops from `0.257/0.310` to `0.171/0.027`, while final IoU only moves `+0.006`.
- Protocol upgrade implemented: `--calibration-source train` now carves the alpha-calibration subset out of training data instead of heldout, leaving final eval independent of alpha selection. Slice robustness columns were added for small/medium/large and center/offcenter AP50 plus fixed-quality AP50. Smoke artifact: `results/det_real_traincalib_slice_smoke.csv`.
- Strict train-calibration 1000-image check: `results/det_real_quality_traincalib_1000img_500step_3seed.csv` keeps the ranking/calibration signal without selecting alpha on heldout. Residual-anchor + two-stage quality gives AP50 `+0.057` (`2/3`), q2 AP50 `+0.112` (`3/3`), fixed AP50 `+0.113` (`3/3`), fixed AP75 `+0.020` (`2/3`), and combined ECE50/ECE75 drops from `0.292/0.338` to `0.148/0.028`, while final IoU moves only `+0.002`.
- Slice robustness from the strict train-calibration run is uneven: fixed-quality AP improves strongly on `large` (`+0.131`, `3/3`) and `center` (`+0.164`, `3/3`) objects, is only marginal on `medium` (`+0.005`, `2/3`), remains unsupported on `small` (`0.000` for both models), and drops on `offcenter` (`-0.029`, `1/3`). Next quality-ranking work should target off-center/small robustness rather than new proposal-state architectures.
- Slice-stress protocol added: `--eval-slice-filter {small,medium,large,center,offcenter}` restricts the final eval split to images containing the requested robustness slice while keeping train/calibration unchanged. Smoke artifact: `results/det_real_offcenter_slice_smoke.csv`. Use this for off-center-only and small-only robustness checks before claiming OOD/slice stability.
- Off-center stress result: `results/det_real_quality_traincalib_offcenter_1000img_500step_3seed.csv` confirms that off-center remains the weak slice. Eval contains `90/83/86` off-center images across seeds. Quality ranking gives fixed AP50 `+0.021` (`2/3`) and fixed AP75 `+0.003` (`2/3`), but base AP50 is `-0.003`, class AP50 is `-0.018`, best IoU is lower, and offcenter fixed AP is `-0.035` (`1/3`). Treat this as a robustness blocker, not a failure of the aggregate quality route.
- Slice-aware calibration hook added: `--calibration-slice-filter {small,medium,large,center,offcenter}` lets alpha selection use only calibration images containing the requested slice. Use this next for off-center-calibrated scoring before changing model structure.
- Off-center-calibrated stress result: `results/det_real_quality_traincalib_offcenter_calib_1000img_500step_3seed.csv` shows slice-aware alpha selection does not fix off-center AP. Calibration uses `66/67/66` off-center train-calibration images and eval uses `90/83/86` off-center images. Quality improves final IoU `+0.002` and fixed AP75 `+0.004`, but AP50 is `-0.005`, fixed AP50 is `-0.001`, class fixed AP50 is `-0.011`, and offcenter fixed AP50 is `-0.015`. ECE still improves. Conclusion: the off-center problem is not just aggregate-alpha mismatch; it needs representation/query or slice-specific scoring work.
- Box-aware off-center quality check: `results/det_real_box_quality_traincalib_offcenter_1000img_500step_3seed.csv` shows that concatenating detached predicted boxes into the quality head also does not fix off-center AP. Fixed AP50 is `+0.021` (`2/3`), but base AP50 is `-0.003`, class AP50 is `-0.018`, and offcenter fixed AP50 is `-0.035`. This matches the query-only quality behavior, so the blocker is not simply missing box geometry in the quality head input.
- Small-object stress result: `results/det_real_quality_traincalib_small_1000img_500step_3seed.csv` shows that small-object robustness is not the same failure mode as off-center. Eval contains `30/31/33` small-slice images. Quality improves final IoU `+0.009`, AP50 `+0.020`, q2 AP50 `+0.033` (`3/3`), fixed AP50 `+0.032` (`3/3`), and fixed AP75 `+0.008` (`2/3`). Treat small-object evidence as positive but high-variance; the main unresolved robustness blocker is off-center.
- Position-conditioned class scoring hook added: `local_anchor_residual_query_box_class_head` and `local_anchor_residual_query_box_class_quality_head` let the class/objectness head consume `[query, detached_pred_box]`. This tests whether off-center weakness comes from class/objectness logits missing position/box context, rather than from the separate quality head.
- Position-conditioned off-center result: `results/det_real_box_class_traincalib_offcenter_1000img_500step_3seed.csv` shows that adding detached box coordinates to the class/objectness head does not fix the off-center slice. `local_anchor_residual_query_box_class_head` loses final IoU `-0.002`, AP50 `-0.014`, class AP50 `-0.014`, and offcenter fixed AP50 `-0.037` vs the base detector. `local_anchor_residual_query_box_class_quality_head` improves ECE but still loses final IoU `-0.009`, fixed AP50 `-0.005`, fixed AP75 `-0.004`, and offcenter fixed AP50 `-0.041`. Conclusion: the off-center blocker is not solved by simply giving class/quality heads detached box coordinates.
- Current off-center decision: stop global alpha sweeps, persistent proposal-state rescue, and simple box-aware head concatenation. The next useful detector work should be diagnostic-first: slice-stratified query assignment, matched class accuracy, TP50 class accuracy, duplicate/high-score false positive behavior, score-IoU correlation, and query identity/position diagnostics for center vs offcenter examples. Only add a new representation/query mechanism after that diagnosis identifies the failure mode.
- Slice-stratified ranking diagnostics added: real DET CSV rows now include `center_*` and `offcenter_*` metrics for matched assignment class accuracy, TP50 class accuracy, score-IoU correlation, objectness AUC, objectness top-k FP rate, fixed-quality combined top-k FP rate, and duplicate-per-GT. Smoke artifacts: `results/det_real_slice_diagnostics_smoke.csv` and `results/det_real_combined_topk_slice_smoke.csv`. Use these fields in the next off-center run before introducing another model variant.
- Formal off-center diagnostic rerun: `results/det_real_offcenter_diagnostics_1000img_500step_3seed.csv` confirms the same AP boundary and exposes the likely failure mode. Quality ranking improves q2/fixed AP50 `+0.021` and center fixed AP50 `+0.130` (`3/3`), and sharply improves combined ECE, but offcenter fixed AP50 drops `-0.035` and class AP50 drops `-0.018`. Diagnostics show very high offcenter top-k FP rates for both base and quality models (`0.965 -> 0.981`), while offcenter objectness AUC does not improve (`0.525 -> 0.507`). Conclusion: quality helps center/large ranking and calibration, but off-center still suffers from high-scoring false candidates / query-position representation, not from missing alpha, box geometry, or basic quality calibration.
- Combined top-k formal rerun: `results/det_real_offcenter_combined_topk_1000img_500step_3seed.csv` confirms that fixed-quality scoring reduces overall combined top-k FP (`0.853 -> 0.819`) and center combined top-k FP (`0.845 -> 0.679`), but not off-center combined top-k FP (`0.965 -> 0.982`). This pins the robustness blocker more tightly: the quality head can reorder center candidates, but it cannot suppress off-center high-score false candidates.
- Off-center representation hook added: `local_grid_residual_query` and `local_grid_residual_query_quality_head` initialize residual queries from a coarse 2D grid over the feature lattice rather than row/column/global anchors. Smoke artifact: `results/det_real_grid_residual_offcenter_smoke.csv`. This is a candidate off-center representation control, not yet a formal result.
- Grid-residual formal result: `results/det_real_grid_residual_offcenter_1000img_500step_3seed.csv` shows a small base-detector AP signal but not a solution to off-center robustness. `local_grid_residual_query` improves AP50 `+0.013` (`2/3`), q/fixed AP50 `+0.013`, center fixed AP50 `+0.022` (`3/3`), and final IoU `+0.002`, but offcenter fixed AP50 is only `+0.005` (`1/3`) and offcenter top-k FP remains very high (`0.965 -> 0.957`). `local_grid_residual_query_quality_head` improves q/fixed AP50 `+0.016` (`3/3`) and center fixed AP50 `+0.098` (`3/3`), but offcenter fixed AP50 drops `-0.020` and offcenter combined top-k FP stays near-saturated (`0.994`). Conclusion: coarse grid query coverage is a useful lightweight control, but it does not solve off-center high-score false candidates.
- Top-k center-distance diagnostics added: real DET CSV rows now include `topk_center_distance`, `combined_topk_center_distance`, and center/offcenter-stratified versions. Smoke artifact: `results/det_real_topk_center_distance_smoke.csv`. Use this to test whether off-center high-score false candidates are center-biased or merely wrong for another reason.
- Top-k center-distance formal result: `results/det_real_topk_center_distance_1000img_500step_3seed.csv` confirms a center-biased high-score failure mode. Fixed-quality scoring improves q/fixed AP50 `+0.021` (`2/3`) and center fixed AP50 `+0.130` (`3/3`), but offcenter fixed AP50 drops `-0.035` (`1/3`) and class AP50 drops `-0.018`. The combined top-k center distance collapses from `0.183 -> 0.121` overall and from `0.168 -> 0.064` on center targets, while offcenter combined top-k center distance also shrinks from `0.165 -> 0.073`. This means quality ranking is selecting more center-near high-score boxes even when evaluating off-center images. Next work should directly address off-center proposal/query position coverage or suppress center-biased false positives, not continue alpha sweeps, persistent proposal-state rescue, or generic score calibration.
- Slice-match breakdown added after code review: off-center top-k diagnostics now split high-score predictions into slice-GT matches, non-slice-GT matches, and no-GT matches. Smoke artifact: `results/det_real_slice_match_breakdown_smoke.csv`.
- Slice-match formal result: `results/det_real_slice_match_breakdown_1000img_500step_3seed.csv` refines the off-center conclusion. For off-center combined top-k predictions, the base detector has slice/non-slice/no-GT rates `0.035/0.121/0.844`; the quality head changes them to `0.018/0.199/0.783`. Thus the previous off-center FP metric was partly inflated by same-image non-slice GT matches, but the dominant failure remains no-GT high-score candidates. Quality ranking increases center-near selection (`offcenter_combined_topk_center_distance 0.165 -> 0.073`) and shifts some high-score candidates toward non-slice GT rather than the off-center object. Next work should use this three-way breakdown as the official diagnostic and target off-center object-specific query assignment/proposal selection.
- Center-distance penalty diagnostic added: `eval_ap50_center_dist1`, `eval_ap50_q_fixed_center_dist1`, and center/offcenter variants multiply scores by predicted-box distance from image center. Smoke artifact: `results/det_real_center_penalty_smoke.csv`.
- Center-distance penalty formal result: `results/det_real_center_penalty_1000img_500step_3seed.csv` shows that eval-only center suppression does not solve off-center AP. The base detector's offcenter AP improves under pure center-distance scoring (`0.067 -> 0.085`), but the quality model remains worse: offcenter center-dist AP `0.056`, offcenter `q_fixed * center_dist` AP `0.054`, versus base offcenter AP `0.067`. Center AP still improves strongly under `q_fixed * center_dist` (`0.123 -> 0.256`). Conclusion: off-center failure cannot be fixed by a simple scoring penalty against center-near boxes; the next step should change query/proposal target binding or off-center proposal selection.
- Edge-grid query control added: `local_edge_grid_residual_query` and `local_edge_grid_residual_query_quality_head` use corner/edge-biased feature tokens as residual query seeds. Smoke artifact: `results/det_real_edge_grid_offcenter_smoke.csv`.
- Edge-grid formal result: `results/det_real_edge_grid_offcenter_1000img_500step_3seed.csv` is negative. Relative to `local_anchor_residual_query`, `local_edge_grid_residual_query` loses final IoU `-0.010` (`0/3` wins), AP50 `-0.009`, and AP75 `-0.004`, while offcenter AP only moves `+0.001`. The quality version loses AP50 `-0.017` (`0/3`), class AP50 `-0.013`, and offcenter fixed AP `-0.021`. The earlier generic grid control remains better: `local_grid_residual_query` keeps the small AP50 signal (`+0.013`) and offcenter center-dist AP signal (`+0.025`). Conclusion: naive edge/corner-biased query coverage is worse than regular coarse grid coverage and should not be pursued. The off-center problem is not solved by simple spatial query seeding.

### Step 0: Persistent Proposal-State Mini Probe

Goal: test whether proposal evidence is merely useful at initialization time or remains useful when consumed across decoder steps.

Implemented minimal controls:

- `local_mask_proposal_nms_query`: existing predicted proposal init-only baseline.
- `local_mask_proposal_nms_query_decode2`: depth-matched init-only control with two shared decoder reads but no proposal re-injection.
- `local_mask_proposal_nms_query_reinject`: two decoder reads; the second read receives the original proposal tokens again.
- `local_mask_proposal_nms_query_persistent`: two decoder reads with a learned proposal state update after each read.
- Oracle counterparts:
  - `local_mask_proposal_oracle_nms_query`
  - `local_mask_proposal_oracle_nms_query_decode2`
  - `local_mask_proposal_oracle_nms_query_reinject`
  - `local_mask_proposal_oracle_nms_query_persistent`

Added summary metric:

```text
proposal_oracle_gap_mode,metric,pred_model,oracle_model,
pred_mean,oracle_mean,direct_gap_mean,closure_vs_init_oracle_gap_mean
```

Definition:

```text
closure_vs_init_oracle_gap =
  (candidate_predicted_metric - init_only_predicted_metric)
  / (init_only_oracle_metric - init_only_predicted_metric)
```

Interpretation:

- `0.0`: candidate does not close the init-only predicted-vs-oracle gap.
- `1.0`: candidate closes the entire init-only oracle gap.
- `<0`: candidate is worse than init-only predicted proposal.
- `>1`: candidate exceeds the init-only oracle reference on that metric.
- If the init-only oracle metric is not higher than the init-only predicted metric, there is no positive oracle headroom and closure is reported as `0.0`.

Smoke command:

```bash
/opt/anaconda3/envs/AIAA/bin/python -u scripts/train_det_real.py \
  --models local_mask_proposal_nms_query local_mask_proposal_nms_query_decode2 \
  local_mask_proposal_nms_query_reinject local_mask_proposal_nms_query_persistent \
  local_mask_proposal_oracle_nms_query local_mask_proposal_oracle_nms_query_decode2 \
  local_mask_proposal_oracle_nms_query_reinject local_mask_proposal_oracle_nms_query_persistent \
  --reference-model local_mask_proposal_nms_query \
  --top-classes 10 --max-samples 80 --max-objects 3 --num-queries 6 \
  --steps 40 --eval-every 40 --eval-batches 2 --batch-size 8 \
  --out results/det_real_proposal_state_smoke.csv \
  --label-map-out results/det_real_proposal_state_smoke_label_map.csv \
  --split-out results/det_real_proposal_state_smoke_split.csv \
  --seeds 1
```

Smoke observation, not formal evidence:

- The new variants train and evaluate on MPS.
- The new `proposal_oracle_gap_mode` table is emitted correctly.
- In this tiny smoke, `persistent` gives positive final-IoU closure before the closure-definition cleanup, but hurts base AP50; `reinject` improves AP50 but does not improve final IoU. This is a mechanism smoke only and should be rerun with the standard 200-image / 2-seed protocol before making any claim.

Standard real-mini result:

- Artifact: `results/det_real_proposal_state_300step_2seed.csv`.
- Protocol: top-10 classes, 200 images, max 3 objects, 6 queries, 300 steps, 2 seeds, eval every 100 steps.

| Model | Final IoU | Best IoU | Recall50 | AP50 | Class AP50 | Mask IoU |
|---|---:|---:|---:|---:|---:|---:|
| `local_mask_proposal_nms_query` | 0.357 | 0.366 | 0.302 | 0.266 | 0.093 | 0.458 |
| `local_mask_proposal_nms_query_decode2` | 0.344 | 0.352 | 0.265 | 0.221 | 0.088 | 0.443 |
| `local_mask_proposal_nms_query_reinject` | 0.367 | 0.373 | 0.293 | 0.258 | 0.111 | 0.386 |
| `local_mask_proposal_nms_query_persistent` | 0.356 | 0.367 | 0.324 | 0.262 | 0.121 | 0.420 |
| `local_mask_proposal_oracle_nms_query` | 0.399 | 0.421 | 0.340 | 0.259 | 0.138 | 1.000 |
| `local_mask_proposal_oracle_nms_query_decode2` | 0.399 | 0.415 | 0.380 | 0.313 | 0.086 | 1.000 |
| `local_mask_proposal_oracle_nms_query_reinject` | 0.402 | 0.416 | 0.354 | 0.279 | 0.103 | 1.000 |
| `local_mask_proposal_oracle_nms_query_persistent` | 0.405 | 0.421 | 0.413 | 0.322 | 0.081 | 1.000 |

Proposal-gap closure summary:

- `decode2`: worse than init-only on predicted final/best IoU; depth alone is not the answer.
- `reinject`: predicted final-IoU closure `0.608`, best-IoU closure `0.140`, but AP50 is slightly lower than init-only.
- `persistent`: predicted final-IoU closure is negative (`-0.335`) and best-IoU closure is near zero (`-0.014`), but class AP50 improves by `+0.027` over init-only with `2/2` wins.
- Oracle persistent is strongest in oracle AP50 (`0.322`) and final IoU (`0.405`), but its class AP50 is weak (`0.081`).

Interpretation:

- `decode2` is a useful negative control: simply adding decoder depth does not explain proposal consumption gains.
- `reinject` is currently the most reliable predicted proposal-consumption candidate for geometry, because it beats init-only on final/best IoU mean.
- `persistent` is not yet supported as the main predicted-proposal architecture. It may improve query/class behavior, but it does not close the predicted-vs-oracle geometry gap under the current update rule.
- Oracle persistent remains promising, which means the consumer idea may have capacity if proposal quality/state update is better controlled.
- Next step should not add teacher, dense positives, or a larger backbone yet. Do only micro-stabilization:
  - `persistent_gated` with a smaller update/read gate,
  - `late_persistent`, only in the last decoder step,
  - `persistent_stopgrad`, detach proposal-state update input,
  - post-detector frozen quality ranking on the completed checkpoints.

Persistent-stability follow-up:

- Implemented predicted variants:
  - `local_mask_proposal_nms_query_persistent_gated`: same persistent state path, but with a small `gate_init=0.01`.
  - `local_mask_proposal_nms_query_late_persistent`: first decoder read is normal; proposal state is injected only into the second read.
  - `local_mask_proposal_nms_query_persistent_stopgrad`: proposal state update consumes detached decoded query features.
- Implemented oracle counterparts:
  - `local_mask_proposal_oracle_nms_query_persistent_gated`
  - `local_mask_proposal_oracle_nms_query_late_persistent`
  - `local_mask_proposal_oracle_nms_query_persistent_stopgrad`
- Standard protocol artifact: `results/det_real_persistent_stability_300step_2seed.csv`.
- Protocol: top-10 classes, 200 images, max 3 objects, 6 queries, 300 steps, 2 seeds, eval every 100 steps.

| Model | Final IoU | Best IoU | Recall50 | AP50 | Class AP50 | Mask IoU |
|---|---:|---:|---:|---:|---:|---:|
| `local_mask_proposal_nms_query` | 0.357 | 0.366 | 0.302 | 0.266 | 0.093 | 0.458 |
| `local_mask_proposal_nms_query_late_persistent` | 0.339 | 0.359 | 0.312 | 0.252 | 0.089 | 0.463 |
| `local_mask_proposal_nms_query_persistent_gated` | 0.328 | 0.349 | 0.264 | 0.234 | 0.099 | 0.459 |
| `local_mask_proposal_nms_query_persistent_stopgrad` | 0.337 | 0.363 | 0.310 | 0.249 | 0.090 | 0.387 |
| `local_mask_proposal_oracle_nms_query` | 0.399 | 0.421 | 0.340 | 0.259 | 0.138 | 1.000 |
| `local_mask_proposal_oracle_nms_query_late_persistent` | 0.406 | 0.412 | 0.386 | 0.270 | 0.095 | 1.000 |
| `local_mask_proposal_oracle_nms_query_persistent_gated` | 0.383 | 0.421 | 0.346 | 0.213 | 0.072 | 1.000 |
| `local_mask_proposal_oracle_nms_query_persistent_stopgrad` | 0.405 | 0.412 | 0.380 | 0.300 | 0.118 | 1.000 |

Paired vs predicted init-only baseline:

- `late_persistent`: final IoU `-0.019`, AP50 `-0.014`, class AP50 `-0.004`.
- `persistent_gated`: final IoU `-0.029`, AP50 `-0.033`, class AP50 `+0.006`.
- `persistent_stopgrad`: final IoU `-0.020`, AP50 `-0.017`, class AP50 `-0.004`.

Interpretation:

- The predicted-proposal persistent variants do not rescue persistent stability. All three lose final IoU and AP50 against the init-only predicted proposal baseline under the standard 200-image / 2-seed / 300-step protocol.
- `persistent_stopgrad` nearly recovers best IoU (`-0.003` vs init-only), but still loses final IoU/AP and therefore is not enough to keep persistent as the main predicted-proposal route.
- Oracle `persistent_stopgrad` is positive: final IoU `+0.047`, AP50 `+0.034`, class AP50 `+0.025`, all `2/2` paired wins vs predicted init-only. This means the consumer idea can work when proposal quality/state inputs are idealized, but the current predicted-proposal state path is too unstable.
- Mainline decision: shrink the active proposal-consumption path to lighter layerwise proposal refresh / `reinject` for predicted proposals. Keep persistent only as an oracle/proposal-quality follow-up, not as the current main architecture.

Layerwise refresh + two-stage quality follow-up:

- Added quality-head variants for the active refresh path:
  - `local_mask_proposal_nms_query_reinject_quality_head`
  - `local_mask_proposal_oracle_nms_query_reinject_quality_head`
- Standard artifact: `results/det_real_reinject_quality_head_400step_2seed.csv`.
- Protocol: top-10 classes, 200 images, max 3 objects, 6 queries, 400 steps, 2 seeds. First 300 steps train the detector normally; after step 301, freeze the detector and train only the quality head. Formal quality score uses fixed `q^2`, `temperature=1.0`.

| Model | Final IoU | Best IoU | AP50 | Class AP50 | Fixed-q2 AP50 | Fixed-q2 Class AP50 | IoU-ref AP50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `local_anchor_residual_query_quality_head` | 0.376 | 0.376 | 0.266 | 0.135 | 0.341 | 0.185 | 0.421 |
| `local_mask_proposal_nms_query_quality_head` | 0.357 | 0.366 | 0.266 | 0.093 | 0.282 | 0.108 | 0.364 |
| `local_mask_proposal_nms_query_reinject_quality_head` | 0.367 | 0.373 | 0.258 | 0.111 | 0.284 | 0.124 | 0.390 |
| `local_mask_proposal_oracle_nms_query_quality_head` | 0.399 | 0.421 | 0.259 | 0.138 | 0.325 | 0.173 | 0.372 |
| `local_mask_proposal_oracle_nms_query_reinject_quality_head` | 0.402 | 0.416 | 0.279 | 0.103 | 0.313 | 0.116 | 0.402 |

Interpretation:

- `reinject_quality_head` keeps the geometry gain of layerwise refresh: final/best IoU `0.367/0.373`, above init-only proposal quality `0.357/0.366`.
- Fixed-q2 quality scoring does not turn predicted `reinject` into the best AP route: AP50 is `0.284`, essentially tied with init-only proposal quality `0.282`, and far below residual-anchor quality `0.341`.
- `reinject` improves class-aware fixed-q2 AP over init-only proposal quality (`0.124` vs `0.108`), so refresh helps class/region coupling somewhat.
- Oracle `reinject` improves geometry and base AP over oracle init-only, but fixed-q2 AP is lower (`0.313` vs `0.325`). This suggests current quality calibration does not fully align with the refreshed proposal boxes.
- Mainline status: keep `reinject` as the active predicted-proposal geometry/coverage path, but do not claim it solves AP. The AP route remains `local_anchor_residual_query + two-stage quality`; proposal refresh remains a coverage path that needs better quality calibration or box-score coupling.

Layerwise refresh gate sweep:

- Added refresh-strength variants:
  - `local_mask_proposal_nms_query_reinject_g001`: gate init `0.01`
  - `local_mask_proposal_nms_query_reinject_g003`: gate init `0.03`
  - `local_mask_proposal_nms_query_reinject_g03`: gate init `0.30`
- Standard artifact: `results/det_real_reinject_gate_sweep_300step_2seed.csv`.
- Protocol: top-10 classes, 200 images, max 3 objects, 6 queries, 300 steps, 2 seeds.

| Model | Final IoU | Best IoU | AP50 | Class AP50 | Mask IoU |
|---|---:|---:|---:|---:|---:|
| `local_mask_proposal_nms_query` | 0.357 | 0.366 | 0.266 | 0.093 | 0.458 |
| `reinject_g001` | 0.348 | 0.348 | 0.239 | 0.075 | 0.406 |
| `reinject_g003` | 0.352 | 0.377 | 0.274 | 0.150 | 0.436 |
| `reinject_g01` | 0.367 | 0.373 | 0.258 | 0.111 | 0.386 |
| `reinject_g03` | 0.366 | 0.366 | 0.242 | 0.119 | 0.468 |

Interpretation:

- Very weak refresh (`0.01`) is underpowered and loses on geometry/AP.
- The default `0.10` remains the strongest final-IoU refresh setting, but it is not the best AP/class setting.
- `0.03` is the best ranking/class candidate: AP50 `0.274`, class AP50 `0.150`, and best IoU `0.377`. It has final-IoU instability, so it should be treated as a candidate for longer training or best-checkpoint/quality-head follow-up, not as a solved default.
- Strong refresh (`0.30`) helps dense mask coverage but hurts AP. This suggests over-refreshing proposal evidence can improve region coverage while worsening score/box ranking.
- Next active refresh control: run `reinject_g003_quality_head` or a longer 500-step `g003` check before adding new architecture.

Best-checkpoint restore before quality head:

- Added `--restore-best-before-quality-head`. When used with `--quality-head-only-after-start`, the script restores the best detector checkpoint observed before the frozen quality-head phase.
- Motivation: `reinject_g003` often reaches its best detector IoU before step 300, then partially regresses before quality-only training begins. Training the quality head on the last detector checkpoint can hide the useful refresh setting.
- Artifact: `results/det_real_reinject_g003_quality_bestrestore_400step_2seed.csv`.
- Protocol: top-10 classes, 200 images, max 3 objects, 6 queries, 400 steps, 2 seeds, quality head starts at step 301, detector restored to best pre-quality checkpoint.

| Model | Final IoU | Best IoU | AP50 | Class AP50 | Fixed-q2 AP50 | Fixed-q2 Class AP50 | IoU-ref AP50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `local_anchor_residual_query_quality_head` | 0.376 | 0.376 | 0.276 | 0.168 | 0.338 | 0.201 | 0.397 |
| `local_mask_proposal_nms_query_reinject_g003_quality_head` | 0.377 | 0.377 | 0.272 | 0.120 | 0.319 | 0.116 | 0.419 |

Interpretation:

- Best-checkpoint restore is a real engineering fix for unstable refresh settings: `g003_quality` final IoU improves from `0.352` without restore to `0.377` with restore.
- `g003` with best restore is now a credible coverage/geometry candidate, matching residual-anchor final IoU while retaining a higher IoU-reference AP headroom (`0.419`).
- It still does not beat residual-anchor quality on fixed-q2 AP (`0.319` vs `0.338`) or class-aware fixed-q2 AP (`0.116` vs `0.201`).
- Current conclusion: restore-best makes layerwise refresh viable as a geometry path, but ranking/class calibration is still the blocker. Do not add new proposal architecture until fixed scoring for refreshed proposal boxes improves.

Box-aware quality head check:

- Added `quality_mode="box"` for `DetectionHead`. The quality head can now receive `[query_feature, detached_pred_box]` instead of query feature alone.
- Added variants:
  - `local_anchor_residual_query_box_quality_head`
  - `local_mask_proposal_nms_query_reinject_g003_box_quality_head`
- Artifact: `results/det_real_box_quality_head_bestrestore_400step_2seed.csv`.
- Protocol: same two-stage frozen quality setup with `--restore-best-before-quality-head`.

| Model | Final IoU | AP50 | Fixed-q2 AP50 | Fixed-q2 Class AP50 | IoU-ref AP50 |
|---|---:|---:|---:|---:|---:|
| `local_anchor_residual_query_box_quality_head` | 0.376 | 0.276 | 0.340 | 0.202 | 0.397 |
| `local_mask_proposal_nms_query_reinject_g003_box_quality_head` | 0.377 | 0.272 | 0.310 | 0.116 | 0.419 |

Interpretation:

- Adding detached predicted box geometry to the quality head does not materially improve refreshed-proposal scoring.
- For residual-anchor, box-aware quality is effectively tied with query-only quality.
- For `g003` refresh, box-aware quality is slightly worse than query-only quality under fixed `q^2` (`0.310` vs `0.319`).
- Current blocker is not solved by a simple box-aware quality input. The next scoring work should change target/calibration or query-box coupling, not just concatenate box geometry into the quality head.

Proposal-residual query init check:

- Added `mask_proposal_residual_nms`: query initialization becomes `learned_query + gate * proposal_query`, instead of replacing learned queries with proposal tokens.
- Added variant:
  - `local_mask_proposal_residual_nms_query_reinject_g003_quality_head`
- Artifact: `results/det_real_proposal_residual_query_400step_2seed.csv`.
- Protocol: same two-stage frozen quality setup with best-detector restore.

| Model | Final IoU | AP50 | Class AP50 | Fixed-q2 AP50 | Fixed-q2 Class AP50 | IoU-ref AP50 |
|---|---:|---:|---:|---:|---:|---:|
| `local_anchor_residual_query_quality_head` | 0.376 | 0.276 | 0.168 | 0.338 | 0.201 | 0.397 |
| `local_mask_proposal_nms_query_reinject_g003_quality_head` | 0.377 | 0.272 | 0.120 | 0.319 | 0.116 | 0.419 |
| `local_mask_proposal_residual_nms_query_reinject_g003_quality_head` | 0.369 | 0.259 | 0.078 | 0.309 | 0.091 | 0.362 |

Interpretation:

- Simple `learned + proposal residual` query initialization does not help. It loses to pure `g003` refresh on final IoU, AP50, class AP50, fixed-q2 AP, and IoU-reference AP.
- This suggests the issue is not that proposal queries replaced learned queries too aggressively. A naive residual mixture can dilute both learned-query ranking and proposal coverage.
- Downgrade proposal-residual query init. Keep pure `g003` refresh as the coverage path and residual-anchor quality as the AP path.

Longer refresh training check:

- Artifact: `results/det_real_reinject_longer_600step_2seed.csv`.
- Protocol: `reinject_g003` vs default `reinject_g01`, 600 steps, 2 seeds, no quality head.

| Model | Final IoU | Best IoU | AP50 | Class AP50 | Mask IoU |
|---|---:|---:|---:|---:|---:|
| `reinject_g003` | 0.354 | 0.377 | 0.208 | 0.104 | 0.460 |
| `reinject_g01` | 0.351 | 0.379 | 0.243 | 0.111 | 0.477 |

Interpretation:

- Longer training does not rescue refresh final performance.
- `g003` still has good best-IoU behavior, but final AP drops by 600 steps. Its signal is an early/best-checkpoint signal, not a "train longer" signal.
- Default `g01` is slightly better on final AP under 600 steps, but it also drops compared with its 300-step result.
- This supports the best-checkpoint restore path and argues against simply extending training length as the next solution.

### Step 1: Detection Scaffold

Implement minimal DETR-style components under `src/attention2d/detection/`:

- `matcher.py`: Hungarian matcher with class, L1 box, and GIoU costs.
- `losses.py`: set criterion with classification, L1 box, GIoU, cardinality, and optional mask auxiliary.
- `heads.py`: MLP box head, classification head, object query utilities.
- `anchor_region_detr.py`: small end-to-end model using existing 2D backbones and DETR-style query decoding.
- `scripts/train_det_toy.py`: small synthetic detection task for shape/gradient sanity.

Success criterion:

- Forward pass returns `pred_logits` and `pred_boxes`.
- Matcher/loss run on variable target counts.
- One dummy backward pass passes tests.

### Step 2: AnchorQueryInit

Turn the current spatial signal into a detector-specific mechanism:

```text
image feature map
  -> online anchor/region interaction
  -> region tokens / anchor summaries
  -> initialize or bias DETR object queries
  -> decoder predicts boxes/classes
```

Controls:

- random learned queries,
- FPN/local feature queries,
- `anchor_only_no_prefill` query initialization,
- region-pool query initialization.

Success criterion:

- Anchor-initialized queries improve convergence or detection quality over random queries on a synthetic/local detection task.

Status:

- Implemented `scripts/train_det_toy.py`.
- Initial smoke was biased because eval used GIoU as IoU, learned/anchor used different model initialization, and torch noise was not strictly paired.
- Fixed eval to use true IoU, shared initialization seed across learned/anchor, and deterministic torch noise for train/eval batches.
- Strict toy result: `results/det_toy_anchor_query_300step_5seeds_strict.csv`.
- Learned queries: final IoU `0.090`, best IoU `0.095`.
- Anchor query init: final IoU `0.110`, best IoU `0.123`.
- Paired delta vs learned: final IoU `+0.020`, best IoU `+0.028`, both `5/5` wins.
- Added formal feature/query 2x2 controls in `results/det_toy_feature_query_2x2_300step_5seeds.csv`:
  - `local_learned`: local-state features + learned queries, final/best IoU `0.079/0.088`.
  - `local_anchor`: local-state features + anchor queries, final/best IoU `0.099/0.131`.
  - `anchor_learned`: anchor-region features + learned queries, final/best IoU `0.090/0.095`.
  - `anchor_anchor`: anchor-region features + anchor queries, final/best IoU `0.110/0.123`.
- In the 2x2 control, content-conditioned anchor query generation over local-state features independently improves over `local_learned`: final IoU delta `+0.020` with `4/5` wins, best IoU delta `+0.042` with `5/5` wins.
- Anchor-region features alone also improve over `local_learned`, but more weakly: final IoU delta `+0.011` with `4/5` wins, best IoU delta `+0.007` with `4/5` wins.
- Combining anchor features and anchor queries gives the best final IoU, but not the best best-IoU; `local_anchor` has the highest best IoU.

Interpretation:

- AnchorQueryInit has a clean positive signal on the single-object square-detection toy task.
- The 2x2 control shows that the content-conditioned anchor query signal is not only an artifact of the anchor-region feature backbone.
- Added `local_anchor_detached` to separate anchor query content from the extra query-side gradient path.
- Under target-matched single-object eval, `results/det_toy_local_anchor_detached_single_300step_5seeds.csv` gives:
  - `local_learned`: final/best IoU `0.195/0.200`.
  - `local_anchor`: final/best IoU `0.241/0.250`, delta vs `local_learned` `+0.046/+0.050`, both `5/5` wins.
  - `local_anchor_detached`: final/best IoU `0.258/0.265`, delta vs `local_learned` `+0.063/+0.065`, final `4/5` wins and best `5/5` wins.
- Single-object interpretation: detached anchor queries preserve and even improve the signal, so simple single-object localization is helped mainly by the anchor query content/geometry summary, not by the extra query-side gradient path.
- This is still a limited sanity result: the toy task is single-object, square-only, and can be helped by row/column projections.
- Added non-overlapping `multi_distractor` toy support. Distractors are red, unlabeled squares; targets are green labeled squares; target-target and target-distractor overlap is controlled by rejection sampling.
- The contaminated overlapping multi-distractor run should not be used as formal evidence. Use `results/det_toy_multi_distractor_nonoverlap_2x2_300step_5seeds.csv`.
- Non-overlap multi-distractor 2x2 result:
  - `local_learned`: final/best IoU `0.176/0.181`.
  - `local_anchor`: final/best IoU `0.192/0.194`, delta vs `local_learned` `+0.016/+0.012`, both `4/5` wins.
  - `anchor_learned`: final/best IoU `0.194/0.198`, delta vs `local_learned` `+0.018/+0.016`, both `3/5` wins.
  - `anchor_anchor`: final/best IoU `0.200/0.200`, delta vs `local_learned` `+0.024/+0.018`, both `3/5` wins.
- Interpretation of the harder toy: anchor/region components still help over local learned queries, but the effect is weaker and less clean than the single-object toy. `anchor_anchor` is best on mean final IoU, but only marginally above `anchor_learned` and `local_anchor`.
- Detached query control on non-overlap multi-distractor, `results/det_toy_local_anchor_detached_multi_distractor_300step_5seeds.csv`:
  - `local_learned`: final/best IoU `0.176/0.181`.
  - `local_anchor`: final/best IoU `0.192/0.194`, delta vs `local_learned` `+0.016/+0.012`, both `4/5` wins.
  - `local_anchor_detached`: final/best IoU `0.163/0.166`, delta vs `local_learned` `-0.013/-0.016`, both `2/5` wins.
- Multi-distractor interpretation: unlike the single-object toy, detached query content alone is not enough and can hurt. The harder task appears to need differentiable online query-feature coupling or a stronger decoder/matcher.
- Current multi-object eval is target-matched IoU/recall-style localization. It does not penalize false-positive queries and should not be reported as AP.
- Added confidence-aware AP50-lite to `scripts/train_det_toy.py`. It ranks queries by object-class probability, matches predictions to targets at IoU `0.50`, and therefore penalizes false-positive ordering unlike target-matched IoU.
- AP-lite non-overlap multi-distractor result: `results/det_toy_aplite_multi_distractor_300step_5seeds.csv`.
  - `local_learned`: final/best IoU `0.176/0.181`, AP50-lite `0.039`.
  - `local_anchor`: final/best IoU `0.192/0.194`, AP50-lite `0.048`, AP delta vs `local_learned` `+0.010` with `4/5` wins.
  - `local_anchor_detached`: final/best IoU `0.163/0.166`, AP50-lite `0.055`, AP delta vs `local_learned` `+0.017` with `4/5` wins.
- AP-lite interpretation: `local_anchor` remains the better localization/coverage model, while `local_anchor_detached` ranks some true-positive queries better despite worse target-matched IoU. This separates "coverage quality" from "confidence ordering"; neither should be treated as detector AP yet.
- Next controls: stronger decoder/AP training and real DET dataset path.

### Step 3: DenseMaskAux

Keep the positive bbox-mask signal as an auxiliary loss:

- add a spatial mask head to detection backbones,
- train with `L_det + lambda * L_mask`,
- reuse bbox rectangle masks as weak dense supervision.

Success criterion:

- Mask auxiliary improves detector box AP or convergence without hurting classification/detection loss stability.

Status:

- Implemented explicit mask-aux variants in `scripts/train_det_toy.py`:
  - `local_learned_maskaux`
  - `local_anchor_maskaux`
  - `local_anchor_detached_maskaux`
- Each mask-aux variant adds a `1x1` dense mask head on `outputs["spatial_features"]`, trained with BCE + Dice against the union of target bbox rectangle masks.
- Formal result: `results/det_toy_densemaskaux_multi_distractor_300step_5seeds.csv`.
  - `local_anchor`: final/best IoU `0.192/0.194`, AP50-lite `0.048`.
  - `local_anchor_maskaux`: final/best IoU `0.189/0.189`, AP50-lite `0.046`, mask IoU/Dice `0.982/0.991`.
  - `local_learned`: final/best IoU `0.176/0.181`, AP50-lite `0.039`.
  - `local_learned_maskaux`: final/best IoU `0.170/0.172`, AP50-lite `0.054`, mask IoU/Dice `0.983/0.991`.
  - `local_anchor_detached`: final/best IoU `0.172/0.172`, AP50-lite `0.044`.
  - `local_anchor_detached_maskaux`: final/best IoU `0.177/0.192`, AP50-lite `0.041`, mask IoU/Dice `0.987/0.993`.
- Paired deltas:
  - `local_anchor_maskaux` vs `local_anchor`: final IoU `-0.003`, best IoU `-0.005`, AP50-lite `-0.002`.
  - `local_learned_maskaux` vs `local_learned`: final IoU `-0.006`, best IoU `-0.010`, AP50-lite `+0.015`.
  - `local_anchor_detached_maskaux` vs `local_anchor_detached`: final IoU `+0.005`, best IoU `+0.020`, AP50-lite `-0.002`.

Interpretation:

- DenseMaskAux is learnable: all mask-aux variants reach mask IoU around `0.98`.
- The auxiliary mask head does not reliably improve detection box IoU or AP-lite in the current tiny detector.
- The best current coverage model remains `local_anchor` without mask auxiliary.
- The useful signal is that dense supervision can train a spatial head, but transferring it into query boxes likely needs a stronger coupling mechanism than a side auxiliary loss.
- Implemented first coupling attempt: `maskpooled_query`.
  - `local_anchor_maskpooled_query` and `local_learned_maskpooled_query` predict an internal dense mask, soft-pool a foreground region summary from spatial features, then add that summary to queries before decoding.
  - Formal result: `results/det_toy_maskpooled_query_multi_distractor_300step_5seeds.csv`.
  - `local_anchor_maskpooled_query`: final/best IoU `0.172/0.179`, AP50-lite `0.034`, mask IoU/Dice `0.978/0.989`.
  - `local_learned_maskpooled_query`: final/best IoU `0.163/0.166`, AP50-lite `0.053`, mask IoU/Dice `0.989/0.995`.
  - Against `local_anchor`, `local_anchor_maskpooled_query` is worse by final IoU `-0.020`, best IoU `-0.015`, AP50-lite `-0.014`.
  - Against `local_learned`, `local_learned_maskpooled_query` is worse by final IoU `-0.012` and best IoU `-0.015`, but AP50-lite improves by `+0.015`.

Maskpooled interpretation:

- The internal mask can be learned, but a single global foreground summary added to all queries is too coarse for multi-object/distractor detection.
- It can shift score ordering for learned queries, but it hurts target coverage.
- The next coupling should be per-query, such as mask-biased query attention or mask proposal query initialization, not one shared pooled region vector.
- Implemented second coupling attempt: `mask_biased_attn`.
  - `local_anchor_mask_biased_attn` and `local_learned_mask_biased_attn` predict an internal dense mask and add `log(sigmoid(mask))` as a soft foreground bias to query-to-spatial attention logits.
  - Formal result: `results/det_toy_mask_biased_attn_multi_distractor_300step_5seeds.csv`.
  - `local_anchor_mask_biased_attn`: final/best IoU `0.189/0.189`, AP50-lite `0.052`, mask IoU/Dice `0.978/0.989`.
  - Against `local_anchor`, it is roughly tied on IoU (`-0.003` final, `-0.005` best; `3/5` final wins) and slightly better on AP50-lite (`+0.004`, `3/5` wins).
  - `local_learned_mask_biased_attn`: final/best IoU `0.222/0.227`, AP50-lite `0.093`, mask IoU/Dice `0.989/0.994`.
  - Against `local_learned`, it improves mean final IoU by `+0.046` and AP50-lite by `+0.054`, but wins only `2/5` seeds on IoU and is strongly pulled by one high-performing seed.

Mask-biased interpretation:

- Mask-biased attention is a better coupling than global mask pooling: it does not collapse anchor coverage and gives a small AP-lite signal for `local_anchor`.
- The large `local_learned_mask_biased_attn` mean is not yet robust because paired wins are weak; treat it as an instability/optimization signal, not a stable new mainline.
- Next step: diagnose and stabilize mask-biased attention with per-seed trajectories, bias-gate schedules, and proposal-style query initialization before making a stronger claim.
- Implemented gate-stability controls for mask-biased attention.
  - Formal result: `results/det_toy_mask_bias_gate_multi_distractor_300step_5seeds.csv`.
  - `local_anchor_mask_biased_attn_gate001`: final/best IoU `0.188/0.188`, AP50-lite `0.041`.
  - `local_anchor_mask_biased_attn_warmup`: final/best IoU `0.175/0.181`, AP50-lite `0.037`.
  - Against `local_anchor`, smaller gate and warmup do not improve the anchor-query branch.
  - `local_learned_mask_biased_attn_gate001`: final/best IoU `0.224/0.224`, AP50-lite `0.097`.
  - `local_learned_mask_biased_attn_warmup`: final/best IoU `0.232/0.234`, AP50-lite `0.103`.
  - Against `local_anchor`, learned-query gate controls improve mean IoU/AP but still win only `2/5` seeds on IoU, so the signal remains unstable.

Gate-stability interpretation:

- Smaller gate and linear warmup do not solve the core instability.
- For learned queries, mask bias sometimes opens a much better optimization path, but the improvement is seed-selective rather than reliable.
- Next step should move from global foreground bias to `mask_proposal_query_init`, where dense masks explicitly seed object queries instead of only nudging attention.
- Implemented `mask_proposal_query_init`.
  - `local_mask_proposal_query` and `anchor_mask_proposal_query` predict a dense mask, take top-k foreground cells, and gather the corresponding spatial feature tokens as object query seeds.
  - Formal result: `results/det_toy_mask_proposal_query_multi_distractor_300step_5seeds.csv`.
  - `local_mask_proposal_query`: final/best IoU `0.328/0.342`, recall50 `0.363`, AP50-lite `0.295`, mask IoU/Dice `0.986/0.993`.
  - `anchor_mask_proposal_query`: final/best IoU `0.321/0.324`, recall50 `0.341`, AP50-lite `0.272`, mask IoU/Dice `0.990/0.995`.
  - Against `local_anchor`, `local_mask_proposal_query` improves final IoU by `+0.136`, best IoU by `+0.149`, AP50-lite by `+0.247`, with `5/5` wins on all three metrics.
  - Against `local_anchor`, `anchor_mask_proposal_query` improves final IoU by `+0.129`, best IoU by `+0.130`, AP50-lite by `+0.224`, with `5/5` wins on all three metrics.

Mask-proposal interpretation:

- This is the first detection toy coupling that gives a large and stable positive signal.
- The result supports the narrower mechanism claim: dense foreground masks are useful when they explicitly seed object queries.
- It does not revive early prefill or generic memory-first claims; the supported path is mask/object-proposal-conditioned query initialization.
- Next step should stress-test proposal queries with top-k diversity, proposal NMS/connected components, object-size stratification, and eventually real DET annotations.
- Implemented top-k proposal diversity with spatial suppression.
  - `local_mask_proposal_nms_query` and `anchor_mask_proposal_nms_query` greedily select high-mask cells while suppressing nearby feature-grid cells with radius `2`.
  - Formal result: `results/det_toy_mask_proposal_nms_multi_distractor_300step_5seeds.csv`.
  - `local_mask_proposal_nms_query`: final/best IoU `0.461/0.463`, recall50 `0.455`, AP50-lite `0.378`, mask IoU/Dice `0.979/0.989`.
  - `anchor_mask_proposal_nms_query`: final/best IoU `0.438/0.465`, recall50 `0.394`, AP50-lite `0.313`, mask IoU/Dice `0.982/0.990`.
  - Against plain `local_mask_proposal_query`, `local_mask_proposal_nms_query` improves final IoU by `+0.133`, best IoU by `+0.120`, AP50-lite by `+0.083`, with `5/5` IoU wins.
  - Against plain `local_mask_proposal_query`, `anchor_mask_proposal_nms_query` improves final IoU by `+0.110`, best IoU by `+0.122`, with `5/5` IoU wins.

Proposal-diversity interpretation:

- Spatial suppression strongly improves query coverage, confirming that plain top-k over-selects nearby foreground peaks.
- Mask IoU is slightly lower than plain top-k, but detection IoU/AP are much better; proposal diversity matters more than perfect union-mask accuracy.
- Current strongest detection toy model is `local_mask_proposal_nms_query`.
- Next step should add stratified diagnostics and visual proposal overlays; if overlays are blurred or uninformative, swap to clearer curated samples rather than relying on weak figures.
- Added object-size and off-center stratified evaluation columns to detection toy CSVs.
  - Formal result: `results/det_toy_proposal_nms_stratified_multi_distractor_300step_5seeds.csv`.
  - `local_mask_proposal_nms_query` vs plain `local_mask_proposal_query`:
    - small IoU `0.363` vs `0.231`;
    - medium IoU `0.561` vs `0.393`;
    - large IoU `0.522` vs `0.443`;
    - center IoU `0.408` vs `0.278`;
    - off-center IoU `0.479` vs `0.344`.
  - `local_mask_proposal_nms_query` remains strongest across all reported strata.

Clear overlay sample-selection rule:

- Prefer images with object area not too tiny, visible non-overlapping objects, and enough object separation to make proposal diversity visible.
- Include at least one small, one medium, one large, one center, and one off-center example.
- Exclude samples whose objects are too small to see at display resolution, heavily overlapping, or visually ambiguous.
- If generated overlay panels are blurry, replace the sample set with clearer high-area / well-separated examples instead of using weak figures.
- Implemented `scripts/select_clear_det_overlay_samples.py` to select clearer qualitative samples directly from the local ILSVRC2013 DET val images and XML annotations.
- The selector scores samples by largest-box area, pairwise box overlap, object separation, border margin, image resolution, object count, and a lightweight image sharpness measure.
- Default balanced output:
  - command: `/opt/anaconda3/envs/AIAA/bin/python scripts/select_clear_det_overlay_samples.py --per-group 3 --extra-fill 6 --max-per-label 3`
  - manifest: `results/clear_det_overlay_samples/manifest.csv`
  - overlays: `results/clear_det_overlay_samples/overlays/`
  - contact sheet: `results/clear_det_overlay_samples/contact_sheet.jpg`
- Figure-ready high-area output:
  - command: `/opt/anaconda3/envs/AIAA/bin/python scripts/select_clear_det_overlay_samples.py --per-group 2 --extra-fill 10 --min-area 0.10 --max-overlap 0.35 --min-margin 0.03 --max-per-label 3 --out-dir results/clear_det_overlay_samples_high_area`
  - manifest: `results/clear_det_overlay_samples_high_area/manifest.csv`
  - overlays: `results/clear_det_overlay_samples_high_area/overlays/`
  - contact sheet: `results/clear_det_overlay_samples_high_area/contact_sheet.jpg`
- Use the high-area sheet first for paper-style figures; use the balanced sheet when the figure needs explicit small/medium/large and center/off-center coverage.
- Fixed high-area prediction overlay:
  - selected the first 8 image ids from `results/clear_det_overlay_samples_high_area/manifest.csv`, covering medium/large and center/off-center examples.
  - command: `/opt/anaconda3/envs/AIAA/bin/python -u scripts/visualize_bbox_mask_predictions.py --models anchor_only_no_prefill no_prefill_local_mix fpn_sum_lite --steps 1000 --mask-size 32 --display-size 192 --image-id-manifest results/clear_det_overlay_samples_high_area/manifest.csv --manifest-top-k 8 --out-dir results/fixed_high_area_bbox_mask_overlays_32x32_seed41`
  - cross-model grid: `results/fixed_high_area_bbox_mask_overlays_32x32_seed41/comparison_grid.png`
  - per-model panels: `results/fixed_high_area_bbox_mask_overlays_32x32_seed41/{anchor_only_no_prefill,no_prefill_local_mix,fpn_sum_lite}/`

### Step 4: Real Detection Dataset Path

Use existing DET annotations before COCO:

- parse `ILSVRC2013_DET_bbox_val` into detection samples,
- support image, boxes, labels, image size,
- add a small train/eval split for smoke,
- report AP50, AP50:95-style approximation, recall, and speed.

Success criterion:

- A tiny detector overfits a small DET subset.
- A 197-class DET subset can train without target-shape or matching issues.

Status:

- Implemented `scripts/train_det_real.py`.
- The script builds one sample per image, not one sample per object row, avoiding object-level train/eval leakage.
- It uses a fixed synset-to-class-id label map from the selected top classes and writes it to disk.
- v1 transform is explicit square resize/stretch: XML boxes are normalized by original image width/height, which is correct for direct `Resize((S,S))`; random crop/flip and letterbox are not enabled.
- Targets are truncated to the largest `--max-objects` boxes per image so the current brute-force Hungarian matcher remains valid with `max_objects <= num_queries`.
- Metrics:
  - class-agnostic target-matched IoU and recall50,
  - objectness AP50-lite,
  - class-aware AP50-lite,
  - mask IoU/Dice when a query-mask branch exists,
  - small/medium/large and center/off-center IoU strata.
- Reproducibility artifacts:
  - label map CSV,
  - image-level train/eval split CSV,
  - metrics CSV.

MPS smoke:

- Command: `/opt/anaconda3/envs/AIAA/bin/python -u scripts/train_det_real.py --models local_learned local_anchor local_mask_proposal_nms_query --reference-model local_learned --top-classes 10 --max-samples 200 --max-objects 3 --num-queries 6 --steps 50 --eval-every 50 --eval-batches 4 --batch-size 16 --out results/det_real_mini_50step_2seed.csv --label-map-out results/det_real_mini_50step_label_map.csv --split-out results/det_real_mini_50step_split.csv --seeds 2`
- Output CSV: `results/det_real_mini_50step_2seed.csv`.
- Label map: `results/det_real_mini_50step_label_map.csv`.
- Split manifest: `results/det_real_mini_50step_split.csv`.
- Device: MPS.
- Result summary:
  - `local_anchor`: IoU `0.361`, recall50 `0.356`, objectness AP50-lite `0.296`, class-aware AP50-lite `0.082`.
  - `local_learned`: IoU `0.314`, recall50 `0.229`, objectness AP50-lite `0.257`, class-aware AP50-lite `0.077`.
  - `local_mask_proposal_nms_query`: IoU `0.306`, recall50 `0.241`, objectness AP50-lite `0.255`, class-aware AP50-lite `0.097`, mask IoU/Dice `0.400/0.537`.
- Paired vs `local_learned`:
  - `local_anchor`: IoU delta `+0.047`, wins `2/2`; class-aware AP50-lite delta `+0.006`, wins `2/2`.
  - `local_mask_proposal_nms_query`: IoU delta `-0.007`, wins `1/2`; class-aware AP50-lite delta `+0.020`, wins `2/2`.

Interpretation:

- The real DET path is now executable end-to-end on real XML boxes.
- On this small 50-step top-10 smoke, `local_anchor` is the strongest real-box coverage/IoU model.
- The toy-detection winner `local_mask_proposal_nms_query` does not yet transfer to real-box IoU, though it learns a dense mask and gives a small class-aware AP-lite signal.
- This result supports continuing real-box validation, but it is not RF-DETR-level evidence and should not be reported as detector mAP.
- Follow-up 300-step real DET mini run:
  - command: `/opt/anaconda3/envs/AIAA/bin/python -u scripts/train_det_real.py --models local_learned local_anchor local_anchor_detached local_mask_proposal_nms_query --reference-model local_learned --top-classes 10 --max-samples 200 --max-objects 3 --num-queries 6 --steps 300 --eval-every 100 --eval-batches 4 --batch-size 16 --out results/det_real_mini_300step_2seed.csv --label-map-out results/det_real_mini_300step_label_map.csv --split-out results/det_real_mini_300step_split.csv --seeds 2`
  - `local_learned`: final/best IoU `0.359/0.367`, recall50 `0.321`, objectness AP50-lite `0.310`, class-aware AP50-lite `0.108`.
  - `local_anchor`: final/best IoU `0.359/0.367`, recall50 `0.312`, objectness AP50-lite `0.236`, class-aware AP50-lite `0.068`.
  - `local_anchor_detached`: final/best IoU `0.343/0.353`, recall50 `0.297`, objectness AP50-lite `0.214`, class-aware AP50-lite `0.049`.
  - `local_mask_proposal_nms_query`: final/best IoU `0.357/0.366`, recall50 `0.302`, objectness AP50-lite `0.266`, class-aware AP50-lite `0.093`, mask IoU/Dice `0.458/0.604`.
- 300-step interpretation:
  - The 50-step `local_anchor` advantage does not remain clearly above `local_learned` after longer training; both are essentially tied on matched IoU.
  - `local_anchor_detached` is consistently weaker than `local_anchor`, so detached anchor content alone is not sufficient on real boxes. Any real-box anchor benefit likely depends on differentiable online query-feature coupling or optimization path, not just static anchor summaries.
  - `local_mask_proposal_nms_query` learns a useful dense mask but still does not produce a real-box IoU advantage. Toy proposal-NMS gains do not transfer automatically.
- Added real DET ranking/classification diagnostics to separate localization from confidence ordering:
  - command: `/opt/anaconda3/envs/AIAA/bin/python -u scripts/train_det_real.py --models local_learned local_anchor local_anchor_detached local_mask_proposal_nms_query --reference-model local_learned --top-classes 10 --max-samples 200 --max-objects 3 --num-queries 6 --steps 300 --eval-every 100 --eval-batches 4 --batch-size 16 --out results/det_real_mini_300step_2seed_diagnostics.csv --label-map-out results/det_real_mini_300step_diagnostics_label_map.csv --split-out results/det_real_mini_300step_diagnostics_split.csv --seeds 2`
  - `local_learned`: matched-assignment class acc `0.390`, score-IoU corr `0.438`, pre-NMS objectness AUC `0.658`, top-k FP rate `0.719`, query assignment entropy `0.904`.
  - `local_anchor`: matched-assignment class acc `0.390`, score-IoU corr `0.305`, pre-NMS objectness AUC `0.584`, top-k FP rate `0.819`, query assignment entropy `0.926`.
  - `local_anchor_detached`: matched-assignment class acc `0.347`, score-IoU corr `0.194`, pre-NMS objectness AUC `0.582`, top-k FP rate `0.848`, query assignment entropy `0.897`.
  - `local_mask_proposal_nms_query`: matched-assignment class acc `0.401`, score-IoU corr `0.380`, pre-NMS objectness AUC `0.598`, top-k FP rate `0.767`, query assignment entropy `0.994`, mask IoU/Dice `0.458/0.604`.
- Ranking-diagnostic interpretation:
  - In this mini setup, `local_learned` and `local_anchor` are tied on matched IoU, but `local_learned` has much better score-IoU correlation, pre-NMS objectness AUC, and top-k false-positive rate. This suggests the current anchor failure is mainly a ranking/objectness/classification problem, not pure box coverage.
  - `local_anchor_detached` is weaker on matched-assignment class accuracy, score-IoU correlation, and top-k false positives, reinforcing that static detached anchor summaries are insufficient for real-box detection.
  - `local_mask_proposal_nms_query` has the highest matched-assignment class accuracy and query assignment entropy, but its objectness/ranking quality still trails `local_learned`. Dense masks are learned, but proposal queries still need better score calibration and query-box refinement.
- Next controls:
  - add score-IoU/ranking loss or class/objectness calibration controls,
  - test soft/gated anchor residual query initialization instead of replacing learned queries,
  - add oracle and detached mask-proposal controls to measure whether proposal quality or query consumption is the bottleneck,
  - add proposal/box overlays on fixed clear samples,
  - add a stronger decoder or iterative box refinement,
  - improve class/objectness training before scaling beyond this subset,
  - only then revisit longer/larger DET or RF-DETR distillation.
- P0 follow-up: oracle proposal and gated residual anchor controls:
  - command: `/opt/anaconda3/envs/AIAA/bin/python -u scripts/train_det_real.py --models local_learned local_anchor local_anchor_residual_query local_anchor_residual_detached_query local_mask_proposal_nms_query local_mask_proposal_oracle_nms_query --reference-model local_learned --top-classes 10 --max-samples 200 --max-objects 3 --num-queries 6 --steps 300 --eval-every 100 --eval-batches 4 --batch-size 16 --out results/det_real_p0_oracle_anchor_residual_300step_2seed.csv --label-map-out results/det_real_p0_oracle_anchor_residual_label_map.csv --split-out results/det_real_p0_oracle_anchor_residual_split.csv --seeds 2`
  - `local_learned`: final/best IoU `0.359/0.367`, recall50 `0.321`, objectness AP50-lite `0.310`, class-aware AP50-lite `0.108`.
  - `local_anchor`: final/best IoU `0.359/0.367`, recall50 `0.312`, objectness AP50-lite `0.236`, class-aware AP50-lite `0.068`.
  - `local_anchor_residual_query`: final/best IoU `0.376/0.376`, recall50 `0.371`, objectness AP50-lite `0.266`, class-aware AP50-lite `0.135`.
  - `local_anchor_residual_detached_query`: final/best IoU `0.367/0.378`, recall50 `0.320`, objectness AP50-lite `0.256`, class-aware AP50-lite `0.076`.
  - `local_mask_proposal_nms_query`: final/best IoU `0.357/0.366`, recall50 `0.302`, objectness AP50-lite `0.266`, class-aware AP50-lite `0.093`, mask IoU/Dice `0.458/0.604`.
  - `local_mask_proposal_oracle_nms_query`: final/best IoU `0.399/0.421`, recall50 `0.340`, objectness AP50-lite `0.259`, class-aware AP50-lite `0.138`, oracle mask IoU/Dice `1.000/1.000`.
- P0 interpretation:
  - In this mini DET setting, oracle NMS proposal queries improve box coverage substantially over `local_learned`: final IoU `+0.040`, best IoU `+0.054`, best-IoU wins `2/2`. This indicates foreground-union proposal quality/extraction is a real coverage bottleneck for the predicted-mask proposal branch.
  - Oracle proposal still does not improve objectness AP50-lite, and its score-IoU correlation/objectness AUC remain weak. Therefore query scoring/calibration is a second bottleneck even when proposal locations are idealized.
  - `local_anchor_residual_query` is stronger than replacement-style `local_anchor` on final IoU, recall50, and class-aware AP50-lite. Soft residual anchor bias is a better real-DET direction than replacing learned queries with anchor queries.
  - `local_anchor_residual_query` also beats the detached residual version on final IoU and class-aware AP, so differentiable anchor-feature coupling still matters.
- Updated next controls:
  - prioritize `local_anchor_residual_query` as the real-DET anchor path, not `local_anchor`;
  - add score-IoU/objectness calibration loss for residual-anchor and oracle-proposal variants;
  - improve predicted proposal quality toward oracle proposal, then retest whether AP follows;
  - add oracle proposal with learned scoring/box refinement decoupled from oracle query locations.
- Score-IoU calibration follow-up:
  - implemented `score_iou_calibration_loss`, which trains object-vs-background query logits to regress each query's detached max IoU to any target.
  - Added calibrated real DET variants:
    - `local_learned_calib`
    - `local_anchor_calib`
    - `local_anchor_residual_query_calib`
  - Main command, weight `0.5`: `/opt/anaconda3/envs/AIAA/bin/python -u scripts/train_det_real.py --models local_learned local_learned_calib local_anchor_residual_query local_anchor_residual_query_calib --reference-model local_anchor_residual_query --top-classes 10 --max-samples 200 --max-objects 3 --num-queries 6 --steps 300 --eval-every 100 --eval-batches 4 --batch-size 16 --score-iou-weight 0.5 --out results/det_real_residual_anchor_score_iou_calib_300step_2seed.csv --label-map-out results/det_real_residual_anchor_score_iou_calib_label_map.csv --split-out results/det_real_residual_anchor_score_iou_calib_split.csv --seeds 2`
  - Weight `0.5` result:
    - `local_learned`: final/best IoU `0.359/0.367`, objectness AP50-lite `0.310`, class-aware AP50-lite `0.108`.
    - `local_learned_calib`: final/best IoU `0.376/0.376`, objectness AP50-lite `0.355`, class-aware AP50-lite `0.144`.
    - `local_anchor_residual_query`: final/best IoU `0.376/0.376`, objectness AP50-lite `0.266`, class-aware AP50-lite `0.135`.
    - `local_anchor_residual_query_calib`: final/best IoU `0.361/0.372`, objectness AP50-lite `0.257`, class-aware AP50-lite `0.102`.
  - Smaller weight command, weight `0.1`: `/opt/anaconda3/envs/AIAA/bin/python -u scripts/train_det_real.py --models local_anchor_residual_query local_anchor_residual_query_calib local_learned_calib --reference-model local_anchor_residual_query --top-classes 10 --max-samples 200 --max-objects 3 --num-queries 6 --steps 300 --eval-every 100 --eval-batches 4 --batch-size 16 --score-iou-weight 0.1 --out results/det_real_residual_anchor_score_iou_calib_w01_300step_2seed.csv --label-map-out results/det_real_residual_anchor_score_iou_calib_w01_label_map.csv --split-out results/det_real_residual_anchor_score_iou_calib_w01_split.csv --seeds 2`
  - Weight `0.1` result:
    - `local_anchor_residual_query`: final/best IoU `0.376/0.376`, objectness AP50-lite `0.266`, class-aware AP50-lite `0.135`.
    - `local_anchor_residual_query_calib`: final/best IoU `0.373/0.373`, objectness AP50-lite `0.250`, class-aware AP50-lite `0.099`.
    - `local_learned_calib`: final/best IoU `0.373/0.382`, objectness AP50-lite `0.262`, class-aware AP50-lite `0.082`.
- Calibration interpretation:
  - Naive class-agnostic max-IoU BCE calibration works for learned queries at weight `0.5`: it improves final IoU, objectness AP50-lite, class-aware AP50-lite, objectness AUC, and top-k FP rate over `local_learned`. It does not improve score-IoU correlation in this run.
  - The same calibration does not work for residual-anchor: at weight `0.5` it hurts final IoU, objectness AP, and class-aware AP; at weight `0.1` it improves score-IoU correlation/objectness AUC but still hurts AP and class-aware AP.
  - This suggests the calibration objective conflicts with residual-anchor query assignment. The likely issue is that max-IoU targets treat duplicate/high-overlap queries as soft positives, while DETR AP and matching need one confident query per object.
  - Next calibration should be matcher-aware quality classification: matched queries get target IoU as class/objectness quality, while unmatched foreground class logits are pushed toward zero and no-object handling remains mainly from standard DETR CE. Do not continue tuning naive all-query max-IoU BCE as the main path.
- Matcher-aware quality classification follow-up:
  - Implemented `matcher_aware_quality_classification_loss`: run Hungarian matching, assign only each matched query's target class a detached IoU-quality soft label, and keep all unmatched foreground class logits at zero.
  - Added variants:
    - `local_learned_matchqual`
    - `local_anchor_matchqual`
    - `local_anchor_residual_query_matchqual`
  - Weight `1.0` command: `/opt/anaconda3/envs/AIAA/bin/python -u scripts/train_det_real.py --models local_learned local_learned_matchqual local_anchor_residual_query local_anchor_residual_query_matchqual --reference-model local_anchor_residual_query --top-classes 10 --max-samples 200 --max-objects 3 --num-queries 6 --steps 300 --eval-every 100 --eval-batches 4 --batch-size 16 --quality-cls-weight 1.0 --out results/det_real_matcher_quality_cls_300step_2seed.csv --label-map-out results/det_real_matcher_quality_cls_label_map.csv --split-out results/det_real_matcher_quality_cls_split.csv --seeds 2`
  - Weight `1.0` result:
    - `local_anchor_residual_query`: final/best IoU `0.376/0.376`, objectness AP50-lite `0.266`, class-aware AP50-lite `0.135`.
    - `local_anchor_residual_query_matchqual`: final/best IoU `0.366/0.392`, objectness AP50-lite `0.281`, class-aware AP50-lite `0.092`.
    - `local_learned`: final/best IoU `0.359/0.367`, objectness AP50-lite `0.310`, class-aware AP50-lite `0.108`.
    - `local_learned_matchqual`: final/best IoU `0.360/0.360`, objectness AP50-lite `0.239`, class-aware AP50-lite `0.080`.
  - Weight `0.25` command: `/opt/anaconda3/envs/AIAA/bin/python -u scripts/train_det_real.py --models local_anchor_residual_query local_anchor_residual_query_matchqual local_learned --reference-model local_anchor_residual_query --top-classes 10 --max-samples 200 --max-objects 3 --num-queries 6 --steps 300 --eval-every 100 --eval-batches 4 --batch-size 16 --quality-cls-weight 0.25 --out results/det_real_matcher_quality_cls_w025_300step_2seed.csv --label-map-out results/det_real_matcher_quality_cls_w025_label_map.csv --split-out results/det_real_matcher_quality_cls_w025_split.csv --seeds 2`
  - Weight `0.25` result:
    - `local_anchor_residual_query`: final/best IoU `0.376/0.376`, objectness AP50-lite `0.266`, class-aware AP50-lite `0.135`.
    - `local_anchor_residual_query_matchqual`: final/best IoU `0.370/0.371`, objectness AP50-lite `0.258`, class-aware AP50-lite `0.133`.
- Matcher-aware interpretation:
  - Matcher-aware quality classification avoids the duplicate-positive problem of all-query max-IoU BCE, but the current BCE-on-foreground-class-logits implementation still does not improve residual-anchor final AP.
  - At weight `1.0`, it improves residual-anchor best IoU and objectness AP50-lite, but lowers final IoU and class-aware AP50-lite. At weight `0.25`, the effect is smaller but still not positive.
  - For learned queries, matcher-aware quality classification is worse than both the base learned model and the naive max-IoU calibration.
  - The next scoring path should not be another weight sweep. It should separate semantic class prediction from localization-quality prediction.
- Separate quality/objectness head follow-up:
  - Implemented `pred_quality_logits` in `DetectionHead`, trained only for `*_quality_head` variants.
  - The quality head uses Hungarian matches: matched queries regress detached matched IoU; unmatched queries target `0`.
  - Evaluation now reports quality-aware AP with `score = class_prob * quality^alpha` for `alpha = 0.5, 1.0, 2.0`, plus quality-IoU correlation and quality AUC.
  - Command: `/opt/anaconda3/envs/AIAA/bin/python -u scripts/train_det_real.py --models local_learned local_learned_quality_head local_anchor_residual_query local_anchor_residual_query_quality_head --reference-model local_anchor_residual_query --top-classes 10 --max-samples 200 --max-objects 3 --num-queries 6 --steps 300 --eval-every 100 --eval-batches 4 --batch-size 16 --quality-head-weight 1.0 --out results/det_real_quality_head_300step_2seed.csv --label-map-out results/det_real_quality_head_label_map.csv --split-out results/det_real_quality_head_split.csv --seeds 2`
  - `local_anchor_residual_query`: final/best IoU `0.376/0.376`, AP50-lite `0.266`, class-aware AP50-lite `0.135`; quality-aware AP fields are intentionally equal to base AP because this model does not train the quality head.
  - `local_anchor_residual_query_quality_head`: final/best IoU `0.359/0.367`, AP50-lite `0.262`, class-aware AP50-lite `0.134`, quality-aware AP50-lite at alpha=1 `0.330`, quality-aware class AP50-lite at alpha=1 `0.175`.
  - `local_learned`: final/best IoU `0.359/0.367`, AP50-lite `0.310`, class-aware AP50-lite `0.108`.
  - `local_learned_quality_head`: final/best IoU `0.374/0.374`, AP50-lite `0.263`, class-aware AP50-lite `0.082`, quality-aware AP50-lite at alpha=1 `0.306`.
- Quality-head interpretation:
  - The separate head supports the ranking-side diagnosis: for residual-anchor, the learned quality score improves ranking when used at inference (`ap50_q1 +0.064` vs the residual-anchor base AP, `2/2` wins).
  - It also improves residual-anchor class-aware quality AP (`ap50_class_q1 +0.040` vs the residual-anchor base class-aware AP), but the gain is not paired-clean on both seeds.
  - The quality head does not improve base AP without quality scoring, and it lowers residual-anchor final IoU. This means query-quality scoring helps ranking, but the current auxiliary loss still perturbs localization/semantic learning.
  - For learned queries, the quality head does not beat the simpler `local_learned` AP reference.
  - Next scoring step should keep the separate-head design but reduce interference: try lower quality-head weights, stop-gradient or late-start quality training, and/or use the quality score only for inference ranking after a base detector is trained.
- Low-interference quality-head follow-up:
  - Added `--quality-head-start-step`, `--quality-head-warmup-steps`, and `--quality-head-only-after-start`.
  - Lower weight `0.5`, command output `results/det_real_quality_head_w05_300step_2seed.csv`:
    - `local_anchor_residual_query_quality_head`: final/best IoU `0.362/0.367`, AP50-lite `0.266`, class-aware AP50-lite `0.090`, quality-aware AP50-lite at alpha=1 `0.335`.
    - Interpretation: qAP signal remains, but residual-anchor IoU is still below base and class-aware scoring is worse.
  - Lower weight `0.25`, command output `results/det_real_quality_head_w025_300step_2seed.csv`:
    - `local_anchor_residual_query_quality_head`: final/best IoU `0.347/0.361`, AP50-lite `0.232`, class-aware AP50-lite `0.104`, quality-aware AP50-lite at alpha=1 `0.278`.
    - Interpretation: too weak/unstable; it loses both IoU and most qAP benefit.
  - Late-start + warmup, command output `results/det_real_quality_head_late151_warm50_300step_2seed.csv`:
    - setting: `--quality-head-start-step 151 --quality-head-warmup-steps 50 --quality-head-weight 1.0`.
    - `local_anchor_residual_query_quality_head`: final/best IoU `0.357/0.374`, AP50-lite `0.274`, class-aware AP50-lite `0.127`, quality-aware AP50-lite at alpha=1 `0.305`.
    - Interpretation: less disruptive than always-on `weight=1.0`, but still does not preserve final IoU.
  - Two-stage quality-only, command output `results/det_real_quality_head_twostage_start301_400step_2seed.csv`:
    - setting: `--steps 400 --quality-head-start-step 301 --quality-head-only-after-start`; first 300 steps train detector normally, final 100 steps freeze detector and train only `head.quality_head`.
    - `local_anchor_residual_query_quality_head`: final/best IoU `0.376/0.376`, AP50-lite `0.266`, class-aware AP50-lite `0.135`, quality-aware AP50-lite at alpha=1 `0.326`, quality-aware class AP50-lite at alpha=1 `0.167`.
    - Interpretation: this is the best current quality-head path. It preserves residual-anchor localization and base AP while improving ranking when quality is used for inference scoring.
- Updated scoring decision:
  - Do not keep tuning quality targets inside class logits.
  - Do not use always-on quality-head auxiliary as the main path because it perturbs detector training.
  - Promote the two-stage quality head as the current scoring/ranking direction: train the detector first, then freeze it and learn query quality for score re-ranking.
  - Next useful controls: train quality-only for longer after start, sweep alpha at eval, and test whether the same two-stage quality head improves oracle-proposal and mask-proposal variants.
- Oracle scoring and correlation diagnostics:
  - Added eval-only oracle scoring: `score = class_prob * true_max_iou_to_any_gt`.
  - Added `combined_iou_corr` / `combined_auc` for `class_prob * predicted_quality`.
  - Re-ran two-stage quality head with these diagnostics:
    - command output: `results/det_real_quality_head_twostage_oracle_start301_400step_2seed.csv`.
    - `local_anchor_residual_query_quality_head`: final/best IoU `0.376/0.376`, AP50-lite `0.266`, AP50 q^0.5/q^1/q^2 `0.329/0.326/0.341`, oracle-IoU AP50 `0.421`.
    - class-aware AP50 base/q^0.5/q^1/q^2/oracle `0.135/0.179/0.167/0.185/0.227`.
    - correlation: `score_iou_corr 0.339`, `quality_iou_corr 0.611`, `combined_iou_corr 0.584`.
  - Interpretation:
    - The ranking bottleneck is real: oracle-IoU scoring raises AP50 from `0.266` to `0.421`.
    - The two-stage quality head closes part of this gap, reaching `0.326-0.341` depending on alpha.
    - Quality score itself tracks IoU better than class score (`0.611` vs `0.339`), and `class_prob * quality` remains much better aligned than class score alone (`0.584` vs `0.339`).
    - There is still headroom versus oracle scoring, so next work should improve quality-head calibration rather than reintroduce quality loss into detector training.
- Quality alpha sweep and oracle-gap accounting:
  - Expanded eval scoring to `q^0.25`, `q^0.5`, `q^1`, `q^2`, and `q^4`.
  - Added automatic post-hoc `best-q` alpha, IoU-reference gap, gap-to-reference, and raw gap-closure ratio fields for both class-agnostic and class-aware AP.
  - Re-ran the same two-stage quality setup:
    - command output: `results/det_real_quality_head_twostage_alphas_oracle_start301_400step_2seed.csv`.
    - `local_anchor_residual_query_quality_head`: AP50/base `0.266`, post-hoc best-q AP50 `0.345`, mean selected alpha `1.25`, IoU-reference AP50 `0.421`.
    - residual-anchor closure: post-hoc best-q closes `0.491` of the class-agnostic IoU-reference ranking gap; class-aware best-q closes `0.584` of the class-aware IoU-reference gap.
    - `local_learned_quality_head`: AP50/base `0.310`, best-q AP50 `0.328`, oracle-IoU AP50 `0.363`, closure `0.240`.
  - Interpretation:
    - Residual-anchor remains the better candidate for quality reranking even though its base AP is lower than learned queries.
    - The quality head can recover roughly half of the residual-anchor oracle ranking gap without changing boxes.
    - Learned queries have less oracle headroom and lower closure, so quality calibration is less impactful there.
    - The `best-q` value is a post-hoc diagnostic selected on the eval set; formal comparisons should still report fixed alphas such as `q^1` or a pre-registered alpha.
    - IoU-reference scoring is diagnostic rather than a strict detector upper bound. Class-aware IoU-reference scoring is especially not a strict upper bound, because true-IoU weighting does not fix class-wrong predictions; use class-agnostic closure as the cleaner ranking-gap measure.
  - Next useful scoring controls:
    - train the frozen quality head longer after detector freeze;
    - try calibration regularizers or temperature scaling for the quality head;
    - test two-stage quality ranking on oracle-proposal and mask-proposal real-DET variants.
- Longer frozen quality-head control:
  - Tested a longer two-stage schedule: first 300 steps train the detector, then 300 additional steps train only the frozen-detector quality head.
  - command output: `results/det_real_quality_head_twostage_alphas_oracle_start301_600step_2seed.csv`.
  - `local_anchor_residual_query_quality_head`: AP50/base `0.266`, post-hoc best-q AP50 `0.344`, mean selected alpha `3.00`, IoU-reference AP50 `0.421`, raw gap closure `0.492`.
  - `local_learned_quality_head`: AP50/base `0.310`, post-hoc best-q AP50 `0.328`, IoU-reference AP50 `0.363`, raw gap closure `0.240`.
  - Interpretation:
    - Extending quality-only training from 100 to 300 steps after freeze does not materially improve residual-anchor ranking (`0.345 -> 0.344` best-q AP50; `0.491 -> 0.492` closure).
    - The quality head is likely capacity/target-limited rather than under-trained in this mini setting.
    - Do not spend more runs simply lengthening the frozen quality stage; next scoring work should change calibration or apply the same two-stage ranking head to proposal-query variants.
- Proposal-query two-stage quality-head control:
  - Added real-DET quality-head variants:
    - `local_mask_proposal_nms_query_quality_head`
    - `local_mask_proposal_oracle_nms_query_quality_head`
  - command output: `results/det_real_proposal_quality_head_twostage_start301_400step_2seed.csv`.
  - Predicted proposal NMS:
    - base `local_mask_proposal_nms_query`: final/best IoU `0.348/0.366`, AP50 `0.248`, IoU-reference AP50 `0.329`.
    - quality `local_mask_proposal_nms_query_quality_head`: final/best IoU `0.357/0.366`, AP50 `0.266`, post-hoc best-q AP50 `0.291`, IoU-reference AP50 `0.364`.
  - Oracle proposal NMS:
    - base `local_mask_proposal_oracle_nms_query`: final/best IoU `0.383/0.421`, AP50 `0.250`, class-aware AP50 `0.108`.
    - quality `local_mask_proposal_oracle_nms_query_quality_head`: final/best IoU `0.399/0.421`, AP50 `0.259`, class-aware AP50 `0.138`, post-hoc best-q AP50 `0.325`, IoU-reference AP50 `0.372`.
  - Interpretation:
    - Two-stage quality ranking transfers to proposal-query variants.
    - For predicted proposal NMS, quality head gives a modest base AP improvement (`0.248 -> 0.266`) and a larger post-hoc quality-ranked AP (`0.291`).
    - For oracle proposal NMS, quality ranking is stronger: best-q AP reaches `0.325`, and class-aware AP improves from `0.108` to `0.138`.
    - The strongest current real-mini direction is now proposal query + two-stage quality ranking, but the best-q result remains post-hoc diagnostic. Fixed-alpha or calibrated scoring is needed before treating it as a finalized inference recipe.
- Fixed-alpha / calibrated scoring follow-up:
  - Added CLI fields:
    - `--fixed-quality-alpha`, default `2.0`;
    - `--quality-score-temperature`, default `1.0`.
  - CSV/summary now report fixed quality AP, fixed alpha, temperature, fixed gap-to-IoU-reference, and fixed raw closure separately from post-hoc `best-q`.
  - Temperature-calibrated control:
    - command output: `results/det_real_proposal_quality_head_fixed_alpha2_temp2_400step_2seed.csv`.
    - setting: fixed `alpha=2.0`, temperature `2.0`.
    - `local_mask_proposal_nms_query_quality_head`: base AP50 `0.266`, fixed AP50 `0.283`, post-hoc best-q AP50 `0.291`, IoU-reference AP50 `0.364`.
    - `local_mask_proposal_oracle_nms_query_quality_head`: base AP50 `0.259`, fixed AP50 `0.312`, post-hoc best-q AP50 `0.325`, IoU-reference AP50 `0.372`.
  - Interpretation:
    - Fixed-alpha scoring is now separated from post-hoc best-q and should be used for formal comparisons.
    - Temperature `2.0` does not improve over the untemperatured `q^2` score for oracle proposals (`0.312` fixed-temp vs `0.325` q^2/best-q).
    - For predicted proposals, temp-scaled fixed score is useful but not better than the best fixed alpha already in the sweep (`q^1`/best-q).
    - Current recommendation: report fixed `q^2` with temperature `1.0` for proposal-quality ranking unless a calibration split, not eval post-hoc selection, justifies another setting.
- Official fixed `q^2,temp=1.0` real-mini summary:
  - Generated `results/det_real_official_fixed_q2_summary.csv` from the current key result files.
  - Summary:
    - `local_learned`: final/best IoU `0.359/0.367`, base AP50 `0.310`, class AP50 `0.108`.
    - `local_anchor_residual_query`: final/best IoU `0.376/0.376`, base AP50 `0.266`, class AP50 `0.135`.
    - `local_anchor_residual_query_quality_head`: final/best IoU `0.376/0.376`, base AP50 `0.266`, fixed-q2 AP50 `0.341`, class fixed-q2 AP50 `0.185`, IoU-reference AP50 `0.421`.
    - `local_mask_proposal_nms_query_quality_head`: final/best IoU `0.357/0.366`, base AP50 `0.266`, fixed-q2 AP50 `0.282`, IoU-reference AP50 `0.364`.
    - `local_mask_proposal_oracle_nms_query_quality_head`: final/best IoU `0.399/0.421`, base AP50 `0.259`, fixed-q2 AP50 `0.325`, IoU-reference AP50 `0.372`.
  - Interpretation:
    - Official fixed scoring keeps `local_anchor_residual_query_quality_head` as the current best AP50 route (`0.341`).
    - Oracle proposal quality gives the strongest localization coverage but still trails residual-anchor quality in fixed-q2 AP (`0.325` vs `0.341`).
    - Proposal/query improvement and quality ranking are both useful, but the current best real-mini recipe is still residual-anchor detector + two-stage fixed-q2 quality ranking.
- Held-out calibration split:
  - Added CLI fields:
    - `--calibration-frac`;
    - `--calibration-batches`.
  - When enabled, the original held-out split is divided into calibration and final eval subsets. The quality model chooses alpha on calibration via `ap50_q_best_alpha`; final eval reports that alpha through `eval_ap50_q_fixed`.
  - Smoke artifact: `results/det_real_quality_calib_split_smoke.csv`.
  - Standard artifact: `results/det_real_quality_calib_split_400step_2seed.csv`.
  - Robustness artifact: `results/det_real_quality_calib_split_400step_3seed.csv`.
  - AP75 artifact: `results/det_real_quality_calib_split_ap75_400step_3seed.csv`.
  - ECE artifact: `results/det_real_quality_calib_split_ece_400step_3seed.csv`.

| Model | Final IoU | AP50 | AP75 | Fixed AP50 | Fixed AP75 | ECE50 | ECE75 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `local_anchor_residual_query` | 0.345 | 0.216 | 0.006 | 0.216 | 0.006 | 0.346 | 0.391 |
| `local_anchor_residual_query_quality_head` | 0.367 | 0.225 | 0.019 | 0.295 | 0.054 | 0.144 | 0.052 |

  - Interpretation:
    - The two-stage quality head remains useful under held-out alpha calibration: final IoU `+0.021`, AP50 `+0.010`, class AP50 `+0.049`, calibrated fixed AP50 `+0.079`, all `3/3` paired wins.
    - AP75-lite also improves: base AP75 `+0.013` and calibrated fixed AP75 `+0.048`, which suggests the ranking head is not only improving loose AP50 ordering.
    - ECE-lite improves strongly: ECE50 drops `0.346 -> 0.144`, and ECE75 drops `0.391 -> 0.052`.
    - The calibrated fixed AP50 (`0.295`) is close to final-eval best-q (`0.298`), so this run does not look like pure eval-set alpha overfitting.
    - Because this split is smaller than the previous full-eval run, the absolute AP values should not be compared directly to `results/det_real_official_fixed_q2_summary.csv`; use it as protocol evidence.
- Query-conditioned mask auxiliary:
  - Implemented per-query dense mask logits: `query_mask_logits_per_query` with shape `[B, Q, H, W]`.
  - Added matched-query mask loss: run Hungarian matching, supervise each matched query with its matched bbox rectangle mask via BCE + Dice; unmatched queries are not mask-supervised.
  - Added toy/real variants:
    - `local_learned_querymask`
    - `local_anchor_residual_query_querymask`
    - `local_anchor_residual_query_querymask_quality_head`
  - Toy smoke:
    - command output: `results/det_toy_querymask_smoke.csv`.
    - In a 40-step multi-distractor sanity check, `local_anchor_residual_query_querymask` learns matched masks (`mask IoU 0.429`) and improves final IoU/AP over `local_anchor_residual_query`. This only validates wiring, not a formal toy benchmark.
  - Real DET mini querymask:
    - command output: `results/det_real_querymask_300step_2seed.csv`.
    - `local_anchor_residual_query`: final/best IoU `0.376/0.376`, AP50 `0.266`, class AP50 `0.135`.
    - `local_anchor_residual_query_querymask`: final/best IoU `0.357/0.357`, AP50 `0.257`, class AP50 `0.058`, matched mask IoU/Dice `0.354/0.477`.
  - Real DET mini querymask + two-stage quality:
    - command output: `results/det_real_querymask_quality_head_400step_2seed.csv`.
    - non-quality `local_anchor_residual_query_querymask` trained to 400 steps: final/best IoU `0.364/0.367`, AP50 `0.288`, class AP50 `0.125`, matched mask IoU/Dice `0.341/0.451`.
    - `local_anchor_residual_query_querymask_quality_head`: final/best IoU `0.357/0.357`, base AP50 `0.257`, fixed-q2 AP50 `0.286`, best-q AP50 `0.290`, matched mask IoU/Dice `0.354/0.477`.
  - Interpretation:
    - Query-conditioned masks are learnable and correctly tied to Hungarian query identity.
    - As an auxiliary loss alone, querymask does not beat the residual-anchor detector on real DET mini box/AP at 300 steps.
    - Two-stage quality ranking gives only a small q-score recovery for querymask and does not make it competitive with residual-anchor quality.
    - This supports the next architectural step: use query masks for box refinement or query refinement, not merely as auxiliary supervision.
  - Query-mask box moment refinement:
    - Implemented `query_mask_refine`: each query predicts a mask, converts it to a differentiable soft cxcywh box moment, and applies a small gated interpolation from the raw box head toward that mask box.
    - Smoke artifact: `results/det_real_querymask_refine_smoke_50step_2seed.csv`.
    - Standard artifact: `results/det_real_querymask_refine_300step_2seed.csv`.

| Model | Final IoU | Best IoU | AP50 | Class AP50 | Mask IoU |
|---|---:|---:|---:|---:|---:|
| `local_anchor_residual_query` | 0.376 | 0.376 | 0.266 | 0.135 | 0.000 |
| `local_anchor_residual_query_querymask_refine` | 0.358 | 0.366 | 0.202 | 0.049 | 0.338 |

  - Interpretation:
    - Query-mask moment refinement is negative under the standard 300-step real DET mini protocol.
    - The branch learns matched masks, but the mask-derived box interpolation hurts final IoU, best IoU, AP50, and class-aware AP50.
    - This means the current query mask is not yet a reliable box-refinement signal. Do not continue this exact moment-refine design without a stronger coupling idea or a separate stabilization mechanism.

### Step 5: RF-DETR Distillation Track

Use RF-DETR as a teacher rather than trying to beat it from scratch:

- teacher boxes/classes on local images,
- distill class logits and boxes,
- optionally distill object-query embeddings or dense masks,
- compare student with and without AnchorQueryInit.

Success criterion:

- Distilled anchor-region student improves over non-distilled/random-query student on the same data.

### Step 6: Engineering Innovation Backlog

Prioritized engineering points:

- Anchor-lite routing: slots `1/2/4`, read-every-2, lowres-only, shared projections.
- Spatial-head interface standardization: every spatial model returns `logits`, `spatial_features`, `memories`, optional `routing_maps`.
- TinyViT spatial adapter: reshape ViT patch tokens back to `[B, C, H, W]` for dense heads.
- Overlay figure composer: same samples across models, one row per image, columns for GT/local/FPN/anchor.
- Result registry: parse CSVs, emit markdown tables, paired deltas, and config metadata.

Immediate implementation order:

1. Detection scaffold and tests.
2. Synthetic detection sanity.
3. AnchorQueryInit prototype.
4. DET annotation dataset loader.
5. DenseMaskAux in detection training.
6. RF-DETR teacher/distillation after the student detector is stable.

## Current Pivot: Local-State-First

The current strongest Pareto baseline is `no_prefill_local_mix`, not any early prefill or full memory-read variant. On the larger all-single-label DET-derived ImageFolder (`197` classes, `11,433` images), 1000-step strict paired MPS training gives `no_prefill_local_mix` final/best `0.250/0.250` at about `470 img/s`. `anchor_only_no_prefill` is slightly higher at `0.251/0.251`, but slower at about `341 img/s`; treat it as the best plugin candidate, not a replacement mainline.

Working interpretation:

- Early spatial prefill is not helping under the current classification setup.
- Cross-stage memory read is not the main source of gains.
- Lightweight online 2D local refinement is the most reliable signal so far.
- Anchor/region ideas should now be treated as optional plugins on top of the stronger no-prefill local backbone.
- The current negative prefill result may specifically be an "early memory" failure: memory built before features become semantic can be stale or noisy.
- The next memory version should be local-state-first, then delayed or stage-wise memory refresh with content-adaptive read.

Immediate strong baselines:

- `no_prefill_local_mix`: current Pareto reference.
- `anchor_only_no_prefill`: best current anchor/plugin candidate.
- `xattnres_no_prefill`: no-prefill cross-layer routing reference.
- `xattnres_style`: historical cross-stage residual-routing reference; weaker than its no-prefill control.
- `tiny_vit`: sequence/ViT reference.
- `anchor_no_prefill`: weak anchor-signal candidate, not the main baseline.

## Current P0: No-Prefill Controls and Delayed Memory Boundary

Run the strict paired control matrix before adding new memory modules:

- `no_prefill_local_mix`
- `xattnres_style`
- `xattnres_no_prefill`
- `anchor_no_prefill`
- `anchor_only_no_prefill`
- `tiny_vit`

Use two paired references:

- primary: `no_prefill_local_mix`
- secondary: `xattnres_style`

Decision rules:

- If `xattnres_no_prefill` improves over `xattnres_style`, then the issue is partly the prefill protocol, not XAttnRes-style routing itself.
- If `anchor_only_no_prefill` is much weaker than `anchor_no_prefill`, the anchor signal comes from the mixed local/anchor/lattice backbone, not anchor read alone.
- If neither beats `no_prefill_local_mix` on accuracy-speed tradeoff, keep local-state refinement as the mainline and treat memory as an optional plugin.

Current result:

- `xattnres_no_prefill` improves over `xattnres_style` by about `+0.007` final accuracy and wins `3/3` split seeds against it.
- `anchor_only_no_prefill` beats `anchor_no_prefill` while using fewer parameters and running much faster.
- `anchor_only_no_prefill` is only about `+0.002` over `no_prefill_local_mix` and runs slower, so it is not yet a new mainline.
- `no_prefill_local_mix` remains the cleanest accuracy-speed reference.

Delayed-memory experiments should come after this control matrix:

- `delayed_xattnres_no_prefill`: implemented, smoke-tested; local warmup before XAttnRes-style reads.
- `stage_refresh_region_slots_2x2`: implemented, smoke-tested; local warmup plus refreshed fixed-grid slots.
- `region_pool_mixer_no_history`: implemented, smoke-tested; current-state fixed-grid slots without cross-stage history.
- `fpn_sum_lite`: implemented, smoke-tested; fixed state fusion control.
- `fpn_concat_lite`: implemented, smoke-tested; fixed concat fusion control.

The FPN/ViTDet boundary is important: fixed stage fusion or fixed pyramid read is a control, not the claim. A new claim requires delayed or refreshed region memory plus content-adaptive read/write whose contribution is measured against `no_prefill_local_mix`.

Next formal matrix:

- `no_prefill_local_mix`
- `anchor_only_no_prefill`
- `xattnres_no_prefill`
- `delayed_xattnres_no_prefill`
- `fpn_sum_lite`
- `fpn_concat_lite`
- `region_pool_mixer_no_history`
- `stage_refresh_region_slots_2x2`

Use the same 197-class, 1000-step, 3-seed strict paired setting with `no_prefill_local_mix` and `anchor_only_no_prefill` as paired references.

Formal result:

- `anchor_only_no_prefill`: final/best `0.251/0.251`, about `338 img/s`.
- `no_prefill_local_mix`: final/best `0.250/0.250`, about `462 img/s`.
- `xattnres_no_prefill`: final/best `0.249/0.249`, about `377 img/s`.
- `delayed_xattnres_no_prefill`: final/best `0.248/0.248`, about `354 img/s`.
- `region_pool_mixer_no_history`: final/best `0.246/0.247`, about `371 img/s`.
- `stage_refresh_region_slots_2x2`: final/best `0.245/0.245`, about `360 img/s`.
- `fpn_sum_lite`: final/best `0.244/0.244`, about `447 img/s`.
- `fpn_concat_lite`: final/best `0.241/0.241`, about `443 img/s`.

Current decisions:

- Keep `no_prefill_local_mix` as the main Pareto baseline.
- Keep `anchor_only_no_prefill` as the strongest plugin candidate.
- Keep `xattnres_no_prefill` as a strong reference, but not the mainline.
- Do not promote delayed XAttnRes, fixed 2x2 region pooling, or stage-refresh grid slots under this classification evidence.
- FPN-like fixed fusion does not explain the local-mix result, because both `fpn_sum_lite` and `fpn_concat_lite` underperform `no_prefill_local_mix`.

Next useful work:

- Add equal-depth/equal-parameter local-mix controls before testing larger delayed or slot models.
- Test a spatially sensitive task: bbox quadrant, object size bin, weak localization heatmap, or small segmentation.
- If staying on classification, optimize `anchor_only_no_prefill` for speed with read-every-2 or cheaper anchor projection.

## Spatial Probe Update

Implemented `scripts/compare_bbox_probe_models.py` for DET-derived bbox probes. It supports:

- `quadrant4`
- `grid9`
- `size3`
- class-balanced train sampling
- class-balanced eval subsets
- raw accuracy
- balanced accuracy
- macro F1
- per-class recall
- confusion matrix
- majority and balanced-random baselines

Balanced `quadrant4` result at 64px, 1000 steps, 3 seeds:

- `fpn_sum_lite`: balanced acc `0.275`, macro F1 `0.221`, about `354 img/s`.
- `no_prefill_local_mix`: balanced acc `0.269`, macro F1 `0.200`, about `361 img/s`.
- `stage_refresh_region_slots_2x2`: balanced acc `0.267`, macro F1 `0.207`, about `298 img/s`.
- `anchor_only_no_prefill`: balanced acc `0.266`, macro F1 `0.188`, about `286 img/s`.
- `xattnres_no_prefill`: balanced acc `0.266`, macro F1 `0.192`, about `300 img/s`.
- `region_pool_mixer_no_history`: balanced acc `0.261`, macro F1 `0.185`, about `292 img/s`.

Interpretation:

- Unbalanced `quadrant4` is invalid as architecture evidence because it stays near the majority baseline.
- Balanced `quadrant4` gives a weak signal above `0.25`, but the signal favors `fpn_sum_lite`, not region/memory routing.
- This does not rescue memory-first or region-slot claims.
- The next spatial task should be harder and less reducible to coarse layout bias: balanced `grid9`, bbox center regression, or weak localization heatmap.

Next spatial P0:

- Balanced `grid9` has been run. `no_prefill_local_mix` and `xattnres_no_prefill` are the best current references at balanced acc `0.136`; `fpn_sum_lite` drops to `0.116`.
- The signal is still weak because the balanced random baseline is `0.111` and macro F1 stays below `0.10` for every model.
- BBox center regression has been run. It also fails to separate architectures: all models sit around mean L2 `0.1524-0.1528`, while the eval split mean-target baseline is about `0.1518`.
- Explicit `16x16` bbox-center heatmap localization has been run with a spatial `1x1` heatmap head. It is a better probe design than global-pooled regression, but the result still does not support a memory/region claim: argmax mean L2 stays around `0.1588-0.1592`, worse than the eval split mean-target baseline `0.1518`, and all paired deltas are tiny.
- Dense `16x16` bbox-mask heatmap has been run and is the first spatial probe with a useful positive signal. All models beat the mean-mask prior IoU `0.364` and center-box prior IoU `0.407`; `anchor_only_no_prefill` is strongest at IoU `0.436`, Dice `0.577`, with 3/3 paired IoU wins over both `no_prefill_local_mix` and `fpn_sum_lite`.
- Dense `32x32` bbox-mask heatmap confirms the signal. `anchor_only_no_prefill` remains strongest at IoU `0.434`, with 3/3 paired wins over both `no_prefill_local_mix` and `fpn_sum_lite`, and it has the highest small/medium/large area-stratified IoU means.
- Overlay visualizations for the `16x16` seed-41 run are under `results/bbox_mask_overlays_16x16_seed41/`.
- Keep both `no_prefill_local_mix` and `xattnres_no_prefill` as spatial references for the next probe.
- Do not spend more time on global-pooled coarse cell classification or global-pooled coordinate regression.
- Do not continue tuning Gaussian center heatmap alone unless there is a clear change in supervision or target construction.
- Next spatial task should build on dense supervision: add `TinyViTHeatmapAdapter`, inspect overlays, run anchor-only lite/slot sweep, and then consider weak segmentation-style masks. Treat the positive signal as anchor/region-style online interaction, not as a revival of early prefill.

## Historical P0: Prefill-AttnRes Baseline

Goal: prove the smallest useful version of the idea.

- Keep image features as `[B, C, H, W]` throughout the model.
- Use `SpatialPrefill2D` to build a complete 2D memory list before read blocks.
- Use `LatticeMemoryRead` to route over `memory depth x 2D offsets`.
- Confirm routing weights are normalized per spatial location.
- Use `scripts/run_toy_task.py` as the first tiny synthetic training task.
- Use `scripts/compare_toy_models.py` to compare against conv-only and flattened Transformer baselines.
- Use `--seeds N` and the generated `results/toy_compare.csv` for less noisy comparisons.

Why first: this directly tests the central hypothesis with minimal machinery.

Current toy tasks:

- `oriented_pair`: classify whether the green marker is right of or below the red marker. This is the default early smoke task.
- `aligned_pair`: classify whether two markers share the same row or column. This is harder and should be used after the local offset task is stable.
- `distractor_aligned_pair`: same target row/column relation with blue/gray distractors. This is the current hard synthetic stress test.

Current short-run observations:

- On `oriented_pair` with 40 CPU steps, `conv_only` and `prefill_lattice_attnres` solve the task, while the small flattened Transformer baseline lags.
- On `aligned_pair` with 100 CPU steps and 3 seeds, `conv_only` and `prefill_lattice_attnres` are both around 0.62 eval accuracy, while the small flattened Transformer is near chance. This is a useful harder task, but not yet evidence of a clear advantage.

Current comparison set includes `conv_only`, `seq_transformer`, `tiny_vit`, `xattnres_style`, `prefill_lattice_attnres`, `anchor_prefill_attnres`, and `graph_prefill_attnres`.

Latest smoke observation: on `aligned_pair` with 60 CPU steps and 2 seeds, `anchor_prefill_attnres` is above the ViT/XAttnRes-style baselines but still comparable to `conv_only`. Treat this as direction-finding only.

Latest hard-task smoke observation: on `distractor_aligned_pair` with 80 CPU steps and 2 seeds, `conv_only` is strongest at about 0.65 eval accuracy; proposed variants and XAttnRes-style are around 0.54-0.56; ViT/sequence baselines stay near chance. This is a negative result for the current graph/anchor implementation and should prevent overclaiming.

Next implementation priority: add true-image support through a local `ImageFolder` runner or a small dataset supplied by the user. Do not make broad claims from synthetic tasks alone.

## P1: Graph-Augmented Memory

Goal: let each patch read semantic neighbors, not only fixed lattice offsets.

- Add optional kNN edges from prefilled feature similarity.
- Compare fixed lattice offsets against lattice plus semantic graph neighbors.
- Keep local grid edges as the default to avoid losing image priors.
- Watch for oversmoothing by tracking feature diversity across blocks.

Why second: Vision GNN suggests graph topology is useful for irregular objects, but dynamic graph construction adds cost and implementation complexity.

## P2: V2M / 2D-SSM Prefill

Goal: replace the current convolutional prefill with a more principled 2D state update.

- Prototype four-corner or synchronous 2D state updates.
- Compare against bidirectional and four-direction scan baselines.
- Keep AttnRes-style memory read unchanged so the prefill mechanism is isolated.

Why third: V2M is closest to true 2D state modeling, but a faithful implementation is heavier than the current v1 demo.

## P3: Real Task Evaluation

Goal: move beyond shape validation.

- Start with small dense-prediction tasks, not ImageNet classification.
- Candidate tasks: synthetic segmentation, boundary detection, toy depth/order prediction.
- Track boundary quality, small-object behavior, and routing visualizations.
- Only consider ImageNet after the toy and dense-prediction checks show a real signal.

Current MPS smoke result on the DET-derived 20-class ImageFolder subset: at 64px and 300 steps, `anchor_prefill_attnres` is the best of the 6-model short run, while `graph_prefill_attnres` does not help. Repeat with more seeds and a stronger train/eval protocol before making any claim.

Implementation note: memory-read residuals now use a small learnable gate initialized to `1e-3`.

Gated 5-seed MPS core result initially suggested `anchor_prefill_attnres` was highest. A stricter paired anchor-vs-XAttnRes mechanism check shows anchor is not yet a stable win: final delta vs `xattnres_style` is about -0.002, with anchor winning 3/5 seeds but running about 2.4x slower. Treat anchor as a promising branch, not a confirmed advantage.

Next priority: anchor ablations against XAttnRes-style and CLS/global-token baselines. Do not claim anchor-prefill superiority until paired deltas are positive and stable.

Current anchor ablation note: a 3-seed, 300-step MPS run shows `multi_cls_vit` is much weaker than the anchor variants, so the anchor signal is not simply explained by adding multiple global Transformer tokens. However, `anchor_no_prefill` slightly outperforms full `anchor_prefill_attnres`, and `anchor_fixed_gamma_0` remains competitive. This means the current evidence does not prove that spatial prefill or learned memory read is the causal source of the gain.

Immediate next ablations:

- Treat `anchor_no_prefill` as the current best P0 branch, not full `anchor_prefill_attnres`.
- Use the implemented `prefill_local_mix` baseline as a strong local-backbone control; it is close to full anchor while much faster.
- Treat `anchor_read_only_no_lattice` and `lattice_only_no_anchor` as negative/weak source-isolation results under the current 300-step setting.
- Log step-wise mechanism stats at eval checkpoints, not only final block stats.
- Use `python -u` or flushed prints for long MPS experiments so runs are observable.

Historical P0 source ablation: an older non-strict run at 64px, 300 steps, 5 seeds had `anchor_no_prefill` at 0.135 mean eval acc and full `anchor_prefill_attnres` at 0.123. This should now be treated as provisional only because it predates the stable model-name seed offsets and shared train-loader shuffle order.

Seeding correction: `scripts/compare_imagefolder_models.py` now uses stable model-name-based seed offsets and shared train-loader shuffle order per split. Older comparison runs used model-list position to choose the initialization seed, and did not strictly pair training minibatch order, so treat older cross-run comparisons as provisional.

Latest strict paired result: at 64px, 300 steps, 5 seeds, `anchor_no_prefill` is the best final-accuracy mean among the current controls at 0.117, but only by a small paired final delta over `xattnres_style` (+0.007) and it does not beat `xattnres_style` on best eval accuracy. Full `anchor_prefill_attnres` ties `xattnres_style` on final accuracy and is worse on best accuracy while much slower. `prefill_local_mix` remains close to the anchor variants and much faster. This means local refinement explains a substantial part of the signal; spatial prefill and learned memory read remain unproven.

Larger-data update: on the all-single-label DET-derived ImageFolder (`197` classes, `11,433` images), 1000-step strict paired MPS training shows `no_prefill_local_mix` is currently strongest and fastest: final `0.249`, best `0.250`, about `442 img/s`. `anchor_no_prefill` reaches final/best `0.246` but runs at about `183 img/s`; `xattnres_style` reaches `0.242`; `tiny_vit` reaches `0.243`. This further downgrades the anchor-specific claim and makes lightweight no-prefill local refinement the strongest immediate baseline.

Next priority:

- Make `no_prefill_local_mix`, `anchor_no_prefill`, and `xattnres_style` the immediate controls for any future anchor-lite or region-mixer variant.
- Do not claim full `anchor_prefill_attnres` superiority under the current evidence.
- Add `xattnres_no_prefill` to test whether removing prefill improves XAttnRes-style too.
- Add `anchor_only_no_prefill` to isolate the online axis-anchor path without lattice reads.
- Treat anchor-lite as an optional add-on to the stronger no-prefill local backbone, not as the main claim.
- Prefer longer/stronger runs only after the no-prefill/local-mix controls are clean.

P0 control update: the dual-reference 197-class run confirms that `xattnres_no_prefill` is a stronger reference than `xattnres_style`, and that `anchor_only_no_prefill` is cleaner than `anchor_no_prefill`. Future work should use `no_prefill_local_mix`, `xattnres_no_prefill`, and `anchor_only_no_prefill` as the active controls.

Why later: classification can hide spatial-routing weaknesses behind global pooling.
## Query-Specific Center Binding Check

This check asks whether the off-center/query-binding issue can be fixed by making each Hungarian-matched query mask put its soft center near the matched GT box center.

Implementation:

- Added `query_mask_center_aux_loss` in `scripts/train_det_toy.py`.
- Added `local_anchor_residual_query_querymask_centeraux`.
- The loss only uses Hungarian matched queries and applies SmoothL1 to the softargmax center of `query_mask_logits_per_query`.
- Added CSV fields `loss_query_center_aux`, `eval_query_mask_center_l1`, `eval_query_mask_center_l2`,
  `eval_query_mask_center_pck025`, and center/offcenter query-mask center slice metrics.

Artifact:

- `results/det_real_query_center_aux_1000img_500step_3seed.csv`
- `results/det_real_query_center_aux_slice_metrics_1000img_500step_3seed.csv`

Protocol:

- 1000 images, top-10 train-label classes, 500 steps, 3 seeds.
- Compared `local_anchor_residual_query`, `local_anchor_residual_query_querymask`, and `local_anchor_residual_query_querymask_centeraux`.

| Variant | Final IoU | AP50 | Class AP50 | AP75 | Center AP50 | Offcenter AP50 | Mask IoU | Query-mask center L1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `local_anchor_residual_query` | 0.414 | 0.246 | 0.116 | 0.045 | 0.298 | 0.071 | 0.000 | 0.000 |
| `local_anchor_residual_query_querymask` | 0.415 | 0.307 | 0.132 | 0.045 | 0.380 | 0.062 | 0.422 | 0.109 |
| `local_anchor_residual_query_querymask_centeraux` | 0.412 | 0.294 | 0.134 | 0.044 | 0.370 | 0.070 | 0.394 | 0.110 |

Slice-level query-mask center diagnostics:

| Variant | Center L2 | Offcenter L2 | Center PCK@0.25 | Offcenter PCK@0.25 |
|---|---:|---:|---:|---:|
| `local_anchor_residual_query_querymask` | 0.146 | 0.223 | 0.893 | 0.627 |
| `local_anchor_residual_query_querymask_centeraux` | 0.139 | 0.238 | 0.911 | 0.596 |

Paired against `local_anchor_residual_query`:

- `querymask`: AP50 `+0.061`, `3/3` wins; class AP50 `+0.015`, `2/3` wins; final IoU only `+0.001`.
- `querymask_centeraux`: AP50 `+0.049`, `3/3` wins; class AP50 `+0.017`, `2/3` wins; final IoU `-0.002`.

Interpretation:

- Query-conditioned masks have a real aggregate AP signal in this larger 1000-image protocol.
- The center auxiliary does not improve the explicit query-mask center metric (`0.110` vs `0.109`) and does not improve detector AP over plain querymask.
- Slice diagnostics make the failure sharper: centeraux slightly improves center-object query-mask center L2/PCK, but worsens offcenter L2/PCK.
- Off-center AP remains weak and is not fixed by soft-center binding. The blocker is likely not just the query-mask center location, but object-specific query/proposal binding and ranking under off-center ambiguity.
- Keep `querymask` as an active coupling probe. Treat `querymask_centeraux` as a weak/negative control unless a future off-center-specific training protocol changes the result.

## Offcenter-Only Training Slice Control

This check asks whether the off-center failure is mainly caused by insufficient off-center training coverage. It adds `--train-slice-filter`, which restricts only the training split and leaves held-out eval unchanged.

Implementation:

- Added CLI flag `--train-slice-filter {none,small,medium,large,center,offcenter}` to `scripts/train_det_real.py`.
- Added split-level unit coverage for offcenter-only train filtering.

Artifact:

- `results/det_real_train_offcenter_1000img_500step_3seed.csv`

Protocol:

- 1000 images, top-10 train-label classes, 500 steps, 3 seeds.
- `--train-slice-filter offcenter`.
- Compared `local_anchor_residual_query` and `local_anchor_residual_query_querymask`.

| Variant | Final IoU | AP50 | Class AP50 | Center AP50 | Offcenter AP50 | Mask IoU | Offcenter query-mask center L2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `local_anchor_residual_query` | 0.368 | 0.139 | 0.055 | 0.145 | 0.108 | 0.000 | 0.000 |
| `local_anchor_residual_query_querymask` | 0.404 | 0.210 | 0.090 | 0.237 | 0.117 | 0.371 | 0.188 |

Interpretation:

- Offcenter-only training improves offcenter AP versus the all-train querymask run (`0.117` vs `0.062`) and improves the querymask offcenter center L2 (`0.188` vs `0.223`).
- It substantially hurts aggregate AP versus all-train querymask (`0.210` vs `0.307`) and center AP (`0.237` vs `0.380`).
- The offcenter weakness is partly representation/data-coverage related, but pure offcenter-only training is not a deployable fix.
- Next active control should be balanced or oversampled offcenter training, not a new scoring alpha or persistent proposal state.

## Offcenter Oversampling Control

This check keeps the full training split but samples offcenter-containing images more often. It tests whether the offcenter-only gain can be kept without discarding normal center-distribution samples.

Implementation:

- Added `--train-slice-oversample` and `--train-slice-oversample-factor`.
- Uses a deterministic `WeightedRandomSampler` over the training subset.

Artifact:

- `results/det_real_train_offcenter_oversample3_1000img_500step_3seed.csv`

Protocol:

- 1000 images, top-10 train-label classes, 500 steps, 3 seeds.
- `--train-slice-oversample offcenter --train-slice-oversample-factor 3`.
- Compared `local_anchor_residual_query` and `local_anchor_residual_query_querymask`.

| Variant | Final IoU | AP50 | Class AP50 | Center AP50 | Offcenter AP50 | Mask IoU | Offcenter query-mask center L2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `local_anchor_residual_query` | 0.413 | 0.261 | 0.119 | 0.332 | 0.067 | 0.000 | 0.000 |
| `local_anchor_residual_query_querymask` | 0.415 | 0.211 | 0.089 | 0.262 | 0.055 | 0.440 | 0.206 |

Interpretation:

- Offcenter oversampling at factor 3 does not recover the offcenter-only positive signal. Querymask offcenter AP is `0.055`, below the all-train querymask result (`0.062`) and well below offcenter-only querymask (`0.117`).
- It also hurts aggregate AP (`0.211` vs all-train querymask `0.307`).
- Oversampling improves the query-mask center geometry versus all-train (`offcenter center L2 0.206` vs `0.223`) but that does not translate into AP.
- Conclusion: the offcenter failure is not solved by simple data weighting. The remaining issue is more likely object-specific query assignment/ranking or feature ambiguity, not raw offcenter sample frequency alone.

## Query Assignment Entropy Diagnostic

This diagnostic asks whether offcenter failure comes from matched GTs collapsing onto too few query slots. It adds center/offcenter query-assignment entropy fields to the real-det evaluator.

Implementation:

- Added `center_query_assignment_entropy` and `offcenter_query_assignment_entropy`.
- They are computed from Hungarian matched-query counts for center/offcenter target slices.

Artifact:

- `results/det_real_query_assignment_entropy_1000img_500step_3seed.csv`

Protocol:

- 1000 images, top-10 train-label classes, 500 steps, 3 seeds.
- Compared `local_anchor_residual_query` and `local_anchor_residual_query_querymask`.

| Variant | AP50 | Center AP50 | Offcenter AP50 | Assignment Entropy | Center Entropy | Offcenter Entropy | Center no-GT top-k | Offcenter no-GT top-k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `local_anchor_residual_query` | 0.246 | 0.298 | 0.071 | 0.911 | 0.787 | 0.868 | 0.813 | 0.865 |
| `local_anchor_residual_query_querymask` | 0.307 | 0.380 | 0.062 | 0.939 | 0.834 | 0.879 | 0.699 | 0.855 |

Interpretation:

- Querymask improves aggregate and center AP, but still does not improve offcenter AP.
- Query assignment entropy is not lower for offcenter targets; querymask slightly increases it. The failure is therefore not a simple query-slot collapse.
- The dominant offcenter ranking issue remains high-score no-GT candidates: offcenter top-k no-GT rate is still very high (`0.855`).
- Next useful direction should target offcenter candidate generation/ranking under ambiguity, not query-count entropy or more center-position penalties.

## Stronger Background / No-Object Weight Control

This control asks whether the high no-GT top-k rate is caused by weak background supervision. The original DETR-style criterion used `no_object_weight=0.1`; this run uses `--no-object-weight 0.3`.

Implementation:

- Added `--no-object-weight` to `scripts/train_det_real.py`.
- It passes through to `DetectionCriterion(no_object_weight=...)`.

Artifact:

- `results/det_real_no_object_w03_1000img_500step_3seed.csv`

Protocol:

- 1000 images, top-10 train-label classes, 500 steps, 3 seeds.
- `--no-object-weight 0.3`.
- Compared `local_anchor_residual_query` and `local_anchor_residual_query_querymask`.

| Variant | Final IoU | AP50 | Class AP50 | AP75 | Center AP50 | Offcenter AP50 | Center no-GT top-k | Offcenter no-GT top-k | ECE50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `local_anchor_residual_query` | 0.422 | 0.323 | 0.147 | 0.052 | 0.400 | 0.082 | 0.690 | 0.807 | 0.208 |
| `local_anchor_residual_query_querymask` | 0.413 | 0.309 | 0.116 | 0.043 | 0.379 | 0.094 | 0.675 | 0.803 | 0.214 |

Comparison to default `no_object_weight=0.1`:

- Residual-anchor baseline improves strongly: AP50 `0.246 -> 0.323`, class AP50 `0.116 -> 0.147`, center AP50 `0.298 -> 0.400`, offcenter AP50 `0.071 -> 0.082`.
- Querymask stays around the same aggregate AP (`0.307 -> 0.309`) and improves offcenter AP (`0.062 -> 0.094`), but no longer beats the stronger residual-anchor baseline.
- Offcenter no-GT top-k rate decreases only modestly (`0.865 -> 0.807` for residual-anchor), so background weighting helps but does not solve the offcenter ranking failure.

Interpretation:

- Weak no-object/background supervision was a real bottleneck for the residual-anchor detector.
- Once background is strengthened, plain querymask is no longer the best AP route, although it retains a small offcenter AP advantage.
- The next scoring/training control should combine stronger background with two-stage quality ranking, because the current best AP path is still detector scoring/ranking rather than more query-position heuristics.

## Stronger Background + Two-Stage Quality Ranking

This follow-up checks whether `--no-object-weight 0.3` stacks with the existing frozen quality-head route.

Implementation:

- Added `--restore-best-metric {iou,ap50,ap50_class,ap75}`.
- Default remains `iou` for backward compatibility.
- Motivation: with stronger background supervision, the best-IoU checkpoint can have much worse AP than later checkpoints; restoring by IoU can therefore damage the quality-head phase.

Artifacts:

- `results/det_real_no_object_w03_quality_traincalib_1000img_500step_3seed.csv`
- `results/det_real_no_object_w03_quality_norestore_traincalib_1000img_500step_3seed.csv`
- `results/det_real_no_object_w03_quality_restore_ap50_traincalib_1000img_500step_3seed.csv`

Protocol:

- 1000 images, top-10 train-label classes, 500 steps, 3 seeds.
- `--no-object-weight 0.3`.
- `--quality-head-start-step 401 --quality-head-only-after-start`.
- Train-sourced alpha calibration: `--calibration-source train --calibration-frac 0.25`.

| Variant | Restore | Final IoU | AP50 | Class AP50 | q-fixed AP50 | AP75 | q-fixed AP75 | ECE50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `local_anchor_residual_query` | none | 0.420 | 0.361 | 0.171 | 0.361 | 0.058 | 0.058 | 0.209 |
| `local_anchor_residual_query_quality_head` | best IoU | 0.421 | 0.316 | 0.148 | 0.377 | 0.045 | 0.057 | 0.214 |
| `local_anchor_residual_query_quality_head` | none | 0.416 | 0.377 | 0.177 | 0.398 | 0.050 | 0.054 | 0.226 |
| `local_anchor_residual_query_quality_head` | best AP50 | 0.406 | 0.388 | 0.181 | 0.389 | 0.051 | 0.052 | 0.235 |

Interpretation:

- `--restore-best-before-quality-head` with the old best-IoU metric is harmful under stronger background: AP50 drops to `0.316` even though q-fixed AP50 recovers to `0.377`.
- No-restore and best-AP50 restore both preserve the stronger detector ranking and give small quality-weighted gains over the base AP50 (`0.398` and `0.389` q-fixed AP50 vs base `0.361`).
- The quality gain is now much smaller than in the old `no_object_weight=0.1` route because stronger background supervision already fixes much of the ranking problem.
- Future stronger-background quality runs should not use best-IoU restore. Use no-restore or `--restore-best-metric ap50`; keep `best_iou` only as a localization diagnostic.

## Stronger Background Offcenter Stress

This reruns the core offcenter stress with stronger background supervision and AP50-based quality restore.

Artifact:

- `results/det_real_no_object_w03_quality_restore_ap50_offcenter_1000img_500step_3seed.csv`

Protocol:

- 1000 images, top-10 train-label classes, 500 steps, 3 seeds.
- `--no-object-weight 0.3`.
- `--eval-slice-filter offcenter`.
- Quality route uses `--restore-best-before-quality-head --restore-best-metric ap50`.

| Variant | Final IoU | AP50 | Class AP50 | q-fixed AP50 | AP75 | q-fixed AP75 | Offcenter AP50 | Offcenter q-fixed AP50 | top-k FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `local_anchor_residual_query` | 0.325 | 0.136 | 0.061 | 0.136 | 0.009 | 0.009 | 0.054 | 0.054 | 0.830 |
| `local_anchor_residual_query_quality_head` | 0.337 | 0.171 | 0.080 | 0.168 | 0.013 | 0.014 | 0.095 | 0.053 | 0.802 |

Paired result vs base:

- Final IoU `+0.012`, `3/3` wins.
- AP50 `+0.035`, `3/3` wins.
- Class AP50 `+0.019`, `3/3` wins.
- q-fixed AP50 `+0.032`, `3/3` wins.
- AP75 / q-fixed AP75 both `+0.004`, `3/3` wins.
- Offcenter AP50 improves (`0.054 -> 0.095`), but offcenter q-fixed AP50 is essentially unchanged (`0.054 -> 0.053`).

Interpretation:

- Stronger background plus AP50 checkpoint restore gives the first clean positive offcenter-stress result for the quality route.
- The gain is partly checkpoint/restoration and base-score ranking, not purely fixed-quality scoring: q-fixed offcenter AP does not improve.
- This revises the offcenter conclusion: offcenter is still the hard slice, but it is not hopeless; no-object/background weight and checkpoint selection matter more than alpha sweeps, edge-grid seeding, or center penalties.
- Next useful work should keep `no_object_weight=0.3` and `restore-best-metric ap50` as the detector-side protocol, then test whether the remaining offcenter q-fixed weakness is due to quality calibration or candidate generation.

## Stronger Background Querymask Offcenter Control

This checks whether query-specific dense masks stack with the stronger offcenter protocol.

Artifact:

- `results/det_real_no_object_w03_querymask_quality_restore_ap50_offcenter_1000img_500step_3seed.csv`

Protocol:

- Same offcenter stress as above.
- Compared `local_anchor_residual_query` against `local_anchor_residual_query_querymask_quality_head`.

| Variant | Final IoU | AP50 | Class AP50 | q-fixed AP50 | AP75 | q-fixed AP75 | Mask IoU | top-k FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `local_anchor_residual_query` | 0.325 | 0.136 | 0.061 | 0.136 | 0.009 | 0.009 | 0.000 | 0.830 |
| `local_anchor_residual_query_querymask_quality_head` | 0.317 | 0.145 | 0.067 | 0.147 | 0.010 | 0.010 | 0.328 | 0.825 |

Paired result vs base:

- AP50 `+0.009`, `2/3` wins.
- Class AP50 `+0.006`, `3/3` wins.
- q-fixed AP50 `+0.010`, `2/3` wins.
- Final IoU `-0.007`, best IoU `-0.012`.

Interpretation:

- Querymask gives a small offcenter ranking gain under the stronger protocol, but it is much weaker than the plain residual quality route (`+0.009` vs `+0.035` AP50).
- It also lowers geometry despite learning masks (`mask IoU 0.328`).
- Keep querymask as a dense-coupling probe, not as the current offcenter mainline. The active detector protocol remains residual-anchor + `no_object_weight=0.3` + AP50 checkpoint restore for quality-stage runs.

Stronger-background offcenter slice-calibration control:

- `results/det_real_no_object_w03_quality_restore_ap50_offcenter_calib_1000img_500step_3seed.csv`
- Protocol: same stronger offcenter stress as above, but alpha calibration is restricted with `--calibration-slice-filter offcenter`.
- Result vs base: final IoU `-0.008` (`1/3` wins), AP50 `-0.001` (`1/3`), class AP50 `+0.008` (`2/3`), q-fixed AP50 `+0.002` (`2/3`), AP75 `-0.013`, q-fixed AP75 `-0.007`, and q-best AP50 `+0.007` (`2/3`).
- Interpretation: offcenter-only calibration does not recover the offcenter ranking failure. It slightly helps class-aware/q-best ranking but hurts geometry and AP75, and it removes offcenter samples from the detector training split. Do not use this as the default protocol.
- Active decision: keep calibration protocol simple under the stronger detector setting. The next useful work should target candidate generation/query binding for offcenter objects, not slice-only alpha calibration, persistent proposal-state rescue, or more alpha/temperature sweeps.
- Follow-up diagnostic from the stronger offcenter CSVs: the high-score offcenter candidates still rarely match the offcenter GT. Under the stronger quality run, offcenter combined top-k slice/non-slice/no-GT rates move from `0.024/0.170/0.806` to `0.005/0.222/0.773`; under querymask they become `0.018/0.214/0.768`. This means the remaining issue is not just calibration: the model often ranks no-GT or non-slice candidates ahead of the offcenter object.
- Next immediate control: test whether increasing background/no-object pressure further reduces no-GT high-score candidates. If `--no-object-weight 0.5` does not improve offcenter slice-match/no-GT breakdown, stop treating background loss as the main lever and move to query/object binding mechanisms.
- `--no-object-weight 0.5` offcenter stress artifact: `results/det_real_no_object_w05_quality_restore_ap50_offcenter_1000img_500step_3seed.csv`.
- `0.5` is mixed, not a simple loss. Base residual-anchor improves offcenter AP50 (`0.054 -> 0.101`) and offcenter combined top-k slice-match (`0.024 -> 0.085`), with final IoU `0.336` and AP50 `0.142`. However the quality-stage version loses final IoU (`0.321`), does not improve AP50 meaningfully (`0.144`), and drops offcenter fixed AP50 to `0.040`.
- Interpretation: stronger background pressure can help the base detector bind offcenter objects, but the current frozen quality stage suppresses or reorders those offcenter candidates poorly. Next check normal heldout with `no_object_weight=0.5` before making it a default; if aggregate performance survives, treat `0.5` as a base-detector candidate and revisit quality training/scoring only after candidate binding is stable.
- Normal heldout `0.5` artifact: `results/det_real_no_object_w05_quality_restore_ap50_1000img_500step_3seed.csv`.
- Aggregate result: `local_anchor_residual_query_quality_head` reaches AP50 `0.368`, fixed AP50 `0.375`, q-best AP50 `0.376`, AP75/fixed AP75 `0.050/0.050`, and combined ECE50/ECE75 `0.142/0.018`. This is close to the prior `0.3` AP50-restore aggregate run (`0.388/0.389` fixed AP50) while improving ECE75.
- Offcenter caveat remains: `0.5` base has offcenter AP50 `0.104`, but quality fixed offcenter AP50 falls to `0.035`. Stronger background helps candidate binding; frozen quality ranking still prefers center/other candidates over offcenter objects.
- Active decision: keep `no_object_weight=0.5` as an aggregate/background candidate, not as a complete offcenter fix. The next detector change should preserve the `0.5` base offcenter candidate benefit while preventing quality-stage suppression, likely through slice-aware quality training targets or query-object binding, not alpha/temperature sweeps.
