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
- Current RF-DETR path signal: max3 crop-fused localization has useful class-agnostic recall, but category transfer is the active bottleneck. Simple native category-id offsets and naive ImageNet/proposal-crop prior fusion do not solve it.
- Current unsupported claims: early prefill, stale history pools, full prefill-lattice memory, graph-prefill, side-only DenseMaskAux, quality loss on class logits, and old memory-first routing.
- RF-DETR-level performance requires a real detector, not just classification or bbox-mask probes.

Active collision-avoidance route:

- Avoid claiming "new DETR query initialization"; frame the detector path as layerwise proposal refresh / `reinject` plus ranking calibration.
- Do not frame persistent proposal state as the active architecture. The standard stability matrix did not rescue predicted persistent; keep it only as an oracle/proposal-quality diagnostic branch.
- Avoid claiming "new quality score"; frame the quality path as a frozen, post-detector, held-out calibrated ranking head.
- Avoid claiming "new local-global backbone"; if this route returns, frame it as ambiguity-triggered interaction scheduling.
- Avoid treating RF-DETR native category failure as an off-by-one bug. The formal category-id transform check peaks at AP50 `0.0194`, far below ImageNet-prior and oracle-category settings.
- See `docs/research_route_collision_avoidance.md` for the detailed roadmap, stage gates, and IP/literature risk framing.

Immediate stage gates:

- P0 calibration gate: frozen ranking should produce meaningful held-out AP75/ECE or score-IoU correlation gains before expanding calibration.
- P1 proposal-consumption gate: layerwise proposal refresh / `reinject` is the active predicted-proposal path. Persistent state can only return to mainline if a future proposal-quality/oracle-gap run beats `reinject` on both geometry and AP without post-hoc tuning.
- P2 RF-DETR category gate: category work should focus on stronger category teachers or integrated detector-side class heads. Existing diagnostics show top-100 localization recall around `0.75-0.77`, but class-aware recall only `0.31-0.32`; the per-category vs global-topK ranking gap is only `0.006-0.013`, so missing/incorrect candidate categories dominate.
- Category source scorecard: `results/rfdetr_category_source_scorecard_full_summary.csv` shows existing torchvision priors/ensembles do not beat the current max3 ResNet50 top-5 prior. This lowers priority for more naive post-hoc category priors.
- Candidate-constrained oracle: `results/rfdetr_candidate_category_oracle_summary.csv` shows the current top-5 candidate set contains the nearest-GT category for only `20.4%` of grouped proposal boxes. Candidate reranking alone is insufficient; even oracle IoU scoring over the constrained candidates reaches AP50 `0.4736`, below the all-category IoU oracle AP50 `0.7515`.
- Stratified RF-DETR candidate-set diagnostic: `results/rfdetr_stratified_seed41_2best_candidate_oracle_summary.csv` and the decoder-class counterpart confirm the same failure on the fair train1000 split. Base grouped-box candidate hit rate is only `36.9%`, decoder+class +1ep falls to `35.3%`, and candidate-oracle AP50 is only `0.4996/0.4974`, far below the nearest-GT relabel oracle AP50 `0.6009`. This means many localized bbox groups do not emit the correct category at all; stop expanding low-interference decoder/class scope and prioritize stronger detector-side category supervision or teacher signals.
- Per-category candidate-set split: `results/rfdetr_stratified_seed41_2best_candidate_oracle_per_category.csv` shows the issue is not just long-tail noise. For categories with at least `20` grouped boxes, base has `10` zero-hit categories, including high-localization wrong-class cases like `n03676483` (`47` groups, mean nearest IoU `0.560`) and `n07695742` (`40`, `0.546`). Decoder+class still has `8` zero-hit high-support categories and lowers the weighted hit rate for `groups>=20` from `0.378` to `0.362`; it is not a broad class-candidate repair.
- Candidate split-count join: `results/rfdetr_stratified_seed41_2best_candidate_oracle_per_category_with_split_counts.csv` shows the worst zero-hit high-support categories are not simply uncovered by training. In the base checkpoint, zero-hit categories with `groups>=20` have train counts from `6` to `20` with median `13.5`; `n03676483` and `n07695742` each have `20` train boxes and high nearest IoU but no correct emitted category. Train count vs hit-rate correlation is weak (`r≈0.27` base, `r≈0.25` decoder+class). This strengthens the class-head/semantic-candidate diagnosis.
- Qualitative candidate-confusion overlays added: `results/rfdetr_stratified_seed41_2best_confusion_overlays/contact_sheet.jpg` and `manifest.csv` show high-IoU wrong-category cases such as `n07695742 -> n01726692`, `n07739125 -> n07749582`, `n02799071 -> n02786058`, and `n03676483 -> n07880968`. These are visibly localized objects with wrong semantic candidates, so the current blocker is category assignment/candidate generation rather than bbox support.
- Hard-category oversampling hook added: `scripts/build_coco_hard_category_oversample.py` builds an RF-DETR-compatible train split by duplicating images containing candidate-missing categories while preserving valid/test. The fixed absolute-symlink smoke dataset `/private/tmp/rfdetr_stratified_seed41_train1000_hardcat_oversample60_abs` passes RF-DETR `--check-only`; train images/annotations become `1000/4409 -> 1087/5221`. Artifact: `results/rfdetr_stratified_seed41_hardcat_oversample60_abs_summary.csv`.
- Hard-category oversampling 1ep class-head result: `results/rfdetr_stratified_seed41_hardcat60_classhead_1ep_test_cocoeval.csv` reaches class `AP/AP50/AP75 = 0.1624/0.1961/0.1729`, a small improvement over the base checkpoint and roughly tied with decoder-class +1ep. However candidate diagnostics are negative: grouped-box hit rate drops to `34.1%`, candidate-oracle AP50 drops to `0.4875`, and the original 10 zero-hit hard categories recover only `3.98%` weighted hit rate. Decision: hard-category oversampling is a weak AP regularizer, not a candidate-generation fix; do not scale this path blindly.
- Low-cost category intervention summary added: `results/rfdetr_stratified_seed41_category_intervention_summary.csv` collects base, class-head, query-class, decoder-class, and hardcat rows. It should be the quick stage-gate table for this branch: hardcat is best/near-best on AP but worst on candidate hit, so future work needs stronger semantic supervision rather than more class-head-scope or oversampling variants.
- Matched-proposal crop classifier check: `results/rfdetr_matched_propcrop_prior_heldout_summary.csv` extends `scripts/train_coco_proposal_crop_classifier.py` with `--train-crop-source matched_predictions`. It is negative: proposal-matched frozen ResNet50 reaches heldout GT-crop top5 `0.6667`, but proposal relabeling AP50 is only `0.0172` with multiply and `0.0103` with keep-score. Do not pursue small-split standalone proposal-crop classifiers as the RF-DETR category source.
- Integrated RF-DETR class-head protocol: `results/rfdetr_integrated_class_head_summary.csv` evaluates full prepared split RF-DETR detector-side finetunes with exported `model_num_classes=200`. Nano 384px improves with longer training: class-aware AP50 rises from `0.0495` at 1 epoch to `0.1224` at 3 epochs and `0.1437` at 5 epochs; class-agnostic AP50 remains much higher (`0.5584 -> 0.6074 -> 0.6214`), confirming that category/class-head learning is the active bottleneck. The 5-epoch Nano result still trails the current max3 crop + ResNet50 top-5 prior AP50 `0.1697`, but it is now the correct integrated detector-side route. Small 384px improves from AP50 `0.0513` at 1 epoch to `0.1178` at 3 epochs and `0.1835` after resumed 5-epoch training, with class-agnostic AP50 `0.6285`. This makes resumed Small the strongest validation-mirrored integrated detector-side checkpoint so far, but not automatically a stronger pseudo-label teacher. Do not return to post-hoc crop priors; next step is stronger teacher selection/filtering or a formal long-train detector-side protocol.
- Detector-side pseudo-label distillation hook added: `scripts/build_coco_pseudolabel_dataset.py` converts teacher COCO prediction JSONs into RF-DETR-compatible pseudo-label split annotations while preserving original images by symlink/copy. Smoke: `/private/tmp/rfdetr_pseudolabel_smoke` used Nano 5ep test predictions as pseudo labels for the test split, produced `808` pseudo annotations at `min_score=0.25/topk=20`, and passed `scripts/train_rfdetr_coco.py --check-only` with `dataset_num_classes=200`. This is the first concrete detector-side distillation substrate; next meaningful run needs teacher predictions for the train split from a stronger teacher, then train RF-DETR on teacher-only or GT+pseudo labels.
- Detector-side pseudo-label distillation smoke completed with train-split Nano 5ep self-teacher predictions. The teacher exported `98,506` train predictions; filtering at `score>=0.25/top20` gives `2,191` train pseudo boxes. On the validation-mirrored heldout protocol, `teacher-only` Nano 384px 1ep slightly improves class AP50 over the GT 1ep baseline (`0.0552` vs `0.0495`) while keeping class-agnostic AP50 comparable (`0.5499` vs `0.5584`). Raw `GT+pseudo` concatenation is negative: class AP50 `0.0484` and class-agnostic AP50 `0.3737`, likely from duplicate/noisy pseudo boxes confusing Hungarian matching. Artifact: `results/rfdetr_detector_side_pseudolabel_summary.csv`. Next distillation should use a stronger train-split teacher or a two-stage teacher-only pretrain -> GT finetune schedule, not naive GT+pseudo appending.
- On the old validation-mirrored protocol, teacher-only pretrain -> GT finetune was the first positive detector-side distillation schedule. Using Nano 5ep self-teacher pseudo labels for 1 epoch, then finetuning on GT for 1 epoch, gives class AP50 `0.0816`, above GT 1ep (`0.0495`), teacher-only (`0.0552`), and raw GT+pseudo (`0.0484`), with class-agnostic AP50 `0.5749`. Extending the GT finetune to 2 and 3 epochs strengthens the result: class AP50 `0.1088 -> 0.1428`, AP75 `0.0820 -> 0.1048`, and off-center AP50 `0.1163 -> 0.1514`. The 1+3 staged schedule nearly ties GT-only Nano 5ep class AP50 (`0.1428` vs `0.1437`) while exceeding its off-center AP50 (`0.1514` vs `0.1454`). A second seeded run (`--seed 43`) reproduces the result with class AP50 `0.1413`, AP75 `0.1070`, class-agnostic AP50 `0.6367`, and off-center AP50 `0.1522`. Extending to 1+4 does not improve class AP50 (`0.1342`) despite stronger localization / large-object AP50 (`loc AP50 0.6290`, large AP50 `0.3127`). This result is now qualified by the independent-test checks below: staged pseudo-pretrain remains useful as a localization/warm-start route, but is not yet a clear equal-budget class-AP50 win. Caveat: RF-DETR's `best_total` checkpoint selected a stale/high EMA (`0.7167`) in this flow, so reported staged-finetune results use `checkpoint_best_regular.pth` and independent `predict_rfdetr_coco.py` export. Next: stronger train-split teacher predictions; do not use raw GT+pseudo concatenation.
- Historical validation-mirrored / stronger-teacher caution: staged13, Small3, and Small-resume5 were not better train pseudo teachers than Nano5 under the old checked filters, despite some being better heldout detectors. This caution was first superseded by Small-resume8 and is now superseded again by Small-resume12 seed41, which clears the train-teacher gate by a wide margin. Keep the general lesson, but treat this as a historical diagnostic note rather than the active route: independent-test RF-DETR train1000 direct training is now the main benchmark path.
- Pseudo-label filtering diagnostic added: `scripts/coco_annotations_to_predictions.py` converts selected pseudo-label COCO annotations back into detection predictions using `teacher_score`, so pseudo-label filters can be evaluated directly against train GT before spending MPS time on RF-DETR training. Artifact: `results/rfdetr_pseudo_filter_quality_summary.csv`. Historical Nano5 result: stricter filters were worse because recall collapsed, and wider `s015/s020` filters looked better under static pseudo AP but collapsed in teacher-only 1ep RF-DETR training. This means static train-GT pseudo AP is not enough; every new teacher/filter needs a training-stability check. Small-resume12 seed41 is now the strongest checked train teacher: raw train class AP50 is `0.6764`, raw loc AP50 is `0.8741`, and filtered `s015/s020/s025 top20` class AP50 is `0.6430/0.6260/0.6170`, well above Small-resume8 seed41 (`0.4221/0.4033/0.3832`) and seed43 resume8 (`0.3804/0.3638/0.3475`). Next pseudo-label work should test teacher-only pretrain -> GT finetune with this stronger teacher; raw GT+pseudo concatenation remains downgraded.
- Engineering protocol correction: `scripts/check_rfdetr_handoff.py` now reports split annotation SHA256 hashes, category-range consistency, and whether valid/test annotations are identical. Current prepared RF-DETR split has consistent `1..200` categories across train/valid/test, but `valid` and `test` annotations are byte-identical. Treat existing `*_test_*` RF-DETR results as validation-mirrored heldout results, not an independent test set. Artifact: `results/rfdetr_offcenter_seed41_handoff_check.json`.
- Independent-test protocol added: `scripts/split_coco_by_images.py --keep-all-categories` creates image-disjoint valid/test annotation JSONs while preserving the original 200-category table. Prepared `rfdetr_offcenter_seed41_indtest_seed43` has train `330/1610`, valid `100/319`, and test `100/365`; handoff check confirms consistent `1..200` categories and `valid_test_annotations_identical=false`. On this true independent test, GT-only Nano 384px 1ep beats teacher-only `s025/top20` 1ep: class AP50 `0.0224` vs `0.0204`, AP75 `0.0201` vs `0.0182`, class-agnostic AP50 `0.2113` vs `0.1908`, and offcenter AP50 `0.0154` vs `0.0145`; the old 1ep teacher-only positive signal is protocol-sensitive. Staged pseudo-pretrain -> GT finetune survives the true independent split: Nano5 `s025/top20` pseudo pretrain for 1ep followed by GT finetune for 3ep reaches class AP50 `0.1618`, AP75 `0.1259`, class-agnostic AP50 `0.6170`, offcenter AP50 `0.1310`, and small AP50 `0.1035`. Equal-total-epoch GT-only 4ep reaches nearly identical class AP50 (`0.1618`) and class-agnostic AP50 (`0.6151`), with staged slightly better on AP/AP75/loc AP50 but GT-only slightly better on offcenter/small/medium AP50. Corrected decision: single-stage teacher-only is not robust; staged pseudo-pretrain is useful but not yet a clear equal-budget class-AP50 win. Next confirmation should be multi-seed/equal-budget. Artifacts: `results/rfdetr_indtest_seed43_detector_side_summary.csv` and `results/rfdetr_offcenter_seed41_indtest_seed43_handoff_check.json`.
- Reporting note: the `offcenter_ap50`, `center_ap50`, `small_ap50`, `medium_ap50`, and `large_ap50` columns in `results/rfdetr_indtest_seed43_detector_side_summary.csv` are slice AP50 values from `scripts/evaluate_coco_slices.py`, not COCO `AP_small/AP_medium/AP_large` area metrics.
- Second-seed equal-budget check completed on the same independent split. With seed41, staged Nano5 `s025/top20` pretrain -> GT finetune reaches class AP50 `0.1502`, AP75 `0.1042`, loc AP50 `0.6263`, center AP50 `0.1710`, small AP50 `0.1216`, and large AP50 `0.2970`; equal-budget GT-only 4ep reaches class AP50 `0.1517`, AP75 `0.1046`, loc AP50 `0.6194`, offcenter AP50 `0.1383`, small AP50 `0.1050`, and large AP50 `0.2404`. Updated decision: staged pseudo-pretrain remains useful as a localization / warm-start route and improves large-object + center slices in seed41, but it still does not produce a clear equal-budget class-AP50 win across seeds. Do not claim self-teacher distillation beats GT-only yet; next RF-DETR distillation step should use a stronger train-split teacher, a longer formal protocol, or a different pseudo-label consumption schedule rather than more self-teacher repeats.
- Long-train Small detector-side class head now becomes the strongest RF-DETR route, but the optimal epoch is seed-sensitive. The resumed Small 384px seed41 run improves with longer training: 5ep -> 8ep -> 12ep -> 16ep -> 20ep class AP50 is `0.2405 -> 0.2677 -> 0.3070 -> 0.3209 -> 0.3625`, AP75 is `0.1795 -> 0.2053 -> 0.2155 -> 0.2268 -> 0.2566`, offcenter AP50 is `0.2238 -> 0.2260 -> 0.2955 -> 0.2881 -> 0.3241`, small AP50 is `0.2457 -> 0.2594 -> 0.2911 -> 0.3258 -> 0.3334`, and large AP50 reaches `0.4588` at 20ep. The second direct Small seed43 run improves through 12ep but drops at 16ep: 5ep -> 8ep -> 12ep -> 16ep class AP50 is `0.1762 -> 0.2066 -> 0.2923 -> 0.2753`, while AP75 improves from `0.1802` to `0.1957` and class-agnostic AP50 improves from `0.6290` to `0.6348`. Direct Small 12ep two-seed mean class AP50 is about `0.2997`; seed41 20ep reaches the current single-run peak `0.3625`, but seed43 shows that longer training is not monotonically better for class AP50. RF-DETR internal validation metrics after resume can be misleadingly high or non-monotonic; official conclusions use independent `predict_rfdetr_coco.py` export from `checkpoint_best_regular.pth`.
- RF-DETR epoch-selection table added: `scripts/summarize_rfdetr_epoch_curve.py` writes `results/rfdetr_direct_small_epoch_curve_indtest_seed43.csv` from the detector-side summary. This should be the quick reference before starting any new long RF-DETR run.
- RF-DETR runner now exposes `--lr-drop`, `--lr-scheduler`, `--lr-min-factor`, and `--warmup-epochs`. The first low-interference schedule check is positive for seed43: direct Small loaded the 12ep `checkpoint_best_regular.pth` through `--pretrain-weights`, used a fresh optimizer with `lr=3e-5`, and trained 4 more epochs. Independent-test class AP50 improves to `0.3102`, above both seed43 12ep (`0.2923`) and resume16 (`0.2753`); AP75 improves to `0.2190`, loc AP50 is `0.6316`, offcenter AP50 is `0.2755`, and medium AP50 is `0.3025`. The seed41 replication from the stronger 20ep checkpoint is negative for peak AP: low-LR fresh continuation reaches class AP50 `0.3546`, below the seed41 20ep peak `0.3625`, although medium AP50 improves to `0.3751`. Current interpretation: lower-LR fresh continuation can rescue a seed after a bad blind resume, but it is not a universal improvement over an already strong checkpoint. Stop schedule sweeping unless a new longer-data or stronger-protocol reason appears.
- Larger random RF-DETR protocol prepared and smoke-tested: `rfdetr_random_seed41_train1000_val200_test200` uses image-disjoint train/valid/test splits with 1000/200/200 images, 2894/681/611 boxes, and a preserved 200-class category table. A Small 384px 2ep seed41 run reaches test class AP50 `0.0864`, AP75 `0.0741`, AP `0.0708`, class-agnostic AP50 `0.5550`, offcenter AP50 `0.0663`, small AP50 `0.0855`, medium AP50 `0.0908`, and large AP50 `0.1878`. Interpretation: the larger protocol is valid and has strong localization even after only 2 epochs, but the class head is far from trained; it is not comparable to the old 330-image 20ep run yet. Next progress should be a low-frequency formal longer run on this larger split, not more short schedule tweaks on the old split.
- Larger random RF-DETR continuation is strongly positive so far. Continuing the 2ep checkpoint in repeated 4-epoch fresh-optimizer blocks at `lr=1e-4` gives class AP50 `0.2196 -> 0.2912 -> 0.3155`, AP75 `0.1841 -> 0.2395 -> 0.2517`, AP `0.1746 -> 0.2190 -> 0.2394`, and class-agnostic AP50 `0.6111 -> 0.6265 -> 0.6427`. The latest 10best regular checkpoint also improves hard slices: offcenter AP50 `0.3081`, small AP50 `0.2792`, medium AP50 `0.3167`, large AP50 `0.3572`. Artifact: `results/rfdetr_random_seed41_train1000_protocol_summary.csv`. This confirms the 1000-image protocol is the active RF-DETR route; class head training is still improving materially, so future progress should be planned as low-frequency longer detector-side runs rather than more old-split schedule sweeps. Caveat: RF-DETR internal validation selected epoch 2 in the last block and dipped at epoch 3, so formal conclusions must continue to use independent `predict_rfdetr_coco.py` export from `checkpoint_best_regular.pth`.
- RF-DETR deformable attention diagnostic added: `scripts/visualize_rfdetr_deformable_attention.py` hooks `MSDeformAttn` during `model.predict`, aggregates the final captured deformable cross-attention sampling locations into a heatmap, and overlays sampling density with GT boxes and top predictions. Smoke artifact: `results/rfdetr_attention_overlays_train1000_6best/contact_sheet.jpg`; the manifest also records GT-box attention mass, top-prediction attention mass, entropy, and peak location. Interpret this as sampling-density / coverage visualization, not a ViT-style full attention matrix or a quantitative AP result.
- Query-level RF-DETR attention diagnostic added to the same script via `--query-overlays K`. The script now runs an equivalent raw RF-DETR forward/postprocess path that preserves each detection's `query_index`, then saves per-query sampling-density overlays and records query-level GT attention mass / prediction attention mass. Smoke artifact: `results/rfdetr_attention_query_overlays_train1000_6best/query_contact_sheet.jpg`. This is the right diagnostic for off-center failures: some high-score queries place most sampling mass inside GT boxes, while failure cases can show zero GT attention mass despite confident predictions.
- Attention manifest summarizer added: `scripts/summarize_rfdetr_attention_manifest.py` flattens image-level and query-level attention diagnostics into CSV and optional JSON diagnostics. Current artifacts: `results/rfdetr_attention_query_overlays_train1000_6best/image_attention_summary.csv`, `results/rfdetr_attention_query_overlays_train1000_6best/query_attention_summary.csv`, and `results/rfdetr_attention_query_overlays_train1000_6best/attention_diagnostics.json`. In the 6-image off-center smoke, top-query GT attention mass has mean `0.605`, median `0.818`, and `2/14` zero-GT-mass query failures; `9/14` top queries have nearest-GT IoU >= `0.5`, with mean IoU `0.581` and median IoU `0.773`. GT attention mass and nearest-GT IoU are strongly aligned in this tiny sample (`Pearson=0.950`, `Spearman=0.753`), with high-IoU queries averaging `0.873` GT mass and low-IoU queries averaging `0.124`; treat this as a failure-mining diagnostic, not a statistical claim.
- Larger top-1 off-center attention mining added: `results/rfdetr_attention_top1_offcenter40_train1000_6best/` contains 40 test images with one query overlay each, CSV summaries, `attention_diagnostics.json`, and bucket contact sheets under `bucket_contact_sheets/`. This run has mean nearest-GT IoU `0.751`, median `0.858`, IoU50 `32/40`, mean GT attention mass `0.834`, and mass-IoU correlations `Pearson=0.726`, `Spearman=0.359`. Buckets: `32` aligned hits, `5` high-mass/low-IoU failures, `2` aligned misses, `1` zero-mass miss. Interpretation: many top predictions do sample inside GT support, so remaining off-center failures are not only "attention misses"; the high-mass/low-IoU bucket should drive box refinement / duplicate / assignment diagnosis, while zero-mass misses still indicate query representation coverage failure.
- Full-test top-1 attention mining added: `results/rfdetr_attention_top1_test200_train1000_6best/attention_diagnostics.json` and the paired CSVs summarize 200 test images without treating it as AP evaluation. Top-1 query overlays exist for `196/200` images above threshold. Mean nearest-GT IoU is `0.835`, median `0.935`, IoU50 is `177/196`, mean GT attention mass is `0.814`, and mass-IoU correlations are `Pearson=0.687`, `Spearman=0.242`. Buckets: `177` aligned hits, `8` high-mass/low-IoU, `8` aligned misses, `3` zero-mass. Nearest-GT category match is `149/196=0.760`, and rises to `147/177=0.831` among IoU50 top-1 queries. This strengthens the failure split: the trained RF-DETR often samples the target support correctly and often labels its highest-score localized query correctly; the remaining class AP gap is more likely caused by full-set ranking, duplicate/false-positive handling, and long-tail candidate coverage than by a universal top-query representation failure.
- RF-DETR full-prediction ranking/category diagnostics added for the same 6best checkpoint. Score-IoU correlation artifacts: `results/rfdetr_random_seed41_train1000_small_384_seed41_6best_score_iou_loc.csv` and `results/rfdetr_random_seed41_train1000_small_384_seed41_6best_score_iou_classaware.csv`; category coverage artifact: `results/rfdetr_random_seed41_train1000_small_384_seed41_6best_category_coverage_gap.csv`. Overall top100 recall at IoU 0.5 is `loc=0.872`, `class-aware=0.665`, and `per-category=0.717`; offcenter top100 is `0.812/0.577/0.624`, and small top100 is `0.720/0.543/0.603`. Full-set score-IoU correlation is weak despite strong top predictions (`loc Pearson=0.058`, Spearman=`-0.009`; class-aware Pearson=`0.386`, Spearman=`0.059`). Interpretation: top-1 query behavior is good, but AP is still limited by full-set category coverage and score/ranking, especially on small and offcenter objects.
- The same ranking/category diagnostics were rerun for the 10best checkpoint. Artifacts: `results/rfdetr_random_seed41_train1000_small_384_seed41_10best_score_iou_loc.csv`, `results/rfdetr_random_seed41_train1000_small_384_seed41_10best_score_iou_classaware.csv`, and `results/rfdetr_random_seed41_train1000_small_384_seed41_10best_category_coverage_gap.csv`. Top100 coverage improves to `loc=0.876`, `class-aware=0.683`, `per-category=0.741`; offcenter top100 becomes `0.826/0.594/0.658`, and small top100 becomes `0.728/0.595/0.647`. Full-set score-IoU correlation is still weak (`loc Pearson=0.064`, Spearman=`0.018`; class-aware Pearson=`0.361`, Spearman=`0.027`). Interpretation: longer RF-DETR class-head training improves candidate coverage, especially small/offcenter category recall, but does not fix full-set score calibration.
- A valid-split post-hoc calibrator was applied to the 10best RF-DETR checkpoint without test-set alpha tuning. It improves score-IoU correlation (`class-aware Spearman 0.027 -> 0.291`, Pearson `0.361 -> 0.402`; loc Spearman `0.018 -> 0.120`) but does not materially improve test AP or top-ranked detection quality (`AP50 0.3155 -> 0.3158`, AP75 `0.2517 -> 0.2521`; class-aware top100 mean IoU `0.9418 -> 0.9283`). Offcenter AP50 improves only slightly (`0.3081 -> 0.3120`) and small AP50 drops slightly (`0.2792 -> 0.2767`). Interpretation: a valid-fitted post-hoc diagnostic calibrator can fit correlation structure, but it is not enough to solve the 10best full-set AP gap; do not continue post-hoc score-rescoring sweeps without a stronger candidate-generation/class-head change.
- RF-DETR Small train1000 was continued from 10best with a fresh optimizer at lower LR `5e-5` for 4 epochs. The best regular checkpoint occurs at continuation epoch 1 (`val mAP/AP50/AP75 = 0.2454/0.3231/0.2557`), while later epochs regress. Independent test improves class AP50 only modestly (`10best 0.3155 -> 14best 0.3242`) and AP75 barely (`0.2517 -> 0.2523`), while loc AP50 drops (`0.6427 -> 0.6271`). Slices: offcenter AP50 is roughly flat/slightly lower (`0.3081 -> 0.3064`), small AP50 improves (`0.2792 -> 0.2980`), and large AP50 improves (`0.3572 -> 0.3864`). Ranking/category diagnostics show weak score-IoU correlation remains (`class-aware Pearson=0.375`, Spearman=`0.033`) and top100 coverage does not clearly improve (`loc=0.872`, `class-aware=0.689`, `per-category=0.733`). Interpretation: low-LR continuation can still add a little class AP, but it is now a diminishing-return training lever rather than a mechanism breakthrough.
- Applying the same valid-split calibrator to the 14best checkpoint gives the first meaningful calibrated AP gain on the train1000 protocol: class `AP/AP50/AP75 0.2421/0.3242/0.2523 -> 0.2494/0.3297/0.2593`, with offcenter AP50 `0.3064 -> 0.3248` and large AP50 `0.3864 -> 0.3960`. Score-IoU correlation improves (`class-aware Spearman 0.033 -> 0.274`, Pearson `0.375 -> 0.413`; loc Spearman `0.071 -> 0.158`) while top100 class-aware coverage slightly drops (`0.689 -> 0.678`) and top100 offcenter class-aware recall also slightly drops (`0.604 -> 0.597`). Interpretation: once the detector is stronger, valid-fitted ranking calibration can recover some AP, especially offcenter, but it is still reordering existing predictions rather than generating missing candidates; it should remain a ranking layer on top of detector-side learning, not a substitute for better candidate/category coverage.
- 14best oracle diagnostics quantify the remaining bottleneck. Same-category oracle IoU scoring reaches class `AP/AP50/AP75 = 0.452/0.569/0.473`, showing large ranking-quality headroom when the predicted category is already correct. Nearest-GT category relabeling while keeping the detector's original score reaches `0.469/0.648/0.482`, much higher than the calibrated score path, so category assignment/candidate coverage is still the dominant class-aware AP50 bottleneck. Full nearest-GT relabel + IoU score gives `0.433/0.512/0.471`, lower AP50 because duplicate ordering/COCO precision behavior changes under pure IoU scoring; treat it as a diagnostic upper-bound variant, not a deployable recipe.
- Per-category coverage diagnostics now make the class-head bottleneck concrete. At `top100 / IoU 0.5 / gt_count >= 3`, several classes have near-perfect localization coverage but zero class-aware recall, including `n02503517` (`gt=11`, loc `1.000`, class `0.000`), `n03188531` (`gt=11`, loc `1.000`, class `0.000`), and `n04004767` (`gt=5`, loc `1.000`, class `0.000`). Valid-calibrated scoring does not fix these categories, and in some cases worsens class recall while loc recall stays high. Interpretation: the next detector-side work should target category assignment / class-head candidate coverage, not more post-hoc score calibration.
- High-IoU category-confusion diagnostics show the wrong-class flow directly. On 14best `top100 / IoU 0.5`, examples include `n07714571 -> n07739125` (`14/18`, mean IoU `0.894`), `n03188531 -> n07747607` (`6/11`, mean IoU `0.916`; calibrated `9/11`), and `n02799071 -> n03720891` (`5/9`, mean IoU `0.904`). These are high-overlap boxes with wrong categories, so the failure is semantic/category assignment after localization, not just missed object support.
- Split-count join reveals a protocol caveat behind the worst class failures. Among test-positive categories in the train1000 split, `2` categories have zero train boxes and `18` have fewer than `3` train boxes; among classes with `test_gt >= 3`, six have `train_gt < 3`. Several worst gaps are therefore not fair class-head architecture failures: `n02503517` has `test=11/train=0`, `n03188531` has `test=11/train=2`, `n04004767` has `test=5/train=2`, `n07714571` has `test=18/train=1`, and `n02799071` has `test=9/train=1`. Interpretation: continue treating current train1000 as a detector-side smoke benchmark, but the next serious RF-DETR protocol should be category-stratified or at least enforce minimum train coverage per evaluated class.
- Category-stratified RF-DETR split builder added: `scripts/build_stratified_rfdetr_coco_split.py` prepares `rfdetr_stratified_seed41_train1000_val200_test200_min3` with image-disjoint train/valid/test counts `1000/200/200`, preserved 200-category table, and train minimum `3` boxes per category (`zero=0`, `lt_min=0`, `min=3`). Handoff check passes with no missing files, no invalid boxes, matching category tables, and `valid_test_annotations_identical=false`. This should become the next serious RF-DETR benchmark before drawing further class-head conclusions from the random split.
- Stratified split RF-DETR Small 2ep smoke completed successfully and establishes the new protocol baseline. Independent test class `AP/AP50/AP75 = 0.0949/0.1122/0.1001`, loc `AP/AP50/AP75 = 0.4111/0.5426/0.4504`, offcenter AP50 `0.0871`, center `0.1348`, small `0.0646`, medium `0.1300`, large `0.2094`. Coverage confirms the split fix: among test-positive categories, train_zero=`0` and train_lt3=`0`, yet top100 still has loc/class/percat recall `0.8295/0.5513/0.6457`, with high loc/low class gaps remaining for trained categories. Interpretation: the new split is a fairer class-head benchmark and should replace random train1000 for future serious RF-DETR runs, but this 2ep result is only a smoke baseline and should not be compared directly against random 14best long training.
- Stronger static train teacher did not translate into a better staged student. Small-resume12 seed41 has much stronger train pseudo-label quality than Small-resume8, but the `s015/top20` teacher-only 1ep -> GT-finetune 3ep student reaches only class AP50 `0.1601`, AP75 `0.1038`, loc AP50 `0.6258`, offcenter AP50 `0.1389`, small AP50 `0.1276`, medium AP50 `0.2103`, and large AP50 `0.2542` on the independent test. This is slightly above the same-schedule seed43 Small-resume8 staged result (`0.1540`) but below seed41 Small-resume8 staged (`0.1855`) and far below direct Small 12ep (`0.3070/0.2923` across seeds). Static pseudo AP is therefore not enough to choose a teacher/filter; teacher-only pseudo-pretrain remains downgraded unless a future schedule changes how pseudo boxes are consumed.
- Small-resume8 `s025/top20` staged pseudo-pretrain is a useful but not decisive warm-start. Seed41 teacher-only 1ep is positive but weak on independent test (class AP50 `0.0547`, loc AP50 `0.3288`), and after 3 GT-finetune epochs reaches class AP50 `0.1855`, AP75 `0.1386`, loc AP50 `0.6300`, offcenter AP50 `0.1486`, small AP50 `0.1269`, and large AP50 `0.3164`. A second seed is weaker: teacher-only AP50 `0.0493`, staged 1+3 AP50 `0.1540`, AP75 `0.1035`, loc AP50 `0.6300`, offcenter AP50 `0.1342`, and small AP50 `0.1251`. Mean staged AP50 is about `0.1697`, but seed43 is below same-seed GT-only 4ep (`0.1618`). Treat stronger-teacher pseudo-pretrain as a weak-Nano warm-start diagnostic, not a robust equal-budget win and not a replacement for direct Small-resume8 (`0.2677`).
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
- Slice-weighted quality hook added: `--quality-head-slice {small,medium,large,center,offcenter}` and `--quality-head-slice-weight` upweight matched quality-head BCE targets for a robustness slice. Default behavior is unchanged.
- Offcenter quality-target weighting result: `results/det_real_no_object_w05_quality_offcenter_weight3_restore_ap50_offcenter_1000img_500step_3seed.csv` shows that `--quality-head-slice offcenter --quality-head-slice-weight 3.0` does not recover the `0.5` base offcenter benefit. Offcenter fixed AP50 only moves `0.040 -> 0.047`, slice-match `0.017 -> 0.028`, and no-GT stays high (`0.769 -> 0.767`). Stop this simple loss-weight branch.
- Restore metric hook extended to `center_ap50` and `offcenter_ap50`.
- Offcenter-restore result: `results/det_real_no_object_w05_quality_restore_offcenter_ap50_offcenter_1000img_500step_3seed.csv` shows that restoring by offcenter AP does not solve the suppression. AP50 drops to `0.123` vs base `0.142`, offcenter fixed AP is only `0.050` vs base `0.101`, and offcenter slice-match remains low (`0.019`). It does improve fixed AP75 (`0.013 -> 0.021`), but this is not enough to recover offcenter AP. Stop checkpoint-metric tweaks for this issue.
- Late quality-start result: `results/det_real_no_object_w05_quality_late501_restore_ap50_offcenter_1000img_600step_3seed.csv` tests letting the `0.5` base detector train through step 500 before freezing and training quality at step 501. It reduces the geometry damage but still does not improve AP50: base AP50/offcenter AP50/fixed offcenter AP50 are `0.161/0.080/0.080`, while quality gives `0.148/0.088/0.056`. Conclusion: quality-stage timing is not the core fix. Stop quality protocol tweaks and move to object-specific query/proposal binding.
- Querymask under the `0.5` offcenter protocol result: `results/det_real_no_object_w05_querymask_quality_restore_ap50_offcenter_1000img_500step_3seed.csv`.
- Protocol: same `0.5` offcenter stress, comparing `local_anchor_residual_query` against `local_anchor_residual_query_querymask_quality_head` with AP50 restore before the frozen quality stage.
- Result: querymask quality improves aggregate AP50 `0.127 -> 0.148` (`3/3` wins), class AP50 `0.051 -> 0.073` (`3/3`), q-fixed AP50 `0.127 -> 0.150` (`3/3`), and reduces no-GT top-k rate `0.857 -> 0.823`.
- However it hurts geometry and the actual offcenter target: final IoU `0.335 -> 0.328`, best IoU `0.342 -> 0.329`, AP75 `0.020 -> 0.009`, offcenter AP50 `0.056 -> 0.036`, and offcenter q-fixed AP50 `0.056 -> 0.038`.
- Slice breakdown explains the failure: offcenter combined top-k slice-match drops `0.041 -> 0.008`, while non-slice matches rise `0.116 -> 0.215`. Querymask suppresses some no-GT candidates but shifts high-score predictions toward other GTs, not the offcenter object.
- Decision: keep querymask as an aggregate ranking / dense-coupling probe, but do not treat it as the offcenter binding fix. The active predicted-proposal route should move to lighter layerwise proposal refresh / `reinject`, not query-mask moment refinement, quality timing, or persistent proposal state.
- Layerwise proposal refresh under the `0.5` offcenter protocol result: `results/det_real_no_object_w05_reinject_quality_restore_ap50_offcenter_1000img_500step_3seed.csv`.
- Protocol: same `0.5` offcenter stress, comparing residual-anchor base, mask-proposal init-only quality, and `local_mask_proposal_nms_query_reinject_g003_quality_head`.
- Init-only proposal quality is not competitive with residual-anchor: final IoU `0.335 -> 0.317`, AP50 `0.127 -> 0.125`, AP75 `0.020 -> 0.018`, and offcenter AP50 `0.056 -> 0.023`, despite learning a dense proposal mask (`mask IoU 0.430`).
- Reinject gives a small aggregate AP50 signal (`0.127 -> 0.138`, `2/3` wins) and class AP50 signal (`0.051 -> 0.058`, `2/3`), but hurts geometry and fine localization: final IoU `0.335 -> 0.306`, best IoU `0.342 -> 0.309`, AP75 `0.020 -> 0.008`, and q-fixed AP75 `0.020 -> 0.008`.
- Offcenter remains unsolved: reinject offcenter AP50 is `0.040` versus base `0.056`, q-fixed offcenter AP50 is `0.029`, and offcenter slice-match is only `0.008` while no-GT remains high (`0.813`).
- Decision: layerwise refresh / `reinject` can remain a lightweight ranking/proposal-consumption probe, but it is not the offcenter object-binding fix under the current predicted proposal quality. The strongest active detector path remains residual-anchor with stronger background (`no_object_weight=0.5`) for base offcenter binding, while future offcenter work should change candidate generation/assignment rather than proposal state, querymask moment refinement, or quality scoring protocol.
- Added `offcenter_only` slice filter for train/calibration/eval/sampling diagnostics. It selects images whose ranked GT boxes include offcenter objects but no center objects. This separates pure offcenter candidate generation from same-image center/non-slice GT competition.
- Smoke artifact: `results/det_real_offcenter_only_slice_smoke.csv`; the split is valid (`7` eval images in a 200-sample smoke).
- Formal `offcenter_only` diagnostic artifact: `results/det_real_no_object_w05_quality_restore_ap50_offcenter_only_1000img_500step_3seed.csv`.
- Important scope caveat: `offcenter_only` is computed after top-class filtering and `max_objects` ranking. It means retained detector GTs have offcenter objects and no center objects; it does not mean the original XML image has no center object. In the formal split, raw annotations still contain center objects in `20/33`, `14/31`, and `9/21` eval images for seeds `41/42/43`, respectively.
- Result: the slice is much harder than the mixed offcenter eval. Base residual-anchor drops to final IoU `0.250`, AP50 `0.037`, class AP50 `0.029`, AP75 `0.015`, and top-k FP `0.969`.
- Frozen quality gives only a loose AP50 gain (`0.037 -> 0.050`, `2/3` wins) while hurting AP75 (`0.015 -> 0.000`) and keeping top-k FP saturated (`0.970`). Fixed-quality AP50 is `0.047`, and IoU-reference AP50 is only `0.130`.
- Interpretation: offcenter failure is not primarily same-image center/non-slice competition. Even when center GTs are absent, the detector often cannot generate/rank high-quality offcenter candidates. The next useful control should test candidate generation capacity itself, such as higher input/feature resolution, before adding more scoring heads.
- Higher-resolution candidate-generation control: `results/det_real_no_object_w05_image96_offcenter_only_1000img_500step_3seed.csv`.
- Protocol: same base residual-anchor detector and `offcenter_only` split, but `--image-size 96` instead of `64`.
- Result: higher input/feature resolution does not recover offcenter-only AP. AP50 is essentially unchanged (`0.037 -> 0.038`), final IoU slightly drops (`0.250 -> 0.245`), AP75 drops (`0.015 -> 0.002`), and top-k FP remains saturated (`0.972`).
- Deeper local-state capacity control: added `--local-blocks` CLI and ran `results/det_real_no_object_w05_localblocks4_offcenter_only_1000img_500step_3seed.csv`.
- Result: `local_blocks=4` is negative on this slice: final IoU `0.237`, AP50 `0.029`, class AP50 `0.025`, AP75 `0.013`, and top-k FP `0.977`.
- Decision: offcenter-only failure is not solved by simple input resolution or local-depth scaling. Stop these capacity-only controls unless a qualitative inspection shows an obvious data/transform artifact. Next step should inspect offcenter-only predictions/overlays to see whether boxes are center-biased, wrong-scale, or class/ranking failures before adding another architecture.
- Added `scripts/visualize_det_real_predictions.py` to train a tiny real-DET base model and save GT-vs-top-prediction overlays for fixed eval samples.
- Smoke artifact: `results/det_real_offcenter_only_overlay_smoke/contact_sheet.jpg`.
- Formal qualitative artifact: `results/det_real_offcenter_only_overlay_w05_base_seed41_500step/contact_sheet.jpg`.
- Overlay panels now distinguish raw XML GT from retained detector GT, which is necessary because `offcenter_only` is label-space filtered rather than raw-image pure.
- Qualitative read: high-score predictions are often large, cluttered, and shifted to the right/right-bottom image region rather than overlapping the offcenter GT. This supports the quantitative top-k FP result: the failure is candidate geometry / query spatial prior, not just score calibration. The next model change should directly alter box/query spatial parameterization or assignment, not quality alpha, temperature, persistent proposal state, querymask moment interpolation, local depth, or input resolution.
- Spatial query-seed control on `offcenter_only`: `results/det_real_no_object_w05_grid_residual_offcenter_only_1000img_500step_3seed.csv`.
- Result: regular coarse-grid query seeding does not solve the issue. Relative to residual-anchor, `local_grid_residual_query` has lower final IoU (`0.243 -> 0.236`), lower AP50 (`0.061 -> 0.029`, `0/3` wins), lower class AP50 (`0.045 -> 0.011`), and higher top-k FP (`0.955 -> 0.983`), although best IoU is slightly higher (`0.248 -> 0.252`, `3/3`).
- Decision: simple spatial query seeding is not enough. The next candidate should change box spatial parameterization itself, for example query/reference-box residual prediction, rather than only changing query feature initialization.
- Grid reference-box residual controls on `offcenter_only`:
  - `results/det_real_no_object_w05_grid_box_residual_offcenter_only_1000img_500step_3seed.csv`
  - `results/det_real_no_object_w05_grid_box_residual_q9_offcenter_only_1000img_500step_3seed.csv`
- Implementation note: fixed a `grid_query_indices()` device bug while adding this control, and changed grid reference coordinates to normalized feature-cell centers `(idx + 0.5) / size`.
- q=6 result: hard grid reference center residual improves AP50 (`0.061 -> 0.081`, `2/3` wins) and reduces top-k FP (`0.955 -> 0.900`), but lowers final IoU (`0.243 -> 0.203`, `0/3` wins), best IoU (`0.248 -> 0.211`), and class AP50 (`0.045 -> 0.034`). This is candidate/ranking signal, not a localization fix.
- q=9 coverage control: the AP50 signal persists (`0.013 -> 0.030`, `3/3` wins; class AP50 `0.007 -> 0.014`), but geometry still drops (final IoU `0.249 -> 0.234`, best IoU `0.269 -> 0.258`). Therefore the q=6 result is not only a two-column grid coverage artifact.
- Soft reference gate control: `results/det_real_no_object_w05_grid_box_soft_residual_offcenter_only_1000img_500step_3seed.csv`.
- Result: low-interference reference-center gating flips the trade-off. It preserves/slightly improves geometry (final IoU `0.243 -> 0.253`, `2/3` wins; best IoU `0.248 -> 0.255`, `2/3`), but loses AP50 (`0.061 -> 0.044`, `0/3` wins), class AP50 (`0.045 -> 0.020`), and keeps top-k FP saturated (`0.955 -> 0.983`).
- Decision: ad-hoc grid reference boxes expose a real AP/geometry tension but do not solve offcenter object binding. Hard reference centers improve candidate AP while damaging geometry; soft reference centers preserve geometry while losing ranking. Stop this path as an active fix. The next detector step should be a principled reference-box decoder/assignment design (DAB-style iterative reference boxes or explicit query-target binding), or a real detector baseline, not another hand-tuned grid/reference heuristic.
- Minimal DAB-style reference-box decoder control: `results/det_real_no_object_w05_anchor_refbox_residual_offcenter_only_1000img_500step_3seed.csv`.
- Implementation: added `local_anchor_refbox_residual_query`, which keeps residual-anchor query features but gives each query a learned reference box. The box head predicts logit-space deltas using `sigmoid(delta + inverse_sigmoid(reference_box))` rather than post-hoc center blending.
- Result: learned reference boxes show a small geometry signal but worsen ranking/class binding. Best IoU improves (`0.248 -> 0.261`, `2/3` wins), but final IoU is flat/slightly lower (`0.243 -> 0.241`), AP50 drops (`0.061 -> 0.046`, `1/3` wins), class AP50 drops (`0.045 -> 0.023`, `0/3` wins), and top-k FP worsens (`0.955 -> 0.983`).
- Reference-aware matching control: `results/det_real_no_object_w05_anchor_refbox_refmatch_offcenter_only_1000img_500step_3seed.csv`.
- Implementation: added `local_anchor_refbox_residual_query_refmatch`, which keeps the same learned reference-box decoder but adds reference-box L1 distance to the Hungarian matching cost. This tests explicit query-target binding without adding a decoder/head.
- Result: simple reference-aware matching is strongly negative. It keeps only a small best-IoU signal (`0.248 -> 0.257`, `2/3` wins) but worsens final IoU (`0.243 -> 0.233`), AP50 (`0.061 -> 0.008`, `0/3` wins), class AP50 (`0.045 -> 0.004`, `0/3` wins), and top-k FP (`0.955 -> 0.991`).
- Minimal multi-layer DAB-style reference update control: `results/det_real_no_object_w05_anchor_refbox_dab_offcenter_only_1000img_500step_3seed.csv`.
- Implementation: added `local_anchor_refbox_dab_query`, which starts from residual-anchor queries, maintains learned reference boxes, runs three cross-attention decoder/update rounds, and decodes the final box as `sigmoid(delta + inverse_sigmoid(reference_box))`. Scope caveat: this is a minimal DAB-style reference-update control, not a full DAB/Deformable-DETR implementation, because the updated reference boxes are not injected back as positional attention constraints.
- Result: DAB-style updates improve geometry but worsen ranking/class binding. Final IoU improves (`0.243 -> 0.267`, `2/3` wins) and best IoU improves (`0.248 -> 0.267`, `2/3` wins), but AP50 drops (`0.061 -> 0.039`, `1/3` wins), class AP50 drops (`0.045 -> 0.025`, `1/3` wins), and top-k FP remains worse (`0.955 -> 0.980`). This repeats the same AP/geometry tension seen in grid/refbox controls.
- Current stop point: the offcenter issue is now a serious assignment/ranking-direction problem, not an implementation bug. Simple spatial query seeds, hard grid reference centers, soft grid reference gates, minimal learned reference boxes, naive reference-aware matching, and minimal multi-layer DAB-style reference updates all expose AP/geometry trade-offs without solving object-specific offcenter binding. Do not add another small reference/matcher heuristic. The next credible step is to move out of this tiny detector scaffold toward a real detector baseline or a substantially more complete DETR decoder where query assignment/ranking machinery is mature enough to test the representation idea.
- Real-detector handoff utility added: `scripts/export_det_manifest_to_coco.py` converts `det_val_manifest.csv` into COCO-style detection JSON, optionally preserving a `train_det_real.py` split CSV by `run_seed`. Smoke artifact: `results/coco_export_smoke/` produced `train.json` with `50` images / `124` annotations and `eval.json` with `33` images / `78` annotations for split seed `41`. This is the first concrete engineering step toward RF-DETR/Detectron/torchvision detector baselines after stopping tiny-detector reference heuristics.
- External-detector smoke runner added: `scripts/train_torchvision_coco_detector.py` trains/evaluates a small `torchvision` Faster R-CNN model on exported COCO JSON. CPU smoke artifact: `results/torchvision_coco_detector_smoke.csv`; MPS smoke artifact: `results/torchvision_coco_detector_mps_smoke.csv`. Both completed a 1-step train/eval loop on `results/coco_export_smoke/{train,eval}.json`. This confirms the real-detector data plumbing works on the local AIAA environment, including MPS. Next real-detector work should scale this runner or replace it with RF-DETR once the package/weights are available; do not resume tiny-detector reference heuristics as the main path.
- External-detector 20-step MPS sanity: `results/torchvision_coco_detector_mps_20step.csv` trains on `20` exported COCO images and evaluates on `10`. The run is stable (`loss 3.2900 -> 1.0110`), but AP remains near zero, which is expected for no-pretraining Faster R-CNN on tiny data. Treat this as a pipeline sanity check, not a performance baseline. The meaningful next step is to attach a pretrained/full detector such as RF-DETR or a pretrained torchvision detector through the same COCO interface.
- Pretrained-handoff hook added: `scripts/train_torchvision_coco_detector.py --weights coco` now loads torchvision COCO Faster R-CNN weights and replaces the box predictor for the exported dataset category count. Default remains `--weights none`, so normal smoke runs do not download weights. Local cache check found no Faster R-CNN weights yet, so this path is ready for later download/manual cache rather than executed as a formal result.
- Local pretrained-weight path added: `--weights-file /path/to/checkpoint.pth` loads a manually downloaded torchvision checkpoint before replacing the predictor, and `--steps 0` runs eval-only. The official torchvision Faster R-CNN MobileNetV3-320 checkpoint was downloaded to `/private/tmp/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth` for local smoke only. Eval-only MPS artifact: `results/torchvision_coco_detector_weightsfile_evalonly_smoke.csv` (`eval_iou 0.205`, AP50 `0.004`). A 20-step MPS finetune smoke (`results/torchvision_coco_detector_weightsfile_mps_20step.csv`) completed after fixing an evaluator `KeyError` on images with zero predictions, but AP collapsed from `0.027` at step 1 to `0.001` at step 20 after replacing the predictor. Treat this as a plumbing and transfer-sanity result, not a performance claim. Meaningful pretrained detector comparison needs more data/steps and likely a proper train schedule.
- Transfer-freeze hook added: `--trainable-parts {all,roi_heads,box_predictor}` supports freezing the backbone/RPN for external-detector transfer checks. Head-only pretrained smoke artifact: `results/torchvision_coco_detector_weightsfile_boxpredictor_mps_20step.csv`; it trains only `56,375` parameters and runs cleanly, but AP still collapses (`0.028 -> 0.001`). This confirms the freeze path works, but the tiny 20-image/20-step split is not a valid performance protocol for pretrained detectors. Next meaningful external-detector experiment should use the full exported split and a longer schedule, or keep this runner as data plumbing until RF-DETR is available.
- ROI-head pretrained smoke on the larger exported split: `results/torchvision_coco_detector_weightsfile_roiheads_fast_smoke.csv` uses `50` train images, `5` eval images, 96px, 5 steps, and pretrained weights. It runs cleanly and loss decreases (`3.1724 -> 2.2268`), but AP remains near zero (`0.001 -> 0.004`). A larger 128px/full-eval run was stopped because it took about 150s to reach step 1. Added eval controls `--detections-per-img` and `--score-threshold` to keep future torchvision smokes practical. Current interpretation remains: the external-detector plumbing works, but meaningful performance needs a real schedule/pretrained detector protocol, not tiny smoke settings.
- External-detector staged-run support added: `--save-checkpoint`, `--resume-checkpoint`, and `--predictions-out` now persist model/optimizer state and COCO-format detection outputs. CLI smoke artifacts:
  - `results/torchvision_checkpoint_smoke_train.csv`
  - `results/torchvision_checkpoint_smoke.pt`
  - `results/torchvision_checkpoint_smoke_train_predictions.json`
  - `results/torchvision_checkpoint_smoke_eval.csv`
  - `results/torchvision_checkpoint_smoke_eval_predictions.json`
- Implementation note: the first resume attempt exposed a PyTorch 2.6+ `weights_only=True` checkpoint loading issue because `Path` objects were stored in `args`. Fixed by serializing checkpoint args as JSON-like scalars instead of disabling safe loading. This is now covered by `tests/test_torchvision_coco_detector.py`.
- External-detector summary utility added: `scripts/summarize_torchvision_detector_results.py` summarizes CSV curves and COCO prediction JSONs. Smoke summary confirms the current external-detector state: pretrained eval-only gives the best smoke IoU (`0.205`) while short finetunes are unstable (`best AP50 0.039` but final AP50 `0.001` for the 20-step all-parameter run). Use this summary script for future external-detector stage gates instead of hand-reading CSVs.
- External-detector COCOeval handoff tightened: `scripts/train_torchvision_coco_detector.py --predictions-out` now exports boxes in original image coordinates instead of resized smoke coordinates, and `scripts/evaluate_coco_predictions.py` runs standard pycocotools COCOeval on those predictions. Internal fast metrics still use resized boxes for cheap smoke checks; official external-detector comparisons should use the COCO JSON + COCOeval path.
- `scripts/summarize_torchvision_detector_results.py` now accepts `--cocoeval` CSVs as well as runner CSVs and prediction JSONs, so future external-detector stage gates can report internal smoke metrics, prediction counts, and standard COCO AP from one summarizer.
- Checkpoint resume tightened after review: `scripts/train_torchvision_coco_detector.py` now stores and reloads the global training step, so resumed staged runs continue step numbering instead of restarting CSV rows at step `1`.
- COCO category-id handling tightened: the torchvision runner now remaps COCO category ids to contiguous positive training labels internally and maps predictions back to original `category_id` values for JSON export. It also rejects train/eval JSONs with inconsistent category mappings to avoid silent label drift.
- Standard COCOeval handoff smoke with local torchvision COCO weights:
  - runner CSV: `results/torchvision_weightsfile_cocoeval_handoff.csv`
  - predictions: `results/torchvision_weightsfile_cocoeval_handoff_predictions.json`
  - COCOeval CSV: `results/torchvision_weightsfile_cocoeval_handoff_cocoeval.csv`
  - Setting: local MobileNetV3-320 Faster R-CNN weights, replaced predictor, eval-only, 10 eval images, 96px, 20 detections/image, MPS.
  - Result: internal resized-coordinate smoke metrics `eval_iou=0.184`, `eval_ap50=0.014`, class-aware `0.001`; standard COCOeval original-coordinate metrics `ap=0.0045`, `ap50=0.0061`, `ap75=0.0055`. This is still a handoff sanity result, not a detector performance claim, but it confirms the COCO prediction export and pycocotools evaluation pipeline are now wired end to end.
- External-detector slice utility added: `scripts/filter_coco_annotations.py` creates filtered COCO annotation JSONs for `offcenter`, `center`, and small/medium/large object area-ratio slices. Smoke artifact: `results/coco_export_smoke/eval_offcenter.json` (`33` images / `72` annotations at center-radius `0.25`), evaluated with the same predictions at `results/torchvision_weightsfile_cocoeval_handoff_offcenter_cocoeval.csv`. This gives the external-detector path the same offcenter/size robustness hooks that drove the tiny-detector stop decision.
- Multi-slice COCOeval added: `scripts/evaluate_coco_slices.py` runs `all/offcenter/center/small/medium/large` slice evaluation for one prediction JSON and writes a single CSV. Smoke artifact: `results/torchvision_weightsfile_cocoeval_handoff_slices.csv`. The script filters predictions per slice image set before calling pycocotools, because `COCO.loadRes` rejects predictions for images absent from the current slice annotation.
- Review tightening: empty-prediction COCOeval now still validates annotation JSON; protocol summaries include per-slice `images` and `annotations` counts so empty slices are not confused with real AP=0. Slice metrics should be read as slice-target COCOeval: predictions are filtered by image id, not by object identity, so same-image non-slice detections can count as false positives for that slice.
- External-detector protocol stage-gate added: `scripts/evaluate_external_detector_protocol.py` takes COCO annotations, prediction JSON, and optional runner CSV, then writes standard COCOeval, slice COCOeval, and a one-line summary CSV. Smoke artifacts:
  - `results/torchvision_weightsfile_cocoeval_protocol_cocoeval.csv`
  - `results/torchvision_weightsfile_cocoeval_protocol_slices.csv`
  - `results/torchvision_weightsfile_cocoeval_protocol_summary.csv`
  - Summary row for the pretrained torchvision eval-only smoke: `coco_ap50=0.0061`, `offcenter_ap50=0.0061`, `center_ap50=0.0`, `medium_ap50=0.0266`. Treat these as protocol sanity values, not model performance claims.
- External-detector comparison helper added: `scripts/compare_external_detector_protocols.py` merges one-line protocol summaries and optionally computes deltas against a named reference. Smoke artifact: `results/torchvision_weightsfile_cocoeval_protocol_comparison.csv`.
- More realistic external-detector eval-only protocol split exported from `results/det_real_train_offcenter_1000img_500step_3seed_split.csv` seed `41`:
  - COCO split: `data/ILSVRC2013_DET_val_supervised/coco_offcenter_seed41/` with train `330` images / `1610` boxes and eval `200` images / `684` boxes, `200` classes.
  - Eval-only runner artifacts: `results/torchvision_offcenter_seed41_evalonly.csv`, `results/torchvision_offcenter_seed41_evalonly_predictions.json`.
  - Protocol artifacts: `results/torchvision_offcenter_seed41_evalonly_protocol_cocoeval.csv`, `results/torchvision_offcenter_seed41_evalonly_protocol_slices.csv`, `results/torchvision_offcenter_seed41_evalonly_protocol_summary.csv`, `results/torchvision_offcenter_seed41_evalonly_protocol_comparison.csv`.
  - Result: internal fast metrics `eval_iou=0.342`, `eval_ap50=0.034`, class-aware AP50 approximately `0`; standard COCOeval is much stricter (`coco_ap=0.00024`, `coco_ap50=0.00061`). Slice protocol reports offcenter AP50 `0.00107`, center AP50 `0`, small AP50 `0.00124`, medium/large AP50 `0`. Treat this as a protocol baseline, not a serious pretrained detector result, because the COCO pretrained head is replaced for 200 ILSVRC classes and not trained.
- RF-DETR local package check (superseded by the controlled install notes below): the first AIAA check had `transformers` but not `rfdetr`, `roboflow`, `supervision`, or `timm`, and no local RF-DETR repo/files were found under this project. The stable interface remains COCO JSON + prediction JSON + protocol stage-gate; RF-DETR should plug into that by producing COCO-format predictions.
- Torchvision finetune ergonomics: `scripts/train_torchvision_coco_detector.py` now supports `--no-eval-first-step` so real finetune runs can avoid an expensive first-step full eval and instead evaluate at `--eval-every` plus the final step.
- Torchvision real finetune baseline run:
  - Training artifacts: `results/torchvision_offcenter_seed41_roiheads_100step_train.csv`, `results/torchvision_offcenter_seed41_roiheads_100step.pt`.
  - Full eval artifacts: `results/torchvision_offcenter_seed41_roiheads_100step_eval.csv`, `results/torchvision_offcenter_seed41_roiheads_100step_predictions.json`.
  - Protocol artifacts: `results/torchvision_offcenter_seed41_roiheads_100step_protocol_summary.csv`, `results/torchvision_offcenter_seed41_evalonly_vs_roiheads_100step_comparison.csv`.
  - Setting: offcenter seed41 COCO split, local MobileNetV3-320 Faster R-CNN weights, replaced 200-class predictor, `roi_heads` trainable, 100 steps, image size 96, MPS, 50-image intermediate eval and 200-image final eval.
  - Result: this is a negative/weak baseline. Intermediate internal metrics dropped from step 50 to 100 (`IoU 0.246 -> 0.187`, `AP50 0.013 -> 0.007` on 50-image eval). Full 200-image eval after checkpoint: internal `IoU=0.231`, `AP50=0.011`, class-aware AP50 `0.00088`, worse than eval-only internal `IoU=0.342`, `AP50=0.034`. Standard COCOeval versus eval-only: `coco_ap50` changes `0.00061 -> 0.00085` but offcenter AP50 drops `0.00107 -> 0.00014`; center and large slices improve from zero but remain tiny. Interpretation: short ROI-head finetune with replaced head is not a meaningful positive baseline and should not be scaled blindly. Next serious baseline needs a proper pretrained detector schedule/RF-DETR path, not more tiny-step torchvision tuning.
- Torchvision low-interference box-predictor-only finetune control:
  - Training artifacts: `results/torchvision_offcenter_seed41_boxpredictor_100step_train.csv`, `results/torchvision_offcenter_seed41_boxpredictor_100step.pt`.
  - Full eval/protocol artifacts: `results/torchvision_offcenter_seed41_boxpredictor_100step_eval.csv`, `results/torchvision_offcenter_seed41_boxpredictor_100step_predictions.json`, `results/torchvision_offcenter_seed41_boxpredictor_100step_protocol_summary.csv`.
  - Combined comparison: `results/torchvision_offcenter_seed41_finetune_comparison.csv`.
  - Result: also negative. 50-image internal metrics are weak (`step50 IoU=0.216/AP50=0.005`, `step100 IoU=0.224/AP50=0.012`). Full 200-image eval: internal `IoU=0.241`, `AP50=0.0105`, class-aware AP50 `0.00165`; standard `coco_ap50=0.00019`, worse than eval-only `0.00061`; offcenter AP50 drops to `0.000027` versus eval-only `0.00107`. The high `max_score≈1.0` with near-zero AP suggests poor calibration/ranking after training the replaced predictor. Conclusion: both ROI-head and box-predictor-only 100-step torchvision finetunes fail to establish a meaningful positive baseline. Treat torchvision as plumbing/protocol baseline; serious finetune should use a proper detector schedule or RF-DETR-style model.
- Protocol now reports class-agnostic localization COCOeval (`loc_*`) in addition to class-aware `coco_*`. This separates category-head failure from proposal/localization quality. Recomputed offcenter seed41 summaries show eval-only `loc_ap50=0.0384`, while ROI-head 100-step drops to `0.0118` and box-predictor-only drops to `0.0111`; comparison deltas are about `-0.0266` and `-0.0274`. Thus the 100-step torchvision finetunes do not merely fail classification; they also damage localization/proposal ranking under class-agnostic scoring.
- Random-weight torchvision eval-only control was attempted on the offcenter seed41 protocol but was stopped because even 50-image eval did not finish promptly. This is an engineering cost signal rather than a result. Do not spend more time on random Faster R-CNN full eval; the useful baseline is COCO-pretrained eval-only versus finetuned checkpoints under the same protocol.
- Proposal recall diagnostic added: `scripts/evaluate_torchvision_proposal_recall.py` evaluates raw torchvision RPN proposals before ROI classification. On the offcenter seed41 200-image eval split with COCO-pretrained MobileNetV3-320 RPN/backbone, top-50 class-agnostic proposal recall is much higher than final detector AP (`all recall@0.5=0.450`, `mean_best_iou=0.424`), but the slice split reveals the real weakness: `offcenter recall@0.5=0.243`, `small recall@0.5=0.069`, while `center recall@0.5=0.649` and `large recall@0.5=0.913`. Raising the budget to top-300 barely changes recall (`all 0.455`, `offcenter 0.249`, `small 0.075`), so this is not just a top-k truncation issue. Interpretation: the external-detector handoff bottleneck is not only the 200-class ROI head/ranking; proposal geometry itself is biased toward centered and larger objects. Next verification should prioritize proposal generator/offcenter representation quality over more ROI-head finetuning.
- Deterministic anchor-grid oracle recall added: `scripts/evaluate_anchor_grid_recall.py` tests whether the same 96x96 geometry can cover GT boxes without relying on learned RPN scores/regression. On the same offcenter seed41 eval split, dense anchors reach `all recall@0.5=0.836`, `offcenter=0.778`, `small=0.644`, `medium=0.985`, `large=1.000` with `mean_best_iou=0.704`. This is far above RPN top-300 (`all=0.455`, `offcenter=0.249`, `small=0.075`), so the weak offcenter/small behavior is not simply caused by low image resolution or impossible anchor geometry. It points to proposal scoring/regression/domain-transfer failure in the pretrained torchvision RPN path.
- Input-size proposal recall sweep: increasing torchvision detector size from `96 -> 128 -> 160` improves RPN top-300 recall@0.5 from `all 0.455 -> 0.556 -> 0.613`, `offcenter 0.249 -> 0.386 -> 0.467`, and `small 0.075 -> 0.163 -> 0.225`. Resolution helps and should be used for serious external-detector handoff, but even 160x160 remains far below the 96x96 anchor-grid oracle (`offcenter 0.778`, `small 0.644`). Treat larger resolution as a necessary engineering baseline, not a complete solution.
- Full torchvision ROI eval at 160x160 was attempted for both 200-image and 50-image eval-only checks but was stopped after exceeding a reasonable smoke window without producing metrics. Keep resolution sweeps on fast proposal-recall diagnostics unless a proper long-running external-detector evaluation budget is explicitly allocated.
- `--trainable-parts rpn` support was added for future proposal-generator adaptation tests, but an RPN-only 50-step 128x128 smoke was stopped after exceeding a reasonable interactive window. Treat RPN-only finetuning as a long-running job, not a quick local diagnostic. The immediate reliable evidence remains: proposal recall improves with resolution but still lags far behind deterministic anchor-grid coverage.
- RPN proposal-as-detection export added: `scripts/export_torchvision_rpn_predictions.py` writes raw RPN proposals and objectness scores as COCO predictions, allowing fast class-agnostic COCOeval without slow ROI heads. On offcenter seed41, RPN-only localization AP improves with resolution (`96x96 loc ap50=0.131`, `160x160 loc ap50=0.197`), confirming that higher resolution helps proposal ranking. However, slice AP remains very weak for the actual hard cases: at `160x160`, `offcenter ap50=0.0396` and `small ap50=0.0055`, while `center ap50=0.2455` and `large ap50=0.4134`. This strengthens the current diagnosis: RPN objectness/ranking is heavily center/large-biased even when proposal recall improves.
- Proposal diagnostics summary helper added: `scripts/summarize_proposal_diagnostics.py` merges RPN recall, RPN objectness AP, and anchor-grid oracle rows into one report (`results/proposal_diagnostics_offcenter_seed41_summary.csv`). This is the canonical table for the current proposal-generator diagnosis.
- RPN oracle-IoU rescoring diagnostic added: `scripts/rescore_coco_predictions_by_oracle_iou.py` replaces proposal objectness with max GT IoU for eval-only upper-bound diagnosis. On 160x160 RPN proposals, class-agnostic AP50 improves from `0.197` to `0.313`, showing that RPN objectness/ranking is a real bottleneck. But hard slices remain weak even under oracle rescoring: `offcenter ap50=0.102`, `small ap50=0.015` at 160x160. Thus ranking calibration alone cannot solve the handoff; proposal box quality/coverage for offcenter/small objects is also insufficient. The same pattern holds at 96x96 (`all ap50 0.131 -> 0.229`, `offcenter 0.026 -> 0.044`, `small 0.001 -> 0.004`).
- RF-DETR route switch started:
  - Official RF-DETR docs require a `train/valid/test` COCO directory, each with `_annotations.coco.json` and image files. Added `scripts/prepare_rfdetr_dataset.py` to convert the existing COCO split into that structure using symlinks/copies/hardlinks.
  - Prepared local artifact: `data/ILSVRC2013_DET_val_supervised/rfdetr_offcenter_seed41/` with train `330` images / `1610` annotations and valid/test `200` images / `684` annotations each. Test currently mirrors valid because no separate test split was provided.
  - Added `scripts/train_rfdetr_coco.py`, a thin RF-DETR runner that validates the dataset structure and calls `RFDETR{Nano,Small,Medium,Large,Base}.train(...)` when `rfdetr` is installed.
  - `scripts/prepare_rfdetr_dataset.py` now supports `--max-train-images`, `--max-valid-images`, and `--max-test-images` so mini RF-DETR protocols can be prepared without editing source COCO JSONs.
  - Earlier AIAA environment check showed only `torch` and `transformers` available; `rfdetr`, `timm`, `supervision`, and `roboflow` were absent before the controlled install below.
  - Initial `pip install rfdetr` attempt in AIAA began resolving/downloading `rfdetr-1.6.5.post2`, `transformers>=5.1`, `huggingface-hub>=1.5`, `supervision`, `peft`, and `accelerate`, but stalled for several minutes during dependency download and was stopped. That immediate post-check still showed `rfdetr=False`, `supervision=False`, `timm=False`.
  - Prepared RF-DETR dataset sanity check: train/valid/test have `200` contiguous category ids, no category drift, and sampled image symlinks resolve. `evaluate_coco_predictions.py` accepts the valid annotation JSON with an empty prediction file, so the stage-gate evaluator is compatible with the RF-DETR annotation layout.
  - Added `scripts/check_rfdetr_handoff.py` to report RF-DETR dependency availability and prepared dataset validity. The first artifact confirmed the dataset was ready but `rfdetr`, `timm`, and `supervision` were absent; the current artifact has since been refreshed after the controlled install below.
- RF-DETR controlled local install update:
  - Minimal AIAA install now reaches `import rfdetr` and `import supervision` after installing `rfdetr --no-deps`, replacing the wrong `deprecate` package with `pyDeprecate==0.7.0`, and adding `peft`, `accelerate`, `supervision`, `albumentations`, `faster-coco-eval`, `protobuf>=5.28,<6`, `transformers==5.8.1`, `huggingface-hub>=1.5`, and `regex>=2025.10.22`.
  - `scripts/train_rfdetr_coco.py --check-only` now validates the prepared dataset and lists available RF-DETR classes without starting training. The current local check exposes `RFDETRNano`, `RFDETRSmall`, `RFDETRMedium`, `RFDETRLarge`, and `RFDETRBase`.
  - `scripts/train_rfdetr_coco.py` now exposes `--num-classes` and fail-fast checks it against the train split category count. The prepared ILSVRC handoff should pass `--num-classes 200` to make the intended task count explicit; RF-DETR may still print a construction-time 90-class checkpoint warning before the training module aligns to the dataset.
  - Added `scripts/predict_rfdetr_coco.py` to export RF-DETR detections into standard COCO prediction JSON from any compatible checkpoint/pretrain weights. This plugs RF-DETR outputs into the existing `evaluate_coco_predictions.py` and slice-evaluation protocol instead of creating a separate evaluator. Tiny smoke: one-image RF-DETR Nano prediction export wrote `/private/tmp/rfdetr_tiny_smoke_predictions.json`, and `evaluate_coco_predictions.py` accepted it, producing `/private/tmp/rfdetr_tiny_smoke_cocoeval.csv` with zero AP as expected for raw smoke predictions.
  - `results/rfdetr_offcenter_seed41_handoff_check.json` now performs real import smoke checks and confirms `rfdetr`, `torch`, `torchvision`, `transformers`, `supervision`, `albumentations`, `faster_coco_eval`, `pytorch_lightning`, and `torchmetrics` are importable; `timm` remains absent. The dataset check now scans all split image files and validates annotation image ids plus bbox area/bounds.
  - Caveat: this install upgraded shared packages (`protobuf`, `transformers`, `huggingface-hub`) and has known dependency conflicts with unrelated packages in AIAA, especially `grpcio-status` and `descript-audiotools`. Treat AIAA as temporarily RF-DETR-oriented until the environment is cleaned or cloned.
  - `RFDETRNano()` has downloaded and MD5-validated `rf-detr-nano.pth` (`349MB`). A tiny 1-epoch MPS train smoke on `/private/tmp/rfdetr_tiny_smoke` completed and saved `/private/tmp/rfdetr_tiny_smoke_out3/checkpoint_best_regular.pth` after adding `albumentations` and `faster-coco-eval`. A second tiny smoke with `--num-classes 200` also completed and saved `/private/tmp/rfdetr_tiny_smoke_out_numclasses200/checkpoint_best_regular.pth`. RF-DETR still prints a construction-time 90-class checkpoint warning, but the training path is not blocked. This verifies local RF-DETR Nano training starts and finishes on MPS; it is not a performance result.
  - Mini real-handoff RF-DETR smoke: prepared `/private/tmp/rfdetr_offcenter_seed41_subset20` with `20` train images / `121` annotations and `10` valid/test images / `36` annotations. A 1-epoch RF-DETR Nano MPS run at `128px` completed and saved `/private/tmp/rfdetr_offcenter_seed41_subset20_out/checkpoint_best_total.pth`, with RF-DETR's validation table reporting `mAP=0.0531`, `AP50=0.0816`, `AP75=0.0816`. Prediction export from that checkpoint produced `/private/tmp/rfdetr_offcenter_seed41_subset20_predictions.json`; `evaluate_coco_predictions.py` accepted it and wrote `/private/tmp/rfdetr_offcenter_seed41_subset20_cocoeval.csv` (`ap=0.0577`, `ap50=0.0892`, `ap75=0.0654`). Slice eval also works on the same predictions: `offcenter_ap50=0.0926`, `center_ap50=0.1000`, `small_ap50=0.0000`, `medium_ap50=0.0926`, `large_ap50=0.1667`. This is still a mini protocol smoke, but it is the first RF-DETR-trained checkpoint that flows through the repo's COCOeval and slice-eval paths.
  - Slightly larger mini RF-DETR smoke: prepared `/private/tmp/rfdetr_offcenter_seed41_subset50` with `50` train images / `281` annotations and `25` valid/test images / `90` annotations. A 1-epoch RF-DETR Nano MPS run at `128px` completed and saved `/private/tmp/rfdetr_offcenter_seed41_subset50_out/checkpoint_best_total.pth`. RF-DETR validation reported `mAP=0.0308`, `AP50=0.0475`, `AP75=0.0324`. Repo COCOeval on exported predictions reported `ap=0.0313`, `ap50=0.0422`, `ap75=0.0347`; slice eval shows the familiar hard slices remain weak (`offcenter_ap50=0.0205`, `small_ap50=0.0000`, `center_ap50=0.0656`, `large_ap50=0.1011`). This confirms the RF-DETR mini protocol scales beyond tiny smoke while preserving the offcenter/small diagnostic pressure.
  - Default-resolution sanity: the same `subset20` protocol at `384px` also trains, predicts, and evaluates cleanly. RF-DETR validation reported `mAP=0.0220`, `AP50=0.0334`, `AP75=0.0334`; repo COCOeval on exported predictions reported `ap=0.0238`, `ap50=0.0357`, `ap75=0.0357`. Slice eval: `offcenter_ap50=0.1111`, `center_ap50=0.0505`, `small_ap50=0.0000`, `medium_ap50=0.0836`, `large_ap50=0.0000`. Interpretation: 384px is runnable on MPS but is not a free win under one-epoch mini conditions; keep resolution as a protocol variable rather than assuming larger is better.
  - `subset100` RF-DETR mini stage-gate: prepared `/private/tmp/rfdetr_offcenter_seed41_subset100` with `100` train images / `510` annotations and `50` valid/test images / `177` annotations. A 1-epoch RF-DETR Nano MPS run at `128px` completed and saved `/private/tmp/rfdetr_offcenter_seed41_subset100_out/checkpoint_best_total.pth`. RF-DETR validation reported `mAP=0.0335`, `AP50=0.0512`, `AP75=0.0307`; repo COCOeval on exported predictions reported `ap=0.0439`, `ap50=0.0676`, `ap75=0.0411`. Slice eval: `offcenter_ap50=0.0686`, `center_ap50=0.0918`, `small_ap50=0.0000`, `medium_ap50=0.1068`, `large_ap50=0.1674`. The 50-image valid stage-gate keeps the same conclusion: RF-DETR mini protocol is working, but small objects remain completely unresolved after 1 epoch at 128px.
  - Small-object focused control: filtered COCO train/eval to small boxes only, then prepared `/private/tmp/rfdetr_small_subset50` with `50` train images / `190` annotations and `25` valid/test images / `119` annotations. A 1-epoch RF-DETR Nano MPS run at `384px` completed and saved `/private/tmp/rfdetr_small_subset50_384_out/checkpoint_best_total.pth`. RF-DETR validation reported `mAP=0.0083`, `AP50=0.0107`, `AP75=0.0097`; repo COCOeval on exported predictions reported `ap=0.0052`, `ap50=0.0072`, `ap75=0.0058`. Interpretation: the small-object bottleneck is not solved by simply filtering to small-object supervision plus 384px for one epoch; next work should inspect bbox pixel-scale distribution and consider tiling/cropping or higher-resolution/longer schedules rather than only resampling.
  - Bbox scale diagnostic added: `scripts/summarize_coco_bbox_stats.py` reports bbox short-side and area percentiles at target detector resolutions. On the full 200-image eval split, `44.7%` of annotations are small; median short side is `25.3px` at 128, `75.9px` at 384, and `101.2px` at 512. On the small-only eval split, median short side is `8.8px` at 128, `26.4px` at 384, and `35.2px` at 512; p10 short side is still only `9.2px` at 384. Interpretation: 384px makes median small boxes visible, but the small-object tail remains very hard; small AP=0 likely mixes pixel scale, long-tail class sparsity, and one-epoch undertraining rather than being a pure resolution issue.
  - Small-object crop/tiling diagnostic added: `scripts/crop_coco_around_boxes.py` creates context crops centered on selected boxes and rewrites COCO boxes into crop coordinates. This is the first concrete small-object rescue path after plain resolution/resampling failed. Crop stats on the eval crop set show median short side improves to `62.5px` at 384 and p10 to `22.5px`.
  - Crop-focused RF-DETR smoke: generated `/private/tmp/rfdetr_small_crops_subset50` from 50 train crops / 458 annotations and 25 valid crops / 236 annotations. A 1-epoch RF-DETR Nano MPS run at `384px` completed and saved `/private/tmp/rfdetr_small_crops_subset50_384_out/checkpoint_best_total.pth`. Repo COCOeval on exported predictions reported `ap=0.0266`, `ap50=0.0629`, `ap75=0.0202`; slice eval on crop coordinates reported `small_ap50=0.0400`, `medium_ap50=0.1429`, `large_ap50=0.2131`, `offcenter_ap50=0.0428`, `center_ap50=0.0985`. Interpretation: crop/tiling has a real early signal for the small-object route, unlike small-only original-image training (`ap50=0.0072`). The caveat is that crop coordinates change area bins, so this should be treated as a tiling-protocol result, not directly comparable to full-image small AP.
  - Crop-to-source remap support added: crop images now preserve `source_image_id`, `source_file_name`, and `crop_box`; `scripts/remap_crop_predictions_to_source_coco.py` maps crop-coordinate predictions back to source-image COCO coordinates. `--max-crops-per-image` prevents crop eval from over-concentrating on a few source images.
  - Diversified small-crop RF-DETR check: generated `/private/tmp/rfdetr_small_crops_diverse_subset50` with 50 train crops from 50 source images and 25 valid crops from 25 source images (`135` valid annotations). At `384px`, eval-crop median short side is `50.9px`, p10 `21.0px`. A 1-epoch RF-DETR Nano MPS run saved `/private/tmp/rfdetr_small_crops_diverse_subset50_384_out/checkpoint_best_total.pth`.
  - Diversified crop-coordinate COCOeval is weaker than the earlier concentrated crop smoke: `ap=0.0159`, `ap50=0.0284`, `ap75=0.0158`; crop-coordinate slice AP50 is `small=0.0125`, `medium=0.0536`, `large=0.1108`, `offcenter=0.0191`, `center=0.0412`. This suggests the earlier crop-coordinate AP50 `0.0629` was partly helped by concentrated/easier crop coverage.
  - Source-coordinate remap is the stricter criterion. On the 25 covered source images, class-aware remap AP50 is only `0.0127`, but class-agnostic remap AP50 is `0.1527`. On the full small-valid slice, class-aware AP50 is `0.0043` and class-agnostic AP50 is `0.0313`. Interpretation: diversified crop/remap does produce some usable localization candidates on covered images, but category scoring/ranking and source-image coverage remain the bottlenecks. The next tiling step should not just train on crops; it should implement multi-crop/full-image fusion with NMS plus class-agnostic and class-aware reporting.
  - Tiling fusion diagnostic added: `scripts/fuse_coco_predictions.py` fuses one or more COCO prediction JSONs with score filtering, per-image NMS, and a class-agnostic NMS mode. Standard per-class NMS does not change the diversified remap result materially: covered-source class-aware AP50 stays `0.0127` and class-agnostic AP50 stays about `0.1525`.
  - Class-agnostic NMS is much more revealing. On the 25 covered source images, class-agnostic remap AP50 rises from `0.1527` to `0.3607`; slice AP50 is `offcenter=0.2823`, `center=0.1798`, `small=0.2000`, `medium=0.3270`, `large=0.1949`. On the full small-valid slice, class-agnostic AP50 rises from `0.0313` to `0.0787`. Class-aware AP50 remains `0.0127`, so the crop/tiling path currently has a real localization signal but a severe category/score-ranking problem.
  - Full-coverage small-valid crop inference confirms the same pattern at the proper source-image coverage. Using one crop for each of the 69 small-valid source images yields 20,550 raw crop predictions. After remap, class-agnostic AP50 is `0.0730` without fusion and `0.2002` after class-agnostic NMS; slice AP50 is `offcenter=0.1517`, `center=0.1039`, `small=0.2002`. Class-aware AP50 after the same class-agnostic NMS is only `0.0030`.
  - Category/score oracle split: on the same full-coverage class-agnostic-NMS boxes, relabeling each prediction to its nearest GT category while keeping original scores raises class-aware AP50 from `0.0030` to `0.3066`. Relabeling and replacing scores with nearest-GT IoU raises AP50 further to `0.6297`; the class-agnostic IoU-score upper bound is `0.5620`. This separates the bottleneck: crop/tiling boxes have substantial localization potential, category prediction is the largest current failure, and score-IoU ranking is the second failure.
  - Image-level category-prior diagnostic: relabeling all fused crop boxes in each source image to the largest-GT category raises class-aware AP50 to `0.2185`; using the most frequent GT category gives `0.1984`. This is not a deployable result because it uses GT categories, but it shows that a coarse image/region classifier could recover much of the class-aware AP if paired with the current crop-localization boxes.
  - Non-oracle lightweight image-prior classifier check: `scripts/train_coco_image_prior_classifier.py` trains a tiny image-level classifier on the RF-DETR train split and relabels fused crop boxes. On small-valid, the 300-step 96px MPS run reaches image-prior top-1 `0.1014` and top-5 `0.3768`, but class-aware detection AP50 is only `0.0023`, essentially no better than the original `0.0030`. Conclusion: category transfer is promising only with a strong classifier/teacher; a tiny classifier trained from the small local split is not enough.
  - Pretrained ImageNet category-transfer diagnostic added: `scripts/relabel_coco_predictions_by_torchvision_imagenet.py` maps torchvision ResNet18 ImageNet outputs to ILSVRC synset categories and relabels fused crop boxes. Using cached ResNet18 weights on full small-valid gives image-prior top-1 `0.2029` and class-aware AP50 `0.0886`. This is a real non-oracle gain over original AP50 `0.0030`, but still far below GT image-prior AP50 `0.2185` and nearest-GT category AP50 `0.3066`.
  - Crop-level ResNet18 prior is not stronger than full-image prior. Running the same cached ResNet18 on the full-coverage crop images and mapping predictions back through `source_image_id` gives crop-prior top-1 `0.0725` and detection AP50 `0.0904`, essentially tied with full-image ResNet18 AP50 `0.0886`. Conclusion: simple ResNet18 category transfer is helpful but not enough; next category teacher should be stronger or detector-aware rather than just crop-level.
  - Stronger torchvision teacher hook added: `scripts/relabel_coco_predictions_by_torchvision_imagenet.py --model` now supports `resnet18`, `resnet50`, `mobilenet_v3_large`, `efficientnet_b0`, and `convnext_tiny`, and the dataset preprocessing now follows the selected model instead of being hard-coded to ResNet18. ResNet50 full-image category transfer improves target-set top-1 to `0.2754` and class-aware AP50 to `0.1243` (`/private/tmp/rfdetr_small_crops_fullcoverage_384_resnet50_imageprior_cocoeval.csv`). ResNet50 crop-level transfer gives AP50 `0.1195`, so crop-level classification is still not better than full-image classification. Interpretation: stronger pretrained category teachers help, but the crop view alone is not the missing ingredient.
  - ConvNeXt-Tiny full-image transfer was also checked after downloading the torchvision checkpoint. It gives target-set top-1 `0.2464`, top-5 `0.3043`, and class-aware AP50 `0.1061` (`/private/tmp/rfdetr_small_crops_fullcoverage_384_convnext_tiny_imageprior_cocoeval.csv`). This is below ResNet50 despite higher top-5, so the current simple one-category-per-image relabeling protocol is driven by top-1 category correctness rather than broad candidate coverage. ResNet50 is the strongest checked torchvision teacher so far.
  - Category-prior top-k expansion added: `--top-k-categories K --category-score-mode {keep,multiply}` can duplicate each detection over the top-k mapped ImageNet categories. ResNet50 top-5 with score multiplication improves AP50 to `0.1607`, versus `0.1243` for hard top-1. The keep-score control drops to AP50 `0.0742`, so the gain comes from classifier-prior ranking, not from blindly duplicating categories. This is the strongest non-oracle class-aware result so far, but it still trails the GT largest-category prior (`0.2185`) and nearest-GT category oracle (`0.3066`).
  - ResNet50 top-k sweep: top-3 multiply gives AP50 `0.1415`, top-5 gives `0.1607`, and top-10 gives `0.1611`. The useful gain saturates around top-5; top-10 adds duplicate predictions with negligible AP50 benefit. Use top-5 multiply as the practical category-transfer setting unless a held-out calibration split later justifies another K.
  - Lightweight teacher checks: EfficientNet-B0 top-5 multiply gives AP50 `0.1253`, and MobileNetV3-Large top-5 multiply gives AP50 `0.1158`. Both are below ResNet50 top-5 multiply, so there is no immediate reason to prefer a lightweight ImageNet teacher for this diagnostic. The next meaningful category step should be a stronger teacher or detector-aware category scoring, not more lightweight torchvision sweeps.
  - ConvNeXt-Tiny top-5 multiply gives AP50 `0.1253`, matching EfficientNet-B0 and still below ResNet50. Even though ConvNeXt-Tiny has the highest target-set top-5 among the checked torchvision teachers (`0.3043`), the detection metric does not convert that into a ResNet50-level AP gain. This closes the current lightweight/modern torchvision teacher check.
  - Simple teacher ensembling was checked by fusing top-5 prediction JSONs with per-class NMS. ResNet50+ConvNeXt-Tiny gives AP50 `0.1595`, and ResNet50+ConvNeXt-Tiny+EfficientNet-B0 gives AP50 `0.1577`, both below single ResNet50 top-5 multiply (`0.1607`). Do not spend more time on naive torchvision teacher ensembles; the next category improvement needs a stronger teacher, better calibration split, or detector-aware category scoring.
  - Slice robustness check for the current best non-oracle category prior: ResNet50 top-5 multiply improves offcenter AP50 from the ResNet50 top-1 prior's `0.1012` to `0.1465`, and center AP50 from `0.0910` to `0.1010`. The category-prior gain is therefore stronger on the offcenter/hard slice, not merely a center-object artifact.
  - Calibration split utility added: `scripts/split_coco_by_images.py` creates image-disjoint COCO annotation splits. On a 34-image calibration / 35-image heldout split of the small-valid source images, ResNet50 AP50 for top1/top3/top5/top10 is `0.1002/0.1033/0.1033/0.1037` on calibration and `0.1161/0.1349/0.1634/0.1655` on heldout. This split is too small for fine K tuning, but it supports the top-k expansion signal and makes future category-prior selection less eval-set tuned.
  - Full small-valid multi-crop coverage check: increasing eval crops from one to at most two per source image raises class-agnostic NMS localization AP50 from `0.2002` to `0.2255`, with offcenter loc AP50 `0.1710`. Raw class-aware AP50 remains `0.0030`, confirming category failure. Applying ResNet50 top-5 multiply to the max2 fused boxes raises class-aware AP50 to `0.1688`; the largest-GT image-category prior reaches `0.2325`. This supports multi-crop coverage plus category-prior transfer as the current strongest RF-DETR tiling path.
  - Max3 crop coverage check: increasing eval crops to at most three per source image raises class-agnostic loc AP50 to `0.2359` and offcenter loc AP50 to `0.1870`, but ResNet50 top-5 class-aware AP50 only reaches `0.1697`. Interpretation: extra crop coverage still improves localization, but class-aware AP is now largely category/ranking limited under the current ResNet50 prior. Do not blindly continue max4/max5 before improving category scoring.
  - Max3 oracle split: nearest-GT category with original scores reaches AP50 `0.3712`, while nearest-GT category with IoU scores reaches AP50 `0.7515`. This widens the gap relative to ResNet50 top-5 AP50 `0.1697` and confirms that the next high-value work is category prediction plus score-IoU ranking, not more crop geometry.
  - Category-prior coverage diagnostic added to the torchvision relabeler: in addition to largest-GT top1/top5, it now reports whether the teacher hits any GT category in the image. On the max3 ResNet50 top-5 setting, largest-category top1/top5 is `0.2754/0.2899`, while any-GT top1/top5 is `0.3768/0.3913`. This means the ImageNet teacher's category coverage is itself limited on this DET subset; the remaining oracle gap is not only a top-k/ranking issue.
  - Score-IoU alignment diagnostic added: `scripts/analyze_coco_score_iou_correlation.py` computes nearest-GT IoU and score-IoU correlations. On max3 fused boxes, original localization score has Pearson/Spearman `0.3914/0.2102` and top-100 mean nearest IoU `0.2518`; ResNet50 top-5 category prior has weaker correlation `0.0951/0.1116` but higher top-100 mean nearest IoU `0.3401`; oracle-IoU score reaches top-100 mean IoU `0.8743`. This supports the next bottleneck diagnosis: category transfer helps retrieval, but scoring is still not calibrated to localization quality.
  - Stage summary table added: `results/rfdetr_tiling_category_transfer_summary.csv`, generated by `scripts/summarize_coco_eval_table.py`, collects the key one-crop/max2/max3 localization and category-transfer rows. Use this table as the current RF-DETR tiling stage gate instead of hand-copying scattered `/private/tmp` COCOeval outputs.
  - Stage-gate note added: `docs/rfdetr_tiling_stage_gate.md` summarizes the current best practical setting, key AP50 rows, interpretation, and stop/continue rules for the RF-DETR tiling/category-transfer path.
  - Current RF-DETR tiling decision: do not claim crop finetuning solves small detection. The next serious step is a two-branch tiling report: (1) class-agnostic localization with class-agnostic NMS to measure proposal/box potential, and (2) class-aware AP after category calibration or category transfer. Multi-crop/full-image fusion should cover more source images, then apply class-agnostic NMS before evaluating localization and only then revisit category scoring.
  - Category-stratified RF-DETR split built and checked: `data/ILSVRC2013_DET_val_supervised/rfdetr_stratified_seed41_train1000_val200_test200_min3` has train `1000` images / `4409` boxes, valid `200`, test `200` / `604`, and no test-positive category with fewer than `3` train boxes. This fixes the random train1000 protocol caveat where some test categories had zero or very low train coverage.
  - Stratified RF-DETR Small 384 2ep smoke: class AP/AP50/AP75 `0.0949/0.1122/0.1001`, loc AP/AP50/AP75 `0.4111/0.5426/0.4504`. This is the new fair-split smoke baseline, not directly comparable with the longer random-split rows.
  - Stratified RF-DETR Small 384 continuation from the 2ep regular checkpoint with fresh lr `1e-4` for 4 more epochs: independent test export from `checkpoint_best_regular.pth` gives class AP/AP50/AP75 `0.1543/0.1869/0.1645` and loc AP/AP50/AP75 `0.4206/0.5584/0.4484`. Slice AP50 is `offcenter=0.1675`, `center=0.1877`, `small=0.1544`, `medium=0.2226`, `large=0.2968`. Use `checkpoint_best_regular.pth`; RF-DETR's `best_total` again selected anomalous EMA and should not be used for independent conclusions.
  - Stratified category diagnostics after continuation: top-100 / IoU 0.5 coverage is `loc=0.8427`, global class-aware `0.6424`, per-category class-aware `0.7119`. Score-IoU correlation remains weak (`loc Spearman=-0.0036`, class-aware Spearman `0.0862`) despite high top-100 nearest IoU (`0.9053` loc, `0.8891` class-aware). Worst trained-category gaps still include classes like `n07695742`, `n04468005`, `n03790512`, `n03761084`, and `n02992211`, so remaining class failures are not only split-coverage artifacts.
  - Stratified oracle headroom confirms the same bottleneck on the fair split. Base AP/AP50/AP75 is `0.1543/0.1869/0.1645`; same-category IoU-score oracle is `0.4257/0.5178/0.4592`; nearest-GT relabel with original scores is `0.4555/0.6009/0.4778`; nearest-GT relabel + IoU score is `0.4515/0.5384/0.4935`. Interpretation: there is large ranking headroom when category is already right and even larger category-assignment headroom; next work should improve detector-side category assignment/scoring rather than returning to persistent proposal state or post-hoc crop-prior patches.
  - Stratified high-IoU confusion overlays added: the largest wrong localized flows include `n07695742 -> n01726692` (`7/9`, mean IoU `0.955`), `n07739125 -> n07749582` (`6/11`, mean IoU `0.893`), and `n02799071 -> n02786058` (`5/6`, mean IoU `0.956`). Artifacts: `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_category_confusions_top100_iou50.csv`, `results/rfdetr_stratified_seed41_2best_category_confusion_overlays/contact_sheet.jpg`, and `results/rfdetr_stratified_seed41_2best_category_confusion_overlays/manifest.csv`.
  - Validation-split score calibration was checked on the stratified 2best continuation. The calibrator learns validation quality signal (`Spearman=0.3051` for class-aware quality, `0.6153` for loc quality), but it does not improve test AP: base class AP/AP50/AP75 `0.1543/0.1869/0.1645`, class-aware calibrated `0.1550/0.1864/0.1650`, loc-quality calibrated `0.1509/0.1794/0.1606`. Conclusion: simple post-hoc scalar calibration is not the fix on the fair split; the next class route needs detector-side category assignment/representation changes.
  - Small RF-DETR deformable-attention probe added for 8 off-center test images with top-3 query overlays. Query mean IoU is `0.5638`, IoU>=0.5 rate is `0.625`, mean GT attention mass is `0.8379`, and median mass is `0.9375`; however nearest-GT category match is only `0.5417` overall and `0.6667` among IoU>=0.5 queries. This supports the current diagnosis: many queries do attend to the object and localize it, but category assignment from query representation remains weak. Artifacts include `results/rfdetr_stratified_seed41_2best_attention_offcenter_probe/contact_sheet.jpg`, `query_contact_sheet.jpg`, and `results/rfdetr_stratified_seed41_2best_attention_offcenter_probe_diagnostics.json`.
  - Low-interference class-head-only continuation added for the stratified RF-DETR Small checkpoint. `scripts/train_rfdetr_coco.py` now supports `--trainable-scope class-head`, implemented by freezing the RF-DETR LightningModule after internal checkpoint construction so the optimizer sees only detector classification heads (`723K` trainable parameters vs `31.6M` frozen). A 1-epoch continuation from the regular 2best checkpoint gives independent test class AP/AP50/AP75 `0.1609/0.1942/0.1704`, up from `0.1543/0.1869/0.1645`; class-agnostic loc AP/AP50/AP75 is `0.4253/0.5661/0.4499`, slightly above the base `0.4206/0.5584/0.4484`. Diagnostics show class-aware score-IoU Spearman improves only mildly (`0.0862 -> 0.0946`), top-100 class-aware nearest IoU improves (`0.8891 -> 0.9023`), and top-100 coverage remains basically unchanged (`loc=0.8510`, global class-aware `0.6440`, per-category `0.7103`). Conclusion: class-head-only finetuning is a real but small positive detector-side signal; it does not close the category assignment/oracle gap by itself. Artifacts: `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_classhead_fixed_1ep_test_cocoeval.csv`, `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_classhead_fixed_1ep_test_loc_cocoeval.csv`, `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_classhead_fixed_1ep_test_slices.csv`, `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_classhead_fixed_1ep_score_iou_classaware.csv`, and `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_classhead_fixed_1ep_category_coverage_gap.csv`.
  - Query+class low-interference continuation checked next via `--trainable-scope query-class-head`, which trains `query_feat.weight`, `refpoint_embed.weight`, and detector classification heads while freezing backbone, decoder, and box heads (`1.7M` trainable, `30.6M` frozen). Independent test class AP/AP50/AP75 is `0.1603/0.1938/0.1707`, essentially tied with but slightly below class-head-only; loc AP/AP50/AP75 is `0.4269/0.5677/0.4547`, slightly higher. Coverage remains flat (`loc=0.8510`, global class-aware `0.6440`, per-category `0.7086`) and class-aware score-IoU Spearman is `0.0938`. Conclusion: simply adapting learned query/reference embeddings is not the missing category fix; it gives a small localization/ranking nudge but does not materially improve class assignment. Artifacts: `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_queryclass_fixed_1ep_test_cocoeval.csv`, `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_queryclass_fixed_1ep_test_loc_cocoeval.csv`, `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_queryclass_fixed_1ep_test_slices.csv`, and `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_queryclass_fixed_1ep_category_coverage_gap.csv`.
  - Decoder+class low-interference continuation checked via `--trainable-scope decoder-class-head`, which trains decoder representation layers, query/refpoint embeddings, and class heads while keeping backbone and box heads frozen (`6.2M` trainable, `26.2M` frozen). Independent test class AP/AP50/AP75 is `0.1615/0.1968/0.1734`, the highest among the 1ep low-interference continuations, but class-agnostic loc AP/AP50/AP75 drops to `0.4222/0.5597/0.4481` versus class-head/query-class. Slices do not improve the hard cases (`offcenter AP50=0.1783`, `small AP50=0.1573`), class-aware score-IoU Spearman falls to `0.0804`, and coverage remains flat-to-worse (`loc=0.8427`, global class-aware `0.6391`, per-category `0.7119`). Conclusion: opening decoder representation gives only a tiny class AP bump while degrading ranking/coverage signals; do not keep expanding trainable scope blindly. Artifacts: `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_decoderclass_fixed_1ep_test_cocoeval.csv`, `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_decoderclass_fixed_1ep_test_loc_cocoeval.csv`, `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_decoderclass_fixed_1ep_test_slices.csv`, and `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_decoderclass_fixed_1ep_category_coverage_gap.csv`.
  - Low-interference scope summary table added: `results/rfdetr_stratified_seed41_low_interference_scope_summary.csv`. Stage-gate decision: class-head/query-class/decoder-class continuations are useful controls, but none closes the category gap. Stop expanding trainable scope locally; next category work should use stronger supervision, teacher signals, or a more formal longer protocol.
  - Full-model low-LR continuation from the same stratified regular checkpoint reverses the low-interference conclusion. Training all parameters for 2 more epochs at lr `5e-5` and evaluating `checkpoint_best_regular.pth` on the independent test split gives class AP/AP50/AP75 `0.1930/0.2558/0.2154` and class-agnostic loc AP/AP50/AP75 `0.4277/0.5754/0.4622`. Slice AP50 improves broadly: `offcenter=0.2252`, `center=0.2582`, `small=0.1772`, `medium=0.2634`, `large=0.3483`. Candidate-set diagnostics also improve instead of only re-ranking: grouped candidate hit rate rises to `40.6%`, mean nearest IoU to `0.3330`, and candidate-oracle AP/AP50/AP75 to `0.4290/0.5627/0.4716`.
  - A high-support category check confirms the candidate gain is not purely a low-support artifact. For categories with at least `20` grouped boxes, weighted hit rate improves from `37.8%` in the base checkpoint to `41.1%` after full low-LR continuation, while zero-hit high-support categories fall from `10` to `8`.
  - Top-k coverage diagnostics agree with the candidate-oracle read: class-aware global top-100 recall improves from `0.6424` to `0.6887`, and per-category class-aware recall improves from `0.7119` to `0.7467`; class-agnostic loc recall is roughly flat/slightly lower (`0.8427 -> 0.8311`). Score-IoU alignment is still not solved (`loc Spearman=-0.1376`, class-aware Spearman `0.1082`), so the improvement is primarily semantic candidate generation rather than scalar calibration.
  - Interpretation: the category-candidate problem is not fixed by class-head-only, query-only, decoder-only, hard-category oversampling, scalar calibration, or persistent proposal state. It does respond to real full detector-side adaptation. The active route should therefore move to a formal longer RF-DETR protocol: repeat full low-LR continuation on another seed and/or extend the best regular checkpoint, while continuing to report candidate hit rate and candidate-oracle AP as primary diagnostics. Do not treat hard-category oversampling as the fix; it slightly regularizes AP but worsens candidate generation.
  - Follow-up full-model continuation from the full low-LR checkpoint at lr `3e-5` for 2 more epochs gives a mixed result. Class AP/AP50/AP75 continues to improve to `0.2016/0.2625/0.2252`, with strong slice AP50 gains on offcenter (`0.2455`), small (`0.1952`), medium (`0.3222`), and large (`0.3566`) objects. However class-agnostic loc AP/AP50/AP75 drops to `0.4131/0.5640/0.4450`, candidate hit rate drops from `40.6%` to `39.8%`, high-support weighted hit drops from `41.1%` to `39.9%`, and high-support zero-hit categories increase from `8` to `13`. Candidate-oracle AP/AP50/AP75 is `0.4242/0.5668/0.4557`, roughly flat in AP50 but below the prior AP/AP75. This means same-seed longer continuation improves deployed class AP but does not continue repairing candidate coverage. Next serious check should be second seed or a formal longer protocol, not indefinite continuation of seed41.
  - Candidate category transition helper added: `scripts/summarize_candidate_category_transitions.py` compares per-category candidate hit transitions across a same-seed chain. For the seed41 base -> full_lr_2ep -> extra_low_lr chain, `results/rfdetr_stratified_seed41_candidate_category_transitions.csv` shows `7` persistent high-support zero-hit categories, `4` categories regressing to zero-hit, only `2` categories rescued from zero-hit, and `38` persistent high-support hit categories. This sharpens the interpretation above: extra-low-LR continuation improves deployed class AP, but candidate coverage repair is not monotonic and still leaves concrete high-support failure categories to inspect.
  - Second-seed setup exposed an RF-DETR runner hazard: the official RF-DETR Small checkpoint has a 90-class head, and the package can leave the model head at 91 outputs even when `num_classes=200` is requested. `scripts/train_rfdetr_coco.py` now explicitly forces the detection head to `num_classes + 1` outputs after model construction and again inside the LightningModule runtime patch, rebuilding criterion/postprocess if needed. A direct probe confirmed the head changes from `(91, 256)` to `(201, 256)`. Treat any earlier second-seed run started before this patch as invalid if it printed only a 90-class head.
  - Handoff audit strengthened: `scripts/check_rfdetr_handoff.py` now records per-split `image_ids_sha256`, pairwise image-id overlap counts, and `image_ids_disjoint`. The refreshed stratified seed41 handoff check reports `train_valid=0`, `train_test=0`, `valid_test=0`, and `image_ids_disjoint=true`, so the stratified train/valid/test split is now explicitly image-disjoint in the machine-readable audit.
  - Forced-head second-seed 2ep sanity completed on `rfdetr_stratified_seed43_train1000_val200_test200_min3`. `predict_rfdetr_coco.py` reports `model_num_classes=200`, confirming the export path also sees the corrected head. Independent test metrics from `checkpoint_best_regular.pth` are class AP/AP50/AP75 `0.1474/0.1888/0.1608` and class-agnostic loc AP/AP50/AP75 `0.3635/0.5068/0.3868`; slice AP50 is `offcenter=0.1326`, `center=0.2097`, `small=0.0900`, `medium=0.2083`, `large=0.3923`. Candidate hit rate is `37.8%`, candidate-oracle AP/AP50/AP75 is `0.3729/0.4849/0.4079`, top-100 coverage is `loc=0.8079`, global class-aware `0.6382`, per-category class-aware `0.7000`, and score-IoU Spearman is weak (`loc=0.0506`, class-aware `0.1321`). Treat this as a repaired-head baseline sanity row only; it is not directly comparable to seed41's later full low-LR continuation, and it does not change the current long-train RF-DETR route.
  - Mainline summary helper added: `scripts/summarize_rfdetr_mainline.py` merges class-aware COCOeval, class-agnostic loc COCOeval, slice AP50, candidate-oracle diagnostics, top-k coverage, score-IoU correlation, and high-support candidate hit/zero-hit diagnostics into one stage-gate CSV. Current summary: `results/rfdetr_stratified_mainline_summary.csv`. This should be the quick comparison table for stratified RF-DETR regular-checkpoint runs before launching new long continuations.
  - Artifacts:
    - `results/rfdetr_stratified_seed41_fulllr5e5_2ep_test_cocoeval.csv`
    - `results/rfdetr_stratified_seed41_fulllr5e5_2ep_test_loc_cocoeval.csv`
    - `results/rfdetr_stratified_seed41_fulllr5e5_2ep_test_slices.csv`
    - `results/rfdetr_stratified_seed41_fulllr5e5_2ep_candidate_oracle_summary.csv`
    - `results/rfdetr_stratified_seed41_fulllr5e5_2ep_candidate_oracle_groupmax_cocoeval.csv`
    - `results/rfdetr_stratified_seed41_fulllr5e5_2ep_candidate_oracle_per_category.csv`
    - `results/rfdetr_stratified_seed41_fulllr5e5_2ep_candidate_oracle_per_category_with_split_counts.csv`
    - `results/rfdetr_stratified_seed41_category_intervention_summary.csv`
    - `results/rfdetr_stratified_seed41_candidate_high_support_summary.csv`
    - `results/rfdetr_stratified_seed41_fulllr5e5_2ep_score_iou_loc.csv`
    - `results/rfdetr_stratified_seed41_fulllr5e5_2ep_score_iou_classaware.csv`
    - `results/rfdetr_stratified_seed41_fulllr5e5_2ep_category_coverage_gap.csv`
    - `results/rfdetr_stratified_seed41_fulllr5e5_2ep_per_category_coverage_gap_with_split_counts.csv`
    - `results/rfdetr_stratified_seed41_fulllr5e5_then3e5_2ep_test_cocoeval.csv`
    - `results/rfdetr_stratified_seed41_fulllr5e5_then3e5_2ep_test_loc_cocoeval.csv`
    - `results/rfdetr_stratified_seed41_fulllr5e5_then3e5_2ep_test_slices.csv`
    - `results/rfdetr_stratified_seed41_fulllr5e5_then3e5_2ep_candidate_oracle_summary.csv`
    - `results/rfdetr_stratified_seed41_fulllr5e5_then3e5_2ep_candidate_oracle_groupmax_cocoeval.csv`
    - `results/rfdetr_stratified_seed41_candidate_category_transitions.csv`
    - `results/rfdetr_stratified_seed41_fulllr5e5_then3e5_2ep_category_coverage_gap.csv`
    - `results/rfdetr_stratified_seed41_fulllr5e5_then3e5_2ep_per_category_coverage_gap_with_split_counts.csv`
    - `results/rfdetr_stratified_seed43_forced_2ep_test_cocoeval.csv`
    - `results/rfdetr_stratified_seed43_forced_2ep_test_loc_cocoeval.csv`
    - `results/rfdetr_stratified_seed43_forced_2ep_test_slices.csv`
    - `results/rfdetr_stratified_seed43_forced_2ep_candidate_oracle_summary.csv`
    - `results/rfdetr_stratified_seed43_forced_2ep_candidate_oracle_groupmax_cocoeval.csv`
    - `results/rfdetr_stratified_seed43_forced_2ep_category_coverage_gap.csv`
    - `results/rfdetr_stratified_seed43_forced_2ep_per_category_coverage_gap_with_split_counts.csv`
    - `results/rfdetr_stratified_mainline_summary.csv`
