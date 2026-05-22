# RF-DETR Tiling Stage Gate

This note records the current RF-DETR small-object tiling decision point. It is a
stage gate, not a final benchmark.

## Current Best Practical Setting

- Detector source: RF-DETR Nano crop checkpoint from the diversified small-crop
  run.
- Inference geometry: up to 3 small-object-centered crops per source image at
  384px.
- Box fusion: remap crop predictions to source coordinates, then class-agnostic
  NMS at IoU 0.5.
- Category transfer: ResNet50 ImageNet prior, restricted to DET categories,
  top-5 category expansion, score multiplied by classifier probability.

Current class-aware AP50: `0.1697`.

## Key Evidence

| Setting | AP50 |
|---|---:|
| one crop, class-agnostic localization | 0.2002 |
| max2 crops, class-agnostic localization | 0.2255 |
| max3 crops, class-agnostic localization | 0.2359 |
| one crop + ResNet50 top-5 category prior | 0.1607 |
| max2 crops + ResNet50 top-5 category prior | 0.1688 |
| max3 crops + ResNet50 top-5 category prior | 0.1697 |
| max3 crops + nearest-GT category, original score | 0.3712 |
| max3 crops + nearest-GT category, IoU score | 0.7515 |

Machine-readable table:

- `results/rfdetr_tiling_category_transfer_summary.csv`

Score-IoU alignment diagnostics:

| Setting | Pearson | Spearman | Top-100 mean nearest IoU |
|---|---:|---:|---:|
| max3 fused localization score | 0.3914 | 0.2102 | 0.2518 |
| max3 + ResNet50 top-5 category prior | 0.0951 | 0.1116 | 0.3401 |
| max3 oracle IoU score | 1.0000 | 1.0000 | 0.8743 |

Held-out score calibration check:

| Setting | Split | AP | AP50 | AP75 | Off-center AP50 | Top-100 mean class-aware IoU |
|---|---|---:|---:|---:|---:|---:|
| max3 + ResNet50 top-5 prior | heldout | 0.1209 | 0.1781 | 0.1470 | 0.1682 | 0.2293 |
| + calibration-split post-hoc quality score | heldout | 0.1230 | 0.1878 | 0.1473 | 0.1933 | 0.2418 |

Calibration artifacts:

- `results/rfdetr_max3_resnet50_top5x_heldout_calibration_summary.csv`
- `results/rfdetr_max3_resnet50_top5x_heldout_base_slices.csv`
- `results/rfdetr_max3_resnet50_top5x_heldout_calibrated_slices.csv`
- `results/rfdetr_max3_resnet50_top5x_heldout_base_filtered_score_iou.csv`
- `results/rfdetr_max3_resnet50_top5x_heldout_calibrated_score_iou.csv`
- `results/rfdetr_score_calibration_heldout_summary.csv`

Held-out prediction recall diagnostic:

| Setting | Class-aware | Top-10 R@50 | Top-50 R@50 | Top-100 R@50 | Top-100 R@75 |
|---|---:|---:|---:|---:|---:|
| max3 fused localization | no | 0.5063 | 0.7342 | 0.7848 | 0.4494 |
| max3 + ResNet50 top-5 prior | yes | 0.1582 | 0.2785 | 0.3228 | 0.1899 |
| + calibration-split post-hoc quality score | yes | 0.1646 | 0.2785 | 0.3228 | 0.1899 |

Recall artifacts:

- `results/rfdetr_max3_heldout_loc_prediction_recall.csv`
- `results/rfdetr_max3_resnet50_top5x_heldout_base_prediction_recall.csv`
- `results/rfdetr_max3_resnet50_top5x_heldout_calibrated_prediction_recall.csv`
- `results/rfdetr_prediction_recall_heldout_summary.csv`

Category coverage gap diagnostic, all-slice R@50:

| Setting | Top-K | Global loc recall | Global class-aware recall | Per-category class-aware recall | Global category retention | Ranking gap |
|---|---:|---:|---:|---:|---:|---:|
| max3 + ResNet50 top-5 prior | 10 | 0.4304 | 0.1392 | 0.1582 | 0.3235 | 0.0190 |
| max3 + ResNet50 top-5 prior | 50 | 0.6962 | 0.2658 | 0.2785 | 0.3818 | 0.0127 |
| max3 + ResNet50 top-5 prior | 100 | 0.7468 | 0.3101 | 0.3228 | 0.4153 | 0.0127 |
| + calibration-split post-hoc quality score | 10 | 0.4873 | 0.1392 | 0.1646 | 0.2857 | 0.0253 |
| + calibration-split post-hoc quality score | 50 | 0.7278 | 0.2785 | 0.2785 | 0.3826 | 0.0000 |
| + calibration-split post-hoc quality score | 100 | 0.7722 | 0.3165 | 0.3228 | 0.4098 | 0.0063 |

Category coverage artifacts:

- `scripts/analyze_coco_category_coverage_gap.py`
- `results/rfdetr_max3_resnet50_top5x_heldout_base_category_coverage_gap.csv`
- `results/rfdetr_max3_resnet50_top5x_calibrated_heldout_category_coverage_gap.csv`

Existing category source scorecard, full split:

| Setting | AP50 | Global top-100 class R@50 | Category retention |
|---|---:|---:|---:|
| max3 + ResNet50 top-5 prior | 0.1697 | 0.2092 | 0.2963 |
| onecrop + ResNet50 top-5 prior | 0.1607 | 0.1634 | 0.2841 |
| onecrop + ConvNeXt-Tiny top-5 prior | 0.1253 | 0.1667 | 0.2881 |
| onecrop + EfficientNet-B0 top-5 prior | 0.1253 | 0.1471 | 0.2557 |
| onecrop + teacher ensemble top-5 prior | 0.1577 | 0.1634 | 0.2890 |
| onecrop + ResNet50/ConvNeXt ensemble top-5 prior | 0.1595 | 0.1699 | 0.2971 |

Category source scorecard artifacts:

- `scripts/summarize_coco_category_sources.py`
- `results/rfdetr_category_source_scorecard_full_summary.csv`

Candidate-constrained category oracle:

| Setting | Candidate hit rate | AP50 | AP75 | AR100 |
|---|---:|---:|---:|---:|
| Existing top-5 candidate set, use candidate score | 0.2039 | 0.2578 | 0.1845 | 0.2723 |
| Existing top-5 candidate set, use group max score | 0.2039 | 0.2578 | 0.1845 | 0.2723 |
| Existing top-5 candidate set, use oracle IoU score | 0.2039 | 0.4736 | 0.2807 | 0.2723 |
| Any category oracle, original score | - | 0.3712 | 0.2346 | 0.4413 |
| Any category oracle, IoU score | - | 0.7515 | 0.4554 | 0.4413 |

Candidate-oracle artifacts:

- `scripts/oracle_coco_category_candidates.py`
- `results/rfdetr_candidate_category_oracle_summary.csv`
- `results/rfdetr_max3_resnet50_top5x_candidate_oracle_candidate_cocoeval.csv`
- `results/rfdetr_max3_resnet50_top5x_candidate_oracle_groupmax_cocoeval.csv`
- `results/rfdetr_max3_resnet50_top5x_candidate_oracle_iou_cocoeval.csv`

Matched-proposal crop classifier check:

| Setting | Eval crop top-5 | AP50 | Global top-100 class R@50 | Category retention |
|---|---:|---:|---:|---:|
| Matched proposal crops, score multiply | 0.6667 | 0.0172 | 0.2722 | 0.3496 |
| Matched proposal crops, keep detector score | 0.6667 | 0.0103 | 0.2911 | 0.4466 |
| Image ResNet50 top-5 prior, heldout base | - | 0.1781 | 0.3101 | 0.4153 |

Matched-propcrop artifacts:

- `scripts/train_coco_proposal_crop_classifier.py`
- `results/rfdetr_matched_propcrop_prior_heldout_summary.csv`
- `results/rfdetr_max3_matched_propcrop_resnet50_top5x_heldout_summary.csv`
- `results/rfdetr_max3_matched_propcrop_resnet50_top5keep_heldout_summary.csv`

Integrated RF-DETR detector-side class-head check:

| Setting | Resolution | Class AP50 | Class AP75 | Class-agnostic AP50 | Off-center AP50 | Small AP50 | Large AP50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| full split Nano, 1 epoch | 128 | 0.0439 | 0.0364 | 0.3081 | 0.0278 | 0.0300 | 0.1036 |
| full split Nano, 1 epoch | 384 | 0.0495 | 0.0401 | 0.5584 | 0.0455 | 0.0630 | 0.1302 |
| full split Nano, 3 epochs | 384 | 0.1224 | 0.0960 | 0.6074 | 0.1060 | 0.1320 | 0.2220 |
| full split Nano, 5 epochs | 384 | 0.1437 | 0.1027 | 0.6214 | 0.1454 | 0.1800 | 0.2715 |
| full split Small, 1 epoch | 384 | 0.0513 | 0.0372 | 0.5464 | 0.0396 | 0.0413 | 0.1349 |
| full split Small, 3 epochs | 384 | 0.1178 | 0.0934 | 0.6168 | 0.1187 | 0.1297 | 0.2264 |
| full split Small, resumed 5 epochs | 384 | 0.1835 | 0.1446 | 0.6285 | 0.1885 | 0.1788 | 0.3130 |

Integrated RF-DETR artifacts:

- `results/rfdetr_integrated_class_head_summary.csv`
- `scripts/build_coco_pseudolabel_dataset.py`
- `results/rfdetr_offcenter_seed41_full_nano_128_1ep_test_cocoeval.csv`
- `results/rfdetr_offcenter_seed41_full_nano_128_1ep_test_loc_cocoeval.csv`
- `results/rfdetr_offcenter_seed41_full_nano_128_1ep_test_slices.csv`
- `results/rfdetr_offcenter_seed41_full_nano_384_1ep_test_cocoeval.csv`
- `results/rfdetr_offcenter_seed41_full_nano_384_1ep_test_loc_cocoeval.csv`
- `results/rfdetr_offcenter_seed41_full_nano_384_1ep_test_slices.csv`
- `results/rfdetr_offcenter_seed41_full_nano_384_3ep_test_cocoeval.csv`
- `results/rfdetr_offcenter_seed41_full_nano_384_3ep_test_loc_cocoeval.csv`
- `results/rfdetr_offcenter_seed41_full_nano_384_3ep_test_slices.csv`
- `results/rfdetr_offcenter_seed41_full_nano_384_5ep_test_cocoeval.csv`
- `results/rfdetr_offcenter_seed41_full_nano_384_5ep_test_loc_cocoeval.csv`
- `results/rfdetr_offcenter_seed41_full_nano_384_5ep_test_slices.csv`
- `results/rfdetr_offcenter_seed41_full_small_384_1ep_test_cocoeval.csv`
- `results/rfdetr_offcenter_seed41_full_small_384_1ep_test_loc_cocoeval.csv`
- `results/rfdetr_offcenter_seed41_full_small_384_1ep_test_slices.csv`
- `results/rfdetr_offcenter_seed41_full_small_384_3ep_test_cocoeval.csv`
- `results/rfdetr_offcenter_seed41_full_small_384_3ep_test_loc_cocoeval.csv`
- `results/rfdetr_offcenter_seed41_full_small_384_3ep_test_slices.csv`
- `results/rfdetr_offcenter_seed41_full_small_384_resume5ep_test_cocoeval.csv`
- `results/rfdetr_offcenter_seed41_full_small_384_resume5ep_test_loc_cocoeval.csv`
- `results/rfdetr_offcenter_seed41_full_small_384_resume5ep_test_slices.csv`

Interpretation:

- Nano 384px continues to improve with longer detector-side training: class-aware
  AP50 rises from `0.1224` at 3 epochs to `0.1437` at 5 epochs, with
  off-center AP50 improving from `0.1060` to `0.1454`.
- Class-agnostic AP50 remains much higher (`0.6214` at 5 epochs), so category
  learning/class-head quality is still the main bottleneck.
- Small 384px at 1 epoch is not a fair stronger-teacher result yet. It verifies
  the stronger model path and 200-class export, but AP50 `0.0513` shows that it
  needs a longer detector-side schedule before it can be judged.
- Small 384px at 3 epochs improves to AP50 `0.1178` and class-agnostic AP50
  `0.6168`, but it remains below Nano 3 epochs (`0.1224`) and Nano 5 epochs
  (`0.1437`) on class-aware AP50.
- Resuming Small from its 3-epoch checkpoint to 5 epochs changes that picture:
  regular-checkpoint export reaches class AP50 `0.1835`, AP75 `0.1446`,
  class-agnostic AP50 `0.6285`, off-center AP50 `0.1885`, and large AP50
  `0.3130`. This is the strongest validation-mirrored integrated detector-side
  checkpoint so far and finally beats the max3 crop + ResNet50 top-5 prior AP50
  `0.1697` on the same mirrored protocol.
- However, this stronger heldout detector does not automatically become a
  stronger pseudo-label teacher. Its filtered train pseudo-label quality still
  trails the original Nano5 teacher under the checked filters. The next useful
  step is a formal long-train / independent-test protocol or better
  teacher-filtering, not another post-hoc crop prior.
- Detector-side pseudo-label distillation is now wired at the dataset level:
  `scripts/build_coco_pseudolabel_dataset.py` can turn teacher COCO predictions
  into RF-DETR-compatible pseudo annotations. A smoke build with Nano 5ep test
  predictions produced `808` pseudo test annotations and passed RF-DETR
  `--check-only` with `dataset_num_classes=200`. The next distillation step
  requires train-split predictions from a stronger teacher, then teacher-only
  vs GT+pseudo RF-DETR training.

Detector-side pseudo-label distillation smoke:

| Setting | Teacher | Train annotations | Class AP50 | Class AP75 | Class-agnostic AP50 | Off-center AP50 | Small AP50 |
|---|---|---:|---:|---:|---:|---:|---:|
| GT baseline, Nano 1 epoch | GT | 1,610 | 0.0495 | 0.0401 | 0.5584 | 0.0455 | 0.0630 |
| teacher-only, Nano 1 epoch | Nano 5ep train predictions, score>=0.25 top20 | 2,191 | 0.0552 | 0.0416 | 0.5499 | 0.0494 | 0.0451 |
| GT+pseudo, Nano 1 epoch | GT + Nano 5ep train predictions, score>=0.25 top20 | 3,801 | 0.0484 | 0.0330 | 0.3737 | 0.0345 | 0.0514 |
| teacher-only pretrain -> GT finetune, Nano 1+1 epochs | Nano 5ep train predictions, score>=0.25 top20 | 1,610 | 0.0816 | 0.0633 | 0.5749 | 0.0708 | 0.0645 |
| teacher-only pretrain -> GT finetune, Nano 1+2 epochs | Nano 5ep train predictions, score>=0.25 top20 | 1,610 | 0.1088 | 0.0820 | 0.6273 | 0.1163 | 0.0964 |
| teacher-only pretrain -> GT finetune, Nano 1+3 epochs | Nano 5ep train predictions, score>=0.25 top20 | 1,610 | 0.1428 | 0.1048 | 0.6198 | 0.1514 | 0.1461 |
| teacher-only pretrain -> GT finetune, Nano 1+4 epochs | Nano 5ep train predictions, score>=0.25 top20 | 1,610 | 0.1342 | 0.0990 | 0.6290 | 0.1323 | 0.1151 |
| teacher-only pretrain -> GT finetune, Nano 1+3 epochs, seed43 | Nano 5ep train predictions, score>=0.25 top20 | 1,610 | 0.1413 | 0.1070 | 0.6367 | 0.1522 | 0.1589 |

Distillation artifacts:

- `results/rfdetr_detector_side_pseudolabel_summary.csv`
- `results/rfdetr_train_teacher_quality_summary.csv`
- `results/rfdetr_pseudo_filter_quality_summary.csv`
- `results/rfdetr_nano5_teacher_only_s025k20_384_1ep_test_cocoeval.csv`
- `results/rfdetr_nano5_teacher_only_s025k20_384_1ep_test_loc_cocoeval.csv`
- `results/rfdetr_nano5_teacher_only_s025k20_384_1ep_test_slices.csv`
- `results/rfdetr_nano5_gt_pseudo_s025k20_384_1ep_test_cocoeval.csv`
- `results/rfdetr_nano5_gt_pseudo_s025k20_384_1ep_test_loc_cocoeval.csv`
- `results/rfdetr_nano5_gt_pseudo_s025k20_384_1ep_test_slices.csv`
- `results/rfdetr_nano5_teacherpre_gtfinetune_384_1ep_regular_test_cocoeval.csv`
- `results/rfdetr_nano5_teacherpre_gtfinetune_384_1ep_regular_test_loc_cocoeval.csv`
- `results/rfdetr_nano5_teacherpre_gtfinetune_384_1ep_regular_test_slices.csv`
- `results/rfdetr_nano5_teacherpre_gtfinetune_384_2ep_regular_test_cocoeval.csv`
- `results/rfdetr_nano5_teacherpre_gtfinetune_384_2ep_regular_test_loc_cocoeval.csv`
- `results/rfdetr_nano5_teacherpre_gtfinetune_384_2ep_regular_test_slices.csv`
- `results/rfdetr_nano5_teacherpre_gtfinetune_384_3ep_regular_test_cocoeval.csv`
- `results/rfdetr_nano5_teacherpre_gtfinetune_384_3ep_regular_test_loc_cocoeval.csv`
- `results/rfdetr_nano5_teacherpre_gtfinetune_384_3ep_regular_test_slices.csv`
- `results/rfdetr_nano5_teacherpre_gtfinetune_384_4ep_regular_test_cocoeval.csv`
- `results/rfdetr_nano5_teacherpre_gtfinetune_384_4ep_regular_test_loc_cocoeval.csv`
- `results/rfdetr_nano5_teacherpre_gtfinetune_384_4ep_regular_test_slices.csv`
- `results/rfdetr_nano5_teacherpre_gtfinetune_384_3ep_seed43_regular_test_cocoeval.csv`
- `results/rfdetr_nano5_teacherpre_gtfinetune_384_3ep_seed43_regular_test_loc_cocoeval.csv`
- `results/rfdetr_nano5_teacherpre_gtfinetune_384_3ep_seed43_regular_test_slices.csv`

True independent-test detector-side distillation check:

| Setting | Split protocol | Seed | Train annotations | Class AP50 | Class AP75 | Loc AP50 | Off-center AP50 | Center AP50 | Small AP50 | Large AP50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GT-only Nano 384px, 1 epoch | indtest_seed43 | 43 | 1,610 | 0.0224 | 0.0201 | 0.2113 | 0.0154 | 0.0266 | 0.0329 | 0.0441 |
| teacher-only Nano 384px, 1 epoch | indtest_seed43 | 43 | 2,191 | 0.0204 | 0.0182 | 0.1908 | 0.0145 | 0.0248 | 0.0038 | 0.0420 |
| GT-only Nano 384px, 4 epochs | indtest_seed43 | 43 | 1,610 | 0.1618 | 0.1180 | 0.6151 | 0.1343 | 0.1658 | 0.1051 | 0.2543 |
| teacher pretrain -> GT finetune, Nano 1+3 epochs | indtest_seed43 | 43 | 1,610 | 0.1618 | 0.1259 | 0.6170 | 0.1310 | 0.1757 | 0.1035 | 0.3023 |
| GT-only Nano 384px, 4 epochs | indtest_seed43 | 41 | 1,610 | 0.1517 | 0.1046 | 0.6194 | 0.1383 | 0.1442 | 0.1050 | 0.2404 |
| teacher pretrain -> GT finetune, Nano 1+3 epochs | indtest_seed43 | 41 | 1,610 | 0.1502 | 0.1042 | 0.6263 | 0.1143 | 0.1710 | 0.1216 | 0.2970 |

Independent-test artifacts:

- `results/rfdetr_indtest_seed43_detector_side_summary.csv`
- `results/rfdetr_offcenter_seed41_indtest_seed43_handoff_check.json`
- `results/rfdetr_indtest_seed43_gt_384_4ep_seed41_regular_test_cocoeval.csv`
- `results/rfdetr_indtest_seed43_gt_384_4ep_seed41_regular_test_loc_cocoeval.csv`
- `results/rfdetr_indtest_seed43_gt_384_4ep_seed41_regular_test_slices.csv`
- `results/rfdetr_indtest_seed43_nano5_teacherpre_gtfinetune_s025k20_384_3ep_seed41_regular_test_cocoeval.csv`
- `results/rfdetr_indtest_seed43_nano5_teacherpre_gtfinetune_s025k20_384_3ep_seed41_regular_test_loc_cocoeval.csv`
- `results/rfdetr_indtest_seed43_nano5_teacherpre_gtfinetune_s025k20_384_3ep_seed41_regular_test_slices.csv`

Independent-test interpretation:

- The old single-stage teacher-only result is protocol-sensitive: on a true
  image-disjoint test split, GT-only 1 epoch beats teacher-only 1 epoch.
- Staged teacher-only pretrain -> GT finetune remains useful as a warm-start /
  localization route, but the equal-budget claim is now conservative. Across
  seeds 43 and 41, staged does not clearly beat GT-only on class AP50.
- In seed41, staged improves class-agnostic AP50 (`0.6263` vs `0.6194`), center
  AP50 (`0.1710` vs `0.1442`), small AP50 (`0.1216` vs `0.1050`), and large AP50
  (`0.2970` vs `0.2404`), while GT-only improves class AP50 (`0.1517` vs
  `0.1502`) and off-center AP50 (`0.1383` vs `0.1143`).
- Current decision: keep staged pseudo-pretrain as a diagnostic and possible
  localization pretraining route, but do not claim self-teacher pseudo-label
  distillation beats equal-budget GT training until a stronger train-split
  teacher or longer formal protocol confirms it.

Pseudo-label filtering update:

- `scripts/coco_annotations_to_predictions.py` converts selected pseudo-label
  annotation JSONs into COCO detection prediction JSONs, preserving
  `teacher_score` as the detection score.
- Direct train-GT evaluation shows that stricter Nano5 pseudo filters reduce
  selected-label AP mostly by losing recall. `score>=0.15/top20` is currently
  the strongest no-training filter (`class AP50 0.4098`, `loc AP50 0.7565`,
  `3,872` selected boxes), followed by `score>=0.20/top20` (`class AP50
  0.3991`, `loc AP50 0.7482`, `2,842` boxes). The current training filter
  `score>=0.25/top20` is lower (`class AP50 0.3689`, `loc AP50 0.7272`,
  `2,191` boxes).
- `score>=0.25/top10` is worse than `score>=0.25/top20`, so top-k truncation is
  not the right cleanup lever here.
- The staged 1+3 student remains a weaker train pseudo-label teacher than
  Nano5 under the same `score>=0.25/top20` filter (`class AP50 0.2255` vs
  `0.3689` on selected labels).
- Small 384px 3ep is also not a stronger train-split pseudo-label teacher:
  raw train prediction AP50 is only `0.2464`, and the same `score>=0.25/top20`
  filter gives selected-label AP50 `0.1728`, far below Nano5 (`0.4474` raw and
  `0.3689` filtered).
- Historical note: resumed Small 5ep is a much stronger detector and raw train teacher
  (`raw train AP50 0.4041`, `loc AP50 0.7764`), but it still does not beat
  Nano5 as a filtered pseudo-label source. Its selected-label AP50 is `0.3507`
  at `score>=0.15/top20`, `0.3340` at `score>=0.20/top20`, and `0.3025` at
  `score>=0.25/top20`, all below the corresponding Nano5 filter quality
  (`0.4098`, `0.3991`, `0.3689`).
- Training gate: despite better static pseudo-label AP, wider Nano5 filters
  collapse badly in teacher-only 1ep training after regular-checkpoint export.
  `score>=0.15/top20` gives `class AP50 0.0120`, `loc AP50 0.1509`,
  `offcenter AP50 0.0077`; `score>=0.20/top20` gives `class AP50 0.0119`,
  `loc AP50 0.1852`, `offcenter AP50 0.0074`. Both are far below the previous
  `score>=0.25/top20` teacher-only result (`class AP50 0.0552`, `loc AP50
  0.5499`). Do not run staged 1+3 on `s015/top20` or `s020/top20`.
- This old Nano5 filtering result is now superseded by Small-resume8 as the
  first train-split teacher that beats Nano5 under the checked filters. The
  lasting lesson is narrower: pseudo filters must be judged by both static
  train-GT quality and training stability.

Protocol caveat:

- `scripts/check_rfdetr_handoff.py` now reports annotation SHA256 hashes and
  split category-range consistency. Current RF-DETR prepared split has
  consistent `1..200` category ranges, but `valid` and `test` annotations are
  byte-identical. Treat existing `*_test_*` RF-DETR numbers as
  validation-mirrored heldout results rather than independent-test numbers until
  a true separate test split is prepared.
- `scripts/split_coco_by_images.py --keep-all-categories` now supports making
  image-disjoint validation/test annotation files while preserving the original
  200-category table. The independent-test prepared dataset
  `rfdetr_offcenter_seed41_indtest_seed43` has train `330/1610`, valid
  `100/319`, and test `100/365` images/annotations; handoff check confirms
  consistent `1..200` categories and `valid_test_annotations_identical=false`.

Independent-test detector-side pseudo-label check:

| Setting | Split | Train annotations | Class AP50 | Class AP75 | Class-agnostic AP50 | Off-center AP50 | Small AP50 |
|---|---|---:|---:|---:|---:|---:|---:|
| GT baseline, Nano 1 epoch | true independent test | 1,610 | 0.0224 | 0.0201 | 0.2113 | 0.0154 | 0.0329 |
| GT baseline, Nano 4 epochs | true independent test | 1,610 | 0.1618 | 0.1180 | 0.6151 | 0.1343 | 0.1051 |
| teacher-only, Nano 1 epoch | true independent test | 2,191 | 0.0204 | 0.0182 | 0.1908 | 0.0145 | 0.0038 |
| teacher-only pretrain -> GT finetune, Nano 1+3 epochs | true independent test | 1,610 | 0.1618 | 0.1259 | 0.6170 | 0.1310 | 0.1035 |

Independent-test artifacts:

- `results/rfdetr_offcenter_seed41_indtest_seed43_handoff_check.json`
- `results/rfdetr_indtest_seed43_detector_side_summary.csv`
- `results/rfdetr_indtest_seed43_gt_384_1ep_regular_test_cocoeval.csv`
- `results/rfdetr_indtest_seed43_gt_384_1ep_regular_test_loc_cocoeval.csv`
- `results/rfdetr_indtest_seed43_gt_384_1ep_regular_test_slices.csv`
- `results/rfdetr_indtest_seed43_gt_384_4ep_regular_test_cocoeval.csv`
- `results/rfdetr_indtest_seed43_gt_384_4ep_regular_test_loc_cocoeval.csv`
- `results/rfdetr_indtest_seed43_gt_384_4ep_regular_test_slices.csv`
- `results/rfdetr_indtest_seed43_nano5_teacher_only_s025k20_384_1ep_regular_test_cocoeval.csv`
- `results/rfdetr_indtest_seed43_nano5_teacher_only_s025k20_384_1ep_regular_test_loc_cocoeval.csv`
- `results/rfdetr_indtest_seed43_nano5_teacher_only_s025k20_384_1ep_regular_test_slices.csv`
- `results/rfdetr_indtest_seed43_nano5_teacherpre_gtfinetune_s025k20_384_3ep_regular_test_cocoeval.csv`
- `results/rfdetr_indtest_seed43_nano5_teacherpre_gtfinetune_s025k20_384_3ep_regular_test_loc_cocoeval.csv`
- `results/rfdetr_indtest_seed43_nano5_teacherpre_gtfinetune_s025k20_384_3ep_regular_test_slices.csv`
- `results/rfdetr_offcenter_seed41_full_small_384_resume5ep_indtest_seed43_cocoeval.csv`
- `results/rfdetr_offcenter_seed41_full_small_384_resume5ep_indtest_seed43_loc_cocoeval.csv`
- `results/rfdetr_offcenter_seed41_full_small_384_resume5ep_indtest_seed43_slices.csv`
- `results/rfdetr_offcenter_seed41_full_small_384_resume8ep_indtest_seed43_cocoeval.csv`
- `results/rfdetr_offcenter_seed41_full_small_384_resume8ep_indtest_seed43_loc_cocoeval.csv`
- `results/rfdetr_offcenter_seed41_full_small_384_resume8ep_indtest_seed43_slices.csv`
- `results/rfdetr_offcenter_seed41_full_small_384_seed41_resume12ep_indtest_seed43_cocoeval.csv`
- `results/rfdetr_offcenter_seed41_full_small_384_seed41_resume12ep_indtest_seed43_loc_cocoeval.csv`
- `results/rfdetr_offcenter_seed41_full_small_384_seed41_resume12ep_indtest_seed43_slices.csv`
- `results/rfdetr_offcenter_seed41_full_small_384_seed43_5ep_indtest_seed43_cocoeval.csv`
- `results/rfdetr_offcenter_seed41_full_small_384_seed43_5ep_indtest_seed43_loc_cocoeval.csv`
- `results/rfdetr_offcenter_seed41_full_small_384_seed43_5ep_indtest_seed43_slices.csv`
- `results/rfdetr_offcenter_seed41_full_small_384_seed43_resume8ep_indtest_seed43_cocoeval.csv`
- `results/rfdetr_offcenter_seed41_full_small_384_seed43_resume8ep_indtest_seed43_loc_cocoeval.csv`
- `results/rfdetr_offcenter_seed41_full_small_384_seed43_resume8ep_indtest_seed43_slices.csv`
- `results/rfdetr_offcenter_seed41_full_small_384_seed43_resume12ep_indtest_seed43_cocoeval.csv`
- `results/rfdetr_offcenter_seed41_full_small_384_seed43_resume12ep_indtest_seed43_loc_cocoeval.csv`
- `results/rfdetr_offcenter_seed41_full_small_384_seed43_resume12ep_indtest_seed43_slices.csv`
- `results/rfdetr_indtest_seed43_small_resume8_teacher_only_s025k20_384_1ep_cocoeval.csv`
- `results/rfdetr_indtest_seed43_small_resume8_teacher_only_s025k20_384_1ep_loc_cocoeval.csv`
- `results/rfdetr_indtest_seed43_small_resume8_teacher_only_s025k20_384_1ep_slices.csv`
- `results/rfdetr_indtest_seed43_small_resume8_teacherpre_gtfinetune_s025k20_384_3ep_cocoeval.csv`
- `results/rfdetr_indtest_seed43_small_resume8_teacherpre_gtfinetune_s025k20_384_3ep_loc_cocoeval.csv`
- `results/rfdetr_indtest_seed43_small_resume8_teacherpre_gtfinetune_s025k20_384_3ep_slices.csv`
- `results/rfdetr_indtest_seed43_small_resume8_teacher_only_s025k20_384_1ep_seed43_cocoeval.csv`
- `results/rfdetr_indtest_seed43_small_resume8_teacher_only_s025k20_384_1ep_seed43_loc_cocoeval.csv`
- `results/rfdetr_indtest_seed43_small_resume8_teacher_only_s025k20_384_1ep_seed43_slices.csv`
- `results/rfdetr_indtest_seed43_small_resume8_teacherpre_gtfinetune_s025k20_384_3ep_seed43_cocoeval.csv`
- `results/rfdetr_indtest_seed43_small_resume8_teacherpre_gtfinetune_s025k20_384_3ep_seed43_loc_cocoeval.csv`
- `results/rfdetr_indtest_seed43_small_resume8_teacherpre_gtfinetune_s025k20_384_3ep_seed43_slices.csv`

Distillation interpretation:

- The train-split teacher-prediction path is now functional. Nano 5ep produced
  `98,506` train predictions; filtered pseudo labels gave `2,191` train boxes at
  `score>=0.25/top20`.
- Teacher-only training is slightly better than the 1-epoch GT baseline on
  validation-mirrored heldout class AP50 (`0.0552` vs `0.0495`), so
  detector-side pseudo-label training can move the class head in that protocol.
  However, the true independent-test check reverses this 1-epoch teacher-only
  signal: teacher-only is below GT-only on class AP50 (`0.0204` vs `0.0224`) and
  class-agnostic AP50 (`0.1908` vs `0.2113`). Treat the old teacher-only
  improvement as protocol-sensitive.
- The staged pseudo-pretrain -> GT-finetune schedule does survive the true
  independent split. Using the same Nano5 `score>=0.25/top20` pseudo pretrain
  checkpoint and finetuning on GT for 3 epochs reaches class AP50 `0.1618`,
  AP75 `0.1259`, class-agnostic AP50 `0.6170`, offcenter AP50 `0.1310`, and
  small AP50 `0.1035`. This is above the independent GT-only 1ep and
  teacher-only 1ep checks by a wide margin.
- The equal-total-epoch GT-only 4ep control reaches essentially the same class
  AP50 (`0.1618`) and class-agnostic AP50 (`0.6151`). Staged pretrain keeps a
  modest AP/AP75/localization edge (`AP 0.1261 vs 0.1151`, `AP75 0.1259 vs
  0.1180`, `loc AP50 0.6170 vs 0.6151`), while GT-only 4ep is slightly better
  on offcenter/small/medium AP50. The corrected conclusion is: single-stage
  teacher-only is not robust; staged pseudo pretrain is a useful schedule but
  is not yet a clear equal-budget class-AP50 win. Next confirmation should be
  multi-seed/equal-budget rather than more single-stage teacher-only.
- Resumed Small 384px 5ep is the strongest independent-test detector-side
  checkpoint so far: class AP50 `0.2405`, AP75 `0.1795`, class-agnostic AP50
  `0.6352`, offcenter AP50 `0.2238`, small AP50 `0.2457`, medium AP50
  `0.2572`, and large AP50 `0.3768`. This shifts the active RF-DETR direction
  away from more self-teacher pseudo-label repeats and toward a formal longer
  detector-side Small/Nano protocol. Pseudo-labeling should return only after a
  train-split teacher also beats Nano5 after filtering.
- Extending that same resumed Small run to 8 epochs improves independent-test
  class AP50 further to `0.2677`, AP75 `0.2053`, center AP50 `0.2527`, and
  small AP50 `0.2594`, while class-agnostic AP50 remains high at `0.6271`.
  This is the current strongest RF-DETR checkpoint in the project.
- Extending seed41 further to 12 epochs improves independent-test class AP50
  again to `0.3070`, AP75 to `0.2155`, class-agnostic AP50 to `0.6397`,
  offcenter AP50 to `0.2955`, small AP50 to `0.2911`, and medium AP50 to
  `0.2997`.
- A second direct Small 5ep run with seed43 reaches independent-test class
  AP50 `0.1762`, AP75 `0.1318`, class-agnostic AP50 `0.6327`, offcenter
  AP50 `0.1506`, small AP50 `0.1220`, and large AP50 `0.3263`. It is weaker
  than the seed41 direct Small run, but still beats same-seed Small-resume8
  staged pseudo-pretrain (`0.1762` vs `0.1540` AP50). Direct Small 5ep
  two-seed mean AP50 is about `0.2084`, above Small-resume8 staged-pseudo
  mean about `0.1697`, so direct detector-side Small training remains the main
  RF-DETR route.
- Resuming the same seed43 direct Small run to 8 epochs improves
  independent-test class AP50 to `0.2066`, AP75 to `0.1491`, offcenter AP50 to
  `0.1795`, small AP50 to `0.1697`, and large AP50 to `0.3973`; class-agnostic
  AP50 remains strong at `0.6135`. Direct Small 8ep now has a two-seed mean
  class AP50 of about `0.2371`, confirming that longer direct Small training is
  more reliable than staged pseudo-pretrain under the current independent-test
  protocol.
- Extending seed43 further to 12 epochs gives another large independent-test
  jump: class AP50 `0.2923`, AP75 `0.1802`, class-agnostic AP50 `0.6290`,
  offcenter AP50 `0.2562`, small AP50 `0.2921`, and large AP50 `0.4235`. This
  confirms the direct-Small long-train route with a two-seed 12ep mean class
  AP50 of about `0.2997`. This is now the strongest current RF-DETR route in
  the project.
- Resuming seed41 direct Small to 16 epochs further improves the independent
  test result despite non-monotonic RF-DETR internal validation logs. Official
  regular-checkpoint export gives class AP50 `0.3209`, AP75 `0.2268`, AP
  `0.2206`, loc AP50 `0.6374`, offcenter AP50 `0.2881`, center AP50 `0.2951`,
  small AP50 `0.3258`, medium AP50 `0.2987`, and large AP50 `0.4014`. Direct
  Small long training remains the active RF-DETR mainline; staged pseudo
  pretraining is now clearly secondary.
- Resuming seed41 direct Small to 20 epochs gives the largest independent-test
  jump so far. Official regular-checkpoint export gives class AP50 `0.3625`,
  AP75 `0.2566`, AP `0.2573`, loc AP50 `0.6273`, offcenter AP50 `0.3241`,
  center AP50 `0.3046`, small AP50 `0.3334`, medium AP50 `0.2885`, and large
  AP50 `0.4588`. This extends the seed41 direct-Small curve from
  `0.2405 -> 0.2677 -> 0.3070 -> 0.3209 -> 0.3625` class AP50 across
  `5/8/12/16/20` epochs. Current decision: keep direct Small long training as
  the main RF-DETR path, and next confirm with seed43 rather than returning to
  pseudo-label distillation or persistent proposal-state experiments.
- The seed43 confirmation is mixed rather than monotonic. Resuming seed43 from
  12 to 16 epochs lowers class AP50 from `0.2923` to `0.2753`, while improving
  AP75 from `0.1802` to `0.1957`, class-agnostic AP50 from `0.6290` to
  `0.6348`, medium AP50 from `0.2555` to `0.2780`, and large AP50 from
  `0.4235` to `0.4359`. This keeps direct Small as the active path, but it
  turns the next question into epoch selection / LR schedule rather than simply
  pushing every seed longer.
- The compact epoch-selection artifact is
  `results/rfdetr_direct_small_epoch_curve_indtest_seed43.csv`, generated by
  `scripts/summarize_rfdetr_epoch_curve.py`. Use it before launching more
  RF-DETR long-train jobs.
- A lower-LR fresh-optimizer continuation fixes the seed43 class-AP50 drop.
  Loading seed43 12ep `checkpoint_best_regular.pth` through `--pretrain-weights`
  and training 4 epochs with `lr=3e-5` reaches independent-test class AP50
  `0.3102`, AP75 `0.2190`, AP `0.2114`, loc AP50 `0.6316`, offcenter AP50
  `0.2755`, center AP50 `0.2832`, small AP50 `0.2890`, medium AP50 `0.3025`,
  and large AP50 `0.4137`. This beats both seed43 12ep (`0.2923` AP50) and
  resume16 (`0.2753` AP50). Current decision: the next direct-Small schedule
  should favor low-LR fresh-optimizer continuation from a strong checkpoint,
  not blind resume of the old optimizer state.
- Small-resume8 also clears the train-teacher gate: raw train AP50 `0.4599`
  beats Nano5 raw train AP50 `0.4474`, and filtered pseudo AP50 beats Nano5 at
  the same filters (`s015/top20 0.4221 vs 0.4098`, `s020/top20 0.4033 vs
  0.3991`, `s025/top20 0.3832 vs 0.3689`). This makes Small-resume8 the first
  checked candidate worth using for a new teacher-only pretrain -> GT finetune
  schedule.
- Small-resume12 seed41 now supersedes Small-resume8 as the checked train
  teacher. Its train raw class AP50 is `0.6764` and class-agnostic AP50 is
  `0.8741`. Filtered pseudo labels also clear the gate by a wide margin:
  `s015/top20` keeps `3,323` boxes with class AP50 `0.6430` and loc AP50
  `0.8508`; `s020/top20` keeps `2,608` boxes with class AP50 `0.6260` and loc
  AP50 `0.8457`; `s025/top20` keeps `2,110` boxes with class AP50 `0.6170` and
  loc AP50 `0.8275`. This is the first train-split teacher that is strong
  enough to justify a renewed teacher-only pretrain -> GT finetune check. It
  does not revive raw GT+pseudo appending.
- The renewed staged check is negative for the main route. Using
  Small-resume12 `s015/top20` pseudo labels for 1 epoch and then GT-finetuning
  for 3 epochs reaches independent-test class AP50 `0.1601`, AP75 `0.1038`,
  loc AP50 `0.6258`, offcenter AP50 `0.1389`, small AP50 `0.1276`, medium AP50
  `0.2103`, and large AP50 `0.2542`. This improves over the same teacher-only
  checkpoint (`0.0629` AP50) and is close to the old weak-student staged range,
  but it is below Small-resume8 seed41 staged (`0.1855` AP50) and far below
  direct Small 12ep (`0.3070` / `0.2923` AP50 across seeds). Static pseudo-label
  quality is therefore not sufficient as a training-success predictor.
  Teacher-only pretrain remains a diagnostic / warm-start branch, not the
  active RF-DETR mainline.
- The seed43 direct Small resume8 checkpoint is not a stronger train teacher.
  Its raw train class AP50 is `0.4274`, below seed41 Small-resume8 `0.4599`.
  After filtering, `s015/s020/s025 top20` class AP50 is
  `0.3804/0.3638/0.3475`, also below seed41 Small-resume8
  `0.4221/0.4033/0.3832`. Use seed43 resume8 as a second direct-training
  detector result, not as the next pseudo-label teacher.
- Small-resume8 `s025/top20` teacher-only 1ep is still only a weak standalone
  model on independent test: class AP50 `0.0547`, AP75 `0.0504`,
  class-agnostic AP50 `0.3288`, offcenter AP50 `0.0437`, and small AP50
  `0.0337`. It is numerically above the earlier GT-only 1ep and old Nano5
  teacher-only 1ep checks on this split, but those are not same-seed controls;
  it remains far below direct GT 4ep and Small long-train. Use it as a
  pretraining-stage candidate, not as a final detector.
- Small-resume8 `s025/top20` teacher-only pretrain followed by 3 GT-finetune
  epochs is a useful but not decisive weak-student warm-start. Seed41 reaches
  independent-test class AP50 `0.1855`, AP75 `0.1386`, class-agnostic AP50
  `0.6300`, offcenter AP50 `0.1486`, small AP50 `0.1269`, and large AP50
  `0.3164`. Seed43 is weaker: teacher-only AP50 `0.0493`, staged 1+3 class
  AP50 `0.1540`, AP75 `0.1035`, class-agnostic AP50 `0.6300`, offcenter AP50
  `0.1342`, small AP50 `0.1251`, and large AP50 `0.2539`. Across the two
  seeds, the staged AP50 mean is about `0.1697`, but seed43 is below same-seed
  GT-only 4ep (`0.1618`). The current decision is to keep stronger-teacher
  pseudo-pretrain as a weak-Nano warm-start diagnostic, not as a robust
  equal-budget win and not as a replacement for direct detector-side Small long
  training.
- Naively appending pseudo labels to GT is negative: GT+pseudo lowers class AP50
  to `0.0484` and class-agnostic AP50 to `0.3737`, suggesting duplicate/noisy
  pseudo boxes interfere with Hungarian matching.
- Historical validation-mirrored result: teacher-only pretraining followed by GT
  finetuning was the first positive detector-side distillation schedule under
  the old mirrored protocol, reaching class AP50 `0.1428` at Nano5 1+3 and
  reproducing around `0.1413` with a second seed. The independent-test and
  equal-budget checks above supersede the older strong wording: staged
  pseudo-pretrain remains useful as a warm-start / localization route, but the
  self-teacher Nano5 result is not a clear class-AP50 win over equal-budget
  GT-only, and Small-resume8 direct training is currently stronger.
- The staged 1+3 student is not a better train-split pseudo-label teacher under
  the current pseudo protocol. On train-split predictions, the original Nano5
  teacher is stronger (`AP50 0.4474`, loc AP50 `0.7814`) than the staged13
  checkpoint (`AP50 0.3137`, loc AP50 `0.7380`). A teacher-only 1ep run using
  staged13 pseudo labels reaches only class AP50 `0.0504`, below the original
  Nano5 teacher-only result (`0.0552`). Treat this as a teacher-selection/filtering
  warning, not as a failure of staged pretrain -> GT finetune itself.
- Lengthening the GT finetune to 4 epochs does not improve class AP50 (`0.1342`)
  even though class-agnostic AP50 (`0.6290`) and large-object AP50 (`0.3127`)
  increase. Current self-teacher schedule should stop at 1ep pseudo pretrain +
  3ep GT finetune for class-aware detection, then move to stronger teacher or
  multi-seed formalization instead of simply extending epochs.
- Engineering caveat: RF-DETR selected an anomalous `best_total` from stale/high
  EMA (`0.7167`) in this pretrain-then-finetune flow. The reported staged
  results use `checkpoint_best_regular.pth`, exported independently through
  `predict_rfdetr_coco.py`.
- This is not yet a strong-teacher result. It is a substrate check using Nano 5ep
  as a self-teacher. The next useful step is a stronger train-split teacher, a
  more formal held-out multi-seed staged-finetune protocol, or stricter pseudo
  filtering; raw GT+pseudo concatenation should stay downgraded.

Frozen DET-aware category-prior check:

| Setting | Eval top-1 | Eval top-5 | AP50 | Top-100 class-aware R@50 | Top-100 mean class-aware IoU |
|---|---:|---:|---:|---:|---:|
| frozen ResNet50 DET linear head, top-5 multiply | 0.1143 | 0.4286 | 0.0223 | 0.3797 | 0.1173 |
| frozen ResNet50 DET linear head, top-5 keep-score | 0.1143 | 0.4286 | 0.0139 | 0.3797 | 0.0034 |
| frozen ResNet50 DET linear head, calibrated top-5 multiply | 0.1143 | 0.4286 | 0.0361 | 0.3797 | 0.0281 |

Frozen DET-prior artifacts:

- `results/rfdetr_max3_frozen_resnet50_detprior_top5x_heldout_summary.csv`
- `results/rfdetr_max3_frozen_resnet50_detprior_top5x_heldout_slices.csv`
- `results/rfdetr_max3_frozen_resnet50_detprior_top5x_heldout_prediction_recall.csv`
- `results/rfdetr_max3_frozen_resnet50_detprior_top5x_heldout_score_iou.csv`
- `results/rfdetr_max3_frozen_resnet50_detprior_top5keep_heldout_summary.csv`
- `results/rfdetr_max3_frozen_resnet50_detprior_top5keep_heldout_slices.csv`
- `results/rfdetr_max3_frozen_resnet50_detprior_top5keep_heldout_prediction_recall.csv`
- `results/rfdetr_max3_frozen_resnet50_detprior_top5keep_heldout_score_iou.csv`
- `results/rfdetr_max3_frozen_resnet50_detprior_top5x_calibrated_heldout_summary.csv`
- `results/rfdetr_max3_frozen_resnet50_detprior_top5x_calibrated_heldout_slices.csv`
- `results/rfdetr_max3_frozen_resnet50_detprior_top5x_calibrated_heldout_prediction_recall.csv`
- `results/rfdetr_max3_frozen_resnet50_detprior_top5x_calibrated_heldout_score_iou.csv`
- `results/rfdetr_frozen_detprior_heldout_summary.csv`

Proposal-crop DET category-prior check:

| Setting | Eval crop top-1 | Eval crop top-5 | AP50 | Top-100 class-aware R@50 | Top-100 mean class-aware IoU |
|---|---:|---:|---:|---:|---:|
| frozen ResNet50 proposal-crop head, top-5 multiply | 0.2911 | 0.4367 | 0.0816 | 0.3291 | 0.1633 |
| frozen ResNet50 proposal-crop head, calibrated top-5 multiply | 0.2911 | 0.4367 | 0.0824 | 0.3291 | 0.1672 |

Proposal-crop artifacts:

- `scripts/train_coco_proposal_crop_classifier.py`
- `results/rfdetr_max3_frozen_resnet50_propcrop_top5x_heldout_summary.csv`
- `results/rfdetr_max3_frozen_resnet50_propcrop_top5x_heldout_slices.csv`
- `results/rfdetr_max3_frozen_resnet50_propcrop_top5x_heldout_prediction_recall.csv`
- `results/rfdetr_max3_frozen_resnet50_propcrop_top5x_heldout_score_iou.csv`
- `results/rfdetr_max3_frozen_resnet50_propcrop_top5x_full_summary.csv`
- `results/rfdetr_max3_frozen_resnet50_propcrop_top5x_calibrated_heldout_summary.csv`
- `results/rfdetr_max3_frozen_resnet50_propcrop_top5x_calibrated_heldout_slices.csv`
- `results/rfdetr_max3_frozen_resnet50_propcrop_top5x_calibrated_heldout_prediction_recall.csv`
- `results/rfdetr_max3_frozen_resnet50_propcrop_top5x_calibrated_heldout_score_iou.csv`
- `results/rfdetr_propcrop_prior_heldout_summary.csv`

Image-prior / proposal-crop fusion check:

| Setting | AP50 | Off-center AP50 | Top-100 mean class-aware IoU |
|---|---:|---:|---:|
| ResNet50 image top-5x + calibration | 0.1878 | 0.1933 | 0.2418 |
| Image prior and proposal-crop intersection, geomean | 0.1049 | 0.0557 | 0.0227 |
| Keep ImageNet primary, boost categories also seen by proposal-crop | 0.1879 | 0.1816 | 0.1066 |

Fusion artifacts:

- `scripts/fuse_coco_category_prior_predictions.py`
- `results/rfdetr_max3_resnet50_image_propcrop_geomean_intersection_heldout_slices.csv`
- `results/rfdetr_max3_resnet50_image_propcrop_geomean_intersection_heldout_prediction_recall.csv`
- `results/rfdetr_max3_resnet50_image_propcrop_geomean_intersection_heldout_score_iou.csv`
- `results/rfdetr_max3_resnet50_image_propcrop_geomean_keepprimary_heldout_slices.csv`
- `results/rfdetr_max3_resnet50_image_propcrop_geomean_keepprimary_heldout_score_iou.csv`
- `results/rfdetr_category_prior_fusion_heldout_summary.csv`

RF-DETR native category-id transform check:

| Setting | AP50 |
|---|---:|
| Native category IDs | 0.0058 |
| Native category IDs, best small offset (+3) | 0.0194 |
| Class-agnostic reference on same class-preserving NMS file | 0.0854 |
| Max3 + ResNet50 image top-5 prior | 0.1697 |
| Max3 + nearest-GT category, original score | 0.3712 |

Native category transform artifacts:

- `scripts/evaluate_coco_category_id_transforms.py`
- `results/rfdetr_native_category_transform_max3_fused_full_summary.csv`

## Interpretation

- More crop coverage still improves class-agnostic localization.
- Class-aware AP saturates after max2/max3 because category prediction and
  score-IoU ranking dominate the remaining error.
- ResNet50 top-5 category expansion is the strongest checked non-oracle
  category-transfer setting.
- Naive lightweight teachers and simple teacher ensembles do not beat ResNet50.
- ResNet50 any-GT top-5 coverage is only `0.3913`, so ImageNet category
  coverage itself is a bottleneck, not just top-k ranking.
- Score-IoU alignment remains weak after category transfer. ResNet50 top-5
  improves top-ranked box quality but does not produce a calibrated IoU score.
- A tiny calibration-split post-hoc quality scorer gives a modest held-out
  gain, especially on off-center images. This supports score calibration as a
  useful next branch, but the gain is not large enough to treat simple
  calibration as the main detector solution.
- Prediction recall makes the category bottleneck explicit: class-agnostic
  fused boxes cover `78.5%` of held-out GT at top-100 / IoU 0.5, while the
  class-aware ResNet50 top-5x candidates cover only `32.3%`. Calibration helps
  top-10 ordering slightly, but cannot recover category coverage.
- A frozen ResNet50 feature head trained on the local DET train split improves
  top-5 label coverage but fails as a detection scorer: top-100 class-aware
  recall reaches `38.0%`, yet AP50 drops to `0.0223` with multiplied scores and
  `0.0139` with kept detector scores. Calibration improves it to `0.0361` AP50,
  still far below the pretrained ImageNet ResNet50 prior. The local split is
  too small/noisy for a standalone DET category head to replace the pretrained
  ImageNet prior.
- A proposal-crop classifier is better aligned with detection than the image
  DET-prior head: held-out AP50 reaches `0.0816` and GT-crop top-1 reaches
  `0.2911`. However, it still trails the pretrained ImageNet full-image prior
  (`0.1878` held-out AP50 after calibration), and calibration only lifts it to
  `0.0824`. This is a useful mechanism signal but not a deployable category
  source yet.
- Simple fusion between ImageNet image prior and proposal-crop prior does not
  improve the current best setting. Intersection fusion throws away too much
  recall, while keep-primary fusion is effectively flat on AP50 and worse on
  off-center AP50. The next category step needs a stronger teacher or integrated
  detector-side class head, not naive post-hoc prior fusion.
- The native RF-DETR category failure is not explained by a simple category-id
  offset. A small offset sweep around the native IDs peaks at `0.0194` AP50,
  still far below the ImageNet-prior and oracle-category settings.
- Category coverage, not intra-image rank order, is the main remaining
  category bottleneck. Under global top-K evaluation, top-100 localization recall
  is about `0.75-0.77`, but class-aware recall is only `0.31-0.32`; the
  per-category-topK recall is only `0.006-0.013` higher. This means most loss is
  from missing/incorrect category candidates, not from correct categories being
  buried far down the global ranking.
- Existing torchvision category sources do not beat the current max3 ResNet50
  top-5 prior. ConvNeXt/EfficientNet and simple teacher ensembles are not enough
  to close the category retention gap; the next category route should be a
  stronger detector-aware teacher or an integrated detector-side class head.
- Candidate-constrained oracle confirms the same bottleneck. The current top-5
  candidate set contains the nearest-GT category for only `20.4%` of grouped
  proposal boxes. Even with oracle IoU scoring, this constrained candidate set
  reaches AP50 `0.4736`, well below the all-category IoU oracle AP50 `0.7515`.
  This rules out pure candidate reranking as the main next step.
- Training a frozen ResNet50 head on matched RF-DETR proposal crops does not
  solve the category source problem. Despite `0.6667` heldout GT-crop top-5
  accuracy, the resulting proposal relabeling is far below the ImageNet image
  prior on AP50 and does not improve top-100 class recall. This suggests the
  small matched-proposal split is too weak/noisy for a standalone crop classifier.
- Integrated RF-DETR Nano full-split finetuning confirms that class-head
  training must happen inside the detector, and longer training immediately
  helps. The 384px run improves from `0.0495` class-aware AP50 at 1 epoch to
  `0.1224` at 3 epochs, while class-agnostic AP50 remains high (`0.6074` at
  3 epochs). This is still below the max3 crop + ResNet50 top-5 prior AP50
  (`0.1697`), but the direction is now clearly better than standalone
  post-hoc crop-prior tuning. The category route should therefore move toward
  longer detector-side class-head training, stronger teacher initialization, or
  detector-side distillation rather than another external crop-prior patch.

## Stop / Continue Rule

Do not continue max4/max5 crop expansion unless category scoring improves first.

Next useful directions:

1. Stronger detector-aware category teacher or teacher distillation into the
   detector-side class head.
2. Longer integrated RF-DETR class-head finetune protocol with explicit
   class-aware and class-agnostic reporting.
3. Category-aware quality calibration on an image-disjoint split after the
   detector-side class head is strong enough to provide candidate coverage.
4. If trying DET-aware category again, prefer detector-integrated training or
   distillation rather than post-hoc crop-prior relabeling.

Avoid:

- Returning to persistent proposal state.
- Sweeping alpha/temperature on the same eval split.
- Claiming crop finetuning solves small-object detection.
