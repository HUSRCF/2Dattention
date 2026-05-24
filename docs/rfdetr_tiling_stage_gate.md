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
- The same lower-LR fresh-optimizer continuation does not push the already
  strong seed41 20ep checkpoint higher. Loading seed41 20ep
  `checkpoint_best_regular.pth` and training 4 more epochs at `lr=3e-5`
  reaches independent-test class AP50 `0.3546`, AP75 `0.2396`, AP `0.2494`,
  loc AP50 `0.6304`, offcenter AP50 `0.3084`, center AP50 `0.3186`, small
  AP50 `0.3111`, medium AP50 `0.3751`, and large AP50 `0.3888`. This is below
  the seed41 20ep peak class AP50 `0.3625`, although medium AP50 improves
  strongly. Updated decision: low-LR fresh continuation is a useful rescue for
  a seed that has started to overstep, not a universal way to extend every
  strong checkpoint. Stop blind schedule sweeping; the next RF-DETR progress
  should come from more data/resolution or a stronger detector-side training
  protocol.
- A larger image-disjoint RF-DETR protocol is now prepared and runnable:
  `rfdetr_random_seed41_train1000_val200_test200` has train/valid/test
  `1000/200/200` images, `2894/681/611` annotations, and the same preserved
  200-category table. A Small 384px seed41 2ep run is not a performance
  competitor yet, but it is a useful cost and direction gate: official test
  export from `checkpoint_best_regular.pth` gives class AP50 `0.0864`, AP75
  `0.0741`, AP `0.0708`, class-agnostic AP50 `0.5550`, offcenter AP50
  `0.0663`, center AP50 `0.1017`, small AP50 `0.0855`, medium AP50 `0.0908`,
  and large AP50 `0.1878`. This confirms the larger split has usable
  localization after only two epochs, but the class head is severely underfit.
  Treat this as the next formal protocol entry point; it needs longer
  detector-side training rather than another short LR/alpha sweep.
- Continuing that 2ep checkpoint for 4 more epochs with a fresh optimizer at
  `lr=1e-4` validates the direction. Official test export now gives class AP50
  `0.2196`, AP75 `0.1841`, AP `0.1746`, class-agnostic AP50 `0.6111`,
  offcenter AP50 `0.2073`, center AP50 `0.2067`, small AP50 `0.1890`,
  medium AP50 `0.2248`, and large AP50 `0.3263`. The improvement over the 2ep
  gate is large (`0.0864 -> 0.2196` class AP50), and class-agnostic AP50 also
  improves (`0.5550 -> 0.6111`). This establishes the larger split as a valid
  long-train protocol rather than a failed smoke. Continue this protocol only
  with longer detector-side training or a clearly stronger training recipe.
- Continuing the same train1000 route in repeated 4-epoch fresh-optimizer stages
  keeps strengthening the result. Official test export from the 6best regular
  checkpoint gives class AP50 `0.2912`, AP75 `0.2395`, AP `0.2190`,
  class-agnostic AP50 `0.6265`, AP75 `0.5350`, offcenter AP50 `0.2603`, small
  AP50 `0.2458`, medium AP50 `0.2865`, and large AP50 `0.3646`. The next
  10best regular checkpoint improves again to class AP50 `0.3155`, AP75
  `0.2517`, AP `0.2394`, class-agnostic AP50 `0.6427`, offcenter AP50
  `0.3081`, small AP50 `0.2792`, medium AP50 `0.3167`, and large AP50
  `0.3572`. The staged progression on this larger split is now class AP50
  `0.0864 -> 0.2196 -> 0.2912 -> 0.3155`, so the active RF-DETR direction is no
  longer schedule tweaking on the old 330-image split. It is controlled,
  low-frequency longer detector-side training on the 1000-image image-disjoint
  protocol. Caveat: the last block's internal validation dipped at the final
  epoch, so keep using independent test export from `checkpoint_best_regular.pth`
  for conclusions.
- RF-DETR deformable attention visualization is now available through
  `scripts/visualize_rfdetr_deformable_attention.py`. The script hooks
  `MSDeformAttn`, aggregates captured deformable cross-attention sampling
  locations, and overlays sampling density with GT boxes and top predicted
  boxes. The first smoke artifact is
  `results/rfdetr_attention_overlays_train1000_6best/contact_sheet.jpg`, with
  per-image GT-box attention mass, top-prediction attention mass, entropy, and
  peak location in `manifest.json`. Treat it as a qualitative sampling-coverage
  diagnostic, not as a ViT-style full attention map and not as a quantitative
  result.
- The same diagnostic now supports query-level overlays with
  `--query-overlays K`. This path preserves RF-DETR query indices before
  postprocessing, then renders the sampled locations for each top prediction's
  originating query. The first artifact is
  `results/rfdetr_attention_query_overlays_train1000_6best/query_contact_sheet.jpg`.
  Use this for failure analysis: a high-score query with low or zero
  `query_gt_attention_mass` indicates a query/representation alignment failure,
  while high GT mass with a bad box points to box refinement or scoring.
- `scripts/summarize_rfdetr_attention_manifest.py` now flattens the attention
  manifest into image-level and query-level CSVs plus optional JSON diagnostics.
  On the first 6-image off-center smoke, top-query GT attention mass has mean
  `0.605`, median `0.818`, and `2/14` zero-GT-mass query failures; `9/14` top
  queries have nearest-GT IoU >= `0.5`, with mean IoU `0.581` and median IoU
  `0.773`. The JSON diagnostic reports strong mass-IoU alignment in this tiny
  sample (`Pearson=0.950`, `Spearman=0.753`), with high-IoU queries averaging
  `0.873` GT mass and low-IoU queries averaging `0.124`. This is not a
  benchmark, but it gives a concrete filter for selecting failure cases before
  deciding whether the next fix should target query representation, box
  refinement, or scoring.
- A larger top-1 off-center mining pass is available at
  `results/rfdetr_attention_top1_offcenter40_train1000_6best/`. It covers 40
  test images with one query overlay per image and writes bucket contact sheets
  under `bucket_contact_sheets/`. Summary: mean nearest-GT IoU `0.751`, median
  `0.858`, IoU50 `32/40`, mean GT attention mass `0.834`, and mass-IoU
  correlations `Pearson=0.726`, `Spearman=0.359`. Bucket counts are `32`
  aligned hits, `5` high-mass/low-IoU failures, `2` aligned misses, and `1`
  zero-mass miss. The new practical diagnostic split is: zero/low-mass misses
  point to query representation coverage, while high-mass/low-IoU cases point
  to box refinement, duplicates, assignment, or scoring rather than a pure
  attention-coverage failure.
- The full 200-image test split now has the same top-1 query attention summary
  at `results/rfdetr_attention_top1_test200_train1000_6best/`. Top-1 overlays
  are available for `196/200` images above threshold. Mean nearest-GT IoU is
  `0.835`, median `0.935`, IoU50 is `177/196`, mean GT attention mass is
  `0.814`, and mass-IoU correlations are `Pearson=0.687`, `Spearman=0.242`.
  Buckets: `177` aligned hits, `8` high-mass/low-IoU, `8` aligned misses, and
  `3` zero-mass. Nearest-GT category match is `149/196=0.760`, and rises to
  `147/177=0.831` among IoU50 top-1 queries. This confirms the attention
  diagnostic should not be framed as "RF-DETR cannot look at off-center
  objects"; for most top predictions it does sample target support and often
  predicts the nearest GT category correctly. The remaining class AP gap is more
  likely a full-set ranking, duplicate/false-positive, and long-tail candidate
  coverage issue than a universal top-query representation failure.
- Full-prediction diagnostics confirm that interpretation. For the same 6best
  checkpoint, top100 recall at IoU 0.5 is `loc=0.872`, `class-aware=0.665`, and
  `per-category=0.717`; offcenter top100 is `0.812/0.577/0.624`, and small
  top100 is `0.720/0.543/0.603`. Full-set score-IoU correlation is weak despite
  the strong top-1 queries (`loc Pearson=0.058`, Spearman=`-0.009`;
  class-aware Pearson=`0.386`, Spearman=`0.059`). The next detector-side work
  should therefore improve category candidate coverage and ranking/calibration
  on the full prediction set, not chase a generic "attention cannot see the
  object" explanation.
- The same diagnostics on the 10best checkpoint show coverage is improving but
  score calibration is still weak. Top100 coverage rises to `loc=0.876`,
  `class-aware=0.683`, and `per-category=0.741`; offcenter top100 becomes
  `0.826/0.594/0.658`, and small top100 becomes `0.728/0.595/0.647`.
  Full-set score-IoU correlation remains low (`loc Pearson=0.064`,
  Spearman=`0.018`; class-aware Pearson=`0.361`, Spearman=`0.027`). This
  supports continuing detector-side training for candidate coverage while
  keeping ranking/calibration as a separate bottleneck.
- A valid-split post-hoc calibrator on the 10best checkpoint confirms that
  score-IoU correlation can be fitted without test-set tuning, but it is not
  sufficient by itself. It raises class-aware score-IoU Spearman from `0.027`
  to `0.291` and loc Spearman from `0.018` to `0.120`, but class-aware COCOeval
  is essentially unchanged (`AP50 0.3155 -> 0.3158`, `AP75 0.2517 -> 0.2521`)
  and class-aware top100 mean IoU drops from `0.9418` to `0.9283`. Offcenter
  AP50 only moves from `0.3081` to `0.3120`, while small AP50 moves from
  `0.2792` to `0.2767`. Treat this as a diagnostic: post-hoc score-rescoring
  sweeps are not the next lever without a stronger candidate-generation or
  class-head change.
- Continuing the 10best checkpoint with a fresh optimizer at lower LR `5e-5`
  for 4 epochs gives a small class AP gain but does not change the diagnosis.
  The best regular checkpoint is continuation epoch 1 (`val mAP/AP50/AP75 =
  0.2454/0.3231/0.2557`); later epochs regress. Independent test reaches class
  `AP/AP50/AP75 = 0.2421/0.3242/0.2523`, but localization drops to
  `loc AP/AP50/AP75 = 0.4733/0.6271/0.5081`. Offcenter AP50 is roughly flat
  (`0.3081 -> 0.3064`), small AP50 improves (`0.2792 -> 0.2980`), and large
  AP50 improves (`0.3572 -> 0.3864`). Coverage/ranking diagnostics remain
  similar (`top100 loc/class/percat = 0.872/0.689/0.733`; class-aware
  score-IoU Pearson/Spearman `0.375/0.033`). This supports low-frequency longer
  RF-DETR training as an incremental baseline improver, not a replacement for
  candidate coverage and class/ranking mechanism work.
- The same valid-split calibrator is more useful on the 14best checkpoint than
  on 10best. It improves class `AP/AP50/AP75` from `0.2421/0.3242/0.2523` to
  `0.2494/0.3297/0.2593`, raises offcenter AP50 from `0.3064` to `0.3248`, and
  raises large AP50 from `0.3864` to `0.3960`. Score-IoU correlation improves
  (`class-aware Pearson/Spearman 0.375/0.033 -> 0.413/0.274`; loc Spearman
  `0.071 -> 0.158`), while top100 class-aware coverage slightly drops
  (`0.689 -> 0.678`) and top100 offcenter class-aware recall moves from
  `0.604` to `0.597`. This keeps the route split clean: stronger detector-side
  training makes post-hoc quality ranking more useful, but the calibrator is
  still reordering existing predictions rather than generating missing
  candidates/categories.
- Oracle rescoring on the 14best checkpoint quantifies that remaining gap.
  Same-category oracle IoU scoring reaches class `AP/AP50/AP75 =
  0.452/0.569/0.473`, so there is still large ranking-quality headroom when
  the predicted category is already correct. Nearest-GT category relabeling
  while keeping the detector's original score reaches `0.469/0.648/0.482`,
  which is much higher than the valid-calibrated score path; this makes
  category assignment/candidate coverage the dominant class-aware AP50
  bottleneck. Full nearest-GT relabel + IoU score gives `0.433/0.512/0.471`,
  lower AP50 because duplicate ordering and COCO precision behavior change
  under pure IoU scoring. Treat these oracle paths as diagnostics, not
  deployable inference recipes.
- Per-category coverage diagnostics turn the category bottleneck into a
  concrete target list. At `top100 / IoU 0.5 / gt_count >= 3`, some classes
  have near-perfect localization coverage but zero class-aware recall:
  `n02503517` (`gt=11`, loc `1.000`, class `0.000`), `n03188531`
  (`gt=11`, loc `1.000`, class `0.000`), and `n04004767` (`gt=5`, loc
  `1.000`, class `0.000`). Valid-calibrated scoring does not repair these
  failures, and can worsen class recall while localization remains high. This
  shifts the next detector-side work toward category assignment / class-head
  candidate coverage rather than more score-only calibration.
- High-IoU category-confusion diagnostics show where those failures go. On
  14best `top100 / IoU 0.5`, representative wrong-class flows include
  `n07714571 -> n07739125` (`14/18`, mean IoU `0.894`), `n03188531 ->
  n07747607` (`6/11`, mean IoU `0.916`; calibrated `9/11`), and `n02799071 ->
  n03720891` (`5/9`, mean IoU `0.904`). These are high-overlap boxes with
  wrong categories, so the failure is semantic/category assignment after
  localization rather than missing object support alone.
- Joining per-category coverage with split counts reveals a protocol caveat.
  In this random train1000 split, `2` test-positive categories have zero train
  boxes and `18` have fewer than `3` train boxes; among categories with
  `test_gt >= 3`, six have `train_gt < 3`. Some of the largest gaps are
  therefore not fair class-head architecture failures: `n02503517` has
  `test=11/train=0`, `n03188531` has `test=11/train=2`, `n04004767` has
  `test=5/train=2`, `n07714571` has `test=18/train=1`, and `n02799071` has
  `test=9/train=1`. Treat the current train1000 protocol as a useful
  detector-side smoke benchmark, but the next serious RF-DETR protocol should
  be category-stratified or should enforce minimum train coverage for evaluated
  classes.
- A category-stratified replacement split is now prepared by
  `scripts/build_stratified_rfdetr_coco_split.py`:
  `rfdetr_stratified_seed41_train1000_val200_test200_min3` keeps the same
  `1000/200/200` image counts, preserves the full 200-category table, and
  guarantees at least `3` train boxes per category (`zero=0`, `lt_min=0`,
  `min=3`). The RF-DETR handoff check passes with no missing files, no invalid
  boxes, matching category tables, and `valid_test_annotations_identical=false`.
  Use this as the next serious RF-DETR benchmark before attributing class gaps
  to architecture.
- RF-DETR Small 2ep smoke on this stratified split completes cleanly and
  establishes the new protocol baseline. Independent test class
  `AP/AP50/AP75 = 0.0949/0.1122/0.1001`, localization `AP/AP50/AP75 =
  0.4111/0.5426/0.4504`, offcenter AP50 `0.0871`, center AP50 `0.1348`,
  small AP50 `0.0646`, medium AP50 `0.1300`, and large AP50 `0.2094`.
  Category coverage confirms the split fix (`train_zero=0`, `train_lt3=0`
  among test-positive categories), but top100 still has loc/class/percat
  recall `0.8295/0.5513/0.6457`, so high-localization / low-class gaps remain
  for trained categories. Treat this as the fairer 2ep smoke baseline for the
  next serious RF-DETR run, not as a direct comparison against the random-split
  14best long-trained checkpoint.
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

## Stratified RF-DETR Split Check

The original random `train1000/valid200/test200` split was later found to have
category-coverage artifacts: some test-positive categories had zero or very few
training boxes. A category-stratified replacement was built at
`data/ILSVRC2013_DET_val_supervised/rfdetr_stratified_seed41_train1000_val200_test200_min3`
with train `1000` images / `4409` boxes, valid `200` images, and test `200`
images / `604` boxes. It enforces at least `3` train boxes for every category
present in the test split.

| Setting | Split | Class AP | Class AP50 | Class AP75 | Loc AP | Loc AP50 | Loc AP75 | Offcenter AP50 | Small AP50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Small 384, 2ep | stratified train1000 | 0.0949 | 0.1122 | 0.1001 | 0.4111 | 0.5426 | 0.4504 | 0.0871 | 0.0646 |
| Small 384, 2best + 4ep fresh lr1e-4 | stratified train1000 | 0.1543 | 0.1869 | 0.1645 | 0.4206 | 0.5584 | 0.4484 | 0.1675 | 0.1544 |

Stratified long-run artifacts:

- `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2ep_test_cocoeval.csv`
- `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2ep_test_loc_cocoeval.csv`
- `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2ep_test_slices.csv`
- `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_freshlr1e4_4ep_test_cocoeval.csv`
- `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_freshlr1e4_4ep_test_loc_cocoeval.csv`
- `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_freshlr1e4_4ep_test_slices.csv`
- `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_score_iou_loc.csv`
- `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_score_iou_classaware.csv`
- `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_category_coverage_gap.csv`
- `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_per_category_coverage_gap_with_split_counts.csv`

Stratified interpretation:

- The split-coverage issue is fixed for this protocol: test-positive categories
  are no longer absent from training.
- The detector-side class head still remains the limiting factor. At top-100 /
  IoU 0.5, localization recall is `0.8427`, global class-aware recall is
  `0.6424`, and per-category class recall is `0.7119`.
- Score-IoU alignment remains weak despite improved AP: loc Spearman is
  `-0.0036`, class-aware Spearman is `0.0862`, while top-100 boxes have high
  nearest IoU (`0.9053` loc, `0.8891` class-aware).
- Off-center and small slices remain weaker than center/large: AP50 is
  `0.1675` off-center versus `0.1877` center, and `0.1544` small versus
  `0.2968` large.
- Per-category diagnostics still show trained categories with high localization
  recall and zero class recall, e.g. `n07695742`, `n04468005`, `n03790512`,
  `n03761084`, and `n02992211`. Therefore the remaining category gap is not
  explained only by missing train coverage.

Stratified oracle headroom:

| Diagnostic | AP | AP50 | AP75 | Meaning |
|---|---:|---:|---:|---|
| Base regular checkpoint | 0.1543 | 0.1869 | 0.1645 | Actual detector output |
| Same-category IoU-score oracle | 0.4257 | 0.5178 | 0.4592 | Ranking headroom when category is already correct |
| Nearest-GT relabel, keep original score | 0.4555 | 0.6009 | 0.4778 | Category-assignment upper bound with current scores |
| Nearest-GT relabel + IoU score | 0.4515 | 0.5384 | 0.4935 | Diagnostic mixed oracle |

Oracle artifacts:

- `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_oracle_classaware_iou_score_cocoeval.csv`
- `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_oracle_relabel_keep_score_cocoeval.csv`
- `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_oracle_relabel_iou_score_cocoeval.csv`

This confirms the same mechanism on the fair split: the model has substantial
box/candidate potential, but detector-side category assignment and score
calibration still leave most AP50 on the table.

Stratified high-IoU category confusions:

- The top wrong-category flows include `n07695742 -> n01726692` (`7/9`,
  mean IoU `0.955`), `n07739125 -> n07749582` (`6/11`, mean IoU `0.893`),
  and `n02799071 -> n02786058` (`5/6`, mean IoU `0.956`).
- These are localized GT-level confusions: for each GT, the diagnostic selects
  the highest-IoU prediction in the image top-100. It is not a full COCO AP
  error decomposition, but it is useful for inspecting class assignment after
  successful localization.
- Overlay artifacts:
  - `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_category_confusions_top100_iou50.csv`
  - `results/rfdetr_stratified_seed41_2best_category_confusion_overlays/contact_sheet.jpg`
  - `results/rfdetr_stratified_seed41_2best_category_confusion_overlays/manifest.csv`

Validation-split score calibration check:

| Scoring | Train target | AP | AP50 | AP75 | Note |
|---|---|---:|---:|---:|---|
| Base regular checkpoint | none | 0.1543 | 0.1869 | 0.1645 | Original detector scores |
| Valid-calibrated class-aware quality | same-category IoU | 0.1550 | 0.1864 | 0.1650 | Essentially flat |
| Valid-calibrated localization quality | nearest IoU | 0.1509 | 0.1794 | 0.1606 | Hurts class-aware AP |

The calibrators learn a signal on the validation split (`Spearman=0.3051` for
class-aware quality and `0.6153` for localization quality), but they do not
translate into test AP gains. This makes simple post-hoc score calibration a
negative result for this fair split. The remaining bottleneck is more likely
detector-side category assignment/representation than a scalar score remap.

Small deformable-attention probe:

- A targeted off-center probe was run on 8 test images with top-3 query overlays
  per image. The query-level mean IoU is `0.5638`; `62.5%` of inspected queries
  have IoU >= 0.5.
- Query attention is already strongly concentrated on GT regions: mean GT
  attention mass is `0.8379`, median is `0.9375`, and no inspected query has
  zero GT mass.
- Category assignment is still weak: nearest-GT category match is `0.5417`
  overall and `0.6667` among IoU>=0.5 queries. There are aligned-hit examples
  with high IoU/high GT attention mass but wrong category.
- This supports the current direction: the main remaining problem is not merely
  off-center attention coverage, but semantic/category assignment from the
  query representation.
- Artifacts:
  - `results/rfdetr_stratified_seed41_2best_attention_offcenter_probe/contact_sheet.jpg`
  - `results/rfdetr_stratified_seed41_2best_attention_offcenter_probe/query_contact_sheet.jpg`
  - `results/rfdetr_stratified_seed41_2best_attention_offcenter_probe_diagnostics.json`
  - `results/rfdetr_stratified_seed41_2best_attention_offcenter_probe_queries.csv`

Low-interference class-head-only continuation:

| Model | Trainable scope | AP | AP50 | AP75 | Loc AP | Loc AP50 | Loc AP75 |
|---|---|---:|---:|---:|---:|---:|---:|
| Base regular checkpoint | all, previous 4ep continuation | 0.1543 | 0.1869 | 0.1645 | 0.4206 | 0.5584 | 0.4484 |
| Class-head-only +1ep | classification heads only | 0.1609 | 0.1942 | 0.1704 | 0.4253 | 0.5661 | 0.4499 |
| Query+class +1ep | query/refpoint embeddings + class heads | 0.1603 | 0.1938 | 0.1707 | 0.4269 | 0.5677 | 0.4547 |
| Decoder+class +1ep | decoder representation + query/refpoint + class heads | 0.1615 | 0.1968 | 0.1734 | 0.4222 | 0.5597 | 0.4481 |

- `scripts/train_rfdetr_coco.py` now supports `--trainable-scope class-head`,
  `--trainable-scope query-class-head`, and
  `--trainable-scope decoder-class-head`.
  RF-DETR rebuilds its LightningModule inside `.train(...)`, so the script
  patches the internal module construction and freezes after checkpoint loading.
  The verified class-head run had `723K` trainable parameters and `31.6M`
  frozen; the query+class run had `1.7M` trainable and `30.6M` frozen; the
  decoder+class run had `6.2M` trainable and `26.2M` frozen.
- The gain is real but small: class AP improves by `+0.0066` and AP50 by
  `+0.0073`. It also slightly improves class-agnostic localization eval, likely
  through score/ranking effects rather than new box geometry.
- Releasing query feature and reference-point embeddings does not improve class
  AP over class-head-only. It slightly improves class-agnostic localization
  metrics, but coverage and class-aware score-IoU alignment remain basically
  unchanged.
- Releasing decoder representation layers gives the highest class AP/AP50 among
  the three 1-epoch low-interference continuations, but the gain over class-head
  only is tiny (`+0.0006` AP, `+0.0026` AP50), while class-aware score-IoU
  Spearman drops to `0.0804` and localization/coverage diagnostics do not
  improve. This is not a strong enough signal to keep expanding trainable scope.
- The remaining bottleneck is not solved: class-aware score-IoU Spearman only
  moves from `0.0862` to `0.0946`, and top-100 category coverage is nearly flat
  (`loc=0.8510`, global class-aware `0.6440`, per-category `0.7103`).
- Interpretation: low-interference class-head finetuning is a useful detector-
  side control, but final class-head tuning, learned query/refpoint tuning, and
  decoder representation tuning all leave the category-assignment/oracle gap
  mostly open. Future work should target stronger category supervision, teacher
  signals, or data/split scale, not only low-interference parameter-scope
  expansion.
- Artifacts:
  - `results/rfdetr_stratified_seed41_low_interference_scope_summary.csv`
  - `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_classhead_fixed_1ep_test_cocoeval.csv`
  - `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_classhead_fixed_1ep_test_loc_cocoeval.csv`
  - `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_classhead_fixed_1ep_test_slices.csv`
  - `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_classhead_fixed_1ep_score_iou_loc.csv`
  - `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_classhead_fixed_1ep_score_iou_classaware.csv`
  - `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_classhead_fixed_1ep_category_coverage_gap.csv`
  - `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_classhead_fixed_1ep_per_category_coverage_gap.csv`
  - `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_queryclass_fixed_1ep_test_cocoeval.csv`
  - `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_queryclass_fixed_1ep_test_loc_cocoeval.csv`
  - `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_queryclass_fixed_1ep_test_slices.csv`
  - `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_queryclass_fixed_1ep_score_iou_loc.csv`
  - `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_queryclass_fixed_1ep_score_iou_classaware.csv`
  - `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_queryclass_fixed_1ep_category_coverage_gap.csv`
  - `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_queryclass_fixed_1ep_per_category_coverage_gap.csv`
  - `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_decoderclass_fixed_1ep_test_cocoeval.csv`
  - `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_decoderclass_fixed_1ep_test_loc_cocoeval.csv`
  - `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_decoderclass_fixed_1ep_test_slices.csv`
  - `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_decoderclass_fixed_1ep_score_iou_loc.csv`
  - `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_decoderclass_fixed_1ep_score_iou_classaware.csv`
  - `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_decoderclass_fixed_1ep_category_coverage_gap.csv`
  - `results/rfdetr_stratified_seed41_train1000_small_384_seed41_2best_decoderclass_fixed_1ep_per_category_coverage_gap.csv`

Category candidate-set diagnostic:

| Model | Candidate hit rate | Mean nearest IoU | Candidate-oracle AP | Candidate-oracle AP50 | Candidate-oracle AP75 |
|---|---:|---:|---:|---:|---:|
| Base regular checkpoint | 0.3686 | 0.3218 | 0.3869 | 0.4996 | 0.4126 |
| Decoder+class +1ep | 0.3526 | 0.3062 | 0.3821 | 0.4974 | 0.4172 |

- RF-DETR postprocess takes the top predictions from the flattened
  `query x class` score table, so identical boxes can appear with multiple
  category candidates. `scripts/oracle_coco_category_candidates.py` groups
  predictions by image and rounded bbox, then checks whether the nearest GT
  category is present in that box group's candidate set.
- The base candidate hit rate is only `36.9%`, and decoder+class decreases it
  to `35.3%`. This is far below the nearest-GT relabel oracle, which reaches
  AP50 `0.6009` by assigning each localized box to its nearest GT category.
- Interpretation: the category bottleneck is not just ranking the right class
  lower within an otherwise good candidate set. Many localized box groups do
  not include the correct category among their emitted candidates at all. This
  points toward stronger detector-side category supervision or teacher signals,
  not further low-interference score calibration or decoder-scope expansion.
- Per-category breakdown sharpens the diagnosis. Among categories with at
  least `20` grouped boxes, the base checkpoint has `10` zero-hit categories,
  including localized-but-wrong classes such as `n03676483` (`47` groups,
  hit rate `0.000`, mean nearest IoU `0.560`) and `n07695742` (`40` groups,
  hit rate `0.000`, mean nearest IoU `0.546`). Decoder+class still has
  `8` zero-hit categories in the same high-support regime, and its weighted
  hit rate for `groups>=20` drops from `0.378` to `0.362`. Its improvements
  are concentrated in a few categories (`n03495258`, `n02131653`,
  `n02374451`, `n04228054`, `n04379243`), while several already-good classes
  degrade. This rules out a broad class-candidate repair from local decoder
  unfreezing.
- Joining split counts confirms this is not only residual data starvation after
  stratification. For base zero-hit categories with `groups>=20`, train box
  counts range from `6` to `20` with median `13.5`; for decoder+class they range
  from `6` to `20` with median `12.0`. The clearest failures,
  `n03676483` and `n07695742`, each have `20` train boxes and high nearest-box
  IoU, but still no emitted correct category candidate. Training count and hit
  rate are only weakly correlated in the high-support set (`r≈0.27` base,
  `r≈0.25` decoder+class).
- A qualitative high-IoU confusion sheet confirms the same pattern visually.
  The selected predictions have high overlap with GT boxes but wrong categories,
  often within fine-grained or semantically nearby groups (`n07695742 ->
  n01726692`, `n07739125 -> n07749582`, `n02799071 -> n02786058`,
  `n02484322 -> n02131653`, `n03676483 -> n07880968`). This is not a simple
  box-alignment bug; the detector is often looking at the right object support
  while emitting the wrong semantic candidate.
- Artifacts:
  - `results/rfdetr_stratified_seed41_2best_candidate_oracle_summary.csv`
  - `results/rfdetr_stratified_seed41_2best_candidate_oracle_groupmax_cocoeval.csv`
  - `results/rfdetr_stratified_seed41_2best_candidate_oracle_per_category.csv`
  - `results/rfdetr_stratified_seed41_2best_candidate_oracle_per_category_with_split_counts.csv`
  - `results/rfdetr_stratified_seed41_2best_decoderclass_candidate_oracle_summary.csv`
  - `results/rfdetr_stratified_seed41_2best_decoderclass_candidate_oracle_groupmax_cocoeval.csv`
  - `results/rfdetr_stratified_seed41_2best_decoderclass_candidate_oracle_per_category.csv`
  - `results/rfdetr_stratified_seed41_2best_decoderclass_candidate_oracle_per_category_with_split_counts.csv`
  - `results/rfdetr_stratified_seed41_2best_confusion_overlays/manifest.csv`
  - `results/rfdetr_stratified_seed41_2best_confusion_overlays/contact_sheet.jpg`

## Stop / Continue Rule

Do not continue max4/max5 crop expansion unless category scoring improves first.

Next useful directions:

1. Continue controlled 1000-image detector-side training only as a low-frequency
   baseline extension, with regular-checkpoint export and explicit class-aware /
   class-agnostic reporting.
2. Stronger detector-aware category teacher or teacher distillation into the
   detector-side class head remains a diagnostic branch, not the active main
   route, unless direct training saturates.
3. Category-aware quality calibration on an image-disjoint split after the
   detector-side class head is strong enough to provide candidate coverage.
4. If trying DET-aware category again, prefer detector-integrated training or
   distillation rather than post-hoc crop-prior relabeling.

Hard-category oversampling hook:

- `scripts/build_coco_hard_category_oversample.py` builds an RF-DETR-compatible
  dataset by duplicating train image records that contain candidate-missing hard
  categories, while preserving the original valid/test splits. It keeps image
  files linked by default through absolute symlinks and rewrites only the train
  COCO annotations.
- The first smoke uses base candidate misses with `groups>=20` and
  `candidate_hit_rate=0`, targets `60` train boxes per hard category, and caps
  per-image repeat at `5`. The generated dataset
  `/private/tmp/rfdetr_stratified_seed41_train1000_hardcat_oversample60_abs` passes
  `scripts/train_rfdetr_coco.py --check-only --num-classes 200`. Train images /
  annotations change from `1000/4409` to `1087/5221`.
- A 1ep class-head continuation from the stratified 2best checkpoint gives only
  a weak AP gain: class `AP/AP50/AP75 = 0.1624/0.1961/0.1729`. It is slightly
  above the base checkpoint and comparable to decoder+class +1ep, but it does
  not fix candidate generation. Grouped-box candidate hit rate drops to
  `34.1%`, candidate-oracle AP50 drops to `0.4875`, and the original `10`
  zero-hit hard categories recover only `3.98%` weighted hit rate. Treat this
  as a weak regularization result, not a route worth scaling blindly.
- The intervention summary table confirms the split: hard-category oversampling
  is the best/near-best AP row among the low-cost class interventions, but it
  has the worst candidate-hit diagnostics. Small AP gains alone should not be
  interpreted as candidate coverage repair.
- Artifact:
  - `results/rfdetr_stratified_seed41_category_intervention_summary.csv`
  - `results/rfdetr_stratified_seed41_hardcat_oversample60_abs_summary.csv`
  - `results/rfdetr_stratified_seed41_hardcat60_classhead_1ep_test_cocoeval.csv`
  - `results/rfdetr_stratified_seed41_hardcat60_classhead_1ep_test_loc_cocoeval.csv`
  - `results/rfdetr_stratified_seed41_hardcat60_classhead_1ep_test_slices.csv`
  - `results/rfdetr_stratified_seed41_hardcat60_classhead_1ep_candidate_oracle_summary.csv`
  - `results/rfdetr_stratified_seed41_hardcat60_classhead_1ep_candidate_oracle_groupmax_cocoeval.csv`
  - `results/rfdetr_stratified_seed41_hardcat60_classhead_1ep_candidate_oracle_per_category.csv`
  - `results/rfdetr_stratified_seed41_hardcat60_classhead_1ep_candidate_oracle_per_category_with_split_counts.csv`

Full low-LR detector continuation:

- A 2-epoch all-parameter continuation from the same stratified regular
  checkpoint, using lr `5e-5`, is the first intervention in this series that
  clearly improves both AP and category candidate generation. Independent test
  metrics from `checkpoint_best_regular.pth` are class AP/AP50/AP75
  `0.1930/0.2558/0.2154` and class-agnostic loc AP/AP50/AP75
  `0.4277/0.5754/0.4622`.
- The slice gains are broad rather than a single-slice artifact:
  `offcenter AP50=0.2252`, `center=0.2582`, `small=0.1772`,
  `medium=0.2634`, and `large=0.3483`.
- Candidate-set diagnostics also move in the right direction. The grouped-box
  candidate hit rate rises to `40.6%`, mean nearest IoU to `0.3330`, and
  candidate-oracle AP/AP50/AP75 to `0.4290/0.5627/0.4716`. This is the opposite
  of the hard-category oversampling result, which slightly improved AP but
  worsened candidate coverage.
- The high-support category summary confirms the improvement is not purely a
  few low-support classes. For categories with at least `20` grouped boxes,
  weighted hit rate improves from `37.8%` in the base checkpoint to `41.1%` in
  the full low-LR continuation; zero-hit high-support categories fall from
  `10` to `8`.
- Standard coverage diagnostics show the same semantic-candidate shift. Top-100
  class-aware global recall improves from `0.6424` in the base checkpoint to
  `0.6887`, and per-category class-aware recall improves from `0.7119` to
  `0.7467`; class-agnostic loc recall is roughly flat/slightly lower
  (`0.8427 -> 0.8311`). Score-IoU alignment is still imperfect: loc Spearman is
  `-0.1376`, while class-aware Spearman is `0.1082`. The gain is candidate
  generation/semantic coverage more than clean scalar calibration.
- Current decision: the active RF-DETR route should be formal full-detector
  adaptation, not further low-interference class-head repair, hard-category
  oversampling, scalar calibration, or persistent proposal state. The next
  serious check is another seed and/or a longer regular-checkpoint continuation,
  with class-aware AP, class-agnostic loc AP, slice AP, candidate hit rate, and
  candidate-oracle AP reported together.
- A same-seed 2-epoch continuation from this checkpoint at lr `3e-5` improves
  deployed class AP further, but it does not keep repairing category candidates.
  Independent test class AP/AP50/AP75 becomes `0.2016/0.2625/0.2252`, with
  strong slice AP50 on offcenter (`0.2455`), small (`0.1952`), medium
  (`0.3222`), and large (`0.3566`) objects. However class-agnostic loc AP/AP50
  drops to `0.4131/0.5640`, grouped candidate hit rate drops to `39.8%`,
  high-support weighted hit drops to `39.9%`, and high-support zero-hit
  categories increase to `13`. Candidate-oracle AP/AP50/AP75 is
  `0.4242/0.5668/0.4557`. This is useful for deployed AP, but it suggests the
  candidate-coverage repair has likely plateaued on this seed.
- Updated decision: do not keep extending seed41 blindly. The next formal
  experiment should either repeat the full low-LR route on a second seed or run
  a longer planned RF-DETR protocol with candidate-hit diagnostics as a stage
  gate. If the second seed repeats the pattern, full detector-side adaptation
  becomes the current main route; if not, the 2ep gain should be treated as a
  seed-specific continuation result.
- Forced-head second-seed 2ep sanity on
  `rfdetr_stratified_seed43_train1000_val200_test200_min3` confirms the RF-DETR
  head-size fix is active in the export path (`model_num_classes=200`). Regular
  checkpoint independent test metrics are class AP/AP50/AP75
  `0.1474/0.1888/0.1608`, loc AP/AP50/AP75 `0.3635/0.5068/0.3868`, candidate
  hit rate `37.8%`, and candidate-oracle AP/AP50/AP75 `0.3729/0.4849/0.4079`.
  This is a repaired-head baseline sanity row, not a new long-train result.
- `scripts/summarize_rfdetr_mainline.py` now produces
  `results/rfdetr_stratified_mainline_summary.csv`, the compact comparison
  table for class AP, loc AP, slice AP50, candidate hit/oracle AP, top-k
  coverage, score-IoU correlation, and high-support candidate zero-hit
  diagnostics across stratified RF-DETR runs.
- `scripts/summarize_candidate_category_transitions.py` now produces
  `results/rfdetr_stratified_seed41_candidate_category_transitions.csv` for
  the seed41 base -> full_lr_2ep -> extra_low_lr chain. It identifies `7`
  persistent high-support zero-hit categories, `4` categories that regress to
  zero-hit, and only `2` categories rescued from zero-hit, so the extra-low-LR
  continuation should be treated as class-AP-positive but candidate-coverage
  mixed rather than a monotonic candidate repair.
- `scripts/summarize_candidate_failure_categories.py` now joins those
  transitions with split counts and base high-IoU confusion flows in
  `results/rfdetr_stratified_seed41_candidate_failure_categories.csv`. The
  clearest qualitative targets are `n03676483`, `n07695742`, and `n04468005`:
  they have nontrivial train/test support, high nearest-box IoU, and stable
  wrong-category flows, so they are better debugging targets than aggregate AP
  alone.
- `scripts/select_coco_category_samples.py` now exports
  `results/rfdetr_stratified_seed41_candidate_failure_test_samples.csv`, a
  fixed 52-annotation test-set target list for those persistent/regressed
  zero-hit categories. Use this for future prediction overlays or manual
  failure inspection.
- `scripts/visualize_coco_category_samples.py` now exports a GT-only contact
  sheet and manifest under
  `results/rfdetr_stratified_seed41_candidate_failure_gt_overlays/`. This
  visualizes the fixed failure target set before any new prediction export, and
  shows that the candidate failures include both clear large objects and
  difficult small/multi-instance cases.
- `scripts/visualize_coco_sample_predictions.py` now overlays GT boxes with the
  nearest grouped RF-DETR prediction for that same fixed failure set. Artifact:
  `results/rfdetr_stratified_seed41_candidate_failure_prediction_overlays/`.
  The current contact sheet samples 30 annotations from persistent/regressed
  zero-hit categories; mean nearest IoU is `0.784`, `24/30` samples have nearest
  IoU >= `0.5`, but only `2/30` nearest prediction groups contain the GT
  category anywhere in the grouped candidate set, and none contain it within
  the displayed top-5. The manifest now records nearest group rank, top score,
  group size, GT-category rank, and input SHA256s in `run_manifest.json`. This
  is a top-k nearest-box diagnostic: it shows local box support exists for many
  fixed failures, while emitted semantic candidates miss the correct category.
- `scripts/summarize_coco_sample_prediction_chain.py` compares those fixed
  failure samples across checkpoint prediction JSONs. Artifact:
  `results/rfdetr_stratified_seed41_candidate_failure_prediction_chain_summary.csv`.
  On all 52 target annotations, nearest-box support remains high across the
  seed41 chain (`base/full_lr_2ep/extra_low_lr` mean nearest IoU
  `0.811/0.810/0.822`), but GT-category candidate presence falls
  `4/52 -> 3/52 -> 0/52`; displayed-top-5 presence is `0` for all three.
  Mean nearest-group rank drifts from `10.2 -> 12.8 -> 13.8`, so these are not
  necessarily top-scored detections, but the nearest-box diagnostic still
  confirms that extra-low-LR continuation can improve deployed class AP while
  regressing concrete fixed failure categories' emitted candidate coverage.
- `scripts/summarize_coco_sample_candidate_flows.py` aggregates the nearest-box
  top-candidate flows from the same chain into
  `results/rfdetr_stratified_seed41_candidate_failure_candidate_flows.csv` and
  `..._candidate_flow_by_category.csv`. Stable wrong-flow examples on the fixed
  failure set include `n03676483 -> n00007846` in the base checkpoint
  (`10` samples, mean nearest IoU `0.888`, no GT candidate), `n07695742 ->
  n00007846` in base (`6`, IoU `0.969`), and after continuation
  `n07695742 -> n01726692` plus `n04468005 -> n04530566`. These flows are now
  the concrete semantic-confusion targets for any future class-head or teacher
  repair.
- `scripts/build_rfdetr_candidate_repair_targets.py` merges the failure-category
  report with the fixed-sample flow table into
  `results/rfdetr_stratified_seed41_candidate_repair_targets.csv`. This is the
  current compact handoff table for repair work: it keeps train/valid/test
  support, aggregate zero-hit stats, high-IoU wrong-flow evidence, and
  checkpoint-specific dominant wrong candidates in one row per target category.
- `scripts/build_rfdetr_hard_confusion_pairs.py` extracts high-IoU fixed-sample
  instances where the GT category is absent from the nearest bbox group's
  candidate set. Artifacts:
  `results/rfdetr_stratified_seed41_candidate_failure_hard_pairs.csv` and
  `..._hard_pair_summary.csv`. With `min_iou=0.5`, it produces `132`
  hard-confusion instances across the three checkpoint labels and `73` aggregate
  pairs. These rows are the candidate-level hard negatives for any future
  class-head contrastive loss or teacher-side semantic repair.
- `scripts/build_rfdetr_semantic_repair_config.py` converts the hard-pair
  summary into a training-consumable JSON config:
  `results/rfdetr_stratified_seed41_semantic_repair_config.json`. It maps
  category names back to COCO category ids, merges duplicate `(GT, negative)`
  pairs across checkpoint labels, caps per-pair weights, and keeps example
  image/annotation ids. With `min_samples=2`, it yields `6` target categories;
  the strongest repair targets are `n03255030`, `n03676483`, and `n07695742`.
- Semantic hard-negative loss substrate added for the local detector criterion.
  `DetectionCriterion` can now add a margin penalty that pushes configured
  hard-negative class logits below the matched GT class. Because COCO/RF-DETR
  category ids are 1-based in this handoff, the conversion artifact is explicit:
  `scripts/convert_semantic_repair_config_to_loss_map.py --category-id-offset 1`
  writes `results/rfdetr_stratified_seed41_semantic_repair_loss_map.json`.
  `scripts/train_det_real.py --semantic-hard-negative-loss-map` then remaps
  entries by category name onto the active runtime `label_to_id` when names are
  present, avoiding silent RF-DETR category-id vs compact-logit-index mismatch.
  This is a training-interface dry run for future class-head/semantic repair
  smoke tests, not evidence yet that the loss improves RF-DETR candidate
  coverage.
- `scripts/check_rfdetr_handoff.py` now writes explicit image-id split
  overlap diagnostics. The refreshed stratified seed41 handoff check reports
  `image_id_overlap_counts={train_valid: 0, train_test: 0, valid_test: 0}` and
  `image_ids_disjoint=true`, closing the previous audit caveat that the note
  only proved annotation-hash/category consistency.
- Artifacts:
  - `results/rfdetr_stratified_seed41_fulllr5e5_2ep_test_cocoeval.csv`
  - `results/rfdetr_stratified_seed41_fulllr5e5_2ep_test_loc_cocoeval.csv`
  - `results/rfdetr_stratified_seed41_fulllr5e5_2ep_test_slices.csv`
  - `results/rfdetr_stratified_seed41_fulllr5e5_2ep_candidate_oracle_summary.csv`
  - `results/rfdetr_stratified_seed41_fulllr5e5_2ep_candidate_oracle_groupmax_cocoeval.csv`
  - `results/rfdetr_stratified_seed41_fulllr5e5_2ep_candidate_oracle_per_category.csv`
  - `results/rfdetr_stratified_seed41_fulllr5e5_2ep_candidate_oracle_per_category_with_split_counts.csv`
  - `results/rfdetr_stratified_seed41_candidate_high_support_summary.csv`
  - `results/rfdetr_stratified_seed41_candidate_category_transitions.csv`
  - `results/rfdetr_stratified_seed41_candidate_failure_categories.csv`
  - `results/rfdetr_stratified_seed41_candidate_failure_test_samples.csv`
  - `results/rfdetr_stratified_seed41_candidate_failure_gt_overlays/contact_sheet.jpg`
  - `results/rfdetr_stratified_seed41_candidate_failure_gt_overlays/manifest.csv`
  - `results/rfdetr_stratified_seed41_candidate_failure_prediction_overlays/contact_sheet.jpg`
  - `results/rfdetr_stratified_seed41_candidate_failure_prediction_overlays/manifest.csv`
  - `results/rfdetr_stratified_seed41_candidate_failure_prediction_overlays/run_manifest.json`
  - `results/rfdetr_stratified_seed41_candidate_failure_prediction_chain.csv`
  - `results/rfdetr_stratified_seed41_candidate_failure_prediction_chain_summary.csv`
  - `results/rfdetr_stratified_seed41_candidate_failure_prediction_chain_summary_run_manifest.json`
  - `results/rfdetr_stratified_seed41_candidate_failure_candidate_flows.csv`
  - `results/rfdetr_stratified_seed41_candidate_failure_candidate_flow_by_category.csv`
  - `results/rfdetr_stratified_seed41_candidate_repair_targets.csv`
  - `results/rfdetr_stratified_seed41_candidate_failure_hard_pairs.csv`
  - `results/rfdetr_stratified_seed41_candidate_failure_hard_pair_summary.csv`
  - `results/rfdetr_stratified_seed41_semantic_repair_config.json`
  - `results/rfdetr_stratified_seed41_semantic_repair_loss_map.json`
  - `results/rfdetr_stratified_seed41_fulllr5e5_2ep_score_iou_loc.csv`
  - `results/rfdetr_stratified_seed41_fulllr5e5_2ep_score_iou_classaware.csv`
  - `results/rfdetr_stratified_seed41_fulllr5e5_2ep_category_coverage_gap.csv`
  - `results/rfdetr_stratified_seed41_fulllr5e5_2ep_per_category_coverage_gap_with_split_counts.csv`
  - `results/rfdetr_stratified_seed41_fulllr5e5_then3e5_2ep_test_cocoeval.csv`
  - `results/rfdetr_stratified_seed41_fulllr5e5_then3e5_2ep_test_loc_cocoeval.csv`
  - `results/rfdetr_stratified_seed41_fulllr5e5_then3e5_2ep_test_slices.csv`
  - `results/rfdetr_stratified_seed41_fulllr5e5_then3e5_2ep_candidate_oracle_summary.csv`
  - `results/rfdetr_stratified_seed41_fulllr5e5_then3e5_2ep_candidate_oracle_groupmax_cocoeval.csv`
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

Avoid:

- Returning to persistent proposal state.
- Sweeping alpha/temperature on the same eval split.
- Claiming crop finetuning solves small-object detection.
