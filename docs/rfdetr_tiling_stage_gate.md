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
  (`0.1437`) on class-aware AP50. This suggests that simply switching to the
  larger Small backbone is not enough under the current short schedule; the
  bottleneck remains category supervision / detector-side class learning.
- The current 5-epoch Nano result still trails the max3 crop + ResNet50 top-5
  category-prior AP50 `0.1697`, but it is the correct integrated detector-side
  route. The next useful step is stronger teacher / detector-side distillation
  or a longer Small/Nano protocol, not another post-hoc crop prior.
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

Distillation artifacts:

- `results/rfdetr_detector_side_pseudolabel_summary.csv`
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

Distillation interpretation:

- The train-split teacher-prediction path is now functional. Nano 5ep produced
  `98,506` train predictions; filtered pseudo labels gave `2,191` train boxes at
  `score>=0.25/top20`.
- Teacher-only training is slightly better than the 1-epoch GT baseline on
  class AP50 (`0.0552` vs `0.0495`), so detector-side pseudo-label training can
  move the class head.
- Naively appending pseudo labels to GT is negative: GT+pseudo lowers class AP50
  to `0.0484` and class-agnostic AP50 to `0.3737`, suggesting duplicate/noisy
  pseudo boxes interfere with Hungarian matching.
- Teacher-only pretraining followed by GT finetuning is the first clear positive
  distillation schedule: class AP50 improves to `0.0816`, above GT 1 epoch
  (`0.0495`), teacher-only (`0.0552`), and raw GT+pseudo (`0.0484`).
- Lengthening the GT finetune to 2 epochs strengthens the staged schedule:
  class AP50 reaches `0.1088`, AP75 `0.0820`, class-agnostic AP50 `0.6273`,
  off-center AP50 `0.1163`, and large-object AP50 `0.2240`.
- Lengthening the GT finetune to 3 epochs pushes class AP50 to `0.1428`, nearly
  tying GT-only Nano 5ep class AP50 (`0.1437`) while exceeding GT-only Nano 5ep
  off-center AP50 (`0.1514` vs `0.1454`). This is the strongest detector-side
  pseudo-label result so far, though class-agnostic AP50 (`0.6198`) remains close
  to the 2ep value (`0.6273`), so the main gain is class/ranking-side rather than
  raw localization.
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
