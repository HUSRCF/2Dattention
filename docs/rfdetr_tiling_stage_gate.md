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

Frozen DET-aware category-prior check:

| Setting | Eval top-1 | Eval top-5 | AP50 | Top-100 class-aware R@50 | Top-100 mean class-aware IoU |
|---|---:|---:|---:|---:|---:|
| frozen ResNet50 DET linear head, top-5 multiply | 0.1143 | 0.4286 | 0.0223 | 0.3797 | 0.1173 |
| frozen ResNet50 DET linear head, top-5 keep-score | 0.1143 | 0.4286 | 0.0139 | 0.3797 | 0.0034 |

Frozen DET-prior artifacts:

- `results/rfdetr_max3_frozen_resnet50_detprior_top5x_heldout_summary.csv`
- `results/rfdetr_max3_frozen_resnet50_detprior_top5x_heldout_slices.csv`
- `results/rfdetr_max3_frozen_resnet50_detprior_top5x_heldout_prediction_recall.csv`
- `results/rfdetr_max3_frozen_resnet50_detprior_top5x_heldout_score_iou.csv`
- `results/rfdetr_max3_frozen_resnet50_detprior_top5keep_heldout_summary.csv`
- `results/rfdetr_max3_frozen_resnet50_detprior_top5keep_heldout_slices.csv`
- `results/rfdetr_max3_frozen_resnet50_detprior_top5keep_heldout_prediction_recall.csv`
- `results/rfdetr_max3_frozen_resnet50_detprior_top5keep_heldout_score_iou.csv`
- `results/rfdetr_frozen_detprior_heldout_summary.csv`

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
  `0.0139` with kept detector scores. The local split is too small/noisy for a
  standalone DET category head to replace the pretrained ImageNet prior.

## Stop / Continue Rule

Do not continue max4/max5 crop expansion unless category scoring improves first.

Next useful directions:

1. Stronger or detector-aware category teacher.
2. Stronger category-aware quality calibration on an image-disjoint split.
3. Score-IoU ranking for fused boxes after category transfer.
4. If trying DET-aware category again, prefer teacher/distillation or
   proposal-conditioned classification rather than a local image-level linear
   head.

Avoid:

- Returning to persistent proposal state.
- Sweeping alpha/temperature on the same eval split.
- Claiming crop finetuning solves small-object detection.
