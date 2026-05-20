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

## Interpretation

- More crop coverage still improves class-agnostic localization.
- Class-aware AP saturates after max2/max3 because category prediction and
  score-IoU ranking dominate the remaining error.
- ResNet50 top-5 category expansion is the strongest checked non-oracle
  category-transfer setting.
- Naive lightweight teachers and simple teacher ensembles do not beat ResNet50.
- ResNet50 any-GT top-5 coverage is only `0.3913`, so ImageNet category
  coverage itself is a bottleneck, not just top-k ranking.

## Stop / Continue Rule

Do not continue max4/max5 crop expansion unless category scoring improves first.

Next useful directions:

1. Stronger or detector-aware category teacher.
2. Category calibration on an image-disjoint calibration split.
3. Score-IoU ranking for fused boxes after category transfer.

Avoid:

- Returning to persistent proposal state.
- Sweeping alpha/temperature on the same eval split.
- Claiming crop finetuning solves small-object detection.
