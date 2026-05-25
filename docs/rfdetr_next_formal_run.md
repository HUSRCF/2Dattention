# RF-DETR Next Formal Run Plan

This note freezes the current practical next step after the stratified
multi-metric route ranking. The planned run has now completed; this file keeps
the exact command plus the post-run interpretation for reproducibility.

## Current Selected Checkpoint

- Benchmark family: category-stratified RF-DETR train1000 / valid200 / test200.
- Dataset:
  `data/ILSVRC2013_DET_val_supervised/rfdetr_stratified_seed41_train1000_val200_test200_min3`
- Current route guardrail:
  `results/rfdetr_stratified_active_route_multi_metric_rank.csv`
- Selected row:
  `seed41_fulllr5e5_then3e5_2ep`
- Current checkpoint:
  `/private/tmp/rfdetr_stratified_seed41_train1000_small_384_seed41_fulllr5e5_then3e5_2ep/checkpoint_best_regular.pth`

This row is the best stratified-only mean-rank point over `class_ap50`,
`offcenter_ap50`, `small_ap50`, and `loc_ap50`. It improves class/offcenter/small
AP50, but candidate coverage remains mixed, so the next run should be treated as
a formal continuation check rather than a mechanism breakthrough.

## Completed Preflight

The selected checkpoint was probed with a one-image export on MPS:

```bash
env MPLCONFIGDIR=/private/tmp/matplotlib-cache NO_ALBUMENTATIONS_UPDATE=1 \
  /opt/anaconda3/envs/AIAA/bin/python scripts/predict_rfdetr_coco.py \
  --annotation-json data/ILSVRC2013_DET_val_supervised/rfdetr_stratified_seed41_train1000_val200_test200_min3/test/_annotations.coco.json \
  --image-root data/ILSVRC2013_DET_val_supervised/rfdetr_stratified_seed41_train1000_val200_test200_min3/test \
  --weights /private/tmp/rfdetr_stratified_seed41_train1000_small_384_seed41_fulllr5e5_then3e5_2ep/checkpoint_best_regular.pth \
  --model-size small \
  --num-classes 200 \
  --device mps \
  --resolution 384 \
  --max-images 1 \
  --out results/rfdetr_stratified_seed41_extra_low_lr_head_probe_predictions.json
```

The export completed with:

- `images: 1`
- `predictions: 300`
- `model_num_classes: 200`
- `actual_class_head_out_features: 201`
- `forced_class_head_out_features: 201 -> 201`

## Next Train Command

Use a fresh optimizer continuation from the selected regular checkpoint. This
keeps the protocol consistent with the prior continuation blocks and avoids RF-DETR
resume-state artifacts.

```bash
env MPLCONFIGDIR=/private/tmp/matplotlib-cache NO_ALBUMENTATIONS_UPDATE=1 \
  /opt/anaconda3/envs/AIAA/bin/python scripts/train_rfdetr_coco.py \
  --dataset-dir data/ILSVRC2013_DET_val_supervised/rfdetr_stratified_seed41_train1000_val200_test200_min3 \
  --output-dir /private/tmp/rfdetr_stratified_seed41_train1000_small_384_seed41_extra_low_lr_plus2ep_lr2e5 \
  --model-size small \
  --num-classes 200 \
  --device mps \
  --resolution 384 \
  --epochs 2 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --lr 2e-5 \
  --num-workers 2 \
  --eval-interval 1 \
  --checkpoint-interval 1 \
  --pretrain-weights /private/tmp/rfdetr_stratified_seed41_train1000_small_384_seed41_fulllr5e5_then3e5_2ep/checkpoint_best_regular.pth \
  --seed 41 \
  --no-tensorboard \
  --no-wandb
```

## Completed Outcome

The run completed on MPS and saved:

- Run directory:
  `/private/tmp/rfdetr_stratified_seed41_train1000_small_384_seed41_extra_low_lr_plus2ep_lr2e5`
- Official checkpoint for independent conclusions:
  `checkpoint_best_regular.pth`
- Prediction export:
  `regular_test_predictions.json`

Independent test metrics from the standard artifact bundle:

- Class AP/AP50/AP75: `0.2100 / 0.2700 / 0.2288`
- Loc AP/AP50/AP75: `0.4280 / 0.5837 / 0.4561`
- Slice AP50: `offcenter=0.2387`, `center=0.2842`, `small=0.2057`,
  `medium=0.2940`, `large=0.3731`
- Candidate hit rate: `0.3828`
- Candidate-oracle AP/AP50/AP75: `0.4070 / 0.5468 / 0.4279`
- Score-IoU Spearman: `class=0.1155`, `loc=-0.0907`
- High-support candidate diagnostics: `61` high-support categories,
  `8` zero-hit categories, weighted hit `0.3855`

Interpretation: this is the best stratified active-route row by the current
multi-metric rank and improves deployed class/small/loc metrics, but candidate
hit rate regresses relative to earlier seed41 continuations. Treat it as a
detector-side AP continuation, not a candidate-generation breakthrough.

## Required Post-Run Checks

After training, report only independent `checkpoint_best_regular.pth` exports:

1. Class-aware COCO eval.
2. Class-agnostic localization COCO eval.
3. Slice AP50 for offcenter/center/small/medium/large.
4. Score-IoU correlation.
5. Candidate coverage / candidate-oracle diagnostics.
6. Updated stratified mainline summary and stratified active-route rank.

Use this concrete post-run template, replacing only `RUN_DIR` / `RUN_NAME` if
the train output directory changes. The preferred path is: export predictions
once, then run the artifact-bundle evaluator.

```bash
RUN_DIR=/private/tmp/rfdetr_stratified_seed41_train1000_small_384_seed41_extra_low_lr_plus2ep_lr2e5
RUN_NAME=rfdetr_stratified_seed41_extra_low_lr_plus2ep_lr2e5
ANN=data/ILSVRC2013_DET_val_supervised/rfdetr_stratified_seed41_train1000_val200_test200_min3/test/_annotations.coco.json
IMG=data/ILSVRC2013_DET_val_supervised/rfdetr_stratified_seed41_train1000_val200_test200_min3/test

env MPLCONFIGDIR=/private/tmp/matplotlib-cache NO_ALBUMENTATIONS_UPDATE=1 \
  /opt/anaconda3/envs/AIAA/bin/python scripts/predict_rfdetr_coco.py \
  --annotation-json "$ANN" \
  --image-root "$IMG" \
  --weights "$RUN_DIR/checkpoint_best_regular.pth" \
  --model-size small \
  --num-classes 200 \
  --device mps \
  --resolution 384 \
  --out "$RUN_DIR/regular_test_predictions.json"

/opt/anaconda3/envs/AIAA/bin/python scripts/evaluate_rfdetr_run_artifacts.py \
  --annotations "$ANN" \
  --predictions "$RUN_DIR/regular_test_predictions.json" \
  --out-prefix "results/${RUN_NAME}" \
  --candidate-score-mode group_max
```

Then update `results/rfdetr_stratified_mainline_summary.csv` with
`scripts/update_rfdetr_mainline_summary.py`, which also regenerates
`results/rfdetr_stratified_active_route_multi_metric_rank.csv`:

```bash
/opt/anaconda3/envs/AIAA/bin/python scripts/update_rfdetr_mainline_summary.py \
  --base-summary results/rfdetr_stratified_mainline_summary.csv \
  --artifact-prefix "results/${RUN_NAME}" \
  --setting seed41_extra_low_lr_plus2ep_lr2e5 \
  --protocol stratified_seed41_extra_low_lr_plus2ep_lr2e5 \
  --notes next_formal_continuation \
  --out results/rfdetr_stratified_mainline_summary.csv \
  --rank-out results/rfdetr_stratified_active_route_multi_metric_rank.csv
```

Then record the result in `docs/TODO.md`.

Go/no-go:

- Continue this route only if class AP50 or hard-slice AP50 improves without a
  candidate-coverage collapse.
- If class AP improves but candidate hit rate / high-support zero-hit categories
  regress further, treat the run as training-gain-only and return to category
  candidate generation rather than more LR continuation.
