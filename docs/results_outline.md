# Results Outline

This document is a paper-style Results skeleton for the current 2Dattention experiments. It separates what is supported by evidence from what has been ruled out or downgraded.

## 1. Main Finding

The current results support a dual-mainline interpretation:

- Classification mainline: `no_prefill_local_mix`.
- Dense spatial localization mainline candidate: `anchor_only_no_prefill`.

The supported mechanism is clean no-prefill online anchor/region interaction. The current experiments do not support early prefill, stale history pooling, full prefill-lattice memory, or memory-first routing as the main claim.

Suggested concise claim:

> Dense bbox-mask localization is the first spatial probe that produces a clear and stable architectural signal. At both 16x16 and 32x32 mask resolutions, `anchor_only_no_prefill` is the strongest model, with consistent paired IoU wins over `no_prefill_local_mix` and `fpn_sum_lite`. The advantage also holds across small, medium, and large object bins. This supports clean online anchor/region interaction for dense spatial prediction, but not early prefill or memory-first designs.

## 2. Classification Result

Task: DET-derived ImageFolder classification, 197 classes, 64px images, 1000 steps, 3 seeds.

Primary result:

| Model | Final acc | Best acc | Speed |
|---|---:|---:|---:|
| `anchor_only_no_prefill` | 0.251 | 0.251 | 338 img/s |
| `no_prefill_local_mix` | 0.250 | 0.250 | 462 img/s |
| `xattnres_no_prefill` | 0.249 | 0.249 | 377 img/s |
| `fpn_sum_lite` | 0.244 | 0.244 | 447 img/s |

Interpretation:

- `anchor_only_no_prefill` has a tiny accuracy edge, but the margin is about `+0.002` over `no_prefill_local_mix`.
- `no_prefill_local_mix` remains the practical Pareto baseline because it is almost equally accurate and substantially faster.
- FPN-like fixed fusion does not explain the classification result, because `fpn_sum_lite` and `fpn_concat_lite` underperform `no_prefill_local_mix`.
- Early history/prefill is harmful in this setup: `xattnres_no_prefill` improves over the older `xattnres_style`.

Results-section wording:

> In classification, the strongest practical reference is the no-prefill local-state model. Anchor-only interaction provides only a small accuracy-side signal, and its speed cost prevents it from replacing local-state refinement as the classification mainline.

## 3. Negative Spatial Probes

These probes are useful because they prevent overclaiming. They should be reported as negative findings, not hidden.

### 3.1 Balanced Quadrant4

Balanced random baseline: `0.250`.

Best result:

| Model | Balanced acc | Macro F1 |
|---|---:|---:|
| `fpn_sum_lite` | 0.275 | 0.221 |
| `no_prefill_local_mix` | 0.269 | 0.200 |
| `stage_refresh_region_slots_2x2` | 0.267 | 0.207 |

Interpretation:

- The probe gives a weak signal above random.
- The signal favors simple fixed fusion, not memory/region routing.
- It is too coarse to support a spatial reasoning claim.

### 3.2 Balanced Grid9

Balanced random baseline: `1/9 = 0.111`.

Best result:

| Model | Balanced acc | Macro F1 |
|---|---:|---:|
| `no_prefill_local_mix` | 0.136 | 0.093 |
| `xattnres_no_prefill` | 0.136 | 0.090 |
| `region_pool_mixer_no_history` | 0.129 | 0.085 |

Interpretation:

- The signal is very weak.
- Macro F1 remains below `0.10` for every model.
- This does not provide a reliable architecture discriminator.

### 3.3 Global-Pooled Center Regression

Eval split mean-target baseline L2: `0.1518`.

All models are essentially tied around `0.1524-0.1528`.

Interpretation:

- The global pooled output bottleneck collapses to the dataset center prior.
- This probe does not measure image-conditioned localization.

### 3.4 Gaussian Center Heatmap

Mean-target baseline L2: `0.1518`.

All models have argmax L2 around `0.1588-0.1592`.

Interpretation:

- The spatial head is a better design than global pooling, but largest-box center is too weak as supervision.
- Softargmax can look close to the prior because smooth heatmaps collapse toward the center prior.
- This probe does not support memory/region claims.

Results-section wording:

> Coarse cell classification, global-pooled regression, and Gaussian center heatmaps either collapse to spatial priors or produce weak/non-discriminative signals. These results motivated the shift from center-only targets to dense bbox-mask supervision.

## 4. Dense BBox-Mask Localization

The bbox-mask probe is the first spatial evaluation that produces a stable architecture signal.

Task: predict a dense rectangle mask for the largest annotated bbox. A cell is positive if its center lies inside the bbox. Training uses BCE plus Dice loss. Metrics include IoU, Dice, balanced cell accuracy, center-from-mask PCK, and area-stratified IoU.

Sanity check:

- `no_prefill_local_mix` overfits 32 samples in 200 steps.
- IoU: `0.739`.
- Dice: `0.838`.
- PCK@0.10: `0.840`.

This verifies that the target, head, and loss are learnable.

### 4.1 16x16 BBox Mask

Mean-mask prior IoU: `0.364`.
Center-box prior IoU: `0.407`.

| Model | IoU | Dice | Center L2 | PCK@0.10 | Speed |
|---|---:|---:|---:|---:|---:|
| `anchor_only_no_prefill` | 0.436 | 0.577 | 0.1494 | 0.391 | 245 img/s |
| `region_pool_mixer_no_history` | 0.426 | 0.569 | 0.1502 | 0.391 | 258 img/s |
| `stage_refresh_region_slots_2x2` | 0.425 | 0.568 | 0.1496 | 0.395 | 255 img/s |
| `fpn_sum_lite` | 0.422 | 0.565 | 0.1510 | 0.385 | 317 img/s |
| `no_prefill_local_mix` | 0.422 | 0.565 | 0.1511 | 0.382 | 313 img/s |
| `xattnres_no_prefill` | 0.421 | 0.563 | 0.1515 | 0.383 | 270 img/s |

Paired against `no_prefill_local_mix`:

- `anchor_only_no_prefill`: IoU delta `+0.014`, `3/3` wins.
- Dice delta `+0.012`, `3/3` wins.
- Center L2 delta `-0.0018`, `3/3` wins.

Paired against `fpn_sum_lite`:

- `anchor_only_no_prefill`: IoU delta `+0.014`, `3/3` wins.

### 4.2 32x32 BBox Mask

Mean-mask prior IoU: `0.363`.
Center-box prior IoU: `0.386`.

| Model | IoU | Small IoU | Medium IoU | Large IoU | Dice | Center L2 | Speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| `anchor_only_no_prefill` | 0.434 | 0.145 | 0.402 | 0.525 | 0.576 | 0.1492 | 239 img/s |
| `region_pool_mixer_no_history` | 0.430 | 0.135 | 0.397 | 0.523 | 0.573 | 0.1494 | 259 img/s |
| `stage_refresh_region_slots_2x2` | 0.425 | 0.139 | 0.398 | 0.512 | 0.568 | 0.1492 | 255 img/s |
| `fpn_sum_lite` | 0.424 | 0.131 | 0.392 | 0.516 | 0.567 | 0.1510 | 296 img/s |
| `no_prefill_local_mix` | 0.424 | 0.131 | 0.390 | 0.517 | 0.567 | 0.1512 | 307 img/s |
| `xattnres_no_prefill` | 0.422 | 0.131 | 0.390 | 0.513 | 0.565 | 0.1509 | 255 img/s |

Paired against `no_prefill_local_mix`:

- `anchor_only_no_prefill`: IoU delta `+0.010`, `3/3` wins.
- Dice delta `+0.009`, `3/3` wins.
- Center L2 delta `-0.0020`, `3/3` wins.

Paired against `fpn_sum_lite`:

- `anchor_only_no_prefill`: IoU delta `+0.010`, `3/3` wins.

Interpretation:

- The positive signal survives the higher `32x32` resolution.
- The advantage is not isolated to one object-size bin: small, medium, and large IoU means are all highest for `anchor_only_no_prefill`.
- `region_pool_mixer_no_history` is the closest follower, suggesting that region-style interaction matters, but the clean anchor-only path is currently the strongest implementation.
- `stage_refresh_region_slots_2x2` does not clearly improve over simpler region pooling, so stage-refresh memory remains unsupported.

Results-section wording:

> Dense bbox-mask localization reveals a robust spatial signal for online anchor/region interaction. The signal persists at 16x16 and 32x32 resolutions and holds across object-size strata, while FPN-style fixed fusion and local-state refinement remain weaker on IoU.

## 5. Qualitative Evidence

Overlay visualizations are available at:

`results/bbox_mask_overlays_16x16_seed41/`

Models:

- `anchor_only_no_prefill`
- `no_prefill_local_mix`
- `fpn_sum_lite`

Each image contains:

- input image,
- GT bbox mask,
- predicted mask,
- FP/FN error map,
- probability map.

Recommended figure layout:

| Sample type | Columns |
|---|---|
| small object | input, GT, local mix, FPN, anchor-only |
| medium object | input, GT, local mix, FPN, anchor-only |
| large object | input, GT, local mix, FPN, anchor-only |
| off-center object | input, GT, local mix, FPN, anchor-only |

The figure should emphasize where anchor-only reduces false negatives or produces more complete object-region coverage. If the overlays do not visually show the quantitative advantage, the qualitative claim should be kept modest.

## 6. Mechanistic Boundary

Supported:

- Clean no-prefill online anchor/region interaction.
- Dense mask supervision as a better spatial probe than center-only targets.
- Dual-mainline result: classification favors lightweight local-state refinement; dense spatial localization favors anchor-only interaction.

Not supported:

- Early prefill.
- Stale history pool.
- Full prefill-lattice memory.
- Old XAttnRes-style history protocol.
- Coarse grid/quadrant probes as strong evidence.
- Center regression or Gaussian center heatmap as reliable spatial probes.

Key boundary sentence:

> The dense-mask gains should be interpreted as evidence for online anchor/region interaction, not as evidence for memory-first or prefill-first vision modeling.

## 7. Engineering Innovation Points

These are concrete implementation directions that could become project contributions if they survive ablation.

### 7.1 Anchor-Lite Routing

Problem: `anchor_only_no_prefill` is spatially strongest but slower than `no_prefill_local_mix` and `fpn_sum_lite`.

Candidate variants:

- `anchor_only_slots_1`
- `anchor_only_slots_2`
- `anchor_only_slots_4`
- `anchor_only_read_every_2`
- `anchor_only_lowres_only`
- shared projection for anchor keys/values
- grouped or low-rank anchor projections

Success criterion:

- Keep at least `95%` of the anchor-only IoU gain.
- Recover speed toward the local/FPN range.

### 7.2 Spatial-Head Interface Standardization

Current heatmap/mask code depends on models returning either `memories` or `spatial_features`.

Engineering improvement:

- Standardize all spatial-capable models to return:
  - `logits`,
  - `spatial_features`,
  - `memories`,
  - optional `routing_maps`,
  - optional `mechanism_stats`.

Benefit:

- Fewer probe-specific wrappers.
- Fairer FPN/local/anchor comparison.
- Easier ViT adapter integration.

### 7.3 TinyViT Spatial Adapter

Problem: current heatmap/mask probes exclude `tiny_vit` because it returns only global logits.

Engineering improvement:

- Add a `TinyViTHeatmapAdapter`.
- Reshape patch tokens back to `[B, C, H, W]`.
- Attach the same `1x1` spatial head used by local/FPN/anchor models.

Benefit:

- Directly tests sequence ViT vs 2D local/anchor state on dense mask output.
- Prevents the spatial results from being criticized as missing a ViT-style control.

### 7.4 Overlay and Figure Curation Pipeline

Current overlay script generates per-model panels, but a paper figure needs cross-model sample grids.

Engineering improvement:

- Add a figure composer that takes the same sample index across models.
- Output one row per sample and one column per model.
- Include compact metric labels: IoU, Dice, FP/FN area.
- Stratify automatically by small/medium/large/off-center.

Benefit:

- Turns raw qualitative images into reproducible paper figures.
- Avoids cherry-picking by recording the sample-selection rule.

### 7.5 Mask-Probe Diagnostics

Add metrics beyond global IoU:

- boundary-band IoU,
- false-negative area ratio,
- false-positive area ratio,
- mask compactness,
- predicted/target area ratio,
- area-stratified center error.

Benefit:

- Identifies whether anchor-only helps by expanding coverage, improving boundary, or reducing center drift.

### 7.6 Multi-Task Classification and Mask Head

After anchor-lite:

```text
L = L_cls + lambda * L_mask
```

Compare:

- `no_prefill_local_mix`,
- `anchor_only_no_prefill`,
- best anchor-lite variant.

Questions:

- Does anchor-only preserve classification accuracy?
- Does mask supervision improve classification representation?
- Can one backbone be competitive on both mainlines?

### 7.7 Reproducible Experiment Registry

Current results are documented manually in `docs/review.md`.

Engineering improvement:

- Add a small result registry script:
  - reads CSVs from `results/`,
  - emits markdown tables,
  - computes paired deltas,
  - records command/config metadata.

Benefit:

- Reduces manual transcription risk.
- Makes future runs easier to audit.

## 8. Next Experiment Order

Recommended order:

1. Curate overlay figure from existing `16x16` outputs.
2. Add anchor-lite variants.
3. Run anchor-lite sweep on `32x32` bbox mask.
4. Add `TinyViTHeatmapAdapter`.
5. Run joint classification + mask training.
6. Add result-registry tooling if experiments keep expanding.

Do not prioritize:

- more center Gaussian tuning,
- more quadrant/grid classification,
- reviving full prefill-lattice before anchor-lite is understood,
- graph extensions before anchor-only is made efficient.

## 9. Detection Scaffold and AnchorQueryInit Toy

The RF-DETR push begins by turning the probe backbone into a minimal DETR-style detector.

Implemented components:

- `src/attention2d/detection/matcher.py`: tiny Hungarian-style matcher with class/L1/GIoU cost.
- `src/attention2d/detection/losses.py`: DETR-style criterion.
- `src/attention2d/detection/heads.py`: class and box heads.
- `src/attention2d/detection/anchor_region_detr.py`: `TinyAnchorRegionDETR` with learned or anchor query initialization.
- `scripts/train_det_toy.py`: synthetic square-detection runner.

Important engineering correction:

- The first toy smoke was not formal evidence because it used GIoU as the evaluation IoU, changed initialization between learned/anchor, and did not fully pair torch noise.
- The strict run now uses true IoU evaluation, shared initialization seed, and deterministic train/eval noise.

Strict toy result:

Output CSV: `results/det_toy_anchor_query_300step_5seeds_strict.csv`.

Setting: one square per image, 64px images, `embed_dim=16`, `num_queries=4`, 300 steps, 5 seeds.

| Query init | Final IoU | Best IoU | Recall@0.50 |
|---|---:|---:|---:|
| learned | 0.090 | 0.095 | 0.048 |
| anchor | 0.110 | 0.123 | 0.047 |

Paired result:

- Final IoU delta: `+0.020`, `5/5` wins for anchor.
- Best IoU delta: `+0.028`, `5/5` wins for anchor.

Interpretation:

- AnchorQueryInit has a real positive signal on the strict single-object toy.
- This supports using online anchor/region summaries as detector query seeds.
- The result is not yet a real detector claim: both query modes share the same anchor-region feature backbone, and the square toy can be solved partly by row/column projections.

Next detector controls:

The first control is now complete: feature backbone and query initialization were separated into a 2x2 matrix.

Output CSV: `results/det_toy_feature_query_2x2_300step_5seeds.csv`.

| Variant | Feature source | Query init | Final IoU | Best IoU | Recall@0.50 | Img/s |
|---|---|---|---:|---:|---:|---:|
| `local_learned` | local-state | learned | 0.079 | 0.088 | 0.033 | 320 |
| `local_anchor` | local-state | anchor | 0.099 | 0.131 | 0.048 | 321 |
| `anchor_learned` | anchor-region | learned | 0.090 | 0.095 | 0.048 | 254 |
| `anchor_anchor` | anchor-region | anchor | 0.110 | 0.123 | 0.047 | 277 |

Paired interpretation:

- `local_anchor` vs `local_learned`: final IoU `+0.020`, `4/5` wins; best IoU `+0.042`, `5/5` wins.
- `anchor_learned` vs `local_learned`: final IoU `+0.011`, `4/5` wins; best IoU `+0.007`, `4/5` wins.
- `anchor_anchor` vs `anchor_learned`: final IoU `+0.020`, `5/5` wins; best IoU `+0.028`, `5/5` wins.
- `anchor_anchor` vs `local_anchor`: final IoU `+0.011`, `4/5` wins; best IoU `-0.008`, `2/5` wins.

Updated interpretation:

- Content-conditioned anchor query generation has an independent positive signal even when the feature backbone is only local-state.
- Anchor-region features also help, but more weakly in this toy setting.
- The combined `anchor_anchor` variant gives the best final IoU, while `local_anchor` gives the best best-IoU and keeps higher throughput.
- This remains toy-level evidence: it supports anchor/region summaries as useful query generation signals, not RF-DETR-scale detector gains.
- Caveat: the current anchor queries are generated differentiably from the feature map. A detached-query control is needed before claiming this is purely an initialization effect.

Next detector controls:

The next control makes the toy harder with multiple labeled targets plus unlabeled distractor squares. An initial overlapping version was treated as contaminated because distractors could cover targets and targets could overlap. The formal toy now uses non-overlap placement.

Output CSV: `results/det_toy_multi_distractor_nonoverlap_2x2_300step_5seeds.csv`.

Setting: 64px images, 1-3 green target squares, 1-3 red distractor squares, controlled non-overlap, `embed_dim=16`, `num_queries=4`, 300 steps, 5 seeds. Evaluation reports target-matched IoU/Recall@0.50, so it measures whether target boxes are covered by queries; it is not AP and does not penalize false-positive queries.

| Variant | Final IoU | Best IoU | Recall@0.50 | Img/s |
|---|---:|---:|---:|---:|
| `local_learned` | 0.176 | 0.181 | 0.059 | 201 |
| `local_anchor` | 0.192 | 0.194 | 0.083 | 208 |
| `anchor_learned` | 0.194 | 0.198 | 0.086 | 182 |
| `anchor_anchor` | 0.200 | 0.200 | 0.096 | 188 |

Paired result vs `local_learned`:

- `local_anchor`: final IoU `+0.016`, `4/5` wins; best IoU `+0.012`, `4/5` wins.
- `anchor_learned`: final IoU `+0.018`, `3/5` wins; best IoU `+0.016`, `3/5` wins.
- `anchor_anchor`: final IoU `+0.024`, `3/5` wins; best IoU `+0.018`, `3/5` wins.

Interpretation:

- The harder non-overlap multi-distractor toy still supports anchor/region components over the local learned-query baseline.
- The result is weaker and less clean than the single-object square toy: wins are not 5/5, and `anchor_anchor` is only marginally above `anchor_learned` and `local_anchor`.
- The safest claim is that anchor/region generation remains promising under distractors, but it is not yet a robust detector-level result.

Detached-query control:

`local_anchor_detached` was added to separate anchor query content from the extra query-side gradient path. It builds anchor queries from `spatial_state.detach()`, so decoder loss cannot flow through the query branch back into the feature map.

Single-object target-matched eval:

Output CSV: `results/det_toy_local_anchor_detached_single_300step_5seeds.csv`.

| Variant | Final IoU | Best IoU | Recall@0.50 | Img/s |
|---|---:|---:|---:|---:|
| `local_learned` | 0.195 | 0.200 | 0.094 | 287 |
| `local_anchor` | 0.241 | 0.250 | 0.138 | 294 |
| `local_anchor_detached` | 0.258 | 0.265 | 0.144 | 286 |

Single-object interpretation:

- Detaching the anchor query branch does not remove the signal; it improves the mean result in this simple setting.
- This suggests that simple square localization benefits strongly from the anchor query content/geometry summary itself.
- The absolute values are not directly comparable to the earlier single-object table because this run uses target-matched IoU eval.

Non-overlap multi-distractor eval:

Output CSV: `results/det_toy_local_anchor_detached_multi_distractor_300step_5seeds.csv`.

| Variant | Final IoU | Best IoU | Recall@0.50 | Img/s |
|---|---:|---:|---:|---:|
| `local_learned` | 0.176 | 0.181 | 0.059 | 203 |
| `local_anchor` | 0.192 | 0.194 | 0.083 | 209 |
| `local_anchor_detached` | 0.163 | 0.166 | 0.077 | 202 |

Multi-distractor interpretation:

- Detached anchor query content does not transfer to the harder setting; it falls below `local_learned`.
- The differentiable `local_anchor` branch remains positive, suggesting that online query-feature coupling matters when multiple targets and distractors are present.
- This is a useful boundary: anchor query content alone is enough for the simplest toy, but the harder toy needs either gradient-coupled query generation, a better decoder, or AP-aware training/evaluation.

AP-lite control:

`scripts/train_det_toy.py` now reports an additional confidence-aware `eval_ap50`. It sorts queries by object-class probability and greedily matches predictions to unmatched targets at IoU `0.50`. This is still a toy AP-lite metric, not COCO AP, but it penalizes false-positive ordering in a way target-matched IoU does not.

Output CSV: `results/det_toy_aplite_multi_distractor_300step_5seeds.csv`.

| Variant | Final IoU | Best IoU | Recall@0.50 | AP50-lite | Img/s |
|---|---:|---:|---:|---:|---:|
| `local_learned` | 0.176 | 0.181 | 0.059 | 0.039 | 161 |
| `local_anchor` | 0.192 | 0.194 | 0.083 | 0.048 | 161 |
| `local_anchor_detached` | 0.163 | 0.166 | 0.077 | 0.055 | 162 |

AP-lite interpretation:

- `local_anchor` remains the stronger target-coverage/localization model: final IoU `+0.016`, best IoU `+0.012`, both `4/5` wins vs `local_learned`.
- `local_anchor_detached` has worse target-matched IoU but higher AP50-lite, suggesting it can rank some true-positive queries better even when coverage is worse.
- This separates two failure modes: query boxes must cover targets, and object scores must rank good boxes before false positives.
- The next detector step should train and evaluate these aspects more explicitly rather than relying only on target-matched IoU.

Next detector controls:

DenseMaskAux control:

The detection toy now supports explicit mask-aux model variants. A variant with suffix `_maskaux` adds a `1x1` dense mask head on the detector spatial feature map and trains it with BCE + Dice against the union of target bbox rectangle masks.

Output CSV: `results/det_toy_densemaskaux_multi_distractor_300step_5seeds.csv`.

| Variant | Final IoU | Best IoU | AP50-lite | Mask IoU | Mask Dice |
|---|---:|---:|---:|---:|---:|
| `local_learned` | 0.176 | 0.181 | 0.039 | N/A | N/A |
| `local_learned_maskaux` | 0.170 | 0.172 | 0.054 | 0.983 | 0.991 |
| `local_anchor` | 0.192 | 0.194 | 0.048 | N/A | N/A |
| `local_anchor_maskaux` | 0.189 | 0.189 | 0.046 | 0.982 | 0.991 |
| `local_anchor_detached` | 0.172 | 0.172 | 0.044 | N/A | N/A |
| `local_anchor_detached_maskaux` | 0.177 | 0.192 | 0.041 | 0.987 | 0.993 |

Paired interpretation:

- `local_anchor_maskaux` vs `local_anchor`: final IoU `-0.003`, best IoU `-0.005`, AP50-lite `-0.002`.
- `local_learned_maskaux` vs `local_learned`: final IoU `-0.006`, best IoU `-0.010`, AP50-lite `+0.015`.
- `local_anchor_detached_maskaux` vs `local_anchor_detached`: final IoU `+0.005`, best IoU `+0.020`, AP50-lite `-0.002`.

DenseMaskAux interpretation:

- Dense mask supervision is easy for the spatial feature map: all `_maskaux` variants learn bbox rectangle masks with mask IoU around `0.98`.
- In the current tiny detector, this side auxiliary does not reliably improve box localization or AP-lite.
- The best current coverage model remains `local_anchor` without DenseMaskAux.
- Therefore, the next useful step is not more side supervision by itself, but a stronger coupling between dense spatial masks and query/box refinement.

Maskpooled-query coupling:

The first direct coupling variant predicts an internal dense mask, soft-pools a foreground region summary from the spatial feature map, and adds that region vector to every query before decoding.

Output CSV: `results/det_toy_maskpooled_query_multi_distractor_300step_5seeds.csv`.

| Variant | Final IoU | Best IoU | AP50-lite | Mask IoU | Mask Dice |
|---|---:|---:|---:|---:|---:|
| `local_anchor` | 0.192 | 0.194 | 0.048 | N/A | N/A |
| `local_anchor_maskaux` | 0.189 | 0.189 | 0.046 | 0.982 | 0.991 |
| `local_anchor_maskpooled_query` | 0.172 | 0.179 | 0.034 | 0.978 | 0.989 |
| `local_learned` | 0.176 | 0.181 | 0.039 | N/A | N/A |
| `local_learned_maskpooled_query` | 0.163 | 0.166 | 0.053 | 0.989 | 0.995 |

Maskpooled interpretation:

- Like DenseMaskAux, the mask branch itself learns the union mask well.
- The global pooled foreground vector does not improve box coverage; it hurts `local_anchor` by final IoU `-0.020` and AP50-lite `-0.014`.
- For `local_learned`, it improves AP50-lite by `+0.015` but lowers final/best IoU, so it mainly changes confidence ordering rather than localization.
- This suggests that query consumption of dense masks must be query-specific. A single foreground summary broadcast to all queries is too coarse for multi-object/distractor detection.

Mask-biased query attention:

The second direct coupling variant predicts an internal dense mask and adds `log(sigmoid(mask))` as a soft foreground bias to the query-to-spatial attention logits. This preserves per-query QK selection while nudging attention toward predicted foreground regions.

Output CSV: `results/det_toy_mask_biased_attn_multi_distractor_300step_5seeds.csv`.

| Variant | Final IoU | Best IoU | AP50-lite | Mask IoU | Mask Dice |
|---|---:|---:|---:|---:|---:|
| `local_anchor` | 0.192 | 0.194 | 0.048 | N/A | N/A |
| `local_anchor_maskaux` | 0.189 | 0.189 | 0.046 | 0.982 | 0.991 |
| `local_anchor_maskpooled_query` | 0.172 | 0.179 | 0.034 | 0.978 | 0.989 |
| `local_anchor_mask_biased_attn` | 0.189 | 0.189 | 0.052 | 0.978 | 0.989 |
| `local_learned` | 0.176 | 0.181 | 0.039 | N/A | N/A |
| `local_learned_maskpooled_query` | 0.163 | 0.166 | 0.053 | 0.989 | 0.995 |
| `local_learned_mask_biased_attn` | 0.222 | 0.227 | 0.093 | 0.989 | 0.994 |

Paired interpretation:

- `local_anchor_mask_biased_attn` vs `local_anchor`: final IoU `-0.003` with `3/5` wins, best IoU `-0.005` with `3/5` wins, AP50-lite `+0.004` with `3/5` wins.
- `local_learned_mask_biased_attn` vs `local_learned`: final IoU `+0.046` but only `2/5` wins, AP50-lite `+0.054` with `3/5` wins.
- `local_learned_mask_biased_attn` vs `local_anchor`: final IoU `+0.030` but only `1/5` wins, AP50-lite `+0.044` with `2/5` wins.

Mask-biased interpretation:

- Mask-biased attention is better than global mask pooling as a coupling mechanism: it avoids the large coverage drop seen in `local_anchor_maskpooled_query`.
- For `local_anchor`, it is roughly tied on IoU and gives a small AP-lite signal, but it does not clearly beat the current coverage baseline.
- For `local_learned`, the mean is much higher but seed wins are weak, so this is an unstable optimization signal rather than robust evidence.
- The next useful step is to stabilize this path with bias-gate schedules and compare it with mask-proposal query initialization.

Mask-bias gate controls:

This control set tests whether the seed instability comes from applying the foreground bias too strongly or too early. It adds a smaller gate initialization (`0.01`) and a linear gate warmup schedule.

Output CSV: `results/det_toy_mask_bias_gate_multi_distractor_300step_5seeds.csv`.

| Variant | Final IoU | Best IoU | AP50-lite | Mask IoU | Mask Dice |
|---|---:|---:|---:|---:|---:|
| `local_anchor` | 0.192 | 0.194 | 0.048 | N/A | N/A |
| `local_anchor_mask_biased_attn` | 0.189 | 0.189 | 0.052 | 0.978 | 0.989 |
| `local_anchor_mask_biased_attn_gate001` | 0.188 | 0.188 | 0.041 | 0.981 | 0.990 |
| `local_anchor_mask_biased_attn_warmup` | 0.175 | 0.181 | 0.037 | 0.977 | 0.988 |
| `local_learned` | 0.176 | 0.181 | 0.039 | N/A | N/A |
| `local_learned_mask_biased_attn` | 0.222 | 0.227 | 0.093 | 0.989 | 0.994 |
| `local_learned_mask_biased_attn_gate001` | 0.224 | 0.224 | 0.097 | 0.985 | 0.992 |
| `local_learned_mask_biased_attn_warmup` | 0.232 | 0.234 | 0.103 | 0.989 | 0.994 |

Paired interpretation:

- `local_anchor_mask_biased_attn_gate001` vs `local_anchor`: final IoU `-0.004` with `2/5` wins, AP50-lite `-0.007` with `1/5` wins.
- `local_anchor_mask_biased_attn_warmup` vs `local_anchor`: final IoU `-0.017` with `1/5` wins, AP50-lite `-0.011` with `1/5` wins.
- `local_learned_mask_biased_attn_gate001` vs `local_learned`: final IoU `+0.048` with `2/5` wins, AP50-lite `+0.058` with `3/5` wins.
- `local_learned_mask_biased_attn_warmup` vs `local_learned`: final IoU `+0.056` with `2/5` wins, AP50-lite `+0.064` with `4/5` wins.
- Against `local_anchor`, the learned-query gate variants improve mean IoU/AP but still win only `2/5` seeds on IoU.

Gate-stability interpretation:

- Smaller gate and warmup do not improve the anchor-query branch.
- For learned queries, they preserve or slightly raise the mean, especially AP-lite, but still do not fix seed instability.
- The high learned-query means remain driven by a subset of seeds, so this path is not yet a stable mainline.
- The next detector step should test `mask_proposal_query_init`, where the dense mask explicitly proposes query seeds instead of only biasing attention.

Mask proposal query initialization:

This variant predicts a dense foreground mask, takes the top-k mask cells, and gathers those spatial feature tokens as object query seeds. This is a harder coupling than mask-biased attention: the dense mask directly controls which spatial tokens become object queries.

Output CSV: `results/det_toy_mask_proposal_query_multi_distractor_300step_5seeds.csv`.

| Variant | Final IoU | Best IoU | Recall50 | AP50-lite | Mask IoU | Mask Dice |
|---|---:|---:|---:|---:|---:|---:|
| `local_anchor` | 0.192 | 0.194 | 0.083 | 0.048 | N/A | N/A |
| `local_learned` | 0.176 | 0.181 | 0.059 | 0.039 | N/A | N/A |
| `local_anchor_mask_biased_attn` | 0.189 | 0.189 | 0.091 | 0.052 | 0.978 | 0.989 |
| `local_learned_mask_biased_attn_warmup` | 0.232 | 0.234 | 0.148 | 0.103 | 0.989 | 0.994 |
| `local_mask_proposal_query` | 0.328 | 0.342 | 0.363 | 0.295 | 0.986 | 0.993 |
| `anchor_mask_proposal_query` | 0.321 | 0.324 | 0.341 | 0.272 | 0.990 | 0.995 |

Paired interpretation:

- `local_mask_proposal_query` vs `local_anchor`: final IoU `+0.136`, best IoU `+0.149`, AP50-lite `+0.247`, with `5/5` paired wins on all three metrics.
- `anchor_mask_proposal_query` vs `local_anchor`: final IoU `+0.129`, best IoU `+0.130`, AP50-lite `+0.224`, with `5/5` paired wins on all three metrics.
- Proposal-query variants also outperform the best mask-biased attention variant in mean IoU, recall50, and AP50-lite.

Mask-proposal interpretation:

- This is the first detection toy result that strongly supports dense mask to query coupling.
- The positive signal is stable across all five seeds and much larger than the previous mask-biased attention signal.
- The supported claim is specific: dense foreground masks help when they explicitly initialize object queries.
- This should not be generalized to early prefill or memory-first routing.

Proposal diversity with spatial suppression:

Plain top-k proposal queries can over-select nearby mask peaks from the same foreground region. The NMS-style variant greedily selects high-mask cells while suppressing nearby feature-grid cells with radius `2`.

Output CSV: `results/det_toy_mask_proposal_nms_multi_distractor_300step_5seeds.csv`.

| Variant | Final IoU | Best IoU | Recall50 | AP50-lite | Mask IoU | Mask Dice |
|---|---:|---:|---:|---:|---:|---:|
| `local_anchor` | 0.192 | 0.194 | 0.083 | 0.048 | N/A | N/A |
| `local_mask_proposal_query` | 0.328 | 0.342 | 0.363 | 0.295 | 0.986 | 0.993 |
| `local_mask_proposal_nms_query` | 0.461 | 0.463 | 0.455 | 0.378 | 0.979 | 0.989 |
| `anchor_mask_proposal_query` | 0.321 | 0.324 | 0.341 | 0.272 | 0.990 | 0.995 |
| `anchor_mask_proposal_nms_query` | 0.438 | 0.465 | 0.394 | 0.313 | 0.982 | 0.990 |

Paired interpretation:

- `local_mask_proposal_nms_query` vs `local_mask_proposal_query`: final IoU `+0.133`, best IoU `+0.120`, AP50-lite `+0.083`; final/best IoU wins are `5/5`, AP50-lite wins are `3/5`.
- `anchor_mask_proposal_nms_query` vs `local_mask_proposal_query`: final IoU `+0.110`, best IoU `+0.122`, AP50-lite `+0.018`; final/best IoU wins are `5/5`, AP50-lite wins are `3/5`.
- Both NMS-style proposal variants remain far above `local_anchor`.

Proposal-diversity interpretation:

- Spatial suppression is a major improvement over plain top-k proposal queries.
- The mask score itself is not the limiting metric: NMS variants have slightly lower mask IoU than plain top-k but much better box IoU and AP-lite.
- This supports the mechanism that query diversity and spatial coverage are central for turning dense masks into detector queries.
- If qualitative proposal figures are blurred or not readable, use curated clearer samples; do not rely on weak visual evidence.

Stratified proposal evaluation:

The detection toy now reports object-size and center/off-center strata. Size is based on bbox area, and off-center is based on bbox-center distance from the image center.

Output CSV: `results/det_toy_proposal_nms_stratified_multi_distractor_300step_5seeds.csv`.

| Variant | Small IoU | Medium IoU | Large IoU | Center IoU | Off-center IoU |
|---|---:|---:|---:|---:|---:|
| `local_anchor` | 0.102 | 0.236 | 0.325 | 0.174 | 0.197 |
| `local_mask_proposal_query` | 0.231 | 0.393 | 0.443 | 0.278 | 0.344 |
| `local_mask_proposal_nms_query` | 0.363 | 0.561 | 0.522 | 0.408 | 0.479 |
| `anchor_mask_proposal_nms_query` | 0.351 | 0.514 | 0.512 | 0.388 | 0.455 |

Stratified interpretation:

- NMS proposal improves over plain top-k on small, medium, and large objects.
- NMS proposal also improves both center and off-center targets, so the gain is not only a center-prior effect.
- The small-object improvement is especially important: `local_mask_proposal_nms_query` raises small-object IoU from `0.231` to `0.363`.
- For qualitative figures, use samples that make these strata visible: include small/medium/large and center/off-center cases, but replace blurry or visually ambiguous panels with clearer examples.
- Clear sample curation is now scripted with `scripts/select_clear_det_overlay_samples.py`.
  - Balanced candidate sheet: `results/clear_det_overlay_samples/contact_sheet.jpg`.
  - Higher-clarity figure candidate sheet: `results/clear_det_overlay_samples_high_area/contact_sheet.jpg`.
  - The high-area sheet should be the first choice for readable paper figures; the balanced sheet is useful when the figure needs explicit small/medium/large and center/off-center diversity.
- Fixed high-area prediction overlays are under `results/fixed_high_area_bbox_mask_overlays_32x32_seed41/`.
  - Main cross-model sheet: `comparison_grid.png`.
  - Columns: original image, GT bbox mask, `anchor_only_no_prefill`, `no_prefill_local_mix`, and `fpn_sum_lite`.
  - The grid uses original JPEGs resized for display, not enlarged 64px model inputs, so it is suitable for qualitative review.

Next detector controls:

1. proposal overlay visualization with curated clear examples.
2. connected-components or soft-NMS proposal variants.
3. confidence-aware loss or decoder refinement for AP-lite.
4. DET annotation loader using real boxes.
5. RF-DETR distillation path with mask/proposal query student.

Real DET mini-detector path:

`scripts/train_det_real.py` now trains the tiny detector variants on real ILSVRC2013 DET XML boxes. This is an image-level dataset path: each sample contains one image with multiple boxes and labels, not one row per object. The v1 transform is direct square resize/stretch, so normalized boxes are computed from original XML coordinates and original image width/height.

Smoke configuration:

```text
/opt/anaconda3/envs/AIAA/bin/python -u scripts/train_det_real.py \
  --models local_learned local_anchor local_mask_proposal_nms_query \
  --reference-model local_learned \
  --top-classes 10 \
  --max-samples 200 \
  --max-objects 3 \
  --num-queries 6 \
  --steps 50 \
  --eval-every 50 \
  --eval-batches 4 \
  --batch-size 16 \
  --out results/det_real_mini_50step_2seed.csv \
  --label-map-out results/det_real_mini_50step_label_map.csv \
  --split-out results/det_real_mini_50step_split.csv \
  --seeds 2
```

Output artifacts:

- Metrics: `results/det_real_mini_50step_2seed.csv`
- Label map: `results/det_real_mini_50step_label_map.csv`
- Image-level split: `results/det_real_mini_50step_split.csv`

| Variant | IoU | Recall50 | Objectness AP50-lite | Class-aware AP50-lite | Mask IoU | Mask Dice |
|---|---:|---:|---:|---:|---:|---:|
| `local_learned` | 0.314 | 0.229 | 0.257 | 0.077 | 0.000 | 0.000 |
| `local_anchor` | 0.361 | 0.356 | 0.296 | 0.082 | 0.000 | 0.000 |
| `local_mask_proposal_nms_query` | 0.306 | 0.241 | 0.255 | 0.097 | 0.400 | 0.537 |

Paired interpretation:

- `local_anchor` improves matched IoU over `local_learned` by `+0.047` with `2/2` seed wins.
- `local_anchor` also improves recall50 and objectness AP50-lite, but the class-aware AP gain is small.
- `local_mask_proposal_nms_query`, the strongest toy detection model, does not yet transfer to real-box IoU in this short run. It does learn a dense mask and has the highest class-aware AP50-lite, but its matched IoU is below `local_learned`.

Current interpretation:

- The real DET mini-detector path is now executable end-to-end.
- The first real-box smoke supports the cleaner online anchor path more than the mask-proposal NMS path.
- This should be treated as an early engineering checkpoint, not RF-DETR-level evidence.
- Next step: run a longer 300-step real DET mini experiment including `local_anchor_detached`, then add proposal/box overlays on the fixed high-clarity image set.

300-step detached-anchor follow-up:

The 300-step follow-up keeps the same top-10 / 200-image / 2-seed mini setup, adds `local_anchor_detached`, and evaluates every 100 steps.

Output artifacts:

- Metrics: `results/det_real_mini_300step_2seed.csv`
- Label map: `results/det_real_mini_300step_label_map.csv`
- Image-level split: `results/det_real_mini_300step_split.csv`

| Variant | Final IoU | Best IoU | Recall50 | Objectness AP50-lite | Class-aware AP50-lite | Mask IoU | Mask Dice |
|---|---:|---:|---:|---:|---:|---:|---:|
| `local_learned` | 0.359 | 0.367 | 0.321 | 0.310 | 0.108 | 0.000 | 0.000 |
| `local_anchor` | 0.359 | 0.367 | 0.312 | 0.236 | 0.068 | 0.000 | 0.000 |
| `local_anchor_detached` | 0.343 | 0.353 | 0.297 | 0.214 | 0.049 | 0.000 | 0.000 |
| `local_mask_proposal_nms_query` | 0.357 | 0.366 | 0.302 | 0.266 | 0.093 | 0.458 | 0.604 |

Paired interpretation:

- `local_anchor` no longer has a stable advantage over `local_learned` at 300 steps: final IoU delta is about `-0.001`, with `1/2` wins.
- `local_anchor_detached` is worse than both `local_anchor` and `local_learned`, suggesting that detached anchor content alone is not enough for real-box detection.
- `local_mask_proposal_nms_query` still learns a dense foreground mask, but the toy proposal-NMS advantage does not automatically transfer to real-box IoU.

Current real-box conclusion:

- The 50-step anchor advantage should be treated as an early-training signal, not a stable detector result.
- On real DET boxes, anchor query content appears to need differentiable coupling or a stronger decoder; static detached summaries are weaker.
- The next engineering bottleneck is no longer dataset plumbing; it is decoder/box refinement and class/objectness training.

300-step ranking/classification diagnostics:

The same 300-step setup was rerun with additional diagnostics for confidence ordering and query assignment.

Output artifacts:

- Metrics: `results/det_real_mini_300step_2seed_diagnostics.csv`
- Label map: `results/det_real_mini_300step_diagnostics_label_map.csv`
- Image-level split: `results/det_real_mini_300step_diagnostics_split.csv`

| Variant | Matched-assignment class acc | Score-IoU corr | Pre-NMS objectness AUC | Top-k FP rate | Duplicate / GT | Query entropy |
|---|---:|---:|---:|---:|---:|---:|
| `local_learned` | 0.390 | 0.438 | 0.658 | 0.719 | 0.252 | 0.904 |
| `local_anchor` | 0.390 | 0.305 | 0.584 | 0.819 | 0.204 | 0.926 |
| `local_anchor_detached` | 0.347 | 0.194 | 0.582 | 0.848 | 0.217 | 0.897 |
| `local_mask_proposal_nms_query` | 0.401 | 0.380 | 0.598 | 0.767 | 0.283 | 0.994 |

Diagnostic interpretation:

- `local_anchor` matches `local_learned` on matched IoU, but its confidence ordering is worse: score-IoU correlation, pre-NMS objectness AUC, AP50-lite, and top-k false-positive rate all favor `local_learned`.
- `local_anchor_detached` is weaker on both box coverage and ranking diagnostics, reinforcing that static detached anchor summaries are insufficient for real-box detection.
- `local_mask_proposal_nms_query` learns a real dense mask and has the highest matched-assignment class accuracy and query assignment entropy, but its score calibration still trails `local_learned`.
- The immediate bottleneck is not whether the model can produce boxes or masks; it is whether query scores and class predictions align with the best-localized boxes.

Updated real-box conclusion:

> The toy-level anchor/query and mask-proposal gains do not stably transfer to 300-step real DET mini. Anchor-like branches can match box IoU, but currently lose on ranking, objectness calibration, and class-aware AP. Future real-detector work should prioritize score-IoU alignment, query classification calibration, gated residual anchor-query initialization, oracle proposal controls, and stronger decoder refinement before scaling toward RF-DETR comparisons.

P0 oracle proposal and gated residual anchor follow-up:

The follow-up directly tests two bottlenecks:

- proposal quality vs query consumption, using GT foreground-union mask oracle proposal queries;
- anchor content vs learned-query replacement, using a small gated anchor residual added to learned queries.

Output artifacts:

- Metrics: `results/det_real_p0_oracle_anchor_residual_300step_2seed.csv`
- Label map: `results/det_real_p0_oracle_anchor_residual_label_map.csv`
- Image-level split: `results/det_real_p0_oracle_anchor_residual_split.csv`

| Variant | Final IoU | Best IoU | Recall50 | Objectness AP50-lite | Class-aware AP50-lite | Score-IoU corr | Objectness AUC | Mask IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `local_learned` | 0.359 | 0.367 | 0.321 | 0.310 | 0.108 | 0.438 | 0.658 | 0.000 |
| `local_anchor` | 0.359 | 0.367 | 0.312 | 0.236 | 0.068 | 0.305 | 0.584 | 0.000 |
| `local_anchor_residual_query` | 0.376 | 0.376 | 0.371 | 0.266 | 0.135 | 0.339 | 0.602 | 0.000 |
| `local_anchor_residual_detached_query` | 0.367 | 0.378 | 0.320 | 0.256 | 0.076 | 0.404 | 0.597 | 0.000 |
| `local_mask_proposal_nms_query` | 0.357 | 0.366 | 0.302 | 0.266 | 0.093 | 0.380 | 0.598 | 0.458 |
| `local_mask_proposal_oracle_nms_query` | 0.399 | 0.421 | 0.340 | 0.259 | 0.138 | 0.182 | 0.537 | 1.000 |

Paired against `local_learned`:

- `local_anchor_residual_query`: final IoU `+0.016`, best IoU `+0.009`, class-aware AP50-lite `+0.027`, all `2/2` wins except objectness AP.
- `local_anchor_residual_detached_query`: final IoU `+0.007`, best IoU `+0.011`, but class-aware AP50-lite `-0.032`.
- `local_mask_proposal_oracle_nms_query`: final IoU `+0.040`, best IoU `+0.054`, best-IoU wins `2/2`, but objectness AP50-lite `-0.051`.

Interpretation:

- In this mini DET setting, oracle NMS proposal queries recover a clear box-coverage gain. This means predicted foreground-union proposal quality/extraction is a real bottleneck for the mask-proposal path.
- Oracle proposals still do not fix AP/ranking. Even with idealized union-mask proposal locations, objectness calibration and score-IoU alignment remain weak.
- Gated residual anchor initialization is a better real-DET anchor path than replacing learned queries with anchor queries. It improves final IoU, recall50, and class-aware AP50-lite over both `local_learned` and replacement-style `local_anchor`.
- Detached residual anchors are weaker than differentiable residual anchors on final IoU and class-aware AP, so query-feature coupling still matters.

Updated real-DET P0 conclusion:

> Real DET P0 now separates two bottlenecks in the current mini setup. Box coverage can improve when foreground-union proposal locations are idealized or when anchor information is added as a soft residual to learned queries. However, AP remains limited by query scoring and class/objectness calibration. The next real-detector step should combine residual anchor queries with score-IoU/objectness calibration and improve predicted proposal quality toward the oracle proposal upper bound.

Score-IoU/objectness calibration follow-up:

This follow-up adds a simple class-agnostic objectness calibration auxiliary loss. For each query, the aggregate foreground-vs-background logit is trained with BCE against the query's detached max IoU to any target. This targets objectness quality, but it is not the same score used by AP-lite, which still ranks by max foreground softmax probability.

Output artifacts:

- Weight `0.5`: `results/det_real_residual_anchor_score_iou_calib_300step_2seed.csv`
- Weight `0.1`: `results/det_real_residual_anchor_score_iou_calib_w01_300step_2seed.csv`

Weight `0.5`:

| Variant | Final IoU | Best IoU | Recall50 | Objectness AP50-lite | Class-aware AP50-lite | Score-IoU corr | Objectness AUC | Top-k FP rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `local_learned` | 0.359 | 0.367 | 0.321 | 0.310 | 0.108 | 0.438 | 0.658 | 0.719 |
| `local_learned_calib` | 0.376 | 0.376 | 0.369 | 0.355 | 0.144 | 0.419 | 0.682 | 0.665 |
| `local_anchor_residual_query` | 0.376 | 0.376 | 0.371 | 0.266 | 0.135 | 0.339 | 0.602 | 0.802 |
| `local_anchor_residual_query_calib` | 0.361 | 0.372 | 0.329 | 0.257 | 0.102 | 0.351 | 0.624 | 0.792 |

Weight `0.1`:

| Variant | Final IoU | Best IoU | Recall50 | Objectness AP50-lite | Class-aware AP50-lite | Score-IoU corr | Objectness AUC | Top-k FP rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `local_anchor_residual_query` | 0.376 | 0.376 | 0.371 | 0.266 | 0.135 | 0.339 | 0.602 | 0.802 |
| `local_anchor_residual_query_calib` | 0.373 | 0.373 | 0.327 | 0.250 | 0.099 | 0.429 | 0.631 | 0.800 |
| `local_learned_calib` | 0.373 | 0.382 | 0.310 | 0.262 | 0.082 | 0.346 | 0.627 | 0.779 |

Calibration interpretation:

- The naive max-IoU calibration loss helps `local_learned` at weight `0.5`: it improves final IoU, objectness AP50-lite, class-aware AP50-lite, objectness AUC, and top-k false-positive rate. It does not improve score-IoU correlation in this run.
- The same loss does not help residual-anchor. At weight `0.5`, it hurts final IoU and AP. At weight `0.1`, it improves score-IoU correlation and objectness AUC, but still hurts objectness AP and class-aware AP.
- This suggests that residual-anchor's AP problem is not solved by all-query max-IoU BCE. The loss likely conflicts with DETR-style one-to-one assignment because duplicate queries with high IoU are treated as soft positives.

Updated calibration conclusion:

> Score-IoU calibration is useful for learned queries, but the naive all-query max-IoU version conflicts with residual-anchor query behavior. The next calibration design should be matcher-aware: matched queries should receive IoU-quality class/objectness targets, while unmatched foreground class logits are pushed toward zero and no-object handling remains mainly from standard DETR CE. This is closer to quality focal / Varifocal-style detection training than raw max-IoU BCE over every query.

Matcher-aware quality classification follow-up:

This follow-up removes the duplicate-positive issue from all-query max-IoU BCE. It first runs the Hungarian matcher, then assigns only each matched query's target class an IoU-quality soft label. All unmatched foreground class logits remain zero.

Output artifacts:

- Weight `1.0`: `results/det_real_matcher_quality_cls_300step_2seed.csv`
- Weight `0.25`: `results/det_real_matcher_quality_cls_w025_300step_2seed.csv`

Weight `1.0`:

| Variant | Final IoU | Best IoU | Recall50 | Objectness AP50-lite | Class-aware AP50-lite | Score-IoU corr | Objectness AUC | Top-k FP rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `local_anchor_residual_query` | 0.376 | 0.376 | 0.371 | 0.266 | 0.135 | 0.339 | 0.602 | 0.802 |
| `local_anchor_residual_query_matchqual` | 0.366 | 0.392 | 0.358 | 0.281 | 0.092 | 0.456 | 0.645 | 0.771 |
| `local_learned` | 0.359 | 0.367 | 0.321 | 0.310 | 0.108 | 0.438 | 0.658 | 0.719 |
| `local_learned_matchqual` | 0.360 | 0.360 | 0.301 | 0.239 | 0.080 | 0.366 | 0.607 | 0.800 |

Weight `0.25`:

| Variant | Final IoU | Best IoU | Recall50 | Objectness AP50-lite | Class-aware AP50-lite | Score-IoU corr | Objectness AUC | Top-k FP rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `local_anchor_residual_query` | 0.376 | 0.376 | 0.371 | 0.266 | 0.135 | 0.339 | 0.602 | 0.802 |
| `local_anchor_residual_query_matchqual` | 0.370 | 0.371 | 0.371 | 0.258 | 0.133 | 0.253 | 0.551 | 0.804 |
| `local_learned` | 0.359 | 0.367 | 0.321 | 0.310 | 0.108 | 0.438 | 0.658 | 0.719 |

Matcher-aware interpretation:

- The matcher-aware loss avoids the duplicate-positive flaw of all-query max-IoU BCE, but the current BCE-on-foreground-class-logits form still does not improve the residual-anchor final result.
- At weight `1.0`, it improves residual-anchor best IoU and objectness AP50-lite, but lowers final IoU and class-aware AP50-lite.
- At weight `0.25`, the effect is smaller and still not positive.
- For learned queries, matcher-aware quality classification is worse than the base learned model and worse than the previous naive max-IoU calibration.

Updated scoring conclusion:

> Matcher-aware quality classification is conceptually cleaner than all-query max-IoU BCE, but this implementation is not the right scoring fix because it still mixes localization quality into class logits. The current path moves quality calibration into a separate query-level quality/objectness head.

Separate query-quality head:

The next scoring control decouples semantic class prediction from localization-quality scoring. `DetectionHead` now emits an independent `pred_quality_logits` tensor. The quality head is trained only for `*_quality_head` variants: Hungarian matched queries receive detached matched IoU targets, while unmatched queries receive target `0`. Standard DETR CE remains unchanged for class logits.

Output artifacts:

- Metrics: `results/det_real_quality_head_300step_2seed.csv`
- Label map: `results/det_real_quality_head_label_map.csv`
- Image-level split: `results/det_real_quality_head_split.csv`

| Variant | Final IoU | Best IoU | AP50-lite | Class-aware AP50-lite | AP50 q^1 | Class AP50 q^1 | Quality-IoU corr | Quality AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `local_anchor_residual_query` | 0.376 | 0.376 | 0.266 | 0.135 | 0.266 | 0.135 | 0.000 | 0.500 |
| `local_anchor_residual_query_quality_head` | 0.359 | 0.367 | 0.262 | 0.134 | 0.330 | 0.175 | 0.619 | 0.701 |
| `local_learned` | 0.359 | 0.367 | 0.310 | 0.108 | 0.310 | 0.108 | 0.000 | 0.500 |
| `local_learned_quality_head` | 0.374 | 0.374 | 0.263 | 0.082 | 0.306 | 0.094 | 0.559 | 0.666 |

Paired against `local_anchor_residual_query`:

- `local_anchor_residual_query_quality_head`: final IoU `-0.017`, best IoU `-0.010`, AP50-lite `-0.004`, class-aware AP50-lite `-0.001`, but quality-aware AP50 q^1 `+0.064` with `2/2` wins.
- `local_learned`: AP50-lite `+0.044` with `2/2` wins, but final IoU `-0.016`.
- `local_learned_quality_head`: final IoU `-0.001`, AP50-lite `-0.003`, quality-aware AP50 q^1 `+0.044` with `2/2` wins.

Quality-head interpretation:

- The separate quality head is the right direction for score calibration because it no longer forces IoU targets into class logits.
- For residual-anchor, quality scores meaningfully improve ranking when used in inference scoring: `AP50 q^1` rises to `0.330`, above the residual-anchor base AP `0.266`.
- However, the auxiliary quality training also lowers residual-anchor final IoU and does not improve the base class-probability AP. Therefore, this is a ranking signal, not yet a detector-level improvement.
- For learned queries, the quality head does not beat the simpler `local_learned` AP reference.
- The next scoring step should reduce interference rather than change the target semantics: lower quality-head weight, late-start quality training, or train the quality head after freezing a base detector.

Low-interference quality-head controls:

Three controls test whether the quality head can keep its ranking benefit without disturbing box/class learning.

| Control | Residual-anchor final IoU | Residual-anchor AP50 | AP50 q^1 | Class AP50 q^1 | Interpretation |
|---|---:|---:|---:|---:|---|
| Base `local_anchor_residual_query` | 0.376 | 0.266 | 0.266 | 0.135 | No quality scoring. |
| Always-on quality, weight 1.0 | 0.359 | 0.262 | 0.330 | 0.175 | Strong qAP signal, but hurts localization. |
| Always-on quality, weight 0.5 | 0.362 | 0.266 | 0.335 | 0.105 | Keeps qAP, still hurts IoU and class-aware quality. |
| Always-on quality, weight 0.25 | 0.347 | 0.232 | 0.278 | 0.113 | Too weak/unstable. |
| Late start 151 + warmup 50 | 0.357 | 0.274 | 0.305 | 0.129 | Less disruptive, but still loses final IoU. |
| Two-stage quality-only after step 300 | 0.376 | 0.266 | 0.326 | 0.167 | Preserves detector, improves ranking. |

Two-stage setting:

```text
--steps 400
--quality-head-start-step 301
--quality-head-only-after-start
```

Output artifact:

- `results/det_real_quality_head_twostage_start301_400step_2seed.csv`

Two-stage interpretation:

- Freezing the detector after 300 steps and training only `head.quality_head` removes the main interference observed in always-on quality training.
- Residual-anchor box localization is preserved: final/best IoU remains `0.376/0.376`.
- Quality-aware AP improves from base AP `0.266` to `0.326`; quality-aware class AP improves from `0.135` to `0.167`.
- This supports a narrower and cleaner claim: query quality is useful as a post-detector ranking head, but coupling it into detector training from the start can hurt localization and semantic learning.

Updated scoring conclusion:

> The most reliable scoring path is now two-stage query quality: train the detector normally, freeze the detector, then train an independent query-quality head for score re-ranking. This preserves the residual-anchor detector while improving quality-aware AP. Always-on quality supervision remains useful diagnostically, but should not be the main training recipe.

Oracle scoring and correlation diagnostics:

The next diagnostic asks whether AP is limited by boxes or by ranking. It evaluates an oracle score multiplier using each query's true max IoU to any GT box:

```text
score = class_prob * true_max_iou
```

It also reports correlation with matched IoU for three scores:

- `class_score`: max foreground class probability;
- `quality_score`: predicted query quality;
- `combined_score`: `class_score * quality_score`.

Output artifact:

- `results/det_real_quality_head_twostage_oracle_start301_400step_2seed.csv`

| Variant | AP50 | AP50 q^0.5 | AP50 q^1 | AP50 q^2 | Oracle-IoU AP50 | Class AP50 | Class AP50 q^1 | Class Oracle AP50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `local_anchor_residual_query_quality_head` | 0.266 | 0.329 | 0.326 | 0.341 | 0.421 | 0.135 | 0.167 | 0.227 |
| `local_learned_quality_head` | 0.310 | 0.315 | 0.310 | 0.317 | 0.363 | 0.108 | 0.109 | 0.112 |

| Variant | Score-IoU corr | Quality-IoU corr | Combined-IoU corr | Objectness AUC | Quality AUC | Combined AUC |
|---|---:|---:|---:|---:|---:|---:|
| `local_anchor_residual_query_quality_head` | 0.339 | 0.611 | 0.584 | 0.602 | 0.686 | 0.683 |
| `local_learned_quality_head` | 0.438 | 0.567 | 0.557 | 0.658 | 0.680 | 0.683 |

Diagnostic interpretation:

- For residual-anchor, oracle-IoU scoring raises AP50 from `0.266` to `0.421`. This confirms that ranking/scoring is a major bottleneck: the boxes contain more usable AP than class scores expose.
- The two-stage quality head closes part of the ranking gap. Its best tested alpha is `q^2` for class-agnostic AP (`0.341`), while `q^0.5` and `q^2` are stronger than `q^1` for class-aware AP in this run.
- Quality score is much more correlated with IoU than class score for residual-anchor (`0.611` vs `0.339`). The combined score remains much better aligned than class score alone (`0.584` vs `0.339`).
- Predicted quality is not yet oracle-level. The remaining gap from `0.341` to `0.421` suggests improving quality calibration is useful, but it should remain a low-interference ranking head rather than an always-on detector loss.

Quality alpha sweep and oracle-gap accounting:

The previous run only exposed `q^0.5`, `q^1`, and `q^2`. The follow-up expands the eval-only alpha sweep to:

```text
q^0.25, q^0.5, q^1, q^2, q^4
```

It also records:

- post-hoc best quality alpha and best quality-aware AP;
- IoU-reference AP gap;
- remaining gap to the IoU-reference score after best quality scoring;
- raw fraction of the IoU-reference gap closed by quality scoring.

Output artifact:

- `results/det_real_quality_head_twostage_alphas_oracle_start301_400step_2seed.csv`

| Variant | AP50 | Post-hoc best-q AP50 | Mean selected alpha | IoU-ref AP50 | IoU-ref gap | Raw gap closed | Class AP50 | Class best-q AP50 | Class IoU-ref AP50 | Class raw gap closed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `local_anchor_residual_query_quality_head` | 0.266 | 0.345 | 1.25 | 0.421 | 0.155 | 0.491 | 0.135 | 0.189 | 0.227 | 0.584 |
| `local_learned_quality_head` | 0.310 | 0.328 | 2.12 | 0.363 | 0.053 | 0.240 | 0.108 | 0.120 | 0.112 | 0.345 |

Diagnostic interpretation:

- Residual-anchor still has the larger ranking headroom: IoU-reference scoring can raise AP50 from `0.266` to `0.421`.
- The two-stage quality head closes about half of that residual-anchor ranking gap while preserving the detector boxes.
- Learned queries have higher base AP50 (`0.310`) but less oracle headroom and lower quality-gap closure, so quality reranking is less useful there.
- The post-hoc best-q value is a diagnostic selected on the eval set. Formal model comparison should still report fixed alphas such as `q^1` or a pre-registered alpha.
- The IoU-reference score is only a diagnostic, not a strict detector upper bound: multiplying by true IoU can still leave class-score and duplicate-query errors. Treat class-aware closure as a rough calibration signal, with class-agnostic AP50 closure as the cleaner ranking-gap measure.
- This strengthens the current scoring recipe: keep quality prediction as a frozen-detector post-ranking head, then improve its calibration rather than pushing quality loss back into detector training.

Longer frozen quality-head control:

The next check asks whether the frozen-detector quality head was simply under-trained. The schedule keeps the detector training unchanged for the first 300 steps, then trains only `head.quality_head` for 300 more steps.

Output artifact:

- `results/det_real_quality_head_twostage_alphas_oracle_start301_600step_2seed.csv`

| Variant | AP50 | Post-hoc best-q AP50 | Mean selected alpha | IoU-ref AP50 | Raw gap closed |
|---|---:|---:|---:|---:|---:|
| `local_anchor_residual_query_quality_head` | 0.266 | 0.344 | 3.00 | 0.421 | 0.492 |
| `local_learned_quality_head` | 0.310 | 0.328 | 1.12 | 0.363 | 0.240 |

Interpretation:

- Longer quality-only training does not materially improve residual-anchor ranking over the 400-step run: best-q AP50 is effectively unchanged (`0.345 -> 0.344`), and raw gap closure is unchanged (`0.491 -> 0.492`).
- The frozen quality head is therefore likely capacity/target-limited rather than simply under-trained under this mini setting.
- The next scoring step should not just lengthen quality-only training. It should either improve calibration/target design or transfer the two-stage ranking head to the stronger proposal-query variants.

Proposal-query two-stage quality-head control:

This control applies the same frozen-detector quality ranking recipe to predicted and oracle proposal-query variants.

Output artifact:

- `results/det_real_proposal_quality_head_twostage_start301_400step_2seed.csv`

| Variant | Final IoU | Best IoU | AP50 | Best-q AP50 | IoU-ref AP50 | Class AP50 |
|---|---:|---:|---:|---:|---:|---:|
| `local_mask_proposal_nms_query` | 0.348 | 0.366 | 0.248 | 0.248 | 0.329 | 0.090 |
| `local_mask_proposal_nms_query_quality_head` | 0.357 | 0.366 | 0.266 | 0.291 | 0.364 | 0.093 |
| `local_mask_proposal_oracle_nms_query` | 0.383 | 0.421 | 0.250 | 0.250 | 0.345 | 0.108 |
| `local_mask_proposal_oracle_nms_query_quality_head` | 0.399 | 0.421 | 0.259 | 0.325 | 0.372 | 0.138 |

Interpretation:

- Two-stage quality ranking transfers to proposal-query variants.
- For predicted proposal NMS, the quality-head variant modestly improves base AP50 (`0.248 -> 0.266`) and lifts post-hoc best-q AP50 to `0.291`.
- For oracle proposal NMS, quality ranking is stronger: best-q AP50 reaches `0.325`, and class-aware AP50 improves from `0.108` to `0.138`.
- This strengthens the proposal-query + quality-ranking direction. The remaining issue is calibration: post-hoc best-q is useful for diagnosis, but fixed-alpha or calibrated scoring is needed before this becomes a clean inference recipe.

Fixed-alpha and temperature-calibrated scoring:

The next change separates fixed scoring from post-hoc best-q diagnostics. `train_det_real.py` now accepts:

```text
--fixed-quality-alpha
--quality-score-temperature
```

and writes fixed-score AP, alpha, temperature, fixed gap-to-IoU-reference, and fixed raw closure fields to the CSV.

Temperature-calibrated control:

- `results/det_real_proposal_quality_head_fixed_alpha2_temp2_400step_2seed.csv`
- fixed `alpha=2.0`
- temperature `2.0`

| Variant | AP50 | Fixed AP50 | Post-hoc best-q AP50 | IoU-ref AP50 | Fixed raw gap closed |
|---|---:|---:|---:|---:|---:|
| `local_mask_proposal_nms_query_quality_head` | 0.266 | 0.283 | 0.291 | 0.364 | 0.210 |
| `local_mask_proposal_oracle_nms_query_quality_head` | 0.259 | 0.312 | 0.325 | 0.372 | 0.485 |

Interpretation:

- Fixed-alpha scoring is now a first-class metric and should be used for formal comparisons instead of post-hoc best-q.
- Temperature `2.0` does not improve the oracle-proposal path over the untemperatured `q^2` score from the same run (`0.312` fixed-temp vs `0.325` q^2/best-q).
- For predicted proposals, temp-scaled fixed scoring is useful but still trails the best fixed alpha in the sweep.
- Current practical recommendation: report fixed `q^2` with temperature `1.0` as the pre-registered quality score unless a separate calibration split justifies changing alpha/temperature.

Official fixed-q2 real-mini summary:

This table uses fixed `q^2` with temperature `1.0` as the formal quality score. Post-hoc best-q remains diagnostic only.

Output artifact:

- `results/det_real_official_fixed_q2_summary.csv`

| Variant | Final IoU | Best IoU | Base AP50 | Class AP50 | Fixed-q2 AP50 | Fixed-q2 class AP50 | IoU-ref AP50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `local_learned` | 0.359 | 0.367 | 0.310 | 0.108 | - | - | - |
| `local_anchor_residual_query` | 0.376 | 0.376 | 0.266 | 0.135 | - | - | - |
| `local_anchor_residual_query_quality_head` | 0.376 | 0.376 | 0.266 | 0.135 | 0.341 | 0.185 | 0.421 |
| `local_mask_proposal_nms_query_quality_head` | 0.357 | 0.366 | 0.266 | 0.093 | 0.282 | 0.108 | 0.364 |
| `local_mask_proposal_oracle_nms_query_quality_head` | 0.399 | 0.421 | 0.259 | 0.138 | 0.325 | 0.173 | 0.372 |

Interpretation:

- Under the official fixed score, `local_anchor_residual_query_quality_head` remains the strongest AP route: fixed-q2 AP50 `0.341`.
- Oracle proposal quality has the strongest localization coverage, but fixed-q2 AP50 remains below residual-anchor quality (`0.325` vs `0.341`).
- The current real-mini recipe is therefore: residual-anchor detector + two-stage fixed-q2 quality ranking for AP, while proposal/oracle-proposal remains the coverage-improvement path that still needs better query-box refinement and scoring.

Query-conditioned mask auxiliary:

This control replaces the previous side-only union-mask auxiliary with one dense mask per object query. For each batch item, the Hungarian matcher selects query-target pairs; each matched query is supervised against its matched bbox rectangle mask with BCE + Dice. Unmatched queries are not mask-supervised. This tests whether dense mask supervision helps more when it is tied to query identity.

Output artifacts:

- `results/det_toy_querymask_smoke.csv`
- `results/det_real_querymask_300step_2seed.csv`
- `results/det_real_querymask_quality_head_400step_2seed.csv`

Toy smoke:

| Variant | Final IoU | AP50-lite | Matched mask IoU | Matched mask Dice |
|---|---:|---:|---:|---:|
| `local_anchor_residual_query` | 0.103 | 0.006 | 0.000 | 0.000 |
| `local_anchor_residual_query_querymask` | 0.148 | 0.015 | 0.429 | 0.565 |

The toy result is only a wiring sanity check: query-specific masks are learnable and gradients flow through the matched-query path.

Real DET mini, 300 steps, 2 seeds:

| Variant | Final IoU | Best IoU | Recall50 | AP50 | Class AP50 | Matched mask IoU | Matched mask Dice |
|---|---:|---:|---:|---:|---:|---:|---:|
| `local_anchor_residual_query` | 0.376 | 0.376 | 0.371 | 0.266 | 0.135 | 0.000 | 0.000 |
| `local_anchor_residual_query_querymask` | 0.357 | 0.357 | 0.284 | 0.257 | 0.058 | 0.354 | 0.477 |

Real DET mini, querymask with two-stage quality:

| Variant | Final IoU | Best IoU | Base AP50 | Fixed-q2 AP50 | Best-q AP50 | Matched mask IoU | Matched mask Dice |
|---|---:|---:|---:|---:|---:|---:|---:|
| `local_anchor_residual_query_querymask` trained 400 steps | 0.364 | 0.367 | 0.288 | - | - | 0.341 | 0.451 |
| `local_anchor_residual_query_querymask_quality_head` | 0.357 | 0.357 | 0.257 | 0.286 | 0.290 | 0.354 | 0.477 |

Interpretation:

- Query-conditioned masks are learnable and are now coupled to Hungarian query identity, unlike the earlier union-mask side auxiliary.
- As an auxiliary loss alone, querymask does not beat the residual-anchor detector on real DET mini box/AP at 300 steps.
- Two-stage quality ranking gives a small scoring recovery for querymask, but not enough to make it competitive with the residual-anchor quality route.
- The next architectural step should use query masks for box/query refinement, for example differentiable mask moments or a mask-conditioned box-refinement MLP, instead of treating query masks as auxiliary supervision only.

Persistent proposal-state mini probe:

This is the first implementation of the `init-only vs layerwise re-inject vs persistent proposal state` control requested by the proposal-consumption route. It should be treated as a smoke test until repeated under the standard real-mini protocol.

Implemented variants:

- `local_mask_proposal_nms_query`: predicted proposal init-only baseline.
- `local_mask_proposal_nms_query_decode2`: depth-matched init-only control.
- `local_mask_proposal_nms_query_reinject`: proposal tokens are re-added before the second decoder read.
- `local_mask_proposal_nms_query_persistent`: proposal state is updated after each decoder read.
- Oracle counterparts use `local_mask_proposal_oracle_nms_*`.

New summary output:

```text
proposal_oracle_gap_mode,metric,pred_model,oracle_model,
pred_mean,oracle_mean,direct_gap_mean,closure_vs_init_oracle_gap_mean
```

The closure metric uses the init-only predicted-vs-oracle gap as denominator:

```text
(candidate_predicted_metric - init_only_predicted_metric)
/ (init_only_oracle_metric - init_only_predicted_metric)
```

If the init-only oracle metric is not higher than the init-only predicted metric, there is no positive oracle headroom and closure is reported as `0.0`.

Smoke artifact:

- `results/det_real_proposal_state_smoke.csv`

Smoke interpretation:

- The implementation runs end-to-end on real DET mini inputs and emits the proposal-gap summary.
- In the 40-step / 80-sample / 1-seed smoke, `persistent` improves final-IoU closure but hurts base AP50, while `reinject` improves AP50 but not final IoU.
- This supports the diagnostic value of the new control but is not yet evidence that persistent proposal state is the right final architecture.

Standard 300-step result:

Artifact:

- `results/det_real_proposal_state_300step_2seed.csv`

Protocol:

- top-10 classes
- 200 images
- max 3 objects
- 6 queries
- 300 training steps
- 2 seeds

| Variant | Final IoU | Best IoU | Recall50 | AP50 | Class AP50 | Mask IoU |
|---|---:|---:|---:|---:|---:|---:|
| `local_mask_proposal_nms_query` | 0.357 | 0.366 | 0.302 | 0.266 | 0.093 | 0.458 |
| `local_mask_proposal_nms_query_decode2` | 0.344 | 0.352 | 0.265 | 0.221 | 0.088 | 0.443 |
| `local_mask_proposal_nms_query_reinject` | 0.367 | 0.373 | 0.293 | 0.258 | 0.111 | 0.386 |
| `local_mask_proposal_nms_query_persistent` | 0.356 | 0.367 | 0.324 | 0.262 | 0.121 | 0.420 |
| `local_mask_proposal_oracle_nms_query` | 0.399 | 0.421 | 0.340 | 0.259 | 0.138 | 1.000 |
| `local_mask_proposal_oracle_nms_query_decode2` | 0.399 | 0.415 | 0.380 | 0.313 | 0.086 | 1.000 |
| `local_mask_proposal_oracle_nms_query_reinject` | 0.402 | 0.416 | 0.354 | 0.279 | 0.103 | 1.000 |
| `local_mask_proposal_oracle_nms_query_persistent` | 0.405 | 0.421 | 0.413 | 0.322 | 0.081 | 1.000 |

Proposal-gap diagnostic:

| Mode | Final-IoU closure | Best-IoU closure | AP50 closure | Class-AP50 closure |
|---|---:|---:|---:|---:|
| `decode2` | -0.148 | -0.217 | 0.000 | -0.047 |
| `reinject` | 0.608 | 0.140 | 0.000 | 0.318 |
| `persistent` | -0.335 | -0.014 | 0.000 | 0.648 |

Interpretation:

- `decode2` is a clean negative depth control: extra decoder depth alone does not help.
- `reinject` is the strongest predicted proposal-consumption geometry candidate in this matrix: final IoU `0.367`, best IoU `0.373`, and positive final-IoU closure.
- `persistent` is not yet supported as the main predicted-proposal architecture. It improves class AP50 over init-only, but it does not close the geometry oracle gap under the current state update rule.
- Oracle persistent has the strongest oracle AP50 and final IoU, so the consumer-side idea should not be discarded; the current bottleneck may be proposal quality, update strength, query identity stability, or ranking calibration.
- Next recommended micro-controls: `persistent_gated`, `late_persistent`, `persistent_stopgrad`, and frozen quality scoring over the completed detector checkpoints.

Persistent-stability controls:

These controls ask whether persistent can be rescued without adding a larger backbone, teacher, dense positives, or a new training recipe.

Artifact:

- `results/det_real_persistent_stability_300step_2seed.csv`

Protocol:

- top-10 classes
- 200 images
- max 3 objects
- 6 queries
- 300 training steps
- 2 seeds

| Variant | Final IoU | Best IoU | Recall50 | AP50 | Class AP50 | Mask IoU |
|---|---:|---:|---:|---:|---:|---:|
| `local_mask_proposal_nms_query` | 0.357 | 0.366 | 0.302 | 0.266 | 0.093 | 0.458 |
| `local_mask_proposal_nms_query_late_persistent` | 0.339 | 0.359 | 0.312 | 0.252 | 0.089 | 0.463 |
| `local_mask_proposal_nms_query_persistent_gated` | 0.328 | 0.349 | 0.264 | 0.234 | 0.099 | 0.459 |
| `local_mask_proposal_nms_query_persistent_stopgrad` | 0.337 | 0.363 | 0.310 | 0.249 | 0.090 | 0.387 |
| `local_mask_proposal_oracle_nms_query` | 0.399 | 0.421 | 0.340 | 0.259 | 0.138 | 1.000 |
| `local_mask_proposal_oracle_nms_query_late_persistent` | 0.406 | 0.412 | 0.386 | 0.270 | 0.095 | 1.000 |
| `local_mask_proposal_oracle_nms_query_persistent_gated` | 0.383 | 0.421 | 0.346 | 0.213 | 0.072 | 1.000 |
| `local_mask_proposal_oracle_nms_query_persistent_stopgrad` | 0.405 | 0.412 | 0.380 | 0.300 | 0.118 | 1.000 |

Paired against the predicted init-only baseline:

| Variant | Final IoU delta | Final IoU wins | AP50 delta | AP50 wins | Class AP50 delta | Class AP50 wins |
|---|---:|---:|---:|---:|---:|---:|
| `late_persistent` | -0.019 | 1/2 | -0.014 | 0/2 | -0.004 | 1/2 |
| `persistent_gated` | -0.029 | 0/2 | -0.033 | 0/2 | +0.006 | 1/2 |
| `persistent_stopgrad` | -0.020 | 0/2 | -0.017 | 0/2 | -0.004 | 1/2 |
| `oracle_late_persistent` | +0.049 | 2/2 | +0.004 | 1/2 | +0.002 | 1/2 |
| `oracle_persistent_gated` | +0.026 | 2/2 | -0.053 | 0/2 | -0.022 | 0/2 |
| `oracle_persistent_stopgrad` | +0.047 | 2/2 | +0.034 | 2/2 | +0.025 | 2/2 |

Interpretation:

- The predicted-proposal persistent variants do not rescue persistent stability. Gating, late injection, and stop-gradient update all underperform the predicted init-only baseline on final IoU and AP50.
- `persistent_stopgrad` nearly recovers best IoU but still loses final IoU/AP, so it is not a sufficient rescue.
- Oracle `persistent_stopgrad` is the useful positive control: with idealized proposals, it improves final IoU, AP50, and class AP50 with `2/2` paired wins. The consumer idea still has capacity, but the current predicted-proposal state path is not stable enough.
- Current route decision: for predicted proposals, shrink the mainline to lighter layerwise proposal refresh / `reinject`. Keep persistent as an oracle/proposal-quality research branch rather than the active detector architecture.

Layerwise refresh plus two-stage quality:

After shrinking the active proposal path to `reinject`, the next control applies the low-interference frozen quality-head recipe to the refresh variants.

Artifact:

- `results/det_real_reinject_quality_head_400step_2seed.csv`

Protocol:

- top-10 classes
- 200 images
- max 3 objects
- 6 queries
- 400 training steps
- 2 seeds
- quality head starts at step 301 with the detector frozen
- official quality score: `class_prob * quality^2`, temperature `1.0`

| Variant | Final IoU | Best IoU | AP50 | Class AP50 | Fixed-q2 AP50 | Fixed-q2 Class AP50 | IoU-ref AP50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `local_anchor_residual_query_quality_head` | 0.376 | 0.376 | 0.266 | 0.135 | 0.341 | 0.185 | 0.421 |
| `local_mask_proposal_nms_query_quality_head` | 0.357 | 0.366 | 0.266 | 0.093 | 0.282 | 0.108 | 0.364 |
| `local_mask_proposal_nms_query_reinject_quality_head` | 0.367 | 0.373 | 0.258 | 0.111 | 0.284 | 0.124 | 0.390 |
| `local_mask_proposal_oracle_nms_query_quality_head` | 0.399 | 0.421 | 0.259 | 0.138 | 0.325 | 0.173 | 0.372 |
| `local_mask_proposal_oracle_nms_query_reinject_quality_head` | 0.402 | 0.416 | 0.279 | 0.103 | 0.313 | 0.116 | 0.402 |

Interpretation:

- Predicted `reinject` remains useful for geometry: it improves final/best IoU over init-only proposal quality (`0.367/0.373` vs `0.357/0.366`).
- Two-stage fixed-q2 scoring does not make predicted `reinject` the AP leader. Fixed AP50 is only `0.284`, essentially tied with init-only proposal quality (`0.282`) and well below residual-anchor quality (`0.341`).
- Predicted `reinject` improves fixed class-aware AP over init-only proposal quality (`0.124` vs `0.108`), so refresh has some class/region coupling value.
- Oracle `reinject` improves base AP and geometry over oracle init-only, but its fixed-q2 AP is lower than oracle init-only (`0.313` vs `0.325`). This points to quality calibration mismatch for refreshed proposal boxes.
- Current detector ranking: residual-anchor plus two-stage quality remains the strongest AP route; layerwise proposal refresh is now the active proposal-coverage path, not the active AP solution.
