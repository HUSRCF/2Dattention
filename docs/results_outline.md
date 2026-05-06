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
