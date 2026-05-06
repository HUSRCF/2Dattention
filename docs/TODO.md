# TODO: 2Dattention Research Roadmap

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
