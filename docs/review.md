# Process Review and Evidence Boundary

## Current Bottom Line

The project has pivoted from a memory-first hypothesis to a local-state-first baseline. The strongest current Pareto result is `no_prefill_local_mix` on the larger all-single-label DET-derived ImageFolder run: final `0.250`, best `0.250`, about `470 img/s`. `anchor_only_no_prefill` is the highest-accuracy current variant by a very small margin: final/best `0.251`, about `341 img/s`. The difference is too small to move the main claim away from local-state-first, but it makes anchor-only interaction the best plugin candidate.

Current evidence does not support claiming that early spatial prefill, full prefill-lattice memory, or learned memory read is effective. The most defensible claim is narrower: lightweight online 2D local refinement is currently the strongest baseline; anchor/region memory remains an optional add-on to test against that baseline.

This should not be overread as evidence that visual memory is useless. A more precise diagnosis is that memory built from very early features is likely too shallow, stale, or distribution-mismatched for later queries. The next memory hypothesis is therefore delayed and stage-wise: first refine the 2D state locally, then build or refresh region memory from denser features, and only then use gated content-adaptive reads.

The boundary with FPN/ViTDet-style designs must stay explicit. If a future model only performs fixed local refinement plus fixed multi-scale fusion, it is an FPN-like or local-ViT-like baseline. The distinct research claim requires state persistence, delayed or refreshed region memory, and content-adaptive read/write paths whose contribution can be measured.

## What Has Been Implemented

- A clean torch scaffold with `src/`, `scripts/`, `docs/`, and `tests/`.
- Proposed model variants:
  - `prefill_lattice_attnres`: 2D prefill plus lattice offset memory reads.
  - `anchor_prefill_attnres`: adds row, column, and global anchor memory.
  - `graph_prefill_attnres`: adds semantic top-k graph memory.
- Baselines:
  - `conv_only`
  - `seq_transformer`
  - `tiny_vit`
  - `xattnres_style`
- Synthetic tasks:
  - `oriented_pair`
  - `aligned_pair`
  - `distractor_aligned_pair`
- ILSVRC2013 DET validation helpers:
  - detection bbox manifest from XML annotations
  - symlinked single-label `ImageFolder` subset for classification smoke tests
  - multi-model ImageFolder comparison script with ViT baseline

## What The Current Evidence Supports

- The code paths run and pass shape/backward tests.
- Routing tensors are normalized and gradients flow.
- On the easiest toy task, the proposed model can learn a simple oriented 2D relation.
- On harder toy tasks, results are mixed or near chance.
- On the small ILSVRC2013 DET-derived ImageFolder subset, the training/eval pipeline runs across Conv, tiny ViT, XAttnRes-style, and proposed variants.

## What The Current Evidence Does Not Support

- It does not support a claim that the method is better than ViT, XAttnRes, ConvNets, or graph models in general.
- It does not support a claim about real image recognition, segmentation, or dense prediction.
- It does not prove that 2D prefill, anchor memory, or semantic graph memory is necessary.
- It does not prove robustness, scale behavior, or data efficiency.

## Why The Earlier Toy Evidence Is Insufficient

- `oriented_pair` is too simple; a small conv baseline can solve it.
- `aligned_pair` is more useful, but current results are close between conv and proposed variants.
- `distractor_aligned_pair` is a better stress test, but early smoke runs are near chance for all models.
- Synthetic marker tasks are controlled probes, not substitutes for natural images.

## Required Next Evidence

- Use `scripts/run_imagefolder_smoke.py` with a local real-image `ImageFolder` dataset.
- Use `scripts/run_unlabeled_image_smoke.py` for forward-only checks on unlabeled real images.
- Treat `data/ILSVRC2013_DET_val_supervised/single_label_imagefolder` as a course/smoke subset, not the standard ImageNet-1K classification benchmark.
- Run multi-seed comparisons with matched parameter counts and training budgets.
- Increase image size and training budget before comparing architecture quality; the current 100-step CPU smoke has Conv > Prefill > tiny ViT but is not a serious benchmark.

## Current MPS ImageFolder Smoke Results

Dataset: `data/ILSVRC2013_DET_val_supervised/single_label_imagefolder`

Setting: 64px images, batch size 16, embed dim 32, 300 steps.

Core 3-model run, 3 seeds:

| Model | Eval accuracy mean |
|---|---:|
| `prefill_lattice_attnres` | 0.130 |
| `conv_only` | 0.112 |
| `tiny_vit` | 0.112 |

Full 6-model run, 2 seeds:

| Model | Eval accuracy mean |
|---|---:|
| `anchor_prefill_attnres` | 0.157 |
| `xattnres_style` | 0.131 |
| `prefill_lattice_attnres` | 0.108 |
| `tiny_vit` | 0.108 |
| `graph_prefill_attnres` | 0.107 |
| `conv_only` | 0.102 |

Interpretation: this is a useful smoke signal that anchor memory may help on this small subset. It is not enough for a claim because the dataset is small, the split is synthetic, and the training budget is short.

## Current Engineering Adjustment

All memory-read blocks now use a learnable residual gate:

```text
state = state + gamma * MemoryRead(...)
```

with `gamma` initialized to `1e-3`. This follows the current interpretation that random early memory reads can destabilize lattice/graph variants. The gate makes the model start close to the local prefill backbone and learn memory usage gradually.

## Gated 5-Seed Core Result

Dataset: `data/ILSVRC2013_DET_val_supervised/single_label_imagefolder`

Setting: MPS, 64px images, batch size 16, embed dim 32, 300 steps, 5 seeds, eval every 100 steps.

| Model | Params | Final eval acc mean | Best eval acc mean | Images/sec mean |
|---|---:|---:|---:|---:|
| `anchor_prefill_attnres` | 29,046 | 0.133 | 0.134 | 128.20 |
| `xattnres_style` | 22,422 | 0.123 | 0.126 | 354.68 |
| `tiny_vit` | 27,700 | 0.118 | 0.118 | 353.92 |
| `prefill_lattice_attnres` | 22,576 | 0.104 | 0.109 | 151.80 |
| `conv_only` | 30,388 | 0.096 | 0.096 | 456.47 |

Interpretation: after gating, `anchor_prefill_attnres` remains the strongest core signal across 5 seeds, but it is much slower than Conv/ViT/XAttnRes-style. `prefill_lattice_attnres` remains unstable. This supports prioritizing anchor/region memory analysis next, not graph expansion.

## Paired Anchor vs XAttnRes Mechanism Check

Setting: MPS, 64px images, batch size 16, embed dim 32, 300 steps, 5 shared split seeds, eval every 100 steps.

| Model | Final eval acc mean | Best eval acc mean | Images/sec mean | Gate mean | Scaled read ratio mean |
|---|---:|---:|---:|---:|---:|
| `anchor_prefill_attnres` | 0.123 | 0.123 | 121.74 | 0.00526 | 0.00919 |
| `xattnres_style` | 0.125 | 0.126 | 291.89 | 0.00305 | 0.01168 |

Paired against `xattnres_style`:

| Model | Final delta mean | Final wins | Best delta mean | Best wins |
|---|---:|---:|---:|---:|
| `anchor_prefill_attnres` | -0.002 | 3/5 | -0.002 | 3/5 |

Interpretation: anchor memory is not yet a stable win over XAttnRes-style. It wins in 3 of 5 split seeds, but mean delta is slightly negative and throughput is about 2.4x lower. The mechanism statistics also do not prove that anchor uses memory more strongly than XAttnRes; its gate is larger, but scaled read ratio is lower on average. Anchor remains a promising branch, not a confirmed mainline advantage.

## Anchor Ablation Check

Setting: MPS, 64px images, batch size 16, embed dim 32, 300 steps, 3 split seeds, eval every 100 steps.

Output CSV: `results/imagefolder_anchor_ablation_300step_3seeds.csv`

| Model | Params | Final eval acc mean | Eval acc std | Best eval acc mean | Images/sec mean | Scaled read ratio mean |
|---|---:|---:|---:|---:|---:|---:|
| `anchor_no_prefill` | 19,894 | 0.142 | 0.026 | 0.142 | 185.21 | 0.00783 |
| `anchor_prefill_attnres` | 29,046 | 0.139 | 0.004 | 0.139 | 116.76 | 0.00846 |
| `anchor_fixed_gamma_0` | 29,044 | 0.133 | 0.022 | 0.137 | 113.84 | 0.00000 |
| `xattnres_style` | 22,422 | 0.127 | 0.014 | 0.127 | 285.41 | 0.00933 |
| `tiny_vit` | 27,700 | 0.123 | 0.026 | 0.131 | 356.08 | n/a |
| `xattnres_equal_params` | 27,814 | 0.113 | 0.009 | 0.135 | 287.29 | 0.01313 |
| `multi_cls_vit` | 27,892 | 0.084 | 0.041 | 0.095 | 355.49 | n/a |

Paired against `xattnres_style`:

| Model | Final delta mean | Final delta std | Final wins | Best delta mean | Best delta std | Best wins |
|---|---:|---:|---:|---:|---:|---:|
| `anchor_no_prefill` | 0.015 | 0.040 | 2/3 | 0.015 | 0.040 | 2/3 |
| `anchor_prefill_attnres` | 0.012 | 0.013 | 2/3 | 0.012 | 0.013 | 2/3 |
| `anchor_fixed_gamma_0` | 0.006 | 0.036 | 1/3 | 0.011 | 0.032 | 1/3 |
| `tiny_vit` | -0.004 | 0.037 | 2/3 | 0.004 | 0.026 | 2/3 |
| `xattnres_equal_params` | -0.013 | 0.019 | 1/3 | 0.008 | 0.010 | 2/3 |
| `multi_cls_vit` | -0.042 | 0.044 | 0/3 | -0.032 | 0.030 | 0/3 |

Interpretation: this ablation weakens the claim that the full anchor-prefill path is the reason for the gain. `multi_cls_vit` is clearly weak in this setting, so the result is not explained by simply adding several global Transformer tokens. However, `anchor_no_prefill` slightly beats full `anchor_prefill_attnres`, and `anchor_fixed_gamma_0` remains competitive despite disabling memory-read contribution. This means the current evidence points to an anchor/region-style architecture being useful, but it does not yet prove that spatial prefill or learned memory read is the causal mechanism. The next experiment should isolate the local mix backbone, anchor read cost, and gamma schedule before making architectural claims.

Implementation follow-up: `prefill_local_mix` has been added as the local-backbone control for this question. A 30-step MPS smoke confirms it runs and writes results to `results/imagefolder_local_mix_smoke.csv`; it is not long enough to interpret as evidence.

## P0 Source Ablation: Prefill vs Anchor vs Lattice

Setting: MPS, 64px images, batch size 16, embed dim 32, 300 steps, 5 split seeds, eval every 100 steps.

Output CSV: `results/imagefolder_anchor_source_p0_300step_5seeds.csv`

Important caveat: this run was produced before the script used stable model-name seed offsets and shared train-loader shuffle order. Keep it as historical context only; use the strict paired result below for current decisions.

| Model | Params | Final eval acc mean | Eval acc std | Best eval acc mean | Images/sec mean | Scaled read ratio mean |
|---|---:|---:|---:|---:|---:|---:|
| `anchor_no_prefill` | 19,894 | 0.135 | 0.024 | 0.137 | 184.23 | 0.01173 |
| `prefill_local_mix` | 20,244 | 0.126 | 0.019 | 0.126 | 419.03 | n/a |
| `anchor_prefill_attnres` | 29,046 | 0.123 | 0.016 | 0.123 | 113.07 | 0.00919 |
| `lattice_only_no_anchor` | 22,576 | 0.115 | 0.019 | 0.116 | 133.54 | 0.00507 |
| `anchor_read_only_no_lattice` | 22,556 | 0.101 | 0.026 | 0.106 | 226.60 | 0.00599 |

Paired against `anchor_prefill_attnres`:

| Model | Final delta mean | Final delta std | Final wins | Best delta mean | Best delta std | Best wins |
|---|---:|---:|---:|---:|---:|---:|
| `anchor_no_prefill` | 0.012 | 0.017 | 3/5 | 0.014 | 0.017 | 3/5 |
| `prefill_local_mix` | 0.002 | 0.025 | 2/5 | 0.002 | 0.025 | 2/5 |
| `lattice_only_no_anchor` | -0.008 | 0.026 | 1/5 | -0.008 | 0.026 | 2/5 |
| `anchor_read_only_no_lattice` | -0.023 | 0.024 | 1/5 | -0.018 | 0.020 | 1/5 |

Interpretation: this older source-ablation run suggested that `anchor_no_prefill` may be stronger than full prefill, but it should not be used as the decisive table because the experiment was not yet strictly paired.

## No-Prefill Local-Mix Control

Implementation update: `no_prefill_local_mix` has been added to the ImageFolder comparison script. It uses the same local refinement blocks as `prefill_local_mix`, but replaces `SpatialPrefill2D` with a one-state identity memory so it tests local refinement without early prefilled memory.

An order-dependent 5-seed run was completed before the seeding scheme was fixed:

Output CSV: `results/imagefolder_no_prefill_local_mix_300step_5seeds.csv`

| Model | Params | Final eval acc mean | Eval acc std | Best eval acc mean | Images/sec mean |
|---|---:|---:|---:|---:|---:|
| `anchor_prefill_attnres` | 29,046 | 0.133 | 0.011 | 0.134 | 112.91 |
| `anchor_no_prefill` | 19,894 | 0.131 | 0.016 | 0.131 | 179.04 |
| `no_prefill_local_mix` | 11,092 | 0.130 | 0.022 | 0.130 | 438.46 |
| `prefill_local_mix` | 20,244 | 0.126 | 0.019 | 0.126 | 411.18 |
| `xattnres_style` | 22,422 | 0.125 | 0.017 | 0.126 | 289.34 |

Important caveat: during this run, model initialization seed still depended on the position of each model in the `--models` list. This has now been fixed in `scripts/compare_imagefolder_models.py` by assigning stable seed offsets per model name. Therefore, this table is useful as a diagnostic signal, but it should not be treated as the final source-ablation result. It suggests `no_prefill_local_mix` is a strong control and may explain much of the anchor signal, but the stable-seed rerun is still required.

Superseded required rerun:

```bash
python -u scripts/compare_imagefolder_models.py \
  --data-root data/ILSVRC2013_DET_val_supervised/single_label_imagefolder \
  --models xattnres_style anchor_no_prefill prefill_local_mix no_prefill_local_mix anchor_prefill_attnres \
  --reference-model xattnres_style \
  --steps 300 \
  --batch-size 16 \
  --image-size 64 \
  --embed-dim 32 \
  --seeds 5 \
  --eval-every 100 \
  --out results/imagefolder_no_prefill_local_mix_300step_5seeds_stable_seeds.csv
```

## Strict Paired No-Prefill / Local-Mix Control

Setting: MPS, 64px images, batch size 16, embed dim 32, 300 steps, 5 split seeds, eval every 100 steps. This run uses stable model-name seed offsets and rebuilds each train loader with the same split-specific shuffle generator, so models are paired on split and training minibatch order.

Output CSV: `results/imagefolder_no_prefill_local_mix_300step_5seeds_strict_paired.csv`

| Model | Params | Final eval acc mean | Eval acc std | Best eval acc mean | Images/sec mean | Scaled read ratio mean |
|---|---:|---:|---:|---:|---:|---:|
| `anchor_no_prefill` | 19,894 | 0.117 | 0.017 | 0.119 | 175.77 | 0.03249 |
| `prefill_local_mix` | 20,244 | 0.113 | 0.026 | 0.113 | 408.78 | n/a |
| `anchor_prefill_attnres` | 29,046 | 0.110 | 0.021 | 0.113 | 100.90 | 0.01942 |
| `xattnres_style` | 22,422 | 0.110 | 0.021 | 0.120 | 288.19 | 0.00767 |
| `no_prefill_local_mix` | 11,092 | 0.102 | 0.019 | 0.111 | 441.47 | n/a |

Paired against `xattnres_style`:

| Model | Final delta mean | Final delta std | Final wins | Best delta mean | Best delta std | Best wins |
|---|---:|---:|---:|---:|---:|---:|
| `anchor_no_prefill` | 0.007 | 0.013 | 3/5 | -0.001 | 0.009 | 2/5 |
| `anchor_prefill_attnres` | 0.000 | 0.019 | 2/5 | -0.007 | 0.009 | 1/5 |
| `no_prefill_local_mix` | -0.007 | 0.010 | 1/5 | -0.008 | 0.010 | 0/5 |
| `prefill_local_mix` | 0.004 | 0.034 | 2/5 | -0.006 | 0.023 | 1/5 |

Interpretation: this strict paired run further weakens the full prefill-memory claim. `anchor_no_prefill` is the best final-accuracy mean in this set, but its advantage over `xattnres_style` is small, noisy, and disappears on best eval accuracy. Full `anchor_prefill_attnres` is essentially tied with `xattnres_style` on final accuracy and below it on best accuracy while being much slower. `prefill_local_mix` is close to the anchor variants and much faster, so local refinement explains a substantial part of the signal. The current evidence supports only a cautious claim: no-prefill anchor/local structures are worth studying, but spatial prefill and learned memory read are not established as the causal mechanism in this setting.

## Larger Single-Label DET Run

Dataset: `data/ILSVRC2013_DET_val_supervised/single_label_imagefolder_all`

This ImageFolder uses all single-label ILSVRC2013 DET validation images available from the local annotations: 197 classes and 11,433 symlinked images.

Setting: MPS, 64px images, batch size 32, embed dim 32, 1000 steps, 3 split seeds, eval every 250 steps. This uses stable model-name seed offsets and shared train-loader shuffle order per split.

Output CSV: `results/imagefolder_all_197cls_1000step_3seeds_strict_paired.csv`

| Model | Params | Final eval acc mean | Eval acc std | Best eval acc mean | Images/sec mean | Scaled read ratio mean |
|---|---:|---:|---:|---:|---:|---:|
| `no_prefill_local_mix` | 16,933 | 0.249 | 0.004 | 0.250 | 442.28 | n/a |
| `anchor_no_prefill` | 25,735 | 0.246 | 0.006 | 0.246 | 182.74 | 0.01329 |
| `tiny_vit` | 33,541 | 0.243 | 0.003 | 0.243 | 315.76 | n/a |
| `xattnres_style` | 28,263 | 0.242 | 0.006 | 0.242 | 347.26 | 0.02023 |
| `prefill_local_mix` | 26,085 | 0.239 | 0.002 | 0.239 | 443.25 | n/a |

Paired against `xattnres_style`:

| Model | Final delta mean | Final delta std | Final wins | Best delta mean | Best delta std | Best wins |
|---|---:|---:|---:|---:|---:|---:|
| `no_prefill_local_mix` | 0.008 | 0.006 | 2/3 | 0.009 | 0.005 | 3/3 |
| `anchor_no_prefill` | 0.004 | 0.004 | 2/3 | 0.004 | 0.004 | 2/3 |
| `tiny_vit` | 0.001 | 0.004 | 2/3 | 0.001 | 0.004 | 2/3 |
| `prefill_local_mix` | -0.002 | 0.004 | 1/3 | -0.002 | 0.004 | 1/3 |

Interpretation: the larger-data result shifts the current center of gravity again. `no_prefill_local_mix` is the best model in this setting and is also the fastest among the tested non-trivial models. `anchor_no_prefill` retains a small accuracy signal over `xattnres_style`, but it is much slower and is slightly below the no-prefill local-mix control. This weakens the argument that anchor interaction is currently the main causal factor. The strongest current conclusion is that removing early prefill and using a lightweight local refinement backbone is the most reliable direction under this DET-derived classification setup. Future anchor work should be treated as an optional add-on to this stronger local/no-prefill baseline, not the main claim.

## P0 No-Prefill Control Matrix

Dataset: `data/ILSVRC2013_DET_val_supervised/single_label_imagefolder_all`

Setting: MPS, 64px images, batch size 32, embed dim 32, 1000 steps, 3 split seeds, eval every 250 steps. This run uses stable model-name seed offsets, shared train-loader shuffle order per split, and two paired references.

Output CSV: `results/imagefolder_all_197cls_p0_controls_1000step_3seeds_dual_ref.csv`

| Model | Params | Final eval acc mean | Eval acc std | Best eval acc mean | Images/sec mean | Scaled read ratio mean |
|---|---:|---:|---:|---:|---:|---:|
| `anchor_only_no_prefill` | 19,245 | 0.251 | 0.005 | 0.251 | 341.33 | 0.01944 |
| `no_prefill_local_mix` | 16,933 | 0.250 | 0.006 | 0.250 | 470.29 | n/a |
| `xattnres_no_prefill` | 19,111 | 0.249 | 0.010 | 0.249 | 374.00 | 0.01801 |
| `anchor_no_prefill` | 25,735 | 0.246 | 0.006 | 0.248 | 178.77 | 0.00871 |
| `tiny_vit` | 33,541 | 0.243 | 0.003 | 0.243 | 363.53 | n/a |
| `xattnres_style` | 28,263 | 0.242 | 0.006 | 0.242 | 353.23 | 0.02023 |

Paired against `no_prefill_local_mix`:

| Model | Final delta mean | Final delta std | Final wins | Best delta mean | Best delta std | Best wins |
|---|---:|---:|---:|---:|---:|---:|
| `anchor_only_no_prefill` | 0.002 | 0.001 | 2/3 | 0.002 | 0.001 | 2/3 |
| `xattnres_no_prefill` | -0.001 | 0.006 | 1/3 | -0.001 | 0.005 | 1/3 |
| `anchor_no_prefill` | -0.003 | 0.000 | 0/3 | -0.002 | 0.002 | 0/3 |
| `tiny_vit` | -0.007 | 0.003 | 0/3 | -0.007 | 0.003 | 0/3 |
| `xattnres_style` | -0.008 | 0.002 | 0/3 | -0.008 | 0.002 | 0/3 |

Paired against `xattnres_style`:

| Model | Final delta mean | Final delta std | Final wins | Best delta mean | Best delta std | Best wins |
|---|---:|---:|---:|---:|---:|---:|
| `anchor_only_no_prefill` | 0.010 | 0.003 | 3/3 | 0.010 | 0.003 | 3/3 |
| `no_prefill_local_mix` | 0.008 | 0.002 | 3/3 | 0.008 | 0.002 | 3/3 |
| `xattnres_no_prefill` | 0.007 | 0.008 | 3/3 | 0.008 | 0.007 | 3/3 |
| `anchor_no_prefill` | 0.005 | 0.002 | 3/3 | 0.006 | 0.002 | 3/3 |
| `tiny_vit` | 0.001 | 0.004 | 2/3 | 0.001 | 0.004 | 2/3 |

Interpretation: this control matrix clarifies the no-prefill route. `xattnres_no_prefill` substantially improves over `xattnres_style`, so the old XAttnRes-style result was likely harmed by the early prefill/history protocol. However, `xattnres_no_prefill` still does not beat `no_prefill_local_mix` on the paired mean. `anchor_only_no_prefill` is the highest-accuracy variant, but its advantage over `no_prefill_local_mix` is only about 0.002 while running about 27% slower. `anchor_no_prefill` is worse and much slower than `anchor_only_no_prefill`, so the prior anchor+lattice mixture should be downgraded. Current mainline remains local-state-first; anchor-only and no-prefill XAttnRes are the best plugin/reference branches to carry forward.

## Delayed Memory Hypothesis

The negative prefill results should be interpreted as a failure of the current early-cache protocol, not as a final rejection of memory. The likely failure mode is:

```text
early shallow feature -> fixed memory cache -> later semantic query reads stale/noisy memory
```

The next testable hypothesis is:

```text
local-state refinement first -> delayed/stage-wise memory build -> gated adaptive read
```

Minimum controls needed before reviving memory as a claim:

- `prefill_at_block_0`: current early prefill behavior.
- `delayed_xattnres_no_prefill`: local warmup before XAttnRes-style reads.
- `stage_refresh_region_slots_2x2`: local warmup plus refreshed fixed-grid slots.
- `fpn_sum_lite`: fixed sum fusion over block states.
- `fpn_concat_lite`: fixed concat fusion over block states.
- `region_pool_mixer_no_history`: current fixed-grid slots only, no cross-stage memory.

The key comparison is not against `tiny_vit` alone. Any delayed-memory variant must first beat or improve the Pareto tradeoff against `no_prefill_local_mix`; otherwise the result is only a more complex local/FPN-like variant.

Implementation status: these P0 controls have been added and passed a short 20-step MPS smoke on the 20-class ImageFolder subset. The smoke confirms the training paths and paired summaries work, but it is not evidence about architecture quality. The formal check should use the same 197-class, 1000-step, 3-seed strict paired protocol as the previous matrix.

## Delayed / FPN / Region Control Matrix

Dataset: `data/ILSVRC2013_DET_val_supervised/single_label_imagefolder_all`

Setting: MPS, 64px images, batch size 32, embed dim 32, 1000 steps, 3 split seeds, eval every 250 steps.

Output CSV: `results/imagefolder_all_197cls_delayed_fpn_region_1000step_3seeds.csv`

| Model | Params | Final eval acc mean | Eval acc std | Best eval acc mean | Images/sec mean | Scaled read ratio mean |
|---|---:|---:|---:|---:|---:|---:|
| `anchor_only_no_prefill` | 19,245 | 0.251 | 0.005 | 0.251 | 338.11 | 0.01944 |
| `no_prefill_local_mix` | 16,933 | 0.250 | 0.006 | 0.250 | 462.28 | n/a |
| `xattnres_no_prefill` | 19,111 | 0.249 | 0.010 | 0.249 | 377.28 | 0.01801 |
| `delayed_xattnres_no_prefill` | 23,431 | 0.248 | 0.007 | 0.248 | 353.77 | 0.02769 |
| `region_pool_mixer_no_history` | 25,511 | 0.246 | 0.008 | 0.247 | 370.83 | 0.01529 |
| `stage_refresh_region_slots_2x2` | 29,831 | 0.245 | 0.005 | 0.245 | 360.39 | 0.01862 |
| `fpn_sum_lite` | 20,101 | 0.244 | 0.004 | 0.244 | 447.40 | n/a |
| `fpn_concat_lite` | 20,037 | 0.241 | 0.004 | 0.241 | 442.64 | n/a |

Paired against `no_prefill_local_mix`:

| Model | Final delta mean | Final delta std | Final wins | Best delta mean | Best delta std | Best wins |
|---|---:|---:|---:|---:|---:|---:|
| `anchor_only_no_prefill` | 0.002 | 0.001 | 2/3 | 0.002 | 0.001 | 2/3 |
| `xattnres_no_prefill` | -0.001 | 0.006 | 1/3 | -0.001 | 0.005 | 1/3 |
| `delayed_xattnres_no_prefill` | -0.002 | 0.003 | 1/3 | -0.002 | 0.003 | 1/3 |
| `region_pool_mixer_no_history` | -0.004 | 0.003 | 0/3 | -0.003 | 0.003 | 0/3 |
| `stage_refresh_region_slots_2x2` | -0.004 | 0.002 | 0/3 | -0.004 | 0.002 | 0/3 |
| `fpn_sum_lite` | -0.006 | 0.002 | 0/3 | -0.006 | 0.002 | 0/3 |
| `fpn_concat_lite` | -0.008 | 0.007 | 0/3 | -0.008 | 0.007 | 0/3 |

Interpretation: this matrix strengthens the local-state-first conclusion. The fixed FPN-like controls underperform `no_prefill_local_mix`, so the current local-state baseline is not explained by simply fusing block states. The delayed XAttnRes variant increases memory-read contribution but does not improve accuracy over `xattnres_no_prefill` or `no_prefill_local_mix`. Fixed 2x2 region pooling and stage-refresh slots also do not beat local mix, so grid-region memory is not a useful next mainline in this classification setting. `anchor_only_no_prefill` remains the best plugin candidate, but its margin over `no_prefill_local_mix` is tiny and comes with a clear speed cost.

## General Evidence Hygiene

- Add at least one dense prediction or segmentation-style task.
- Report negative results directly, not only successful toy runs.
- Treat all current numbers as debugging evidence until tested on real images.

## Balanced BBox Quadrant Probe

Dataset: ILSVRC2013 DET validation images with the largest annotated object box per image.

Task: predict the normalized bbox center quadrant (`quadrant4`). The first unbalanced version was inconclusive because all models stayed close to the majority-class baseline, about `0.335`. The balanced version uses class-balanced train sampling and a class-balanced eval subset, and reports raw accuracy, balanced accuracy, macro F1, per-class recall, and confusion matrices.

Output CSV: `results/bbox_probe_quadrant4_balanced_1000step_3seeds.csv`

Setting: MPS, 64px images, batch size 32, embed dim 32, 1000 steps, 3 split seeds, eval every 250 steps.

| Model | Balanced acc mean | Macro F1 mean | Best eval acc mean | Majority baseline | Images/sec mean |
|---|---:|---:|---:|---:|---:|
| `fpn_sum_lite` | 0.275 | 0.221 | 0.275 | 0.250 | 353.68 |
| `no_prefill_local_mix` | 0.269 | 0.200 | 0.270 | 0.250 | 361.41 |
| `stage_refresh_region_slots_2x2` | 0.267 | 0.207 | 0.267 | 0.250 | 298.15 |
| `anchor_only_no_prefill` | 0.266 | 0.188 | 0.268 | 0.250 | 285.70 |
| `xattnres_no_prefill` | 0.266 | 0.192 | 0.268 | 0.250 | 300.02 |
| `region_pool_mixer_no_history` | 0.261 | 0.185 | 0.262 | 0.250 | 292.19 |

Paired against `no_prefill_local_mix`, `fpn_sum_lite` improves balanced accuracy by about `+0.006` and wins `3/3` seeds. Region-pool and stage-refresh slot variants do not improve over local mix. This means balanced `quadrant4` does provide a weak spatial signal above random, but the signal favors fixed fusion rather than memory/region routing. The current result does not rescue memory-first or region-slot claims; it instead suggests that this coarse quadrant task may reward simple multi-state fusion and low-level layout bias.

Next spatial probes should be stronger than quadrant classification: balanced `grid9`, bbox center regression, object size bin, or a weak localization heatmap. Any positive region/memory claim should require improvement in balanced accuracy or center-error metrics over both `no_prefill_local_mix` and `fpn_sum_lite`.

## Balanced BBox Grid9 Probe

Task: predict the normalized bbox center cell in a 3x3 grid (`grid9`). The raw label distribution is extremely center-biased: center cell has `12054 / 18680 = 0.645` of all examples, so balanced train sampling and balanced eval are required. Balanced random baseline is `1/9 = 0.111`.

Output CSV: `results/bbox_probe_grid9_balanced_1000step_3seeds.csv`

Setting: MPS, 64px images, batch size 32, embed dim 32, 1000 steps, 3 split seeds, eval every 250 steps.

| Model | Balanced acc mean | Macro F1 mean | Best eval acc mean | Corner recall | Edge recall | Center recall | Images/sec mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `no_prefill_local_mix` | 0.136 | 0.093 | 0.141 | 0.158 | 0.122 | 0.099 | 523.47 |
| `xattnres_no_prefill` | 0.136 | 0.090 | 0.155 | 0.105 | 0.143 | 0.229 | 377.44 |
| `region_pool_mixer_no_history` | 0.129 | 0.085 | 0.155 | 0.142 | 0.136 | 0.050 | 407.58 |
| `anchor_only_no_prefill` | 0.126 | 0.091 | 0.140 | 0.144 | 0.115 | 0.100 | 333.71 |
| `stage_refresh_region_slots_2x2` | 0.124 | 0.081 | 0.142 | 0.130 | 0.103 | 0.180 | 433.52 |
| `fpn_sum_lite` | 0.116 | 0.063 | 0.133 | 0.140 | 0.099 | 0.083 | 550.93 |

Paired against `fpn_sum_lite`, `no_prefill_local_mix` and `xattnres_no_prefill` improve balanced accuracy by about `+0.020`, with `2/3` balanced wins and `3/3` macro-F1 wins. Paired against `no_prefill_local_mix`, no model clearly improves: `xattnres_no_prefill` ties balanced accuracy on average but wins only `1/3` seeds.

Interpretation: balanced `grid9` is harder than `quadrant4` and gives only a weak spatial signal. `fpn_sum_lite` is no longer strongest; `no_prefill_local_mix` and `xattnres_no_prefill` are the best current references. However, balanced accuracy remains close to random (`0.136` vs `0.111`), and macro F1 is below `0.10` for all models. This is not enough to claim strong spatial understanding or to rescue memory/region routing. The useful next step is a continuous bbox-center regression or heatmap-style localization probe, where small improvements can be measured by center error rather than coarse cell accuracy.

## BBox Center Regression Probe

Task: predict the largest-object normalized bbox center `(x_c, y_c)` with a 2D regression head. Training uses SmoothL1 loss on sigmoid-constrained coordinates. Metrics include MAE, mean/median L2 error, PCK at `0.05/0.10/0.20`, and two baselines: always predicting image center `(0.5, 0.5)` and always predicting the eval split mean target.

Output CSV: `results/bbox_center_regression_1000step_3seeds.csv`

Setting: MPS, 64px images, batch size 32, embed dim 32, 1000 steps, 3 split seeds, eval every 250 steps.

| Model | Mean L2 | Median L2 | PCK@0.10 | PCK@0.20 | Best mean L2 | Mean-target baseline L2 | Images/sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| `no_prefill_local_mix` | 0.1524 | 0.1309 | 0.377 | 0.717 | 0.1516 | 0.1518 | 301.55 |
| `xattnres_no_prefill` | 0.1524 | 0.1305 | 0.377 | 0.716 | 0.1514 | 0.1518 | 257.37 |
| `fpn_sum_lite` | 0.1524 | 0.1308 | 0.377 | 0.715 | 0.1519 | 0.1518 | 298.30 |
| `region_pool_mixer_no_history` | 0.1527 | 0.1305 | 0.376 | 0.717 | 0.1514 | 0.1518 | 254.72 |
| `stage_refresh_region_slots_2x2` | 0.1527 | 0.1310 | 0.376 | 0.717 | 0.1516 | 0.1518 | 242.01 |
| `anchor_only_no_prefill` | 0.1528 | 0.1305 | 0.376 | 0.716 | 0.1518 | 0.1518 | 235.35 |
| `tiny_vit` | 0.1528 | 0.1309 | 0.377 | 0.715 | 0.1514 | 0.1518 | 248.50 |

Interpretation: this regression probe is also inconclusive as an architecture discriminator. All models are essentially tied and remain at or slightly worse than the eval split mean-target baseline (`0.1518`). The models can learn the dataset-level center prior, but this setup does not show meaningful image-conditioned localization. The negative result is useful: coarse cell classification and direct global-pooled center regression are both too weak to validate region/memory claims. The next localization probe should expose spatial outputs directly, such as a heatmap head or patch-level objective, instead of only regressing coordinates from a global pooled classifier head.

## BBox Heatmap Localization Probe

Task: predict a `16x16` Gaussian heatmap centered at the largest-object bbox center. Unlike the coordinate-regression probe, this head is attached directly to the final 2D spatial feature map through a `1x1` convolution. For FPN controls, the heatmap head uses the fused FPN spatial feature rather than the last unfused memory state.

Output CSV: `results/bbox_heatmap_16x16_1000step_3seeds.csv`

Setting: MPS, 64px images, batch size 32, embed dim 32, 1000 steps, 3 split seeds, eval every 250 steps, heatmap size `16`, Gaussian sigma `1.5`.

| Model | Argmax mean L2 | Softargmax mean L2 | PCK@0.10 | Top1 cell acc | Best argmax L2 | Mean-target baseline L2 | Images/sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| `region_pool_mixer_no_history` | 0.1588 | 0.1506 | 0.351 | 0.066 | 0.1565 | 0.1518 | 185.07 |
| `stage_refresh_region_slots_2x2` | 0.1588 | 0.1514 | 0.357 | 0.066 | 0.1568 | 0.1518 | 183.18 |
| `fpn_sum_lite` | 0.1590 | 0.1516 | 0.354 | 0.067 | 0.1571 | 0.1518 | 192.83 |
| `no_prefill_local_mix` | 0.1590 | 0.1515 | 0.355 | 0.069 | 0.1568 | 0.1518 | 197.65 |
| `anchor_only_no_prefill` | 0.1592 | 0.1520 | 0.356 | 0.067 | 0.1568 | 0.1518 | 174.56 |
| `xattnres_no_prefill` | 0.1592 | 0.1515 | 0.354 | 0.067 | 0.1567 | 0.1518 | 170.22 |

Paired against `no_prefill_local_mix`, all argmax-L2 deltas are within about `0.0003` mean. `region_pool_mixer_no_history` and `stage_refresh_region_slots_2x2` have slightly lower mean argmax L2, but the effect is tiny and not aligned with clearly better PCK or top-1 cell accuracy. `anchor_only_no_prefill` does not improve heatmap localization, and it remains slower than the local/FPN controls.

Interpretation: the explicit heatmap head is a better probe design than global-pooled coordinate regression, but this DET-derived largest-box center target still does not provide strong architecture discrimination. The main warning sign is that argmax mean L2 (`0.1588-0.1592`) is still worse than the eval split mean-target baseline (`0.1518`). Softargmax mean L2 can fall near or below that baseline, but this is likely because smooth heatmaps collapse toward the dataset center prior. Therefore, the current heatmap result does not support a memory/region claim. The next spatial probe should provide denser supervision, such as a bbox mask heatmap or weak segmentation-style target, and should report IoU/Dice in addition to center PCK.
