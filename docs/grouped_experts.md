# Grouped Private Experts

The prototype now supports an optional grouped lattice read in
`TinyPrefillLatticeAttnRes`:

```python
model = TinyPrefillLatticeAttnRes(embed_dim=48, expert_groups=2)
result = model(images)
private = result["expert_readouts"]  # one [B, E, D, H, W] tensor per read block
```

With `expert_groups=1` the original shared-channel `LatticeMemoryRead` path
is unchanged. With multiple groups, each channel group has its own query,
candidate routing distribution, and value projection. The concatenated read
still has shape `[B, C, H, W]`.

The module exports two optional regularizers:

```python
from attention2d import cross_group_decorrelation_loss, variance_floor_loss

regularizer = sum(
    cross_group_decorrelation_loss(readout) + variance_floor_loss(readout)
    for readout in result["expert_readouts"]
)
loss = task_loss + lambda_cross * regularizer
```

These losses only constrain private readouts. They do not establish semantic
roles by themselves. Use controlled interventions and selective expert
ablations to test whether groups acquire useful specialization. PCA is not
performed online; it remains an optional initialization or comparison to add
in a separate experiment.

This implementation targets the repository's tiny prototype. The external
RF-DETR training scripts do not automatically use it.

## CPU Smoke Result (2026-09-08)

On `aligned_pair`, using CPU, `embed_dim=16`, one prefill round, one read
block, 120 AdamW steps, batch size 32, and seeds 21/22/23:

| Variant | Eval accuracy (per seed) | Mean |
| --- | --- | ---: |
| `expert_groups=1` | 0.672, 0.613, 0.486 | 0.590 |
| `expert_groups=2` | 0.637, 0.652, 0.637 | 0.642 |

This is an encouraging preliminary `+5.1` percentage-point signal, but the
short synthetic run has high seed variance and remains near a toy-task
ceiling. A longer multi-seed run and controlled intervention/ablation are
required before treating the mechanism as an effective specialization method.
