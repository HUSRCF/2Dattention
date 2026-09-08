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
