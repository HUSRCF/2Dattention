# Literature Notes

These notes summarize the papers fetched under `/tmp/2dattention_lit` for this prototype.

## AttnRes: Attention Residuals

- arXiv: `2603.15031`
- Core idea: standard PreNorm residuals accumulate previous layer outputs with fixed unit weights.
- AttnRes replaces fixed residual accumulation with softmax attention over preceding layer outputs.
- Block AttnRes reduces overhead by attending over block-level summaries instead of every layer output.
- Prototype implication: use learned pseudo-query routing over memory sources instead of blindly summing features.

## XAttnRes: Cross-Stage Attention Residuals

- arXiv: `2604.03297`
- Extends AttnRes from same-shape LLM layers to multi-scale encoder-decoder segmentation features.
- Maintains a global feature history pool and aligns spatial/channel dimensions before aggregation.
- Prototype implication: a visual AttnRes module should treat feature history as a memory pool, not only as the previous block.

## Mamba-3

- arXiv: `2603.15569`
- Improves sequence modeling through more expressive recurrence, complex-valued state tracking, and MIMO SSM formulation.
- It is still a sequence-model paper, so it should be treated as an information-flow inspiration rather than copied directly into the first vision demo.
- Prototype implication: state tracking matters, but v1 should avoid implementing a full SSM kernel.

## V2M: Visual 2-Dimensional Mamba

- arXiv: `2410.10382`
- Argues that flattening image tokens into 1D scans disrupts 2D coherence and locality.
- Generalizes SSM formulation toward 2D state updates and uses four image corners to respect non-sequential image structure.
- Prototype implication: future prefill modules should move from local conv updates toward true synchronous 2D state updates.

## Vision GNN

- arXiv: `2206.00272`
- Treats image patches as graph nodes and connects nearest neighbors for graph-level processing.
- Motivates graph edges for irregular object topology beyond rectangular grids.
- Prototype implication: graph memory should be added as an optional routing source after the lattice baseline is validated.
