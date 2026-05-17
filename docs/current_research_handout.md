# 2Dattention 当前研究任务 Handout

## 研究问题

我们最初的问题是：视觉模型是否应该继续把图像主要当作一维 token 序列处理？如果图像天然是二维空间结构，那么更合理的路线应当让模型在二维 feature field 上进行在线交互，并在需要检测/定位时显式利用区域或目标级线索。

经过多轮实验，当前项目已经从早期的“prefill memory / memory-first”概念探索，收敛为一个更稳的双主线：

- 分类任务：`no_prefill_local_mix` 是当前最强实用基线。
- 密集空间定位与检测方向：clean online anchor/region interaction 与 two-stage quality ranking 是当前最有价值的信号。
- 后续旗舰路线：`Proposal Consumption + Frozen Rank Calibration`，详见 `docs/research_route_collision_avoidance.md`。

一句话总结：

> 当前证据不支持 early prefill / memory-first；更稳的方向是 local-state-first，并在空间/检测任务中加入 clean anchor/region 或 proposal-query 机制。

## 已经被削弱的路线

以下方向在当前实验中没有形成稳定正证据：

- early spatial prefill：过早构建 memory 容易缓存低层、噪声或 stale feature。
- full prefill-lattice memory：复杂但没有稳定超过强基线。
- old XAttnRes-style early history protocol：no-prefill control 更强，说明早期 history 可能拖累。
- coarse quadrant/grid spatial probes：信号弱，容易被 label imbalance 或 dataset prior 主导。
- global-pooled center regression / Gaussian center heatmap：容易退化成中心先验，不能有效测空间能力。
- always-on quality auxiliary：会干扰 detector 主训练，不适合作为主 recipe。

这些 negative results 很重要，因为它们防止我们把“空间/记忆”叙事讲得过强。

## 当前成立的证据线

### 1. 分类主线：local-state-first

在 DET-derived 197-class、64px、1000-step 分类实验中：

| 模型 | Final Acc | Best Acc | 速度 |
|---|---:|---:|---:|
| `no_prefill_local_mix` | 0.250 | 0.250 | 约 462 img/s |
| `anchor_only_no_prefill` | 0.251 | 0.251 | 约 338 img/s |
| `xattnres_no_prefill` | 0.249 | 0.249 | 约 377 img/s |
| `fpn_sum_lite` | 0.244 | 0.244 | 约 447 img/s |

解释：

- `anchor_only_no_prefill` 有极小精度优势，但速度更慢。
- `no_prefill_local_mix` 仍是 practical Pareto baseline。
- FPN-like fixed fusion 不能解释 local mix 的分类优势。

因此分类主张应写成：

> 对低分辨率分类，轻量在线二维 local-state refinement 比 early memory routing 更可靠。

### 2. 密集空间定位：anchor-only no-prefill

真正有区分力的空间 probe 是 bbox mask localization，而不是中心点或粗网格分类。

在 16x16 与 32x32 bbox-mask 任务中：

- `anchor_only_no_prefill` 是当前最强 dense spatial model。
- 它稳定超过 `no_prefill_local_mix` 和 `fpn_sum_lite`。
- 优势在 small / medium / large object bins 中都保持。

解释：

> clean online anchor/region interaction 对 dense spatial prediction 有正信号，但这不等于 early prefill 或 memory-first 成立。

### 3. Detection toy：anchor/query 与 mask proposal 有正信号

在 synthetic detection toy 中：

- `AnchorQueryInit` 在 single-object square toy 上稳定优于 learned query。
- multi-distractor toy 中，anchor/region 信号仍存在但变弱。
- `DenseMaskAux` 作为旁路监督可学，但不会自动改善 query box。
- 真正有效的是 mask proposal query init：让 dense mask 显式产生 query seed。
- 加上 proposal diversity / spatial suppression 后，toy detection 的 IoU/AP 明显提升。

解释：

> dense mask signal 必须被 query 显式消费；仅作为 side auxiliary 不够。

## 当前 Real DET Mini 结论

我们已经从 toy 进入 real DET mini。当前 setting 是一个小规模 smoke：

- ILSVRC2013 DET-derived images
- top-10 classes
- 200 images
- max 3 objects
- 6 queries
- MPS 上快速训练与诊断

当前官方固定评分配置是：

```text
score = class_prob * quality^2
temperature = 1.0
```

post-hoc best-q 只作为诊断，不作为正式推理配置。

### Official fixed-q2 summary

| 模型 | Final IoU | Best IoU | Base AP50 | Class AP50 | Fixed-q2 AP50 | IoU-ref AP50 |
|---|---:|---:|---:|---:|---:|---:|
| `local_learned` | 0.359 | 0.367 | 0.310 | 0.108 | - | - |
| `local_anchor_residual_query` | 0.376 | 0.376 | 0.266 | 0.135 | - | - |
| `local_anchor_residual_query_quality_head` | 0.376 | 0.376 | 0.266 | 0.135 | 0.341 | 0.421 |
| `local_mask_proposal_nms_query_quality_head` | 0.357 | 0.366 | 0.266 | 0.093 | 0.282 | 0.364 |
| `local_mask_proposal_oracle_nms_query_quality_head` | 0.399 | 0.421 | 0.259 | 0.138 | 0.325 | 0.372 |

当前最稳解释：

- AP 主线：`local_anchor_residual_query_quality_head`
- localization / coverage 候选：`local_mask_proposal_oracle_nms_query_quality_head`
- fixed-q2 quality score 明确改善 ranking。
- temperature=2.0 没有超过 temperature=1.0，因此暂时不继续在 eval 上调温度。

## 为什么 two-stage quality head 重要

我们发现 real DET mini 的一个核心瓶颈不是 box 完全不行，而是 score ranking 没有对齐 box quality。

失败路径：

- 把 IoU target 直接压到 class logits 上，会混淆 semantic classification 和 localization quality。
- always-on quality auxiliary 会干扰 detector 主训练。

成功路径：

1. 先正常训练 detector。
2. 冻结 detector。
3. 只训练独立 query quality head。
4. 推理时用固定：

```text
score = class_prob * quality^2
```

这条路径能在不破坏 box 的情况下改善 AP ranking。

## 当前项目主张

最稳的论文式表述是：

> Early prefill and memory-first routing are not supported by the current evidence. In classification, lightweight no-prefill local-state refinement is the strongest practical baseline. In dense spatial prediction and detection-style tasks, clean online anchor/region interaction and proposal-conditioned query initialization provide useful signals. For real DET mini, the current best AP route is residual-anchor query initialization plus a two-stage frozen quality head with fixed q² scoring.

更短版本：

> Classification 看 local-state-first；dense spatial / detection 看 anchor-region query bias；AP ranking 需要 two-stage quality scoring。

## 当前限制

这仍然不是 RF-DETR 级结果。当前 real DET mini 只是用于快速判断方向，限制包括：

- 数据规模小；
- 训练步数短；
- decoder 很弱；
- AP 是 lite diagnostic，不是完整 COCO AP；
- proposal query 还没有真正形成强 real-detector AP 优势；
- post-hoc best-q 不能作为正式配置；
- 还没有和 RF-DETR teacher / full detector pipeline 做直接对比。

因此现在不能说我们已经冲击 RF-DETR，只能说：

> 我们已经找到一个更清晰的 detector-building direction：proposal/query + residual anchor + two-stage quality ranking。

## 下一步计划

P0：固定 real-mini 评估口径

- 固定 `q^2, temperature=1.0` 作为正式 scoring。
- post-hoc best-q 和 IoU-reference 只作诊断。
- 继续记录 score-IoU correlation、quality-IoU correlation、combined-IoU correlation。
- 当前 robustness blocker 是 off-center；不要继续扫 alpha，也不要回到 persistent proposal state。

P1：诊断 off-center query/representation

- 增加 center/offcenter-stratified 的 query assignment、TP50 class accuracy、duplicate FP、score-IoU correlation、matched-class accuracy。
- 已排除的简单解释：aggregate alpha mismatch、quality head 缺 box 坐标、class/objectness head 缺 box 坐标。
- 只有诊断显示具体失败模式后，再考虑新的 query-position representation 或 off-center-aware scoring。

P2：做 calibration split

- 不再在 eval set 上调 alpha / temperature。
- 使用独立 calibration split 选择 alpha 或 temperature。
- held-out eval split 只用于最终报告。

P3：走向 RF-DETR path

- 使用 RF-DETR 作为 teacher。
- 蒸馏 boxes/classes/query behavior。
- 对比 student with / without anchor/proposal/quality mechanisms。
- 从 mini-detector 逐步扩大到真实 detector benchmark。

## 当前可展示的一页结论

```text
任务目标：
  从“图像不应只是一维 token 序列”出发，寻找更适合二维空间与检测任务的视觉结构。

已否定：
  early prefill / memory-first / stale history routing。

分类主线：
  no_prefill_local_mix。

空间定位主线：
  anchor_only_no_prefill。

检测方向：
  residual-anchor query + two-stage quality head。

正式 scoring：
  class_prob * quality^2, temperature=1.0。

当前结论：
  local-state-first 适合分类；
  online anchor/region interaction 适合 dense spatial；
  real DET mini 的 AP 瓶颈主要在 ranking/scoring；
  下一步应提升 proposal/query coupling，并用 RF-DETR teacher 进入更真实 detector path。
```
