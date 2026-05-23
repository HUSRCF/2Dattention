# 后续研究路线与撞车规避报告

## 执行摘要

当前项目值得继续，但主线必须从 `2D prefill / memory-first` 收敛到：

```text
local-state backbone
+ anchor/region query coupling
+ low-interference quality ranking
```

最稳判断：

- 分类靠 `no_prefill_local_mix` 所代表的 local-state refinement。
- 密集空间定位靠 `anchor_only_no_prefill` 所代表的 clean online anchor/region interaction。
- real DET mini 的 AP 提升靠 `local_anchor_residual_query + two-stage quality head + held-out calibrated quality scoring`；固定 `q^2` 仍是预注册基线。

不应继续试图证明：

```text
early prefill memory 是视觉建模核心
```

更稳的研究句式是：

```text
分类任务需要稳定局部状态更新；
dense spatial / detection 任务需要 proposal 或 anchor-region signal 被 query 持续消费；
AP ranking 需要与 localization quality 解耦。
```

## 已降级方向

以下方向已有足够负证据，不再作为主 claim：

- early prefill
- memory-first
- full prefill-lattice
- graph-prefill
- old XAttnRes early-history protocol
- side-only DenseMaskAux
- quality loss directly on foreground class logits
- coarse quadrant/grid probes
- global-pooled center regression
- Gaussian center heatmap

这些方向可作为 negative findings 保留，但不应继续投入大量工程。

## 当前正信号

### 1. Classification: Local-State First

`no_prefill_local_mix` 是 DET-derived classification 的 practical Pareto baseline。它说明低分辨率分类更受益于在线 2D local-state refinement，而不是 early memory routing。

### 2. Dense Spatial Localization: Online Anchor/Region

`anchor_only_no_prefill` 在 16x16 和 32x32 bbox-mask localization 中稳定领先，并且 small / medium / large 分层都保持优势。这个信号支持 clean online anchor/region interaction，但不支持 early prefill。

### 3. Detection / Ranking: Residual Anchor + Frozen Quality

当前最稳 real-mini recipe：

```text
local_anchor_residual_query
+ two-stage frozen quality head
+ held-out calibrated score, with fixed q^2 as the pre-registered baseline
+ temperature = 1.0
```

它说明 detector 的一部分瓶颈是 score-to-IoU misalignment。quality head 应作为低干扰 post-detector ranking head，而不是 always-on auxiliary 或 class-logit calibration。

## 撞车地图

### 高撞车区

以下泛化表述与近年 DETR / RT-DETR / calibration 文献高度重叠：

- new DETR query initialization
- dynamic anchor boxes / reference points
- proposal-guided decoder
- IoU-aware classification score
- improved local-global vision backbone
- DETR distillation from strong teacher
- improved RT-DETR for a specific industrial scenario

如果使用这些表述，很容易撞上 DETR、Deformable DETR、Conditional DETR、DAB-DETR、Anchor DETR、DINO、RT-DETR、Sparse R-CNN、DDQ、HPR、RAQG、PaQ-DETR、IoU-Net、VarifocalNet、GFLV2、Rank-DETR、Cascade-DETR、Cal-DETR、DETRDistill、QSKD 等方向。

### 更安全的贡献句式

| 高风险表述 | 更安全表述 |
|---|---|
| 新的 DETR query initialization | layerwise proposal refresh / `reinject` 与 frozen ranking calibration 的机制诊断 |
| 新的 quality score | frozen, post-detector, held-out calibrated ranking head |
| 新的 local-global backbone | ambiguity-triggered interaction schedule |
| 新的 detector distillation | hard-negative query ranking 与 proposal-use behavior 的 targeted distillation |
| 改进 RT-DETR 用于某行业 | proposal consumption / ranking calibration 机制，不绑定行业场景 |

专利规避上，应避免把项目写成“改进 RT-DETR/DETR + 多尺度增强 + 辅助头 + 某行业目标检测”。更安全的是围绕 proposal utilization gap、frozen ranking calibrator、predicted-vs-oracle proposal diagnostic protocol 来表述。

## 推荐旗舰路线

当前最新实验已经进一步收缩了路线。`reinject`、querymask、grid query、reference-box
controls 和 minimal DAB-style updates 都暴露了 AP/geometry trade-off，但没有解决
off-center object binding。因此 6-12 个月主项目不应继续停留在 tiny detector
scaffold，而应定义为：

```text
Real RF-DETR Adaptation
+ Category-Candidate Diagnostics
+ Robustness Slice Protocol
```

分三阶段执行：

1. 正式 RF-DETR 长训 / continuation：使用 `checkpoint_best_regular.pth`、独立 test
   export、class-aware/class-agnostic/slice/candidate-oracle 共同报告。
2. 类别候选修复：围绕 category assignment / candidate coverage，而不是 post-hoc
   crop-prior 或 scalar calibration。
3. off-center robustness：把 tiny-detector 的 offcenter 失败结论转化为真实 detector
   诊断，不再在 tiny scaffold 上堆 query/reference 小启发式。

`persistent proposal state` 仍只保留为 oracle/proposal-quality 诊断分支。`reinject`
也降级为轻量 proposal-consumption probe，不再作为 active detector architecture。

## 方向 A：冻结式排序校准头

目标：

```text
冻结 detector 主体；
只训练轻量 quality / rank calibrator；
让 score 更好反映 localization quality。
```

现有证据：

- two-stage quality head 已能在 real DET mini 上提升 AP。
- always-on quality 会干扰 detector 主训练。
- fixed `q^2, temperature=1.0` 是预注册 scoring；启用 calibration split 时，正式 fixed score 使用 calibration-selected alpha，并在 final eval 上固定。

建议实验：

- independent calibration split，而不是 eval-set tuning。
- baseline: class score only。
- calibration baselines: temperature scaling, Platt-like score head, simple IoU predictor。
- metrics: AP50 / AP75, class-aware AP, ECE, score-IoU correlation, quality-IoU correlation, false-positive rate at matched recall。

阶段门：

```text
AP75 +0.8 或 ECE 下降 15% 以上，才继续扩大 calibration 线。
```

## 方向 B：Layerwise Proposal Refresh / Reinject

目标：

```text
proposal 不只初始化 query；
proposal geometry / mask / pooled feature 在 decoder 后续层以轻量 refresh 方式再次暴露给 query。
```

核心问题：

```text
predicted proposal 与 oracle proposal 存在缺口；
当前 proposal 多数是 init-only 或 hard proposal；
stateful persistent consumer 在 predicted proposal 下不稳定；
layerwise refresh / reinject 是当前更可靠的 predicted-proposal 路径。
```

建议对照：

- learned queries
- anchor residual query
- proposal init-only
- layerwise proposal re-injection
- oracle proposal re-injection
- persistent proposal state only as oracle/proposal-quality diagnostic

主指标：

- AP50 / AP75
- oracle-gap closing ratio
- ambiguity split AP
- same-class distractor false positives
- query collapse / duplicate predictions
- query assignment entropy

阶段门：

```text
reinject/refresh 必须同时改善 geometry 与 AP，才继续扩大。
persistent 只有在 future proposal-quality 或 oracle-gap run 同时超过 reinject 的 geometry/AP 后，才允许回到 mainline。
```

当前状态更新：

- 标准 mini 协议下，`reinject` 能给出有限 aggregate AP/ranking 信号，但不能解决
  offcenter object binding。
- 在 `no_object_weight=0.5` offcenter stress 中，`reinject_g003_quality_head`
  对 aggregate AP50 有小幅帮助，但 final/best IoU、AP75 和 offcenter AP50 都下降。
- 结论：`reinject` 是机制诊断，不是当前主线。不要继续扩大 gate/alpha/persistent
  sweep；如果再碰这条线，只应作为 oracle-gap / proposal-quality 诊断。

## 方向 B2：Real RF-DETR Direct Adaptation

目标：

```text
使用真实 RF-DETR Small/Nano 训练协议验证类别候选、定位和 slice robustness；
把 tiny-detector 的机制假设迁移到成熟 detector，而不是继续修补玩具 scaffold。
```

当前证据：

- Stratified RF-DETR Small 384 2ep 是 fair-split smoke baseline。
- Seed41 full low-LR all-parameter continuation 首次同时改善 class AP 和 category
  candidate generation。
- 同 seed 继续低 LR 能继续提高 deployed AP，但 candidate coverage 不再单调改善。
- Seed43 forced-head 2ep sanity 已确认 `num_classes=200` head 修复在推理路径生效；
  它是 repaired-head baseline row，不改变长训主线。
- 更长 direct Small 训练已成为当前最强 RF-DETR 路线，但 epoch optimum 是
  seed-sensitive；结论必须来自 independent `predict_rfdetr_coco.py` export，而不是
  RF-DETR internal EMA/best-total。

主指标：

- class AP/AP50/AP75
- class-agnostic loc AP/AP50/AP75
- offcenter / center / small / medium / large slice AP50
- candidate hit rate
- candidate-oracle AP/AP50/AP75
- top-k loc/class/per-category coverage
- score-IoU correlation

阶段门：

```text
如果 longer direct RF-DETR 在第二 seed 上也稳定提高 class AP50 和 slice AP50，
主线转为 real RF-DETR long-train / category-candidate repair。
如果 AP 提升但 candidate hit/candidate-oracle 不升，说明只是 scoring/ranking 改善，
不能宣称 candidate generation 被修复。
```

## 方向 C：任务自适应 Local-State Interaction

这不是当前第一优先级，但可作为后续 backbone 方向。

目标：

```text
默认 local-state refinement；
只有在 high ambiguity / high proposal entropy / same-class crowding 时触发 sparse region interaction。
```

规避表述：

- 不写“新 local-global backbone”。
- 写“ambiguity-triggered interaction schedule”。

成功标准：

- clean split 不降速或少降速。
- distractor / crowded split 的收益明显大于 overall split。

## 方向 D：Teacher-Guided Proposal-Ranking Distillation

可选增强，不应先做。

目标：

```text
不做全量 feature/logit distillation；
只蒸馏 hard-negative query ranking、proposal use、quality-aware selection。
```

注意：

- RF-DETR 可作为 teacher，但要检查 license 和 weight redistribution。
- 不要把贡献写成普通 detector distillation。

## 年度计划

| 时间 | 任务 |
|---|---|
| 2026-06 | 固定 strict split、slice 协议、RF-DETR regular-checkpoint export |
| 2026-07 | 完成 RF-DETR direct Small 多 seed / epoch curve |
| 2026-08 | category-candidate diagnostics：hit rate、oracle AP、per-category zero-hit |
| 2026-09 | offcenter/small robustness slice + qualitative attention / prediction overlays |
| 2026-10 | 若 direct training 饱和，再做 teacher-guided category / proposal-use distillation |
| 2026-11 | 更大协议或外部 benchmark 复验 |
| 2026-12 | license / IP / FTO 复核 |
| 2027-01 | 论文主表、negative findings、消融表整理 |
| 2027-02 | 外部复现与补实验 |
| 2027-03 | 投稿/专利评估 |

## 预算口径

若已有本地算力和数据流水线，实验预算可控制在较低区间。若需要云 GPU、补标、teacher 缓存和 IP 检索，需要单独预留。

| 项目 | 保守版 | 扩展版 |
|---|---:|---:|
| 云算力 / GPU 折旧 | 8,000 | 18,000 |
| teacher 推理与缓存 | 2,000 | 6,000 |
| 数据清洗、补标与 QA | 6,000 | 18,000 |
| 自动评测与实验工程 | 3,000 | 8,000 |
| 存储、备份与版本管理 | 1,000 | 3,000 |
| IP / FTO 检索预留 | 5,000 | 15,000 |
| 机动缓冲 | 3,000 | 10,000 |
| 小计，不含全职工资 | 28,000 | 78,000 |

若计入 0.5-1 FTE，项目总预算更合理区间约为 `48,000-158,000`，具体取决于本地算力和标注支持。

## 当前立即执行项

P0:

- 继续使用 RF-DETR `checkpoint_best_regular.pth` 做独立 prediction export；不要用
  `best_total`/EMA 做结论。
- 维护 class-aware、class-agnostic、slice、candidate-oracle、coverage、score-IoU
  统一报告。
- 把 forced-head seed43 2ep 作为 repaired-head sanity row，不要和 seed41 long
  continuation 混作同阶段比较。

P1:

- RF-DETR direct Small long-train / continuation 是 active detector route。
- 只在 direct route 饱和后，再考虑 teacher-guided category/proposal-use distillation。
- 类别候选修复优先于 post-hoc crop-prior 或 scalar calibration。

P2:

- 建立 ambiguity diagnostic split：
  - same-class distractor
  - small object
  - crowded object
  - off-center object
  - high proposal entropy

P3:

- 仅在 direct RF-DETR route 明确饱和后，再做 teacher distillation。

## 最终判断

最稳项目叙事不是：

```text
我们提出一种新的 2D prefill Transformer。
```

而是：

```text
我们系统比较 memory-first、local-state-first 与 anchor/proposal query coupling。
实验显示 early memory 不稳；
classification 更受益于 local-state refinement；
dense spatial 与 detection 更需要 proposal/anchor signal 被 query 持续消费；
real DET 的 AP ranking 需要低干扰 quality calibration；
但真正 RF-DETR 路线目前更受 category-candidate generation 和长训协议支配。
```

一句话：

```text
tiny detector 已经完成机制排雷；
下一阶段用真实 RF-DETR regular-checkpoint 长训和 category-candidate diagnostics 推进。
```
