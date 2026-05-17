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
- real DET mini 的 AP 提升靠 `local_anchor_residual_query + two-stage quality head + fixed q^2 scoring`。

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
+ fixed score = class_prob * quality^2
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
| 新的 DETR query initialization | init-only 与 persistent proposal consumption 的性能缺口诊断 |
| 新的 quality score | frozen, post-detector, held-out calibrated ranking head |
| 新的 local-global backbone | ambiguity-triggered interaction schedule |
| 新的 detector distillation | hard-negative query ranking 与 proposal-use behavior 的 targeted distillation |
| 改进 RT-DETR 用于某行业 | proposal consumption / ranking calibration 机制，不绑定行业场景 |

专利规避上，应避免把项目写成“改进 RT-DETR/DETR + 多尺度增强 + 辅助头 + 某行业目标检测”。更安全的是围绕 proposal utilization gap、frozen ranking calibrator、predicted-vs-oracle proposal diagnostic protocol 来表述。

## 推荐旗舰路线

建议把 6-12 个月主项目定义为：

```text
Proposal Consumption + Frozen Rank Calibration
```

分两阶段执行：

1. 低成本验证：冻结式排序校准头。
2. 中风险创新：持久化 proposal state decoder。

这样可以先回答 ranking/calibration 是否真是瓶颈；如果成立，再推进 proposal 在 decoder 中是否被持续消费。

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
- fixed `q^2, temperature=1.0` 是当前正式 scoring。

建议实验：

- independent calibration split，而不是 eval-set tuning。
- baseline: class score only。
- calibration baselines: temperature scaling, Platt-like score head, simple IoU predictor。
- metrics: AP50 / AP75, class-aware AP, ECE, score-IoU correlation, quality-IoU correlation, false-positive rate at matched recall。

阶段门：

```text
AP75 +0.8 或 ECE 下降 15% 以上，才继续扩大 calibration 线。
```

## 方向 B：持久化 Proposal State Decoder

目标：

```text
proposal 不只初始化 query；
proposal geometry / mask / pooled feature / uncertainty 在 decoder 多层中持续被 query 消费。
```

核心问题：

```text
predicted proposal 与 oracle proposal 存在缺口；
当前 proposal 多数是 init-only 或 hard proposal；
decoder 没有被证明持续消费 proposal state。
```

建议对照：

- learned queries
- anchor residual query
- proposal init-only
- layerwise proposal re-injection
- persistent proposal state
- oracle proposal state

主指标：

- AP50 / AP75
- oracle-gap closing ratio
- ambiguity split AP
- same-class distractor false positives
- query collapse / duplicate predictions
- query assignment entropy

阶段门：

```text
oracle-gap 缩小 25% 或 ambiguity split AP 明显高于 clean split 改进，才继续深挖。
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
| 2026-06 | 固定 baseline、slice 协议、predicted-vs-oracle proposal gap |
| 2026-07 | frozen rank calibrator 原型与 calibration split |
| 2026-08 | calibration / ranking 消融，阶段门 1 |
| 2026-09 | persistent proposal state decoder 原型 |
| 2026-10 | init-only / re-inject / persistent-state 对照 |
| 2026-11 | ambiguity split 大规模评测，阶段门 2 |
| 2026-12 | 可选 teacher-guided proposal-ranking distillation |
| 2027-01 | license / IP / FTO 复核，阶段门 3 |
| 2027-02 | 论文写作、图表整理、外部复现 |
| 2027-03 | 补实验、投稿/专利评估 |

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

- 保持 official scoring：`class_prob * quality^2`, `temperature=1.0`。
- 建立 calibration split，不再用 eval set 调 alpha / temperature。
- 补 score-IoU / quality-IoU / combined-score-IoU correlation。
- 汇总 predicted-vs-oracle proposal gap。

P1:

- 实现 persistent proposal state decoder 的最小版本。
- 做 init-only vs layerwise re-inject vs persistent-state 对照。
- 记录 oracle-gap closing ratio。

P2:

- 建立 ambiguity diagnostic split：
  - same-class distractor
  - small object
  - crowded object
  - off-center object
  - high proposal entropy

P3:

- 仅在 P0/P1 成立后，再做 RF-DETR teacher distillation。

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
real DET 的 AP ranking 需要低干扰 quality calibration。
```

一句话：

```text
先用 frozen ranking calibrator 低成本确认 score-quality gap；
再用 persistent proposal state decoder 解决 proposal consumption gap。
```
