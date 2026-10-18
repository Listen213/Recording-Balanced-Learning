# 论文实验章节建议结构

## 1. Experimental Setup

交代：

- ShipsEar raw recordings
- 5 类 coarse taxonomy
- frozen BEATs
- 5 s non-overlapping segments
- recording-first split
- near-duplicate component grouping
- 2/3/4 recordings-per-class

## 2. Compared Methods

### S0
普通 segment ERM。

### S1
recording/class-balanced segment learning。

目的：验证 segment pseudo-replication 的影响。

### PB
equal-recording prototype。

目的：验证纯 recording-level class center。

### PT
low-rank prototype tuning。

目的：验证极少 recording 下 parameter-efficient prototype adaptation。

### F0 / DGE
Dual-Granularity Evidence Fusion：

\[
\log p_c
=
0.5 \log p_{S1,c}
+
0.5 \log p_{PT,c}
\]

作为最终 proposed。

## 3. Main Results

主表：

```text
Method × {2-rec, 3-rec, 4-rec}
Macro-F1 mean±std
Balanced Accuracy mean±std
```

## 4. Why Recording Balancing Matters

重点写：

```text
S0 -> S1
```

并讨论：

- 每条 recording segment 数量不同；
- ordinary segment ERM 会让长 recording 获得更大训练权重；
- recording balancing 把独立 recording 恢复为统计基本单位。

## 5. Dual-Granularity Complementarity

分析：

- S1 segment evidence
- PT recording-prototype evidence
- prediction disagreement
- per-class recall
- support-size trend

不要把固定 0.5 描述成“复杂自适应机制”，而应强调：

> 在极少 independent recordings 下，固定预注册融合比高自由度 reliability gate 更稳健。

## 6. Statistical Reliability

报告：

- 20 support draws
- hierarchical paired bootstrap
- 95% CI
- final test component-aware resampling

## 7. Complexity

报告：

- BEATs frozen
- downstream trainable params
- downstream fit time
- prototype tuning rank

不要使用“轻量部署”表述。
