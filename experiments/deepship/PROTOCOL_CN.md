# DeepShip 50/class External-Validation Protocol V3

## 1. 独立样本单位

- 一级 DeepShip session directory = 1 recording/session unit；
- 类别目录中的顶层 WAV = 1 recording unit；
- session 内多个 WAV 仍属于同一 recording/session；
- 5 秒 segment 永远不算独立 shot。

## 2. Hard duplicate identity

只使用 exact byte-level evidence：

1. complete unit SHA256 相同；
2. 共享完全相同 WAV SHA256。

同日期不 hard merge。
Acoustic cosine 不 hard merge。

## 3. Acoustic similarity

对最终选中的 200 条 recording 计算 fingerprint，相似度只作为 QC/诊断。
阈值 0.997 仅用于报告，不用于合并、删样本或重新 split。

## 4. Balanced subset

- 50 official recordings/class；
- 30/class support pool；
- 20/class frozen test；
- seed = 20261015。

## 5. Date correlation control

同一天 recordings 保持为不同 recording。

日期仅用于：

- split 时尽量把同一天 recordings 放同一侧；
- few-recording draw 时尽量避免同一天重复抽样。

## 6. Few-recording

- 2 recordings/class × 20 draws；
- 3 recordings/class × 20 draws；
- 4 recordings/class × 20 draws；
- full support 只作 reference。

## 7. Methods

S0 / S1 / PB / PT / F0 与 ShipsEar 锁定版本完全一致。

## 8. Statistics

5000 hierarchical paired bootstrap resamples，检验：

- S1 - S0；
- PT - PB；
- F0 - S0；
- F0 - S1；
- F0 - PB；
- F0 - PT。

指标：Macro-F1 / Balanced Accuracy / Accuracy。
