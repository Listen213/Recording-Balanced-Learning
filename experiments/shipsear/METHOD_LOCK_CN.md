# 方法锁定说明

截至本工程：

## 保留

- Frozen BEATs
- S1 Recording-balanced Segment Evidence
- PB Equal-recording Prototype
- PT Low-rank Prototype Tuning
- F0 Fixed Geometric Fusion

## 不再继续

- GMM/UBM distribution route
- Gaussian/Bures route
- Sinkhorn / mixture OT route
- attention MIL / DSMIL route
- SupCon
- LoRA
- residual adapter
- hierarchical bag loss
- query-adaptive gate
- class-wise reliability gate
- prototype perturbation
- uncertainty-gated prototype movement

这些分支可以在论文中作为 development evidence / ablation history 使用，但不进入最终方法。

## 论文 Proposed

建议名称：

**Dual-Granularity Evidence Fusion (DGE)**

核心贡献不是“0.5 这个数字”，而是：

1. recording-balanced segment evidence；
2. equal-recording / low-rank prototype evidence；
3. 两种 granularity 的固定、预注册、低自由度 evidence fusion；
4. few-recording recording-first evaluation protocol。

是否将“协议贡献”写成 novelty，需要在正式文献综述后谨慎表述，不要直接宣称首创。
