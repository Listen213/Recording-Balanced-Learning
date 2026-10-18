# 第二数据集：论文写作阶段再执行

当前工程暂不跑第二数据集。

到论文写作阶段再补时，按以下规则：

1. 选择公开 UATR 数据集；
2. 先按 raw recording / source session 做独立性审计；
3. 绝不先切 segment 再随机划分；
4. 如果存在 near-duplicate / same-session recording，做 group-aware split；
5. 固定同一套 S0 / S1 / PB / PT / F0；
6. 不因为第二数据集结果修改 ShipsEar hyperparameters；
7. 可根据第二数据集自身 development split 设定必要的数据读取参数，但不能改变核心方法；
8. 单独冻结 final test；
9. 报同样的 2/3/4-rec protocol（若数据规模允许）；
10. 最终用于验证跨数据集泛化，而不是继续方法搜索。

论文中第二数据集的作用是：

> external validation / cross-dataset evidence

而不是新的 tuning dataset。
