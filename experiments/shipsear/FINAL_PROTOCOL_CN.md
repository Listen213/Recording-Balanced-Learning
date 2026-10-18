# ShipsEar Final Protocol

## A. 数据独立性

- shot 的单位：raw recording
- segment 不算独立样本
- split 在 segmentation 之前完成
- near-duplicate connected component 为 leakage-control 原子组
- component 不允许跨 support_pool / final test

## B. Final split

默认预注册参数：

```text
split seed = 20261014
test ratio = 0.30
near-duplicate threshold = 0.997
minimum support components/class = 6
fingerprint = start/middle/end 3 windows
fingerprint window = 20 s
fingerprint sample rate = 8 kHz
segment = 5 s
hop = 5 s
```

split 生成后不得为了性能重新换 seed。

## C. 方法冻结

最终方法集合固定：

```text
S0
S1
PB
PT
F0
```

PT：

```text
rank = 2
steps = 160
lr = 0.05
l2 = 0.3
temperature = 0.25
```

F0：

```text
alpha = 0.5
```

不做 test-driven tuning。

## D. Few-recording protocol

```text
2 / 3 / 4 recordings per class
20 support draws each
support seeds 101–120
```

同一 support draw 内不得选择同一 near-duplicate component 中的两个 recordings。

## E. Final test policy

- 方法、split、support seeds、hyperparameters 全部先 lock；
- test 运行时不输出中间 metrics；
- 完成后一次性输出全部 final results；
- final outcome 不再用于修改 ShipsEar 方法。

## F. 统计

主指标：

- recording-level Macro-F1
- recording-level Balanced Accuracy
- recording-level Accuracy

辅助：

- component-level Macro-F1
- per-class recall
- confusion matrix
- runtime
- downstream trainable parameters

比较：

```text
F0 vs S0
F0 vs S1
F0 vs PB
F0 vs PT
```

使用 hierarchical paired bootstrap：

- resample support draws
- resample final-test components
- paired comparison
- 95% CI

## G. 第二数据集

本工程暂不执行第二数据集。

论文写作阶段再补，并使用独立冻结协议；不能利用第二数据集结果回头改 ShipsEar final method。
