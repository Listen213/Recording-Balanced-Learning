# FAST Final Engine 技术说明

## 1. Checkpoint

每个 support draw 保存：

```text
runs/final_fast/checkpoints/draws/<setting>_seed<seed>.npz
runs/final_fast/checkpoints/draws/<setting>_seed<seed>.json
```

NPZ 只保存：

- frozen test labels / IDs / component IDs；
- S0/S1/PB/PT/F0 probabilities。

JSON 只保存：

- FINAL_LOCK hash；
- support draw recording IDs；
- runtime / trainable params；
- PT non-test training diagnostics。

不在中途生成 human-readable final F1/Accuracy。

写入使用 temporary file + `os.replace`，所以进程在写文件中途被 Kill 时不会留下“半个有效 checkpoint”。

## 2. Exact fast bootstrap

旧版 bootstrap 的随机调用顺序为：

1. support draw indices with replacement；
2. class 0 components with replacement；
3. class 1 components with replacement；
4. ...；
5. 下一 replicate。

快速版本仍按同一顺序生成随机 index，因此在相同 seed 下与慢速 reference 使用同一批 resamples。

差别只在 metric 计算：

- 旧版：Python loop + sklearn；
- 新版：NumPy vectorized confusion statistics。

Smoke test 会直接比较 fast 与 slow 的：

```text
mean_delta
low95
high95
p_delta_le_0
```

对 Macro-F1 / Balanced Accuracy / Accuracy 三个指标要求 `atol <= 1e-12`。

## 3. Bootstrap checkpoint

每完成一个：

```text
setting × baseline comparison
```

就写：

```text
runs/final_fast/checkpoints/bootstrap_results.json
```

总 block 数从旧版“36 个 metric calls”简化为：

```text
3 settings × 4 baselines = 12 blocks
```

每个 block 同时返回三个指标。

## 4. Resume

Resume 不改变：

- FINAL_LOCK；
- support draw seed；
- method；
- prediction；
- bootstrap seed；
- bootstrap resampling definition。

因此属于技术性断点恢复，不属于重新调参或重复选择 final test。
