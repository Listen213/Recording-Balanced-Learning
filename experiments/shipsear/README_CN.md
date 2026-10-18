# RIFT_FEWREC_FINAL_FAST_FULL

这是上一版 `RIFT_FEWREC_FINAL_FULL` 的**执行优化版**。

**方法、fresh split、FINAL_LOCK、support seeds、PT 参数、F0 融合权重、5000 次 hierarchical paired bootstrap 定义全部不变。**

本版只解决两个工程问题：

1. 上一版 61/61 后才开始慢速 bootstrap，而且没有进度；
2. 所有 prediction 都只放在内存里，进程被 `Killed` 后必须从头算。

新版本加入：

- 每个 support draw 完成后立即原子 checkpoint；
- 进程被杀后可从最近 draw 继续；
- bootstrap 每个 comparison block 完成后立即 checkpoint；
- bootstrap 改成 NumPy 向量化实现；
- 快速实现与旧慢速参考实现使用**完全相同的 resampling 定义和 RNG 调用顺序**；
- 一次 bootstrap block 同时算 Macro-F1 / Balanced Accuracy / Accuracy，避免重复抽样三次；
- 实时显示 EVAL、BOOT、elapsed、ETA；
- `PROGRESS.json` 可随时查看状态；
- final metric 仍然在全部流程结束前隐藏，不破坏 frozen final protocol。

---

## 1. 当前用户最推荐流程：沿用已经冻结的 final split

你服务器当前已有：

```text
/root/autodl-tmp/RIFT_FEWREC_FINAL_FULL
```

其中已经生成：

```text
manifests_final/
cache/beats_final/
final_lock/FINAL_LOCK.json
```

**不要重新生成 split，不要重新 freeze。**

上传本工程到 `/root/autodl-tmp/` 后：

```bash
cd /root/autodl-tmp
unzip RIFT_FEWREC_FINAL_FAST_FULL.zip
cd RIFT_FEWREC_FINAL_FAST_FULL
```

迁移原来已经冻结的 final assets：

```bash
bash scripts/00_migrate_from_final.sh \
  /root/autodl-tmp/RIFT_FEWREC_FINAL_FULL \
  --delete-old
```

脚本会迁移：

```text
manifests_final
cache/beats_final
final_lock
third_party/beats
weights/BEATs_iter3_plus_AS2M.pt
```

并且重新执行 FINAL_LOCK hash verification。

原来被 Kill 的日志会保存在：

```text
legacy/killed_attempt/
```

不会迁移旧 `runs/final_test` 作为新结果。

---

## 2. 工程检查

```bash
bash scripts/01_smoke_test.sh
```

必须看到：

```text
SMOKE TEST PASSED ['S0', 'S1', 'PB', 'PT', 'F0']
checkpoint save/load/resume path PASS
FAST bootstrap == SLOW reference EXACT (tested all 3 metrics)
```

第三行非常重要：快速 bootstrap 已经和旧版逐循环参考实现做过数值一致性验证。

---

## 3. 从头重新跑 61 个 evaluation

你要求“从头跑”，所以新工程使用全新的：

```text
runs/final_fast/
```

启动：

```bash
bash scripts/05_run_final_fast_background.sh
```

实时看详细过程：

```bash
tail -f logs/final_fast.log
```

会显示：

```text
[EVAL 1/61] 2-rec seed=101 | draw=... | elapsed=... | ETA~... | checkpoint SAVED
[EVAL 2/61] 2-rec seed=102 | draw=... | elapsed=... | ETA~... | checkpoint SAVED
...
[EVAL 61/61] full-support-reference ... | checkpoint SAVED

[BOOT 1/12] 2-rec F0 vs S0 | 5000 resamples × 3 metrics | block=... | ETA~... | checkpoint SAVED
...
[BOOT 12/12] 4-rec F0 vs PT ...
```

注意：

**过程可见，但 final accuracy / F1 在全部完成前仍不会显示。**

---

## 4. 随时查看“已经算到哪里”

一次查看：

```bash
bash scripts/06_status.sh
```

持续查看：

```bash
watch -n 3 'bash scripts/06_status.sh'
```

示例：

```text
STATUS: EVALUATION
evaluation: 37/61
bootstrap: 0/12
current: 3-rec seed=117
elapsed: 08:42
ETA: 05:31
pid: 12345 alive
```

进入 bootstrap 后：

```text
STATUS: BOOTSTRAP
evaluation: 61/61
bootstrap: 7/12
current: 3-rec F0 vs PB
elapsed: 16:10
ETA: 00:02
```

---

## 5. 如果再次被平台 Killed

不要删除任何文件。

检查：

```bash
bash scripts/06_status.sh
```

然后：

```bash
bash scripts/05b_resume_final_fast.sh
```

它会自动：

```text
已完成 draw checkpoint -> 跳过
已完成 bootstrap block -> 跳过
未完成部分 -> 继续
```

例如已经完成：

```text
EVAL 43/61
```

重新启动后会从未完成的第 44 个任务继续，不会重算前 43 个。

如果已经进入：

```text
BOOT 8/12
```

则从剩余 4 个 comparison blocks 继续。

---

## 6. 为什么 bootstrap 会快很多

旧版每个 bootstrap replicate：

```text
5000 replicates
× 20 resampled support draws
× test components
× sklearn metric function
```

而且：

```text
3 settings × 4 comparisons × 3 metrics
= 36 次完整 bootstrap
```

新版先把 recording posterior 一次性聚合为 component posterior，然后预生成与旧版相同的 resample indices，再用 NumPy 批量计算 confusion statistics。

并且每个 comparison：

```text
同一组 5000 resamples
→ 同时得到 Macro-F1 / Balanced Accuracy / Accuracy
```

因此只有：

```text
3 settings × 4 comparisons = 12 个快速 blocks
```

**统计定义没有改变。**

本地合成 benchmark 中，`5000 × 20 draws × 30 components` 的一个 block 约为亚秒级；真实服务器速度会不同，但不会再出现旧版几十分钟的 Python/sklearn 内层循环。

---

## 7. 最终结束判断

只有出现：

```text
runs/final_fast/FINAL_COMPLETE.json
```

才算最终完成。

状态检查：

```bash
bash scripts/06_status.sh
```

应该显示：

```text
STATUS: COMPLETE
```

---

## 8. 查看结果

完成后：

```bash
bash scripts/07_show_results.sh
```

---

## 9. 生成论文表格

```bash
bash scripts/08_make_paper_tables.sh
```

输出：

```text
paper_outputs/FINAL_RESULTS.md
paper_outputs/FINAL_RESULTS.tex
```

---

## 10. 打包结果发给 ChatGPT

```bash
bash scripts/09_pack_results.sh
```

生成：

```text
fewrec_final_fast_results_YYYYMMDD_HHMMSS.zip
```

把这个 zip 发回来即可。

---

# 最终协议仍然不变

Methods：

```text
S0 Ordinary Segment ERM
S1 Recording-Balanced Segment Learning
PB Equal-Recording Prototype
PT Low-Rank Prototype Tuning
F0 Fixed Dual-Granularity Evidence Fusion
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

Few-recording：

```text
2 / 3 / 4 recordings per class
20 support draws each
seeds 101–120
```

Statistics：

```text
hierarchical paired bootstrap
5000 replicates
support-draw resampling
class-stratified near-duplicate-component resampling
```

这次优化的是**计算与恢复机制**，不是重新做方法选择。
