# RIFT_DEEPSHIP_50CLASS_V3_FULL

这是 DeepShip 第二数据集的**独立外部验证工程 V3**。

这版修正了 V2 的核心问题：**声学相似度不再用于判断两条官方 recording/session 是否是同一条 recording。**

## 最终固定的数据协议

每类固定使用：

```text
50 official recording/session units
├─ 30 support pool
└─ 20 frozen test
```

四类共：

```text
Cargo           50
Passengership   50
Tanker          50
Tug             50
------------------
Total           200 recordings
```

### Hard duplicate：只用真正确定的 exact evidence

只有以下情况会被视为同一个 hard duplicate component：

```text
完整 recording/session 的 SHA256 完全相同
或
不同 recording/session 共享字节完全相同的 WAV SHA256
```

**不会**因为下面这些原因把两条 recording 合并：

```text
同一天
频谱很像
acoustic fingerprint cosine >= 0.997
```

### 日期的作用

同一天的多个官方 recordings 仍然是多条 recording。

日期只用于软相关性控制：

1. 30/20 support-test split 时尽量让同一天的选中 recordings 在同一侧；
2. 2/3/4-rec support draw 时，若日期信息充足，尽量不在同一个 draw 中抽两条同一天 recording。

Cargo 顶层数字 WAV 没有日期信息时，自动按普通 recording 抽样。

### Acoustic fingerprint 的作用

V3 仍会给选中的 200 条 recording 计算 acoustic fingerprint，但**只生成诊断文件**：

```text
manifests_deepship_50/acoustic_similarity_diagnostics.json
```

它会报告：

- 每类相似度分布；
- 最大/中位数/P95/P99；
- cosine >= 0.997 的 pair 数量；
- 跨 support/test 的高相似 pair；
- top similar pairs。

这些信息**不会重新合并 recording，也不会重新划分 support/test**。

这正是 V3 与 V2 的关键区别。

---

# 方法完全沿用 ShipsEar 锁定版本

```text
S0  ordinary segment ERM
S1  recording-balanced segment learning
PB  equal-recording prototype
PT  low-rank prototype tuning
F0  fixed geometric fusion
```

锁定参数：

```text
PT rank = 2
steps = 160
lr = 0.05
l2 = 0.3
temperature = 0.25

F0 alpha = 0.5
```

DeepShip 不用于重新调参。

few-recording 设置：

```text
2-rec/class × 20 draws
3-rec/class × 20 draws
4-rec/class × 20 draws
full-support reference
```

---

# 执行流程

假设服务器已有：

```text
/root/autodl-tmp/DeepShip
/root/autodl-tmp/RIFT_FEWREC_FINAL_FAST_FULL
```

## 1. 解压

```bash
cd /root/autodl-tmp
unzip RIFT_DEEPSHIP_50CLASS_V3_FULL.zip
cd RIFT_DEEPSHIP_50CLASS_V3_FULL
```

## 2. 物理复制 BEATs 资产

不使用软链接：

```bash
bash scripts/00_copy_assets_from_shipsear.sh \
  /root/autodl-tmp/RIFT_FEWREC_FINAL_FAST_FULL
```

## 3. smoke test

```bash
bash scripts/01_smoke_test.sh
```

## 4. 数据审计

```bash
bash scripts/02_audit_dataset.sh
```

你的当前 DeepShip 已审计为接近完整：

```text
Cargo          109
Passengership  170
Tanker         240
Tug             69
Total          588 recordings
Duration       ~43.84 h
```

## 5. 生成 V3 固定 50/class split

```bash
bash scripts/03_prepare_external_split.sh
```

这一步会：

```text
588 raw recordings
↓
exact SHA256 deduplication（很快）
↓
每类固定随机选 50 条 official recordings
↓
日期 soft-blocking
↓
30/class support + 20/class frozen test
↓
只对选中的 200 条做 acoustic similarity diagnostic
```

成功后查看：

```bash
cat manifests_deepship_50/split_summary.json
cat manifests_deepship_50/leakage_audit.json
cat manifests_deepship_50/date_split_diagnostics.json
```

硬泄漏必须满足：

```text
recording_id_overlap = 0
wav_sha256_overlap = 0
exact_component_overlap = 0
hard_leakage_clean = true
```

声学诊断可查看：

```bash
cat manifests_deepship_50/acoustic_similarity_diagnostics.json
```

注意：其中即使存在很多 cosine >= 0.997 的 pair，也只是诊断，**不再把它们判成相同 recording**。

## 6. 建立 BEATs cache

```bash
bash scripts/04_build_cache.sh
```

只缓存最终选中的：

```text
120 support recordings
80 test recordings
```

## 7. 冻结 DeepShip 协议

```bash
bash scripts/05_freeze_protocol.sh
```

应看到：

```text
DEEPSHIP V3 LOCK VERIFIED
```

## 8. 正式运行

```bash
bash scripts/06_run_background.sh
```

看过程：

```bash
tail -f logs/deepship_final.log
```

或者：

```bash
watch -n 3 'bash scripts/07_status.sh'
```

会显示：

```text
EVAL x/61
BOOT x/18
elapsed
ETA
checkpoint
```

如果服务器进程被 kill：

```bash
bash scripts/06b_resume.sh
```

会从 checkpoint 继续。

## 9. 完成后

```bash
bash scripts/08_show_results.sh
bash scripts/09_make_paper_tables.sh
bash scripts/10_pack_results.sh
```

最后得到：

```text
deepship_fewrec_results_YYYYMMDD_HHMMSS.zip
```

把这个 ZIP 发给我即可。

---

# 论文重点看什么

第二数据集最重要的是验证：

```text
S1 vs S0
```

即 ShipsEar 上的 recording-balanced learning 是否在 DeepShip 也保持正增益。

其次：

```text
PT vs PB
F0 vs S1/PT
```

DeepShip 结果出来以后不再依据 test 结果修改 ShipsEar 锁定的方法。
