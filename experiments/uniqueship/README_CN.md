# UniqueShip 8-class FINAL CLEAN

这是**只负责 UniqueShip 第三数据集**的最终干净工程。不要在这里跑 ShipsEar/DeepShip 补实验。

## 最终只保留三件事
1. S0 vs S1：2/3/4-vessel × 100 draws；
2. ClassBal mechanism control；
3. 5000 次 paired hierarchical bootstrap（同时给 S1-S0、ClassBal-S0、S1-ClassBal）。

不再在 UniqueShip 上浪费时间跑 sampler / 多 head / scaler 等大批 controls；这些放到另一个补充实验工程里做 ShipsEar + DeepShip。

## 服务器固定路径
- 数据：`/root/autodl-tmp/UniqueShip/allclass_5hours`
- 原 BEATs 工程：`/root/autodl-tmp/RIFT_FEWREC_FINAL_FAST_FULL`

## 如果旧 V3 还在：优先迁移，避免重新提 BEATs
先把这个工程解压，然后：
```bash
cd /root/autodl-tmp/RBL_UNIQUESHIP_8CLASS_FINAL_CLEAN
bash scripts/migrate_from_v3.sh
```
检查：
```bash
ls -lh cache/
find outputs -maxdepth 2 -type f | sort
```
如果已经有 primary + classbal 的 metrics/predictions，直接：
```bash
bash run_analysis_only.sh 2>&1 | tee logs/final_analysis.log
```
这样**不用重新跑 BEATs，也不用重新跑 100 draws**。

## 如果旧 V3 已经删了：从头跑
```bash
cd /root/autodl-tmp/RBL_UNIQUESHIP_8CLASS_FINAL_CLEAN
pip install -r requirements.txt
bash run_from_scratch.sh 2>&1 | tee logs/from_scratch.log
```

## 最终结果
核心目录：
```text
outputs/final_summary/
  method_summary.csv
  full_support.csv
  paired_draw_bootstrap.csv
  hierarchical_bootstrap_all_pairs.csv
```

打包：
```bash
bash pack_results.sh
```
生成 `UniqueShip_8class_FINAL_results.zip`。
