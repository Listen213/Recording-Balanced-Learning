# 迁移与资源说明

当前推荐路径不是重新下载/重新生成 final split，而是从上一版：

```text
/root/autodl-tmp/RIFT_FEWREC_FINAL_FULL
```

迁移已经冻结的：

```text
manifests_final/
cache/beats_final/
final_lock/
third_party/beats/
weights/BEATs_iter3_plus_AS2M.pt
```

执行：

```bash
bash scripts/00_migrate_from_final.sh \
  /root/autodl-tmp/RIFT_FEWREC_FINAL_FULL \
  --delete-old
```

迁移脚本会在删除旧项目之前重新执行：

```bash
python verify_final_lock.py --lock final_lock/FINAL_LOCK.json
```

如果 hash 不一致，脚本会直接停止，旧工程不会删除。

本 zip 不包含大型 BEATs checkpoint、raw ShipsEar WAV 或 frozen final cache。
