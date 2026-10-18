# 与 ShipsEar 工程的物理分离

本工程不会创建软链接。

执行：

```bash
bash scripts/00_copy_assets_from_shipsear.sh \
  /root/autodl-tmp/RIFT_FEWREC_FINAL_FAST_FULL
```

会物理复制：

```text
third_party/beats/
weights/BEATs_iter3_plus_AS2M.pt
```

使用 `cp -aL` 解引用复制，并检查新工程内不存在 symlink。

DeepShip manifests、cache、run results、lock 都只保存在本工程中。
