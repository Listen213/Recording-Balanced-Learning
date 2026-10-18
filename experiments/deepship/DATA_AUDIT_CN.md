# DeepShip V3 数据说明

当前数据审计：

```text
Cargo          109
Passengership  170
Tanker         240
Tug             69
Total          588
Duration       43.84 h
```

V2 报错的原因不是数据不够，而是 acoustic fingerprint 被错误地用成 hard duplicate identity，导致 Tug 69 条被链式合并成 17 个 component。

V3 已彻底取消这一规则。

V3 只对 SHA256 完全相同的数据做 hard deduplication；acoustic fingerprint 只做诊断。
