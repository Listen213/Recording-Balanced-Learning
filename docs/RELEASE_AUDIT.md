# Release preparation audit

- **shipsear**: kept 187 files, excluded 23 archive entries; input `RIFT_FEWREC_FINAL_FAST_FULL(6).zip`.
- **deepship**: kept 197 files, excluded 24 archive entries; input `RIFT_DEEPSHIP_50CLASS_V3_FULL(5).zip`.
- **uniqueship**: kept 36 files, excluded 40 archive entries; input `RBL_UNIQUESHIP_8CLASS_FINAL_CLEAN(5).zip`.

- Python source syntax checks: 54 files; errors: 0.
- Basic token / private key pattern scan hits: 0.
- Exclusions: Python bytecode, logs, PID files, nested ZIP archives, locally cached embeddings, model weights, vendorized BEATs third-party source.
- GPU training / model reproduction: **not executed**.
- Manual review required: dataset licenses, any metadata embedded in split manifests, all manuscript-to-result correspondences, final author code license.

