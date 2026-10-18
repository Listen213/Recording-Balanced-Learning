# Final UniqueShip protocol

- Dataset: UniqueShip `12 Class - 5 Hours Each`, official Fold 1.
- Background excluded because it has `MMSI=0` and is not a vessel identity.
- Eligibility is metadata-only and fixed before model results: at least 20 train MMSIs and at least 10 test MMSIs.
- Eligible classes: cargo, fishing, other, passengership, pleasurecraft, sailing, tanker, tug.
- Support pool: 20 train vessels/class selected deterministically with seed 20261016 (160 vessels total).
- Test: entire official Fold-1 test set for the 8 eligible classes (524 vessels in the observed v1.0 metadata).
- Support regimes: K = 2/3/4 vessels/class, 100 paired support draws.
- Frozen representation: BEATs_iter3_plus_AS2M, 16 kHz input, official 5-s clips.
- Primary comparison: S0 vs S1.
- Mechanism control: ClassBal.
- Evaluation unit: vessel/MMSI. Segment posteriors are averaged within vessel.
- Main uncertainty: paired hierarchical bootstrap over support draws and test vessels, 5000 iterations.

The observed 5-hours-each release has near-uniform clip counts among vessels within many classes. Therefore the finalized interpretation is intentionally diagnostic: if S1 ≈ ClassBal, UniqueShip acts as an identity-explicit low-within-class-imbalance control rather than evidence of a large extra vessel-balancing effect.
