# DeepShip External Validation

## Macro-F1 (%)

| Method | 2-rec | 3-rec | 4-rec | full-support-reference |
|---|---:|---:|---:|---:|
| Segment ERM | 37.35±6.13 | 39.24±6.18 | 45.62±7.61 | 58.97±0.00 |
| Recording-balanced | 43.15±5.78 | 47.80±6.76 | 53.93±6.95 | 71.09±0.00 |
| Equal-recording Proto | 41.95±6.49 | 47.35±6.58 | 53.97±6.19 | 63.33±0.00 |
| Low-rank ProtoTune | 41.91±6.43 | 47.83±5.99 | 54.93±6.11 | 67.43±0.00 |
| DGE Fusion | 43.61±6.22 | 48.64±6.37 | 55.05±5.76 | 66.42±0.00 |

## Balanced Accuracy (%)

| Method | 2-rec | 3-rec | 4-rec | full-support-reference |
|---|---:|---:|---:|---:|
| Segment ERM | 44.19±4.02 | 44.94±4.67 | 50.19±5.93 | 61.25±0.00 |
| Recording-balanced | 46.69±4.32 | 49.56±6.50 | 55.00±6.20 | 71.25±0.00 |
| Equal-recording Proto | 44.50±5.40 | 48.62±6.58 | 54.44±5.87 | 63.75±0.00 |
| Low-rank ProtoTune | 44.50±5.28 | 48.81±5.90 | 55.19±5.89 | 67.50±0.00 |
| DGE Fusion | 46.50±4.91 | 49.62±5.86 | 55.69±5.32 | 66.25±0.00 |

## Hierarchical paired bootstrap

| Setting | Comparison | Metric | Δ mean (pp) | 95% CI (pp) | P(Δ≤0) |
|---|---|---|---:|---:|---:|
| 2-rec | S1_minus_S0 | macro_f1 | 5.68 | [2.03, 9.44] | 0.0014 |
| 2-rec | S1_minus_S0 | balanced_accuracy | 2.48 | [-0.69, 5.69] | 0.0588 |
| 2-rec | S1_minus_S0 | accuracy | 2.48 | [-0.69, 5.69] | 0.0590 |
| 2-rec | PT_minus_PB | macro_f1 | -0.03 | [-1.21, 1.16] | 0.5196 |
| 2-rec | PT_minus_PB | balanced_accuracy | 0.01 | [-1.19, 1.25] | 0.5034 |
| 2-rec | PT_minus_PB | accuracy | 0.01 | [-1.19, 1.25] | 0.5042 |
| 2-rec | F0_minus_S0 | macro_f1 | 6.17 | [1.51, 10.71] | 0.0052 |
| 2-rec | F0_minus_S0 | balanced_accuracy | 2.32 | [-1.94, 6.56] | 0.1420 |
| 2-rec | F0_minus_S0 | accuracy | 2.32 | [-1.94, 6.56] | 0.1422 |
| 2-rec | F0_minus_S1 | macro_f1 | 0.46 | [-2.12, 3.25] | 0.3704 |
| 2-rec | F0_minus_S1 | balanced_accuracy | -0.18 | [-2.69, 2.44] | 0.5590 |
| 2-rec | F0_minus_S1 | accuracy | -0.18 | [-2.69, 2.44] | 0.5596 |
| 2-rec | F0_minus_PB | macro_f1 | 1.66 | [-1.28, 4.75] | 0.1352 |
| 2-rec | F0_minus_PB | balanced_accuracy | 2.02 | [-0.69, 4.94] | 0.0726 |
| 2-rec | F0_minus_PB | accuracy | 2.02 | [-0.69, 4.94] | 0.0724 |
| 2-rec | F0_minus_PT | macro_f1 | 1.66 | [-0.95, 4.52] | 0.1076 |
| 2-rec | F0_minus_PT | balanced_accuracy | 1.98 | [-0.37, 4.56] | 0.0508 |
| 2-rec | F0_minus_PT | accuracy | 1.98 | [-0.38, 4.56] | 0.0504 |
| 3-rec | S1_minus_S0 | macro_f1 | 8.52 | [3.81, 13.14] | 0.0000 |
| 3-rec | S1_minus_S0 | balanced_accuracy | 4.64 | [0.56, 8.75] | 0.0126 |
| 3-rec | S1_minus_S0 | accuracy | 4.64 | [0.56, 8.75] | 0.0124 |
| 3-rec | PT_minus_PB | macro_f1 | 0.46 | [-1.45, 2.57] | 0.3296 |
| 3-rec | PT_minus_PB | balanced_accuracy | 0.17 | [-1.62, 2.06] | 0.4294 |
| 3-rec | PT_minus_PB | accuracy | 0.17 | [-1.62, 2.06] | 0.4316 |
| 3-rec | F0_minus_S0 | macro_f1 | 9.34 | [4.02, 14.81] | 0.0004 |
| 3-rec | F0_minus_S0 | balanced_accuracy | 4.68 | [0.00, 9.50] | 0.0250 |
| 3-rec | F0_minus_S0 | accuracy | 4.68 | [0.00, 9.50] | 0.0248 |
| 3-rec | F0_minus_S1 | macro_f1 | 0.88 | [-1.50, 3.37] | 0.2394 |
| 3-rec | F0_minus_S1 | balanced_accuracy | 0.09 | [-2.25, 2.50] | 0.4716 |
| 3-rec | F0_minus_S1 | accuracy | 0.09 | [-2.25, 2.50] | 0.4742 |
| 3-rec | F0_minus_PB | macro_f1 | 1.32 | [-1.54, 4.22] | 0.1810 |
| 3-rec | F0_minus_PB | balanced_accuracy | 1.02 | [-1.75, 3.88] | 0.2394 |
| 3-rec | F0_minus_PB | accuracy | 1.02 | [-1.75, 3.88] | 0.2384 |
| 3-rec | F0_minus_PT | macro_f1 | 0.84 | [-1.98, 3.76] | 0.2718 |
| 3-rec | F0_minus_PT | balanced_accuracy | 0.83 | [-1.94, 3.69] | 0.2722 |
| 3-rec | F0_minus_PT | accuracy | 0.83 | [-1.94, 3.69] | 0.2740 |
| 4-rec | S1_minus_S0 | macro_f1 | 8.31 | [4.70, 11.99] | 0.0000 |
| 4-rec | S1_minus_S0 | balanced_accuracy | 4.79 | [1.50, 8.19] | 0.0036 |
| 4-rec | S1_minus_S0 | accuracy | 4.79 | [1.50, 8.19] | 0.0034 |
| 4-rec | PT_minus_PB | macro_f1 | 0.98 | [-0.81, 2.87] | 0.1446 |
| 4-rec | PT_minus_PB | balanced_accuracy | 0.76 | [-1.00, 2.56] | 0.1990 |
| 4-rec | PT_minus_PB | accuracy | 0.76 | [-1.00, 2.56] | 0.2000 |
| 4-rec | F0_minus_S0 | macro_f1 | 9.42 | [5.01, 13.95] | 0.0000 |
| 4-rec | F0_minus_S0 | balanced_accuracy | 5.47 | [1.37, 9.56] | 0.0034 |
| 4-rec | F0_minus_S0 | accuracy | 5.47 | [1.38, 9.56] | 0.0034 |
| 4-rec | F0_minus_S1 | macro_f1 | 1.11 | [-1.29, 3.77] | 0.1894 |
| 4-rec | F0_minus_S1 | balanced_accuracy | 0.68 | [-1.50, 3.00] | 0.2790 |
| 4-rec | F0_minus_S1 | accuracy | 0.68 | [-1.50, 3.00] | 0.2784 |
| 4-rec | F0_minus_PB | macro_f1 | 1.11 | [-1.35, 3.45] | 0.1784 |
| 4-rec | F0_minus_PB | balanced_accuracy | 1.26 | [-0.94, 3.38] | 0.1258 |
| 4-rec | F0_minus_PB | accuracy | 1.26 | [-0.94, 3.37] | 0.1254 |
| 4-rec | F0_minus_PT | macro_f1 | 0.14 | [-2.39, 2.54] | 0.4430 |
| 4-rec | F0_minus_PT | balanced_accuracy | 0.51 | [-1.81, 2.75] | 0.3258 |
| 4-rec | F0_minus_PT | accuracy | 0.51 | [-1.81, 2.75] | 0.3274 |
