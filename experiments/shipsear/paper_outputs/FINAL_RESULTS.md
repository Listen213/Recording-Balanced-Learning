# Final ShipsEar Results

## Macro-F1 (%)

| Method | 2-rec | 3-rec | 4-rec | full-support-reference |
|---|---:|---:|---:|---:|
| Segment ERM | 43.45±12.18 | 52.58±7.06 | 53.36±8.29 | 66.14±0.00 |
| Recording-balanced | 48.81±9.55 | 57.45±7.30 | 57.72±8.18 | 77.83±0.00 |
| Equal-recording Proto | 49.07±8.88 | 57.27±6.16 | 59.74±5.98 | 69.67±0.00 |
| Low-rank ProtoTune | 50.78±9.65 | 57.89±6.38 | 59.23±5.59 | 69.67±0.00 |
| DGE Fusion (Proposed) | 50.71±8.98 | 56.95±5.66 | 59.44±7.65 | 72.95±0.00 |

## Balanced Accuracy (%)

| Method | 2-rec | 3-rec | 4-rec | full-support-reference |
|---|---:|---:|---:|---:|
| Segment ERM | 46.14±12.19 | 54.67±7.97 | 56.26±8.05 | 64.33±0.00 |
| Recording-balanced | 51.44±10.00 | 59.36±7.84 | 61.18±7.82 | 77.56±0.00 |
| Equal-recording Proto | 53.01±9.69 | 60.71±6.23 | 64.20±6.41 | 70.89±0.00 |
| Low-rank ProtoTune | 54.90±10.34 | 61.02±6.42 | 63.31±5.56 | 70.89±0.00 |
| DGE Fusion (Proposed) | 54.01±10.38 | 59.11±5.99 | 63.79±7.74 | 73.11±0.00 |

## Proposed-vs-baseline hierarchical bootstrap

| Setting | Comparison | Metric | Δ mean (pp) | 95% CI (pp) | P(Δ≤0) |
|---|---|---|---:|---:|---:|
| 2-rec | F0_minus_S0 | macro_f1 | 7.50 | [1.98, 13.41] | 0.0026 |
| 2-rec | F0_minus_S0 | balanced_accuracy | 7.91 | [2.44, 14.38] | 0.0020 |
| 2-rec | F0_minus_S0 | accuracy | 6.43 | [1.00, 12.20] | 0.0092 |
| 2-rec | F0_minus_S1 | macro_f1 | 1.77 | [-1.25, 5.22] | 0.1298 |
| 2-rec | F0_minus_S1 | balanced_accuracy | 2.33 | [-0.64, 5.91] | 0.0674 |
| 2-rec | F0_minus_S1 | accuracy | 1.81 | [-1.20, 5.40] | 0.1304 |
| 2-rec | F0_minus_PB | macro_f1 | 1.98 | [-2.52, 6.23] | 0.1764 |
| 2-rec | F0_minus_PB | balanced_accuracy | 1.22 | [-3.91, 5.77] | 0.2860 |
| 2-rec | F0_minus_PB | accuracy | 1.16 | [-3.60, 5.40] | 0.2816 |
| 2-rec | F0_minus_PT | macro_f1 | 0.41 | [-3.90, 4.43] | 0.4122 |
| 2-rec | F0_minus_PT | balanced_accuracy | -0.45 | [-4.94, 3.79] | 0.5834 |
| 2-rec | F0_minus_PT | accuracy | -0.46 | [-4.40, 3.40] | 0.5942 |
| 3-rec | F0_minus_S0 | macro_f1 | 4.42 | [-1.51, 10.17] | 0.0718 |
| 3-rec | F0_minus_S0 | balanced_accuracy | 4.40 | [-1.10, 9.86] | 0.0594 |
| 3-rec | F0_minus_S0 | accuracy | 4.55 | [-1.20, 10.40] | 0.0608 |
| 3-rec | F0_minus_S1 | macro_f1 | -0.97 | [-4.03, 2.29] | 0.7420 |
| 3-rec | F0_minus_S1 | balanced_accuracy | -0.74 | [-3.72, 2.08] | 0.6970 |
| 3-rec | F0_minus_S1 | accuracy | -1.00 | [-4.20, 2.20] | 0.7274 |
| 3-rec | F0_minus_PB | macro_f1 | 0.29 | [-3.13, 3.54] | 0.4222 |
| 3-rec | F0_minus_PB | balanced_accuracy | -1.10 | [-4.17, 1.59] | 0.7718 |
| 3-rec | F0_minus_PB | accuracy | -0.18 | [-4.00, 3.20] | 0.5302 |
| 3-rec | F0_minus_PT | macro_f1 | -0.18 | [-3.62, 2.93] | 0.5310 |
| 3-rec | F0_minus_PT | balanced_accuracy | -1.17 | [-4.30, 1.40] | 0.8000 |
| 3-rec | F0_minus_PT | accuracy | -0.60 | [-3.80, 2.40] | 0.6510 |
| 4-rec | F0_minus_S0 | macro_f1 | 4.47 | [-0.77, 10.14] | 0.0516 |
| 4-rec | F0_minus_S0 | balanced_accuracy | 5.06 | [-0.14, 10.58] | 0.0290 |
| 4-rec | F0_minus_S0 | accuracy | 4.41 | [-2.20, 11.00] | 0.0952 |
| 4-rec | F0_minus_S1 | macro_f1 | 0.20 | [-2.26, 2.66] | 0.4252 |
| 4-rec | F0_minus_S1 | balanced_accuracy | 0.34 | [-1.80, 2.70] | 0.3790 |
| 4-rec | F0_minus_S1 | accuracy | 0.18 | [-2.80, 3.20] | 0.4558 |
| 4-rec | F0_minus_PB | macro_f1 | -0.24 | [-3.23, 2.79] | 0.5690 |
| 4-rec | F0_minus_PB | balanced_accuracy | -0.39 | [-3.67, 2.51] | 0.6092 |
| 4-rec | F0_minus_PB | accuracy | -0.38 | [-3.01, 2.20] | 0.6152 |
| 4-rec | F0_minus_PT | macro_f1 | 0.38 | [-2.61, 3.70] | 0.4100 |
| 4-rec | F0_minus_PT | balanced_accuracy | 0.49 | [-2.33, 3.67] | 0.3760 |
| 4-rec | F0_minus_PT | accuracy | 0.22 | [-2.40, 2.80] | 0.4388 |
