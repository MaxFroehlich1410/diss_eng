# Hybrid Krotov Hyperparameter Sweep Report

## Overview

This sweep searches over three hyperparameters of the hybrid Krotov optimizer:
- `hybrid_switch_iteration`: when to switch from online to batch phase
- `hybrid_online_step_size`: step size during the online phase
- `hybrid_batch_step_size`: step size during the batch phase

Three models are tested, each with 3 random seeds per configuration.

## HEA (Iris)

- Grid: switch=[3, 5, 10, 20], online=[0.1, 0.3, 0.5, 1.0], batch=[0.5, 1.0, 2.0, 3.0]
- Total configs: 64
- Seeds per config: 3
- Loss threshold: 0.4

### Best configuration

| Parameter | Value |
|---|---|
| switch_iteration | 20 |
| online_step_size | 1.0 |
| batch_step_size | 3.0 |
| **Mean final loss** | **0.2424 ± 0.015** |
| Mean test accuracy | 0.983 ± 0.024 |
| Mean wall time | 14.4s |
| Success rate (≤0.4) | 1.00 |
| Tail loss std | 0.0000 |

### Compared to current baseline (sw=10, on=0.3, bat=1.0)

| Metric | Baseline | Best | Δ |
|---|---|---|---|
| Final loss | 0.2636 | 0.2424 | +0.0212 (+8.0%) |
| Test accuracy | 1.000 | 0.983 | -0.017 |
| Wall time | 13.2s | 14.4s | +1.2s |

### Top 5 configurations

| Rank | switch | online | batch | loss (mean±std) | test acc | wall (s) |
|---|---|---|---|---|---|---|
| 1 | 20 | 1.0 | 3.0 | 0.2424±0.015 | 0.983 | 14.4 |
| 2 | 20 | 1.0 | 2.0 | 0.2424±0.015 | 0.983 | 15.0 |
| 3 | 20 | 1.0 | 1.0 | 0.2424±0.015 | 0.983 | 14.4 |
| 4 | 20 | 1.0 | 0.5 | 0.2424±0.015 | 0.983 | 14.8 |
| 5 | 10 | 1.0 | 3.0 | 0.2519±0.015 | 0.983 | 13.9 |

