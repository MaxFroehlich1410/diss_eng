# Perez-Salinas Sweep

## Overview

Hyperparameter sweeps for the Perez-Salinas style benchmark.

## Run

```bash
python -m experiments.two_moons_perez_salinas_sweep.run --optimizer adam
python -m experiments.two_moons_perez_salinas_sweep.run --optimizer krotov_hybrid
```

## Outputs

Generated artifacts are written under `results/` and are ignored by git.
