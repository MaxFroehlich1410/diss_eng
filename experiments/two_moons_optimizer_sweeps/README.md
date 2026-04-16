# Two-Moons Optimizer Sweeps

## Overview

Hyperparameter sweeps for optimizer families on the two-moons benchmark.

## Run

```bash
python -m experiments.two_moons_optimizer_sweeps.run --optimizer adam
python -m experiments.two_moons_optimizer_sweeps.run --optimizer qng
python -m experiments.two_moons_optimizer_sweeps.run --optimizer krotov_hybrid
```

## Outputs

Generated artifacts are written under `results/` and are ignored by git.
