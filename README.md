# Krotov QML Benchmark Repository

This repository is organised around four main areas:

- `datasets/` contains reusable classification dataset loaders and synthetic task generators, including Iris, two-moons, and Perez-Salinas style problems such as `crown`.
- `qml_models/` contains reusable QML model definitions and objective functions.
- `optimizers/` contains reusable optimizer implementations, including the hybrid Krotov method and its baselines.
- `experiments/` contains runnable experiment folders, each with its own `README.md`, runner script, and `results/` directory.

## Main Experiments

- `experiments/two_moons_baseline/`: baseline two-moons benchmark.
- `experiments/two_moons_optimizer_sweeps/`: optimizer hyperparameter sweeps on two-moons variants.
- `experiments/two_moons_hybrid_sweep/`: hybrid Krotov schedule sweep.
- `experiments/two_moons_scaling_sweep/`: Krotov scaling experiments.
- `experiments/two_moons_alternative_models/`: alternative QML model benchmarks.
- `experiments/two_moons_perez_salinas_benchmark/`: Perez-Salinas style data-reuploading benchmark.
- `experiments/two_moons_perez_salinas_sweep/`: Perez-Salinas sweep experiments.
- `experiments/two_moons_readout_heads_hea/`, `two_moons_readout_heads_chen/`, `two_moons_readout_heads_simonetti/`, `two_moons_readout_heads_hea_two_moons/`: readout-head comparisons.
- `experiments/two_moons_dense_angle/`: dense-angle circuit benchmark.
- `experiments/two_moons_krotov_variants/`: Krotov variant comparison.
- `experiments/iris_hea/`: Iris HEA benchmark.
- `experiments/vqe_hubbard_1x2/`: exact-statevector 1x2 Fermi-Hubbard VQE sweeps.

Generated outputs live under each experiment's `results/` directory and are ignored by git.

## Install

```bash
pip install -r requirements.txt
```

## Notes

The manuscript source files currently remain under `experiments/krotov_paper/` as archival material.
