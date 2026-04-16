# Configurable QML Test

## Overview

- Created: `2026-04-16T15:27:32`
- Model: `HEA`
- Dataset: `two_moons`
- Loss: `BCE`
- Optimizers: `krotov_hybrid, adam`
- Seeds: `[0]`

## Summary

| Optimizer | Runs | Final loss | Final train acc | Final test acc | Wall time (s) | Cost units |
|---|---:|---:|---:|---:|---:|---:|
| Hybrid Krotov | 1 | 0.3255 ± 0.0000 | 0.8400 ± 0.0000 | 0.8133 ± 0.0000 | 43.99 ± 0.00 | 62850.0 ± 0.0 |
| Adam | 1 | 0.4354 ± 0.0000 | 0.8914 ± 0.0000 | 0.8867 ± 0.0000 | 33.16 ± 0.00 | 62850.0 ± 0.0 |

## Experiment Parameters

```json
{
  "timestamp": "2026-04-16T15:27:32",
  "command": "./run_configurable_qml_test.py --model hea --optimizers krotov_hybrid adam --loss-function bce --n-qubits 4 --n-layers 3 --max-iterations 40",
  "results_dir": "/Users/maximilianfrohlich/Desktop/GitHub/krotov/results_configurable_qml_tests/20260416_152732_hea_krotov_hybrid_adam",
  "dataset": "two_moons",
  "model_label": "HEA",
  "loss_label": "BCE",
  "args": {
    "model": "hea",
    "dataset": "auto",
    "loss_function": "bce",
    "optimizers": [
      "krotov_hybrid",
      "adam"
    ],
    "seeds": [
      0
    ],
    "run_name": null,
    "results_root": "/Users/maximilianfrohlich/Desktop/GitHub/krotov/results_configurable_qml_tests",
    "n_samples": 500,
    "noise": 0.15,
    "test_fraction": 0.3,
    "input_encoding": "tanh_0_pi",
    "perez_problem": "crown",
    "n_qubits": 4,
    "n_layers": 3,
    "entangler": "ring",
    "observable": "Z0Z1",
    "no_entanglement": false,
    "no_affine_head": false,
    "simonetti_sublayers": 4,
    "simonetti_entangler": "cnot_01",
    "chen_macro_layers": 2,
    "chen_encoding_axes": [
      "y"
    ],
    "chen_readout": "simple_z0",
    "souza_variant": "reduced",
    "souza_neurons": 4,
    "max_iterations": 40,
    "progress_interval": 5,
    "early_stopping": false,
    "early_stopping_patience": 12,
    "early_stopping_min_delta": 0.0001,
    "early_stopping_warmup": 20,
    "adam_lr": 0.05,
    "adam_batch_size": null,
    "adam_switch_iteration": 0,
    "lbfgs_maxiter": 40,
    "lbfgs_maxcor": 20,
    "lbfgs_gtol": 1e-07,
    "qng_lr": 0.5,
    "qng_lam": 0.01,
    "qng_approx": "full",
    "qng_batch_size": null,
    "qng_switch_iteration": 0,
    "krotov_step_size": 0.3,
    "krotov_lr_schedule": "constant",
    "krotov_decay": 0.05,
    "krotov_batch_size": null,
    "krotov_online_step_size": 0.3,
    "krotov_online_schedule": "constant",
    "krotov_online_decay": 0.05,
    "krotov_batch_step_size": 1.0,
    "krotov_batch_schedule": "constant",
    "krotov_batch_decay": 0.05,
    "hybrid_switch_iteration": 20,
    "hybrid_online_step_size": 0.3,
    "hybrid_batch_step_size": 1.0,
    "hybrid_online_schedule": "constant",
    "hybrid_batch_schedule": "constant",
    "hybrid_online_decay": 0.05,
    "hybrid_batch_decay": 0.05,
    "hybrid_scaling_mode": "none",
    "hybrid_scaling_apply_phase": "both",
    "hybrid_scaling_config": null
  },
  "config": {
    "n_samples": 500,
    "moon_noise": 0.15,
    "test_fraction": 0.3,
    "input_encoding": "tanh_0_pi",
    "model_architecture": "hea",
    "n_qubits": 4,
    "n_layers": 3,
    "entangler": "ring",
    "observable": "Z0Z1",
    "max_iterations": 40,
    "adam_lr": 0.05,
    "adam_batch_size": null,
    "adam_switch_iteration": 0,
    "lbfgs_maxiter": 40,
    "lbfgs_maxcor": 20,
    "lbfgs_gtol": 1e-07,
    "qng_lr": 0.5,
    "qng_batch_size": null,
    "qng_switch_iteration": 0,
    "qng_lam": 0.01,
    "qng_approx": null,
    "early_stopping_enabled": false,
    "early_stopping_patience": 12,
    "early_stopping_min_delta": 0.0001,
    "early_stopping_warmup": 20,
    "krotov_step_size": 0.3,
    "krotov_lr_schedule": "constant",
    "krotov_decay": 0.05,
    "krotov_batch_size": null,
    "krotov_target_loss": 0.4,
    "krotov_online_step_size": 0.3,
    "krotov_online_schedule": "constant",
    "krotov_online_decay": 0.05,
    "krotov_batch_step_size": 1.0,
    "krotov_batch_schedule": "constant",
    "krotov_batch_decay": 0.05,
    "hybrid_switch_iteration": 20,
    "hybrid_online_step_size": 0.3,
    "hybrid_batch_step_size": 1.0,
    "hybrid_online_schedule": "constant",
    "hybrid_batch_schedule": "constant",
    "hybrid_online_decay": 0.05,
    "hybrid_batch_decay": 0.05,
    "hybrid_scaling_mode": "none",
    "hybrid_scaling_apply_phase": "both",
    "hybrid_scaling_config": null,
    "seeds": [
      0
    ],
    "optimizers": [
      "krotov_hybrid",
      "adam"
    ],
    "run_krotov_batch_sweep": false,
    "krotov_batch_sweep_step_sizes": [
      1.0,
      0.3,
      0.1,
      0.05,
      0.02
    ],
    "krotov_batch_sweep_schedules": [
      "constant",
      "inverse"
    ],
    "run_krotov_hybrid_sweep": false,
    "hybrid_switch_iterations": [
      5,
      10,
      20,
      30,
      50
    ],
    "loss_threshold": 0.4,
    "loss_thresholds": [
      0.45,
      0.4,
      0.38,
      0.36
    ],
    "results_dir": "results_configurable_qml_tests",
    "plots_dir": "plots"
  },
  "n_result_files": 2
}
```

## Generated Files

- `config.json`
- `summary.json`
- `loss_vs_iteration.pdf` and `loss_vs_iteration.png`
- `loss_vs_time.pdf` and `loss_vs_time.png`
- `test_accuracy_vs_iteration.pdf` and `test_accuracy_vs_iteration.png`
- `test_accuracy_vs_time.pdf` and `test_accuracy_vs_time.png`
- `result_<optimizer>_seed<seed>.json` for each run

