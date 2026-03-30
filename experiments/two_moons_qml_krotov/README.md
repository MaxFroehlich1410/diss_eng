# Two-Moons QML Optimizer Benchmark: Krotov vs Baselines

## Overview

Noiseless variational quantum classifier benchmark on the two-moons dataset,
comparing a Krotov sequential gate-update optimizer against Adam and L-BFGS-B.

## Model

- **Qubits:** 4
- **Encoding:** `Ry(x1)` on q0,q2; `Ry(x2)` on q1,q3 (feature re-upload)
- **Ansatz:** 3-layer hardware-efficient ansatz (HEA)
  - Each layer: `Ry(θ)·Rz(φ)` per qubit, then CNOT ring `(0,1),(1,2),(2,3),(3,0)`
- **Trainable parameters:** 24 (4 qubits × 2 rotations × 3 layers)
- **Observable:** `z = 0.5·(⟨Z₀⟩ + ⟨Z₁⟩)`
- **Output:** `p = clip((z+1)/2, ε, 1-ε)`
- **Loss:** Binary cross-entropy
- **Simulation:** Exact statevector (noiseless, no shots)
- **Gradients:** Parameter-shift rule (exact)

## Dataset

- **Source:** `sklearn.datasets.make_moons`
- **Samples:** 500 (350 train / 150 test)
- **Noise:** 0.15
- **Preprocessing:** StandardScaler → tanh mapping to [0, π]

## Optimizers

| Optimizer | Description | Key hyperparameter |
|-----------|-------------|-------------------|
| **Krotov** | Sequential gate-by-gate parameter update using forward/backward propagation. Each sample triggers a sweep through all parameterized gates, updating each one using the current (updated) forward state and the old backward co-state. | step_size = 0.3 |
| **Adam** | Standard Adam with parameter-shift gradients (full-batch). | lr = 0.05 |
| **L-BFGS-B** | Quasi-Newton method via `scipy.optimize.minimize` with parameter-shift gradients. | default L-BFGS-B settings |

All optimizers use the same initial parameters per seed and the same dataset split.

## Results (10 seeds, 100 iterations)

| Optimizer | Final Loss (mean±std) | Train Acc | Test Acc | Wall Time |
|-----------|-----------------------|-----------|----------|-----------|
| **Adam** | 0.349 ± 0.034 | 0.847 ± 0.020 | 0.858 ± 0.030 | 97.6s |
| **Krotov** | 0.424 ± 0.098 | 0.835 ± 0.053 | 0.852 ± 0.036 | 143.6s |
| **L-BFGS-B** | 0.349 ± 0.033 | 0.839 ± 0.022 | 0.849 ± 0.025 | 98.0s |

## Key Findings

1. **Krotov achieves the fastest initial convergence.** After just 1 epoch, Krotov
   reaches ~0.38-0.45 loss, while Adam and L-BFGS-B are still at ~0.5-0.7. This is
   the expected behavior: Krotov's sequential gate updates exploit local structure
   and can make large coordinated moves in a single sweep.

2. **Adam and L-BFGS-B reach lower final losses.** Both gradient-based baselines
   converge to ~0.35 loss with low variance, while Krotov plateaus at ~0.42 on
   average with significant oscillation.

3. **Krotov shows higher variance and instability.** The per-seed traces exhibit
   large oscillations (loss spikes from 0.35 to 0.79 mid-training), and one seed
   failed to converge well (final loss 0.71). The sample-by-sample update with a
   fixed step size causes overshooting.

4. **Test accuracy is comparable across all optimizers** (~84-86%), suggesting the
   model's generalization is limited by the ansatz expressivity rather than the optimizer.

5. **L-BFGS-B is the most efficient** in terms of convergence per step (converges
   in ~20 steps), though it sometimes uses many extra evaluations for line search.

6. **Decision boundaries** are qualitatively similar across all three optimizers,
   confirming that the task is learnable and the model has sufficient capacity.

## Caveats

- **Function evaluation counting is not perfectly fair.** Krotov processes each
  sample individually (2 circuit evals per sample × 350 samples = 700 per epoch),
  while Adam/L-BFGS-B evaluate 2×24 = 48 shifted circuits per step (each on all
  350 samples). The "loss vs step" and "loss vs wall-clock time" plots are the
  fairest comparisons.

- **Krotov step size was not extensively tuned.** A smaller or adaptive step size
  may reduce the oscillations. However, the benchmark uses reasonable defaults for
  all optimizers.

- **No Krotov implementation existed in the repository.** The Krotov optimizer was
  implemented from scratch based on the sequential gate-update variant of Krotov's
  optimal control method adapted for parameterized quantum circuits.

## Reproduction

```bash
# Run the full benchmark (takes ~45 minutes)
cd experiments/two_moons_qml_krotov
python run_experiment.py

# Generate plots
python plot_results.py

# Print summary statistics
python analyze_results.py
```

## File Structure

```
experiments/two_moons_qml_krotov/
├── README.md              # This file
├── config.py              # Experiment configuration
├── dataset.py             # Two-moons data generation
├── model.py               # 4-qubit VQC with HEA
├── optimizers.py          # Krotov, Adam, L-BFGS-B implementations
├── run_experiment.py      # Main experiment runner
├── plot_results.py        # Publication-quality plot generation
├── analyze_results.py     # Summary statistics
├── results/               # Raw JSON results per optimizer per seed
│   ├── config.json
│   ├── summary.json
│   └── result_<opt>_seed<N>.json
└── plots/                 # PNG + PDF plots
    ├── loss_vs_evals.{png,pdf}
    ├── loss_vs_step.{png,pdf}
    ├── loss_vs_time.{png,pdf}
    ├── boxplot_loss.{png,pdf}
    ├── boxplot_accuracy.{png,pdf}
    ├── success_rate.{png,pdf}
    └── decision_boundaries.{png,pdf}
```
