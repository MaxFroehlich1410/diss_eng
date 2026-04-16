# Parity PennyLane Comparison

This experiment compares the repository-native hybrid Krotov optimizer against
the PennyLane optimizers that are fair like-for-like baselines for the fixed
4-bit parity classifier.

The PennyLane package is expected in the repo-local folder:

- `vendor/pennylane`

Run from the repository root:

```bash
python3 -m experiments.parity_pennylane_comparison.run
```

Example with explicit settings:

```bash
python3 -m experiments.parity_pennylane_comparison.run \
  --seeds 0 \
  --repeats 8 \
  --n-layers 2 \
  --max-iterations 20 \
  --adam-lr 0.05 \
  --qng-lr 0.1 \
  --qng-lam 0.001
```

Notes:

- Included PennyLane optimizers for this benchmark:
  `Adam`, `Adagrad`, `GradientDescent`, `Momentum`, `NesterovMomentum`,
  `RMSProp`, `SPSA`, `QNG`, and `Rotosolve`.
- `Rotosolve` is included on the same circuit family. The scalar bias is refreshed
  analytically after each rotosolve sweep because the quantum angles are periodic
  but the bias is not.
- Excluded PennyLane optimizers:
  `AdaptiveOptimizer` because it grows the circuit from an operator pool,
  `RiemannianGradientOptimizer` because it optimizes directly on the unitary
  manifold rather than this fixed parameterization,
  `RotoselectOptimizer` because it changes the gate set while optimizing,
  `ShotAdaptiveOptimizer` because it is designed for finite-shot single-QNode
  objectives, and `QNSPSAOptimizer` because it expects a QNode-level objective
  and metric estimation pipeline rather than the supervised dataset-averaged MSE
  objective used here.
