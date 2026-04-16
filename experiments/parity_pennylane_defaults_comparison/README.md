# Parity PennyLane Defaults Comparison

This experiment is the strict-default counterpart to
`experiments/parity_pennylane_comparison`.

It keeps the same 4-qubit parity circuit family but removes the trainable
classical bias so that the objective is purely quantum:

- `f(x; weights) = <Z0>`
- loss = mean squared error on labels in `{-1, +1}`

Included PennyLane optimizers are only those that can be applied to the
supervised dataset-averaged objective without custom gradient plumbing,
custom metric tensors, or custom frequency metadata:

- `Adam`
- `Adagrad`
- `GradientDescent`
- `Momentum`
- `NesterovMomentum`
- `RMSProp`
- `SPSA`

Excluded from this strict-default benchmark:

- `QNG`, because PennyLane's default QNG expects a QNode-level objective rather
  than this dataset-averaged supervised loss.
- `Rotosolve`, because the generic supervised objective requires extra
  frequency metadata.
- `Adaptive`, `Rotoselect`, `RiemannianGradient`, `ShotAdaptive`, and
  `QNSPSA`, because they do not match the fixed-model supervised setup here.

Run from the repository root:

```bash
python3 -m experiments.parity_pennylane_defaults_comparison.run
```

Example:

```bash
python3 -m experiments.parity_pennylane_defaults_comparison.run \
  --seeds 0 \
  --repeats 4 \
  --n-layers 2 \
  --max-iterations 12
```
