# Perez-Salinas PennyLane Comparison

This experiment mirrors the existing native Perez-Salinas re-uploading model in
PennyLane and compares Hybrid Krotov against the same family of first-order and
QNG optimizers used in the parity benchmark.

Included optimizers:
- `krotov_hybrid`
- `pennylane_adam`
- `pennylane_adagrad`
- `pennylane_gradient_descent`
- `pennylane_momentum`
- `pennylane_nesterov`
- `pennylane_rmsprop`
- `pennylane_spsa`
- `pennylane_qng`

`Rotosolve` is intentionally omitted. For this data-reuploading objective, the
input-scaled gates induce sample-dependent parameter frequencies, so a faithful
frequency specification for the dataset-averaged weighted-fidelity loss is not
available without additional derivation.

Smoke run:

```bash
python3 -m experiments.two_moons_perez_salinas_pennylane_comparison.run \
  --problem crown \
  --n-qubits 4 \
  --n-layers 8 \
  --n-samples 120 \
  --max-iterations 4 \
  --seeds 0 \
  --run-name smoke
```
