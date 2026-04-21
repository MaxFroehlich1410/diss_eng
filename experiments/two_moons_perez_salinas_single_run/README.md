# Perez-Salinas Single Run

Single-run benchmark for the native Perez-Salinas classifier with an explicit
train/test overlap check and a learned decision-boundary plot.

Example:

```bash
python3 -m experiments.two_moons_perez_salinas_single_run.run \
  --problem crown \
  --optimizer krotov_hybrid \
  --n-qubits 4 \
  --n-layers 8 \
  --n-samples 240 \
  --max-iterations 12 \
  --hybrid-switch-iteration 10 \
  --hybrid-online-step-size 0.3 \
  --hybrid-batch-step-size 0.5
```
