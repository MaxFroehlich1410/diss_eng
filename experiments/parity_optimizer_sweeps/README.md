# Parity Optimizer Sweeps

Large hyperparameter sweeps for the 4-bit parity classifier on the corrected
leakage-free split.

Default benchmark:
- 10 unique training bitstrings
- 6 disjoint unique test bitstrings
- 2 parity layers
- 3 seeds per hyperparameter combination

Run the full sweep:

```bash
python3 -m experiments.parity_optimizer_sweeps.run \
  --seeds 0 1 2 \
  --train-size 10 \
  --test-size 6 \
  --n-layers 2 \
  --max-iterations 12
```

Run only a subset:

```bash
python3 -m experiments.parity_optimizer_sweeps.run \
  --optimizers krotov_hybrid pennylane_rmsprop pennylane_rotosolve \
  --seeds 0 1 2
```

Outputs:
- `raw_results.json`
- `analysis.json`
- `experiment.md`
- `best_configs_loss_vs_iteration.(pdf|png)`
- `best_configs_loss_vs_time.(pdf|png)`

The script records all optimizer hyperparameters that were scanned in the local
PennyLane and Krotov implementations, and then sweeps the practical subset that
materially changes behavior on this benchmark.
