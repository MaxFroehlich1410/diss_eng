# Perez-Salinas Optimizer Sweeps

Sharded hyperparameter sweeps for the entangling 4-qubit, 8-layer
Perez-Salinas classifier with the classical affine head enabled.

QNG is intentionally excluded from this sweep because it is too slow here and
the full-metric variant is not compatible with the 4-wire device.

Defaults:
- `problem=crown`
- `n_samples=600`
- `test_fraction=0.3`
- `seeds=0 1 2`
- `max_iterations=20`

The script explicitly aborts if the train/test split contains duplicate points.
It also checkpoints after every completed optimization run by updating:
- `raw_results.json`
- `analysis.json`
- `experiment.md`

Example shard:

```bash
python3 -m experiments.two_moons_perez_salinas_optimizer_sweeps.run \
  --num-shards 4 \
  --shard-index 0 \
  --run-name shard0
```
