## Hydrogen wavefunctions + Q-Alchemy state preparation

This experiment computes a hydrogenic wavefunction on an x-z slice, flattens it
into a statevector, and sends it to Q-Alchemy to produce a Qiskit circuit for
state preparation.

### Setup

From the repo root:

```bash
pip install -r requirements.txt
pip install scipy
```

Set your Q-Alchemy API key (and optionally a host override):

```bash
export Q_ALCHEMY_API_KEY="..."
# export Q_ALCHEMY_HOST="jobs.api.q-alchemy.com"
```

### Run

```bash
python experiments/hydrogen_qalchemy/run_hydrogen_qalchemy.py
```

### Notes

- `num_qubits` must be even. The grid size is `2 ** (num_qubits / 2)`.
- `build_qalchemy_circuit()` calls the Q-Alchemy API when the circuit is
  decomposed to a concrete gate sequence.
