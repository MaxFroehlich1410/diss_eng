# Dissipative Lindblad Tail

Deterministic density-matrix simulation of a **unitary state-preparation
circuit** followed by a **dissipative Lindblad tail**, designed to measure
whether engineered dissipation can increase fidelity to a fixed pure target
state |psi*> after a shallow approximate circuit U_init.

## Key design choices

| Aspect | Choice |
|---|---|
| Simulation | Deterministic density-matrix (no trajectories) |
| Circuit backend | Qiskit (circuit construction, decomposition, `Operator`) |
| Dissipation | Liouville-space matrix exponentiation (`scipy.linalg.expm`) |
| System size | n <= 6 qubits practical (Liouvillian is d^2 x d^2) |
| Dissipation stage | **Tail only** — applied after the circuit, not during gates |

## Package structure

```
diss_lindblad_tail/
├── diss_lindblad/              # Core package
│   ├── __init__.py
│   ├── density_matrix.py       # DM construction, fidelity, physicality checks
│   ├── lindblad.py             # Liouvillian, Lindblad operators, time evolution
│   ├── circuits.py             # Qiskit circuit helpers, target-state generators
│   └── experiment.py           # End-to-end experiment orchestration
├── tests/
│   └── test_sanity.py          # Sanity checks (standalone + pytest compatible)
├── main.py                     # CLI entry point
├── requirements.txt
└── README.md
```

## Quick start

```bash
# From the repo root (Diss_eng/):
cd diss_lindblad_tail

# Install dependencies (into the existing venv)
pip install -r requirements.txt

# Run the default experiment (3-qubit random state, cooling dissipation)
python main.py

# Run with a GHZ target, plot results
python main.py --n-qubits 3 --target ghz --gate-fraction 0.5 --plot

# Run sanity checks
python main.py --check
```

## CLI options

| Flag | Default | Description |
|---|---|---|
| `--n-qubits` | 3 | Number of qubits |
| `--target` | random | Target state: `random`, `ghz`, `w` |
| `--seed` | 42 | Random seed (for `random` target) |
| `--gate-fraction` | 0.5 | Fraction of exact-circuit gates to keep |
| `--dissipation` | cooling | Channel: `cooling`, `amplitude_damping`, `dephasing` |
| `--gamma` | 1.0 | Dissipation rate (uniform) |
| `--t-max` | 5.0 | Max evolution time |
| `--n-steps` | 50 | Number of time steps |
| `--plot` | off | Show matplotlib figure |
| `--check` | off | Run sanity checks and exit |

## Dissipation types

- **cooling**: `L_k = |psi*><psi_k^perp|` for k = 1 .. d-1.
  Drives any state to |psi*> (guaranteed fidelity increase).
- **amplitude_damping**: `sigma^- = |0><1|` on each qubit.
  Standard T1 decay toward |0...0>.
- **dephasing**: `Z` on each qubit.
  Standard T2 dephasing (kills off-diagonal coherences).

## How it works

1. A target pure state |psi*> is generated.
2. The exact unitary U* with U*|0> = |psi*> is built and decomposed via
   Qiskit's `UnitaryGate.decompose()` into elementary gates.
3. The circuit is **truncated** to `gate_fraction` of its gates, yielding
   an approximate unitary U_approx.
4. The initial density matrix is `rho_0 = U_approx |0><0| U_approx^dag`.
5. A Liouvillian superoperator is constructed from the chosen Lindblad
   operators in the column-stacking vectorisation convention.
6. Time evolution `rho(t) = exp(L t) rho_0` is computed via
   `scipy.linalg.expm` (incremental propagation for trajectory).
7. Fidelity `F(t) = <psi*|rho(t)|psi*>`, trace, and purity are recorded.

## Tests

```bash
# Via pytest
python -m pytest tests/test_sanity.py -v

# Standalone
python tests/test_sanity.py
```

The sanity checks verify:
- Pure-state density matrices are valid
- Unitary evolution preserves physicality
- gamma=0 leaves the state unchanged
- Trace is preserved under all Lindblad channels
- rho(t) remains positive semi-definite
- Cooling drives rho -> |psi*><psi*| as t -> inf
- Full circuit faithfully reproduces the target state
- Incremental and direct evolution agree
