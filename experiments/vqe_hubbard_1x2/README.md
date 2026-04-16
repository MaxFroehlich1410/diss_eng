# Exact 1x2 Fermi-Hubbard HV-VQE Benchmark

This folder contains a clean exact-statevector VQE benchmark for the 4-qubit,
5-layer Hamiltonian-variational (HV) ansatz on the 1x2 Fermi-Hubbard model at
half filling.

## Physics Setup

- Qubit ordering: `q0=(site 1, up)`, `q1=(site 2, up)`, `q2=(site 1, down)`, `q3=(site 2, down)`
- Physical Hamiltonian:
  `H(U, t) = t * H_hop_unit + U * H_onsite_unit`, with `t=-1` by default
- Hopping generator:
  `H_hop_unit = 0.5 (X0X1 + Y0Y1) + 0.5 (X2X3 + Y2Y3)`
- Onsite generator:
  `H_onsite_unit = |11><11|_(0,2) + |11><11|_(1,3)`
- Half filling means 2 fermions total; the reference state is the exact
  non-interacting (`U=0`) ground state in that 2-particle sector.

## Ansatz

- Layers: 5
- Parameters per layer: 2
- Total trainable parameters: 10
- Ordering:
  `theta = [phi_1, tau_1, phi_2, tau_2, phi_3, tau_3, phi_4, tau_4, phi_5, tau_5]`
- Layer convention:
  `U_layer(phi_l, tau_l) = exp(+i * tau_l * H_hop_unit) @ exp(+i * phi_l * H_onsite_unit)`

Within each layer, one shared `phi_l` acts on both onsite pairs `(0,2)` and
`(1,3)`, and one shared `tau_l` acts on both hopping pairs `(0,1)` and `(2,3)`.

## Loss Function

The VQE objective is the exact energy expectation

`E(theta) = <psi(theta)|H|psi(theta)>`

with exact dense statevectors throughout:

- no shots
- no measurement sampling
- no device noise
- no black-box finite-difference gradients as the primary gradient method

## Krotov Interface

The module `hubbard_1x2_hv.py` provides:

- standalone helpers for Hamiltonian construction, reference-state preparation,
  ansatz application, forward states, exact energy, and exact energy gradient
- `Hubbard1x2HVVQEProblem`, a thin wrapper exposing:
  - `get_gate_sequence_and_states(theta)`
  - `gate_derivative_generator(param_idx)`
  - `parameter_metadata()`

This keeps the physics/model definition separate from any later hybrid Krotov
optimizer implementation.

## Optimizer Sweeps

`run_vqe_optimizer_sweeps.py` runs matched exact-statevector sweeps for:

- `adam`
- `bfgs`
- `qng`
- `krotov_hybrid`

Example commands:

```bash
python -m experiments.vqe_hubbard_1x2.run --optimizer adam
python -m experiments.vqe_hubbard_1x2.run --optimizer bfgs
python -m experiments.vqe_hubbard_1x2.run --optimizer qng
python -m experiments.vqe_hubbard_1x2.run --optimizer krotov_hybrid
```

By default the script sweeps each optimizer over `U in {2,4,8}` and writes raw
JSON plus a short Markdown report under `experiments/vqe_hubbard_1x2/results/optimizer_sweeps/`.

To build the consolidated plots and LaTeX summary after the sweeps finish:

```bash
python -m experiments.vqe_hubbard_1x2.build_report
```

This writes figures and `vqe_optimizer_sweep_report.tex` under
`experiments/vqe_hubbard_1x2/results/report/` and compiles the PDF automatically
when `pdflatex` is available.
