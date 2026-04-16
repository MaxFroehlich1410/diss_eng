# Alternative Two-Moons QML Model Notes

## Simonetti, Perri, Gervasi (2022)

- Paper-faithful core: 2 qubits, 4 repeated sublayers, each sublayer uses `Ry(q0)`, `Ry(q1)`, `Rz(q0)`, `Rz(q1)`, then `CNOT(0 -> 1)`.
- Primary reconstruction: a classical dense `2 -> 16` map is represented exactly as angle contributions `w_0 x_0 + w_1 x_1 + b` on each gate.
- Readout reconstruction: the paper-level description of the post-quantum classical head is not specific enough to recover a unique implementation, so the model uses `[<Z0>, <Z1>] -> Dense(2 -> 1) -> sigmoid`.
- Explicit-angle ablation: direct trainable angle offsets are added on top of a fixed repeated raw-feature encoding `[x1, x2, x1, x2]` per sublayer. This keeps the ablation trainable and input-dependent; a fully data-independent replacement would not define a useful classifier.
- Output-layer parameters are classical and support analytic gradients, but not gate-by-gate Krotov updates.

## PennyLane Variational Classifier Demo (4-bit parity)

- Architecture: 4 qubits, basis-state encoding of a 4-bit input, then repeated layers of per-qubit `Rot(phi, theta, omega)` gates followed by the ring `CNOT(0,1)`, `CNOT(1,2)`, `CNOT(2,3)`, `CNOT(3,0)`.
- Repository implementation decomposes each `Rot` exactly into `RZ(phi)`, `RY(theta)`, `RZ(omega)` one-parameter gates so the existing gatewise gradient and Krotov interfaces remain usable.
- Readout is the scalar `q(x)=<Z0>` plus one trainable classical bias.
- Loss is mean squared error on labels in `{-1, +1}` and prediction uses the sign of the scalar output.

## Souza et al. (2024)

- Primary benchmark model: reduced single-qubit classifier with `m` sequential `Ry(beta_k(x))` layers.
- Default basis: `[1, x1, x2, x1^2, x1*x2, x2^2]`.
- Each coefficient multiplies one basis feature and is represented as its own same-axis rotation factor, which preserves exact angles because same-axis rotations commute.
- Optional extended variant: `Rz(alpha_k(x)) Ry(beta_k(x)) Rz(gamma_k(x))`.
- Probability mapping follows the paper-oriented classifier convention `p(y=1|x) = (1 - <Z>) / 2`, implemented as `obs = -Z`.

## Chen et al. (2025)

- Paper-faithful high-level structure: 4 qubits, default repeated angle encoding of the two classical features, then a 2-layer brick-wall ansatz.
- Each two-qubit block is implemented as an ordered product of 15 one-parameter exponentials over the `su(4)` Pauli basis:
  `XI, YI, ZI, IX, IY, IZ, XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ`.
- This is a fixed 15-parameter differentiable `SU(4)` decomposition. It is not a minimal-CNOT KAK circuit, but it is a valid universal fallback consistent with the task brief.
- Default readout is the pure quantum `Z0` head. A hybrid linear readout over `[<Z0>, <Z1>, <Z2>, <Z3>]` is also implemented as an ablation.
- Hybrid readout parameters are classical and therefore not exposed through the gatewise derivative interface.
