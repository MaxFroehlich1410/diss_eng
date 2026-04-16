# Task: Implement 3 trainable QML models from the two-moons literature and benchmark 4 training methods on them

## Goal

Implement **three quantum machine learning models** from the two-moons classification literature that are genuinely trainable with gradient-based or other trainable optimization methods, and integrate them into a **single benchmark framework**.

The final goal is:

1. implement the **model structure** of each trainable QML model as faithfully as possible,
2. expose each model to **4 interchangeable training methods**,
3. train all model–method combinations on the **same two-moons benchmark protocol**,
4. compare performance fairly using:
  - training accuracy
  - validation accuracy
  - test accuracy
  - training loss
  - wall-clock time
  - number of function evaluations / gradient evaluations
  - robustness across random seeds

---

## Important scope decisions

Do **not** include kernel-only models such as Suzuki or Shastry in this benchmark, because those papers do **not** train variational quantum circuit parameters in the published setup. This benchmark should focus only on models where a trainable quantum or hybrid parameterization exists.

Use these **3 models**:

1. **Simonetti–Perri–Gervasi (2022)** hybrid VQC classifier
2. **Souza et al. (2024)** single-qubit quantum neural network (SQQNN) classifier
3. **Chen et al. (2025)** SUN-VQC classifier

---

## Non-negotiable implementation principles

### 1. Separate model definition from training method

The model architecture must be independent of the optimization rule.

Create a clean abstraction like:

- `model.forward(x, params)`
- `model.predict(x, params)`
- `model.loss(batch, params)`
- `model.get_initial_params(seed)`
- `model.parameter_metadata()`

and a separate training interface like:

- `trainer.fit(model, dataset, init_params, config)`

The 4 training methods must be pluggable without changing model code.

---

### 2. Be explicit about ambiguities

Some papers do **not** fully specify every implementation detail needed for exact reproduction.

Whenever something is not uniquely specified:

- make the **most paper-faithful reasonable choice**,
- document it clearly in comments and in a `MODEL_NOTES.md`,
- keep it configurable.

Do **not** silently invent architecture details.

---

### 3. Keep comparison fair

For the main benchmark:

- use the **same standardized dataset generation**
- use the **same train/val/test protocol**
- use the **same number of seeds**
- use the **same stopping rules**
- use the **same parameter initialization policy**
- use the **same metric logging**

Paper-faithful reproductions are useful as supplemental runs, but the main training-method comparison must be standardized.

---

### 4. Expose both quantum and classical parameters cleanly

Some models are hybrid quantum-classical models.

Implement parameter handling so that:

- quantum parameters are identifiable separately,
- classical parameters are identifiable separately,
- a training method can optimize:
  - only quantum parameters,
  - only classical parameters,
  - or both.

Default benchmark mode: **optimize all trainable parameters** unless the user specifies otherwise.

---



---

# Model 1 — Simonetti, Perri, Gervasi (2022)

## High-level idea

This is a **hybrid classical–quantum–classical classifier**:

- a classical dense input layer maps 2D features to circuit angles,
- a small variational quantum circuit processes them,
- a classical output layer maps quantum measurements to class logits.

## Paper-faithful structure to implement

### Input

- two-moons sample: `x = (x1, x2) in R^2`

### Classical input layer

Use a dense layer that maps the 2D input into the angles needed by the quantum circuit.

Main paper-faithful reconstruction:

- number of qubits: **2**
- number of repeated quantum blocks / sublayers: **4**
- each sublayer contains:
  - `Ry` on qubit 0
  - `Ry` on qubit 1
  - `Rz` on qubit 0
  - `Rz` on qubit 1
  - then a **CNOT chain**

That means the classical input layer should output **16 real values** total:

- 4 angles per sublayer
- 4 sublayers
- total = 16 angles

So implement:

- `Dense(2 -> 16)` as the classical feature-to-angle map

### Quantum circuit

Use 2 qubits initialized in `|00>`.

For each of the 4 sublayers:

1. apply `Ry(theta_l,0)` on qubit 0
2. apply `Ry(theta_l,1)` on qubit 1
3. apply `Rz(phi_l,0)` on qubit 0
4. apply `Rz(phi_l,1)` on qubit 1
5. apply entangling CNOT chain

Default CNOT chain for 2 qubits:

- `CNOT(0 -> 1)`

Make the entangler configurable in case a bidirectional chain or repeated chain is needed, but default to a single `CNOT(0,1)` per sublayer.

### Readout

The paper uses a hybrid output layer after the quantum circuit, but the exact readout observables are not fully specified in a uniquely reconstructable way.

Use the following faithful and practical reconstruction:

- measure the expectation values:
  - `<Z0>`
  - `<Z1>`
- treat the 2-dimensional measurement vector as quantum features
- apply a final classical output layer:
  - `Dense(2 -> 1)` for binary logit
  - sigmoid for class probability

So the full model is:

`x (2D) -> Dense(2,16) -> 2-qubit circuit (4 x [Ry,Ry,Rz,Rz,CNOT]) -> [<Z0>, <Z1>] -> Dense(2,1) -> sigmoid`

## Trainable parameters

- classical input dense weights + bias
- classical output dense weights + bias

Important:
The quantum gate angles are **data-dependent outputs** of the first dense layer, not necessarily independent free variational circuit parameters.

Still, this is a trainable hybrid model because the classical layer determines the effective quantum angles.

## Training in the paper

The paper trains the hybrid model with **SGD** for a fixed number of epochs.

## Gradient compatibility

This model is compatible with:

- standard autodiff through the hybrid stack,
- parameter-shift for the quantum part if needed,
- custom training rules acting on the effective trainable parameterization.

## Implementation note

Because the quantum circuit angles come from a classical layer, your implementation should support two modes:

### Mode A — paper-faithful hybrid mode

Train the dense layers and keep the circuit topology fixed.

### Mode B — explicit-angle mode

Replace the input dense layer by directly trainable circuit angles (still with 2 qubits and 4 sublayers) so that custom quantum training rules can be tested more directly.

Implement **both**. Use Mode A as the primary reproduction and Mode B as an additional ablation.

---

# Model 2 — Souza et al. (2024): SQQNN classifier

## High-level idea

This model is a **single-qubit quantum neural network**.

The paper introduces a general single-qubit neuron, but for classification it also studies a **reduced classifier structure** that is particularly easy to implement and train.

For this benchmark, implement the **classification-oriented reduced SQQNN** as the main benchmark model, and optionally the full single-qubit neuron model as an extra.

---

## Main benchmark version: reduced SQQNN classifier

### Input

- two-moons sample: `x = (x1, x2) in R^2`

### Quantum register

- **1 qubit**

### Circuit structure

Use a sequence of `m` single-qubit layers (called neurons in the paper context), where each layer applies a rotation:

- `Ry(beta_k(x; a_k))`

for `k = 1, ..., m`

So the circuit is:

`|0> -> Ry(beta_1(x)) -> Ry(beta_2(x)) -> ... -> Ry(beta_m(x))`

### Angle parameterization

Each `beta_k(x)` is a trainable function of the classical input.

Implement it as a **polynomial feature map in the 2D input**:

`beta_k(x) = sum_j a_{k,j} * phi_j(x)`

where `phi_j(x)` are polynomial basis functions of `(x1, x2)`.

Use a configurable polynomial basis.

Default basis:

- constant `1`
- linear terms: `x1, x2`
- quadratic terms: `x1^2, x1*x2, x2^2`

So default:
`phi(x) = [1, x1, x2, x1^2, x1*x2, x2^2]`

Then each neuron has its own coefficient vector `a_k`.

### Number of neurons

Make `m` configurable.

Recommended defaults:

- `m = 4`
- `m = 6`

because the paper reports strong performance with multiple neurons.

### Readout

Measure:

- `<Z>`

Convert to binary probability via:

- `p(y=1|x) = (1 - <Z>) / 2`

or equivalently pass `<Z>` through a logit mapping. Keep the choice consistent across runs.

Default:

- use `(1 - <Z>) / 2` as the positive-class probability.

### Full model

The reduced benchmark model is:

`x -> polynomial features phi(x) -> beta_k(x) for each neuron -> 1-qubit chain of Ry rotations -> <Z> -> binary probability`

---

## Optional extended version: full single-qubit neuron model

Also implement an optional full neuron with a generic single-qubit unitary decomposition such as:

`Rz(gamma_k(x)) Ry(beta_k(x)) Rz(alpha_k(x))`

This is not the primary benchmark model, but it is useful if we want a richer trainable single-qubit baseline.

---

## Trainable parameters

For reduced SQQNN:

- polynomial coefficients for each neuron:
  - `a_k`

For full SQQNN:

- coefficient sets for each angle function:
  - `alpha_k(x)`
  - `beta_k(x)`
  - `gamma_k(x)`

---

## Training in the paper

Important distinction:

- for some settings, the paper derives **analytic gradients**
- for classification, the reduced model is trained via a **linear least squares (LLS)** strategy

For **this benchmark**, do **not** restrict yourself to the paper’s LLS training only.

Instead:

1. implement the reduced classifier as a trainable parametric model,
2. allow all 4 user-provided training methods to optimize its coefficients,
3. optionally include the paper-style **LLS baseline** as an extra comparison.

---

## Gradient compatibility

This model is compatible with:

- parameter-shift, because it uses standard rotation gates,
- analytic gradients,
- standard autodiff,
- custom optimization rules over the polynomial coefficients.

Because all rotations are on a single qubit, it is also a very clean sanity-check model for new training methods.

---

# Model 3 — Chen et al. (2025): SUN-VQC

## High-level idea

This is a **4-qubit variational quantum classifier** using a **brick-wall architecture** built from trainable **two-qubit SU(4) blocks**.

This is the most direct gradient-centric benchmark of the three.

---

## Paper-faithful target structure

### Input

- two-moons sample: `x = (x1, x2) in R^2`

### Quantum register

- **4 qubits**

### Data encoding

The paper encodes the data into a 4-qubit register, but the exact feature-to-gate encoding should be implemented as a configurable module if the paper does not uniquely specify every gate in the accessible description.

Implement a configurable input encoding with the following default:

#### Default encoding

Repeat the two classical features across 4 qubits with angle encoding:

- qubit 0: `Ry(x1)`
- qubit 1: `Ry(x2)`
- qubit 2: `Ry(x1)`
- qubit 3: `Ry(x2)`

Optional extension:

- append `Rz(x1)` / `Rz(x2)` in the same repeated pattern if needed.

Keep this encoding modular so it can be swapped if a more exact paper-faithful encoding is recovered.

---

## Variational core: SUN blocks

### Macro-layer definition

Define one macro-layer as a nearest-neighbor brick-wall of trainable two-qubit SU(4) blocks:

- even layer:
  - block on qubits `(0,1)`
  - block on qubits `(2,3)`
- odd layer:
  - block on qubits `(1,2)`

This is one standard brick-wall sweep.

### Number of macro-layers

Use:

- **2 macro-layers**

because the paper uses a **two-layer SUN-VQC** for the two-moons task.

So the total trainable circuit is:

`Encoding(x) -> BrickWallLayer1 -> BrickWallLayer2 -> measurement`

---

## SU(4) block implementation

Each two-qubit block should be implemented as a differentiable, explicitly parameterized **15-parameter SU(4)** unitary.

Use one of these options:

### Preferred option

A standard KAK / Cartan-style SU(4) decomposition with 15 trainable parameters.

### Acceptable fallback

Any well-documented universal 2-qubit 15-parameter decomposition that is:

- differentiable,
- reproducible,
- fixed across all training methods.

Document the exact decomposition used.

Do **not** replace the SU(4) block by a much simpler hardware-efficient block unless explicitly running an ablation.

---

## Readout

For binary classification, implement both:

### Readout A — simple quantum readout

- measure `<Z0>`
- convert to class probability

### Readout B — richer hybrid readout

- measure `[<Z0>, <Z1>, <Z2>, <Z3>]`
- apply a small classical linear layer `Dense(4 -> 1)`

Default benchmark:

- use **Readout A** for the purest VQC benchmark
- include Readout B as an ablation

---

## Trainable parameters

- all SU(4) block parameters
- optionally output linear layer parameters if using hybrid readout

---

## Training in the paper

This paper explicitly uses a **generalized parameter-shift rule** and gradient descent.

So this is the cleanest direct baseline for custom quantum training methods.

---

## Gradient compatibility

This model must support:

- generalized parameter-shift
- autodiff if the backend allows it
- custom user-defined training rules

This model is the most important benchmark in the suite.

---

# Unified benchmarking protocol

## Dataset

Use `sklearn.datasets.make_moons`.

### Main benchmark dataset

Use a standardized common benchmark:

- `n_samples = 1000`
- `noise = 0.1`
- stratified split:
  - 60% train
  - 20% validation
  - 20% test

Standardize features if helpful, but keep preprocessing identical across all models.

### Reproducibility

Run at least:

- `5 random seeds`

Store the exact split for each seed.

---

## Loss

Use binary cross-entropy for all primary runs.

Optional extras:

- MSE on probability output
- hinge loss

But the main benchmark should use:

- **binary cross-entropy**

---

## Training budget

To compare methods fairly, evaluate under **multiple fairness criteria**:

### Criterion A — fixed epochs / steps

Same number of optimization steps across methods.

### Criterion B — fixed function-evaluation budget

Same total number of forward evaluations or objective calls.

### Criterion C — fixed wall-clock budget

Same training time budget.

At minimum, implement A and B.

---

## Metrics to log

For every epoch / optimization step, log:

- training loss
- validation loss
- training accuracy
- validation accuracy
- test accuracy at best validation checkpoint
- number of forward calls
- number of backward / gradient calls
- wall-clock time
- parameter norm
- gradient norm if available

Also save:

- decision boundary plot
- training curve plot
- per-seed summary JSON / CSV

---

## Training-method integration requirements

The code must allow plugging in **4 user-provided training methods**.

Implement a clean registry such as:

- `trainer_name -> trainer object`

Each trainer should support:

- initialization from config
- step/update function
- logging hooks
- optional access to gradients
- optional access to parameter-shift evaluations
- optional access to stochastic mini-batches

Do not hard-code assumptions that all methods are standard optimizers.

---

## Suggested experiment matrix

For each model:

- run each of 4 training methods
- run 5 seeds
- evaluate under standardized dataset split

Core matrix:

- `3 models x 4 methods x 5 seeds = 60 runs`

Optional extra matrix:

- Simonetti explicit-angle mode
- Souza full SQQNN mode
- Chen hybrid-readout mode

---

# Deliverables

## 1. Code

Implement clean modules:

- `models/`
  - `simonetti_hybrid.py`
  - `souza_sqqnn.py`
  - `chen_sun_vqc.py`
- `trainers/`
  - generic trainer interface
  - 4 method adapters
- `experiments/`
  - benchmark runner
  - config files
- `utils/`
  - plotting
  - logging
  - seed control
  - metrics

---

## 2. Documentation

Create:

- `README.md`
- `MODEL_NOTES.md`
- `BENCHMARK_PROTOCOL.md`

In `MODEL_NOTES.md`, explicitly document:

- what is directly paper-faithful
- what had to be reconstructed
- what is configurable
- any deviations from the papers

---

## 3. Output artifacts

For each run, save:

- config
- final metrics
- best checkpoint
- training curves
- decision boundary plot
- confusion matrix
- seed summary

Also create a final aggregate table:

- rows = model + training method
- columns = mean/std over seeds for:
  - train accuracy
  - validation accuracy
  - test accuracy
  - loss
  - wall-clock time
  - function evaluations

---

# Acceptance criteria

The task is complete only if:

1. all 3 models are implemented in a way that matches the published model family as closely as possible,
2. all 4 training methods can be swapped in without changing model code,
3. the benchmark is fair and reproducible,
4. all ambiguities are documented,
5. final results include both per-run logs and aggregate comparisons.

---

# Summary of the 3 benchmark models

## Simonetti 2022

- 2 qubits
- 4 repeated blocks
- each block: `Ry(q0), Ry(q1), Rz(q0), Rz(q1), CNOT`
- classical dense input layer produces 16 angles
- classical dense output layer maps measurements to class logit
- hybrid model
- trainable with SGD/autodiff; parameter-shift compatible at the quantum level

## Souza 2024

- 1 qubit
- sequence of `Ry(beta_k(x))` layers
- `beta_k(x)` is a trainable polynomial function of 2D input
- readout from `<Z>`
- paper classification uses LLS, but model is fully trainable with custom methods
- excellent minimal benchmark for new training rules

## Chen 2025

- 4 qubits
- 2 brick-wall macro-layers
- each block is a trainable 2-qubit SU(4) unitary
- gradient-centric VQC
- primary analytic-gradient benchmark
- most important model in the suite for comparing training methods

