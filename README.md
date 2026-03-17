# Dissipative Engineering – PinexQ Circuit Builder + Qiskit Simulation

This repo is a minimal, reproducible harness to:

- install and inspect the [`pinexq-client`](https://pypi.org/project/pinexq-client/#files) package
- call a **PinexQ processing step** (e.g. your Tucker-tensor circuit builder) to produce a circuit
- simulate that circuit locally using **Qiskit Aer**, and compare the output state to a chosen target state

## What `pinexq-client` is (quick analysis)

`pinexq-client` is **not** a local circuit-synthesis library. It is a hypermedia client for the **PinexQ platform**
that lets you run *server-side processing steps* by name (think “remote functions”) via jobs:

- create a job
- select a processing step (`function_name`, optional version)
- pass JSON parameters (and/or input workdata)
- wait for completion
- read the JSON result and/or download output workdata blobs

## Setup

Create and activate a Python 3.11 virtualenv, then install:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## PinexQ credentials (online mode)

To actually call your Tucker circuit builder, you need:

- `PINEXQ_API_ENDPOINT` (base URL of your PinexQ Job Management API)
- `PINEXQ_API_KEY` (sent as `x-api-key`)

Example:

```bash
export PINEXQ_API_ENDPOINT="https://myapihost.com:80"
export PINEXQ_API_KEY="...secret..."
```

## Discover the circuit builder step

List available processing steps:

```bash
python pinexq_qiskit_demo.py --list-processing-steps
```

Describe a specific step (prints parameter/return schemas, data slot specs, tags, etc.):

```bash
python pinexq_qiskit_demo.py --describe-processing-step "<FUNCTION_NAME>"
```

## Build + simulate (online mode)

Run a processing step and simulate the returned circuit:

```bash
python pinexq_qiskit_demo.py \
  --processing-step "<YOUR_TUCKER_CIRCUIT_BUILDER_FUNCTION_NAME>" \
  --n-qubits 3 \
  --target ghz \
  --print-qc
```

If your processing step requires a custom parameter schema, pass parameters directly as JSON:

```bash
python pinexq_qiskit_demo.py \
  --processing-step "<YOUR_STEP>" \
  --params-json '{"someKey": 123, "someOtherKey": "abc"}'
```

The script will try to extract a circuit from either:

- `job.get_result()` (JSON result containing a QASM string somewhere), or
- output workdata files attached to the job (downloaded and heuristically parsed as OpenQASM)

## Offline fallback (no PinexQ)

This produces a state-preparation circuit locally (Qiskit `initialize`) and simulates it:

```bash
python pinexq_qiskit_demo.py --offline --n-qubits 3 --target ghz --print-qc
```

