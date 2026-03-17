#!/usr/bin/env python3
"""
PinexQ → Qiskit demo
====================

This script is intentionally "schema-flexible" because `pinexq-client` is an API client:
the actual Tucker/tensor circuit builder lives as a *server-side processing step* in PinexQ.

It can:
  - list and describe available processing steps (to find the circuit builder step)
  - submit a job to a chosen processing step, passing JSON parameters
  - retrieve a circuit from either the job result or output WorkData (e.g. OpenQASM)
  - simulate the circuit with Qiskit Aer and compute fidelity to a target state

Environment variables (for online mode):
  - PINEXQ_API_ENDPOINT  e.g. "https://myapihost.com:80"
  - PINEXQ_API_KEY       your x-api-key
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


def _normalize_state(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.complex128)
    nrm = np.linalg.norm(vec)
    if nrm == 0:
        raise ValueError("Target statevector has zero norm.")
    return vec / nrm


def target_statevector(name: str, n_qubits: int) -> np.ndarray:
    dim = 2**n_qubits
    name = name.lower().strip()

    if name == "zero":
        vec = np.zeros(dim, dtype=np.complex128)
        vec[0] = 1.0
        return vec

    if name in {"bell", "bell00"}:
        if n_qubits != 2:
            raise ValueError("Bell state requires n_qubits=2.")
        vec = np.zeros(4, dtype=np.complex128)
        vec[0] = 1 / np.sqrt(2)
        vec[3] = 1 / np.sqrt(2)
        return vec

    if name == "ghz":
        if n_qubits < 2:
            raise ValueError("GHZ requires n_qubits>=2.")
        vec = np.zeros(dim, dtype=np.complex128)
        vec[0] = 1 / np.sqrt(2)
        vec[-1] = 1 / np.sqrt(2)
        return vec

    if name == "w":
        if n_qubits < 2:
            raise ValueError("W requires n_qubits>=2.")
        vec = np.zeros(dim, dtype=np.complex128)
        # |100..0> + |010..0> + ... + |000..1>
        for i in range(n_qubits):
            vec[1 << (n_qubits - 1 - i)] = 1.0
        return _normalize_state(vec)

    raise ValueError(f"Unknown target state '{name}'. Try: zero, bell, ghz, w")


def statevector_to_json(vec: np.ndarray) -> list[dict[str, float]]:
    """JSON-safe encoding for complex amplitudes."""
    vec = np.asarray(vec, dtype=np.complex128)
    return [{"re": float(z.real), "im": float(z.imag)} for z in vec]


def try_circuit_from_qasm_str(qasm: str) -> QuantumCircuit:
    qasm = qasm.strip()
    if not qasm:
        raise ValueError("Empty QASM.")

    # QASM 3 usually starts with "OPENQASM 3;" (or sometimes "OPENQASM 3.0;")
    if qasm.upper().startswith("OPENQASM 3"):
        try:
            from qiskit import qasm3  # type: ignore

            return qasm3.loads(qasm)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Failed to parse OpenQASM 3: {e}") from e

    # QASM 2 usually starts with "OPENQASM 2.0;"
    if qasm.upper().startswith("OPENQASM 2"):
        try:
            from qiskit import qasm2  # type: ignore

            return qasm2.loads(qasm)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Failed to parse OpenQASM 2: {e}") from e

    # Some tools return raw gate lines without header; try qasm2 as best effort.
    try:
        from qiskit import qasm2  # type: ignore

        return qasm2.loads(qasm)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Unrecognized QASM header and qasm2 fallback failed: {e}") from e


def fidelity_up_to_global_phase(a: np.ndarray, b: np.ndarray) -> float:
    a = _normalize_state(a)
    b = _normalize_state(b)
    return float(np.abs(np.vdot(a, b)) ** 2)


def simulate_statevector(qc: QuantumCircuit) -> np.ndarray:
    # Aer only returns a statevector if it is explicitly saved.
    qc2 = qc.copy()
    qc2.save_statevector()

    sim = AerSimulator(method="statevector")
    tqc = transpile(qc2, sim)
    result = sim.run(tqc).result()
    sv = result.get_statevector(tqc)
    return np.asarray(sv, dtype=np.complex128)


def build_offline_stateprep_circuit(vec: np.ndarray) -> QuantumCircuit:
    vec = _normalize_state(vec)
    n_qubits = int(np.log2(vec.size))
    qc = QuantumCircuit(n_qubits, name="offline_stateprep")
    qc.initialize(vec, list(range(n_qubits)))
    return qc


@dataclass(frozen=True)
class PinexqConfig:
    endpoint: str
    api_key: str


def load_pinexq_config_from_env() -> Optional[PinexqConfig]:
    endpoint = os.getenv("PINEXQ_API_ENDPOINT", "").strip()
    api_key = os.getenv("PINEXQ_API_KEY", "").strip()
    if endpoint and api_key:
        return PinexqConfig(endpoint=endpoint, api_key=api_key)
    return None


def _require_pinexq(config: Optional[PinexqConfig]) -> PinexqConfig:
    if config is None:
        raise SystemExit(
            "Missing PINEXQ credentials. Set PINEXQ_API_ENDPOINT and PINEXQ_API_KEY "
            "or run with --offline."
        )
    return config


def pinexq_list_processing_steps(config: PinexqConfig, limit: int = 200) -> list[dict[str, Any]]:
    from pinexq.client.job_management.enterjma import create_pinexq_client, enter_jma
    from pinexq.client.job_management.model import ProcessingStepQueryParameters

    client = create_pinexq_client(config.endpoint, config.api_key, use_client_cache=False)
    try:
        entry = enter_jma(client)
        root = entry.processing_step_root_link.navigate()
        if not root.query_action:
            raise RuntimeError("ProcessingStep query action not available for this API key.")

        query = root.query_action.execute(ProcessingStepQueryParameters())
        out: list[dict[str, Any]] = []
        for ps in query.iter_flat():
            out.append(
                {
                    "function_name": ps.function_name,
                    "version": ps.version,
                    "title": ps.title,
                    "short_description": ps.short_description,
                    "tags": ps.tags,
                }
            )
            if len(out) >= limit:
                break
        return out
    finally:
        client.close()


def pinexq_describe_processing_step(config: PinexqConfig, function_name: str, version: str | None) -> dict[str, Any]:
    from pinexq.client.job_management.enterjma import create_pinexq_client
    from pinexq.client.job_management.tool.processing_step import ProcessingStep

    client = create_pinexq_client(config.endpoint, config.api_key, use_client_cache=False)
    try:
        ps = ProcessingStep.from_name(client, function_name, version=version or "0")
        hco = ps.processing_step_hco
        if hco is None:
            raise RuntimeError("Failed to load processing step HCO.")
        return {
            "function_name": hco.function_name,
            "version": hco.version,
            "title": hco.title,
            "short_description": hco.short_description,
            "long_description": hco.long_description,
            "tags": hco.tags,
            "has_parameters": hco.has_parameters,
            "parameter_schema": hco.parameter_schema,
            "default_parameters": hco.default_parameters,
            "return_schema": hco.return_schema,
            "input_data_slot_specification": [s.model_dump(mode="json") for s in (hco.input_data_slot_specification or [])],
            "output_data_slot_specification": [s.model_dump(mode="json") for s in (hco.output_data_slot_specification or [])],
        }
    finally:
        client.close()


def _iter_possible_qasm_payloads(result: Any) -> Iterable[str]:
    """Heuristically pull QASM-ish strings out of a JSON-ish result."""
    if isinstance(result, str):
        yield result
        return

    if isinstance(result, dict):
        for key in ("qasm", "openqasm", "openqasm2", "openqasm3", "circuit_qasm", "circuit"):
            val = result.get(key)
            if isinstance(val, str):
                yield val
        # nested
        for v in result.values():
            yield from _iter_possible_qasm_payloads(v)
        return

    if isinstance(result, list):
        for v in result:
            yield from _iter_possible_qasm_payloads(v)


def pinexq_build_circuit_via_job(
    config: PinexqConfig,
    *,
    processing_step: str,
    processing_step_version: str | None,
    job_name: str,
    parameters: dict[str, Any],
    timeout_s: float | None,
    try_output_workdata: bool = True,
) -> tuple[QuantumCircuit, dict[str, Any]]:
    """
    Run a PinexQ job for the given processing_step and attempt to recover a circuit.

    Returns (circuit, debug_info).
    """
    from pinexq.client.job_management.enterjma import create_pinexq_client
    from pinexq.client.job_management.tool.job import Job
    from pinexq.client.job_management.tool.workdata import WorkData

    client = create_pinexq_client(config.endpoint, config.api_key, use_client_cache=False)
    debug: dict[str, Any] = {"job_result": None, "output_workdata_candidates": []}
    try:
        job = Job(client).create(name=job_name)
        job.select_processing(function_name=processing_step, function_version=processing_step_version)
        job.configure_parameters(**parameters)
        job.start().wait_for_completion(timeout_s=timeout_s)

        result = job.get_result()
        debug["job_result"] = result

        # 1) Try to parse a QASM payload directly from the job result.
        for qasm in _iter_possible_qasm_payloads(result):
            try:
                qc = try_circuit_from_qasm_str(qasm)
                return qc, debug
            except Exception:  # noqa: BLE001
                continue

        # 2) Try to download output WorkData and parse it as QASM (common for larger outputs).
        if try_output_workdata:
            for slot in job.get_output_data_slots():
                for wd_hco in slot.assigned_workdatas:
                    wd = WorkData.from_hco(wd_hco)
                    payload = wd.download()
                    debug["output_workdata_candidates"].append(
                        {
                            "name": wd_hco.name,
                            "media_type": wd_hco.media_type,
                            "size_in_bytes": wd_hco.size_in_bytes,
                        }
                    )
                    try:
                        text = payload.decode("utf-8", errors="strict")
                    except UnicodeDecodeError:
                        continue
                    if "OPENQASM" in text.upper() or "qreg" in text:
                        qc = try_circuit_from_qasm_str(text)
                        return qc, debug

        raise RuntimeError(
            "PinexQ job completed but no circuit could be extracted. "
            "Inspect debug output (schemas + job result + output workdata candidates)."
        )
    finally:
        client.close()


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PinexQ circuit builder → Qiskit simulation demo")

    p.add_argument("--n-qubits", type=int, default=3)
    p.add_argument("--target", type=str, default="ghz", help="zero | bell | ghz | w")
    p.add_argument("--offline", action="store_true", help="Skip PinexQ and build a state-prep circuit locally with Qiskit.")

    p.add_argument("--processing-step", type=str, default="", help="PinexQ processing step function_name to run.")
    p.add_argument("--processing-step-version", type=str, default=None, help="Optional processing step version.")
    p.add_argument("--job-name", type=str, default="stateprep-demo")
    p.add_argument("--timeout-s", type=float, default=300.0)

    p.add_argument(
        "--params-json",
        type=str,
        default="",
        help="Raw JSON dict to pass as job parameters. If omitted, a default payload based on target statevector is used.",
    )

    p.add_argument("--list-processing-steps", action="store_true", help="List available processing steps and exit.")
    p.add_argument("--describe-processing-step", type=str, default="", help="Describe a processing step by function_name and exit.")

    p.add_argument("--print-qc", action="store_true", help="Print the circuit.")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    cfg = load_pinexq_config_from_env()

    if args.list_processing_steps:
        cfg = _require_pinexq(cfg)
        steps = pinexq_list_processing_steps(cfg)
        print(json.dumps(steps, indent=2))
        return 0

    if args.describe_processing_step:
        cfg = _require_pinexq(cfg)
        info = pinexq_describe_processing_step(cfg, args.describe_processing_step, args.processing_step_version)
        print(json.dumps(info, indent=2))
        return 0

    vec = target_statevector(args.target, args.n_qubits)

    if args.offline:
        qc = build_offline_stateprep_circuit(vec)
        debug = {"mode": "offline"}
    else:
        cfg = _require_pinexq(cfg)
        if not args.processing_step:
            raise SystemExit(
                "Missing --processing-step. Tip: run --list-processing-steps (with PINEXQ env vars set) "
                "and look for your Tucker/tensor circuit builder step."
            )

        if args.params_json:
            try:
                params = json.loads(args.params_json)
            except json.JSONDecodeError as e:
                raise SystemExit(f"--params-json is not valid JSON: {e}") from e
            if not isinstance(params, dict):
                raise SystemExit("--params-json must be a JSON object (dict).")
        else:
            # Generic default payload (your processing step may require different keys).
            params = {
                "n_qubits": int(args.n_qubits),
                "target_statevector": statevector_to_json(vec),
                "target_statevector_convention": "qiskit_little_endian",
                "requested_output": "openqasm",
            }

        qc, debug = pinexq_build_circuit_via_job(
            cfg,
            processing_step=args.processing_step,
            processing_step_version=args.processing_step_version,
            job_name=args.job_name,
            parameters=params,
            timeout_s=float(args.timeout_s) if args.timeout_s else None,
        )

    if args.print_qc:
        print(qc)

    # Simulate
    out_sv = simulate_statevector(qc)
    fid = fidelity_up_to_global_phase(out_sv, vec)

    print(json.dumps({"fidelity": fid, "debug": debug}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

