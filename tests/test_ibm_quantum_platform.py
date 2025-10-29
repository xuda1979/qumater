"""Tests for the IBM Quantum platform integration helpers."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from qumater.platform import IBMQuantumPlatform, IBMQuantumPlatformError, SimpleCircuit


class DummyJob:
    def __init__(self, result: Mapping[str, Any]):
        self._result = result

    def result(self) -> Mapping[str, Any]:
        return self._result


class DummyService:
    def __init__(self):
        self.calls: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []

    def run(self, program_id: str, inputs: Mapping[str, Any], options: Mapping[str, Any]):
        self.calls.append((program_id, inputs, options))
        if program_id == "estimator":
            return DummyJob({"values": [0.42], "variances": [0.02]})
        return DummyJob({"quasi_dists": [{"0101": 0.7, "1111": 0.3}]})


def test_example_library_exposes_problem_domains():
    platform = IBMQuantumPlatform()
    examples = list(platform.list_examples())

    assert "portfolio_optimization" in examples
    assert "credit_risk_analysis" in examples
    assert "logistics_routing" in examples
    assert "carbon_capture_simulation" in examples
    assert "fraud_detection_classifier" in examples
    assert "drug_discovery_vqe" in examples
    assert len(examples) >= 7

    portfolio = platform.get_example("portfolio_optimization")
    circuit = portfolio.build_circuit()

    if isinstance(circuit, SimpleCircuit):
        assert circuit.num_qubits == 4
        assert circuit.operations[0] == "h q[0]"
    else:  # pragma: no cover - executed only when Qiskit is installed
        assert hasattr(circuit, "num_qubits")
        assert circuit.num_qubits == 4

    assert portfolio.metadata["business_value"].startswith("Optimise capital")


def test_runtime_payload_contains_metadata():
    platform = IBMQuantumPlatform()
    example = platform.get_example("logistics_routing")
    payload = example.build_runtime_payload()

    assert payload["metadata"]["business_value"].startswith("Balance transportation cost")
    assert "observables" in payload


def test_run_example_uses_provided_service():
    platform = IBMQuantumPlatform()
    service = DummyService()

    result = platform.run_example(
        "portfolio_optimization", service=service, runtime_options={"backend": "ibmq_qasm_simulator"}
    )

    assert service.calls[0][0] == "sampler"
    assert service.calls[0][2]["backend"] == "ibmq_qasm_simulator"
    assert result["most_likely_bitstring"] == "0101"


def test_run_example_decodes_estimator_results():
    platform = IBMQuantumPlatform()
    service = DummyService()

    result = platform.run_example("grid_load_balancing", service=service)

    assert service.calls[-1][0] == "estimator"
    assert pytest.approx(result["expectation"], rel=1e-6) == 0.42
    assert pytest.approx(result["variance"], rel=1e-6) == 0.02


def test_run_example_decodes_sampler_for_fraud_detection():
    platform = IBMQuantumPlatform()
    service = DummyService()

    result = platform.run_example("fraud_detection_classifier", service=service)

    assert service.calls[-1][0] == "sampler"
    assert result["most_likely_bitstring"] == "0101"


def test_ensure_service_requires_runtime():
    platform = IBMQuantumPlatform()

    with pytest.raises(IBMQuantumPlatformError):
        platform.ensure_service()

