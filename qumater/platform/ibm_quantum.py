"""Utilities for orchestrating examples on the IBM Quantum platform.

The real IBM Quantum service is only available when the optional
``qiskit-ibm-runtime`` dependency is installed.  The helpers below degrade
gracefully when Qiskit is missing, returning lightweight placeholders that can
still be inspected, serialised and unit tested.  This mirrors how application
teams often prototype workflows before connecting to live hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional

try:  # pragma: no cover - optional import used when available
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import EfficientSU2, RealAmplitudes
    from qiskit.quantum_info import SparsePauliOp
except Exception:  # pragma: no cover - the tests exercise the fallback path
    QuantumCircuit = None  # type: ignore[assignment]
    EfficientSU2 = None  # type: ignore[assignment]
    RealAmplitudes = None  # type: ignore[assignment]
    SparsePauliOp = None  # type: ignore[assignment]


class IBMQuantumPlatformError(RuntimeError):
    """Raised when IBM Quantum integration prerequisites are not satisfied."""


@dataclass
class SimpleCircuit:
    """Minimal circuit description used when Qiskit is unavailable.

    The representation intentionally mirrors the structure that IBM Quantum's
    runtime clients expect (a gate sequence over a fixed number of qubits).
    It is serialisable and easy to inspect within documentation examples or
    unit tests, while avoiding the heavy Qiskit dependency for contributors
    focusing on the classical parts of the stack.
    """

    num_qubits: int
    description: str
    operations: list[str] = field(default_factory=list)

    def summary(self) -> Mapping[str, Any]:
        """Return a JSON-serialisable overview of the circuit."""

        return {
            "num_qubits": self.num_qubits,
            "description": self.description,
            "operations": list(self.operations),
        }


def _build_portfolio_circuit() -> Any:
    """Create a circuit that prepares a QAOA ansatz for portfolio optimisation."""

    if QuantumCircuit is not None and EfficientSU2 is not None:
        feature_map = EfficientSU2(num_qubits=4, reps=2, entanglement="linear")
        circuit = QuantumCircuit(feature_map.num_qubits, name="portfolio-qaoa")
        circuit.compose(feature_map, inplace=True)
        return circuit

    return SimpleCircuit(
        num_qubits=4,
        description="QAOA-style layers for a 4-asset portfolio selection",
        operations=[
            "h q[0]",
            "h q[1]",
            "h q[2]",
            "h q[3]",
            "cz q[0],q[1]",
            "cz q[1],q[2]",
            "cz q[2],q[3]",
            "rz(0.7) q[0]",
            "rz(0.7) q[1]",
            "rz(0.7) q[2]",
            "rz(0.7) q[3]",
        ],
    )


def _build_logistics_circuit() -> Any:
    """Create a circuit representing a supply chain routing problem."""

    if QuantumCircuit is not None and RealAmplitudes is not None:
        ansatz = RealAmplitudes(num_qubits=3, reps=2, entanglement="pairwise")
        circuit = QuantumCircuit(ansatz.num_qubits, name="logistics-routing")
        circuit.compose(ansatz, inplace=True)
        return circuit

    return SimpleCircuit(
        num_qubits=3,
        description="Warm-start circuit for quantum supply chain routing",
        operations=[
            "ry(1.2) q[0]",
            "ry(0.8) q[1]",
            "ry(0.5) q[2]",
            "cx q[0],q[1]",
            "cx q[1],q[2]",
            "rz(0.3) q[0]",
            "rz(0.3) q[1]",
            "rz(0.3) q[2]",
        ],
    )


def _build_credit_risk_circuit() -> Any:
    """Circuit encoding correlated credit risk factors."""

    if QuantumCircuit is not None:
        circuit = QuantumCircuit(3, name="credit-risk")
        for qubit in range(3):
            circuit.h(qubit)
        circuit.cz(0, 1)
        circuit.cx(1, 2)
        circuit.rz(0.5, 0)
        circuit.ry(0.3, 1)
        circuit.rz(0.5, 2)
        return circuit

    return SimpleCircuit(
        num_qubits=3,
        description="Amplitude encoding of correlated credit risk factors",
        operations=[
            "h q[0]",
            "h q[1]",
            "h q[2]",
            "cz q[0],q[1]",
            "cx q[1],q[2]",
            "rz(0.5) q[0]",
            "ry(0.3) q[1]",
            "rz(0.5) q[2]",
        ],
    )


def _build_grid_balancing_circuit() -> Any:
    """Ansatz approximating load balancing decisions for smart grids."""

    if QuantumCircuit is not None and EfficientSU2 is not None:
        feature_map = EfficientSU2(num_qubits=4, reps=1, entanglement="full")
        circuit = QuantumCircuit(feature_map.num_qubits, name="grid-balancing")
        circuit.compose(feature_map, inplace=True)
        return circuit

    return SimpleCircuit(
        num_qubits=4,
        description="Layered entanglement exploring smart grid load distributions",
        operations=[
            "ry(0.4) q[0]",
            "ry(0.4) q[1]",
            "ry(0.4) q[2]",
            "ry(0.4) q[3]",
            "cz q[0],q[1]",
            "cz q[0],q[2]",
            "cz q[0],q[3]",
            "rz(0.2) q[1]",
            "rz(0.2) q[2]",
            "rz(0.2) q[3]",
        ],
    )


def _build_fraud_detection_circuit() -> Any:
    """Circuit performing feature map embedding for fraud detection."""

    if QuantumCircuit is not None:
        circuit = QuantumCircuit(2, name="fraud-detection")
        circuit.h(0)
        circuit.ry(1.1, 1)
        circuit.cx(0, 1)
        circuit.rz(0.9, 0)
        circuit.ry(0.4, 1)
        circuit.cx(1, 0)
        return circuit

    return SimpleCircuit(
        num_qubits=2,
        description="Feature map embedding for binary fraud classification",
        operations=[
            "h q[0]",
            "ry(1.1) q[1]",
            "cx q[0],q[1]",
            "rz(0.9) q[0]",
            "ry(0.4) q[1]",
            "cx q[1],q[0]",
        ],
    )


def _build_drug_discovery_circuit() -> Any:
    """Circuit exploring molecular conformations for drug discovery."""

    if QuantumCircuit is not None and RealAmplitudes is not None:
        ansatz = RealAmplitudes(num_qubits=3, reps=3, entanglement="circular")
        circuit = QuantumCircuit(ansatz.num_qubits, name="drug-discovery")
        circuit.compose(ansatz, inplace=True)
        return circuit

    return SimpleCircuit(
        num_qubits=3,
        description="Variational ansatz for molecular conformations",
        operations=[
            "ry(0.6) q[0]",
            "ry(0.6) q[1]",
            "ry(0.6) q[2]",
            "cz q[0],q[1]",
            "cz q[1],q[2]",
            "rz(0.4) q[0]",
            "rz(0.4) q[1]",
            "rz(0.4) q[2]",
            "cx q[0],q[2]",
        ],
    )


def _build_energy_observable() -> Any:
    """Return an observable measuring parity interactions for energy estimation."""

    if SparsePauliOp is not None:
        return SparsePauliOp.from_list([
            ("ZZII", 0.8),
            ("IIZZ", -0.6),
            ("ZIZI", 0.5),
        ])

    return {
        "observable": "0.8 ZZII - 0.6 IIZZ + 0.5 ZIZI",
        "format": "pauli_list",
    }


def _build_credit_risk_observable() -> Any:
    """Observable for correlated default probability estimation."""

    if SparsePauliOp is not None:
        return SparsePauliOp.from_list([
            ("ZIZ", 0.7),
            ("IZZ", -0.4),
            ("ZZI", 0.6),
        ])

    return {
        "observable": "0.7 ZIZ - 0.4 IZZ + 0.6 ZZI",
        "format": "pauli_list",
    }


def _build_drug_discovery_observable() -> Any:
    """Observable approximating molecular energy terms for VQE."""

    if SparsePauliOp is not None:
        return SparsePauliOp.from_list([
            ("ZZI", -1.2),
            ("IZZ", 0.9),
            ("ZIZ", 0.5),
            ("XIX", -0.3),
        ])

    return {
        "observable": "-1.2 ZZI + 0.9 IZZ + 0.5 ZIZ - 0.3 XIX",
        "format": "pauli_list",
    }


def _build_climate_circuit() -> Any:
    """Circuit evaluating carbon capture catalyst configurations."""

    if QuantumCircuit is not None:
        circuit = QuantumCircuit(2, name="carbon-capture")
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.ry(0.4, 0)
        circuit.ry(0.4, 1)
        return circuit

    return SimpleCircuit(
        num_qubits=2,
        description="Entangled ansatz probing catalyst conformations",
        operations=[
            "h q[0]",
            "cx q[0],q[1]",
            "ry(0.4) q[0]",
            "ry(0.4) q[1]",
        ],
    )


ResultDecoder = Callable[[Any], Mapping[str, Any]]
RuntimeCallable = Callable[[str, MutableMapping[str, Any], MutableMapping[str, Any]], Any]


@dataclass
class IBMQuantumExample:
    """Describes a self-contained IBM Quantum runtime example."""

    name: str
    description: str
    problem_domain: str
    algorithm: str
    runtime_program: str
    circuit_builder: Callable[[], Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    observable_builder: Optional[Callable[[], Any]] = None
    postprocess: Optional[Callable[[Any], Mapping[str, Any]]] = None

    def build_circuit(self) -> Any:
        """Generate the quantum circuit for the example."""

        return self.circuit_builder()

    def build_runtime_payload(self) -> MutableMapping[str, Any]:
        """Return the payload compatible with IBM Quantum runtime programs."""

        payload: MutableMapping[str, Any] = {
            "circuits": [self.build_circuit()],
            "metadata": dict(self.metadata),
        }
        if self.observable_builder is not None:
            payload["observables"] = [self.observable_builder()]
        return payload

    def interpret(self, raw_result: Any) -> Mapping[str, Any]:
        """Interpret raw results returned by the runtime service."""

        if self.postprocess is None:
            if isinstance(raw_result, Mapping):
                return raw_result  # type: ignore[return-value]
            return {"result": raw_result}
        return self.postprocess(raw_result)


class IBMQuantumPlatform:
    """Facade for curating IBM Quantum runtime examples within QuMater."""

    def __init__(self, service: Optional[Any] = None):
        self._service = service
        self._examples = self._build_examples()

    # ------------------------------------------------------------------
    # Example library helpers
    # ------------------------------------------------------------------
    def _build_examples(self) -> Dict[str, IBMQuantumExample]:
        examples = [
            IBMQuantumExample(
                name="portfolio_optimization",
                description=(
                    "Map a four-asset portfolio selection problem to a QAOA "
                    "ansatz and execute it via the Sampler runtime program."
                ),
                problem_domain="Finance",
                algorithm="QAOA",
                runtime_program="sampler",
                circuit_builder=_build_portfolio_circuit,
                metadata={
                    "business_value": "Optimise capital allocation under risk constraints",
                    "shots": 2048,
                },
                postprocess=self._decode_sampler_distribution,
            ),
            IBMQuantumExample(
                name="credit_risk_analysis",
                description=(
                    "Estimate correlated default probabilities using an ansatz "
                    "tailored for amplitude-style risk aggregation."
                ),
                problem_domain="Finance",
                algorithm="Amplitude estimation",
                runtime_program="estimator",
                circuit_builder=_build_credit_risk_circuit,
                metadata={
                    "business_value": "Quantify joint credit exposure across counterparties",
                },
                observable_builder=_build_credit_risk_observable,
                postprocess=self._decode_estimator_expectation,
            ),
            IBMQuantumExample(
                name="logistics_routing",
                description=(
                    "Demonstrate constrained route selection for supply chains "
                    "using a warm-start VQE configured for the Estimator runtime."
                ),
                problem_domain="Supply chain",
                algorithm="VQE",
                runtime_program="estimator",
                circuit_builder=_build_logistics_circuit,
                metadata={
                    "business_value": "Balance transportation cost against on-time delivery",
                },
                observable_builder=_build_energy_observable,
                postprocess=self._decode_estimator_expectation,
            ),
            IBMQuantumExample(
                name="grid_load_balancing",
                description=(
                    "Optimise smart grid load balancing decisions under "
                    "renewable generation uncertainty with a variational ansatz."
                ),
                problem_domain="Energy",
                algorithm="VQE",
                runtime_program="estimator",
                circuit_builder=_build_grid_balancing_circuit,
                metadata={
                    "business_value": "Stabilise grid frequency while integrating renewables",
                },
                observable_builder=_build_energy_observable,
                postprocess=self._decode_estimator_expectation,
            ),
            IBMQuantumExample(
                name="carbon_capture_simulation",
                description=(
                    "Explore catalytic site configurations relevant to carbon "
                    "capture using short-depth entangling layers."
                ),
                problem_domain="Climate science",
                algorithm="State tomography",
                runtime_program="sampler",
                circuit_builder=_build_climate_circuit,
                metadata={
                    "business_value": "Screen catalyst candidates for greenhouse gas mitigation",
                    "shots": 1024,
                },
                postprocess=self._decode_sampler_distribution,
            ),
            IBMQuantumExample(
                name="fraud_detection_classifier",
                description=(
                    "Embed transaction features into a shallow circuit to "
                    "illustrate fraud classification workflows on the sampler."
                ),
                problem_domain="Financial crime",
                algorithm="Quantum kernel methods",
                runtime_program="sampler",
                circuit_builder=_build_fraud_detection_circuit,
                metadata={
                    "business_value": "Detect anomalous payment activity in near real time",
                    "shots": 4096,
                },
                postprocess=self._decode_sampler_distribution,
            ),
            IBMQuantumExample(
                name="drug_discovery_vqe",
                description=(
                    "Approximate molecular energy landscapes for drug discovery "
                    "pipelines via the Estimator runtime."
                ),
                problem_domain="Life sciences",
                algorithm="VQE",
                runtime_program="estimator",
                circuit_builder=_build_drug_discovery_circuit,
                metadata={
                    "business_value": "Prioritise candidate molecules with promising binding profiles",
                },
                observable_builder=_build_drug_discovery_observable,
                postprocess=self._decode_estimator_expectation,
            ),
        ]

        return {example.name: example for example in examples}

    def list_examples(self) -> Iterable[str]:
        """Return the names of the available IBM Quantum examples."""

        return self._examples.keys()

    def get_example(self, name: str) -> IBMQuantumExample:
        """Retrieve a registered example by name."""

        try:
            return self._examples[name]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise IBMQuantumPlatformError(f"Unknown IBM Quantum example: {name}") from exc

    # ------------------------------------------------------------------
    # Service interaction
    # ------------------------------------------------------------------
    def ensure_service(self) -> Any:
        """Ensure an IBM Quantum runtime service instance is available."""

        if self._service is not None:
            return self._service

        raise IBMQuantumPlatformError(
            "No IBM Quantum service provided. Install `qiskit-ibm-runtime` and "
            "initialise `QiskitRuntimeService` with valid credentials."
        )

    def run_example(
        self,
        name: str,
        *,
        service: Optional[Any] = None,
        runtime_options: Optional[MutableMapping[str, Any]] = None,
        result_decoder: Optional[ResultDecoder] = None,
    ) -> Mapping[str, Any]:
        """Execute an example using the provided or default runtime service."""

        example = self.get_example(name)
        runtime_payload = example.build_runtime_payload()
        runtime_options = runtime_options or {}

        runtime_service = service or self.ensure_service()
        run_callable: RuntimeCallable = getattr(runtime_service, "run")  # type: ignore[assignment]

        job = run_callable(example.runtime_program, runtime_payload, runtime_options)
        if hasattr(job, "result"):
            raw_result = job.result()
        else:
            raw_result = job

        if result_decoder is not None:
            return result_decoder(raw_result)
        return example.interpret(raw_result)

    # ------------------------------------------------------------------
    # Result decoding helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _decode_sampler_distribution(result: Any) -> Mapping[str, Any]:
        """Convert Sampler results into a friendly dictionary."""

        if isinstance(result, Mapping) and "quasi_dists" in result:
            quasi = result["quasi_dists"][0]
            return {
                "most_likely_bitstring": max(quasi, key=quasi.get),
                "distribution": dict(quasi),
            }
        return {"distribution": result}

    @staticmethod
    def _decode_estimator_expectation(result: Any) -> Mapping[str, Any]:
        """Convert Estimator expectation values into a structured payload."""

        if isinstance(result, Mapping) and "values" in result:
            return {
                "expectation": float(result["values"][0]),
                "variance": float(result.get("variances", [0.0])[0]),
            }
        return {"expectation": result}


__all__ = [
    "IBMQuantumPlatform",
    "IBMQuantumExample",
    "SimpleCircuit",
    "IBMQuantumPlatformError",
]

