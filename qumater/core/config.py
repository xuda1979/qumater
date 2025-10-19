"""Configuration primitives for orchestrating end-to-end quantum workflows."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

from qumater.qsim import PauliHamiltonian, PauliTerm


@dataclass
class MaterialSelection:
    """Describe how a workflow should identify a material entry."""

    name: Optional[str] = None
    tags: Tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.name and not self.tags:
            raise ValueError("MaterialSelection requires either a name or tags")

    def to_query(self) -> Dict[str, Any]:
        query: Dict[str, Any] = {}
        if self.name:
            query["name"] = self.name
        if self.tags:
            query["tags"] = list(self.tags)
        return query


@dataclass
class AnsatzConfig:
    """Configuration for hardware-agnostic ansatz construction."""

    num_qubits: int
    layers: int = 1

    def validate(self) -> None:
        if self.num_qubits < 1:
            raise ValueError("AnsatzConfig.num_qubits must be positive")
        if self.layers < 1:
            raise ValueError("AnsatzConfig.layers must be positive")


@dataclass
class AlgorithmConfig:
    """Describe which algorithm to instantiate and how to run it."""

    name: str
    options: MutableMapping[str, Any] = field(default_factory=dict)
    run_options: MutableMapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("AlgorithmConfig.name must not be empty")


@dataclass
class HamiltonianConfig:
    """Serialisable description of a Pauli Hamiltonian."""

    terms: Sequence[Tuple[float, str]]

    def validate(self) -> None:
        if not self.terms:
            raise ValueError("HamiltonianConfig.terms must contain at least one term")
        for weight, pauli_string in self.terms:
            if not isinstance(weight, (float, int)):
                raise TypeError("Hamiltonian term weights must be numeric")
            if not pauli_string:
                raise ValueError("Pauli string must not be empty")

    def build(self) -> PauliHamiltonian:
        self.validate()
        pauli_terms = [PauliTerm(float(weight), pauli) for weight, pauli in self.terms]
        return PauliHamiltonian(pauli_terms)


@dataclass
class WorkflowConfig:
    """Container aggregating all configuration fragments required for a run."""

    material: MaterialSelection
    hamiltonian: HamiltonianConfig
    ansatz: AnsatzConfig
    algorithm: AlgorithmConfig
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.material.validate()
        self.hamiltonian.validate()
        self.ansatz.validate()
        self.algorithm.validate()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkflowConfig":
        def _require(section: str) -> Mapping[str, Any]:
            if section not in data:
                raise KeyError(f"Missing '{section}' section in workflow configuration")
            value = data[section]
            if not isinstance(value, Mapping):
                raise TypeError(f"Section '{section}' must be a mapping")
            return value

        material_data = _require("material")
        hamiltonian_data = _require("hamiltonian")
        ansatz_data = _require("ansatz")
        algorithm_data = _require("algorithm")

        material = MaterialSelection(
            name=material_data.get("name"),
            tags=tuple(material_data.get("tags", ())) if material_data.get("tags") else (),
        )
        terms_raw = hamiltonian_data.get("terms")
        if not isinstance(terms_raw, Iterable):
            raise TypeError("Hamiltonian terms must be an iterable of [weight, pauli] pairs")
        terms: list[Tuple[float, str]] = []
        for idx, item in enumerate(terms_raw):
            if not isinstance(item, Iterable):
                raise TypeError(f"Hamiltonian term #{idx} must be iterable")
            try:
                weight, pauli = item  # type: ignore[misc]
            except ValueError as exc:  # unpacking errors
                raise ValueError("Hamiltonian terms must contain exactly two items") from exc
            terms.append((float(weight), str(pauli)))

        ansatz = AnsatzConfig(
            num_qubits=int(ansatz_data.get("num_qubits")),
            layers=int(ansatz_data.get("layers", 1)),
        )
        algorithm = AlgorithmConfig(
            name=str(algorithm_data.get("name")),
            options=dict(algorithm_data.get("options", {})),
            run_options=dict(algorithm_data.get("run", {})),
        )

        metadata = dict(data.get("metadata", {}))
        config = cls(
            material=material,
            hamiltonian=HamiltonianConfig(tuple(terms)),
            ansatz=ansatz,
            algorithm=algorithm,
            metadata=metadata,
        )
        config.validate()
        return config

    def to_dict(self) -> Dict[str, Any]:
        return {
            "material": self.material.to_query(),
            "hamiltonian": {"terms": list(self.hamiltonian.terms)},
            "ansatz": {
                "num_qubits": self.ansatz.num_qubits,
                "layers": self.ansatz.layers,
            },
            "algorithm": {
                "name": self.algorithm.name,
                "options": dict(self.algorithm.options),
                "run": dict(self.algorithm.run_options),
            },
            "metadata": dict(self.metadata),
        }


__all__ = [
    "MaterialSelection",
    "AnsatzConfig",
    "AlgorithmConfig",
    "HamiltonianConfig",
    "WorkflowConfig",
]
