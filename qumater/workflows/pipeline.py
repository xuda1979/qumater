"""High level workflow orchestration building on configuration primitives."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from qumater.core import ObjectivePlanner, WorkflowConfig, default_objectives
from qumater.materials import MaterialEntry, QuantumMaterialDatabase
from qumater.qsim import HardwareAgnosticAnsatz, OptimizationHistory, get_algorithm_registry


@dataclass
class WorkflowReport:
    """Structured summary returned after executing a quantum workflow."""

    material: MaterialEntry
    algorithm_name: str
    algorithm_result: Any
    final_energy: Optional[float]
    converged: Optional[bool]
    steps: List[str]
    objective_summary: Dict[str, Dict[str, float]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON serialisable dictionary describing the report."""

        def convert(value: Any) -> Any:
            if isinstance(value, OptimizationHistory):
                return value.to_dict()
            if isinstance(value, np.ndarray):
                return value.tolist()
            if is_dataclass(value):
                return {key: convert(item) for key, item in asdict(value).items()}
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(item) for item in value]
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return repr(value)

        return {
            "material": convert(self.material),
            "algorithm_name": self.algorithm_name,
            "algorithm_result": convert(self.algorithm_result),
            "final_energy": self.final_energy,
            "converged": self.converged,
            "steps": list(self.steps),
            "objective_summary": convert(self.objective_summary),
            "metadata": convert(self.metadata),
        }


class QuantumWorkflow:
    """End-to-end workflow runner with explicit planning stages."""

    def __init__(
        self,
        config: WorkflowConfig,
        *,
        database: Optional[QuantumMaterialDatabase] = None,
        registry=None,
    ) -> None:
        self.config = config
        self.config.validate()
        self.database = database or QuantumMaterialDatabase.demo()
        self.registry = registry or get_algorithm_registry()
        self._planner = ObjectivePlanner(default_objectives())
        self._steps: List[str] = []

    @property
    def planner(self) -> ObjectivePlanner:
        return self._planner

    def plan(self) -> Iterable[str]:
        """Return the ordered list of task identifiers for the workflow."""

        return [task.name for objective in self._planner.objectives for task in objective.tasks]

    def _select_material(self) -> MaterialEntry:
        selection = self.config.material
        if selection.name:
            material = self.database.get(selection.name)
        elif selection.tags:
            matches = self.database.filter(tags=selection.tags)
            if not matches:
                raise LookupError(f"No material matches tags {selection.tags!r}")
            material = matches[0]
        else:  # pragma: no cover - protected by validation
            raise RuntimeError("Material selection is invalid")
        self._planner.mark_completed("material-catalogue-selection")
        self._steps.append(f"material:{material.name}")
        return material

    def _build_hamiltonian(self):
        hamiltonian = self.config.hamiltonian.build()
        self._planner.mark_completed("hamiltonian-build")
        self._steps.append("hamiltonian")
        return hamiltonian

    def _build_ansatz(self):
        ansatz_cfg = self.config.ansatz
        ansatz = HardwareAgnosticAnsatz(
            num_qubits=ansatz_cfg.num_qubits,
            layers=ansatz_cfg.layers,
        )
        self._planner.mark_completed("ansatz-construction")
        self._steps.append("ansatz")
        return ansatz

    def _build_algorithm(self, **dependencies):
        algorithm_cfg = self.config.algorithm
        algorithm = self.registry.create(
            algorithm_cfg.name,
            **dependencies,
            **algorithm_cfg.options,
        )
        self._planner.mark_completed("algorithm-instantiation")
        self._steps.append(f"algorithm:{algorithm_cfg.name}")
        return algorithm

    def execute(self) -> WorkflowReport:
        material = self._select_material()
        hamiltonian = self._build_hamiltonian()
        ansatz = self._build_ansatz()
        algorithm = self._build_algorithm(hamiltonian=hamiltonian, ansatz=ansatz)

        result = None
        final_energy: Optional[float] = None
        converged: Optional[bool] = None
        if hasattr(algorithm, "run"):
            run_options = dict(self.config.algorithm.run_options)
            result = algorithm.run(**run_options)
            if isinstance(result, OptimizationHistory):
                final_energy = float(result.energies[-1]) if result.energies else None
                converged = bool(result.converged)
        else:
            result = algorithm
        self._planner.mark_completed("result-reporting")
        self._steps.append("report")

        summary = self._planner.summary()
        metadata = {
            "material_parameters": dict(material.parameters),
            "ansatz_parameter_count": ansatz.parameter_count,
            "config_metadata": dict(self.config.metadata),
        }
        return WorkflowReport(
            material=material,
            algorithm_name=self.config.algorithm.name,
            algorithm_result=result,
            final_energy=final_energy,
            converged=converged,
            steps=list(self._steps),
            objective_summary=summary,
            metadata=metadata,
        )


__all__ = ["QuantumWorkflow", "WorkflowReport"]
