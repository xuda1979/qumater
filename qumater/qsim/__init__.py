"""Quantum simulation utilities inspired by hardware-agnostic SOTA methods."""

from .ansatz import HardwareAgnosticAnsatz
from .algorithms import (
    GroverResult,
    GroverSearch,
    LowDepthVQE,
    OptimizationHistory,
    PhaseEstimationResult,
    QuantumFourierTransform,
    QuantumPhaseEstimation,
)
from .circuit_optimization import (
    CircuitOptimisationResult,
    CircuitOptimiser,
    CircuitProfile,
    GateSpec,
)
from .hamiltonian import PauliHamiltonian, PauliTerm, group_commuting_terms
from .hybrid import HybridQuantumAISimulator, HybridSimulationReport, HybridSimulationStep
from .modules import (
    AlgorithmModule,
    AlgorithmRegistry,
    ENTRY_POINT_GROUP,
    get_algorithm_registry,
    load_entry_point_modules,
    register_algorithm_module,
)

__all__ = [
    "HardwareAgnosticAnsatz",
    "GroverResult",
    "GroverSearch",
    "LowDepthVQE",
    "OptimizationHistory",
    "PhaseEstimationResult",
    "QuantumFourierTransform",
    "QuantumPhaseEstimation",
    "CircuitOptimisationResult",
    "CircuitOptimiser",
    "CircuitProfile",
    "GateSpec",
    "PauliHamiltonian",
    "PauliTerm",
    "group_commuting_terms",
    "HybridQuantumAISimulator",
    "HybridSimulationReport",
    "HybridSimulationStep",
    "AlgorithmModule",
    "AlgorithmRegistry",
    "ENTRY_POINT_GROUP",
    "get_algorithm_registry",
    "load_entry_point_modules",
    "register_algorithm_module",
]
