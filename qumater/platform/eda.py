"""Digital quantum design automation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ..qsim.ansatz import HardwareAgnosticAnsatz
from ..qsim.hamiltonian import PauliHamiltonian


@dataclass
class CircuitStatistics:
    """Summary describing a synthesised variational circuit."""

    depth: int
    single_qubit_gates: int
    entangling_gates: int


class QuantumEDA:
    """Provide helper routines inspired by quantum EDA tooling."""

    def synthesise_ansatz(self, num_qubits: int, layers: int = 1) -> HardwareAgnosticAnsatz:
        """Construct a hardware agnostic ansatz for a given layout."""

        return HardwareAgnosticAnsatz(num_qubits=num_qubits, layers=layers)

    @staticmethod
    def estimate_statistics(ansatz: HardwareAgnosticAnsatz) -> CircuitStatistics:
        """Estimate depth and gate counts for *ansatz*."""

        depth = ansatz.layers * 2  # rotation block + entangling block
        single_qubit = ansatz.parameter_count
        entangling = ansatz.num_qubits * ansatz.layers
        return CircuitStatistics(depth=depth, single_qubit_gates=single_qubit, entangling_gates=entangling)

    def parameter_sweep(
        self,
        hamiltonian: PauliHamiltonian,
        ansatz: HardwareAgnosticAnsatz,
        grid: Iterable[Sequence[float]],
    ) -> tuple[np.ndarray, float]:
        """Evaluate *hamiltonian* across a grid of ansatz parameters."""

        best_energy = float("inf")
        best_parameters = None
        for parameters in grid:
            parameters = np.asarray(parameters, dtype=float)
            if parameters.size != ansatz.parameter_count:
                raise ValueError("Parameter grid entries must match ansatz.parameter_count")
            state = ansatz.prepare_state(parameters)
            energy = hamiltonian.expectation(state)
            if energy < best_energy:
                best_energy = float(energy)
                best_parameters = parameters
        if best_parameters is None:
            raise ValueError("Parameter grid must not be empty")
        return best_parameters, best_energy
