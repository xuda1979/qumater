"""Quantum program development helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ..qsim.algorithms import LowDepthVQE
from ..qsim.ansatz import HardwareAgnosticAnsatz
from ..qsim.hamiltonian import PauliHamiltonian, PauliTerm


@dataclass
class CompilationResult:
    """Result of compiling a high level program description."""

    hamiltonian: PauliHamiltonian
    metadata: dict[str, float | int | str]


class ProgramDevelopmentPlatform:
    """Utilities that mimic a quantum programming environment."""

    def compile_to_hamiltonian(self, specification: Iterable[tuple[complex, str]]) -> CompilationResult:
        """Compile a sequence of weighted Pauli strings into a Hamiltonian."""

        terms = [PauliTerm(coefficient=coeff, pauli_string=pauli) for coeff, pauli in specification]
        hamiltonian = PauliHamiltonian(terms)
        metadata = {"num_terms": len(terms), "num_qubits": hamiltonian.num_qubits}
        return CompilationResult(hamiltonian=hamiltonian, metadata=metadata)

    def build_variational_program(
        self,
        hamiltonian: PauliHamiltonian,
        ansatz: HardwareAgnosticAnsatz,
        *,
        learning_rate: float = 0.2,
        max_iterations: int = 120,
    ) -> LowDepthVQE:
        """Instantiate a :class:`~qumater.qsim.algorithms.LowDepthVQE` solver."""

        return LowDepthVQE(
            hamiltonian=hamiltonian,
            ansatz=ansatz,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
        )

    @staticmethod
    def validate_parameters(ansatz: HardwareAgnosticAnsatz, parameters: Sequence[float]) -> np.ndarray:
        """Validate parameter arrays emitted by developer tooling."""

        parameters = np.asarray(parameters, dtype=float)
        if parameters.size != ansatz.parameter_count:
            raise ValueError(
                f"Parameter count mismatch: expected {ansatz.parameter_count}, received {parameters.size}"
            )
        return parameters
