"""Measurement orchestration utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from ..qsim.hamiltonian import PauliHamiltonian, PauliTerm, group_commuting_terms


@dataclass
class MeasurementSchedule:
    """Description of a grouped measurement routine."""

    groups: tuple[tuple[PauliTerm, ...], ...]
    total_groups: int


class MeasurementControlSystem:
    """Produce deterministic measurement strategies."""

    def create_schedule(self, hamiltonian: PauliHamiltonian) -> MeasurementSchedule:
        groups = tuple(tuple(group) for group in group_commuting_terms(hamiltonian.terms))
        return MeasurementSchedule(groups=groups, total_groups=len(groups))

    def simulate_expectations(
        self,
        hamiltonian: PauliHamiltonian,
        state: Sequence[complex],
    ) -> Dict[str, float]:
        state = np.asarray(state, dtype=complex)
        expectations: Dict[str, float] = {}
        for term in hamiltonian.terms:
            matrix = term.matrix()
            expectations[term.pauli_string] = float(np.real(np.vdot(state, matrix @ state)))
        return expectations
