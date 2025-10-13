"""Pauli Hamiltonian utilities with measurement grouping."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Sequence

import numpy as np

PAULI_MATRICES = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


@dataclass(frozen=True)
class PauliTerm:
    """A weighted Pauli string."""

    coefficient: complex
    pauli_string: str

    def __post_init__(self) -> None:
        if not self.pauli_string:
            raise ValueError("pauli_string must be non-empty")
        for char in self.pauli_string.upper():
            if char not in PAULI_MATRICES:
                raise ValueError(f"Invalid Pauli operator '{char}'")

    @property
    def num_qubits(self) -> int:
        return len(self.pauli_string)

    def matrix(self) -> np.ndarray:
        matrices = [PAULI_MATRICES[char] for char in self.pauli_string.upper()]
        result = matrices[0]
        for matrix in matrices[1:]:
            result = np.kron(result, matrix)
        return self.coefficient * result


def _pauli_commutes(p1: str, p2: str) -> bool:
    anti_commuting = 0
    for a, b in zip(p1, p2):
        if a == "I" or b == "I" or a == b:
            continue
        if {a, b} in ({"X", "Y"}, {"Y", "Z"}, {"X", "Z"}):
            anti_commuting += 1
    return anti_commuting % 2 == 0


def group_commuting_terms(terms: Sequence[PauliTerm]) -> List[List[PauliTerm]]:
    """Greedy grouping of mutually commuting Pauli terms."""

    groups: List[List[PauliTerm]] = []
    for term in terms:
        placed = False
        for group in groups:
            if all(_pauli_commutes(term.pauli_string, other.pauli_string) for other in group):
                group.append(term)
                placed = True
                break
        if not placed:
            groups.append([term])
    return groups


class PauliHamiltonian:
    """Collection of Pauli terms with helper methods for VQE style workloads."""

    def __init__(self, terms: Iterable[PauliTerm]):
        terms = list(terms)
        if not terms:
            raise ValueError("Hamiltonian requires at least one term")
        num_qubits = {term.num_qubits for term in terms}
        if len(num_qubits) != 1:
            raise ValueError("All terms must act on the same number of qubits")
        self._num_qubits = num_qubits.pop()
        self._terms = terms
        self._matrix_cache: np.ndarray | None = None

    @property
    def terms(self) -> Sequence[PauliTerm]:
        return tuple(self._terms)

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    def matrix(self) -> np.ndarray:
        if self._matrix_cache is None:
            matrix = np.zeros((2**self._num_qubits, 2**self._num_qubits), dtype=complex)
            for term in self._terms:
                matrix += term.matrix()
            self._matrix_cache = matrix
        return self._matrix_cache

    def expectation(self, state: np.ndarray) -> float:
        """Return the expectation value of the Hamiltonian with respect to *state*."""

        state = np.asarray(state, dtype=complex)
        if state.ndim != 1 or state.size != 2**self._num_qubits:
            raise ValueError("State has incompatible shape")
        groups = group_commuting_terms(self._terms)
        energy = 0.0
        for group in groups:
            matrix = np.zeros((state.size, state.size), dtype=complex)
            for term in group:
                matrix += term.matrix()
            energy += np.real(np.vdot(state, matrix @ state))
        return float(energy)

    def apply(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=complex)
        if state.ndim != 1 or state.size != 2**self._num_qubits:
            raise ValueError("State has incompatible shape")
        result = np.zeros_like(state)
        for term in self._terms:
            result += term.matrix() @ state
        return result

    def variance(self, state: np.ndarray) -> float:
        """Return the energy variance ``⟨H²⟩ - ⟨H⟩²`` for *state*.

        The variance quantifies the quality of a variational eigenstate
        candidate and mirrors the metrics used in industrial VQE benchmarks.  A
        vanishing variance indicates that the supplied state is an eigenstate of
        the Hamiltonian.
        """

        state = np.asarray(state, dtype=complex)
        if state.ndim != 1 or state.size != 2**self._num_qubits:
            raise ValueError("State has incompatible shape")

        expectation_value = self.expectation(state)
        ham_state = self.apply(state)
        h2_expectation = np.real(np.vdot(ham_state, ham_state))
        variance = h2_expectation - expectation_value**2
        if variance < 0 and abs(variance) < 1e-12:
            variance = 0.0
        return float(variance)


__all__ = ["PauliTerm", "PauliHamiltonian", "group_commuting_terms"]
