"""Variational quantum algorithms with natural gradient support."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .ansatz import HardwareAgnosticAnsatz
from .hamiltonian import PauliHamiltonian


@dataclass
class OptimizationHistory:
    """Container describing the optimisation trace."""

    parameters: List[np.ndarray]
    energies: List[float]
    converged: bool


class LowDepthVQE:
    """Low depth VQE with approximate quantum natural gradient updates.

    Phasecraft's public announcements emphasise *hardware-agnostic* compilation
    and fast convergence on NISQ era devices.  The implementation below mirrors
    those design goals by combining:

    * a depth-efficient ansatz (:class:`HardwareAgnosticAnsatz`);
    * measurement grouping to reduce estimator variance (handled by
      :class:`~qumater.qsim.hamiltonian.PauliHamiltonian`);
    * a light-weight natural gradient optimiser that uses the Fubini–Study
      metric tensor derived from the ansatz gradients.
    """

    def __init__(
        self,
        hamiltonian: PauliHamiltonian,
        ansatz: HardwareAgnosticAnsatz,
        *,
        learning_rate: float = 0.2,
        max_iterations: int = 200,
        tolerance: float = 1e-6,
        regularisation: float = 1e-6,
    ) -> None:
        self.hamiltonian = hamiltonian
        self.ansatz = ansatz
        self.learning_rate = float(learning_rate)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.regularisation = float(regularisation)

    def _energy_gradient(self, parameters: np.ndarray, state: np.ndarray) -> np.ndarray:
        ham_state = self.hamiltonian.apply(state)
        gradients = self.ansatz.parameter_gradient(parameters)
        grad = np.zeros(parameters.shape, dtype=float)
        for idx in range(gradients.shape[0]):
            derivative = gradients[idx]
            grad[idx] = 2.0 * np.real(np.vdot(derivative, ham_state))
        return grad

    def _metric_tensor(self, parameters: np.ndarray, state: np.ndarray) -> np.ndarray:
        gradients = self.ansatz.parameter_gradient(parameters)
        overlaps = gradients @ np.conjugate(state)
        gram = gradients @ np.conjugate(gradients.T)
        metric = np.real(gram - np.outer(overlaps, np.conjugate(overlaps)))
        metric += self.regularisation * np.eye(metric.shape[0])
        return metric

    def run(self, initial_parameters: Optional[Sequence[float]] = None) -> OptimizationHistory:
        if initial_parameters is None:
            initial_parameters = np.zeros(self.ansatz.parameter_count)
        parameters = np.asarray(initial_parameters, dtype=float)
        history_params: List[np.ndarray] = []
        history_energies: List[float] = []
        converged = False
        last_direction_norm = None

        for _ in range(self.max_iterations):
            state = self.ansatz.prepare_state(parameters)
            energy = self.hamiltonian.expectation(state)
            history_params.append(parameters.copy())
            history_energies.append(float(energy))

            gradient = self._energy_gradient(parameters, state)
            metric = self._metric_tensor(parameters, state)
            try:
                direction = np.linalg.solve(metric, gradient)
            except np.linalg.LinAlgError:
                direction = gradient

            update_norm = np.linalg.norm(direction)
            last_direction_norm = update_norm
            parameters = parameters - self.learning_rate * direction

            if update_norm * self.learning_rate < self.tolerance:
                converged = True
                break

        if not converged:
            # record final state if loop exited without break
            state = self.ansatz.prepare_state(parameters)
            energy = self.hamiltonian.expectation(state)
            history_params.append(parameters.copy())
            history_energies.append(float(energy))

            gradient = self._energy_gradient(parameters, state)
            gradient_norm = np.linalg.norm(gradient)
            energy_delta = (
                abs(history_energies[-1] - history_energies[-2])
                if len(history_energies) >= 2
                else None
            )
            if (
                gradient_norm < self.tolerance * 10
                or (
                    energy_delta is not None
                    and energy_delta < self.tolerance * 10
                )
                or (
                    last_direction_norm is not None
                    and last_direction_norm * self.learning_rate < self.tolerance * 10
                )
            ):
                converged = True

        return OptimizationHistory(history_params, history_energies, converged)


__all__ = ["LowDepthVQE", "OptimizationHistory"]
