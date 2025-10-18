"""Hybrid quantum-AI simulation workflows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .ansatz import HardwareAgnosticAnsatz
from .hamiltonian import PauliHamiltonian


@dataclass(frozen=True)
class HybridSimulationStep:
    """Snapshot of one hybrid optimisation iteration."""

    iteration: int
    energy: float
    parameters: np.ndarray
    guidance: np.ndarray
    learning_rate: float


@dataclass(frozen=True)
class HybridSimulationReport:
    """Aggregate output of a hybrid quantum-AI optimisation run."""

    steps: tuple[HybridSimulationStep, ...]
    final_parameters: np.ndarray
    final_energy: float

    @property
    def converged(self) -> bool:
        if len(self.steps) < 2:
            return False
        last = self.steps[-1].energy
        prev = self.steps[-2].energy
        return abs(last - prev) < 1e-6

    @property
    def best_energy(self) -> float:
        return min(step.energy for step in self.steps) if self.steps else float("inf")


class HybridQuantumAISimulator:
    """Blend analytic gradients with adaptive AI guidance."""

    def __init__(
        self,
        *,
        hamiltonian: PauliHamiltonian,
        ansatz: HardwareAgnosticAnsatz,
        controller,
        max_iterations: int = 40,
        ai_guidance_strength: float = 0.5,
    ) -> None:
        self.hamiltonian = hamiltonian
        self.ansatz = ansatz
        self.controller = controller
        self.max_iterations = int(max_iterations)
        self.ai_guidance_strength = float(ai_guidance_strength)

    def _energy_gradient(self, parameters: np.ndarray, state: np.ndarray) -> np.ndarray:
        gradients = self.ansatz.parameter_gradient(parameters)
        ham_state = self.hamiltonian.apply(state)
        grad = np.zeros(parameters.shape, dtype=float)
        for idx, derivative in enumerate(gradients):
            grad[idx] = 2.0 * np.real(np.vdot(derivative, ham_state))
        return grad

    def _ai_guidance(self, gradient: np.ndarray, iteration: int) -> np.ndarray:
        norm = np.linalg.norm(gradient)
        if norm == 0:
            return np.zeros_like(gradient)
        schedule = 1.0 / (1.0 + iteration / 10.0)
        return -schedule * gradient / norm

    def simulate(
        self, initial_parameters: Optional[Sequence[float]] = None
    ) -> HybridSimulationReport:
        if initial_parameters is None:
            parameters = np.linspace(0.01, 0.02 * self.ansatz.parameter_count, self.ansatz.parameter_count)
        else:
            parameters = np.asarray(initial_parameters, dtype=float)
        self.controller.reset(self.ansatz.parameter_count)
        steps: List[HybridSimulationStep] = []

        for iteration in range(1, self.max_iterations + 1):
            state = self.ansatz.prepare_state(parameters)
            energy = self.hamiltonian.expectation(state)
            gradient = self._energy_gradient(parameters, state)
            guidance = self._ai_guidance(gradient, iteration)
            signal = self.controller.propose_update(gradient, energy)
            update_direction = signal.direction + self.ai_guidance_strength * guidance
            parameters = parameters + signal.learning_rate * update_direction
            steps.append(
                HybridSimulationStep(
                    iteration=iteration,
                    energy=float(energy),
                    parameters=parameters.copy(),
                    guidance=guidance.copy(),
                    learning_rate=float(signal.learning_rate),
                )
            )

        final_parameters = steps[-1].parameters.copy() if steps else parameters
        final_energy = steps[-1].energy if steps else float("inf")
        return HybridSimulationReport(
            steps=tuple(steps),
            final_parameters=final_parameters,
            final_energy=float(final_energy),
        )


__all__ = [
    "HybridQuantumAISimulator",
    "HybridSimulationReport",
    "HybridSimulationStep",
]
