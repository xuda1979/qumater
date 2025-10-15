"""Quantum machine learning helper utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ..qsim.ansatz import HardwareAgnosticAnsatz
from ..qsim.hamiltonian import PauliHamiltonian


@dataclass
class QuantumRegressor:
    """Linear feature map whose outputs are variational expectations."""

    ansatz: HardwareAgnosticAnsatz
    hamiltonian: PauliHamiltonian
    weights: np.ndarray
    bias: np.ndarray

    def predict(self, features: Sequence[float]) -> float:
        features = np.asarray(features, dtype=float)
        parameters = self.bias + self.weights @ features
        state = self.ansatz.prepare_state(parameters)
        return self.hamiltonian.expectation(state)


class QuantumMachineLearningPlatform:
    """Gradient based learners operating on variational expectations."""

    def __init__(self, *, learning_rate: float = 0.1, epochs: int = 200) -> None:
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)

    def _expectation_and_gradient(
        self,
        ansatz: HardwareAgnosticAnsatz,
        hamiltonian: PauliHamiltonian,
        parameters: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        state = ansatz.prepare_state(parameters)
        energy = hamiltonian.expectation(state)
        gradients = ansatz.parameter_gradient(parameters)
        ham_state = hamiltonian.matrix() @ state
        grad = np.zeros(parameters.shape, dtype=float)
        for idx in range(gradients.shape[0]):
            grad[idx] = 2.0 * np.real(np.vdot(gradients[idx], ham_state))
        return energy, grad

    def fit_expectation_regressor(
        self,
        dataset: Iterable[tuple[Sequence[float], float]],
        ansatz: HardwareAgnosticAnsatz,
        hamiltonian: PauliHamiltonian,
    ) -> QuantumRegressor:
        dataset = [(np.asarray(x, dtype=float), float(y)) for x, y in dataset]
        if not dataset:
            raise ValueError("Dataset must contain at least one sample")
        feature_dim = dataset[0][0].size
        param_count = ansatz.parameter_count
        weights = np.zeros((param_count, feature_dim), dtype=float)
        bias = np.linspace(0.01, 0.01 * param_count, param_count, dtype=float)

        for epoch in range(self.epochs):
            grad_w = np.zeros_like(weights)
            grad_b = np.zeros_like(bias)
            for features, target in dataset:
                parameters = bias + weights @ features
                energy, grad_theta = self._expectation_and_gradient(ansatz, hamiltonian, parameters)
                error = energy - target
                grad_common = 2.0 * error * grad_theta
                grad_b += grad_common
                grad_w += np.outer(grad_common, features)
            weights -= (self.learning_rate / len(dataset)) * grad_w
            bias -= (self.learning_rate / len(dataset)) * grad_b

        return QuantumRegressor(ansatz=ansatz, hamiltonian=hamiltonian, weights=weights, bias=bias)
