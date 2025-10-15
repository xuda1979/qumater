"""Application level workflows built on top of core modules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..qsim.algorithms import LowDepthVQE, OptimizationHistory
from ..qsim.ansatz import HardwareAgnosticAnsatz
from ..qsim.hamiltonian import PauliHamiltonian


@dataclass
class ApplicationReport:
    """Reusable structure summarising application runs."""

    name: str
    result: float
    parameters: np.ndarray
    history: OptimizationHistory


class QuantumApplicationSuite:
    """Domain inspired entry points for finance, chemistry and control."""

    def __init__(self, *, learning_rate: float = 0.2, max_iterations: int = 150) -> None:
        self.learning_rate = float(learning_rate)
        self.max_iterations = int(max_iterations)

    def run_vqe_pipeline(
        self,
        name: str,
        hamiltonian: PauliHamiltonian,
        ansatz: HardwareAgnosticAnsatz,
        initial_parameters: Optional[Sequence[float]] = None,
    ) -> ApplicationReport:
        optimiser = LowDepthVQE(
            hamiltonian=hamiltonian,
            ansatz=ansatz,
            learning_rate=self.learning_rate,
            max_iterations=self.max_iterations,
        )
        history = optimiser.run(initial_parameters)
        result = history.energies[-1]
        parameters = history.parameters[-1]
        return ApplicationReport(name=name, result=result, parameters=parameters, history=history)

    def optimise_portfolio(
        self,
        expected_returns: Sequence[float],
        covariance: np.ndarray,
        risk_aversion: float,
    ) -> dict[str, float]:
        expected_returns = np.asarray(expected_returns, dtype=float)
        covariance = np.asarray(covariance, dtype=float)
        if covariance.shape[0] != covariance.shape[1]:
            raise ValueError("Covariance matrix must be square")
        if covariance.shape[0] != expected_returns.size:
            raise ValueError("Dimension mismatch between covariance and returns")
        inv_cov = np.linalg.pinv(covariance)
        ones = np.ones(expected_returns.shape)
        numerator = inv_cov @ expected_returns
        denominator = ones @ inv_cov @ expected_returns
        weights = (1 - risk_aversion) * (numerator / denominator) + risk_aversion * (inv_cov @ ones) / (ones @ inv_cov @ ones)
        return {
            "expected_return": float(weights @ expected_returns),
            "risk": float(weights @ covariance @ weights),
        }
