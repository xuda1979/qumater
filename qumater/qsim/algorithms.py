"""Reference implementations of established quantum algorithms.

The module started out focusing on variational algorithms inspired by
Phasecraft's hardware agnostic research.  To make the package more useful as a
general algorithm hub we now expose a curated collection of well-known
algorithms alongside the existing VQE optimiser.  Each implementation favours
clarity over micro-optimisations so that new contributions can follow the same
patterns and easily register themselves as modules via
``qumater.qsim.modules``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np

from .ansatz import HardwareAgnosticAnsatz
from .hamiltonian import PauliHamiltonian
from .modules import AlgorithmModule, register_algorithm_module


@dataclass
class OptimizationHistory:
    """Container describing the optimisation trace."""

    parameters: List[np.ndarray]
    energies: List[float]
    converged: bool


@dataclass
class GroverResult:
    """Result of a Grover search iteration."""

    probabilities: np.ndarray

    def most_likely_state(self) -> int:
        """Return the computational basis state with the highest probability."""

        return int(np.argmax(self.probabilities))


@dataclass
class PhaseEstimationResult:
    """Outcome of a quantum phase estimation run."""

    phase: float
    binary: str

    @property
    def decimal(self) -> float:
        r"""Return the phase estimate expressed as a fraction of :math:`2\pi`."""

        return self.phase


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
        # 计算 |ψ(θ)⟩ 在哈密顿量作用下的状态，用于后续的梯度评估
        ham_state = self.hamiltonian.apply(state)
        # ansatz.parameter_gradient 返回对每个可调参数的导数向量
        gradients = self.ansatz.parameter_gradient(parameters)
        grad = np.zeros(parameters.shape, dtype=float)
        for idx in range(gradients.shape[0]):
            derivative = gradients[idx]
            # 近似量子自然梯度基于能量对参数的导数
            grad[idx] = 2.0 * np.real(np.vdot(derivative, ham_state))
        return grad

    def _metric_tensor(self, parameters: np.ndarray, state: np.ndarray) -> np.ndarray:
        # Fubini–Study 度量张量由梯度的 Gram 矩阵减去投影项组成
        gradients = self.ansatz.parameter_gradient(parameters)
        overlaps = gradients @ np.conjugate(state)
        gram = gradients @ np.conjugate(gradients.T)
        metric = np.real(gram - np.outer(overlaps, np.conjugate(overlaps)))
        # 添加轻量正则项避免张量奇异
        metric += self.regularisation * np.eye(metric.shape[0])
        return metric

    def run(self, initial_parameters: Optional[Sequence[float]] = None) -> OptimizationHistory:
        if initial_parameters is None:
            # 为避免完全零向量导致解析梯度停滞，使用确定性扰动打破对称性
            parameters = np.arange(1, self.ansatz.parameter_count + 1, dtype=float) * 1e-2
            parameters[1::2] *= -1.0
        else:
            parameters = np.array(initial_parameters, dtype=float, copy=True)
        history_params: List[np.ndarray] = []
        history_energies: List[float] = []
        converged = False
        last_direction_norm = None

        for _ in range(self.max_iterations):
            # 准备当前参数下的量子态并评估期望能量
            state = self.ansatz.prepare_state(parameters)
            energy = self.hamiltonian.expectation(state)
            history_params.append(parameters.copy())
            history_energies.append(float(energy))

            # 量子自然梯度方向需要梯度向量与度量张量
            gradient = self._energy_gradient(parameters, state)
            metric = self._metric_tensor(parameters, state)
            try:
                direction = np.linalg.solve(metric, gradient)
            except np.linalg.LinAlgError:
                direction = gradient

            update_norm = np.linalg.norm(direction)
            last_direction_norm = update_norm
            # 使用固定学习率执行梯度下降步
            parameters = parameters - self.learning_rate * direction

            if update_norm * self.learning_rate < self.tolerance:
                converged = True
                break

        if not converged:
            # 若循环未提前收敛，则补充记录最终的参数与能量
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


def _low_depth_vqe_factory(
    hamiltonian: PauliHamiltonian,
    ansatz: HardwareAgnosticAnsatz,
    **kwargs: float,
) -> LowDepthVQE:
    return LowDepthVQE(hamiltonian=hamiltonian, ansatz=ansatz, **kwargs)


register_algorithm_module(
    AlgorithmModule(
        name="low_depth_vqe",
        summary="Variational eigensolver with approximate quantum natural gradient updates.",
        factory=_low_depth_vqe_factory,
        keywords=("vqe", "natural-gradient", "phasecraft"),
        package=__name__,
    ),
    overwrite=True,
)


class GroverSearch:
    """Classical simulator for Grover's search algorithm.

    Parameters
    ----------
    num_qubits:
        Size of the search space in qubits.
    oracle:
        Either a callable returning ``True`` for marked states or an iterable of
        marked indices.
    iterations:
        Number of Grover iterations.  When omitted the canonical optimal count
        is used based on the number of marked states.
    """

    def __init__(
        self,
        *,
        num_qubits: int,
        oracle: Callable[[int], bool] | Iterable[int],
        iterations: Optional[int] = None,
    ) -> None:
        if num_qubits < 1:
            raise ValueError("num_qubits must be positive")

        self.num_qubits = int(num_qubits)
        self.dimension = 2**self.num_qubits
        self._oracle_phase = np.ones(self.dimension, dtype=float)

        if callable(oracle):
            marked_indices = [index for index in range(self.dimension) if oracle(index)]
        else:
            marked_indices = list(dict.fromkeys(int(idx) for idx in oracle))

        if not marked_indices:
            raise ValueError("GroverSearch requires at least one marked state")

        for index in marked_indices:
            if index < 0 or index >= self.dimension:
                raise ValueError(
                    f"Marked state index {index} is outside the {self.dimension}-dimensional space"
                )
            self._oracle_phase[index] = -1.0

        solution_ratio = len(marked_indices) / self.dimension
        if iterations is None:
            theta = np.arcsin(np.sqrt(solution_ratio))
            optimal = int(max(1, round((np.pi / (4 * theta)) - 0.5)))
            self.iterations = optimal
        else:
            self.iterations = int(iterations)
            if self.iterations < 1:
                raise ValueError("iterations must be positive")

    def run(self) -> GroverResult:
        """Execute Grover iterations and return the resulting probabilities."""

        state = np.ones(self.dimension, dtype=complex) / np.sqrt(self.dimension)
        diffusion = (2.0 / self.dimension) * np.ones((self.dimension, self.dimension)) - np.eye(
            self.dimension, dtype=complex
        )

        oracle_matrix = np.diag(self._oracle_phase.astype(complex))
        for _ in range(self.iterations):
            # 交替施加 Oracle 相位翻转与扩散算符，实现振幅放大
            state = oracle_matrix @ state
            state = diffusion @ state

        probabilities = np.abs(state) ** 2
        return GroverResult(probabilities=probabilities)


def _grover_search_factory(
    *, num_qubits: int, oracle: Callable[[int], bool] | Iterable[int], iterations: Optional[int] = None
) -> GroverSearch:
    return GroverSearch(num_qubits=num_qubits, oracle=oracle, iterations=iterations)


register_algorithm_module(
    AlgorithmModule(
        name="grover_search",
        summary="Grover amplitude amplification over a black-box oracle.",
        factory=_grover_search_factory,
        keywords=("search", "amplitude-amplification", "grover"),
        package=__name__,
    ),
    overwrite=True,
)


class QuantumFourierTransform:
    """Discrete quantum Fourier transform implemented via FFT."""

    def __init__(self, *, num_qubits: int) -> None:
        if num_qubits < 1:
            raise ValueError("num_qubits must be positive")
        self.num_qubits = int(num_qubits)
        self.dimension = 2**self.num_qubits

    def run(self, state: Sequence[complex]) -> np.ndarray:
        """Apply the QFT to ``state`` and return the transformed amplitudes."""

        amplitudes = np.asarray(state, dtype=complex)
        if amplitudes.size != self.dimension:
            raise ValueError(
                f"Expected a state with {self.dimension} amplitudes, received {amplitudes.size}"
            )
        # 经典 FFT 提供与量子傅里叶变换等价的变换矩阵
        transformed = np.fft.fft(amplitudes) / np.sqrt(self.dimension)
        return transformed

    def inverse(self, state: Sequence[complex]) -> np.ndarray:
        """Apply the inverse QFT to ``state``."""

        amplitudes = np.asarray(state, dtype=complex)
        if amplitudes.size != self.dimension:
            raise ValueError(
                f"Expected a state with {self.dimension} amplitudes, received {amplitudes.size}"
            )
        # 逆 QFT 对应归一化的 IFFT
        transformed = np.fft.ifft(amplitudes) * np.sqrt(self.dimension)
        return transformed


def _qft_factory(*, num_qubits: int) -> QuantumFourierTransform:
    return QuantumFourierTransform(num_qubits=num_qubits)


register_algorithm_module(
    AlgorithmModule(
        name="quantum_fourier_transform",
        summary="Quantum Fourier transform over computational basis states.",
        factory=_qft_factory,
        keywords=("qft", "fourier"),
        package=__name__,
    ),
    overwrite=True,
)


class QuantumPhaseEstimation:
    """Estimate the eigenphase of a unitary given one of its eigenstates.

    Parameters
    ----------
    unitary:
        Square matrix representing the unitary whose eigenphase is estimated.
    eigenstate:
        An eigenvector of ``unitary`` corresponding to the eigenphase of
        interest.
    precision_qubits:
        Number of readout qubits used to discretise the phase estimate.
    unitarity_tol:
        Absolute tolerance used when verifying that ``unitary`` is indeed
        unitary.  Providing a dedicated tolerance mirrors the behaviour of
        production simulators that validate user provided operators before
        executing costly workflows.
    """

    def __init__(
        self,
        *,
        unitary: np.ndarray,
        eigenstate: Sequence[complex],
        precision_qubits: int,
        unitarity_tol: float = 1e-8,
    ) -> None:
        if precision_qubits < 1:
            raise ValueError("precision_qubits must be positive")

        unitary = np.asarray(unitary, dtype=complex)
        if unitary.ndim != 2 or unitary.shape[0] != unitary.shape[1]:
            raise ValueError("unitary must be a square matrix")

        eigenstate = np.asarray(eigenstate, dtype=complex)
        if eigenstate.ndim != 1:
            raise ValueError("eigenstate must be a vector")
        if eigenstate.size != unitary.shape[0]:
            raise ValueError("eigenstate dimension must match unitary")

        norm = np.linalg.norm(eigenstate)
        if norm == 0:
            raise ValueError("eigenstate must be non-zero")

        if not np.allclose(
            unitary.conjugate().T @ unitary,
            np.eye(unitary.shape[0], dtype=complex),
            atol=unitarity_tol,
            rtol=0.0,
        ):
            raise ValueError("unitary must be unitary within the provided tolerance")

        self.unitary = unitary
        self.eigenstate = eigenstate / norm
        self.precision_qubits = int(precision_qubits)
        self.dimension = unitary.shape[0]
        self.unitarity_tol = float(unitarity_tol)

    def run(self) -> PhaseEstimationResult:
        """Return a phase estimate rounded to ``precision_qubits`` bits."""

        evolved = self.unitary @ self.eigenstate
        overlap = np.vdot(self.eigenstate, evolved)
        if np.isclose(np.linalg.norm(evolved), 0.0):
            raise RuntimeError("Unitary produced a zero vector")

        # 获取酉作用前后态矢量的相位差，并根据精度量化到离散格点
        phase_fraction = (np.angle(overlap) / (2 * np.pi)) % 1.0
        scaling = 2**self.precision_qubits
        discrete = int(np.floor(phase_fraction * scaling + 0.5)) % scaling
        estimate = discrete / scaling
        # 同时返回便于展示的二进制字符串表示
        binary = format(discrete, f"0{self.precision_qubits}b")
        return PhaseEstimationResult(phase=estimate, binary=binary)


def _qpe_factory(
    *, unitary: np.ndarray, eigenstate: Sequence[complex], precision_qubits: int, unitarity_tol: float = 1e-8
) -> QuantumPhaseEstimation:
    return QuantumPhaseEstimation(
        unitary=unitary,
        eigenstate=eigenstate,
        precision_qubits=precision_qubits,
        unitarity_tol=unitarity_tol,
    )


register_algorithm_module(
    AlgorithmModule(
        name="quantum_phase_estimation",
        summary="Canonical Kitaev-style quantum phase estimation.",
        factory=_qpe_factory,
        keywords=("qpe", "phase", "estimation"),
        package=__name__,
    ),
    overwrite=True,
)


__all__ = [
    "GroverResult",
    "GroverSearch",
    "LowDepthVQE",
    "OptimizationHistory",
    "PhaseEstimationResult",
    "QuantumFourierTransform",
    "QuantumPhaseEstimation",
    "_grover_search_factory",
    "_low_depth_vqe_factory",
    "_qft_factory",
    "_qpe_factory",
]
