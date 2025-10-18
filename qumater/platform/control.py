"""Intelligent control loops driving hybrid quantum-AI workloads."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class ControlSignal:
    """Encapsulate an update signal for parameter optimisation."""

    learning_rate: float
    direction: np.ndarray


class AdaptiveQuantumController:
    """Adaptive controller combining momentum with energy-aware scheduling."""

    def __init__(
        self,
        *,
        base_learning_rate: float = 0.2,
        momentum: float = 0.9,
        acceleration: float = 0.1,
        damping: float = 0.5,
        min_learning_rate: float = 1e-3,
        max_learning_rate: float = 1.0,
    ) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        self.base_learning_rate = float(base_learning_rate)
        self.momentum = float(momentum)
        self.acceleration = float(acceleration)
        self.damping = float(damping)
        self.min_learning_rate = float(min_learning_rate)
        self.max_learning_rate = float(max_learning_rate)
        self._velocity: np.ndarray | None = None
        self._current_lr = float(base_learning_rate)
        self._last_energy: float | None = None

    def reset(self, dimension: int) -> None:
        self._velocity = np.zeros(dimension, dtype=float)
        self._current_lr = float(self.base_learning_rate)
        self._last_energy = None

    def propose_update(self, gradient: np.ndarray, energy: float) -> ControlSignal:
        gradient = np.asarray(gradient, dtype=float)
        if self._velocity is None or self._velocity.shape != gradient.shape:
            self.reset(gradient.size)
        if self._last_energy is not None:
            improvement = self._last_energy - float(energy)
            if improvement > 0:
                self._current_lr = min(
                    self._current_lr * (1.0 + self.acceleration), self.max_learning_rate
                )
            else:
                self._current_lr = max(self._current_lr * self.damping, self.min_learning_rate)
        self._velocity = self.momentum * self._velocity - (1.0 - self.momentum) * gradient
        self._last_energy = float(energy)
        return ControlSignal(learning_rate=self._current_lr, direction=self._velocity.copy())


class IntelligentControlSuite:
    """Registry managing reusable controllers across workflows."""

    def __init__(self) -> None:
        self._controllers: Dict[str, AdaptiveQuantumController] = {}

    def create_controller(self, name: str, **kwargs) -> AdaptiveQuantumController:
        controller = AdaptiveQuantumController(**kwargs)
        self._controllers[name] = controller
        return controller

    def get_controller(self, name: str) -> AdaptiveQuantumController:
        try:
            return self._controllers[name]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise KeyError(f"Controller '{name}' not registered") from exc

    def ensure_default(self) -> AdaptiveQuantumController:
        if "default" not in self._controllers:
            return self.create_controller("default")
        return self._controllers["default"]


__all__ = ["AdaptiveQuantumController", "ControlSignal", "IntelligentControlSuite"]
