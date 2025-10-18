"""Circuit optimisation helpers blending hardware-aware heuristics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .ansatz import HardwareAgnosticAnsatz


@dataclass(frozen=True)
class GateSpec:
    """Light-weight description of a quantum gate."""

    name: str
    qubits: Tuple[int, ...]
    parameter: float | None = None

    def with_parameter(self, parameter: float | None) -> "GateSpec":
        return GateSpec(self.name, self.qubits, parameter)

    @property
    def is_two_qubit(self) -> bool:
        return len(self.qubits) == 2

    @property
    def is_single_qubit(self) -> bool:
        return len(self.qubits) == 1


@dataclass(frozen=True)
class CircuitProfile:
    """Summary statistics describing a circuit."""

    depth: int
    total_gates: int
    single_qubit_gates: int
    entangling_gates: int


@dataclass(frozen=True)
class CircuitOptimisationResult:
    """Return value describing an optimisation pass."""

    gates: Tuple[GateSpec, ...]
    original_profile: CircuitProfile
    optimised_profile: CircuitProfile
    transformations: Tuple[str, ...]

    @property
    def depth_reduction(self) -> int:
        return self.original_profile.depth - self.optimised_profile.depth

    @property
    def gate_reduction(self) -> int:
        return self.original_profile.total_gates - self.optimised_profile.total_gates


class CircuitOptimiser:
    """Apply lightweight gate simplifications to variational circuits."""

    def __init__(self, *, amplitude_threshold: float = 1e-3) -> None:
        self.amplitude_threshold = float(amplitude_threshold)

    def describe_ansatz(
        self, ansatz: HardwareAgnosticAnsatz, parameters: Sequence[float]
    ) -> Tuple[GateSpec, ...]:
        parameters = np.asarray(parameters, dtype=float)
        if parameters.size != ansatz.parameter_count:
            raise ValueError(
                f"Expected {ansatz.parameter_count} parameters, received {parameters.size}"
            )
        index = 0
        gates: List[GateSpec] = []
        for _ in range(ansatz.layers):
            for qubit in range(ansatz.num_qubits):
                theta = float(parameters[index])
                phi = float(parameters[index + 1])
                index += 2
                gates.append(GateSpec("RY", (qubit,), theta))
                gates.append(GateSpec("RZ", (qubit,), phi))
            for qubit in range(ansatz.num_qubits):
                target = (qubit + 1) % ansatz.num_qubits
                if target == qubit:
                    continue
                gates.append(GateSpec("CZ", (qubit, target)))
        return tuple(gates)

    def profile(self, gates: Sequence[GateSpec], num_qubits: int) -> CircuitProfile:
        if num_qubits < 1:
            raise ValueError("num_qubits must be positive")
        total = len(gates)
        single = sum(1 for gate in gates if gate.is_single_qubit)
        entangling = sum(1 for gate in gates if gate.is_two_qubit)
        depth_per_qubit = [0] * num_qubits
        max_depth = 0
        for gate in gates:
            involved = gate.qubits
            level = max(depth_per_qubit[q] for q in involved) + 1
            for q in involved:
                depth_per_qubit[q] = level
            max_depth = max(max_depth, level)
        return CircuitProfile(
            depth=max_depth,
            total_gates=total,
            single_qubit_gates=single,
            entangling_gates=entangling,
        )

    def optimise(
        self, gates: Sequence[GateSpec], num_qubits: int
    ) -> CircuitOptimisationResult:
        original_profile = self.profile(gates, num_qubits)
        working = list(gates)
        log: List[str] = []

        merged: List[GateSpec] = []
        for gate in working:
            if (
                merged
                and gate.is_single_qubit
                and merged[-1].name == gate.name
                and merged[-1].qubits == gate.qubits
                and merged[-1].parameter is not None
                and gate.parameter is not None
            ):
                last = merged.pop()
                combined = float(last.parameter + gate.parameter)
                merged.append(last.with_parameter(combined))
                log.append(
                    f"Merged consecutive {gate.name} rotations on qubit {gate.qubits[0]}"
                )
            else:
                merged.append(gate)

        pruned: List[GateSpec] = []
        for gate in merged:
            if (
                gate.is_single_qubit
                and gate.parameter is not None
                and abs(gate.parameter) < self.amplitude_threshold
            ):
                log.append(
                    f"Removed near-identity {gate.name} on qubit {gate.qubits[0]}"
                )
                continue
            pruned.append(gate)

        cancelled: List[GateSpec] = []
        for gate in pruned:
            if (
                gate.name == "CZ"
                and cancelled
                and cancelled[-1].name == "CZ"
                and cancelled[-1].qubits == gate.qubits
            ):
                cancelled.pop()
                log.append(
                    f"Cancelled consecutive CZ pair on qubits {gate.qubits}"
                )
            else:
                cancelled.append(gate)

        optimised_profile = self.profile(cancelled, num_qubits)
        return CircuitOptimisationResult(
            gates=tuple(cancelled),
            original_profile=original_profile,
            optimised_profile=optimised_profile,
            transformations=tuple(log),
        )

    def optimise_ansatz(
        self, ansatz: HardwareAgnosticAnsatz, parameters: Sequence[float]
    ) -> CircuitOptimisationResult:
        gates = self.describe_ansatz(ansatz, parameters)
        return self.optimise(gates, ansatz.num_qubits)


__all__ = [
    "CircuitOptimisationResult",
    "CircuitOptimiser",
    "CircuitProfile",
    "GateSpec",
]
