"""Facade tying individual components together."""
from __future__ import annotations

from typing import Iterable, Sequence

from ..qsim.ansatz import HardwareAgnosticAnsatz
from ..qsim.hamiltonian import PauliHamiltonian
from .applications import QuantumApplicationSuite
from .eda import QuantumEDA
from .machine_learning import QuantumMachineLearningPlatform
from .measurement import MeasurementControlSystem
from .programming import ProgramDevelopmentPlatform
from .tasks import PlatformTask, TaskRegistry


class QuantumSoftwarePlatform:
    """One-stop quantum software bundle mirroring industrial tooling."""

    def __init__(self) -> None:
        self.eda = QuantumEDA()
        self.programming = ProgramDevelopmentPlatform()
        self.ml = QuantumMachineLearningPlatform()
        self.measurement = MeasurementControlSystem()
        self.applications = QuantumApplicationSuite()
        self.tasks = TaskRegistry()
        self._register_default_tasks()

    def _register_default_tasks(self) -> None:
        self.tasks.register(
            PlatformTask(
                name="synthesise_ansatz",
                description="Create a hardware agnostic ansatz for a given topology.",
                handler=self.eda.synthesise_ansatz,
            ),
            overwrite=True,
        )
        self.tasks.register(
            PlatformTask(
                name="compile_program",
                description="Compile weighted Pauli strings into a Hamiltonian.",
                handler=self.programming.compile_to_hamiltonian,
            ),
            overwrite=True,
        )
        self.tasks.register(
            PlatformTask(
                name="run_vqe",
                description="Execute a VQE optimisation using the integrated stack.",
                handler=self._run_vqe_task,
            ),
            overwrite=True,
        )

    def _run_vqe_task(
        self,
        name: str,
        hamiltonian: PauliHamiltonian,
        ansatz: HardwareAgnosticAnsatz,
        initial_parameters: Sequence[float] | None = None,
    ):
        return self.applications.run_vqe_pipeline(
            name=name,
            hamiltonian=hamiltonian,
            ansatz=ansatz,
            initial_parameters=initial_parameters,
        )

    def execute_task(self, name: str, *args, **kwargs):
        return self.tasks.run(name, *args, **kwargs)

    def schedule_measurements(self, hamiltonian: PauliHamiltonian):
        return self.measurement.create_schedule(hamiltonian)

    def fit_quantum_regressor(
        self,
        dataset: Iterable[tuple[Sequence[float], float]],
        ansatz: HardwareAgnosticAnsatz,
        hamiltonian: PauliHamiltonian,
    ):
        return self.ml.fit_expectation_regressor(dataset, ansatz, hamiltonian)
