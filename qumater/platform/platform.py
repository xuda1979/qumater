"""Facade tying individual components together."""
from __future__ import annotations

from typing import Iterable, Sequence

from ..qsim.ansatz import HardwareAgnosticAnsatz
from ..qsim.circuit_optimization import CircuitOptimiser, CircuitOptimisationResult
from ..qsim.hamiltonian import PauliHamiltonian
from ..qsim.hybrid import HybridQuantumAISimulator, HybridSimulationReport
from .control import AdaptiveQuantumController, IntelligentControlSuite
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
        self.control = IntelligentControlSuite()
        self.default_controller: AdaptiveQuantumController = self.control.ensure_default()
        self.circuit_optimiser = CircuitOptimiser()
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
        self.tasks.register(
            PlatformTask(
                name="optimise_circuit",
                description="Run structural optimisation over an ansatz circuit.",
                handler=self._optimise_circuit_task,
            ),
            overwrite=True,
        )
        self.tasks.register(
            PlatformTask(
                name="hybrid_simulation",
                description="Execute a hybrid quantum-AI simulation loop.",
                handler=self._hybrid_simulation_task,
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

    def _optimise_circuit_task(
        self, ansatz: HardwareAgnosticAnsatz, parameters: Sequence[float]
    ) -> CircuitOptimisationResult:
        return self.circuit_optimiser.optimise_ansatz(ansatz, parameters)

    def _hybrid_simulation_task(
        self,
        hamiltonian: PauliHamiltonian,
        ansatz: HardwareAgnosticAnsatz,
        *,
        controller_name: str = "default",
        initial_parameters: Sequence[float] | None = None,
        max_iterations: int | None = None,
    ) -> HybridSimulationReport:
        controller = self.control.get_controller(controller_name)
        simulator = HybridQuantumAISimulator(
            hamiltonian=hamiltonian,
            ansatz=ansatz,
            controller=controller,
            max_iterations=max_iterations or 40,
        )
        return simulator.simulate(initial_parameters=initial_parameters)

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

    def optimise_ansatz_circuit(
        self, ansatz: HardwareAgnosticAnsatz, parameters: Sequence[float]
    ) -> CircuitOptimisationResult:
        return self.circuit_optimiser.optimise_ansatz(ansatz, parameters)

    def run_hybrid_simulation(
        self,
        hamiltonian: PauliHamiltonian,
        ansatz: HardwareAgnosticAnsatz,
        *,
        controller_name: str = "default",
        initial_parameters: Sequence[float] | None = None,
        max_iterations: int | None = None,
    ) -> HybridSimulationReport:
        return self._hybrid_simulation_task(
            hamiltonian,
            ansatz,
            controller_name=controller_name,
            initial_parameters=initial_parameters,
            max_iterations=max_iterations,
        )
