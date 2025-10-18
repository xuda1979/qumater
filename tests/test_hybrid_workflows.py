import numpy as np

from qumater.platform.control import AdaptiveQuantumController
from qumater.qsim.ansatz import HardwareAgnosticAnsatz
from qumater.qsim.circuit_optimization import CircuitOptimiser, GateSpec
from qumater.qsim.hamiltonian import PauliHamiltonian, PauliTerm
from qumater.qsim.hybrid import HybridQuantumAISimulator


def test_circuit_optimiser_merges_and_prunes():
    optimiser = CircuitOptimiser(amplitude_threshold=1e-3)
    gates = (
        GateSpec("RY", (0,), 0.2),
        GateSpec("RY", (0,), -0.2),
        GateSpec("RZ", (0,), 5e-4),
        GateSpec("CZ", (0, 1)),
        GateSpec("CZ", (0, 1)),
    )
    result = optimiser.optimise(gates, num_qubits=2)
    assert result.optimised_profile.total_gates == 0
    assert any("Merged" in entry for entry in result.transformations)
    assert any("Removed" in entry for entry in result.transformations)
    assert any("Cancelled" in entry for entry in result.transformations)


def test_adaptive_controller_adjusts_learning_rate():
    controller = AdaptiveQuantumController(
        base_learning_rate=0.1, momentum=0.5, acceleration=0.5, damping=0.5
    )
    controller.reset(2)
    signal1 = controller.propose_update(np.array([1.0, -1.0]), energy=1.0)
    signal2 = controller.propose_update(np.array([0.5, -0.5]), energy=0.2)
    assert signal2.learning_rate >= signal1.learning_rate
    signal3 = controller.propose_update(np.array([0.5, -0.5]), energy=0.4)
    assert signal3.learning_rate <= signal2.learning_rate


def test_hybrid_simulator_generates_guided_steps():
    ansatz = HardwareAgnosticAnsatz(num_qubits=1, layers=1)
    hamiltonian = PauliHamiltonian([PauliTerm(1.0, "Z")])
    controller = AdaptiveQuantumController(base_learning_rate=0.2)
    simulator = HybridQuantumAISimulator(
        hamiltonian=hamiltonian, ansatz=ansatz, controller=controller, max_iterations=4
    )
    report = simulator.simulate()
    assert len(report.steps) == 4
    assert all(step.guidance.shape == (ansatz.parameter_count,) for step in report.steps)
    assert np.isfinite(report.final_energy)
