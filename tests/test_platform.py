import numpy as np
import pytest

from qumater.platform import (
    MeasurementControlSystem,
    QuantumApplicationSuite,
    QuantumEDA,
    QuantumMachineLearningPlatform,
    QuantumSoftwarePlatform,
)
from qumater.qsim.ansatz import HardwareAgnosticAnsatz
from qumater.qsim.hamiltonian import PauliHamiltonian, PauliTerm


def build_simple_hamiltonian():
    return PauliHamiltonian([PauliTerm(1.0, "Z")])


def test_quantum_eda_statistics_and_sweep():
    eda = QuantumEDA()
    ansatz = eda.synthesise_ansatz(num_qubits=1, layers=2)
    stats = eda.estimate_statistics(ansatz)
    assert stats.depth == 4
    assert stats.single_qubit_gates == 4
    assert stats.entangling_gates == 2

    hamiltonian = build_simple_hamiltonian()
    grid = [np.zeros(ansatz.parameter_count), np.ones(ansatz.parameter_count) * 0.1]
    params, energy = eda.parameter_sweep(hamiltonian, ansatz, grid)
    assert params.shape == (ansatz.parameter_count,)
    assert isinstance(energy, float)


def test_programming_compile_and_validate():
    platform = QuantumSoftwarePlatform()
    spec = [(1.0, "Z")]
    result = platform.programming.compile_to_hamiltonian(spec)
    assert result.metadata["num_terms"] == 1
    assert result.metadata["num_qubits"] == 1
    ansatz = HardwareAgnosticAnsatz(num_qubits=1, layers=1)
    validated = platform.programming.validate_parameters(ansatz, [0.1, 0.2])
    assert validated.shape == (2,)


def test_quantum_machine_learning_regressor():
    ansatz = HardwareAgnosticAnsatz(num_qubits=1, layers=1)
    hamiltonian = build_simple_hamiltonian()
    true_weights = np.array([[0.4], [0.0]])
    true_bias = np.array([0.2, 0.0])
    features = [np.array([0.0]), np.array([1.0]), np.array([-1.0])]
    dataset = []
    for feature in features:
        params = true_bias + true_weights @ feature
        state = ansatz.prepare_state(params)
        dataset.append((feature, hamiltonian.expectation(state)))

    ml = QuantumMachineLearningPlatform(learning_rate=0.2, epochs=500)
    regressor = ml.fit_expectation_regressor(dataset, ansatz, hamiltonian)
    for feature, target in dataset:
        prediction = regressor.predict(feature)
        assert pytest.approx(target, abs=1e-2) == prediction


def test_measurement_control_schedule():
    hamiltonian = PauliHamiltonian([PauliTerm(1.0, "X"), PauliTerm(1.0, "Z")])
    control = MeasurementControlSystem()
    schedule = control.create_schedule(hamiltonian)
    assert schedule.total_groups == 2
    state = np.zeros(2, dtype=complex)
    state[0] = 1.0
    expectations = control.simulate_expectations(hamiltonian, state)
    assert expectations == {"X": 0.0, "Z": 1.0}


def test_application_suite_vqe_and_portfolio():
    ansatz = HardwareAgnosticAnsatz(num_qubits=1, layers=1)
    hamiltonian = build_simple_hamiltonian()
    applications = QuantumApplicationSuite(max_iterations=20)
    report = applications.run_vqe_pipeline("ground_state", hamiltonian, ansatz)
    assert report.name == "ground_state"
    assert report.parameters.shape == (ansatz.parameter_count,)

    covariance = np.eye(3)
    returns = np.array([0.1, 0.2, 0.15])
    metrics = applications.optimise_portfolio(returns, covariance, risk_aversion=0.5)
    assert set(metrics.keys()) == {"expected_return", "risk"}


def test_platform_tasks_and_execution():
    platform = QuantumSoftwarePlatform()
    ansatz = platform.execute_task("synthesise_ansatz", num_qubits=1, layers=1)
    hamiltonian = build_simple_hamiltonian()
    compilation = platform.execute_task("compile_program", [(1.0, "Z")])
    assert compilation.metadata["num_terms"] == 1
    report = platform.execute_task("run_vqe", "demo", hamiltonian, ansatz)
    assert report.name == "demo"
    schedule = platform.schedule_measurements(hamiltonian)
    assert schedule.total_groups == 1
    regressor = platform.fit_quantum_regressor([(np.array([0.0]), 1.0)], ansatz, hamiltonian)
    assert isinstance(regressor.predict([0.0]), float)


def test_platform_circuit_optimisation_and_hybrid_simulation():
    platform = QuantumSoftwarePlatform()
    ansatz = HardwareAgnosticAnsatz(num_qubits=2, layers=1)
    parameters = np.zeros(ansatz.parameter_count)
    parameters[0] = 5e-4
    optimisation = platform.optimise_ansatz_circuit(ansatz, parameters)
    assert optimisation.original_profile.total_gates >= optimisation.optimised_profile.total_gates
    assert optimisation.original_profile.depth >= optimisation.optimised_profile.depth

    hamiltonian = PauliHamiltonian([PauliTerm(1.0, "ZZ")])
    report = platform.run_hybrid_simulation(hamiltonian, ansatz, max_iterations=6)
    assert len(report.steps) == 6
    assert all(step.learning_rate > 0 for step in report.steps)
    assert report.final_parameters.shape == (ansatz.parameter_count,)
