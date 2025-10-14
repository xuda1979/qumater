import numpy as np
import pytest

from qumater.qsim import (
    GroverSearch,
    QuantumFourierTransform,
    QuantumPhaseEstimation,
    get_algorithm_registry,
)


def test_grover_search_module_marks_state():
    registry = get_algorithm_registry()
    module = registry.get("grover_search")
    grover = module.create(num_qubits=3, oracle=[5])

    result = grover.run()
    assert isinstance(grover, GroverSearch)
    assert result.most_likely_state() == 5
    assert result.probabilities[result.most_likely_state()] == max(result.probabilities)


def test_quantum_fourier_transform_behaves_like_fft():
    module = get_algorithm_registry().get("quantum_fourier_transform")
    qft = module.create(num_qubits=2)

    state = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    transformed = qft.run(state)

    expected = np.full(4, 0.5, dtype=complex)
    assert np.allclose(transformed, expected)
    assert np.allclose(qft.inverse(transformed), state)


def test_quantum_phase_estimation_returns_expected_phase():
    registry = get_algorithm_registry()
    module = registry.get("quantum_phase_estimation")

    phase = 0.25
    unitary = np.diag([1.0, np.exp(2j * np.pi * phase)])
    eigenstate = np.array([0.0, 1.0], dtype=complex)

    qpe = module.create(unitary=unitary, eigenstate=eigenstate, precision_qubits=3)
    result = qpe.run()

    assert isinstance(qpe, QuantumPhaseEstimation)
    assert result.binary == "010"
    assert abs(result.phase - phase) < 1 / (2**3)


def test_quantum_phase_estimation_rejects_non_unitary_operator():
    registry = get_algorithm_registry()
    module = registry.get("quantum_phase_estimation")

    non_unitary = np.array([[1.0, 0.0], [0.0, 0.5]], dtype=complex)
    eigenstate = np.array([1.0, 0.0], dtype=complex)

    with pytest.raises(ValueError):
        module.create(unitary=non_unitary, eigenstate=eigenstate, precision_qubits=2)


def test_grover_search_requires_marked_states():
    with pytest.raises(ValueError):
        GroverSearch(num_qubits=2, oracle=[])
