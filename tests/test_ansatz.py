import numpy as np

from qumater.qsim.ansatz import HardwareAgnosticAnsatz


def test_parameter_gradient_matches_finite_difference():
    ansatz = HardwareAgnosticAnsatz(num_qubits=2, layers=1)
    parameters = np.linspace(0.1, 1.0, ansatz.parameter_count)

    gradient = ansatz.parameter_gradient(parameters)

    eps = 1e-6
    base_state = ansatz.prepare_state(parameters)
    reference = np.zeros_like(gradient)
    for index in range(ansatz.parameter_count):
        shifted = parameters.copy()
        shifted[index] += eps
        reference[index] = (ansatz.prepare_state(shifted) - base_state) / eps

    assert np.allclose(gradient, reference, atol=1e-6)
