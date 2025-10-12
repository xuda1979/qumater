import numpy as np

from qumater.qsim import HardwareAgnosticAnsatz, LowDepthVQE, PauliHamiltonian, PauliTerm


def test_low_depth_vqe_finds_ground_state_of_single_qubit_z():
    ansatz = HardwareAgnosticAnsatz(num_qubits=1, layers=1)
    hamiltonian = PauliHamiltonian([PauliTerm(1.0, "Z")])
    optimiser = LowDepthVQE(hamiltonian, ansatz, learning_rate=0.1, max_iterations=60)
    history = optimiser.run()
    assert history.energies[0] > history.energies[-1]
    assert history.energies[-1] < -0.8
    assert history.converged
