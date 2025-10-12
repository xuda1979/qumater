import numpy as np

from qumater.qsim import PauliHamiltonian, PauliTerm, group_commuting_terms


def test_commuting_groups():
    terms = [
        PauliTerm(1.0, "ZI"),
        PauliTerm(0.5, "IZ"),
        PauliTerm(0.25, "ZZ"),
        PauliTerm(0.1, "XX"),
    ]
    groups = group_commuting_terms(terms)
    assert any(len(group) >= 3 for group in groups)


def test_expectation_matches_matrix():
    state = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    hamiltonian = PauliHamiltonian([PauliTerm(1.0, "ZI"), PauliTerm(-0.5, "IZ")])
    grouped = hamiltonian.expectation(state)
    matrix = hamiltonian.matrix()
    direct = np.real(np.vdot(state, matrix @ state))
    assert np.isclose(grouped, direct)
