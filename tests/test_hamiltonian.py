import numpy as np
import pytest

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


def test_density_expectation_matches_state_projection():
    state = np.array([1 / np.sqrt(2), 0.0, 0.0, 1 / np.sqrt(2)], dtype=complex)
    density = np.outer(state, np.conjugate(state))
    hamiltonian = PauliHamiltonian([PauliTerm(1.0, "ZI"), PauliTerm(2.0, "IZ")])
    energy_state = hamiltonian.expectation(state)
    energy_density = hamiltonian.expectation_density(density)
    assert energy_state == pytest.approx(energy_density)


def test_commuting_groups_cached():
    terms = [PauliTerm(1.0, "XX"), PauliTerm(0.5, "YY"), PauliTerm(-0.75, "ZZ")]
    hamiltonian = PauliHamiltonian(terms)
    cached_groups = hamiltonian.commuting_groups
    # 由于各项彼此对易，因此应只缓存一个分组。
    assert len(cached_groups) == 1
    assert set(cached_groups[0]) == set(terms)


def test_variance_identifies_eigenstates_and_superpositions():
    hamiltonian = PauliHamiltonian([PauliTerm(1.0, "Z")])

    ground_state = np.array([1.0, 0.0], dtype=complex)
    assert hamiltonian.variance(ground_state) == pytest.approx(0.0, abs=1e-12)

    plus_state = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
    assert hamiltonian.variance(plus_state) == pytest.approx(1.0)
