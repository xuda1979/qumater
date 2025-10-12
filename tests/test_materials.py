import numpy as np

from qumater.materials import QuantumMaterialDatabase, hubbard_square_lattice


def test_demo_database_contains_phasecraft_targets():
    db = QuantumMaterialDatabase.demo()
    names = {entry.name for entry in db.as_dict().values()}
    assert "LiH minimal basis" in names
    assert any("superconductor" in entry.tags for entry in db.query("superconductor"))


def test_hubbard_square_lattice_adjacency():
    lattice = hubbard_square_lattice(4, hopping=1.0, onsite=4.0)
    expected = np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ]
    )
    assert np.array_equal(lattice.adjacency, expected)
    assert lattice.hopping == 1.0
    assert lattice.onsite_interaction == 4.0
