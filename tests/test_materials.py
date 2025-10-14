import numpy as np
import pytest

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


def test_database_filter_allows_semantic_and_numeric_constraints():
    db = QuantumMaterialDatabase.demo()
    superconductors = db.filter(tags=["superconductor"], parameter_bounds={"Tc": (60.0, None)})
    assert [entry.name for entry in superconductors] == ["FeSe monolayer"]

    # Open ended bounds should act as inclusive filters and exclude missing parameters.
    strongly_correlated = db.filter(parameter_bounds={"u": (4.0, 4.0)})
    assert [entry.name for entry in strongly_correlated] == ["Fermi-Hubbard 2x2"]


def test_summary_returns_serialisable_view():
    db = QuantumMaterialDatabase.demo()
    summary = db.summary()
    assert [item["name"] for item in summary] == [
        "Fermi-Hubbard 2x2",
        "FeSe monolayer",
        "LiH minimal basis",
    ]
    fe_se = summary[1]
    fe_se["parameters"]["Tc"] = 70.0
    # The original entry should remain unchanged because the summary returns copies.
    assert db.get("FeSe monolayer").parameters["Tc"] == 65.0


def test_register_rejects_duplicate_names():
    db = QuantumMaterialDatabase.demo()
    first = db.get("LiH minimal basis")

    with pytest.raises(ValueError):
        db.register(first)


def test_filter_rejects_invalid_parameter_bounds():
    db = QuantumMaterialDatabase.demo()

    with pytest.raises(ValueError):
        db.filter(parameter_bounds={"Tc": (70.0, 60.0)})
