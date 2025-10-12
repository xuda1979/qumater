"""Materials subpackage offering curated datasets and lattice utilities."""

from .datasets import QuantumMaterialDatabase, MaterialEntry
from .lattice import LatticeModel, hubbard_square_lattice

__all__ = [
    "QuantumMaterialDatabase",
    "MaterialEntry",
    "LatticeModel",
    "hubbard_square_lattice",
]
