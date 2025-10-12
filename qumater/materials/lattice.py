"""Lattice Hamiltonian utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class LatticeModel:
    """Simple Hubbard-like lattice description.

    This data structure captures the subset of details that our hardware
    agnostic variational algorithms need: the graph layout and the associated
    hopping and interaction strengths.
    """

    adjacency: np.ndarray
    hopping: float
    onsite_interaction: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "adjacency": self.adjacency.tolist(),
            "hopping": float(self.hopping),
            "onsite_interaction": float(self.onsite_interaction),
        }


def hubbard_square_lattice(size: int, hopping: float, onsite: float) -> LatticeModel:
    """Return a square-lattice Hubbard model.

    Parameters
    ----------
    size:
        Number of lattice sites.  Must form a perfect square.
    hopping:
        Nearest neighbour hopping amplitude ``t``.
    onsite:
        On-site repulsion ``U``.
    """

    length = int(np.sqrt(size))
    if length * length != size:
        raise ValueError("Size must be a perfect square for square lattices")

    adjacency = np.zeros((size, size), dtype=int)
    for r in range(length):
        for c in range(length):
            idx = r * length + c
            if c + 1 < length:
                adjacency[idx, idx + 1] = adjacency[idx + 1, idx] = 1
            if r + 1 < length:
                adjacency[idx, idx + length] = adjacency[idx + length, idx] = 1

    return LatticeModel(adjacency=adjacency, hopping=hopping, onsite_interaction=onsite)
