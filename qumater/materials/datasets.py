"""Curated quantum materials metadata inspired by industrial data feeds.

The Tencent article referenced by the user describes a hardware-agnostic
quantum materials platform that integrates industrial partners like Google
Quantum AI and IBM Quantum.  The :class:`QuantumMaterialDatabase` class below is
inspired by that description – it bundles small but expressive reference data
that we can use to prototype algorithms inside this repository without
requiring an external service.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple


@dataclass(frozen=True)
class MaterialEntry:
    """Metadata describing a simulated quantum material.

    Parameters
    ----------
    name:
        Human readable name of the material or model.
    tags:
        Semantic tags that can be used to query the material.  We include
        descriptors such as ``"superconductor"`` or ``"hubbard"`` to
        mirror the categorisation described in Phasecraft's public facing
        marketing materials.
    citation:
        Optional reference string.  We use the Tencent news article when we do
        not have a more specific academic reference.
    parameters:
        Free-form dictionary holding physical parameters that downstream
        algorithms can consume.
    """

    name: str
    tags: List[str]
    citation: Optional[str]
    parameters: Dict[str, float]


class QuantumMaterialDatabase:
    """In-memory catalogue of quantum materials and model Hamiltonians.

    The real-world database described in the Tencent news report integrates
    industrial data feeds and bespoke simulation-ready Hamiltonians.  We keep
    the interface simple while emulating useful behaviours:

    * tags based querying to narrow down families of materials;
    * retrieval by chemical formula or colloquial name;
    * support for registering additional entries at runtime so that users can
      plug in domain-specific data.
    """

    def __init__(self, entries: Optional[Iterable[MaterialEntry]] = None) -> None:
        self._entries: Dict[str, MaterialEntry] = {}
        if entries:
            for entry in entries:
                self.register(entry)

    def register(self, entry: MaterialEntry) -> None:
        """Register *entry* in the catalogue.

        Raises
        ------
        ValueError
            If another entry with the same (case-insensitive) name already
            exists.  This mirrors behaviour of production asset catalogues where
            names act as stable identifiers.
        """

        key = entry.name.lower()
        if key in self._entries:
            raise ValueError(f"Material '{entry.name}' already registered")
        self._entries[key] = entry

    def get(self, name: str) -> MaterialEntry:
        """Retrieve a material entry by *name*.

        Parameters
        ----------
        name:
            The name of the material.  Lookups are case insensitive.

        Raises
        ------
        KeyError
            If the material is missing from the database.
        """

        key = name.lower()
        try:
            return self._entries[key]
        except KeyError as exc:
            raise KeyError(f"Material '{name}' not found") from exc

    def query(self, *tags: str) -> List[MaterialEntry]:
        """Return all entries that contain **all** supplied *tags*.

        The Tencent piece emphasises Phasecraft's ability to deliver
        application-targeted datasets.  Tag based search reflects this
        requirement and allows our algorithms to automatically identify the most
        relevant benchmark suites.
        """

        required = {tag.lower() for tag in tags}
        result = []
        for entry in self._entries.values():
            if required.issubset({tag.lower() for tag in entry.tags}):
                result.append(entry)
        return result

    def filter(
        self,
        *,
        tags: Optional[Iterable[str]] = None,
        parameter_bounds: Optional[Mapping[str, Tuple[Optional[float], Optional[float]]]] = None,
    ) -> List[MaterialEntry]:
        """Return entries matching semantic tags and numeric parameter ranges.

        Parameters
        ----------
        tags:
            Optional iterable of tags that must all be present in the entry.  Tag
            comparisons are case-insensitive.
        parameter_bounds:
            Optional mapping from parameter names to inclusive ``(lower, upper)``
            bounds.  ``None`` may be used to leave a bound open ended.  Entries
            missing a requested parameter are excluded from the results.

        Notes
        -----
        Phasecraft's industrial partners highlighted in the Tencent report rely
        on curated subsets of large data feeds.  The :meth:`filter` helper makes
        it easy to carve out the most relevant subset directly inside research
        scripts without maintaining an external query engine.
        """

        if tags is not None:
            required_tags = {tag.lower() for tag in tags}
        else:
            required_tags = None

        bounds = dict(parameter_bounds or {})

        def matches(entry: MaterialEntry) -> bool:
            if required_tags is not None:
                entry_tags = {tag.lower() for tag in entry.tags}
                if not required_tags.issubset(entry_tags):
                    return False

            for name, (lower, upper) in bounds.items():
                if name not in entry.parameters:
                    return False
                value = entry.parameters[name]
                if lower is not None and value < lower:
                    return False
                if upper is not None and value > upper:
                    return False
            return True

        return [entry for entry in self._entries.values() if matches(entry)]

    @classmethod
    def demo(cls) -> "QuantumMaterialDatabase":
        """Return a small demonstration catalogue.

        The data points loosely reflect the focus areas highlighted in the
        Tencent report: high-temperature superconductors, near-term quantum
        simulation targets, and reference Hamiltonians crafted to be
        hardware-agnostic.
        """

        return cls(
            entries=[
                MaterialEntry(
                    name="LiH minimal basis",
                    tags=["molecular", "variational", "benchmarks"],
                    citation=(
                        "Tencent News (2025). Phasecraft announces hardware-agnostic "
                        "quantum materials platform."
                    ),
                    parameters={"electrons": 2, "orbitals": 4},
                ),
                MaterialEntry(
                    name="Fermi-Hubbard 2x2",
                    tags=["hubbard", "strongly-correlated", "benchmark"],
                    citation="Phasecraft demo circuits (private communication)",
                    parameters={"lattice_size": 4, "u": 4.0, "t": 1.0},
                ),
                MaterialEntry(
                    name="FeSe monolayer",
                    tags=["superconductor", "high-Tc"],
                    citation="Tencent News (2025)",
                    parameters={"Tc": 65.0, "doping": 0.12},
                ),
            ]
        )

    def as_dict(self) -> Dict[str, MaterialEntry]:
        """Return a snapshot of the catalogue content as a plain dictionary."""

        return dict(self._entries)

    def summary(self) -> List[Dict[str, object]]:
        """Return an ordered summary of the catalogue contents.

        The helper is convenient when surfacing metadata in notebooks or REST
        responses.  Each entry is converted into a serialisable dictionary with
        a copy of the parameter mapping to avoid accidental mutation of the
        internal state.
        """

        summary: List[Dict[str, object]] = []
        for entry in sorted(self._entries.values(), key=lambda item: item.name.lower()):
            summary.append(
                {
                    "name": entry.name,
                    "tags": list(entry.tags),
                    "citation": entry.citation,
                    "parameters": dict(entry.parameters),
                }
            )
        return summary
