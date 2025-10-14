# QuMater – Hardware-Agnostic Quantum Simulation Toolkit

QuMater packages a lightweight yet expressive collection of quantum-material
datasets, variational ansätze, and reference quantum algorithms inspired by the
industrial initiatives highlighted in Tencent's 2025 coverage of Phasecraft.
The project demonstrates how a *hardware-agnostic* research workflow can be
captured in a reusable Python package with production-grade ergonomics and
testing.

> 中文摘要：QuMater 旨在复刻 Phasecraft 在腾讯新闻报道中提到的“硬件无关”量子模
> 拟平台原型，提供可复用的数据目录、变分线路与常见量子算法实现，帮助研究者快速搭建
> 工程化原型并开展后续实验。

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Capabilities](#key-capabilities)
3. [Repository Layout](#repository-layout)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Built-in Algorithm Modules](#built-in-algorithm-modules)
7. [Extending the Registry](#extending-the-registry)
8. [Testing & Quality Assurance](#testing--quality-assurance)
9. [Reproducing the README Examples](#reproducing-the-readme-examples)
10. [Support & Contribution](#support--contribution)
11. [License](#license)

## Project Overview

The toolkit intentionally mirrors the workflow described in the Tencent article:

- **Curated materials metadata** for benchmarking near-term quantum hardware.
- **Hardware-agnostic ansätze** that maintain shallow circuit depth while still
  capturing relevant correlations.
- **A modular algorithm registry** that exposes both QuMater's reference
  implementations and third-party extensions discovered through
  `importlib.metadata` entry points.

The package is deliberately dependency-light (only NumPy) so that researchers
can embed it directly into notebooks, service prototypes, or academic
repositories.

## Key Capabilities

- **Materials & model catalogue** – `qumater.materials` provides a
  demonstration database (`QuantumMaterialDatabase.demo()`) with metadata-rich
  entries and utilities such as tag-based filtering and numeric range queries.
- **Hardware-friendly ansatz** – `HardwareAgnosticAnsatz` implements an
  alternating layered structure (parameterised single-qubit rotations + ring CZ
  entanglers) with analytic gradients for natural-gradient optimisation.
- **Low-depth VQE** – `LowDepthVQE` combines measurement-grouped Pauli
  Hamiltonians with an approximate quantum natural gradient update rule.
- **Canonical algorithms** – Grover search, Quantum Fourier Transform, and
  Quantum Phase Estimation are available as easily-instantiable modules.
- **Extensible registry** – `AlgorithmModule`/`AlgorithmRegistry` form a
  registry that automatically loads additional modules defined via
  `qumater.qsim.algorithms` entry points.

## Repository Layout

```
qumater/
├── materials/           # Material metadata catalogue & lattice helpers
├── qsim/                # Ansatz, Hamiltonian, algorithms, and registry tools
reports/readme_examples_output.md  # Captured outputs for documentation snippets
tests/                   # Comprehensive pytest suite
```

Each module exposes an explicit `__all__` to simplify consumption from other
projects.  The tests exercise functional behaviour as well as critical error
paths (e.g. invalid parameter bounds, malformed unitaries).

## Installation

QuMater targets Python 3.10+.

```bash
git clone https://github.com/<your-org>/qumater.git
cd qumater
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

The editable install keeps the package in sync with local code edits—ideal for
research workflows and downstream experimentation.

## Quick Start

The following notebook-style snippet demonstrates the complete workflow: loading
materials, constructing a hardware-agnostic ansatz, optimising with VQE, and
using the algorithm registry.

```python
import numpy as np

from qumater.materials import QuantumMaterialDatabase
from qumater.qsim import (
    HardwareAgnosticAnsatz,
    LowDepthVQE,
    PauliHamiltonian,
    PauliTerm,
    get_algorithm_registry,
)

db = QuantumMaterialDatabase.demo()
print(db.summary())

lih = db.get("LiH minimal basis")
ansatz = HardwareAgnosticAnsatz(num_qubits=1, layers=1)
hamiltonian = PauliHamiltonian([PauliTerm(1.0, "Z")])

optimiser = LowDepthVQE(hamiltonian, ansatz, learning_rate=0.1)
history = optimiser.run()
print("Final energy:", history.energies[-1])

# Density matrix expectation value helper
rho = np.eye(2, dtype=complex) / 2
print("Expectation via density matrix:", hamiltonian.expectation_density(rho))

# Instantiate algorithms via the registry
registry = get_algorithm_registry()
low_depth = registry.create(
    "low_depth_vqe", hamiltonian=hamiltonian, ansatz=ansatz, learning_rate=0.05
)
print(isinstance(low_depth, LowDepthVQE))
```

For convenience the resulting console output is captured in
[`reports/readme_examples_output.md`](reports/readme_examples_output.md).

## Built-in Algorithm Modules

QuMater exposes a discoverable registry of algorithms.  The helper below lists
all available modules along with their short descriptions:

```python
from qumater.qsim import get_algorithm_registry

registry = get_algorithm_registry()
for module in registry.available():
    print(f"{module.name:>25} :: {module.summary}")
```

Grover search, QFT, and QPE can be instantiated directly through the registry
or via the concrete classes imported from `qumater.qsim`.  Each implementation
includes validation logic (e.g. QPE rejects non-unitary matrices and Grover
requires at least one marked state) to emulate production robustness.

## Extending the Registry

Third-party packages can expose new algorithms by registering them with the
global registry or by exporting an entry point in `pyproject.toml` under the
`qumater.qsim.algorithms` group.  Manual registration is equally simple:

```python
from qumater.qsim.modules import AlgorithmModule, register_algorithm_module


class CustomAlgorithm:
    def __init__(self, parameter: float) -> None:
        self.parameter = parameter

    def run(self) -> float:
        return self.parameter ** 2


register_algorithm_module(
    AlgorithmModule(
        name="custom_algorithm",
        summary="Squares the provided parameter.",
        factory=lambda parameter: CustomAlgorithm(parameter),
        keywords=("demo", "prototype"),
    ),
    overwrite=True,
)
```

Packages that rely on entry points can return an `AlgorithmModule` object or a
callable receiving the registry and performing custom registration.

## Testing & Quality Assurance

The project ships with a comprehensive pytest suite covering materials,
Hamiltonians, variational ansätze, the algorithm registry, and all built-in
algorithms.  Run the default suite via:

```bash
pytest
```

Useful options:

- `pytest -vv` – verbose assertion output.
- `pytest -k "grover"` – focus on a specific subset of tests.
- `pytest --maxfail=1` – abort on first failure, useful for rapid iteration.

All tests are deterministic and execute quickly on commodity hardware, making
them suitable for CI pipelines and gated deployments.

## Reproducing the README Examples

To recreate the outputs documented in
`reports/readme_examples_output.md`, execute the snippets from the
[Quick Start](#quick-start) and [Built-in Algorithm Modules](#built-in-algorithm-modules)
sections within an interactive Python session.  Updating the report with fresh
results ensures the documentation always reflects the current behaviour of the
codebase.

## Support & Contribution

Contributions are welcome via pull requests.  Please accompany new features or
bug fixes with tests and documentation updates.  Issues and feature requests can
be filed through the repository's tracker.

For internal adoption, consider pinning QuMater's version in downstream
`pyproject.toml` files or through a constraints file to guarantee reproducible
builds.

## License

QuMater is distributed under the terms of the [MIT License](LICENSE).

