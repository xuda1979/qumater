# QuMater — A Toolkit for Hardware-Agnostic Quantum Simulation

QuMater provides a lightweight yet expressive suite of quantum material datasets, variational circuits, and reference quantum algorithm implementations, inspired by a 2025 Tencent News report on Phasecraft. The project demonstrates how to encapsulate a "hardware-agnostic" research workflow into a production-ready Python package with a comprehensive testing suite.

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Practical Objectives and Tasks](#practical-objectives-and-tasks)
3.  [Architecture and Module Division](#architecture-and-module-division)
4.  [Core Capabilities](#core-capabilities)
5.  [Repository Structure](#repository-structure)
6.  [Installation Steps](#installation-steps)
7.  [Configuration-Driven Workflows](#configuration-driven-workflows)
8.  [Command-Line Interface](#command-line-interface)
9.  [Quick Start](#quick-start)
10. [Built-in Algorithm Modules](#built-in-algorithm-modules)
11. [Extensible Registry](#extensible-registry)
12. [Testing and Quality Assurance](#testing-and-quality-assurance)
13. [Reproducing Experimental Outputs](#reproducing-experimental-outputs)
14. [Support and Contribution](#support-and-contribution)
15. [License](#license)

## Project Overview

This toolkit is intentionally designed to align with the workflow described in the Tencent report:

-   **Curated Material Metadata**: Facilitates the evaluation of near-term quantum hardware performance.
-   **Hardware-Agnostic Variational Circuits**: Maintain shallow circuit depth while capturing critical correlation effects.
-   **Modular Algorithm Registry**: Exposes both QuMater's built-in implementations and allows discovery of third-party extensions via `importlib.metadata` entry points.

The entire package depends only on NumPy, making it easy for researchers to embed it directly into notebooks, service prototypes, or academic repositories.

## Practical Objectives and Tasks

To establish QuMater as a **comprehensive quantum computing software** covering materials, algorithms, and platform orchestration, we have broken down our development goals into three practice-oriented dimensions, which are explicitly tracked in the code via `qumater.core.objectives`:

| Objective                         | Description                                                                          | Key Tasks                                                      |
| --------------------------------- | ------------------------------------------------------------------------------------ | -------------------------------------------------------------- |
| Robust Data-to-Model Pipeline     | Create a closed loop from the material catalogue to Hamiltonian construction, reducing preparatory overhead. | `material-catalogue-selection`, `hamiltonian-build`            |
| Evolvable Algorithms & Orchestration | Link circuits, algorithms, and reports in a modular structure for easy debugging and extension. | `ansatz-construction`, `algorithm-instantiation`, `result-reporting` |
| User-Friendly Interface           | Provide a consistent CLI/API to rapidly deploy algorithms in real-world projects.      | `interface-automation`                                         |

The `ObjectivePlanner` checks off tasks during workflow execution, and the CLI provides progress feedback, helping the team continuously assess whether the architecture meets its goals.

## Architecture and Module Division

The new version of QuMater builds upon the original `materials`, `qsim`, and `platform` modules by adding the following core layers to enhance maintainability and extensibility:

-   **`qumater.core`**: Hosts configuration data classes like `WorkflowConfig` and `AlgorithmConfig`, as well as the `ObjectivePlanner` system aligned with the objectives table above.
-   **`qumater.workflows`**: The `QuantumWorkflow` orchestrates the pipeline: "select material → build Hamiltonian → generate Ansatz → run algorithm → summarise report." All steps feature explicit dependency injection points for easy testing and debugging.
-   **`qumater.cli`**: The command-line entry point loads JSON/TOML configurations, allowing for both dry-runs to preview tasks and execution of workflows with reports generated in English.

This layering decouples configuration, orchestration, and interface, supporting both rapid prototyping for research and integration with logging, monitoring, or external schedulators in a production environment.

The CLI output automatically includes an "Innovation Insights" section, displaying differentiation and risk hedging scores, maturity labels, and follow-up recommendations, enabling quantum algorithm results to be framed in directly citable business terms.

## Core Capabilities

-   **Material and Model Catalogue**: `qumater.materials` provides a demo database (`QuantumMaterialDatabase.demo()`) with rich metadata and tools for filtering by tags and numerical ranges.
-   **Hardware-Friendly Ansatz**: `HardwareAgnosticAnsatz` implements an alternating layer structure (parameterised single-qubit rotations + ring CZ entanglement) and provides analytical gradients to support natural gradient optimisation.
-   **Low-Depth VQE**: `LowDepthVQE` combines grouped-measurement Pauli Hamiltonians with approximate quantum natural gradient updates.
-   **Classical Algorithm Reference Implementations**: Built-in Grover search, Quantum Fourier Transform (QFT), and Quantum Phase Estimation (QPE) are all directly instantiable.
-   **Extensible Registry**: The registration system, composed of `AlgorithmModule` and `AlgorithmRegistry`, automatically loads additional modules declared under the `qumater.qsim.algorithms` entry point.
-   **Strategic Insight Engine**: `InnovationInsightEngine` reads material tags, Hamiltonian structures, and optimisation trajectories to generate differentiation/risk-hedging scores and highlight summaries, helping teams quickly articulate business value.

## Repository Structure

```
qumater/
├── core/                # Core infrastructure like WorkflowConfig, ObjectivePlanner
├── materials/           # Material metadata catalogue and lattice tools
├── workflows/           # QuantumWorkflow and report data structures
├── qsim/                # Variational circuits, Hamiltonians, algorithms, and registration tools
├── platform/            # Abstractions for platforms and application scenarios
├── cli.py               # Command-line entry point (`python -m qumater.cli`)
reports/readme_examples_output.md  # Console output for documentation examples
tests/                   # Comprehensive pytest test suite
```

Each module explicitly exports `__all__` for easy referencing by other projects. The new `core` and `workflows` modules separate configuration, processes, and interfaces. Tests also cover configuration parsing, task planning, and CLI behaviour, ensuring critical error branches (like invalid parameter ranges or unknown task identifiers) are caught.

## Installation Steps

QuMater is targeted to run on Python 3.10 and above.

```bash
git clone https://github.com/<your-org>/qumater.git
cd qumater
python -m venv .venv
source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

Installing in editable mode allows source code changes to be reflected in real-time during research or secondary development.

## Configuration-Driven Workflows

The configuration data classes provided by `qumater.core.config` allow workflows to be described via JSON/TOML files, eliminating the need for hardcoding in scripts. Here is a minimal example:

```json
{
  "material": {"name": "LiH minimal basis"},
  "hamiltonian": {"terms": [[1.0, "Z"]]},
  "ansatz": {"num_qubits": 1, "layers": 1},
  "algorithm": {
    "name": "low_depth_vqe",
    "options": {"learning_rate": 0.2, "max_iterations": 20}
  }
}
```

`WorkflowConfig.from_dict()` automatically validates field legality (e.g., non-empty Pauli strings, positive circuit layers) and converts them into internal objects like `PauliHamiltonian` and `HardwareAgnosticAnsatz` before execution.

## Command-Line Interface

`qumater.cli` integrates configuration parsing, workflow orchestration, and task progress tracking into a single command:

```bash
python -m qumater.cli path/to/workflow.json
```

Common options:

-   `--dry-run`: Only outputs the task plan (i.e., `material-catalogue-selection`, etc.) and its current completion status, suitable for validating configurations before integration.
-   Different suffixes: Both `.json` and `.toml` are supported, with the latter relying on the `tomllib` built into Python 3.11+.

The console output after execution includes material, algorithm, energy convergence details, and the completion percentage of the three major objectives mentioned earlier, helping research teams quickly assess if the process meets expectations.

## Quick Start

The following notebook-like snippet demonstrates a complete workflow: loading a material, constructing a hardware-agnostic ansatz, optimising with VQE, and calling the algorithm registry.

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

# Helper function to calculate expectation value using a density matrix
rho = np.eye(2, dtype=complex) / 2
print("Expectation via density matrix:", hamiltonian.expectation_density(rho))

# Instantiate an algorithm via the registry
registry = get_algorithm_registry()
low_depth = registry.create(
    "low_depth_vqe", hamiltonian=hamiltonian, ansatz=ansatz, learning_rate=0.05
)
print(isinstance(low_depth, LowDepthVQE))
```

The corresponding console output has been collated in [`reports/readme_examples_output.md`](reports/readme_examples_output.md).

To run the same process in a configuration-driven manner, use `QuantumWorkflow`:

```python
from qumater.core import AlgorithmConfig, AnsatzConfig, HamiltonianConfig, MaterialSelection, WorkflowConfig
from qumater.workflows import QuantumWorkflow

config = WorkflowConfig(
    material=MaterialSelection(name="LiH minimal basis"),
    hamiltonian=HamiltonianConfig(terms=((1.0, "Z"),)),
    ansatz=AnsatzConfig(num_qubits=1, layers=1),
    algorithm=AlgorithmConfig(name="low_depth_vqe", options={"learning_rate": 0.1}),
)

report = QuantumWorkflow(config).execute()
print(report.final_energy, report.objective_summary)
```

## Built-in Algorithm Modules

QuMater exposes a discoverable algorithm registry. The helper code below lists all available modules and their summaries:

```python
from qumater.qsim import get_algorithm_registry

registry = get_algorithm_registry()
for module in registry.available():
    print(f"{module.name:>25} :: {module.summary}")
```

Built-in Grover search, QFT, and QPE can be created via the registry or by directly importing their specific classes from `qumater.qsim`. Each implementation includes necessary validation logic (e.g., QPE rejects non-unitary matrices, Grover requires at least one marked state) to align with production-level robustness.

## Extensible Registry

Third-party packages can manually add algorithms to the global registry or export entry points under the `qumater.qsim.algorithms` group in their `pyproject.toml`. Here is an example of manual registration:

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
        summary="Returns the square of a parameter.",
        factory=lambda parameter: CustomAlgorithm(parameter),
        keywords=("demo", "prototype"),
    ),
    overwrite=True,
)
```

When using the entry point mechanism, you can either return an `AlgorithmModule` object directly or a callable that accepts the registry and performs custom registration.

## Testing and Quality Assurance

The project comes with a comprehensive pytest suite covering the materials module, Hamiltonians, variational circuits, the algorithm registry, and all built-in algorithms. To run the default test command:

```bash
pytest
```

Common options:

-   `pytest -vv` — Outputs more detailed assertion information.
-   `pytest -k "grover"` — Focuses on a specific subset of tests.
-   `pytest --maxfail=1` — Stops at the first failure, useful for rapid iteration.

All tests are designed to be deterministic and can complete quickly on standard hardware, making them suitable as a quality gate in a CI pipeline or before a release.

## Reproducing Experimental Outputs

To reproduce the results in [`reports/readme_examples_output.md`](reports/readme_examples_output.md), run the code snippets from the [Quick Start](#quick-start) and [Built-in Algorithm Modules](#built-in-algorithm-modules) sections in an interactive Python environment. Regularly refreshing the report helps keep the documentation consistent with the current code behaviour.

## Support and Contribution

Contributions are welcome via Pull Requests. When adding new features or fixing defects, please also add tests and documentation. If you have questions or feature requests, please submit them in the repository's Issue section.

When using QuMater in an internal environment, it is recommended to lock its version in a downstream `pyproject.toml` or constraints file to ensure reproducible build results.

## License

QuMater is released under the [MIT License](LICENSE).
