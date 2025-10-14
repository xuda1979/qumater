# Extended Quantum Algorithm Module Test Report

## Pytest Results

Command: `pytest tests/test_algorithm_modules.py`

```
===================================================== test session starts ======================================================
platform linux -- Python 3.12.10, pytest-8.4.1, pluggy-1.6.0
rootdir: /workspace/qumater
configfile: pyproject.toml
collected 6 items

tests/test_algorithm_modules.py ......                                                                                   [100%]

====================================================== 6 passed in 0.48s =======================================================
```

## Example Execution

Command:

```bash
python example_custom_module.py
```

Inlined command:

```python
from qumater.materials import QuantumMaterialDatabase
from qumater.qsim import (
    HardwareAgnosticAnsatz,
    LowDepthVQE,
    PauliHamiltonian,
    PauliTerm,
    get_algorithm_registry,
)
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
    )
)

db = QuantumMaterialDatabase.demo()
summary = db.summary()
ansatz = HardwareAgnosticAnsatz(num_qubits=1, layers=1)
h = PauliHamiltonian([PauliTerm(1.0, "Z")])
optimiser = LowDepthVQE(h, ansatz, max_iterations=5)
result = optimiser.run()
print("Final energy:", result.energies[-1])
print("Registry has low_depth_vqe:", isinstance(
    get_algorithm_registry().create("low_depth_vqe", hamiltonian=h, ansatz=ansatz),
    LowDepthVQE,
))
print("Custom algorithm output:", get_algorithm_registry().create("custom_algorithm", parameter=3.0).run())
print(summary)
```

Output:

```
Final energy: 0.9999999999899957
Registry has low_depth_vqe: True
Custom algorithm output: 9.0
[{'name': 'Fermi-Hubbard 2x2', 'tags': ['hubbard', 'strongly-correlated', 'benchmark'], 'citation': 'Phasecraft demo circuits (private communication)', 'parameters': {'lattice_size': 4, 'u': 4.0, 't': 1.0}}, {'name': 'FeSe monolayer', 'tags': ['superconductor', 'high-Tc'], 'citation': 'Tencent News (2025)', 'parameters': {'Tc': 65.0, 'doping': 0.12}}, {'name': 'LiH minimal basis', 'tags': ['molecular', 'variational', 'benchmarks'], 'citation': 'Tencent News (2025). Phasecraft announces hardware-agnostic quantum materials platform.', 'parameters': {'electrons': 2, 'orbitals': 4}}]
```
