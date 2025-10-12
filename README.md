# QuMater：硬件无关的量子模拟与材料工具包

QuMater 聚合了轻量却表达力十足的构件，用于搭建硬件无关的量子模拟工作流。设计灵感来自 [腾讯 2025 年报道](https://news.qq.com/rain/a/20250903A07NHG00)中介绍的 Phasecraft 前沿算法，重点聚焦于：

- 具备丰富元数据的材料目录，贴近工业级数据集；
- 低深度、硬件高效且稀疏纠缠的变分参数化；
- 结合测量分组启发式的量子自然梯度优化。

## 功能特性

- `qumater.materials` 提供演示用材料数据库与 Hubbard 晶格工具，可作为模拟基准的起点。
- `qumater.qsim` 实现了 Pauli 哈密顿量辅助函数、可交换项分组以及基于 Fubini–Study 度量的低深度 VQE 求解器。

## 快速开始

```bash
pip install -e .
pytest
```

## 示例

```python
from qumater.materials import QuantumMaterialDatabase
from qumater.qsim import HardwareAgnosticAnsatz, LowDepthVQE, PauliHamiltonian, PauliTerm

db = QuantumMaterialDatabase.demo()
lih = db.get("LiH minimal basis")
ansatz = HardwareAgnosticAnsatz(num_qubits=1, layers=1)
h = PauliHamiltonian([PauliTerm(1.0, "Z")])
optimiser = LowDepthVQE(h, ansatz)
result = optimiser.run()
print(result.energies[-1])
```
