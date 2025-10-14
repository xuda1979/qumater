# QuMater：硬件无关的量子模拟与材料工具包

QuMater 聚合了轻量却表达力十足的构件，用于搭建硬件无关的量子模拟工作流，并帮助研究者快速构建量子算法模块。设计灵感来自 [腾讯 2025 年报道](https://news.qq.com/rain/a/20250903A07NHG00)中介绍的 Phasecraft 前沿算法，软件内置包括 Grover 搜索、QFT 与 QPE 在内的经典量子算法实现，可直接复用，同时也提供面向多个科学领域（不仅限于材料科学）的模型化能力。框架重点聚焦于：

- 具备丰富元数据的材料目录，贴近工业级数据集；
- 低深度、硬件高效且稀疏纠缠的变分参数化；
- 结合测量分组启发式的量子自然梯度优化。

## 功能特性

- `qumater.materials` 提供演示用材料数据库与 Hubbard 晶格工具，可作为模拟基准的起点。
- `qumater.qsim` 实现了 Pauli 哈密顿量辅助函数、可交换项分组（带缓存矩阵加速）以及基于 Fubini–Study 度量的低深度 VQE 求解器。
- `qumater.qsim.modules` 提供量子算法模块注册表，可发现通过 `entry_points` 发布的第三方算法。
- `qumater.qsim.algorithms` 现内置 Grover 搜索、量子傅里叶变换 (QFT) 与量子相位估计 (QPE) 等经典算法实现，可直接通过模块注册表获取并运行。
- 材料目录提供 `summary()` 辅助方法，可直接生成用于可视化或 API 的序列化视图。
- `PauliHamiltonian` 现支持对密度矩阵求期望值，便于混合态或实验测量数据的分析。

## 快速开始

```bash
pip install -e .
pytest
```

## 示例

```python
from qumater.materials import QuantumMaterialDatabase
from qumater.qsim import (
    HardwareAgnosticAnsatz,
    LowDepthVQE,
    PauliHamiltonian,
    PauliTerm,
    get_algorithm_registry,
)

db = QuantumMaterialDatabase.demo()
print(db.summary())  # 结构化查看目录内容
lih = db.get("LiH minimal basis")
ansatz = HardwareAgnosticAnsatz(num_qubits=1, layers=1)
h = PauliHamiltonian([PauliTerm(1.0, "Z")])
optimiser = LowDepthVQE(h, ansatz)
result = optimiser.run()
print(result.energies[-1])

# 利用实验或模拟获得的密度矩阵来评估能量
import numpy as np
rho = np.eye(2, dtype=complex) / 2
print(h.expectation_density(rho))

# 通过模块注册表按名称实例化算法
registry = get_algorithm_registry()
low_depth = registry.create(
    "low_depth_vqe",
    hamiltonian=h,
    ansatz=ansatz,
    learning_rate=0.1,
)
print(isinstance(low_depth, LowDepthVQE))
```

## 扩展量子算法模块

`qumater.qsim.modules` 允许使用者轻松包装并分发新的量子算法：

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
    )
)
```

第三方包可在 `pyproject.toml` 中声明 `qumater.qsim.algorithms` 入口点，
QuMater 会在运行时自动发现并注册这些算法。
