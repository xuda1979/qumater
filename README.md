# QuMater：硬件无关的量子模拟工具包

QuMater 聚合了轻量却表达力十足的组件，帮助研究者搭建硬件无关的量子模拟原型。项目灵感来自腾讯 2025 年报道中提到的 Phasecraft 工作：仓库实现了示例性的材料元数据目录、低深度变分线路以及多个经典量子算法的参考版本，便于直接复用或作为进一步研究的基准。

## 核心能力概览

- **材料与模型元数据** —— `qumater.materials` 提供内建演示用目录（`QuantumMaterialDatabase.demo()`）以及基础的 Hubbard 晶格工具。所有条目都带有标签、引用信息和参数字典，方便按照语义或参数范围过滤，同时支持 `summary()` 方法导出结构化视图。
- **低深度、稀疏纠缠的变分参数化** —— `HardwareAgnosticAnsatz` 通过交替的单比特旋转与环形 CZ 纠缠门生成状态，强调硬件友好的线路深度，并暴露梯度计算接口以供上层优化器使用。
- **带测量分组的量子自然梯度 VQE** —— `LowDepthVQE` 利用 `PauliHamiltonian` 对可交换项进行启发式分组并缓存矩阵，结合基于 Fubini–Study 度量的近似量子自然梯度更新策略。
- **量子算法模块注册表** —— `qumater.qsim.modules` 提供轻量级注册表，内置 Grover 搜索、量子傅里叶变换 (QFT) 与量子相位估计 (QPE) 的参考实现，并支持通过 `entry_points` 发现第三方算法。

## 快速开始

```bash
pip install -e .
pytest
```

## 使用示例

```python
from qumater.materials import QuantumMaterialDatabase
from qumater.qsim import (
    HardwareAgnosticAnsatz,
    LowDepthVQE,
    PauliHamiltonian,
    PauliTerm,
    get_algorithm_registry,
)

# 载入演示材料目录并查看结构化摘要
db = QuantumMaterialDatabase.demo()
print(db.summary())
lih = db.get("LiH minimal basis")

# 构建低深度硬件无关 ansatz 和简单哈密顿量
ansatz = HardwareAgnosticAnsatz(num_qubits=1, layers=1)
h = PauliHamiltonian([PauliTerm(1.0, "Z")])
optimiser = LowDepthVQE(h, ansatz)
result = optimiser.run()
print(result.energies[-1])

# 利用密度矩阵数据计算期望值
import numpy as np
rho = np.eye(2, dtype=complex) / 2
print(h.expectation_density(rho))

# 通过算法注册表按名称实例化内建算法
registry = get_algorithm_registry()
low_depth = registry.create(
    "low_depth_vqe",
    hamiltonian=h,
    ansatz=ansatz,
    learning_rate=0.1,
)
print(isinstance(low_depth, LowDepthVQE))
```

### 调用内建的经典量子算法

通过算法注册表，可以直接实例化仓库已包含的 Grover 搜索、量子傅里叶变换 (QFT) 与量子相位估计 (QPE) 参考实现：

```python
import numpy as np

from qumater.qsim import get_algorithm_registry

registry = get_algorithm_registry()

# Grover 搜索：提供被标记的基态索引，返回概率分布及最可能的结果
grover = registry.create("grover_search", num_qubits=3, oracle=[5])
grover_result = grover.run()
print(grover_result.most_likely_state())  # 5
print(grover_result.probabilities)

# 量子傅里叶变换：对态矢量执行 QFT 及其逆操作
qft = registry.create("quantum_fourier_transform", num_qubits=2)
state = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
transformed = qft.run(state)
print(transformed)            # 接近均匀分布的幅度
print(qft.inverse(transformed))  # 还原原始态

# 量子相位估计：给出作用于本征态的酉矩阵与期望精度，返回估计的相位二进制与浮点表示
unitary = np.diag([1.0, np.exp(2j * np.pi * 0.25)])
eigenstate = np.array([0.0, 1.0], dtype=complex)
qpe = registry.create(
    "quantum_phase_estimation",
    unitary=unitary,
    eigenstate=eigenstate,
    precision_qubits=3,
)
phase_result = qpe.run()
print(phase_result.binary)  # '010'
print(phase_result.phase)
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

第三方包可在 `pyproject.toml` 中声明 `qumater.qsim.algorithms` 入口点，QuMater 会在运行时自动发现并注册这些算法。
