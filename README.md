# QuMater —— 面向硬件无关量子模拟的工具包

QuMater 提供了一套轻量但表达力充足的量子材料数据集、变分线路和参考量子算法实现，灵感来自腾讯新闻在 2025 年对 Phasecraft 的报道。项目展示了如何将“硬件无关”的研究流程封装成具有生产级易用性的 Python 包，并配套测试体系。

## 目录

1. [项目概览](#项目概览)
2. [核心能力](#核心能力)
3. [代码结构](#代码结构)
4. [安装方式](#安装方式)
5. [快速上手](#快速上手)
6. [内置算法模块](#内置算法模块)
7. [扩展注册表](#扩展注册表)
8. [测试与质量保障](#测试与质量保障)
9. [复现自述文件示例](#复现自述文件示例)
10. [支持与贡献](#支持与贡献)
11. [许可证](#许可证)

## 项目概览

工具包刻意对齐了腾讯报道中提到的研发流程：

- **精选材料数据元信息**：便于在近端量子硬件上进行基准测试。
- **硬件无关的变分线路**：保持浅层电路深度，同时捕获关键相关性。
- **模块化算法注册表**：同时暴露 QuMater 的参考实现与通过 `importlib.metadata` 入口点自动发现的第三方扩展。

整个包只依赖 NumPy，研究人员可直接将其嵌入笔记本、服务原型或学术仓库中使用。

## 核心能力

- **材料与模型目录** —— `qumater.materials` 提供示例数据库（`QuantumMaterialDatabase.demo()`），包含富元数据条目及标签过滤、数值范围查询等工具。
- **硬件友好的变分线路** —— `HardwareAgnosticAnsatz` 实现交替层结构（参数化单量子比特旋转 + 环形 CZ 纠缠），并给出用于自然梯度优化的解析梯度。
- **低深度 VQE** —— `LowDepthVQE` 结合测量分组的 Pauli 哈密顿量与近似量子自然梯度更新规则。
- **经典量子算法集合** —— 提供 Grover 搜索、量子傅里叶变换（QFT）、量子相位估计（QPE）等模块化实现。
- **可扩展注册表** —— `AlgorithmModule` / `AlgorithmRegistry` 组成的注册机制会自动加载定义在 `qumater.qsim.algorithms` 入口点下的扩展模块。

## 代码结构

```
qumater/
├── materials/           # 材料元数据目录与晶格工具
├── qsim/                # 变分线路、哈密顿量、算法与注册表
reports/readme_examples_output.md  # 自述文件代码片段的输出记录
tests/                   # 完整的 pytest 测试套件
```

各模块都显式导出了 `__all__`，便于下游项目引用。测试既覆盖功能行为，也覆盖关键错误路径（例如参数越界、非正规酉矩阵等）。

## 安装方式

QuMater 适配 Python 3.10 及以上版本。

```bash
git clone https://github.com/<your-org>/qumater.git
cd qumater
python -m venv .venv
source .venv/bin/activate  # Windows 请使用 .venv\\Scripts\\activate
pip install --upgrade pip
pip install -e .
```

采用可编辑安装可以让包与本地代码改动保持同步，非常适合科研工作流与下游实验。

## 快速上手

以下类 notebook 的示例展示了完整流程：加载材料、构建硬件无关变分线路、使用 VQE 优化、调用算法注册表。

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

# 密度矩阵期望值工具
rho = np.eye(2, dtype=complex) / 2
print("Expectation via density matrix:", hamiltonian.expectation_density(rho))

# 通过注册表实例化算法
registry = get_algorithm_registry()
low_depth = registry.create(
    "low_depth_vqe", hamiltonian=hamiltonian, ansatz=ansatz, learning_rate=0.05
)
print(isinstance(low_depth, LowDepthVQE))
```

为了便于对照，执行上述代码的输出记录在 [`reports/readme_examples_output.md`](reports/readme_examples_output.md) 中。

## 内置算法模块

QuMater 暴露了一个可发现的算法注册表。以下辅助代码将列出所有可用模块及其简介：

```python
from qumater.qsim import get_algorithm_registry

registry = get_algorithm_registry()
for module in registry.available():
    print(f"{module.name:>25} :: {module.summary}")
```

Grover 搜索、QFT 与 QPE 均可通过注册表实例化，或直接从 `qumater.qsim` 导入对应类。每个实现都包含健壮性校验（例如 QPE 会拒绝非酉矩阵，Grover 要求至少存在一个标记态），以模拟生产环境需求。

## 扩展注册表

第三方包可以通过向全局注册表注册，或在 `pyproject.toml` 中暴露 `qumater.qsim.algorithms` 入口点来提供新算法。手动注册同样十分简单：

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
        summary="返回参数的平方值。",
        factory=lambda parameter: CustomAlgorithm(parameter),
        keywords=("演示", "原型"),
    ),
    overwrite=True,
)
```

依赖入口点的包可以返回 `AlgorithmModule` 对象，或返回一个接收注册表并执行自定义注册的可调用对象。

## 测试与质量保障

项目内置全面的 pytest 套件，覆盖材料、哈密顿量、变分线路、算法注册表以及所有内置算法。默认测试命令如下：

```bash
pytest
```

常用参数：

- `pytest -vv` —— 输出更详细的断言信息。
- `pytest -k "grover"` —— 仅运行包含关键字的测试子集。
- `pytest --maxfail=1` —— 首次失败即停止，适合快速迭代。

全部测试都是确定性的，并可在普通硬件上快速完成，非常适合持续集成与发布管控。

## 复现自述文件示例

若需复现 [`reports/readme_examples_output.md`](reports/readme_examples_output.md) 中的结果，请在交互式 Python 会话中执行 [快速上手](#快速上手) 和 [内置算法模块](#内置算法模块) 两节的代码。定期更新该报告可以确保文档始终反映代码库的当前行为。

## 支持与贡献

欢迎通过拉取请求贡献代码。提交新功能或修复缺陷时，请附上相应测试与文档更新。若有问题或功能需求，可在仓库的问题追踪中反馈。

对于内部使用场景，建议在下游项目的 `pyproject.toml` 或依赖约束文件中固定 QuMater 版本，以确保可复现的构建。

## 许可证

QuMater 依据 [MIT License](LICENSE) 发布。

