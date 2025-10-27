
# QuMater —— 面向硬件无关量子模拟的工具包

QuMater 提供了一套轻量但表达力充足的量子材料数据集、变分线路和参考量子算法实现，灵感来自腾讯新闻在 2025 年对 Phasecraft 的报道。项目展示了如何将“硬件无关”的研究流程封装成具有生产级易用性的 Python 包，并配套测试体系。


## 目录

1. [项目概览](#项目概览)
2. [实用目标与任务](#实用目标与任务)
3. [体系架构与模块划分](#体系架构与模块划分)
4. [核心能力](#核心能力)
5. [仓库结构](#仓库结构)
6. [安装步骤](#安装步骤)
7. [配置化工作流](#配置化工作流)
8. [命令行界面](#命令行界面)
9. [快速上手](#快速上手)
10. [内置算法模块](#内置算法模块)
11. [扩展注册表](#扩展注册表)
12. [测试与质量保障](#测试与质量保障)
13. [复现实验输出](#复现实验输出)
14. [支持与贡献](#支持与贡献)
15. [许可证](#许可证)

## 项目概览

本工具包刻意贴近腾讯报道中的工作流：

- **精选材料元数据**：便于评估近期量子硬件性能。
- **硬件无关变分线路**：保持浅层电路深度，同时捕捉关键关联效应。
- **模块化算法注册表**：既暴露 QuMater 内置实现，也可通过 `importlib.metadata` 的入口点机制发现第三方扩展。

整个包仅依赖 NumPy，方便研究者直接在笔记本、服务原型或学术仓库中嵌入使用。

## 实用目标与任务

为了让 QuMater 成为涵盖材料、算法与平台编排的**综合性量子计算软件**，我们将研发目标拆解为三个面向实践的维度，并在代码中通过 `qumater.core.objectives` 明确跟踪：

| 目标 | 说明 | 关键任务 |
| --- | --- | --- |
| 数据到模型的稳健流程 | 从材料目录到哈密顿量构建形成闭环，降低前期准备成本。 | `material-catalogue-selection`、`hamiltonian-build` |
| 可演化的算法与编排 | 以模块化结构衔接线路、算法与报表，便于调试与扩展。 | `ansatz-construction`、`algorithm-instantiation`、`result-reporting` |
| 易用的操作界面 | 提供一致的 CLI/API，让算法能快速落地到实际项目。 | `interface-automation` |

`ObjectivePlanner` 会在工作流执行过程中逐项勾选任务，CLI 也会反馈进度，帮助团队随时审视架构是否达标。

## 体系架构与模块划分

新版 QuMater 在原有 `materials`、`qsim`、`platform` 模块基础上，新增以下核心层级，增强可维护性与扩展性：

- **`qumater.core`**：承载 `WorkflowConfig`、`AlgorithmConfig` 等配置数据类，以及与上表目标一致的规划体系 `ObjectivePlanner`。
- **`qumater.workflows`**：`QuantumWorkflow` 负责编排“选材 → 构建哈密顿量 → 生成 Ansatz → 运行算法 → 汇总报告”的流水线，所有步骤都具备显式的依赖注入点，方便测试与调试。
- **`qumater.cli`**：命令行入口统一加载 JSON/TOML 配置，既可 dry-run 预览任务，也可执行工作流并输出中文报表。

这种分层让配置、编排与接口解耦，既能支撑研究原型的快速试验，也易于在生产环境中接入日志、监控或外部调度器。

CLI 的执行结果会自动附带“创新洞察”章节，展示差异化与风险对冲评分、成熟度标签以及后续建议，使量子算法成果具备直接可引用的业务阐述。

## 核心能力

- **材料与模型目录**：`qumater.materials` 提供示例数据库（`QuantumMaterialDatabase.demo()`），包含丰富的元数据以及按标签过滤、数值区间筛选等工具。
- **硬件友好型 Ansatz**：`HardwareAgnosticAnsatz` 实现交替层结构（参数化单比特旋转 + 环形 CZ 纠缠），并给出解析梯度以支持自然梯度优化。
- **低深度 VQE**：`LowDepthVQE` 将分组测量的 Pauli 哈密顿量与近似量子自然梯度更新结合。
- **经典算法参考实现**：内置 Grover 搜索、量子傅里叶变换（QFT）与量子相位估计（QPE），均可直接实例化。
- **可扩展注册表**：`AlgorithmModule` 与 `AlgorithmRegistry` 组成的注册体系会自动加载在 `qumater.qsim.algorithms` 入口点下声明的额外模块。
- **战略洞察引擎**：`InnovationInsightEngine` 会读取材料标签、哈密顿量结构与优化轨迹，生成差异化/风险对冲评分与亮点摘要，帮助团队快速说明业务价值。

## 仓库结构


```
qumater/
├── core/                # WorkflowConfig、ObjectivePlanner 等核心基建
├── materials/           # 材料元数据目录与晶格工具
├── workflows/           # QuantumWorkflow 及报告数据结构
├── qsim/                # 变分线路、哈密顿量、算法与注册工具
├── platform/            # 平台与应用场景抽象
├── cli.py               # 命令行入口（`python -m qumater.cli`）
reports/readme_examples_output.md  # 文档示例的控制台输出
tests/                   # 覆盖全面的 pytest 测试套件
```

各模块均显式导出 `__all__`，便于其他项目引用。新增的 `core` 与 `workflows` 将配置、流程与接口拆分开来，测试也覆盖了配置解析、任务规划与 CLI 行为，确保关键错误分支（如非法参数区间、未知任务标识等）都能被捕获。

## 安装步骤

QuMater 目标运行环境为 Python 3.10 及以上版本。

```bash
git clone https://github.com/<your-org>/qumater.git
cd qumater
python -m venv .venv
source .venv/bin/activate  # Windows 请执行 .venv\\Scripts\\activate
pip install --upgrade pip
pip install -e .
```

以可编辑模式安装，便于在研究或二次开发时实时同步源码改动。

## 配置化工作流

`qumater.core.config` 提供的配置数据类使得工作流可以通过 JSON/TOML 文件描述，无需在脚本中硬编码。一个最小示例如下：

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

`WorkflowConfig.from_dict()` 会自动校验字段合法性（例如 Pauli 字符串非空、线路层数为正等），并在执行前转换成 `PauliHamiltonian`、`HardwareAgnosticAnsatz` 等内部对象。

## 命令行界面

`qumater.cli` 将配置解析、工作流编排和任务进度跟踪整合在一个命令中：

```bash
python -m qumater.cli path/to/workflow.json
```

常用选项：

- `--dry-run`：仅输出任务规划（即 `material-catalogue-selection` 等）及当前完成度，适合在集成前验证配置。
- 不同后缀：`.json` 与 `.toml` 均被支持，后者依赖 Python 3.11+ 自带的 `tomllib`。

执行后的控制台输出会包含材料、算法、能量收敛情况以及上文三大目标的完成比例，便于研究团队快速了解流程是否满足预期。

## 快速上手

以下类笔记本片段演示完整流程：载入材料、构造硬件无关 ansatz、使用 VQE 优化，并调用算法注册表。

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

# 利用密度矩阵求期望值的辅助函数
rho = np.eye(2, dtype=complex) / 2
print("Expectation via density matrix:", hamiltonian.expectation_density(rho))

# 通过注册表实例化算法
registry = get_algorithm_registry()
low_depth = registry.create(
    "low_depth_vqe", hamiltonian=hamiltonian, ansatz=ansatz, learning_rate=0.05
)
print(isinstance(low_depth, LowDepthVQE))
```

相关控制台输出已整理至 [`reports/readme_examples_output.md`](reports/readme_examples_output.md)。

若希望以配置驱动的方式运行相同流程，可以使用 `QuantumWorkflow`：

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

## 内置算法模块

QuMater 暴露可发现的算法注册表。下面的助手代码会列出所有可用模块及其简介：

```python
from qumater.qsim import get_algorithm_registry

registry = get_algorithm_registry()
for module in registry.available():
    print(f"{module.name:>25} :: {module.summary}")
```

内置的 Grover 搜索、QFT 与 QPE 既可通过注册表创建，也能直接从 `qumater.qsim` 导入具体类。每个实现都包含必要的校验逻辑（例如 QPE 会拒绝非酉矩阵，Grover 至少需要一个标记态），以贴近生产环境的健壮性。

## 扩展注册表

第三方包可以手动向全局注册表添加算法，或在 `pyproject.toml` 中的 `qumater.qsim.algorithms` 分组下导出入口点。手动注册示例如下：

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

若通过入口点机制，既可以直接返回 `AlgorithmModule` 对象，也可以返回接受注册表并执行自定义注册的可调用对象。

## 测试与质量保障

项目自带全面的 pytest 套件，覆盖材料模块、哈密顿量、变分线路、算法注册表以及所有内置算法。执行默认测试命令：

```bash
pytest
```

常用选项：

- `pytest -vv` —— 输出更详细的断言信息。
- `pytest -k "grover"` —— 聚焦特定子集的测试。
- `pytest --maxfail=1` —— 首次失败即中止，便于快速迭代。

全部测试均为确定性设计，可在常规硬件上迅速完成，适合作为 CI 流水线或发布前的质量门禁。

## 复现实验输出

若需重现 [`reports/readme_examples_output.md`](reports/readme_examples_output.md) 中的结果，请在交互式 Python 环境中运行 [快速上手](#快速上手) 与 [内置算法模块](#内置算法模块) 两节的代码片段。定期刷新报告有助于保持文档与当前代码行为一致。

## 支持与贡献

欢迎通过 Pull Request 贡献代码。新增功能或修复缺陷时，请同时补充测试与文档。若有问题或功能需求，可在仓库 Issue 区提交。

在内部环境中使用时，建议在下游 `pyproject.toml` 或约束文件中锁定 QuMater 的版本，以确保可重复的构建结果。

## 许可证

QuMater 依据 [MIT 许可证](LICENSE) 发布。
