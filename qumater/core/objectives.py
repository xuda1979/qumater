"""Planning primitives that encode practical objectives for QuMater workflows."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List


@dataclass
class TaskDefinition:
    """Describe an actionable unit required to reach a practical objective."""

    name: str
    description: str
    category: str
    completed: bool = False

    def mark_completed(self) -> None:
        self.completed = True


@dataclass
class PracticalObjective:
    """Group a set of related tasks under a user facing objective."""

    name: str
    summary: str
    tasks: List[TaskDefinition] = field(default_factory=list)

    def progress(self) -> float:
        if not self.tasks:
            return 1.0
        completed = sum(1 for task in self.tasks if task.completed)
        return completed / len(self.tasks)

    def counts(self) -> Dict[str, int]:
        return {
            "completed": sum(1 for task in self.tasks if task.completed),
            "total": len(self.tasks),
        }


class ObjectivePlanner:
    """Track progress against a set of :class:`PracticalObjective` instances."""

    def __init__(self, objectives: Iterable[PracticalObjective]):
        self._objectives: List[PracticalObjective] = []
        self._task_index: Dict[str, TaskDefinition] = {}
        for objective in objectives:
            self._objectives.append(objective)
            for task in objective.tasks:
                if task.name in self._task_index:
                    raise ValueError(f"Duplicate task name '{task.name}' detected")
                self._task_index[task.name] = task

    @property
    def objectives(self) -> List[PracticalObjective]:
        return self._objectives

    def mark_completed(self, task_name: str) -> None:
        try:
            task = self._task_index[task_name]
        except KeyError as exc:
            raise KeyError(f"Unknown task '{task_name}'") from exc
        task.mark_completed()

    def summary(self) -> Dict[str, Dict[str, float]]:
        report: Dict[str, Dict[str, float]] = {}
        for objective in self._objectives:
            counts = objective.counts()
            report[objective.name] = {
                "completed": float(counts["completed"]),
                "total": float(counts["total"]),
                "progress": objective.progress(),
            }
        return report

    def completed_tasks(self) -> List[str]:
        return [name for name, task in self._task_index.items() if task.completed]


def default_objectives() -> List[PracticalObjective]:
    """Return the default practical objectives highlighted in the README."""

    return [
        PracticalObjective(
            name="数据到模型的稳健流程",
            summary="确保材料数据、哈密顿量建模与线路生成形成闭环。",
            tasks=[
                TaskDefinition(
                    name="material-catalogue-selection",
                    description="从量子材料目录中筛选或获取目标样本。",
                    category="数据准备",
                ),
                TaskDefinition(
                    name="hamiltonian-build",
                    description="将配置化的 Pauli 项构造成可直接模拟的哈密顿量。",
                    category="模型构建",
                ),
            ],
        ),
        PracticalObjective(
            name="可演化的算法与编排",
            summary="搭建可维护、可扩展且可调试的算法执行栈。",
            tasks=[
                TaskDefinition(
                    name="ansatz-construction",
                    description="依据配置创建硬件无关的变分线路。",
                    category="量子线路",
                ),
                TaskDefinition(
                    name="algorithm-instantiation",
                    description="通过注册表实例化所需量子算法或工作流。",
                    category="算法管理",
                ),
                TaskDefinition(
                    name="result-reporting",
                    description="生成包含关键诊断指标的结构化汇总。",
                    category="可观测性",
                ),
            ],
        ),
        PracticalObjective(
            name="易用的操作界面",
            summary="提供面向研究人员与工程师的一致化入口。",
            tasks=[
                TaskDefinition(
                    name="interface-automation",
                    description="通过 CLI 或 API 自动化执行并输出结果。",
                    category="用户体验",
                )
            ],
        ),
    ]


__all__ = [
    "TaskDefinition",
    "PracticalObjective",
    "ObjectivePlanner",
    "default_objectives",
]
