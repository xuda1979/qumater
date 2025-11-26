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
            name="Robust Data-to-Model Pipeline",
            summary="Ensure a closed loop from material data and Hamiltonian modeling to circuit generation.",
            tasks=[
                TaskDefinition(
                    name="material-catalogue-selection",
                    description="Filter or retrieve target samples from the quantum material catalogue.",
                    category="Data Preparation",
                ),
                TaskDefinition(
                    name="hamiltonian-build",
                    description="Construct a directly simulatable Hamiltonian from configured Pauli terms.",
                    category="Model Construction",
                ),
            ],
        ),
        PracticalObjective(
            name="Evolvable Algorithms & Orchestration",
            summary="Build a maintainable, extensible, and debuggable algorithm execution stack.",
            tasks=[
                TaskDefinition(
                    name="ansatz-construction",
                    description="Create a hardware-agnostic variational circuit based on the configuration.",
                    category="Quantum Circuits",
                ),
                TaskDefinition(
                    name="algorithm-instantiation",
                    description="Instantiate the required quantum algorithm or workflow via the registry.",
                    category="Algorithm Management",
                ),
                TaskDefinition(
                    name="result-reporting",
                    description="Generate a structured summary containing key diagnostic metrics.",
                    category="Observability",
                ),
            ],
        ),
        PracticalObjective(
            name="User-Friendly Interface",
            summary="Provide a unified entry point for researchers and engineers.",
            tasks=[
                TaskDefinition(
                    name="interface-automation",
                    description="Automate execution and output results via CLI or API.",
                    category="User Experience",
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
