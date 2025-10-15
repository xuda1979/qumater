"""Light-weight task registry powering the integrated platform."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass
class PlatformTask:
    """Encapsulate a runnable platform level workflow."""

    name: str
    description: str
    handler: Callable[..., Any]

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the task using the stored handler."""

        return self.handler(*args, **kwargs)


class TaskRegistry:
    """Register and execute reusable platform tasks."""

    def __init__(self) -> None:
        self._tasks: Dict[str, PlatformTask] = {}

    def register(self, task: PlatformTask, *, overwrite: bool = False) -> None:
        """Register *task* optionally overwriting existing entries."""

        if not overwrite and task.name in self._tasks:
            raise ValueError(f"Task '{task.name}' already registered")
        self._tasks[task.name] = task

    def get(self, name: str) -> PlatformTask:
        try:
            return self._tasks[name]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise KeyError(f"Task '{name}' not found") from exc

    def run(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Execute a registered task by name."""

        task = self.get(name)
        return task.run(*args, **kwargs)

    @property
    def tasks(self) -> Dict[str, PlatformTask]:
        """Expose the registered tasks for inspection and documentation."""

        return dict(self._tasks)
