"""Infrastructure for packaging quantum algorithms as reusable modules.

This module introduces a light-weight registry that keeps track of available
quantum algorithm factories.  It is designed for two complementary workflows:

* **Local prototyping** – developers can register ad-hoc factories while
  iterating on new ideas.
* **Distribution as Python packages** – third parties can expose their
  algorithms via ``entry_points`` so that QuMater can discover and load them at
  runtime.

The central abstraction is :class:`AlgorithmModule`, a dataclass describing a
factory function together with short metadata.  The global registry returned by
 :func:`get_algorithm_registry` is populated with QuMater's built-in algorithms
and any entry points declared under the ``qumater.qsim.algorithms`` group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, List, MutableMapping, Optional, Tuple

from importlib import metadata as _importlib_metadata

AlgorithmFactory = Callable[..., Any]


@dataclass(frozen=True)
class AlgorithmModule:
    """Metadata describing a quantum algorithm factory."""

    name: str
    summary: str
    factory: AlgorithmFactory
    keywords: Tuple[str, ...] = ()
    package: Optional[str] = None

    def create(self, *args: Any, **kwargs: Any) -> Any:
        """Instantiate the underlying algorithm using the stored factory."""

        return self.factory(*args, **kwargs)


class AlgorithmRegistry:
    """Container mapping short names to :class:`AlgorithmModule` instances."""

    def __init__(self) -> None:
        self._modules: MutableMapping[str, AlgorithmModule] = {}

    @staticmethod
    def _key(name: str) -> str:
        return name.casefold()

    def register(self, module: AlgorithmModule, *, overwrite: bool = False) -> None:
        """Register an :class:`AlgorithmModule`.

        Parameters
        ----------
        module:
            The module definition to register.
        overwrite:
            When ``False`` (default) a :class:`ValueError` is raised if a module
            with the same name already exists.  Set to ``True`` to replace the
            existing entry.
        """

        key = self._key(module.name)
        if not overwrite and key in self._modules:
            raise ValueError(f"Algorithm module '{module.name}' is already registered")
        self._modules[key] = module

    def get(self, name: str) -> AlgorithmModule:
        """Return the module registered under *name*.

        Raises
        ------
        KeyError
            If the module is unknown to the registry.
        """

        return self._modules[self._key(name)]

    def create(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Instantiate an algorithm registered under *name*."""

        return self.get(name).create(*args, **kwargs)

    def available(self) -> List[AlgorithmModule]:
        """Return a list of registered modules ordered by their name."""

        return sorted(self._modules.values(), key=lambda module: module.name)

    def __contains__(self, name: object) -> bool:  # pragma: no cover - 简单逻辑
        if not isinstance(name, str):
            return False
        return self._key(name) in self._modules

    def __iter__(self) -> Iterator[AlgorithmModule]:  # pragma: no cover - 简单逻辑
        yield from self.available()

    # ------------------------------------------------------------------
    # 装饰器辅助工具
    # ------------------------------------------------------------------
    def algorithm(self, *, name: str, summary: str, keywords: Iterable[str] = ()) -> Callable[[AlgorithmFactory], AlgorithmFactory]:
        """Decorator to register *factory* functions as modules."""

        keywords_tuple = tuple(keywords)

        def decorator(factory: AlgorithmFactory) -> AlgorithmFactory:
            self.register(
                AlgorithmModule(
                    name=name,
                    summary=summary,
                    factory=factory,
                    keywords=keywords_tuple,
                )
            )
            return factory

        return decorator


ENTRY_POINT_GROUP = "qumater.qsim.algorithms"

_GLOBAL_REGISTRY = AlgorithmRegistry()
_ENTRY_POINTS_LOADED = False


def get_algorithm_registry() -> AlgorithmRegistry:
    """Return the global registry used by :mod:`qumater.qsim`."""

    _ensure_entry_points_loaded(_GLOBAL_REGISTRY)
    return _GLOBAL_REGISTRY


def register_algorithm_module(module: AlgorithmModule, *, overwrite: bool = False) -> None:
    """Register *module* with the global registry."""

    get_algorithm_registry().register(module, overwrite=overwrite)


def load_entry_point_modules(
    registry: Optional[AlgorithmRegistry] = None,
    *,
    entry_points_fn: Optional[Callable[[], Any]] = None,
    group: str = ENTRY_POINT_GROUP,
) -> AlgorithmRegistry:
    """Discover algorithm modules from ``importlib.metadata`` entry points."""

    if registry is None:
        registry = get_algorithm_registry()

    if entry_points_fn is None:
        entry_points_fn = _importlib_metadata.entry_points

    entry_points = entry_points_fn()
    if hasattr(entry_points, "select"):
        selected = entry_points.select(group=group)
    else:  # pragma: no cover - 兼容 Python 3.10 之前的版本
        selected = entry_points.get(group, [])

    for entry_point in selected:
        loaded = entry_point.load()
        if isinstance(loaded, AlgorithmModule):
            try:
                registry.register(loaded)
            except ValueError:
                continue
        elif callable(loaded):
            result = loaded(registry)
            if isinstance(result, AlgorithmModule):
                try:
                    registry.register(result)
                except ValueError:
                    continue
        else:
            raise TypeError(
                f"Entry point {entry_point!r} did not return an AlgorithmModule or a callable"
            )

    return registry


def _ensure_entry_points_loaded(registry: AlgorithmRegistry) -> None:
    global _ENTRY_POINTS_LOADED
    if not _ENTRY_POINTS_LOADED:
        load_entry_point_modules(registry=registry)
        _ENTRY_POINTS_LOADED = True


__all__ = [
    "AlgorithmModule",
    "AlgorithmRegistry",
    "ENTRY_POINT_GROUP",
    "get_algorithm_registry",
    "load_entry_point_modules",
    "register_algorithm_module",
]

