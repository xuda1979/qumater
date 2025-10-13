import pytest

from qumater.qsim import (
    HardwareAgnosticAnsatz,
    LowDepthVQE,
    PauliHamiltonian,
    PauliTerm,
    get_algorithm_registry,
)
from qumater.qsim.modules import (
    AlgorithmModule,
    AlgorithmRegistry,
    load_entry_point_modules,
)


class DummyAlgorithm:
    def __init__(self, scale: float) -> None:
        self.scale = scale

    def run(self, value: float) -> float:
        return self.scale * value


def test_registry_registers_and_creates_instances():
    registry = AlgorithmRegistry()
    module = AlgorithmModule(
        name="dummy",
        summary="Simple scaling algorithm.",
        factory=lambda scale: DummyAlgorithm(scale),
        keywords=("test",),
    )
    registry.register(module)

    created = registry.create("dummy", scale=3.0)
    assert isinstance(created, DummyAlgorithm)
    assert created.run(2.0) == pytest.approx(6.0)


def test_algorithm_decorator_registers_factory():
    registry = AlgorithmRegistry()

    @registry.algorithm(name="decorated", summary="Decorator registration")
    def factory(scale: float) -> DummyAlgorithm:
        return DummyAlgorithm(scale)

    assert "decorated" in registry
    assert registry.create("decorated", scale=5.0).run(1.0) == pytest.approx(5.0)


def test_register_raises_when_name_conflicts():
    registry = AlgorithmRegistry()
    module = AlgorithmModule("conflict", "", lambda: DummyAlgorithm(1.0))
    registry.register(module)

    with pytest.raises(ValueError):
        registry.register(module)


def test_entry_point_loader_accepts_module_objects():
    registry = AlgorithmRegistry()
    module = AlgorithmModule("ep", "entry point", lambda: DummyAlgorithm(4.0))

    class FakeEntryPoint:
        def __init__(self, value):
            self.value = value

        def load(self):
            return self.value

    class FakeEntryPoints(list):
        def select(self, *, group):
            return self

    def fake_entry_points():
        return FakeEntryPoints([FakeEntryPoint(module)])

    load_entry_point_modules(registry=registry, entry_points_fn=fake_entry_points)
    assert registry.create("ep").run(2.0) == pytest.approx(8.0)


def test_entry_point_loader_accepts_callables():
    registry = AlgorithmRegistry()

    def provider(target_registry: AlgorithmRegistry) -> None:
        target_registry.register(
            AlgorithmModule("callable", "callable provider", lambda: DummyAlgorithm(2.5))
        )

    class FakeEntryPoint:
        def load(self):
            return provider

    class FakeEntryPoints(list):
        def select(self, *, group):
            return self

    def fake_entry_points():
        return FakeEntryPoints([FakeEntryPoint()])

    load_entry_point_modules(registry=registry, entry_points_fn=fake_entry_points)
    assert registry.create("callable").run(3.0) == pytest.approx(7.5)


def test_global_registry_exposes_built_in_low_depth_vqe():
    registry = get_algorithm_registry()
    module = registry.get("low_depth_vqe")

    ansatz = HardwareAgnosticAnsatz(num_qubits=1, layers=1)
    hamiltonian = PauliHamiltonian([PauliTerm(1.0, "Z")])
    optimiser = module.create(hamiltonian=hamiltonian, ansatz=ansatz)

    assert isinstance(optimiser, LowDepthVQE)
    history = optimiser.run()
    assert history.energies[-1] < history.energies[0]
