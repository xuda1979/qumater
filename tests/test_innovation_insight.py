from qumater.core import (
    AlgorithmConfig,
    AnsatzConfig,
    HamiltonianConfig,
    MaterialSelection,
    WorkflowConfig,
)
from qumater.materials import QuantumMaterialDatabase
from qumater.qsim import HardwareAgnosticAnsatz, PauliHamiltonian, PauliTerm
from qumater.workflows import InnovationInsightEngine, QuantumWorkflow


def test_innovation_engine_scores_are_bounded():
    config = WorkflowConfig(
        material=MaterialSelection(name="LiH minimal basis"),
        hamiltonian=HamiltonianConfig(terms=((1.0, "Z"),)),
        ansatz=AnsatzConfig(num_qubits=1, layers=1),
        algorithm=AlgorithmConfig(
            name="low_depth_vqe",
            options={"learning_rate": 0.2, "max_iterations": 10},
        ),
    )
    workflow = QuantumWorkflow(config)
    report = workflow.execute()
    insight = report.innovation_insight
    assert insight is not None
    assert 0.0 <= insight.differentiation_score <= 1.0
    assert 0.0 <= insight.risk_reduction_score <= 1.0
    assert insight.maturity_level in {"enterprise-ready", "pilot-ready", "exploratory"}
    assert insight.contributing_metrics["hamiltonian_complexity"] >= 0.0


def test_innovation_engine_handles_non_history_results():
    engine = InnovationInsightEngine()
    database = QuantumMaterialDatabase.demo()
    material = database.get("LiH minimal basis")
    hamiltonian = PauliHamiltonian([PauliTerm(1.0, "Z")])
    ansatz = HardwareAgnosticAnsatz(num_qubits=1, layers=1)
    insight = engine.evaluate(
        material=material,
        hamiltonian=hamiltonian,
        ansatz=ansatz,
        algorithm_name="static",
        algorithm_result={"result": 1},
        objective_summary={"dummy": {"completed": 1.0, "total": 1.0, "progress": 1.0}},
    )
    assert 0.0 <= insight.differentiation_score <= 1.0
    assert insight.recommendations
