from qumater.core import AlgorithmConfig, AnsatzConfig, HamiltonianConfig, MaterialSelection, WorkflowConfig
from qumater.workflows import QuantumWorkflow


def test_quantum_workflow_executes_vqe():
    config = WorkflowConfig(
        material=MaterialSelection(name="LiH minimal basis"),
        hamiltonian=HamiltonianConfig(terms=((1.0, "Z"),)),
        ansatz=AnsatzConfig(num_qubits=1, layers=1),
        algorithm=AlgorithmConfig(
            name="low_depth_vqe",
            options={"learning_rate": 0.2, "max_iterations": 20},
        ),
        metadata={"experiment": "unit-test"},
    )

    workflow = QuantumWorkflow(config)
    report = workflow.execute()

    assert report.algorithm_name == "low_depth_vqe"
    assert report.final_energy is not None
    assert report.final_energy <= 0.0
    assert "material:LiH minimal basis" == report.steps[0]
    assert report.objective_summary["数据到模型的稳健流程"]["completed"] == 2.0
    assert report.metadata["config_metadata"]["experiment"] == "unit-test"

    payload = report.to_dict()
    assert payload["material"]["name"] == "LiH minimal basis"
    assert payload["algorithm_result"]["converged"] in {True, False}
    assert isinstance(payload["algorithm_result"]["energies"], list)
