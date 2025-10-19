import json

from qumater import cli as run_cli


def test_cli_runs_workflow(tmp_path, capsys):
    config_path = tmp_path / "config.json"
    config = {
        "material": {"name": "LiH minimal basis"},
        "hamiltonian": {"terms": [[1.0, "Z"]]},
        "ansatz": {"num_qubits": 1, "layers": 1},
        "algorithm": {
            "name": "low_depth_vqe",
            "options": {"learning_rate": 0.2, "max_iterations": 10},
        },
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    exit_code = run_cli([str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "材料: LiH minimal basis" in captured.out
    assert "实用目标完成进度" in captured.out
