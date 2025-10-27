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


def test_cli_supports_json_output(tmp_path, capsys):
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

    exit_code = run_cli([str(config_path), "--format", "json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["material"]["name"] == "LiH minimal basis"
    assert payload["algorithm_name"] == "low_depth_vqe"
    assert payload["objective_summary"]


def test_cli_writes_markdown_report(tmp_path, capsys):
    config_path = tmp_path / "config.json"
    output_path = tmp_path / "report.md"
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

    exit_code = run_cli([
        str(config_path),
        "--format",
        "markdown",
        "--output",
        str(output_path),
    ])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "报告已写入" in captured.out
    report_text = output_path.read_text(encoding="utf-8")
    assert "# QuMater 工作流报告" in report_text
