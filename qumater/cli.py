"""Command line interface for running QuMater quantum workflows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional

from qumater.core import WorkflowConfig
from qumater.workflows import QuantumWorkflow

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - for Python <3.11 environments
    tomllib = None  # type: ignore


def _load_config(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file '{path}' does not exist")
    suffix = path.suffix.lower()
    if suffix in {".json", ".js"}:
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".toml", ".tml"}:
        if tomllib is None:
            raise RuntimeError("TOML support requires Python 3.11 or newer")
        return tomllib.loads(path.read_text(encoding="utf-8"))
    raise ValueError("Configuration files must be JSON or TOML")


def _format_objective_summary(summary: Mapping[str, Mapping[str, float]]) -> str:
    lines = ["实用目标完成进度:"]
    for name, stats in summary.items():
        completed = int(stats["completed"])
        total = int(stats["total"])
        percentage = stats["progress"] * 100.0 if total else 100.0
        lines.append(f"- {name}: {completed}/{total} ({percentage:.1f}%)")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="运行 QuMater 工作流")
    parser.add_argument("config", help="JSON 或 TOML 配置文件路径")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅输出任务规划信息而不执行工作流",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    raw_config = _load_config(config_path)
    workflow_config = WorkflowConfig.from_dict(raw_config)

    workflow = QuantumWorkflow(workflow_config)
    if args.dry_run:
        plan = list(workflow.plan())
        print("规划任务序列:")
        for item in plan:
            print(f"- {item}")
        print(_format_objective_summary(workflow.planner.summary()))
        return 0

    report = workflow.execute()
    workflow.planner.mark_completed("interface-automation")
    summary_text = _format_objective_summary(workflow.planner.summary())

    print(f"材料: {report.material.name}")
    print(f"算法: {report.algorithm_name}")
    if report.final_energy is not None:
        status = "收敛" if report.converged else "未完全收敛"
        print(f"最终能量: {report.final_energy:.6f} ({status})")
    print(summary_text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
