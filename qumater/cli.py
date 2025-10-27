"""Command line interface for running QuMater quantum workflows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional

from qumater.core import WorkflowConfig
from qumater.qsim import OptimizationHistory
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


def _format_report_text(report, summary_text: str) -> str:
    lines = [
        f"材料: {report.material.name}",
        f"算法: {report.algorithm_name}",
    ]
    if report.final_energy is not None:
        status = "收敛" if report.converged else "未完全收敛"
        lines.append(f"最终能量: {report.final_energy:.6f} ({status})")
    lines.append(summary_text)
    return "\n".join(lines)


def _format_report_markdown(report) -> str:
    lines = [
        "# QuMater 工作流报告",
        "",
        f"- **材料**: {report.material.name}",
        f"- **算法**: {report.algorithm_name}",
    ]
    if report.final_energy is not None:
        status = "收敛" if report.converged else "未完全收敛"
        symbol = "✅" if report.converged else "⚠️"
        lines.append(f"- **最终能量**: `{report.final_energy:.6f}` ({symbol} {status})")
    lines.append("")
    lines.append("## 目标完成进度")
    for name, stats in report.objective_summary.items():
        completed = int(stats["completed"])
        total = int(stats["total"])
        percentage = stats["progress"] * 100.0 if total else 100.0
        lines.append(f"- **{name}**: {completed}/{total} ({percentage:.1f}%)")
    lines.append("")
    lines.append("## 元数据")
    for key, value in report.metadata.items():
        if isinstance(value, (dict, list, tuple)):
            formatted = json.dumps(value, ensure_ascii=False)
        else:
            formatted = value
        lines.append(f"- **{key}**: {formatted}")

    result = report.algorithm_result
    if isinstance(result, OptimizationHistory):
        lines.append("")
        lines.append("## 优化轨迹")
        lines.append(f"- 迭代次数: {len(result.energies)}")
        if result.energies:
            lines.append(f"- 最终能量: `{result.energies[-1]:.6f}`")
        lines.append(f"- 收敛: {'是' if result.converged else '否'}")
    elif result is not None:
        lines.append("")
        lines.append("## 算法结果")
        lines.append("```")
        lines.append(repr(result))
        lines.append("```")

    return "\n".join(lines)


def _format_dry_run(plan: list[str], summary: Mapping[str, Mapping[str, float]], fmt: str) -> str:
    if fmt == "json":
        payload = {"plan": plan, "objective_summary": summary}
        return json.dumps(payload, ensure_ascii=False, indent=2)
    if fmt == "markdown":
        lines = ["# QuMater 规划预览", "", "## 任务序列"]
        for item in plan:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("## 目标完成进度")
        for name, stats in summary.items():
            completed = int(stats["completed"])
            total = int(stats["total"])
            percentage = stats["progress"] * 100.0 if total else 100.0
            lines.append(f"- **{name}**: {completed}/{total} ({percentage:.1f}%)")
        return "\n".join(lines)

    lines = ["规划任务序列:"]
    for item in plan:
        lines.append(f"- {item}")
    lines.append(_format_objective_summary(summary))
    return "\n".join(lines)


def _emit_output(text: str, destination: Optional[Path]) -> None:
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not text.endswith("\n"):
            text_to_write = text + "\n"
        else:
            text_to_write = text
        destination.write_text(text_to_write, encoding="utf-8")
        print(f"报告已写入: {destination}")
    else:
        print(text)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="运行 QuMater 工作流")
    parser.add_argument("config", help="JSON 或 TOML 配置文件路径")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅输出任务规划信息而不执行工作流",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json", "markdown"),
        default="text",
        help="控制输出格式，默认为面向终端的文本",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="若提供，则将结果写入目标文件而非直接打印",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    raw_config = _load_config(config_path)
    workflow_config = WorkflowConfig.from_dict(raw_config)

    workflow = QuantumWorkflow(workflow_config)
    if args.dry_run:
        plan = list(workflow.plan())
        summary = workflow.planner.summary()
        formatted = _format_dry_run(plan, summary, args.format)
        output_path = Path(args.output) if args.output else None
        _emit_output(formatted, output_path)
        return 0

    report = workflow.execute()
    workflow.planner.mark_completed("interface-automation")
    final_summary = workflow.planner.summary()
    report.objective_summary = final_summary
    summary_text = _format_objective_summary(final_summary)
    output_path = Path(args.output) if args.output else None

    if args.format == "json":
        payload = report.to_dict()
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    elif args.format == "markdown":
        text = _format_report_markdown(report)
    else:
        text = _format_report_text(report, summary_text)

    _emit_output(text, output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
