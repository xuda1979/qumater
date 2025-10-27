"""Strategic insight helpers that emphasise QuMater's differentiators."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

from qumater.materials import MaterialEntry
from qumater.qsim import HardwareAgnosticAnsatz, OptimizationHistory, PauliHamiltonian


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Return *value* clamped to ``[lower, upper]``."""

    return max(lower, min(upper, value))


def _scale_between(value: float, low: float, high: float) -> float:
    """Map *value* into ``[0, 1]`` using the provided range."""

    if math.isclose(high, low):
        return 1.0
    return _clamp((value - low) / (high - low))


@dataclass
class InnovationInsight:
    """Rich insight describing why a workflow run is strategically valuable."""

    differentiation_score: float
    risk_reduction_score: float
    maturity_level: str
    highlight: str
    contributing_metrics: Dict[str, float]
    recommendations: Sequence[str] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON serialisable representation of the insight."""

        return {
            "differentiation_score": float(self.differentiation_score),
            "risk_reduction_score": float(self.risk_reduction_score),
            "maturity_level": self.maturity_level,
            "highlight": self.highlight,
            "contributing_metrics": dict(self.contributing_metrics),
            "recommendations": list(self.recommendations),
        }


class InnovationInsightEngine:
    """Compute innovation-centric insights for workflow executions.

    The heuristics aggregate properties of the material catalogue entry, the
    Hamiltonian structure and the optimiser trace.  This surfaces why QuMater is
    valuable compared to generic quantum workflow runners: decision makers gain a
    ready-to-present narrative explaining the strategic advantages of each run.
    """

    def __init__(self, *, variance_reference: float = 1.5) -> None:
        self.variance_reference = float(variance_reference)

    def _extract_history(self, result: Any) -> Optional[OptimizationHistory]:
        if isinstance(result, OptimizationHistory):
            return result
        return None

    def evaluate(
        self,
        *,
        material: MaterialEntry,
        hamiltonian: PauliHamiltonian,
        ansatz: HardwareAgnosticAnsatz,
        algorithm_name: str,
        algorithm_result: Any,
        objective_summary: Mapping[str, Mapping[str, float]],
    ) -> InnovationInsight:
        history = self._extract_history(algorithm_result)
        metrics: Dict[str, float] = {}

        tag_depth = _scale_between(len(material.tags), 1.0, 6.0)
        parameter_richness = _scale_between(len(material.parameters), 1.0, 8.0)
        metrics["material_scope"] = round(0.6 * tag_depth + 0.4 * parameter_richness, 4)

        if objective_summary:
            objective_progress = [
                float(stats.get("progress", 0.0))
                for stats in objective_summary.values()
                if isinstance(stats, Mapping)
            ]
            progress_mean = sum(objective_progress) / len(objective_progress)
        else:
            progress_mean = 0.0
        metrics["objective_progress"] = round(_clamp(progress_mean), 4)

        term_count = len(hamiltonian.terms)
        group_count = max(len(hamiltonian.commuting_groups), 1)
        metrics["hamiltonian_complexity"] = round(_scale_between(term_count, 1.0, 24.0), 4)
        metrics["measurement_efficiency"] = round(
            _scale_between(group_count / max(term_count, 1), 0.2, 1.0), 4
        )

        parameter_count = ansatz.parameter_count
        expected_min = ansatz.num_qubits * 2
        metrics["ansatz_capacity"] = round(
            0.5
            + 0.5
            * _scale_between(parameter_count, expected_min, max(expected_min * 4, expected_min + 1)),
            4,
        )

        energy_gain_ratio = 0.0
        iteration_balance = 0.5
        convergence_factor = 0.5
        variance_score = 0.5

        if history is not None and history.energies:
            initial_energy = history.energies[0]
            final_energy = history.energies[-1]
            improvement = max(0.0, initial_energy - final_energy)
            baseline = max(abs(initial_energy), abs(final_energy), 1.0)
            energy_gain_ratio = _clamp(improvement / baseline)
            metrics["energy_improvement"] = round(energy_gain_ratio, 4)

            iteration_count = max(len(history.energies), 1)
            # Encourage purposeful iterations (6-24 iterations score highly)
            sweet_spot = 12.0
            iteration_balance = math.exp(-abs(iteration_count - sweet_spot) / sweet_spot)
            metrics["iteration_balance"] = round(iteration_balance, 4)

            convergence_factor = 1.0 if history.converged else 0.55
            metrics["convergence_factor"] = round(convergence_factor, 4)

            try:
                final_parameters = history.parameters[-1]
                final_state = ansatz.prepare_state(final_parameters)
                variance = hamiltonian.variance(final_state)
                variance_score = 1.0 - _scale_between(variance, 0.0, self.variance_reference)
            except Exception:  # pragma: no cover - defensive fallback
                variance_score = 0.5
            metrics["variance_resilience"] = round(_clamp(variance_score), 4)
        else:
            metrics.setdefault("energy_improvement", 0.0)
            metrics.setdefault("iteration_balance", iteration_balance)
            metrics.setdefault("convergence_factor", convergence_factor)
            metrics.setdefault("variance_resilience", variance_score)

        # Aggregate innovation and risk signals into high-level scores
        differentiation = (
            0.28 * metrics["material_scope"]
            + 0.22 * metrics["hamiltonian_complexity"]
            + 0.15 * metrics["measurement_efficiency"]
            + 0.18 * metrics["ansatz_capacity"]
            + 0.17 * metrics["energy_improvement"]
        )
        risk_reduction = (
            0.3 * metrics["variance_resilience"]
            + 0.27 * metrics["convergence_factor"]
            + 0.18 * metrics["measurement_efficiency"]
            + 0.15 * metrics["iteration_balance"]
            + 0.1 * metrics["objective_progress"]
        )
        differentiation = round(_clamp(differentiation), 4)
        risk_reduction = round(_clamp(risk_reduction), 4)

        composite = 0.6 * differentiation + 0.4 * risk_reduction
        if composite >= 0.8:
            maturity = "enterprise-ready"
        elif composite >= 0.6:
            maturity = "pilot-ready"
        else:
            maturity = "exploratory"

        progress_phrase: str
        if energy_gain_ratio >= 0.5:
            progress_phrase = "显著压低基态能量"
        elif energy_gain_ratio >= 0.2:
            progress_phrase = "稳定推动能量下降"
        else:
            progress_phrase = "构建面向工业指标的收敛轨迹"

        highlight = (
            f"{algorithm_name} 在 {material.name} 上{progress_phrase}，"
            f"利用 {group_count} 组对易测量涵盖 {term_count} 项哈密顿量，"
            f"体现出硬件无关线路的创新潜力。"
        )

        recommendations = []
        if energy_gain_ratio < 0.2:
            recommendations.append("尝试调整学习率或增加自然梯度步长以提升能量改进幅度")
        if metrics["variance_resilience"] < 0.6:
            recommendations.append("考虑增加 Ansatz 深度或引入误差缓解以降低能量方差")
        if metrics["material_scope"] < 0.5:
            recommendations.append("为材料补充更多标签或参数，以加强任务筛选的可解释性")
        if maturity == "exploratory":
            recommendations.append("结合工作流目标进度，规划后续验证实验以支撑业务落地")

        return InnovationInsight(
            differentiation_score=differentiation,
            risk_reduction_score=risk_reduction,
            maturity_level=maturity,
            highlight=highlight,
            contributing_metrics=metrics,
            recommendations=tuple(recommendations),
        )


__all__ = ["InnovationInsight", "InnovationInsightEngine"]
