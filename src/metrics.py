"""KPI tracking and benchmarking for ReturnShield AI."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import ActionType, FraudType, PipelineResult, RiskLevel


@dataclass
class MetricsSummary:
    total_cases: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    total_processing_ms: float = 0.0
    action_distribution: dict[str, int] = field(default_factory=dict)
    risk_distribution: dict[str, int] = field(default_factory=dict)
    fraud_type_distribution: dict[str, int] = field(default_factory=dict)
    total_value_at_risk: float = 0.0
    value_protected: float = 0.0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def false_positive_rate(self) -> float:
        denom = self.false_positives + self.true_negatives
        return self.false_positives / denom if denom > 0 else 0.0

    @property
    def mean_latency_ms(self) -> float:
        return self.total_processing_ms / self.total_cases if self.total_cases > 0 else 0.0

    @property
    def loss_prevention_rate(self) -> float:
        return self.value_protected / self.total_value_at_risk if self.total_value_at_risk > 0 else 0.0


def compute_metrics(
    results: list[PipelineResult],
    ground_truth: dict[str, FraudType],
) -> MetricsSummary:
    """Compute all KPIs given pipeline results and ground truth labels."""
    summary = MetricsSummary()

    for r in results:
        summary.total_cases += 1
        summary.total_processing_ms += r.total_processing_time_ms

        # Action distribution
        action = r.action.recommended_action.value
        summary.action_distribution[action] = (
            summary.action_distribution.get(action, 0) + 1
        )

        # Risk distribution
        risk = r.policy.risk_level.value
        summary.risk_distribution[risk] = (
            summary.risk_distribution.get(risk, 0) + 1
        )

        # Fraud type distribution
        ftype = r.policy.most_likely_fraud_type.value
        summary.fraud_type_distribution[ftype] = (
            summary.fraud_type_distribution.get(ftype, 0) + 1
        )

        # Ground truth comparison
        gt = ground_truth.get(r.return_id, FraudType.LEGITIMATE)
        predicted_fraud = r.policy.most_likely_fraud_type != FraudType.LEGITIMATE
        actual_fraud = gt != FraudType.LEGITIMATE
        item_value = r.intake.normalized_request.item.price

        if actual_fraud:
            summary.total_value_at_risk += item_value

        if predicted_fraud and actual_fraud:
            summary.true_positives += 1
            summary.value_protected += item_value
        elif predicted_fraud and not actual_fraud:
            summary.false_positives += 1
        elif not predicted_fraud and actual_fraud:
            summary.false_negatives += 1
        else:
            summary.true_negatives += 1

    return summary


def format_metrics_report(summary: MetricsSummary) -> str:
    lines = [
        "=" * 60,
        "RETURNSHIELD AI - PERFORMANCE REPORT",
        "=" * 60,
        "",
        f"Total Cases Processed: {summary.total_cases}",
        "",
        "--- Detection Performance ---",
        f"  Precision:          {summary.precision:.1%}",
        f"  Recall:             {summary.recall:.1%}",
        f"  F1 Score:           {summary.f1_score:.1%}",
        f"  False Positive Rate: {summary.false_positive_rate:.1%}",
        "",
        "--- Operational Metrics ---",
        f"  Mean Latency:       {summary.mean_latency_ms:.1f} ms",
        f"  Loss Prevention:    {summary.loss_prevention_rate:.1%}",
        f"  Value at Risk:      ${summary.total_value_at_risk:,.2f}",
        f"  Value Protected:    ${summary.value_protected:,.2f}",
        "",
        "--- Action Distribution ---",
    ]
    for action, count in sorted(summary.action_distribution.items()):
        pct = count / summary.total_cases * 100 if summary.total_cases else 0
        lines.append(f"  {action:30s} {count:4d} ({pct:.1f}%)")

    lines.extend(["", "--- Risk Distribution ---"])
    for risk, count in sorted(summary.risk_distribution.items()):
        pct = count / summary.total_cases * 100 if summary.total_cases else 0
        lines.append(f"  {risk:30s} {count:4d} ({pct:.1f}%)")

    lines.extend(["", "--- Fraud Type Distribution ---"])
    for ftype, count in sorted(summary.fraud_type_distribution.items()):
        pct = count / summary.total_cases * 100 if summary.total_cases else 0
        lines.append(f"  {ftype:30s} {count:4d} ({pct:.1f}%)")

    lines.append("=" * 60)
    return "\n".join(lines)
