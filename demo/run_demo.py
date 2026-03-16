"""Interactive demo runner for live hackathon presentation."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from demo.scenarios import ALL_SCENARIOS
from src.models import FraudType, RiskLevel
from src.nova_client import NovaClient
from src.orchestrator import ReturnIntegrityPipeline

console = Console()

RISK_COLORS = {
    RiskLevel.LOW: "green",
    RiskLevel.MEDIUM: "yellow",
    RiskLevel.HIGH: "red",
    RiskLevel.CRITICAL: "bold red",
}


def display_result(result, ground_truth: FraudType, scenario_name: str) -> None:
    req = result.intake.normalized_request
    pol = result.policy
    act = result.action

    is_correct = (
        (pol.most_likely_fraud_type == FraudType.LEGITIMATE)
        == (ground_truth == FraudType.LEGITIMATE)
    )
    verdict_str = "[green]CORRECT[/green]" if is_correct else "[red]INCORRECT[/red]"

    console.print()
    console.print(Panel(
        f"[bold]{scenario_name}[/bold]\n"
        f"Item: {req.item.title} | ${req.item.price:.2f}\n"
        f"Return Reason: {req.return_reason}",
        title="Return Case",
        border_style="blue",
    ))

    # Risk assessment table
    risk_table = Table(title="Risk Assessment")
    risk_table.add_column("Metric", style="cyan")
    risk_table.add_column("Value", justify="right")
    risk_color = RISK_COLORS.get(pol.risk_level, "white")
    risk_table.add_row("Risk Score", f"[{risk_color}]{pol.risk_score:.3f}[/{risk_color}]")
    risk_table.add_row("Risk Level", f"[{risk_color}]{pol.risk_level.value.upper()}[/{risk_color}]")
    risk_table.add_row("Fraud Type", pol.most_likely_fraud_type.value)
    risk_table.add_row("Confidence", f"{pol.confidence:.0%}")
    risk_table.add_row("Ground Truth", ground_truth.value)
    risk_table.add_row("Prediction", verdict_str)
    console.print(risk_table)

    # Flags
    if result.intake.initial_flags:
        console.print(Panel(
            "\n".join(f"  - {f}" for f in result.intake.initial_flags),
            title="Initial Flags",
            border_style="yellow",
        ))

    # Evidence
    console.print(Panel(
        result.evidence.evidence_summary or "No evidence anomalies.",
        title="Evidence Analysis",
        border_style="magenta",
    ))

    # Behavior
    console.print(Panel(
        result.behavior.behavior_summary,
        title="Behavioral Analysis",
        border_style="cyan",
    ))

    # Action
    console.print(Panel(
        f"Action: [bold]{act.recommended_action.value}[/bold]\n\n"
        f"Customer Message:\n{act.customer_message}\n\n"
        f"Internal Notes:\n{act.internal_notes}",
        title="Recommended Action",
        border_style="green" if act.recommended_action.value == "auto_approve" else "red",
    ))

    # Latency
    console.print(
        f"  Processing time: [bold]{result.total_processing_time_ms:.0f} ms[/bold]"
    )
    console.print()


def main() -> None:
    console.print(Panel(
        "[bold blue]ReturnShield AI[/bold blue]\n"
        "Multi-Agent Fraud Detection powered by Amazon Nova",
        title="DEMO",
        border_style="bold blue",
    ))

    nova = NovaClient()
    pipeline = ReturnIntegrityPipeline(nova)

    for name, scenario_fn in ALL_SCENARIOS.items():
        request, ground_truth = scenario_fn()
        console.print(f"\n[bold]Running scenario: {name}[/bold]")
        console.print("-" * 50)
        result = pipeline.process(request)
        display_result(result, ground_truth, name)
        console.print("=" * 60)


if __name__ == "__main__":
    main()
