"""Main entry point for ReturnShield AI."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from .metrics import compute_metrics, format_metrics_report
from .models import (
    AccountProfile,
    FraudType,
    OrderItem,
    ReturnRequest,
    ShipmentEvent,
)
from .nova_client import NovaClient
from .orchestrator import ReturnIntegrityPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-18s] %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_synthetic_cases() -> tuple[list[ReturnRequest], dict[str, FraudType]]:
    """Load synthetic test cases from JSON."""
    cases_path = DATA_DIR / "synthetic_cases.json"
    with open(cases_path, encoding="utf-8") as f:
        raw_cases = json.load(f)

    requests: list[ReturnRequest] = []
    ground_truth: dict[str, FraudType] = {}

    for case in raw_cases:
        account = AccountProfile(**case["account"])
        item = OrderItem(**case["item"])
        events = [ShipmentEvent(**e) for e in case.get("shipment_events", [])]

        req = ReturnRequest(
            order_id=case["order_id"],
            account=account,
            item=item,
            return_reason=case["return_reason"],
            return_submitted_at=datetime.fromisoformat(
                case["return_submitted_at"]
            ),
            order_delivered_at=datetime.fromisoformat(
                case["order_delivered_at"]
            ),
            shipment_events=events,
            return_images=case.get("return_images", []),
            customer_message=case.get("customer_message"),
        )
        requests.append(req)
        ground_truth[req.return_id] = FraudType(case["ground_truth"])

    return requests, ground_truth


def main() -> None:
    logger.info("=" * 60)
    logger.info("RETURNSHIELD AI - Powered by Amazon Nova")
    logger.info("=" * 60)

    nova = NovaClient()
    pipeline = ReturnIntegrityPipeline(nova)

    requests, ground_truth = load_synthetic_cases()
    logger.info("Loaded %d synthetic test cases", len(requests))

    results = pipeline.process_batch(requests)

    # Compute and display metrics
    summary = compute_metrics(results, ground_truth)
    report = format_metrics_report(summary)
    print("\n" + report)

    # Print individual case summaries
    print("\n" + "=" * 60)
    print("INDIVIDUAL CASE RESULTS")
    print("=" * 60)
    for r in results:
        gt = ground_truth.get(r.return_id, FraudType.LEGITIMATE)
        predicted = r.policy.most_likely_fraud_type
        match = "CORRECT" if (
            (predicted == FraudType.LEGITIMATE) == (gt == FraudType.LEGITIMATE)
        ) else "WRONG"
        print(
            f"\n[{match}] Return {r.return_id}: "
            f"{r.intake.normalized_request.item.title}"
        )
        print(f"  Ground truth: {gt.value}")
        print(f"  Predicted:    {predicted.value} (confidence {r.policy.confidence:.0%})")
        print(f"  Risk:         {r.policy.risk_score:.3f} ({r.policy.risk_level.value})")
        print(f"  Action:       {r.action.recommended_action.value}")
        print(f"  Latency:      {r.total_processing_time_ms:.0f} ms")
        if r.intake.initial_flags:
            print(f"  Flags:        {', '.join(r.intake.initial_flags)}")

    # Save results to JSON
    output_path = DATA_DIR / "results.json"
    output = {
        "results": [r.model_dump() for r in results],
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
