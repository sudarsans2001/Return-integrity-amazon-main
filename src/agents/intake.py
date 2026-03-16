"""Intake Agent: normalizes and validates incoming return requests."""

from __future__ import annotations

from typing import Any

from ..models import IntakeResult, ReturnRequest
from .base import BaseAgent


class IntakeAgent(BaseAgent):
    name = "IntakeAgent"

    def execute(self, *, request: ReturnRequest, **_: Any) -> IntakeResult:
        flags: list[str] = []

        hours_since_delivery = (
            request.return_submitted_at - request.order_delivered_at
        ).total_seconds() / 3600

        if hours_since_delivery < 24:
            flags.append("RAPID_RETURN: returned within 24 hours of delivery")
        if request.item.price > 500:
            flags.append("HIGH_VALUE_ITEM: item price exceeds $500")
        if request.account.return_rate > 0.30:
            flags.append("HIGH_RETURN_RATE: account return rate above 30%")
        if request.account.previous_fraud_flags > 0:
            flags.append(
                f"PRIOR_FLAGS: {request.account.previous_fraud_flags} previous fraud flags"
            )
        if request.account.account_age_days < 30:
            flags.append("NEW_ACCOUNT: account less than 30 days old")
        if not request.item.serial_number and request.item.price > 100:
            flags.append("MISSING_SERIAL: high-value item without serial number")
        if not request.return_images:
            flags.append("NO_IMAGES: no return images provided")

        completeness_checks = [
            request.item.serial_number is not None,
            request.item.weight_grams is not None,
            len(request.shipment_events) > 0,
            len(request.return_images) > 0,
            request.customer_message is not None,
        ]
        completeness = sum(completeness_checks) / len(completeness_checks)

        return IntakeResult(
            return_id=request.return_id,
            normalized_request=request,
            initial_flags=flags,
            data_completeness_score=round(completeness, 2),
        )
