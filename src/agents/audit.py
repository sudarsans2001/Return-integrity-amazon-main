"""Audit Agent: generates immutable case summaries and human-readable explanations."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..models import (
    ActionResult,
    AuditRecord,
    BehaviorResult,
    EvidenceResult,
    PolicyResult,
    ReturnRequest,
    RiskLevel,
)
from .base import BaseAgent


class AuditAgent(BaseAgent):
    name = "AuditAgent"

    SYSTEM_PROMPT = (
        "You are an audit and compliance agent. Generate a clear, human-readable "
        "explanation of a return fraud decision that can be used for appeals, "
        "compliance reviews, and operational transparency. "
        "The explanation must be factual, cite specific evidence, and avoid "
        "discriminatory language. Output JSON: decision_explanation (string), "
        "appeal_eligible (bool)."
    )

    def execute(
        self,
        *,
        request: ReturnRequest,
        evidence: EvidenceResult,
        behavior: BehaviorResult,
        policy: PolicyResult,
        action: ActionResult,
        processing_time_ms: float = 0.0,
        **_: Any,
    ) -> AuditRecord:
        explanation = self._generate_explanation(
            request, evidence, behavior, policy, action
        )

        appeal_eligible = policy.risk_level != RiskLevel.LOW

        return AuditRecord(
            return_id=request.return_id,
            timestamp=datetime.utcnow(),
            risk_score=policy.risk_score,
            risk_level=policy.risk_level,
            fraud_type=policy.most_likely_fraud_type,
            action_taken=action.recommended_action,
            evidence_summary=evidence.evidence_summary,
            behavior_summary=behavior.behavior_summary,
            policy_reasoning=policy.policy_reasoning,
            customer_message=action.customer_message,
            decision_explanation=explanation,
            appeal_eligible=appeal_eligible,
            processing_time_ms=processing_time_ms,
        )

    def _generate_explanation(
        self,
        request: ReturnRequest,
        evidence: EvidenceResult,
        behavior: BehaviorResult,
        policy: PolicyResult,
        action: ActionResult,
    ) -> str:
        prompt = (
            f"Generate audit explanation for return {request.return_id}.\n"
            f"Item: {request.item.title} (${request.item.price})\n"
            f"Risk: {policy.risk_score:.2f} ({policy.risk_level.value})\n"
            f"Fraud type: {policy.most_likely_fraud_type.value}\n"
            f"Action: {action.recommended_action.value}\n"
            f"Evidence: {evidence.evidence_summary}\n"
            f"Behavior: {behavior.behavior_summary}\n"
            f"Policy reasoning: {policy.policy_reasoning}\n"
        )
        try:
            result = self.nova.reason_json(self.SYSTEM_PROMPT, prompt)
            return result.get("decision_explanation", "")
        except Exception:
            return self._template_explanation(
                request, evidence, behavior, policy, action
            )

    def _template_explanation(
        self,
        request: ReturnRequest,
        evidence: EvidenceResult,
        behavior: BehaviorResult,
        policy: PolicyResult,
        action: ActionResult,
    ) -> str:
        lines = [
            f"CASE SUMMARY: Return {request.return_id}",
            f"Item: {request.item.title} | ASIN: {request.item.asin} | Value: ${request.item.price:.2f}",
            f"Return Reason: {request.return_reason}",
            "",
            "RISK ASSESSMENT:",
            f"  Overall Risk Score: {policy.risk_score:.3f} ({policy.risk_level.value.upper()})",
            f"  Classification: {policy.most_likely_fraud_type.value} (confidence: {policy.confidence:.0%})",
            "",
            "EVIDENCE FINDINGS:",
            f"  {evidence.evidence_summary}",
        ]

        if evidence.weight_anomaly is not None:
            lines.append(f"  Weight anomaly: {'YES' if evidence.weight_anomaly else 'No'}")
        if evidence.serial_match is not None:
            lines.append(f"  Serial match: {'YES' if evidence.serial_match else 'MISMATCH'}")

        lines.extend([
            "",
            "BEHAVIORAL ANALYSIS:",
            f"  {behavior.behavior_summary}",
            "",
            "DECISION:",
            f"  Action: {action.recommended_action.value}",
            f"  Reasoning: {policy.policy_reasoning}",
        ])

        if policy.contributing_factors:
            lines.append(f"  Contributing factors: {', '.join(policy.contributing_factors)}")

        if action.escalation_required:
            lines.append(f"  ESCALATION: {action.escalation_reason}")

        lines.append(f"\nAppeal eligible: {'Yes' if policy.risk_level != RiskLevel.LOW else 'N/A'}")

        return "\n".join(lines)
