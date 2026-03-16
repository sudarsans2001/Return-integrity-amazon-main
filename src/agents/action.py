"""Action Agent: proposes actions and drafts customer/internal messaging."""

from __future__ import annotations

from typing import Any

from ..models import ActionResult, ActionType, PolicyResult, ReturnRequest, RiskLevel
from .base import BaseAgent


class ActionAgent(BaseAgent):
    name = "ActionAgent"

    SYSTEM_PROMPT = (
        "You are an action recommendation agent for Amazon return processing. "
        "Based on the risk assessment, recommend an appropriate action and draft "
        "a customer-facing message. The message must be professional, empathetic, "
        "and transparent. For denials, always include appeal instructions. "
        "Output JSON: action (auto_approve/enhanced_verification/manual_review/"
        "deny_with_rationale), customer_message (string), internal_notes (string), "
        "escalation_required (bool), escalation_reason (string or null)."
    )

    ACTION_MAP = {
        RiskLevel.LOW: ActionType.AUTO_APPROVE,
        RiskLevel.MEDIUM: ActionType.ENHANCED_VERIFICATION,
        RiskLevel.HIGH: ActionType.MANUAL_REVIEW,
        RiskLevel.CRITICAL: ActionType.DENY_WITH_RATIONALE,
    }

    def execute(
        self,
        *,
        request: ReturnRequest,
        policy: PolicyResult,
        **_: Any,
    ) -> ActionResult:
        base_action = self.ACTION_MAP[policy.risk_level]

        # Override: never auto-deny items above max_auto_deny_value
        if (
            base_action == ActionType.DENY_WITH_RATIONALE
            and request.item.price > 200
        ):
            base_action = ActionType.MANUAL_REVIEW

        escalation_required = request.item.price > 500
        escalation_reason = None
        if request.item.price > 2000:
            escalation_reason = (
                f"High-value item (${request.item.price:.2f}) requires supervisor review"
            )
        elif escalation_required:
            escalation_reason = (
                f"Item value (${request.item.price:.2f}) exceeds escalation threshold"
            )

        # Nova-generated messaging
        customer_msg, internal_notes = self._generate_messaging(
            request, policy, base_action
        )

        # Generate voice escalation script (Nova Sonic integration concept)
        voice_script = None
        if escalation_required:
            case_summary = (
                f"Item: {request.item.title} (${request.item.price:.2f})\n"
                f"Return reason: {request.return_reason}\n"
                f"Risk level: {policy.risk_level.value}\n"
                f"Fraud type: {policy.most_likely_fraud_type.value}\n"
                f"Confidence: {policy.confidence:.0%}\n"
                f"Escalation reason: {escalation_reason}"
            )
            voice_script = self.nova.generate_voice_escalation_script(case_summary)

        return ActionResult(
            return_id=request.return_id,
            recommended_action=base_action,
            customer_message=customer_msg,
            internal_notes=internal_notes,
            escalation_required=escalation_required,
            escalation_reason=escalation_reason,
            voice_escalation_script=voice_script,
        )

    def _generate_messaging(
        self,
        request: ReturnRequest,
        policy: PolicyResult,
        action: ActionType,
    ) -> tuple[str, str]:
        prompt = (
            f"Generate messaging for return action '{action.value}'.\n"
            f"Item: {request.item.title}\n"
            f"Risk level: {policy.risk_level.value}\n"
            f"Fraud type: {policy.most_likely_fraud_type.value}\n"
            f"Confidence: {policy.confidence:.0%}\n"
            f"Reasoning: {policy.policy_reasoning}\n"
        )
        try:
            result = self.nova.reason_json(self.SYSTEM_PROMPT, prompt)
            return (
                result.get("customer_message", ""),
                result.get("internal_notes", ""),
            )
        except Exception:
            return self._template_messaging(request, policy, action)

    def _template_messaging(
        self,
        request: ReturnRequest,
        policy: PolicyResult,
        action: ActionType,
    ) -> tuple[str, str]:
        item = request.item.title

        if action == ActionType.AUTO_APPROVE:
            customer = (
                f"Your return for '{item}' has been approved. "
                "Your refund will be processed within 3-5 business days."
            )
            internal = (
                f"Auto-approved: risk score {policy.risk_score:.2f}, "
                f"no significant fraud indicators."
            )

        elif action == ActionType.ENHANCED_VERIFICATION:
            customer = (
                f"Thank you for your return request for '{item}'. "
                "To process your return, we need additional verification. "
                "Please provide photos of the item and packaging, "
                "including any serial numbers visible on the product."
            )
            internal = (
                f"Enhanced verification triggered: risk {policy.risk_score:.2f}, "
                f"likely {policy.most_likely_fraud_type.value}. "
                f"Factors: {', '.join(policy.contributing_factors)}."
            )

        elif action == ActionType.MANUAL_REVIEW:
            customer = (
                f"Your return request for '{item}' is being reviewed by our team. "
                "We aim to complete the review within 24-48 hours. "
                "You will receive an update via email."
            )
            internal = (
                f"Manual review required: risk {policy.risk_score:.2f}, "
                f"classified as {policy.most_likely_fraud_type.value} "
                f"(confidence {policy.confidence:.0%}). "
                f"Reasoning: {policy.policy_reasoning}"
            )

        else:
            customer = (
                f"After careful review, we are unable to process your return for '{item}' "
                "at this time. Our records indicate discrepancies with this return. "
                "If you believe this is an error, you may file an appeal through "
                "your account's Order History page or contact Customer Service."
            )
            internal = (
                f"Return denied: risk {policy.risk_score:.2f}, "
                f"fraud type {policy.most_likely_fraud_type.value} "
                f"(confidence {policy.confidence:.0%}). "
                f"Evidence: {policy.policy_reasoning}. "
                f"Factors: {', '.join(policy.contributing_factors)}."
            )

        return customer, internal
