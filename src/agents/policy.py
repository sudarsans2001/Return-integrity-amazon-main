"""Policy Agent: synthesizes evidence and behavior into risk assessment and fraud classification."""

from __future__ import annotations

import json
from typing import Any

from ..models import (
    BehaviorResult,
    EvidenceResult,
    FraudType,
    PolicyResult,
    ReturnRequest,
    RiskLevel,
)
from .base import BaseAgent


class PolicyAgent(BaseAgent):
    name = "PolicyAgent"

    SYSTEM_PROMPT = (
        "You are a policy decision agent for Amazon return fraud detection. "
        "Given evidence signals and behavioral analysis, determine the overall "
        "risk score, classify the most likely fraud type, and provide transparent reasoning. "
        "You must balance fraud prevention with customer experience - false positives "
        "damage trust. Output JSON: risk_score (0-1), risk_level (low/medium/high/critical), "
        "fraud_type (EMPTY_BOX/COUNTERFEIT_SWAP/USED_AS_NEW/SERIAL_MISMATCH/"
        "POLICY_GAMING/MULTI_ACCOUNT/INR_FRAUD/LEGITIMATE), confidence (0-1), "
        "fraud_type_probabilities (dict), reasoning (string), contributing_factors (list)."
    )

    RISK_BOUNDARIES = [
        (0.85, RiskLevel.CRITICAL),
        (0.60, RiskLevel.HIGH),
        (0.30, RiskLevel.MEDIUM),
        (0.00, RiskLevel.LOW),
    ]

    def execute(
        self,
        *,
        request: ReturnRequest,
        evidence: EvidenceResult,
        behavior: BehaviorResult,
        **_: Any,
    ) -> PolicyResult:
        # Build composite risk from evidence and behavior
        evidence_risk = self._score_evidence(evidence)
        behavior_risk = behavior.account_risk_score

        # 60/40 weighting: evidence matters more than history
        composite = 0.60 * evidence_risk + 0.40 * behavior_risk
        composite = round(min(composite, 1.0), 3)

        risk_level = self._classify_risk(composite)

        # Nova-enhanced fraud classification
        fraud_probs, fraud_type, confidence, reasoning, factors = (
            self._nova_classify(request, evidence, behavior, composite)
        )

        return PolicyResult(
            return_id=request.return_id,
            risk_score=composite,
            risk_level=risk_level,
            fraud_type_probabilities=fraud_probs,
            most_likely_fraud_type=fraud_type,
            confidence=confidence,
            policy_reasoning=reasoning,
            contributing_factors=factors,
        )

    def _score_evidence(self, evidence: EvidenceResult) -> float:
        scores: list[float] = []

        if evidence.weight_anomaly is True:
            weight_signals = [
                s for s in evidence.signals if "WEIGHT" in s.signal_type
            ]
            scores.append(max((s.confidence for s in weight_signals), default=0.7))

        if evidence.serial_match is False:
            scores.append(0.90)

        if evidence.packaging_integrity is not None:
            scores.append(1.0 - evidence.packaging_integrity)

        if evidence.image_similarity_score is not None:
            if evidence.image_similarity_score < 0.70:
                scores.append(1.0 - evidence.image_similarity_score)

        for sig in evidence.signals:
            if sig.signal_type not in ("WEIGHT_WARNING", "WEIGHT_CRITICAL"):
                scores.append(sig.confidence)

        if not scores:
            return 0.10
        return round(sum(scores) / len(scores), 3)

    def _classify_risk(self, score: float) -> RiskLevel:
        for threshold, level in self.RISK_BOUNDARIES:
            if score >= threshold:
                return level
        return RiskLevel.LOW

    def _nova_classify(
        self,
        request: ReturnRequest,
        evidence: EvidenceResult,
        behavior: BehaviorResult,
        composite_risk: float,
    ) -> tuple[dict[str, float], FraudType, float, str, list[str]]:
        prompt = self._build_classification_prompt(
            request, evidence, behavior, composite_risk
        )
        try:
            result = self.nova.reason_json(self.SYSTEM_PROMPT, prompt)
            fraud_probs = result.get("fraud_type_probabilities", {})
            fraud_type_str = result.get("fraud_type", "LEGITIMATE")
            fraud_type = FraudType(fraud_type_str)
            confidence = result.get("confidence", 0.7)
            reasoning = result.get("reasoning", "")
            factors = result.get("contributing_factors", [])
            return fraud_probs, fraud_type, confidence, reasoning, factors
        except Exception:
            return self._rule_based_classify(evidence, behavior, composite_risk)

    def _rule_based_classify(
        self,
        evidence: EvidenceResult,
        behavior: BehaviorResult,
        composite: float,
    ) -> tuple[dict[str, float], FraudType, float, str, list[str]]:
        factors: list[str] = []
        probs: dict[str, float] = {t.value: 0.0 for t in FraudType}

        if evidence.weight_anomaly:
            probs["EMPTY_BOX"] += 0.4
            factors.append("weight anomaly detected")
        if evidence.serial_match is False:
            probs["SERIAL_MISMATCH"] += 0.5
            probs["COUNTERFEIT_SWAP"] += 0.3
            factors.append("serial number mismatch")
        if any("INR" in s.signal_type for s in evidence.signals):
            probs["INR_FRAUD"] += 0.5
            factors.append("item not received claim with delivery confirmation")
        if behavior.return_velocity_flag:
            probs["POLICY_GAMING"] += 0.3
            factors.append("high return velocity")
        if behavior.address_cluster_flag:
            probs["MULTI_ACCOUNT"] += 0.3
            factors.append("address cluster detected")
        if behavior.value_pattern_flag:
            probs["COUNTERFEIT_SWAP"] += 0.2
            factors.append("high-value targeting pattern")

        if composite < 0.3:
            probs["LEGITIMATE"] = max(probs.values()) + 0.2

        # Normalize
        total = sum(probs.values()) or 1.0
        probs = {k: round(v / total, 3) for k, v in probs.items()}
        best = max(probs, key=lambda k: probs[k])
        fraud_type = FraudType(best)
        confidence = probs[best]
        reasoning = f"Rule-based classification: {', '.join(factors) or 'no strong signals'}."

        return probs, fraud_type, confidence, reasoning, factors

    def _build_classification_prompt(
        self,
        request: ReturnRequest,
        evidence: EvidenceResult,
        behavior: BehaviorResult,
        composite_risk: float,
    ) -> str:
        return (
            f"Return case analysis:\n"
            f"Item: {request.item.title} (${request.item.price})\n"
            f"Return reason: {request.return_reason}\n"
            f"Composite risk score: {composite_risk}\n\n"
            f"Evidence summary: {evidence.evidence_summary}\n"
            f"Weight anomaly: {evidence.weight_anomaly}\n"
            f"Serial match: {evidence.serial_match}\n"
            f"Image similarity: {evidence.image_similarity_score}\n"
            f"Evidence signals: {len(evidence.signals)}\n\n"
            f"Behavior summary: {behavior.behavior_summary}\n"
            f"Account risk: {behavior.account_risk_score}\n"
            f"Velocity flag: {behavior.return_velocity_flag}\n"
            f"Address cluster: {behavior.address_cluster_flag}\n"
            f"Value pattern: {behavior.value_pattern_flag}\n\n"
            "Classify this return and explain your reasoning."
        )
