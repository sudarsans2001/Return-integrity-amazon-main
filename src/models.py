"""Pydantic data models for the ReturnShield AI pipeline."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class FraudType(str, Enum):
    EMPTY_BOX = "EMPTY_BOX"
    COUNTERFEIT_SWAP = "COUNTERFEIT_SWAP"
    USED_AS_NEW = "USED_AS_NEW"
    SERIAL_MISMATCH = "SERIAL_MISMATCH"
    POLICY_GAMING = "POLICY_GAMING"
    MULTI_ACCOUNT = "MULTI_ACCOUNT"
    INR_FRAUD = "INR_FRAUD"
    LEGITIMATE = "LEGITIMATE"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionType(str, Enum):
    AUTO_APPROVE = "auto_approve"
    ENHANCED_VERIFICATION = "enhanced_verification"
    MANUAL_REVIEW = "manual_review"
    DENY_WITH_RATIONALE = "deny_with_rationale"


class OrderItem(BaseModel):
    asin: str
    title: str
    category: str
    price: float
    serial_number: Optional[str] = None
    weight_grams: Optional[float] = None


class ShipmentEvent(BaseModel):
    event_type: str
    timestamp: datetime
    carrier: str
    tracking_id: str
    weight_grams: Optional[float] = None
    location: Optional[str] = None


class AccountProfile(BaseModel):
    account_id: str
    account_age_days: int
    total_orders: int
    total_returns: int
    return_rate: float
    high_value_return_rate: float
    linked_addresses: list[str] = Field(default_factory=list)
    linked_payment_methods: int = 1
    previous_fraud_flags: int = 0
    category_return_concentration: dict[str, float] = Field(default_factory=dict)


class ReturnRequest(BaseModel):
    return_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    order_id: str
    account: AccountProfile
    item: OrderItem
    return_reason: str
    return_submitted_at: datetime
    order_delivered_at: datetime
    shipment_events: list[ShipmentEvent] = Field(default_factory=list)
    return_images: list[str] = Field(default_factory=list)
    return_image_data: list[tuple[bytes, str]] = Field(
        default_factory=list,
        description="List of (image_bytes, format) tuples from uploaded images",
    )
    customer_message: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class EvidenceSignal(BaseModel):
    signal_type: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    raw_data: Optional[dict] = None


class IntakeResult(BaseModel):
    return_id: str
    normalized_request: ReturnRequest
    initial_flags: list[str] = Field(default_factory=list)
    data_completeness_score: float = Field(ge=0.0, le=1.0)


class EvidenceResult(BaseModel):
    return_id: str
    signals: list[EvidenceSignal] = Field(default_factory=list)
    weight_anomaly: Optional[bool] = None
    serial_match: Optional[bool] = None
    image_similarity_score: Optional[float] = None
    packaging_integrity: Optional[float] = None
    evidence_summary: str = ""


class BehaviorResult(BaseModel):
    return_id: str
    account_risk_score: float = Field(ge=0.0, le=1.0)
    return_velocity_flag: bool = False
    address_cluster_flag: bool = False
    category_concentration_flag: bool = False
    value_pattern_flag: bool = False
    timing_anomaly_flag: bool = False
    linked_account_ids: list[str] = Field(default_factory=list)
    behavior_summary: str = ""


class PolicyResult(BaseModel):
    return_id: str
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    fraud_type_probabilities: dict[str, float] = Field(default_factory=dict)
    most_likely_fraud_type: FraudType
    confidence: float = Field(ge=0.0, le=1.0)
    policy_reasoning: str = ""
    contributing_factors: list[str] = Field(default_factory=list)


class ActionResult(BaseModel):
    return_id: str
    recommended_action: ActionType
    customer_message: str = ""
    internal_notes: str = ""
    escalation_required: bool = False
    escalation_reason: Optional[str] = None
    voice_escalation_script: Optional[str] = None


class AuditRecord(BaseModel):
    return_id: str
    case_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    risk_score: float
    risk_level: RiskLevel
    fraud_type: FraudType
    action_taken: ActionType
    evidence_summary: str
    behavior_summary: str
    policy_reasoning: str
    customer_message: str
    decision_explanation: str
    appeal_eligible: bool = True
    processing_time_ms: float = 0.0


class PipelineResult(BaseModel):
    return_id: str
    intake: IntakeResult
    evidence: EvidenceResult
    behavior: BehaviorResult
    policy: PolicyResult
    action: ActionResult
    audit: AuditRecord
    total_processing_time_ms: float = 0.0
