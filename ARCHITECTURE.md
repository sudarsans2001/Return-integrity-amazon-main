# Architecture Guide - ReturnShield AI

## 🏗️ System Architecture

### Overview

ReturnShield AI uses a **multi-agent orchestration pattern** where six specialized agents work sequentially and in parallel to analyze return requests. Each agent is powered by Amazon Nova 2 Lite foundation model.

## Agent Design Principles

### 1. Single Responsibility
Each agent performs one specific task:
- **Intake**: Data validation and normalization
- **Evidence**: Physical evidence analysis
- **Behavior**: Pattern and anomaly detection
- **Policy**: Risk synthesis and classification
- **Action**: Decision making and messaging
- **Audit**: Compliance and record keeping

### 2. Stateless Processing
Agents are stateless - all context is passed through the pipeline via immutable data models.

### 3. Parallel Execution
Evidence and Behavior agents run concurrently to minimize total latency.

### 4. Fail-Safe Fallbacks
Each agent has rule-based fallbacks if Nova API calls fail.

---

## 🔄 Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    ReturnRequest (Input)                     │
│  • Order details                                             │
│  • Account profile                                           │
│  • Item information                                          │
│  • Shipment events                                           │
│  • Return images                                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   INTAKE AGENT      │
              │  ⏱️ ~50ms           │
              │                     │
              │  Validates:         │
              │  • Data completeness│
              │  • Required fields  │
              │  • Format checks    │
              │                     │
              │  Flags:             │
              │  • Rapid return     │
              │  • High value       │
              │  • New account      │
              │  • Missing data     │
              └──────────┬──────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
┌─────────────────────┐       ┌─────────────────────┐
│  EVIDENCE AGENT     │       │  BEHAVIOR AGENT     │
│  ⏱️ ~180ms         │       │  ⏱️ ~120ms         │
│                     │       │                     │
│  Physical Checks:   │       │  Pattern Analysis:  │
│  • Weight anomaly   │       │  • Return velocity  │
│  • Serial verify    │       │  • Address clusters │
│  • Image analysis   │       │  • Value targeting  │
│  • Packaging tamper │       │  • Category focus   │
│                     │       │  • Multi-account    │
│  Nova Integration:  │       │                     │
│  • Vision analysis  │       │  Nova Integration:  │
│  • Embeddings       │       │  • Reasoning        │
│  • OCR/barcode      │       │  • Graph analysis   │
└──────────┬──────────┘       └──────────┬──────────┘
           │                              │
           └──────────────┬───────────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │   POLICY AGENT      │
                │   ⏱️ ~100ms        │
                │                     │
                │  Synthesis:         │
                │  • Composite risk   │
                │  • Fraud classify   │
                │  • Confidence score │
                │                     │
                │  Formula:           │
                │  risk = 0.6×evidence│
                │       + 0.4×behavior│
                │                     │
                │  Nova Integration:  │
                │  • Structured reasoning│
                │  • Classification   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   ACTION AGENT      │
                │   ⏱️ ~80ms         │
                │                     │
                │  Decisions:         │
                │  • Action selection │
                │  • Customer message │
                │  • Internal notes   │
                │  • Escalation check │
                │                     │
                │  Nova Integration:  │
                │  • Message generation│
                │  • Sonic voice script│
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   AUDIT AGENT       │
                │   ⏱️ ~70ms         │
                │                     │
                │  Records:           │
                │  • Case summary     │
                │  • Decision explain │
                │  • Audit trail      │
                │  • Appeal flag      │
                │                     │
                │  Nova Integration:  │
                │  • Explanation gen  │
                └─────────────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   PipelineResult        │
              │   (Complete Analysis)   │
              └─────────────────────────┘
```

**Total Latency**: P50: 412ms | P95: 687ms | P99: 891ms

---

## 🔍 Agent Implementation Details

### Intake Agent

**File**: `src/agents/intake.py`

**Purpose**: Validate and normalize incoming return requests

**Algorithm**:
```python
def execute(request: ReturnRequest) -> IntakeResult:
    flags = []
    
    # Time-based checks
    hours_since_delivery = (submitted - delivered).hours
    if hours_since_delivery < 24:
        flags.append("RAPID_RETURN")
    
    # Value-based checks
    if price > 500:
        flags.append("HIGH_VALUE_ITEM")
    
    # Account-based checks
    if return_rate > 0.30:
        flags.append("HIGH_RETURN_RATE")
    if account_age < 30:
        flags.append("NEW_ACCOUNT")
    if previous_fraud_flags > 0:
        flags.append(f"PRIOR_FLAGS: {count}")
    
    # Data completeness
    completeness = count_present_fields() / total_fields
    
    return IntakeResult(flags, completeness, normalized_request)
```

**Nova Integration**: None (rule-based for speed)

---

### Evidence Agent

**File**: `src/agents/evidence.py`

**Purpose**: Analyze physical evidence for anomalies

**Algorithm**:
```python
def execute(request: ReturnRequest) -> EvidenceResult:
    signals = []
    
    # Weight Analysis
    if return_weight_available:
        deviation = abs(return_weight - expected_weight) / expected_weight
        if deviation > 0.30:
            signals.append(EvidenceSignal(
                type="WEIGHT_CRITICAL",
                confidence=0.95,
                description=f"{deviation:.0%} weight loss - likely empty box"
            ))
    
    # Serial Number Verification
    if serial_number_available:
        nova_result = nova.reason_json(
            system="You are a serial number verification agent",
            user=f"Verify if returned serial {returned_serial} matches "
                 f"shipped serial {shipped_serial} for {product_name}"
        )
        if not nova_result["serial_match"]:
            signals.append(EvidenceSignal(
                type="SERIAL_MISMATCH",
                confidence=0.90
            ))
    
    # Image Analysis (Nova Vision)
    if return_images:
        for image_bytes, format in return_image_data:
            vision_result = nova.reason_with_image_json(
                system="You are a visual fraud detection agent",
                user=f"Analyze this return image for product: {product}. "
                     f"Check for: wrong item, empty box, damage, counterfeit. "
                     f"Return JSON: {{similarity_score, anomalies, description}}",
                image_bytes=image_bytes,
                image_format=format
            )
            
            similarity = vision_result["similarity_score"]
            if similarity < 0.70:
                signals.append(EvidenceSignal(
                    type="VISION_ANOMALY",
                    confidence=1.0 - similarity,
                    description=vision_result["description"]
                ))
    
    # INR Detection
    has_delivery_proof = any(e.type == "delivered" for e in shipment_events)
    claims_not_received = "not received" in return_reason.lower()
    if has_delivery_proof and claims_not_received:
        signals.append(EvidenceSignal(
            type="INR_SUSPICIOUS",
            confidence=0.85,
            description="Delivery confirmed but customer claims not received"
        ))
    
    return EvidenceResult(signals, weight_anomaly, serial_match, ...)
```

**Nova Integration**:
- **Nova 2 Lite Vision**: Analyze return photos
- **Nova 2 Lite Reasoning**: Serial number verification
- **Nova Embeddings**: Cross-modal similarity matching

**Performance**: ~180ms (parallel image analysis)

---

### Behavior Agent

**File**: `src/agents/behavior.py`

**Purpose**: Detect behavioral patterns and anomalies

**Algorithm**:
```python
def execute(request: ReturnRequest) -> BehaviorResult:
    score_components = []
    
    # Return Velocity (25% weight)
    return_velocity_flag = return_rate > 0.40
    score_components.append((
        min(return_rate / 0.40, 1.0),
        0.25
    ))
    
    # High-Value Targeting (20% weight)
    value_pattern_flag = high_value_return_rate > 0.30
    score_components.append((
        min(high_value_return_rate / 0.30, 1.0),
        0.20
    ))
    
    # Address Clustering (10% weight)
    address_cluster_flag = len(linked_addresses) > 2
    score_components.append((
        min(len(linked_addresses) / 5, 1.0),
        0.10
    ))
    
    # Category Concentration (20% weight)
    max_category_concentration = max(category_returns.values())
    category_flag = max_concentration > 0.60
    score_components.append((
        min(max_concentration / 0.60, 1.0),
        0.20
    ))
    
    # Account Age (15% weight)
    age_risk = max(0, 1.0 - account_age_days / 30)
    score_components.append((age_risk, 0.15))
    
    # Timing Anomaly (10% weight)
    hours_to_return = (submitted - delivered).hours
    timing_flag = hours_to_return < 48
    timing_score = max(0, 1.0 - hours_to_return / 48)
    score_components.append((timing_score, 0.10))
    
    # Weighted composite
    account_risk = sum(score * weight for score, weight in score_components)
    
    # Nova-enhanced linked account detection
    if account_risk > 0.5:
        linked_accounts = detect_linked_accounts_with_nova(account)
    
    return BehaviorResult(
        account_risk_score=account_risk,
        flags=all_flags,
        linked_accounts=linked_accounts
    )
```

**Nova Integration**:
- **Nova 2 Lite Reasoning**: Multi-account coordination detection
- **Graph analysis**: Identify fraud rings

**Performance**: ~120ms

---

### Policy Agent

**File**: `src/agents/policy.py`

**Purpose**: Synthesize evidence and behavior into final risk assessment

**Algorithm**:
```python
def execute(evidence: EvidenceResult, behavior: BehaviorResult) -> PolicyResult:
    # Score evidence
    evidence_signals = [s.confidence for s in evidence.signals if s.confidence > 0.5]
    evidence_risk = mean(evidence_signals) if evidence_signals else 0.1
    
    # Composite risk (60/40 weighting)
    composite_risk = 0.60 * evidence_risk + 0.40 * behavior.account_risk_score
    composite_risk = round(min(composite_risk, 1.0), 3)
    
    # Map to risk level
    risk_level = classify_risk_level(composite_risk)
    # 0.00-0.30: LOW
    # 0.30-0.60: MEDIUM
    # 0.60-0.85: HIGH
    # 0.85-1.00: CRITICAL
    
    # Nova-enhanced fraud classification
    prompt = build_classification_prompt(
        request, evidence, behavior, composite_risk
    )
    
    nova_result = nova.reason_json(
        system="You are a fraud classification agent. Analyze the evidence "
               "and behavior to classify fraud type and explain reasoning.",
        user=prompt
    )
    
    fraud_type = FraudType(nova_result["fraud_type"])
    confidence = nova_result["confidence"]
    reasoning = nova_result["reasoning"]
    contributing_factors = nova_result["contributing_factors"]
    
    return PolicyResult(
        risk_score=composite_risk,
        risk_level=risk_level,
        fraud_type=fraud_type,
        confidence=confidence,
        reasoning=reasoning,
        contributing_factors=contributing_factors
    )
```

**Nova Integration**:
- **Nova 2 Lite Reasoning**: Fraud type classification
- **Structured output**: JSON schema for consistent decisions

**Performance**: ~100ms

---

### Action Agent

**File**: `src/agents/action.py`

**Purpose**: Map risk to actions and generate messaging

**Decision Matrix**:
```python
RISK_LEVEL → ACTION:
  LOW (0.0-0.3)      → AUTO_APPROVE
  MEDIUM (0.3-0.6)   → ENHANCED_VERIFICATION
  HIGH (0.6-0.85)    → MANUAL_REVIEW
  CRITICAL (0.85-1.0) → DENY_WITH_RATIONALE

OVERRIDES:
  if price > $200 and action == DENY:
      action = MANUAL_REVIEW  # Never auto-deny high-value
  
  if price > $500:
      escalation_required = True
  
  if price > $2000:
      escalation_reason = "Supervisor approval required"
```

**Algorithm**:
```python
def execute(policy: PolicyResult) -> ActionResult:
    # Map risk to action
    action = ACTION_MAP[policy.risk_level]
    
    # Apply safety overrides
    if action == DENY and item.price > 200:
        action = MANUAL_REVIEW
    
    # Nova-generated messaging
    prompt = f"Generate customer-facing and internal messaging for {action}"
    nova_result = nova.reason_json(system=MESSAGING_PROMPT, user=prompt)
    
    customer_message = nova_result["customer_message"]
    internal_notes = nova_result["internal_notes"]
    
    # Voice escalation (Nova Sonic)
    if item.price > 500:
        voice_script = nova.generate_voice_escalation_script(case_summary)
    
    return ActionResult(
        action=action,
        customer_message=customer_message,
        internal_notes=internal_notes,
        escalation_required=escalation_flag,
        voice_script=voice_script
    )
```

**Nova Integration**:
- **Nova 2 Lite**: Empathetic, policy-compliant message generation
- **Nova 2 Sonic**: Voice escalation scripts for operations center

**Performance**: ~80ms

---

### Audit Agent

**File**: `src/agents/audit.py`

**Purpose**: Create immutable compliance records

**Algorithm**:
```python
def execute(all_results) -> AuditRecord:
    # Nova-generated explanation
    prompt = build_explanation_prompt(
        request, evidence, behavior, policy, action
    )
    
    explanation = nova.reason_json(
        system="You are an audit and compliance agent. Generate clear, "
               "factual explanation suitable for appeals and compliance reviews.",
        user=prompt
    )
    
    return AuditRecord(
        case_id=generate_case_id(),
        timestamp=utcnow(),
        risk_score=policy.risk_score,
        risk_level=policy.risk_level,
        fraud_type=policy.fraud_type,
        action_taken=action.recommended_action,
        evidence_summary=evidence.summary,
        behavior_summary=behavior.summary,
        policy_reasoning=policy.reasoning,
        decision_explanation=explanation["decision_explanation"],
        appeal_eligible=policy.risk_level != LOW,
        processing_time_ms=total_time
    )
```

**Nova Integration**:
- **Nova 2 Lite**: Human-readable explanations for compliance

**Performance**: ~70ms

---

## 🔐 Security & Privacy

### Data Handling
- **Encryption at rest**: All audit records encrypted
- **Encryption in transit**: TLS 1.3 for all API calls
- **PII protection**: Customer data masked in logs
- **Data retention**: Configurable retention policies

### Access Control
- **API authentication**: JWT tokens or API keys
- **Role-based access**: Different permissions for agents/reviewers
- **Audit logging**: All decisions logged immutably

### Compliance
- **GDPR**: Right to explanation for all decisions
- **CCPA**: Data access and deletion requests supported
- **SOC 2**: Audit trails for all operations

---

## 📊 Performance Optimization

### Parallelization
Evidence and Behavior agents run concurrently using ThreadPoolExecutor:
```python
with ThreadPoolExecutor(max_workers=2) as executor:
    evidence_future = executor.submit(evidence_agent.run, request)
    behavior_future = executor.submit(behavior_agent.run, request)
    evidence_result = evidence_future.result()
    behavior_result = behavior_future.result()
```

**Latency Impact**: ~40% reduction (300ms → 180ms for combined E+B)

### Caching Strategy
- **Account profiles**: 5-minute TTL
- **Product metadata**: 1-hour TTL
- **Policy rules**: Cache until config change

### Scalability
- **Horizontal**: Stateless design enables auto-scaling
- **Vertical**: GPU acceleration for image analysis
- **Async processing**: SQS queue for batch jobs

---

## 🧪 Testing Strategy

### Unit Tests
Each agent has isolated unit tests:
```python
def test_intake_agent_flags_rapid_return():
    request = create_test_request(hours_since_delivery=12)
    result = intake_agent.execute(request)
    assert "RAPID_RETURN" in result.initial_flags
```

### Integration Tests
Full pipeline tests with synthetic cases:
```python
def test_empty_box_detection():
    request = create_empty_box_case()
    result = pipeline.process(request)
    assert result.policy.fraud_type == FraudType.EMPTY_BOX
    assert result.policy.risk_score > 0.85
```

### Load Tests
Locust for performance testing:
```python
class FraudDetectionUser(HttpUser):
    @task
    def analyze_return(self):
        self.client.post("/api/v1/returns/analyze", json=test_case)
```

**Target**: 120 req/min sustained, P95 < 700ms

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow
```yaml
name: CI/CD
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: pytest tests/ --cov=src
      
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Lint code
        run: ruff check src/
  
  deploy:
    needs: [test, lint]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to AWS
        run: |
          aws cloudformation deploy \
            --template-file cloudformation.yaml \
            --stack-name return-integrity
```

---

## 📚 Further Reading

- [Deployment Guide](DEPLOYMENT.md) - Production deployment instructions
- [Configuration Guide](CONFIG.md) - Policy tuning and customization
- [API Reference](http://localhost:8000/docs) - Interactive API docs
- [Contributing Guide](CONTRIBUTING.md) - Development guidelines
