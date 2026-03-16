# ReturnShield AI

[![Powered by Amazon Nova](https://img.shields.io/badge/Powered%20by-Amazon%20Nova-FF9900?style=for-the-badge&logo=amazon-aws)](https://aws.amazon.com/nova/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)

> **Multi-agent return fraud detection** powered by Amazon Nova foundation models.

Built for the **Amazon Nova AI Hackathon 2026** | Track: **Agentic AI** | **#AmazonNova**

---

## The Problem

Return fraud costs Amazon sellers an estimated **$2.4 billion annually**. Fraud tactics range from empty-box returns and counterfeit swaps to coordinated multi-account abuse. Traditional rule-based systems achieve roughly 65 % accuracy with high false-positive rates, damaging customer trust while still missing sophisticated fraud.

## Our Solution

ReturnShield AI is a **six-agent AI pipeline** that processes return requests in near-real-time. Each agent is specialized and powered by Amazon Nova:

1. **Intake Agent** -- validates and normalizes the request, raises initial flags.
2. **Evidence Agent** -- analyzes weight, serial numbers, images (Nova Vision), and packaging.
3. **Behavior Agent** -- detects return velocity spikes, address clustering, and value targeting.
4. **Policy Agent** -- synthesizes a composite risk score (`0.6 x evidence + 0.4 x behavior`) and classifies fraud type via Nova structured reasoning.
5. **Action Agent** -- maps risk to an action (approve / verify / review / deny) and generates customer messaging and voice escalation scripts.
6. **Audit Agent** -- produces an immutable, explainable case record.

Evidence and Behavior run **in parallel** for lower latency.

### Live Natural-Language Analysis

Users can describe **any** return scenario in plain English (e.g. *"Damaged box of iPhone 15 Pro, returned same day"*). Nova 2 Lite parses the description into a full structured case and runs it through the entire pipeline -- no static data lookup.

### Amazon Nova Models Used

| Model | Role |
|-------|------|
| **Nova 2 Lite** (`us.amazon.nova-2-lite-v1:0`) | Orchestration, structured reasoning, fraud classification, message generation, natural-language case parsing |
| **Nova 2 Lite Vision** (multimodal) | Return photo analysis for counterfeit / damage / empty-box detection |
| **Nova Multimodal Embeddings** (`amazon.nova-2-multimodal-embeddings-v1:0`) | Cross-modal similarity between product listing and return evidence |
| **Nova 2 Sonic** (`us.amazon.nova-2-sonic-v1:0`) | Voice escalation scripts for operations center |

---

## Architecture

```
                         ReturnRequest
                              │
                    ┌─────────┴─────────┐
                    │   INTAKE AGENT    │   ~50 ms
                    └─────────┬─────────┘
                              │
              ┌───────────────┼───────────────┐
              │  (parallel)                   │
    ┌─────────┴─────────┐          ┌─────────┴─────────┐
    │  EVIDENCE AGENT   │          │  BEHAVIOR AGENT   │
    │  Weight / Serial  │          │  Velocity / Addr  │
    │  Vision / INR     │          │  Category / Time  │
    └─────────┬─────────┘          └─────────┬─────────┘
              └───────────────┬───────────────┘
                    ┌─────────┴─────────┐
                    │   POLICY AGENT    │   risk = 0.6E + 0.4B
                    └─────────┬─────────┘
                    ┌─────────┴─────────┐
                    │   ACTION AGENT    │   approve / verify / review / deny
                    └─────────┬─────────┘
                    ┌─────────┴─────────┐
                    │   AUDIT AGENT     │   explainable record
                    └─────────────────────┘
```

---

## Fraud Types Detected

| Type | Method |
|------|--------|
| Empty Box | Weight deviation > 30 % |
| Counterfeit Swap | Serial mismatch + visual similarity |
| Serial Mismatch | Nova reasoning on serial verification |
| INR Fraud | Delivery proof vs "not received" claim |
| Multi-Account | Address/payment graph analysis |
| Policy Gaming | Return velocity and timing patterns |
| Used-as-New | Usage indicators + image analysis |

---

## Quick Start

### Prerequisites

- Python 3.11+
- AWS account with Bedrock access and Nova models enabled (us-east-1 recommended)

### Install

```bash
python -m venv .venv
# Linux/Mac: source .venv/bin/activate
# Windows:   .venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env with your AWS credentials
```

### Run the Dashboard (recommended)

```bash
streamlit run dashboard/app.py
# Open http://localhost:8501
# Use the Live Case Analysis tab to describe any return scenario
```

### Run the API Server

```bash
uvicorn api.server:app --reload
# API docs: http://localhost:8000/docs
```

### Run the Batch Pipeline

```bash
python -m src.main
# Processes the 8 synthetic test cases and prints metrics
```

---

## API Reference

### `POST /api/v1/returns/analyze-text`

Analyze a return described in natural language.

```json
{
  "description": "Customer returned an iPhone 15 Pro with a damaged box, same-day return"
}
```

**Response:**
```json
{
  "return_id": "abc123",
  "risk_score": 0.82,
  "risk_level": "high",
  "fraud_type": "EMPTY_BOX",
  "confidence": 0.88,
  "recommended_action": "manual_review",
  "customer_message": "Your return is under review...",
  "processing_time_ms": 487.3,
  "timestamp": "2026-03-16T12:00:00Z"
}
```

### `POST /api/v1/returns/analyze`

Analyze a structured return request (full JSON with account, item, shipment events).

### `POST /api/v1/returns/analyze-with-images`

Multipart upload for return photo analysis via Nova Vision.

### `GET /api/v1/metrics`

Real-time performance metrics computed from processed cases.

### `GET /health`

Health check and agent status.

Full interactive docs at **http://localhost:8000/docs** (Swagger UI).

---

## Docker

```bash
cp .env.example .env   # add your AWS credentials
docker-compose up --build

# API:       http://localhost:8000
# Dashboard: http://localhost:8501
```

---

## Project Structure

```
returnshield-ai/
├── api/
│   └── server.py              # FastAPI REST API
├── config/
│   ├── scope.yaml             # Domain scope and success metrics
│   └── policies.yaml          # Policy rules and thresholds
├── dashboard/
│   └── app.py                 # Streamlit dashboard (live analysis + benchmarks)
├── data/
│   └── synthetic_cases.json   # 8 test scenarios for benchmarking
├── demo/
│   ├── scenarios.py           # Pre-built demo scenarios
│   └── run_demo.py            # Terminal demo with Rich output
├── src/
│   ├── main.py                # Batch pipeline entry point
│   ├── orchestrator.py        # 6-agent pipeline orchestrator
│   ├── nova_client.py         # AWS Bedrock Nova client wrapper
│   ├── case_builder.py        # Natural language → ReturnRequest (Nova)
│   ├── models.py              # Pydantic data models
│   ├── metrics.py             # KPI computation
│   └── agents/
│       ├── base.py            # Base agent class
│       ├── intake.py          # Agent 1: Data validation
│       ├── evidence.py        # Agent 2: Physical evidence + vision
│       ├── behavior.py        # Agent 3: Pattern detection
│       ├── policy.py          # Agent 4: Risk synthesis
│       ├── action.py          # Agent 5: Decision + messaging
│       └── audit.py           # Agent 6: Compliance records
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── ARCHITECTURE.md
└── README.md
```

---

## License

MIT

---

**#AmazonNova** | **#AgenticAI** | **Amazon Nova AI Hackathon 2026**
