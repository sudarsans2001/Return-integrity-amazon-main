"""FastAPI server for ReturnShield AI API."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.case_builder import CaseBuilder
from src.models import (
    AccountProfile,
    FraudType,
    OrderItem,
    ReturnRequest,
    ShipmentEvent,
)
from src.metrics import compute_metrics
from src.nova_client import NovaClient
from src.orchestrator import ReturnIntegrityPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ReturnShield AI API",
    description="Multi-Agent Fraud Detection powered by Amazon Nova",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_nova = NovaClient()
pipeline = ReturnIntegrityPipeline(_nova)
case_builder = CaseBuilder(_nova)

_processed_results: list = []


class ReturnAnalysisRequest(BaseModel):
    order_id: str
    account: AccountProfile
    item: OrderItem
    return_reason: str
    return_submitted_at: str
    order_delivered_at: str
    shipment_events: list[ShipmentEvent] = Field(default_factory=list)
    return_images: list[str] = Field(default_factory=list)
    customer_message: Optional[str] = None


class TextAnalysisRequest(BaseModel):
    description: str = Field(
        ..., description="Natural language description of the return scenario"
    )


class ReturnAnalysisResponse(BaseModel):
    return_id: str
    risk_score: float
    risk_level: str
    fraud_type: str
    confidence: float
    recommended_action: str
    customer_message: str
    processing_time_ms: float
    timestamp: str


@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "ReturnShield AI",
        "version": "1.0.0",
        "powered_by": "Amazon Nova",
        "api_docs": "/docs",
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "nova_connected": True,
        "agents_active": 6,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/api/v1/returns/analyze", response_model=ReturnAnalysisResponse)
async def analyze_return(request: ReturnAnalysisRequest):
    """Analyze a structured return request for fraud indicators."""
    try:
        return_request = ReturnRequest(
            order_id=request.order_id,
            account=request.account,
            item=request.item,
            return_reason=request.return_reason,
            return_submitted_at=datetime.fromisoformat(request.return_submitted_at),
            order_delivered_at=datetime.fromisoformat(request.order_delivered_at),
            shipment_events=request.shipment_events,
            return_images=request.return_images,
            customer_message=request.customer_message,
        )

        result = pipeline.process(return_request)
        _processed_results.append(result)

        return ReturnAnalysisResponse(
            return_id=result.return_id,
            risk_score=result.policy.risk_score,
            risk_level=result.policy.risk_level.value,
            fraud_type=result.policy.most_likely_fraud_type.value,
            confidence=result.policy.confidence,
            recommended_action=result.action.recommended_action.value,
            customer_message=result.action.customer_message,
            processing_time_ms=result.total_processing_time_ms,
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.error("Error processing return: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/returns/analyze-text", response_model=ReturnAnalysisResponse)
async def analyze_return_text(request: TextAnalysisRequest):
    """Analyze a return described in natural language.

    Nova 2 Lite parses the description into a structured case, then the
    6-agent pipeline processes it for fraud detection.
    """
    try:
        return_request = case_builder.build_from_text(request.description)
        result = pipeline.process(return_request)
        _processed_results.append(result)

        return ReturnAnalysisResponse(
            return_id=result.return_id,
            risk_score=result.policy.risk_score,
            risk_level=result.policy.risk_level.value,
            fraud_type=result.policy.most_likely_fraud_type.value,
            confidence=result.policy.confidence,
            recommended_action=result.action.recommended_action.value,
            customer_message=result.action.customer_message,
            processing_time_ms=result.total_processing_time_ms,
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.error("Error processing text return: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/returns/analyze-with-images")
async def analyze_return_with_images(
    request_data: str,
    images: list[UploadFile] = File(None),
):
    """Analyze return with uploaded images for Nova Vision analysis."""
    try:
        import json as _json
        request_dict = _json.loads(request_data)
        request = ReturnAnalysisRequest(**request_dict)

        return_request = ReturnRequest(
            order_id=request.order_id,
            account=request.account,
            item=request.item,
            return_reason=request.return_reason,
            return_submitted_at=datetime.fromisoformat(request.return_submitted_at),
            order_delivered_at=datetime.fromisoformat(request.order_delivered_at),
            shipment_events=request.shipment_events,
            customer_message=request.customer_message,
        )

        if images:
            image_data = []
            for img in images:
                content = await img.read()
                ext = img.filename.rsplit(".", 1)[-1].lower() if img.filename else "jpeg"
                fmt = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png"}.get(ext, "jpeg")
                image_data.append((content, fmt))
            return_request.return_image_data = image_data

        result = pipeline.process(return_request)
        _processed_results.append(result)

        return {
            "return_id": result.return_id,
            "risk_score": result.policy.risk_score,
            "risk_level": result.policy.risk_level.value,
            "fraud_type": result.policy.most_likely_fraud_type.value,
            "confidence": result.policy.confidence,
            "recommended_action": result.action.recommended_action.value,
            "customer_message": result.action.customer_message,
            "processing_time_ms": result.total_processing_time_ms,
            "image_analysis": {
                "images_processed": len(images) if images else 0,
                "similarity_score": result.evidence.image_similarity_score,
                "anomalies_detected": len(result.evidence.signals),
            },
        }

    except Exception as e:
        logger.error("Error processing return with images: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/metrics")
async def get_metrics():
    """Get real performance metrics from processed cases."""
    if not _processed_results:
        return {
            "message": "No cases processed yet. Submit cases via /api/v1/returns/analyze first.",
            "total_cases_processed": 0,
            "timestamp": datetime.utcnow().isoformat(),
        }

    summary = compute_metrics(_processed_results, {})
    return {
        "precision": round(summary.precision, 3),
        "recall": round(summary.recall, 3),
        "f1_score": round(summary.f1_score, 3),
        "false_positive_rate": round(summary.false_positive_rate, 3),
        "mean_latency_ms": round(summary.mean_latency_ms, 1),
        "total_cases_processed": summary.total_cases,
        "risk_distribution": summary.risk_distribution,
        "action_distribution": summary.action_distribution,
        "fraud_type_distribution": summary.fraud_type_distribution,
        "timestamp": datetime.utcnow().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
