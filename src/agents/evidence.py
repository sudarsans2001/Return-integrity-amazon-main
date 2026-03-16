"""Evidence Agent: performs multimodal checks on return evidence."""

from __future__ import annotations

from typing import Any

from ..models import EvidenceResult, EvidenceSignal, ReturnRequest
from .base import BaseAgent


class EvidenceAgent(BaseAgent):
    name = "EvidenceAgent"

    SYSTEM_PROMPT = (
        "You are an evidence analysis agent for Amazon return fraud detection. "
        "Analyze the provided return evidence and identify anomalies. "
        "Check for: weight discrepancies, serial number mismatches, "
        "packaging tampering, item condition inconsistencies. "
        "Output JSON with fields: signals (list of {type, description, confidence}), "
        "weight_anomaly (bool), serial_match (bool), packaging_integrity (0-1), "
        "evidence_summary (string)."
    )

    def execute(
        self, *, request: ReturnRequest, **_: Any
    ) -> EvidenceResult:
        signals: list[EvidenceSignal] = []
        self._vision_signals: list[EvidenceSignal] = []
        weight_anomaly = None
        serial_match = None
        packaging_integrity = None

        # Weight analysis
        if request.item.weight_grams and request.shipment_events:
            return_weight = next(
                (e.weight_grams for e in request.shipment_events
                 if e.event_type == "return_received" and e.weight_grams),
                None,
            )
            if return_weight is not None:
                expected = request.item.weight_grams
                deviation = abs(return_weight - expected) / expected
                if deviation > 0.30:
                    weight_anomaly = True
                    signals.append(EvidenceSignal(
                        signal_type="WEIGHT_CRITICAL",
                        description=f"Return weight {return_weight}g vs expected {expected}g ({deviation:.0%} deviation)",
                        confidence=min(0.95, 0.5 + deviation),
                    ))
                elif deviation > 0.05:
                    weight_anomaly = True
                    signals.append(EvidenceSignal(
                        signal_type="WEIGHT_WARNING",
                        description=f"Minor weight deviation: {deviation:.1%}",
                        confidence=0.4 + deviation,
                    ))
                else:
                    weight_anomaly = False

        # Serial number verification
        if request.item.serial_number:
            nova_prompt = (
                f"Return for {request.item.title} (ASIN: {request.item.asin}). "
                f"Shipped serial: {request.item.serial_number}. "
                f"Return reason: {request.return_reason}. "
                f"Customer message: {request.customer_message or 'None'}. "
                "Assess serial number match likelihood."
            )
            try:
                nova_response = self.nova.reason_json(
                    self.SYSTEM_PROMPT, nova_prompt
                )
                serial_match_val = nova_response.get("serial_match", True)
                serial_match = bool(serial_match_val)
                if not serial_match:
                    signals.append(EvidenceSignal(
                        signal_type="SERIAL_MISMATCH",
                        description="Nova analysis indicates serial number discrepancy",
                        confidence=nova_response.get("confidence", 0.85),
                        raw_data=nova_response,
                    ))
            except Exception:
                serial_match = None

        # Packaging / image analysis via Nova reasoning
        evidence_prompt = self._build_evidence_prompt(request)
        try:
            nova_result = self.nova.reason_json(
                self.SYSTEM_PROMPT, evidence_prompt
            )
            packaging_integrity = nova_result.get("packaging_integrity", 0.8)

            for sig in nova_result.get("signals", []):
                signals.append(EvidenceSignal(
                    signal_type=sig.get("type", "NOVA_SIGNAL"),
                    description=sig.get("description", ""),
                    confidence=sig.get("confidence", 0.5),
                ))
            evidence_summary = nova_result.get("evidence_summary", "")
        except Exception:
            evidence_summary = self._rule_based_summary(request, signals)

        # INR (Item Not Received) detection
        has_delivery = any(
            e.event_type == "delivered" for e in request.shipment_events
        )
        has_return_received = any(
            e.event_type == "return_received" for e in request.shipment_events
        )
        reason_lower = request.return_reason.lower()
        if has_delivery and not has_return_received and (
            "not received" in reason_lower or "never received" in reason_lower
        ):
            delivery_event = next(
                (e for e in request.shipment_events if e.event_type == "delivered"),
                None,
            )
            confidence = 0.85
            detail = "Tracking shows delivery but customer claims item not received"
            if delivery_event and delivery_event.location:
                loc = delivery_event.location.lower()
                if "handed" in loc or "resident" in loc:
                    confidence = 0.90
                    detail += f"; delivered to '{delivery_event.location}' (direct handoff)"
                elif "signed" in loc:
                    confidence = 0.93
                    detail += f"; signed delivery at '{delivery_event.location}'"
            signals.append(EvidenceSignal(
                signal_type="INR_SUSPICIOUS",
                description=detail,
                confidence=confidence,
            ))

        # Image analysis via Nova Lite multimodal vision
        image_similarity = None
        if request.return_image_data:
            image_similarity = self._analyze_return_images(request)
            signals.extend(self._vision_signals)
        elif request.return_images:
            image_similarity = self._compute_image_similarity(request)

        return EvidenceResult(
            return_id=request.return_id,
            signals=signals,
            weight_anomaly=weight_anomaly,
            serial_match=serial_match,
            image_similarity_score=image_similarity,
            packaging_integrity=packaging_integrity,
            evidence_summary=evidence_summary or self._rule_based_summary(
                request, signals
            ),
        )

    def _build_evidence_prompt(self, request: ReturnRequest) -> str:
        parts = [
            f"Analyze return for: {request.item.title}",
            f"ASIN: {request.item.asin}, Price: ${request.item.price:.2f}",
            f"Return reason: {request.return_reason}",
        ]
        if request.customer_message:
            parts.append(f"Customer message: {request.customer_message}")
        if request.item.weight_grams:
            parts.append(f"Expected weight: {request.item.weight_grams}g")
        if request.return_images:
            parts.append(f"Return images provided: {len(request.return_images)}")
        return "\n".join(parts)

    def _analyze_return_images(self, request: ReturnRequest) -> float:
        """Analyze uploaded return images using Nova Lite multimodal vision."""
        vision_prompt = (
            f"Analyze this return image for product: {request.item.title} "
            f"(ASIN: {request.item.asin}, Category: {request.item.category}).\n"
            f"Return reason: {request.return_reason}\n"
            "Check for: wrong item, empty box, visible damage, counterfeit "
            "indicators, missing accessories, tampered packaging.\n"
            "Return JSON with: similarity_score (0-1 how likely this is the "
            "genuine product), anomalies (list of strings), "
            "description (brief analysis)."
        )
        system = (
            "You are a visual evidence analyst for Amazon return fraud detection. "
            "Carefully examine the image and assess whether the returned item "
            "matches the expected product. Output valid JSON only."
        )
        scores = []
        for img_bytes, img_format in request.return_image_data:
            try:
                result = self.nova.reason_with_image_json(
                    system, vision_prompt, img_bytes, img_format
                )
                score = float(result.get("similarity_score", 0.5))
                scores.append(score)
                for anomaly in result.get("anomalies", []):
                    self._vision_signals.append(EvidenceSignal(
                        signal_type="VISION_ANOMALY",
                        description=str(anomaly),
                        confidence=max(0.0, min(1.0, 1.0 - score)),
                    ))
            except Exception:
                scores.append(0.5)
        return round(sum(scores) / len(scores), 3) if scores else 0.75

    def _compute_image_similarity(self, request: ReturnRequest) -> float:
        """Fallback text-proxy similarity when no actual images are uploaded."""
        product_text = f"{request.item.title} {request.item.asin} genuine product"
        return_text = f"returned item {request.return_reason}"
        try:
            emb_product = self.nova.embed_text(product_text)
            emb_return = self.nova.embed_text(return_text)
            dot = sum(a * b for a, b in zip(emb_product, emb_return))
            mag_a = sum(a**2 for a in emb_product) ** 0.5
            mag_b = sum(b**2 for b in emb_return) ** 0.5
            if mag_a > 0 and mag_b > 0:
                return round(dot / (mag_a * mag_b), 3)
        except Exception:
            pass
        return 0.75

    @staticmethod
    def _rule_based_summary(
        request: ReturnRequest, signals: list[EvidenceSignal]
    ) -> str:
        if not signals:
            return f"No evidence anomalies detected for return of {request.item.title}."
        critical = [s for s in signals if s.confidence > 0.7]
        if critical:
            descs = "; ".join(s.description for s in critical)
            return f"Critical evidence signals: {descs}"
        descs = "; ".join(s.description for s in signals)
        return f"Evidence signals detected: {descs}"
