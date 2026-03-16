"""ReturnShield AI orchestrator: wires all six agents into a sequential pipeline."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

from .agents import (
    ActionAgent,
    AuditAgent,
    BehaviorAgent,
    EvidenceAgent,
    IntakeAgent,
    PolicyAgent,
)
from .models import PipelineResult, ReturnRequest
from .nova_client import NovaClient

logger = logging.getLogger(__name__)


class ReturnIntegrityPipeline:
    """Orchestrates the six-agent pipeline for return fraud assessment."""

    def __init__(self, nova: NovaClient | None = None) -> None:
        self.nova = nova or NovaClient()
        self.intake = IntakeAgent(self.nova)
        self.evidence = EvidenceAgent(self.nova)
        self.behavior = BehaviorAgent(self.nova)
        self.policy = PolicyAgent(self.nova)
        self.action = ActionAgent(self.nova)
        self.audit = AuditAgent(self.nova)

    def process(self, request: ReturnRequest) -> PipelineResult:
        """Run the full pipeline on a single return request."""
        pipeline_start = time.perf_counter()

        logger.info("=" * 60)
        logger.info("Processing return %s", request.return_id)
        logger.info("=" * 60)

        # Stage 1: Intake
        intake_result, t1 = self.intake.run(request=request)

        # Stage 2: Evidence + Behavior (run in parallel)
        with ThreadPoolExecutor(max_workers=2) as executor:
            ev_future = executor.submit(self.evidence.run, request=request)
            bh_future = executor.submit(self.behavior.run, request=request)
            evidence_result, t2 = ev_future.result()
            behavior_result, t3 = bh_future.result()

        # Stage 3: Policy (depends on evidence + behavior)
        policy_result, t4 = self.policy.run(
            request=request,
            evidence=evidence_result,
            behavior=behavior_result,
        )

        # Stage 4: Action (depends on policy)
        action_result, t5 = self.action.run(
            request=request,
            policy=policy_result,
        )

        # Stage 5: Audit (depends on everything)
        elapsed_so_far = t1 + t2 + t3 + t4 + t5
        audit_result, t6 = self.audit.run(
            request=request,
            evidence=evidence_result,
            behavior=behavior_result,
            policy=policy_result,
            action=action_result,
            processing_time_ms=elapsed_so_far,
        )

        total_ms = (time.perf_counter() - pipeline_start) * 1000

        logger.info("-" * 60)
        logger.info(
            "Pipeline complete: risk=%.3f (%s), action=%s, %.0f ms",
            policy_result.risk_score,
            policy_result.risk_level.value,
            action_result.recommended_action.value,
            total_ms,
        )

        return PipelineResult(
            return_id=request.return_id,
            intake=intake_result,
            evidence=evidence_result,
            behavior=behavior_result,
            policy=policy_result,
            action=action_result,
            audit=audit_result,
            total_processing_time_ms=round(total_ms, 1),
        )

    def process_batch(
        self, requests: list[ReturnRequest]
    ) -> list[PipelineResult]:
        """Process multiple return requests."""
        results = []
        for req in requests:
            results.append(self.process(req))
        return results
