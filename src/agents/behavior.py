"""Behavior Agent: detects temporal, graph, and pattern anomalies in account behavior."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from ..models import AccountProfile, BehaviorResult, ReturnRequest
from .base import BaseAgent

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "policies.yaml"


def _load_thresholds() -> dict[str, float]:
    """Load behavioral thresholds from policies.yaml."""
    defaults = {
        "max_return_rate": 0.40,
        "max_high_value_rate": 0.30,
        "min_account_age": 30,
        "concentration_threshold": 0.60,
    }
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        thresholds = cfg.get("behavioral_signals", {}).get("thresholds", {})
        return {
            "max_return_rate": thresholds.get("max_return_rate_30d", defaults["max_return_rate"]),
            "max_high_value_rate": thresholds.get("max_high_value_return_rate", defaults["max_high_value_rate"]),
            "min_account_age": thresholds.get("min_account_age_days", defaults["min_account_age"]),
            "concentration_threshold": defaults["concentration_threshold"],
        }
    except Exception as exc:
        logger.warning("Could not load policies.yaml, using defaults: %s", exc)
        return defaults


_THRESHOLDS = _load_thresholds()


class BehaviorAgent(BaseAgent):
    name = "BehaviorAgent"

    SYSTEM_PROMPT = (
        "You are a behavioral analysis agent for Amazon return fraud detection. "
        "Analyze account history and return patterns to identify anomalous behavior. "
        "Look for: return velocity spikes, address clustering, category concentration, "
        "value targeting patterns, timing anomalies, and multi-account coordination. "
        "Output JSON with: account_risk_score (0-1), flags (object of booleans), "
        "linked_accounts (list), behavior_summary (string)."
    )

    MAX_RETURN_RATE = _THRESHOLDS["max_return_rate"]
    MAX_HIGH_VALUE_RATE = _THRESHOLDS["max_high_value_rate"]
    MIN_ACCOUNT_AGE = _THRESHOLDS["min_account_age"]
    CONCENTRATION_THRESHOLD = _THRESHOLDS["concentration_threshold"]

    def execute(
        self, *, request: ReturnRequest, **_: Any
    ) -> BehaviorResult:
        acct = request.account
        score_components: list[tuple[float, float]] = []

        # Return velocity
        return_velocity_flag = acct.return_rate > self.MAX_RETURN_RATE
        score_components.append(
            (min(acct.return_rate / self.MAX_RETURN_RATE, 1.0), 0.25)
        )

        # High-value return concentration
        value_pattern_flag = acct.high_value_return_rate > self.MAX_HIGH_VALUE_RATE
        score_components.append(
            (min(acct.high_value_return_rate / self.MAX_HIGH_VALUE_RATE, 1.0), 0.20)
        )

        # Address cluster detection
        address_cluster_flag = len(acct.linked_addresses) > 2
        addr_score = min(len(acct.linked_addresses) / 5, 1.0)
        score_components.append((addr_score, 0.10))

        # Category concentration
        max_concentration = max(
            acct.category_return_concentration.values(), default=0.0
        )
        category_flag = max_concentration > self.CONCENTRATION_THRESHOLD
        score_components.append(
            (min(max_concentration / self.CONCENTRATION_THRESHOLD, 1.0), 0.20)
        )

        # Account age risk
        age_risk = max(0, 1.0 - acct.account_age_days / self.MIN_ACCOUNT_AGE)
        score_components.append((age_risk, 0.15))

        # Timing anomaly: very fast return
        hours = (
            request.return_submitted_at - request.order_delivered_at
        ).total_seconds() / 3600
        timing_anomaly_flag = hours < 48
        timing_score = max(0, 1.0 - hours / 48) if hours < 48 else 0.0
        score_components.append((timing_score, 0.10))

        # Weighted risk score
        account_risk = sum(s * w for s, w in score_components)
        account_risk = round(min(account_risk, 1.0), 3)

        # Nova-enhanced analysis for high-risk accounts
        linked_ids: list[str] = []
        if account_risk > 0.5:
            linked_ids = self._detect_linked_accounts(acct)

        summary = self._build_summary(
            acct, account_risk, return_velocity_flag,
            address_cluster_flag, category_flag, value_pattern_flag,
            timing_anomaly_flag,
        )

        return BehaviorResult(
            return_id=request.return_id,
            account_risk_score=account_risk,
            return_velocity_flag=return_velocity_flag,
            address_cluster_flag=address_cluster_flag,
            category_concentration_flag=category_flag,
            value_pattern_flag=value_pattern_flag,
            timing_anomaly_flag=timing_anomaly_flag,
            linked_account_ids=linked_ids,
            behavior_summary=summary,
        )

    def _detect_linked_accounts(self, acct: AccountProfile) -> list[str]:
        """Use Nova reasoning to identify potentially linked accounts."""
        prompt = (
            f"Account {acct.account_id} has {len(acct.linked_addresses)} linked addresses, "
            f"{acct.linked_payment_methods} payment methods, "
            f"return rate {acct.return_rate:.0%}, "
            f"{acct.previous_fraud_flags} prior fraud flags. "
            "Identify if this pattern suggests multi-account coordination."
        )
        try:
            result = self.nova.reason_json(self.SYSTEM_PROMPT, prompt)
            return result.get("linked_accounts", [])
        except Exception:
            return []

    @staticmethod
    def _build_summary(
        acct: AccountProfile,
        risk: float,
        velocity: bool,
        cluster: bool,
        category: bool,
        value: bool,
        timing: bool,
    ) -> str:
        flags = []
        if velocity:
            flags.append(f"high return velocity ({acct.return_rate:.0%})")
        if cluster:
            flags.append(f"{len(acct.linked_addresses)} linked addresses")
        if category:
            flags.append("concentrated category returns")
        if value:
            flags.append(f"high-value targeting ({acct.high_value_return_rate:.0%})")
        if timing:
            flags.append("rapid return timing")
        if not flags:
            return f"Account {acct.account_id}: low behavioral risk ({risk:.2f})."
        flag_str = ", ".join(flags)
        return f"Account {acct.account_id}: risk {risk:.2f} - {flag_str}."
