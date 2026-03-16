"""Pre-built demo scenarios for live presentation."""

from __future__ import annotations

from datetime import datetime, timedelta

from src.models import (
    AccountProfile,
    FraudType,
    OrderItem,
    ReturnRequest,
    ShipmentEvent,
)

NOW = datetime(2026, 3, 3, 12, 0, 0)


def scenario_empty_box_ps5() -> tuple[ReturnRequest, FraudType]:
    """High-confidence empty box fraud on a PS5."""
    return ReturnRequest(
        order_id="DEMO-EB-001",
        account=AccountProfile(
            account_id="DEMO-ACCT-FRAUD-1",
            account_age_days=45,
            total_orders=12,
            total_returns=5,
            return_rate=0.42,
            high_value_return_rate=0.60,
            linked_addresses=["123 Main St", "456 Elm St", "789 Oak Ave"],
            linked_payment_methods=2,
            previous_fraud_flags=1,
            category_return_concentration={"Consumer Electronics": 0.80},
        ),
        item=OrderItem(
            asin="B0D1XD1ZV3",
            title="PlayStation 5 Digital Edition",
            category="Gaming Consoles",
            price=449.99,
            serial_number="PS5-2026-ABX9912",
            weight_grams=3900,
        ),
        return_reason="Item arrived damaged",
        return_submitted_at=NOW,
        order_delivered_at=NOW - timedelta(hours=20),
        shipment_events=[
            ShipmentEvent(
                event_type="delivered",
                timestamp=NOW - timedelta(hours=20),
                carrier="UPS",
                tracking_id="1Z999AA10123456784",
                weight_grams=4500,
                location="Front Door",
            ),
            ShipmentEvent(
                event_type="return_received",
                timestamp=NOW + timedelta(days=2),
                carrier="UPS",
                tracking_id="1Z999AA10123456799",
                weight_grams=950,
                location="Amazon FC",
            ),
        ],
        return_images=["return_img_001.jpg"],
        customer_message="Box was damaged on arrival.",
    ), FraudType.EMPTY_BOX


def scenario_legitimate_tablet() -> tuple[ReturnRequest, FraudType]:
    """Clearly legitimate return from a trusted long-time customer."""
    return ReturnRequest(
        order_id="DEMO-LEG-001",
        account=AccountProfile(
            account_id="DEMO-ACCT-GOOD-1",
            account_age_days=1200,
            total_orders=150,
            total_returns=8,
            return_rate=0.05,
            high_value_return_rate=0.02,
            linked_addresses=["300 Loyal Customer Ave"],
            linked_payment_methods=1,
            previous_fraud_flags=0,
            category_return_concentration={"Books": 0.50, "Home": 0.30},
        ),
        item=OrderItem(
            asin="B0D5C5K1X9",
            title="Samsung Galaxy Tab S9 FE",
            category="Tablets",
            price=349.99,
            serial_number="SN-SAMG-S9FE-77430",
            weight_grams=523,
        ),
        return_reason="Item defective",
        return_submitted_at=NOW,
        order_delivered_at=NOW - timedelta(days=6),
        shipment_events=[
            ShipmentEvent(
                event_type="delivered",
                timestamp=NOW - timedelta(days=6),
                carrier="UPS",
                tracking_id="1Z777CC30345678901",
                weight_grams=900,
            ),
            ShipmentEvent(
                event_type="return_received",
                timestamp=NOW + timedelta(days=2),
                carrier="UPS",
                tracking_id="1Z777CC30345678999",
                weight_grams=890,
            ),
        ],
        return_images=["img_005a.jpg", "img_005b.jpg"],
        customer_message="Dead pixel area appeared after 3 days of use.",
    ), FraudType.LEGITIMATE


def scenario_counterfeit_swap_macbook() -> tuple[ReturnRequest, FraudType]:
    """Counterfeit swap on a high-value MacBook."""
    return ReturnRequest(
        order_id="DEMO-CS-001",
        account=AccountProfile(
            account_id="DEMO-ACCT-FRAUD-2",
            account_age_days=180,
            total_orders=28,
            total_returns=8,
            return_rate=0.29,
            high_value_return_rate=0.38,
            linked_addresses=["100 Tech Blvd"],
            linked_payment_methods=1,
            previous_fraud_flags=0,
            category_return_concentration={"Consumer Electronics": 0.75},
        ),
        item=OrderItem(
            asin="B0DCSRPG5Q",
            title="Apple MacBook Air 15-inch M3 256GB",
            category="Laptops",
            price=1299.00,
            serial_number="FVFXM3Q2024A",
            weight_grams=1510,
        ),
        return_reason="Item not as described",
        return_submitted_at=NOW,
        order_delivered_at=NOW - timedelta(days=5),
        shipment_events=[
            ShipmentEvent(
                event_type="delivered",
                timestamp=NOW - timedelta(days=5),
                carrier="AMZL",
                tracking_id="TBA123456789000",
                weight_grams=2100,
            ),
            ShipmentEvent(
                event_type="return_received",
                timestamp=NOW + timedelta(days=3),
                carrier="UPS",
                tracking_id="1Z888BB20234567890",
                weight_grams=2050,
            ),
        ],
        return_images=["img_002a.jpg", "img_002b.jpg"],
        customer_message="Not the laptop I ordered. Screen quality is different.",
    ), FraudType.COUNTERFEIT_SWAP


ALL_SCENARIOS = {
    "empty_box_ps5": scenario_empty_box_ps5,
    "legitimate_tablet": scenario_legitimate_tablet,
    "counterfeit_swap_macbook": scenario_counterfeit_swap_macbook,
}
