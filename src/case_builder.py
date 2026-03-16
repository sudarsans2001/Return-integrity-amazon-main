"""Natural language case builder: converts free-form text into a structured ReturnRequest.

Uses Amazon Nova 2 Lite to parse user descriptions like "damaged box of iPhone 15 Pro,
returned same day" into fully populated ReturnRequest objects that can be processed
by the 6-agent pipeline.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any

from .models import (
    AccountProfile,
    OrderItem,
    ReturnRequest,
    ShipmentEvent,
)
from .nova_client import NovaClient

logger = logging.getLogger(__name__)

CASE_BUILDER_SYSTEM_PROMPT = """\
You are a return fraud case generator for Amazon's ReturnShield AI system.
Given a natural language description of a return scenario, generate a realistic,
complete JSON representation of a return case.

You MUST output valid JSON with exactly these fields:

{
  "order_id": "ORD-YYYY-NNN",
  "account": {
    "account_id": "ACCT-NNNN",
    "account_age_days": <int 1-3000>,
    "total_orders": <int 1-500>,
    "total_returns": <int 0-100>,
    "return_rate": <float 0.0-1.0>,
    "high_value_return_rate": <float 0.0-1.0>,
    "linked_addresses": ["<address1>", ...],
    "linked_payment_methods": <int 1-5>,
    "previous_fraud_flags": <int 0-10>,
    "category_return_concentration": {"<category>": <float>, ...}
  },
  "item": {
    "asin": "<realistic ASIN>",
    "title": "<full product name>",
    "category": "<product category>",
    "price": <float>,
    "serial_number": "<SN-prefix-alphanumeric>",
    "weight_grams": <float>
  },
  "return_reason": "<reason string>",
  "return_submitted_at": "<ISO 8601 datetime>",
  "order_delivered_at": "<ISO 8601 datetime>",
  "shipment_events": [
    {
      "event_type": "delivered",
      "timestamp": "<ISO 8601>",
      "carrier": "<UPS|FedEx|AMZL|USPS>",
      "tracking_id": "<tracking number>",
      "weight_grams": <float>,
      "location": "<delivery location>"
    },
    {
      "event_type": "return_received",
      "timestamp": "<ISO 8601>",
      "carrier": "<carrier>",
      "tracking_id": "<tracking number>",
      "weight_grams": <float>,
      "location": "Amazon FC"
    }
  ],
  "return_images": [],
  "customer_message": "<what the customer says>"
}

RULES:
- If the scenario describes FRAUD, make the data reflect fraud signals:
  * Empty box: return_received weight should be ~20-25% of item weight
  * Counterfeit swap: weight close but serial number should hint at mismatch
  * Item not received: omit the return_received shipment event, use "not received" in reason
  * Policy gaming: high return_rate (>0.40), many total_returns
  * Multi-account: multiple linked_addresses (3+), multiple payment methods, previous_fraud_flags > 0
  * Serial mismatch: returned weight normal but serial hints at different unit
- If the scenario describes a LEGITIMATE return, make the data reflect trust signals:
  * Long account age (500+ days), low return rate (<0.10), 1 address, 0 fraud flags
- Use realistic product prices, weights, and ASINs for the described item
- Timestamps should be recent (within the last 2 weeks)
- The customer_message should sound authentic and match the scenario
- return_rate must equal total_returns / total_orders
- Ensure all fields are present and correctly typed
"""


class CaseBuilder:
    """Converts natural language return descriptions into structured ReturnRequest objects."""

    def __init__(self, nova: NovaClient) -> None:
        self.nova = nova

    def build_from_text(self, description: str) -> ReturnRequest:
        """Parse a natural language description into a ReturnRequest via Nova."""
        now = datetime.utcnow()
        user_prompt = (
            f"Current date/time: {now.isoformat()}Z\n\n"
            f"Generate a complete return case JSON for this scenario:\n\n"
            f"{description}\n\n"
            "Respond ONLY with valid JSON matching the schema above."
        )

        try:
            result = self.nova.reason_json(
                CASE_BUILDER_SYSTEM_PROMPT,
                user_prompt,
                temperature=0.3,
                max_tokens=2048,
            )
            return self._parse_nova_response(result)
        except Exception as exc:
            logger.warning("Nova case parsing failed, using heuristic builder: %s", exc)
            return self._heuristic_build(description)

    def _parse_nova_response(self, data: dict[str, Any]) -> ReturnRequest:
        """Convert Nova's JSON output into a validated ReturnRequest."""
        if "raw_response" in data:
            raise ValueError("Nova returned unparseable response")

        account = AccountProfile(
            account_id=data["account"]["account_id"],
            account_age_days=data["account"]["account_age_days"],
            total_orders=data["account"]["total_orders"],
            total_returns=data["account"]["total_returns"],
            return_rate=data["account"]["return_rate"],
            high_value_return_rate=data["account"]["high_value_return_rate"],
            linked_addresses=data["account"].get("linked_addresses", []),
            linked_payment_methods=data["account"].get("linked_payment_methods", 1),
            previous_fraud_flags=data["account"].get("previous_fraud_flags", 0),
            category_return_concentration=data["account"].get(
                "category_return_concentration", {}
            ),
        )

        item = OrderItem(
            asin=data["item"]["asin"],
            title=data["item"]["title"],
            category=data["item"]["category"],
            price=float(data["item"]["price"]),
            serial_number=data["item"].get("serial_number"),
            weight_grams=float(data["item"].get("weight_grams", 0)) or None,
        )

        events = []
        for ev in data.get("shipment_events", []):
            events.append(ShipmentEvent(
                event_type=ev["event_type"],
                timestamp=datetime.fromisoformat(ev["timestamp"].replace("Z", "+00:00")),
                carrier=ev["carrier"],
                tracking_id=ev["tracking_id"],
                weight_grams=ev.get("weight_grams"),
                location=ev.get("location"),
            ))

        return ReturnRequest(
            order_id=data.get("order_id", f"ORD-LIVE-{id(data) % 10000:04d}"),
            account=account,
            item=item,
            return_reason=data.get("return_reason", "Customer initiated return"),
            return_submitted_at=datetime.fromisoformat(
                data["return_submitted_at"].replace("Z", "+00:00")
            ),
            order_delivered_at=datetime.fromisoformat(
                data["order_delivered_at"].replace("Z", "+00:00")
            ),
            shipment_events=events,
            return_images=data.get("return_images", []),
            customer_message=data.get("customer_message"),
        )

    def _heuristic_build(self, description: str) -> ReturnRequest:
        """Fallback: build a plausible ReturnRequest from keyword analysis."""
        desc_lower = description.lower()
        now = datetime.utcnow()

        item_title, category, price, weight = self._extract_product_info(desc_lower)

        is_fraud_scenario = any(kw in desc_lower for kw in [
            "fraud", "scam", "empty", "counterfeit", "swap", "fake",
            "stolen", "mismatch", "suspicious", "gaming", "abuse",
        ])

        is_inr = any(kw in desc_lower for kw in [
            "not received", "never received", "didn't arrive", "lost",
        ])

        rapid = any(kw in desc_lower for kw in [
            "same day", "immediately", "right away", "next day", "within hours",
        ])
        hours_offset = 6 if rapid else 120

        if is_fraud_scenario:
            account = AccountProfile(
                account_id=f"ACCT-{abs(hash(description)) % 9000 + 1000}",
                account_age_days=45,
                total_orders=12,
                total_returns=5,
                return_rate=0.42,
                high_value_return_rate=0.55,
                linked_addresses=["123 Main St", "456 Elm St", "789 Oak Ave"],
                linked_payment_methods=3,
                previous_fraud_flags=1,
                category_return_concentration={category: 0.80},
            )
        else:
            account = AccountProfile(
                account_id=f"ACCT-{abs(hash(description)) % 9000 + 1000}",
                account_age_days=800,
                total_orders=120,
                total_returns=6,
                return_rate=0.05,
                high_value_return_rate=0.02,
                linked_addresses=["300 Trusted Ave"],
                linked_payment_methods=1,
                previous_fraud_flags=0,
                category_return_concentration={category: 0.30, "Books": 0.40},
            )

        item = OrderItem(
            asin=f"B0{''.join(str(abs(hash(item_title + str(i))) % 10) for i in range(8))}",
            title=item_title,
            category=category,
            price=price,
            serial_number=f"SN-{category[:4].upper()}-{abs(hash(item_title)) % 90000 + 10000}",
            weight_grams=weight,
        )

        delivered_at = now - timedelta(hours=hours_offset)
        submitted_at = now

        events = [
            ShipmentEvent(
                event_type="delivered",
                timestamp=delivered_at,
                carrier="UPS",
                tracking_id=f"1Z{abs(hash(description)) % 10**16:016d}",
                weight_grams=weight * 1.3,
                location="Front Door",
            ),
        ]

        if not is_inr:
            return_weight = weight * 0.20 if "empty" in desc_lower else weight * 0.98
            events.append(ShipmentEvent(
                event_type="return_received",
                timestamp=now + timedelta(days=2),
                carrier="UPS",
                tracking_id=f"1Z{abs(hash(description + 'ret')) % 10**16:016d}",
                weight_grams=return_weight,
                location="Amazon FC",
            ))

        reason = "Item not received" if is_inr else "Item arrived damaged"
        if "defective" in desc_lower:
            reason = "Item defective or doesn't work"
        elif "wrong" in desc_lower or "not as described" in desc_lower:
            reason = "Item not as described"
        elif "changed my mind" in desc_lower:
            reason = "Changed my mind"

        customer_msg = description.strip()
        if len(customer_msg) < 10:
            customer_msg = f"Returning this item: {description}"

        return ReturnRequest(
            order_id=f"ORD-LIVE-{abs(hash(description)) % 9000 + 1000}",
            account=account,
            item=item,
            return_reason=reason,
            return_submitted_at=submitted_at,
            order_delivered_at=delivered_at,
            shipment_events=events,
            return_images=[],
            customer_message=customer_msg,
        )

    @staticmethod
    def _extract_product_info(desc: str) -> tuple[str, str, float, float]:
        """Extract product name, category, estimated price, and weight from description."""
        product_db = {
            "iphone": ("Apple iPhone 15 Pro Max 256GB", "Smartphones", 1199.00, 221),
            "iphone 15": ("Apple iPhone 15 Pro Max 256GB", "Smartphones", 1199.00, 221),
            "iphone 16": ("Apple iPhone 16 Pro Max 256GB", "Smartphones", 1299.00, 227),
            "samsung": ("Samsung Galaxy S24 Ultra 256GB", "Smartphones", 1299.99, 232),
            "galaxy": ("Samsung Galaxy S24 Ultra 256GB", "Smartphones", 1299.99, 232),
            "pixel": ("Google Pixel 9 Pro 128GB", "Smartphones", 999.00, 199),
            "macbook": ("Apple MacBook Air 15-inch M3 256GB", "Laptops", 1299.00, 1510),
            "laptop": ("Dell XPS 15 Laptop 512GB", "Laptops", 1499.99, 1860),
            "ps5": ("PlayStation 5 Digital Edition", "Gaming Consoles", 449.99, 3900),
            "playstation": ("PlayStation 5 Digital Edition", "Gaming Consoles", 449.99, 3900),
            "xbox": ("Xbox Series X 1TB Console", "Gaming Consoles", 499.99, 4450),
            "switch": ("Nintendo Switch OLED Model", "Gaming Consoles", 349.99, 420),
            "airpods": ("Apple AirPods Pro 2nd Gen USB-C", "Audio Equipment", 249.00, 51),
            "headphones": ("Sony WH-1000XM5 Wireless Headphones", "Audio Equipment", 348.00, 250),
            "camera": ("Sony Alpha a7 IV Mirrorless Camera", "Cameras", 2498.00, 658),
            "gopro": ("GoPro HERO12 Black", "Cameras", 399.99, 154),
            "ipad": ("Apple iPad Pro 11-inch M4 256GB", "Tablets", 999.00, 444),
            "tablet": ("Samsung Galaxy Tab S9 FE 128GB", "Tablets", 349.99, 523),
            "kindle": ("Amazon Kindle Paperwhite 16GB", "Tablets", 149.99, 205),
            "watch": ("Apple Watch Ultra 2", "Wearables", 799.00, 61),
            "tv": ("Samsung 65-inch OLED 4K Smart TV", "Televisions", 1799.99, 22000),
            "monitor": ("LG 27-inch 4K UHD Monitor", "Monitors", 449.99, 6200),
            "keyboard": ("Logitech MX Keys S Wireless Keyboard", "Computer Accessories", 109.99, 810),
            "mouse": ("Logitech MX Master 3S", "Computer Accessories", 99.99, 141),
            "speaker": ("Sonos Era 300 Smart Speaker", "Audio Equipment", 449.00, 4470),
            "earbuds": ("Apple AirPods Pro 2nd Gen USB-C", "Audio Equipment", 249.00, 51),
            "drone": ("DJI Mini 4 Pro Drone", "Cameras", 759.00, 249),
            "gpu": ("NVIDIA GeForce RTX 4090 24GB", "Computer Components", 1599.99, 2350),
            "graphics card": ("NVIDIA GeForce RTX 4090 24GB", "Computer Components", 1599.99, 2350),
        }

        for keyword, info in product_db.items():
            if keyword in desc:
                return info

        return ("Electronics Item", "Consumer Electronics", 299.99, 500)
