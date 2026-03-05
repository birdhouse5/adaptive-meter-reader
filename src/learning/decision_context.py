"""Decision Agent context builder — assembles context from the knowledge base.

Provides the Decision Agent with device expectations, proven instructions,
consistency signals, and calibration data for informed routing decisions.

See docs/architecture.md section 8 (Two-Agent Prompt Architecture).
"""

import json
from typing import Any

from src.data.vector_store import KnowledgeBase
from src.learning.consistency import ConsistencyChecker
from src.learning.expectations import DeviceExpectations


class DecisionContextBuilder:
    """Assembles Decision Agent context from all knowledge sources.

    See docs/architecture.md section 8.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase | None = None,
        expectations: DeviceExpectations | None = None,
        consistency: ConsistencyChecker | None = None,
    ) -> None:
        self.kb = knowledge_base or KnowledgeBase()
        self.expectations = expectations or DeviceExpectations()
        self.consistency = consistency or ConsistencyChecker()

    async def build(
        self,
        extraction: dict[str, Any],
        turn_number: int = 1,
        turn_history: list[dict] | None = None,
    ) -> str:
        """Build context string for the Decision Agent."""
        parts: list[str] = []

        device_type = extraction.get("device_type", "unknown")
        extracted_fields = extraction.get("extracted_fields", {})

        # 1. Device expectations
        expected = self.expectations.get_expectations(device_type)
        if expected:
            parts.append(
                f"Device expectations for '{device_type}': {json.dumps(expected)}"
            )

        # 2. Proven guidance from similar situations
        situation_desc = self._describe_situation(extraction)
        similar = self.kb.find_similar_interactions(situation_desc, n_results=3)
        if similar:
            guidance_lines = []
            for s in similar:
                guidance = s.get("guidance_text", "")
                outcome = s.get("outcome", "")
                turns = s.get("turns_to_success", "?")
                if guidance:
                    guidance_lines.append(
                        f'  - "{guidance}" (outcome: {outcome}, '
                        f"resolved in {turns} turn(s))"
                    )
            if guidance_lines:
                parts.append(
                    "Proven guidance for similar situations:\n"
                    + "\n".join(guidance_lines)
                )

        # 3. Consistency signals
        device_id = self._extract_device_identifier(extracted_fields)
        if device_id:
            violations = self.consistency.check(
                device_identifier=device_id,
                device_type=device_type,
                extracted_fields=extracted_fields,
            )
            if violations:
                parts.append("Consistency warnings: " + "; ".join(violations))

        return "\n\n".join(parts) if parts else ""

    @staticmethod
    def _describe_situation(extraction: dict[str, Any]) -> str:
        """Create a text description of the current extraction situation."""
        device_type = extraction.get("device_type", "unknown")
        quality = extraction.get("image_quality", {})
        issues = quality.get("issues", [])
        usability = quality.get("overall_usability", 0)

        parts = [f"{device_type} device"]
        if issues:
            parts.append(f"issues: {', '.join(issues)}")
        if usability < 0.5:
            parts.append(f"low usability ({usability:.2f})")
        return ", ".join(parts)

    @staticmethod
    def _extract_device_identifier(extracted_fields: dict) -> str | None:
        """Try to find a device identifier from extracted fields."""
        for key in ("serial_number", "identifier", "device_id", "id"):
            field = extracted_fields.get(key)
            if isinstance(field, dict):
                val = field.get("value")
                if val:
                    return str(val)
            elif field:
                return str(field)
        return None
