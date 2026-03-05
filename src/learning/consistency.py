"""Cross-session consistency checks for device readings.

Validates new readings against device history — serial stability, reading
monotonicity, rate bounds. Flags physics-violating readings as likely errors.

See docs/architecture.md section 4 (Ground Truth Validation Tiers), Tier 2.
"""

from typing import Any

from src.data import database as db


class ConsistencyChecker:
    """Checks new readings against device history for consistency.

    See docs/architecture.md section 4, Tier 2.
    """

    def check(
        self,
        device_identifier: str,
        device_type: str,
        extracted_fields: dict[str, Any],
        db_path: Any = None,
    ) -> list[str]:
        """Return list of consistency violations (empty = consistent)."""
        violations: list[str] = []

        history = db.get_device_history(device_identifier, limit=10, db_path=db_path)
        if not history:
            return violations

        # Check reading monotonicity (counters should go up)
        new_value = self._extract_numeric_reading(extracted_fields)
        if new_value is not None:
            last = history[0]
            last_value = self._parse_numeric(last.get("reading_value"))
            if last_value is not None and new_value < last_value:
                violations.append(
                    f"Reading decreased: {new_value} < previous {last_value} "
                    f"(counters normally increase)"
                )

            # Rate bounds — flag implausible jumps
            if last_value is not None and last_value > 0:
                ratio = new_value / last_value
                if ratio > 10:
                    violations.append(
                        f"Reading jumped {ratio:.1f}x from {last_value} to "
                        f"{new_value} (implausible increase)"
                    )

        # Check device type stability
        past_types = {h.get("device_type") for h in history}
        if device_type != "unknown" and past_types and device_type not in past_types:
            violations.append(
                f"Device type changed to '{device_type}' but previously "
                f"identified as {past_types}"
            )

        return violations

    @staticmethod
    def _extract_numeric_reading(extracted_fields: dict) -> float | None:
        """Try to extract a numeric reading value from extracted fields."""
        for key in ("display_value", "reading_value", "current_reading"):
            field = extracted_fields.get(key)
            if isinstance(field, dict):
                val = field.get("value")
            else:
                val = field
            parsed = ConsistencyChecker._parse_numeric(val)
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _parse_numeric(value: Any) -> float | None:
        """Parse a value as a float, returning None on failure."""
        if value is None:
            return None
        try:
            return float(str(value).replace(",", "."))
        except (ValueError, TypeError):
            return None
