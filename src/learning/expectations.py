"""Device type field expectations — learned from confirmed readings.

Aggregates what fields are typically present for each device type, allowing
the Decision Agent to judge extraction completeness.

See docs/architecture.md section 3 (Knowledge Base Growth).
"""

from typing import Any

from src.data import database as db


class DeviceExpectations:
    """Learns what fields to expect from each device type.

    Built from confirmed_readings — discovers field schemas organically
    rather than prescribing them.

    See docs/architecture.md section 3.
    """

    def get_expectations(
        self,
        device_type: str,
        db_path: Any = None,
    ) -> dict[str, Any]:
        """Return expected fields and their frequency for a device type.

        Returns empty dict on cold start (no confirmed readings for this type).
        """
        readings = db.get_confirmed_readings(
            device_type=device_type, limit=100, db_path=db_path
        )
        if not readings:
            return {}

        field_counts: dict[str, int] = {}
        total = len(readings)

        for reading in readings:
            confirmed = reading.get("confirmed_fields", {})
            if isinstance(confirmed, str):
                import json

                confirmed = json.loads(confirmed)
            for field_name in confirmed:
                field_counts[field_name] = field_counts.get(field_name, 0) + 1

        expected_fields = {}
        for field_name, count in field_counts.items():
            frequency = count / total
            expected_fields[field_name] = {
                "frequency": round(frequency, 2),
                "seen_count": count,
                "expected": frequency >= 0.5,
            }

        return {
            "device_type": device_type,
            "total_confirmed": total,
            "fields": expected_fields,
        }
