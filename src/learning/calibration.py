"""Confidence calibration — tracks model confidence vs actual accuracy.

Records each field extraction's confidence and whether it was correct after
confirmation. Reports calibration data per device type so the Decision Agent
can make better accept/reject decisions.

See docs/architecture.md section 5 (Optimization Problem).
"""

from typing import Any

from src.data import database as db


class CalibrationTracker:
    """Tracks model confidence vs correctness per device type.

    See docs/architecture.md section 5.
    """

    def record(
        self,
        device_type: str,
        field_name: str,
        model_confidence: float,
        was_correct: bool,
        db_path: Any = None,
    ) -> None:
        """Record a single calibration data point."""
        db.insert_calibration_data(
            device_type=device_type,
            field_name=field_name,
            model_confidence=model_confidence,
            was_correct=was_correct,
            db_path=db_path,
        )

    def get_calibration(
        self,
        device_type: str | None = None,
        db_path: Any = None,
    ) -> dict[str, Any]:
        """Return calibration summary: binned confidence vs accuracy."""
        data = db.get_calibration_data(device_type=device_type, db_path=db_path)
        if not data:
            return {"total_points": 0, "bins": {}}

        # Bin by confidence range
        bins: dict[str, dict] = {}
        for row in data:
            conf = row["model_confidence"]
            correct = bool(row["was_correct"])

            # Bin into 0.0-0.2, 0.2-0.4, ... 0.8-1.0
            bin_idx = min(int(conf * 5), 4)
            bin_label = f"{bin_idx * 0.2:.1f}-{(bin_idx + 1) * 0.2:.1f}"

            if bin_label not in bins:
                bins[bin_label] = {"total": 0, "correct": 0}
            bins[bin_label]["total"] += 1
            if correct:
                bins[bin_label]["correct"] += 1

        # Calculate accuracy per bin
        for bin_label, counts in bins.items():
            counts["accuracy"] = (
                round(counts["correct"] / counts["total"], 2)
                if counts["total"] > 0
                else 0.0
            )

        return {"total_points": len(data), "bins": bins}
