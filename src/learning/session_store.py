"""Processes confirmed sessions into all knowledge stores.

When a session is confirmed (operator validates the reading), this module
extracts learning signals and distributes them across the knowledge base
collections, device history, and calibration data.

See docs/architecture.md section 3 (Knowledge Base Growth).
"""

import json
import logging
from typing import Any

from src.data import database as db
from src.data.vector_store import KnowledgeBase
from src.learning.calibration import CalibrationTracker

logger = logging.getLogger(__name__)


class SessionProcessor:
    """Processes confirmed sessions into all learning stores.

    See docs/architecture.md section 3.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase | None = None,
        calibration: CalibrationTracker | None = None,
    ) -> None:
        self.kb = knowledge_base or KnowledgeBase()
        self.calibration = calibration or CalibrationTracker()

    def process_confirmed_session(
        self,
        session_id: int,
        confirmed_fields: dict[str, Any],
        original_fields: dict[str, Any],
        was_corrected: bool = False,
        correction_details: dict[str, Any] | None = None,
        image_path: str = "",
    ) -> dict[str, Any]:
        """Process a confirmed session and distribute learning signals."""
        session = db.get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        turns = db.get_turns_for_session(session_id)
        if not turns:
            return {"session_id": session_id, "signals_stored": 0}

        device_type = session.get("meter_type", "unknown")
        total_turns = len(turns)
        signals_stored = 0

        # 1. Store confirmed image in confirmed_images collection
        final_turn = turns[-1]
        description = final_turn.get("description", f"{device_type} device")
        self.kb.add_confirmed_image(
            image_id=f"session_{session_id}",
            description=description,
            device_type=device_type,
            confirmed_fields=json.dumps(confirmed_fields),
            image_path=image_path,
        )
        signals_stored += 1

        # 2. Store guidance outcomes in interaction_patterns
        for turn in turns:
            message = turn.get("operator_message", "")
            if not message or turn.get("routing") == "sufficient":
                continue

            situation = self._describe_turn(turn, device_type)
            primary_issue = self._get_primary_issue(turn)
            turn_num = turn.get("turn_number", 1)
            turns_remaining = total_turns - turn_num

            self.kb.add_interaction_pattern(
                interaction_id=f"session_{session_id}_turn_{turn_num}",
                situation_description=situation,
                guidance_text=message,
                outcome="success" if total_turns > turn_num else "final",
                turns_to_success=turns_remaining,
                device_type=device_type,
                primary_issue=primary_issue,
                effectiveness_rate=1.0 if total_turns > turn_num else 0.0,
            )

            # Update instruction effectiveness in SQLite
            situation_sig = f"{device_type}:{primary_issue}"
            db.upsert_instruction_effectiveness(
                situation_signature=situation_sig,
                instruction_text=message,
                success=True,
                turns_after=turns_remaining,
            )
            signals_stored += 1

        # 3. Store correction patterns if the reading was corrected
        if was_corrected and correction_details:
            for field_name, detail in correction_details.items():
                original_val = str(detail.get("original", ""))
                corrected_val = str(detail.get("corrected", ""))
                if original_val and corrected_val:
                    self.kb.add_correction_pattern(
                        correction_id=f"session_{session_id}_{field_name}",
                        error_description=(
                            f"{device_type}: model read '{original_val}' "
                            f"for {field_name} but correct was '{corrected_val}'"
                        ),
                        device_type=device_type,
                        field_name=field_name,
                        original_value=original_val,
                        corrected_value=corrected_val,
                    )
                    signals_stored += 1

        # 4. Update device history
        device_id = self._find_device_id(confirmed_fields)
        if device_id:
            reading_val = self._find_reading_value(confirmed_fields)
            reading_unit = self._find_reading_unit(confirmed_fields)
            db.insert_device_history(
                device_identifier=device_id,
                device_type=device_type,
                reading_value=reading_val,
                reading_unit=reading_unit,
                session_id=session_id,
            )
            signals_stored += 1

        # 5. Record calibration data
        for field_name, field_data in original_fields.items():
            if isinstance(field_data, dict):
                confidence = field_data.get("confidence", 0.0)
                original_value = field_data.get("value")
            else:
                continue

            confirmed_value = confirmed_fields.get(field_name)
            if confirmed_value is None:
                continue

            # Compare original extraction to confirmed value
            was_correct = str(original_value) == str(confirmed_value)
            self.calibration.record(
                device_type=device_type,
                field_name=field_name,
                model_confidence=confidence,
                was_correct=was_correct,
            )
            signals_stored += 1

        db.log_agent_activity(
            "session_processor",
            "session_confirmed",
            f"session={session_id} turns={total_turns} "
            f"signals={signals_stored} corrected={was_corrected}",
        )

        logger.info(
            "Processed session %d: %d turns, %d signals stored, corrected=%s",
            session_id,
            total_turns,
            signals_stored,
            was_corrected,
        )

        return {
            "session_id": session_id,
            "total_turns": total_turns,
            "signals_stored": signals_stored,
            "was_corrected": was_corrected,
        }

    @staticmethod
    def _describe_turn(turn: dict, device_type: str) -> str:
        """Create a text description of a turn's situation."""
        parts = [f"{device_type} device"]

        quality = turn.get("image_quality", {})
        if isinstance(quality, dict):
            issues = quality.get("issues", [])
            if issues:
                parts.append(f"issues: {', '.join(issues)}")

        turn_issues = turn.get("issues_identified", [])
        if isinstance(turn_issues, list) and turn_issues:
            parts.append(f"identified: {', '.join(turn_issues)}")

        return ", ".join(parts)

    @staticmethod
    def _get_primary_issue(turn: dict) -> str:
        """Extract the primary issue from a turn."""
        issues = turn.get("issues_identified", [])
        if isinstance(issues, list) and issues:
            return issues[0]
        return "unknown"

    @staticmethod
    def _find_device_id(fields: dict) -> str | None:
        """Try to find a device identifier from confirmed fields."""
        for key in ("serial_number", "identifier", "device_id"):
            val = fields.get(key)
            if val:
                return str(val)
        return None

    @staticmethod
    def _find_reading_value(fields: dict) -> str | None:
        """Try to find the reading value from confirmed fields."""
        for key in ("display_value", "reading_value", "current_reading"):
            val = fields.get(key)
            if val:
                return str(val)
        return None

    @staticmethod
    def _find_reading_unit(fields: dict) -> str | None:
        """Try to find the reading unit from confirmed fields."""
        for key in ("reading_unit", "unit"):
            val = fields.get(key)
            if val:
                return str(val)
        return None
