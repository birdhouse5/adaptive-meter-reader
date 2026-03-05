"""FastAPI endpoints for the adaptive meter reading system.

Provides session management, image upload, reading confirmation, and stats.

See docs/architecture.md section 2 (Session Lifecycle) for how these endpoints
map to the multi-turn interaction flow.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.agents.decision import DecisionAgent
from src.agents.orchestrator import Orchestrator
from src.agents.vision import VisionAgent
from src.data import database as db
from src.data.vector_store import KnowledgeBase
from src.learning.calibration import CalibrationTracker
from src.learning.consistency import ConsistencyChecker
from src.learning.decision_context import DecisionContextBuilder
from src.learning.expectations import DeviceExpectations
from src.learning.prompt_builder import VisionPromptBuilder
from src.learning.session_store import SessionProcessor

router = APIRouter()

# Shared instances — initialised via init_services()
_orchestrator: Orchestrator | None = None
_session_processor: SessionProcessor | None = None
_knowledge_base: KnowledgeBase | None = None
_calibration: CalibrationTracker | None = None


def init_services() -> None:
    """Wire up all shared service instances. Called once on app startup."""
    global _orchestrator, _session_processor, _knowledge_base, _calibration

    _knowledge_base = KnowledgeBase()
    expectations = DeviceExpectations()
    consistency = ConsistencyChecker()
    _calibration = CalibrationTracker()

    vision_prompt_builder = VisionPromptBuilder(_knowledge_base)
    decision_context_builder = DecisionContextBuilder(
        _knowledge_base, expectations, consistency
    )

    vision = VisionAgent(prompt_builder=vision_prompt_builder)
    decision = DecisionAgent(decision_context_builder=decision_context_builder)
    _orchestrator = Orchestrator(vision_agent=vision, decision_agent=decision)
    _session_processor = SessionProcessor(_knowledge_base, _calibration)


def _save_upload(file: UploadFile) -> Path:
    """Save an uploaded file to a temp path and return it."""
    suffix = Path(file.filename or "image.jpg").suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    shutil.copyfileobj(file.file, tmp)
    tmp.close()
    return Path(tmp.name)


# -- Session endpoints ---------------------------------------------------------


@router.post("/session/start")
async def start_session(file: UploadFile = File(...)):
    """Start a new reading session with the first photo upload."""
    if _orchestrator is None:
        raise HTTPException(503, "Services not initialised")

    image_path = _save_upload(file)
    try:
        session_id = db.create_session()

        result = await _orchestrator.process(
            {"image_path": str(image_path), "turn_number": 1}
        )
        output = result.output

        # Try to find device_type from the extraction
        meter_type = output.get("device_type", "unknown")
        if meter_type == "unknown":
            # Vision Agent puts device_type at top level of its output
            steps = output.get("steps", [])
            if steps:
                first_step = steps[0] if isinstance(steps[0], dict) else {}
                meter_type = first_step.get("output", {}).get("device_type", "unknown")

        routing = output.get("routing", "retry")
        status = "active"
        if routing == "sufficient":
            status = "completed"
        elif routing == "escalate":
            status = "escalated"

        db.update_session(
            session_id, meter_type=meter_type, status=status, total_turns=1
        )

        db.insert_turn(
            session_id=session_id,
            turn_number=1,
            routing=routing,
            image_path=str(image_path),
            extracted_fields=output.get("extracted_fields"),
            image_quality=output.get("image_quality"),
            operator_message=output.get("operator_message", ""),
            decision_reasoning=output.get("decision_reasoning", ""),
            issues_identified=output.get("issues_identified"),
            description=output.get("description", ""),
        )

        db.log_agent_activity(
            "orchestrator",
            "session_started",
            f"session={session_id} type={meter_type} routing={routing}",
        )

        return {"session_id": session_id, **output}
    except Exception:
        image_path.unlink(missing_ok=True)
        raise


@router.post("/session/{session_id}/upload")
async def upload_retry(session_id: int, file: UploadFile = File(...)):
    """Upload a retry photo for an existing session."""
    if _orchestrator is None:
        raise HTTPException(503, "Services not initialised")

    session = db.get_session(session_id)
    if session is None:
        raise HTTPException(404, f"Session {session_id} not found")
    if session["status"] != "active":
        raise HTTPException(
            400, f"Session {session_id} is {session['status']}, not active"
        )

    turn_number = session["total_turns"] + 1
    image_path = _save_upload(file)

    try:
        # Build turn history for context
        previous_turns = db.get_turns_for_session(session_id)
        turn_history = [
            {
                "turn_number": t["turn_number"],
                "routing": t["routing"],
                "issues_identified": t.get("issues_identified", []),
            }
            for t in previous_turns
        ]

        result = await _orchestrator.process(
            {
                "image_path": str(image_path),
                "turn_number": turn_number,
                "turn_history": turn_history,
            }
        )
        output = result.output

        routing = output.get("routing", "retry")
        status = "active"
        if routing == "sufficient":
            status = "completed"
        elif routing == "escalate":
            status = "escalated"

        meter_type = session.get("meter_type", "unknown")

        db.update_session(
            session_id,
            meter_type=meter_type,
            status=status,
            total_turns=turn_number,
        )

        db.insert_turn(
            session_id=session_id,
            turn_number=turn_number,
            routing=routing,
            image_path=str(image_path),
            extracted_fields=output.get("extracted_fields"),
            image_quality=output.get("image_quality"),
            operator_message=output.get("operator_message", ""),
            decision_reasoning=output.get("decision_reasoning", ""),
            issues_identified=output.get("issues_identified"),
            description=output.get("description", ""),
        )

        db.log_agent_activity(
            "orchestrator",
            "turn_processed",
            f"session={session_id} turn={turn_number} routing={routing}",
        )

        return {"session_id": session_id, "turn_number": turn_number, **output}
    except Exception:
        image_path.unlink(missing_ok=True)
        raise


class ConfirmRequest(BaseModel):
    """Request body for the /confirm endpoint."""

    confirmed_fields: dict[str, Any]
    corrections: dict[str, Any] | None = None


@router.post("/session/{session_id}/confirm")
async def confirm_reading(session_id: int, body: ConfirmRequest):
    """Confirm or correct the extracted reading for a session.

    Triggers knowledge base update via SessionProcessor.
    See docs/architecture.md section 4, Tier 1 (User Confirmation).
    """
    session = db.get_session(session_id)
    if session is None:
        raise HTTPException(404, f"Session {session_id} not found")

    turns = db.get_turns_for_session(session_id)
    if not turns:
        raise HTTPException(400, "Session has no turns")

    # Determine if corrections were made
    was_corrected = body.corrections is not None and len(body.corrections) > 0
    correction_details = {}
    if was_corrected and body.corrections:
        for field_name, corrected_value in body.corrections.items():
            original = body.confirmed_fields.get(field_name)
            correction_details[field_name] = {
                "original": original,
                "corrected": corrected_value,
            }
            # Apply corrections to confirmed_fields
            body.confirmed_fields[field_name] = corrected_value

    # Get original extraction fields for calibration
    final_turn = turns[-1]
    original_fields = final_turn.get("extracted_fields", {})

    # Get image path from the final turn
    image_path = final_turn.get("image_path", "")

    # Store confirmed reading in database
    db.insert_confirmed_reading(
        session_id=session_id,
        device_type=session.get("meter_type", "unknown"),
        confirmed_fields=body.confirmed_fields,
        original_fields=original_fields,
        was_corrected=was_corrected,
        correction_details=correction_details,
        image_path=image_path,
    )

    # Update session validation status
    db.update_session(
        session_id,
        validation_status="confirmed" if not was_corrected else "corrected",
    )

    # Trigger session processing (knowledge base update)
    result = {"session_id": session_id, "was_corrected": was_corrected}
    if _session_processor:
        try:
            processing = _session_processor.process_confirmed_session(
                session_id=session_id,
                confirmed_fields=body.confirmed_fields,
                original_fields=original_fields,
                was_corrected=was_corrected,
                correction_details=correction_details,
                image_path=image_path,
            )
            result["learning"] = processing
        except Exception:
            pass  # Don't fail the confirmation if learning fails

    return result


@router.get("/session/{session_id}")
async def get_session(session_id: int):
    """Get full session details including all turns."""
    session = db.get_session(session_id)
    if session is None:
        raise HTTPException(404, f"Session {session_id} not found")
    turns = db.get_turns_for_session(session_id)
    return {**session, "turns": turns}


@router.get("/sessions")
async def list_sessions(limit: int = 50):
    """List recent sessions."""
    return db.list_sessions(limit=limit)


# -- Stats & activity ----------------------------------------------------------


@router.get("/stats")
async def get_stats():
    """Return optimization metrics and knowledge base stats."""
    stats = db.get_stats()
    effectiveness = db.get_instruction_effectiveness()
    kb_stats = _knowledge_base.stats if _knowledge_base else {}
    calibration = _calibration.get_calibration() if _calibration else {}
    return {
        **stats,
        "instruction_effectiveness": effectiveness,
        "knowledge_base_stats": kb_stats,
        "calibration": calibration,
    }


@router.get("/activity")
async def get_activity(limit: int = 100):
    return db.get_agent_activity(limit=limit)


@router.get("/health")
async def health():
    return {"status": "ok"}
