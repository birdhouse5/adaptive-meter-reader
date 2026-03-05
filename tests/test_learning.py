"""Tests for the adaptive learning layer: knowledge base, expectations,
consistency, calibration, prompt builder, decision context, session processor."""

import pytest

from src.data.database import (
    create_session,
    get_confirmed_readings,
    get_device_history,
    get_instruction_effectiveness,
    get_stats,
    init_db,
    insert_confirmed_reading,
    insert_device_history,
    insert_turn,
    update_session,
)
from src.data.vector_store import KnowledgeBase
from src.learning.calibration import CalibrationTracker
from src.learning.consistency import ConsistencyChecker
from src.learning.expectations import DeviceExpectations
from src.learning.prompt_builder import VisionPromptBuilder
from src.learning.session_store import SessionProcessor


# -- KnowledgeBase (ChromaDB) --------------------------------------------------


def test_knowledge_base_confirmed_images(tmp_path):
    kb = KnowledgeBase(persist_dir=tmp_path / "chroma")
    assert kb.stats["confirmed_images"] == 0

    kb.add_confirmed_image(
        image_id="img1",
        description="White Brunata HCA on radiator, well lit",
        device_type="brunata_hca",
        confirmed_fields='{"serial_number": "BH-2847", "display_value": "4521"}',
    )
    assert kb.stats["confirmed_images"] == 1

    results = kb.find_similar_images("brunata HCA device", n_results=1)
    assert len(results) == 1
    assert results[0]["device_type"] == "brunata_hca"


def test_knowledge_base_interaction_patterns(tmp_path):
    kb = KnowledgeBase(persist_dir=tmp_path / "chroma")

    kb.add_interaction_pattern(
        interaction_id="i1",
        situation_description="dark image of brunata_hca, display not readable",
        guidance_text="Use your phone flashlight.",
        outcome="success",
        turns_to_success=1,
        device_type="brunata_hca",
        primary_issue="low_brightness",
    )

    results = kb.find_similar_interactions("dark brunata in basement", n_results=1)
    assert len(results) == 1
    assert "flashlight" in results[0]["guidance_text"].lower()


def test_knowledge_base_correction_patterns(tmp_path):
    kb = KnowledgeBase(persist_dir=tmp_path / "chroma")

    kb.add_correction_pattern(
        correction_id="c1",
        error_description="brunata_hca: model read '1' for display but correct was '7'",
        device_type="brunata_hca",
        field_name="display_value",
        original_value="1",
        corrected_value="7",
    )

    results = kb.find_similar_corrections("brunata display digit", n_results=1)
    assert len(results) == 1
    assert results[0]["original_value"] == "1"
    assert results[0]["corrected_value"] == "7"


def test_knowledge_base_empty_queries(tmp_path):
    kb = KnowledgeBase(persist_dir=tmp_path / "chroma_empty")
    assert kb.find_similar_images("anything") == []
    assert kb.find_similar_interactions("anything") == []
    assert kb.find_similar_corrections("anything") == []


# -- Database: confirmed_readings, device_history, calibration -----------------


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test.db"
    init_db(path)
    return path


def test_confirmed_readings_crud(db_path):
    session_id = create_session(db_path=db_path)
    reading_id = insert_confirmed_reading(
        session_id=session_id,
        device_type="brunata_hca",
        confirmed_fields={"serial_number": "BH-2847", "display_value": "4521"},
        original_fields={"serial_number": {"value": "BH-2847", "confidence": 0.9}},
        was_corrected=False,
        db_path=db_path,
    )
    assert reading_id > 0

    readings = get_confirmed_readings(device_type="brunata_hca", db_path=db_path)
    assert len(readings) == 1
    assert readings[0]["device_type"] == "brunata_hca"
    assert readings[0]["confirmed_fields"]["serial_number"] == "BH-2847"


def test_device_history_crud(db_path):
    insert_device_history(
        device_identifier="BH-2847",
        device_type="brunata_hca",
        reading_value="4521",
        reading_unit="kWh",
        db_path=db_path,
    )

    history = get_device_history("BH-2847", db_path=db_path)
    assert len(history) == 1
    assert history[0]["reading_value"] == "4521"


def test_stats_includes_confirmation(db_path):
    session_id = create_session(db_path=db_path)
    update_session(session_id, status="completed", total_turns=1, db_path=db_path)
    insert_confirmed_reading(
        session_id=session_id,
        device_type="brunata_hca",
        confirmed_fields={"display_value": "4521"},
        original_fields={},
        was_corrected=True,
        db_path=db_path,
    )

    stats = get_stats(db_path=db_path)
    assert stats["confirmation_stats"]["total_confirmed"] == 1
    assert stats["confirmation_stats"]["total_corrected"] == 1


# -- DeviceExpectations --------------------------------------------------------


def test_expectations_cold_start(db_path):
    import src.data.database

    original = src.data.database.DATABASE_PATH
    src.data.database.DATABASE_PATH = db_path
    try:
        exp = DeviceExpectations()
        result = exp.get_expectations("brunata_hca")
        assert result == {}
    finally:
        src.data.database.DATABASE_PATH = original


def test_expectations_with_data(db_path):
    import src.data.database

    original = src.data.database.DATABASE_PATH
    src.data.database.DATABASE_PATH = db_path
    try:
        # Add confirmed readings
        for i in range(5):
            s_id = create_session(db_path=db_path)
            insert_confirmed_reading(
                session_id=s_id,
                device_type="brunata_hca",
                confirmed_fields={
                    "serial_number": f"BH-{i}",
                    "display_value": f"{i}00",
                },
                original_fields={},
                db_path=db_path,
            )

        exp = DeviceExpectations()
        result = exp.get_expectations("brunata_hca")
        assert result["total_confirmed"] == 5
        assert result["fields"]["serial_number"]["expected"] is True
        assert result["fields"]["display_value"]["expected"] is True
    finally:
        src.data.database.DATABASE_PATH = original


# -- ConsistencyChecker --------------------------------------------------------


def test_consistency_no_history(db_path):
    import src.data.database

    original = src.data.database.DATABASE_PATH
    src.data.database.DATABASE_PATH = db_path
    try:
        checker = ConsistencyChecker()
        violations = checker.check(
            device_identifier="BH-2847",
            device_type="brunata_hca",
            extracted_fields={"display_value": {"value": "4521", "confidence": 0.9}},
        )
        assert violations == []
    finally:
        src.data.database.DATABASE_PATH = original


def test_consistency_reading_decreased(db_path):
    import src.data.database

    original = src.data.database.DATABASE_PATH
    src.data.database.DATABASE_PATH = db_path
    try:
        insert_device_history(
            device_identifier="BH-2847",
            device_type="brunata_hca",
            reading_value="5000",
            db_path=db_path,
        )

        checker = ConsistencyChecker()
        violations = checker.check(
            device_identifier="BH-2847",
            device_type="brunata_hca",
            extracted_fields={"display_value": {"value": "3000", "confidence": 0.9}},
        )
        assert len(violations) >= 1
        assert "decreased" in violations[0].lower()
    finally:
        src.data.database.DATABASE_PATH = original


# -- CalibrationTracker --------------------------------------------------------


def test_calibration_tracker(db_path):
    import src.data.database

    original = src.data.database.DATABASE_PATH
    src.data.database.DATABASE_PATH = db_path
    try:
        tracker = CalibrationTracker()
        tracker.record("brunata_hca", "serial_number", 0.9, True)
        tracker.record("brunata_hca", "serial_number", 0.9, True)
        tracker.record("brunata_hca", "serial_number", 0.3, False)

        cal = tracker.get_calibration("brunata_hca")
        assert cal["total_points"] == 3
        assert len(cal["bins"]) >= 1
    finally:
        src.data.database.DATABASE_PATH = original


# -- VisionPromptBuilder -------------------------------------------------------


async def test_prompt_builder_base_only(tmp_path):
    kb = KnowledgeBase(persist_dir=tmp_path / "chroma")
    builder = VisionPromptBuilder(knowledge_base=kb)

    prompt = await builder.build()
    assert "device_type" in prompt
    assert "extracted_fields" in prompt
    # No enrichment without description
    assert "confirmed examples" not in prompt.lower()


async def test_prompt_builder_with_knowledge(tmp_path):
    kb = KnowledgeBase(persist_dir=tmp_path / "chroma")
    kb.add_confirmed_image(
        image_id="img1",
        description="dark brunata HCA image",
        device_type="brunata_hca",
        confirmed_fields='{"serial_number": "BH-2847"}',
    )
    kb.add_correction_pattern(
        correction_id="c1",
        error_description="dark brunata HCA image: digit confusion on display",
        device_type="brunata_hca",
        field_name="display_value",
        original_value="1",
        corrected_value="7",
    )

    builder = VisionPromptBuilder(knowledge_base=kb)
    prompt = await builder.build("brunata HCA dark image")

    assert "confirmed examples" in prompt.lower()
    assert "brunata_hca" in prompt
    assert "errors to watch" in prompt.lower()


# -- DecisionContextBuilder ----------------------------------------------------


async def test_decision_context_builder(tmp_path, db_path):
    import src.data.database

    original = src.data.database.DATABASE_PATH
    src.data.database.DATABASE_PATH = db_path
    try:
        from src.learning.decision_context import DecisionContextBuilder

        kb = KnowledgeBase(persist_dir=tmp_path / "chroma")
        exp = DeviceExpectations()
        consistency = ConsistencyChecker()

        builder = DecisionContextBuilder(kb, exp, consistency)
        context = await builder.build(
            extraction={
                "device_type": "brunata_hca",
                "extracted_fields": {},
                "image_quality": {"overall_usability": 0.8, "issues": []},
            },
            turn_number=1,
        )

        # Cold start: context may be empty
        assert isinstance(context, str)
    finally:
        src.data.database.DATABASE_PATH = original


# -- SessionProcessor ----------------------------------------------------------


def _create_two_turn_session(db_path):
    """Helper: create a session with 2 turns (retry → sufficient)."""
    session_id = create_session(db_path=db_path)

    insert_turn(
        session_id=session_id,
        turn_number=1,
        routing="retry",
        extracted_fields={
            "serial_number": {"value": None, "confidence": 0.2},
            "display_value": {"value": "???", "confidence": 0.3},
        },
        image_quality={"overall_usability": 0.3, "issues": ["too dark"]},
        operator_message="The image is too dark. Please use your flashlight.",
        decision_reasoning="Display not readable due to darkness",
        issues_identified=["low_brightness"],
        description="Dark image of brunata HCA",
        db_path=db_path,
    )

    insert_turn(
        session_id=session_id,
        turn_number=2,
        routing="sufficient",
        extracted_fields={
            "serial_number": {"value": "BH-2847", "confidence": 0.90},
            "display_value": {"value": "4521", "confidence": 0.92},
        },
        image_quality={"overall_usability": 0.85, "issues": []},
        operator_message="We read serial BH-2847, value 4521. Correct?",
        decision_reasoning="All fields high confidence",
        issues_identified=[],
        description="Well-lit brunata HCA",
        db_path=db_path,
    )

    update_session(
        session_id,
        meter_type="brunata_hca",
        status="completed",
        total_turns=2,
        db_path=db_path,
    )
    return session_id


def test_session_processor(db_path, tmp_path):
    import src.data.database

    original = src.data.database.DATABASE_PATH
    src.data.database.DATABASE_PATH = db_path
    try:
        session_id = _create_two_turn_session(db_path)

        kb = KnowledgeBase(persist_dir=tmp_path / "chroma")
        calibration = CalibrationTracker()
        processor = SessionProcessor(knowledge_base=kb, calibration=calibration)

        result = processor.process_confirmed_session(
            session_id=session_id,
            confirmed_fields={"serial_number": "BH-2847", "display_value": "4521"},
            original_fields={
                "serial_number": {"value": "BH-2847", "confidence": 0.90},
                "display_value": {"value": "4521", "confidence": 0.92},
            },
        )

        assert result["session_id"] == session_id
        assert result["total_turns"] == 2
        assert result["signals_stored"] >= 1

        # Knowledge base should have entries
        assert kb.stats["confirmed_images"] == 1
        assert kb.stats["interaction_patterns"] >= 1
    finally:
        src.data.database.DATABASE_PATH = original


def test_session_processor_with_correction(db_path, tmp_path):
    import src.data.database

    original = src.data.database.DATABASE_PATH
    src.data.database.DATABASE_PATH = db_path
    try:
        session_id = _create_two_turn_session(db_path)

        kb = KnowledgeBase(persist_dir=tmp_path / "chroma")
        calibration = CalibrationTracker()
        processor = SessionProcessor(knowledge_base=kb, calibration=calibration)

        result = processor.process_confirmed_session(
            session_id=session_id,
            confirmed_fields={"serial_number": "BH-2847", "display_value": "4522"},
            original_fields={
                "serial_number": {"value": "BH-2847", "confidence": 0.90},
                "display_value": {"value": "4521", "confidence": 0.92},
            },
            was_corrected=True,
            correction_details={
                "display_value": {"original": "4521", "corrected": "4522"},
            },
        )

        assert result["was_corrected"] is True
        # Should have correction pattern stored
        assert kb.stats["correction_patterns"] >= 1
    finally:
        src.data.database.DATABASE_PATH = original


def test_session_processor_updates_effectiveness(db_path, tmp_path):
    import src.data.database

    original = src.data.database.DATABASE_PATH
    src.data.database.DATABASE_PATH = db_path
    try:
        session_id = _create_two_turn_session(db_path)

        kb = KnowledgeBase(persist_dir=tmp_path / "chroma")
        calibration = CalibrationTracker()
        processor = SessionProcessor(knowledge_base=kb, calibration=calibration)

        processor.process_confirmed_session(
            session_id=session_id,
            confirmed_fields={"serial_number": "BH-2847", "display_value": "4521"},
            original_fields={},
        )

        effectiveness = get_instruction_effectiveness(db_path=db_path)
        assert len(effectiveness) >= 1
    finally:
        src.data.database.DATABASE_PATH = original
