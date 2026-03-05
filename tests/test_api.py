"""Tests for the FastAPI endpoints with the 2-agent architecture."""

import importlib
from io import BytesIO
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.agents.base import AgentResult
from src.data.database import init_db
from src.main import app


@pytest.fixture(autouse=True)
def setup_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr("src.data.database.DATABASE_PATH", db_path)
    monkeypatch.setattr("src.config.DATABASE_PATH", db_path)
    init_db(db_path)


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_health(client):
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_stats_empty(client):
    resp = await client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_sessions"] == 0
    assert "knowledge_base_stats" in data
    assert "confirmation_stats" in data


async def test_sessions_empty(client):
    resp = await client.get("/api/sessions")
    assert resp.status_code == 200
    assert resp.json() == []


def _make_sufficient_result() -> AgentResult:
    return AgentResult(
        agent_name="orchestrator",
        output={
            "routing": "sufficient",
            "extracted_fields": {
                "serial_number": {"value": "BH-2847", "confidence": 0.90},
                "display_value": {"value": "4521", "confidence": 0.92},
            },
            "image_quality": {
                "overall_usability": 0.85,
                "issues": [],
                "suggestions": [],
            },
            "operator_message": "We read serial BH-2847, value 4521. Correct?",
            "decision_reasoning": "All fields high confidence",
            "issues_identified": [],
            "description": "Brunata HCA on radiator",
            "steps": [],
        },
        confidence=0.90,
    )


def _make_retry_result() -> AgentResult:
    return AgentResult(
        agent_name="orchestrator",
        output={
            "routing": "retry",
            "extracted_fields": {
                "serial_number": {"value": None, "confidence": 0.2},
            },
            "image_quality": {
                "overall_usability": 0.3,
                "issues": ["too dark"],
                "suggestions": ["use flashlight"],
            },
            "operator_message": "The image is too dark. Please use your flashlight.",
            "decision_reasoning": "Display not readable",
            "issues_identified": ["display_value unclear"],
            "description": "Dark image",
            "steps": [],
        },
        confidence=0.2,
    )


async def test_start_session_sufficient(client):
    from PIL import Image

    img = Image.new("RGB", (10, 10), color="green")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    with patch("src.api.routes._orchestrator") as mock_orch:
        mock_orch.process = AsyncMock(return_value=_make_sufficient_result())
        resp = await client.post(
            "/api/session/start",
            files={"file": ("test.jpg", buf, "image/jpeg")},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert data["routing"] == "sufficient"
    assert data["operator_message"] != ""


async def test_session_lifecycle(client):
    """Start → retry → sufficient lifecycle."""
    from PIL import Image

    img = Image.new("RGB", (10, 10), color="red")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    # Turn 1: retry
    with patch("src.api.routes._orchestrator") as mock_orch:
        mock_orch.process = AsyncMock(return_value=_make_retry_result())
        resp1 = await client.post(
            "/api/session/start",
            files={"file": ("test.jpg", buf, "image/jpeg")},
        )

    assert resp1.status_code == 200
    session_id = resp1.json()["session_id"]
    assert resp1.json()["routing"] == "retry"
    assert "flashlight" in resp1.json()["operator_message"].lower()

    # Turn 2: sufficient
    buf.seek(0)
    with patch("src.api.routes._orchestrator") as mock_orch:
        mock_orch.process = AsyncMock(return_value=_make_sufficient_result())
        resp2 = await client.post(
            f"/api/session/{session_id}/upload",
            files={"file": ("test2.jpg", buf, "image/jpeg")},
        )

    assert resp2.status_code == 200
    assert resp2.json()["routing"] == "sufficient"
    assert resp2.json()["turn_number"] == 2

    # Verify session state
    resp3 = await client.get(f"/api/session/{session_id}")
    assert resp3.status_code == 200
    session = resp3.json()
    assert session["status"] == "completed"
    assert session["total_turns"] == 2
    assert len(session["turns"]) == 2


async def test_confirm_endpoint(client):
    """Test the /confirm endpoint with and without corrections."""
    from PIL import Image

    img = Image.new("RGB", (10, 10), color="green")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    # Start session
    with patch("src.api.routes._orchestrator") as mock_orch:
        mock_orch.process = AsyncMock(return_value=_make_sufficient_result())
        resp = await client.post(
            "/api/session/start",
            files={"file": ("test.jpg", buf, "image/jpeg")},
        )
    session_id = resp.json()["session_id"]

    # Confirm without corrections
    with patch("src.api.routes._session_processor") as mock_proc:
        mock_proc.process_confirmed_session = lambda **kwargs: {
            "session_id": session_id,
            "signals_stored": 3,
        }
        resp = await client.post(
            f"/api/session/{session_id}/confirm",
            json={
                "confirmed_fields": {
                    "serial_number": "BH-2847",
                    "display_value": "4521",
                }
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == session_id
    assert data["was_corrected"] is False


async def test_confirm_with_corrections(client):
    """Test /confirm endpoint with corrections triggers learning."""
    from PIL import Image

    img = Image.new("RGB", (10, 10), color="green")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    with patch("src.api.routes._orchestrator") as mock_orch:
        mock_orch.process = AsyncMock(return_value=_make_sufficient_result())
        resp = await client.post(
            "/api/session/start",
            files={"file": ("test.jpg", buf, "image/jpeg")},
        )
    session_id = resp.json()["session_id"]

    with patch("src.api.routes._session_processor") as mock_proc:
        mock_proc.process_confirmed_session = lambda **kwargs: {
            "session_id": session_id,
            "signals_stored": 5,
        }
        resp = await client.post(
            f"/api/session/{session_id}/confirm",
            json={
                "confirmed_fields": {
                    "serial_number": "BH-2847",
                    "display_value": "4521",
                },
                "corrections": {"display_value": "4522"},
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["was_corrected"] is True


def test_dashboard_import():
    """Smoke test: dashboard modules import without error."""
    mod = importlib.import_module("dashboard.api")
    assert hasattr(mod, "API_BASE")
    importlib.import_module("dashboard.diagrams")


async def test_upload_to_nonexistent_session(client):
    from PIL import Image

    img = Image.new("RGB", (10, 10))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    with patch("src.api.routes._orchestrator", new=AsyncMock()):
        resp = await client.post(
            "/api/session/9999/upload",
            files={"file": ("test.jpg", buf, "image/jpeg")},
        )
    assert resp.status_code == 404
