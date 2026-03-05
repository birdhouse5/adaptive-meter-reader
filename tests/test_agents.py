"""Tests for the 2-agent framework: Vision Agent, Decision Agent, Orchestrator."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.base import AgentResult, BaseAgent
from src.agents.decision import DecisionAgent
from src.agents.vision import VisionAgent


# -- BaseAgent -----------------------------------------------------------------


class DummyAgent(BaseAgent):
    name = "dummy"

    async def _run(self, payload):
        return AgentResult(
            agent_name=self.name,
            output={"echo": payload},
            confidence=0.99,
        )


async def test_base_agent_timing():
    agent = DummyAgent()
    result = await agent.process({"hello": "world"})
    assert result.agent_name == "dummy"
    assert result.confidence == 0.99
    assert result.processing_time_ms > 0


async def test_base_agent_error():
    class FailAgent(BaseAgent):
        name = "fail"

        async def _run(self, payload):
            raise ValueError("boom")

    agent = FailAgent()
    with pytest.raises(ValueError, match="boom"):
        await agent.process({})


# -- VisionAgent (mocked OpenAI) ----------------------------------------------

_MOCK_VISION_RESPONSE = {
    "device_type": "heat cost allocator",
    "extracted_fields": {
        "serial_number": {"value": "BH-2847", "confidence": 0.87},
        "display_value": {"value": "4521", "confidence": 0.92},
        "reading_unit": {"value": "kWh", "confidence": 0.96},
    },
    "image_quality": {
        "overall_usability": 0.85,
        "issues": [],
        "suggestions": [],
    },
    "description": "White Brunata HCA mounted on a radiator",
}


def _make_mock_openai_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content))]
    return resp


async def test_vision_agent_mocked(tmp_path):
    from PIL import Image

    img = Image.new("RGB", (10, 10), color="red")
    img_path = tmp_path / "test.jpg"
    img.save(img_path)

    agent = VisionAgent()
    agent.client = MagicMock()
    agent.client.chat.completions.create = AsyncMock(
        return_value=_make_mock_openai_response(json.dumps(_MOCK_VISION_RESPONSE))
    )

    result = await agent.process({"image_path": str(img_path)})

    assert result.agent_name == "vision"
    assert result.output["device_type"] == "heat cost allocator"
    assert result.output["extracted_fields"]["serial_number"]["value"] == "BH-2847"
    assert result.output["extracted_fields"]["display_value"]["confidence"] == 0.92
    # Overall confidence = min of field confidences (0.87)
    assert result.confidence == pytest.approx(0.87, abs=0.01)


async def test_vision_agent_json_fallback(tmp_path):
    from PIL import Image

    img = Image.new("RGB", (10, 10), color="blue")
    img_path = tmp_path / "test.png"
    img.save(img_path)

    agent = VisionAgent()
    agent.client = MagicMock()
    agent.client.chat.completions.create = AsyncMock(
        return_value=_make_mock_openai_response("I cannot read this meter.")
    )

    result = await agent.process({"image_path": str(img_path)})
    assert result.output["device_type"] == "unknown"
    assert result.confidence == 0.0


async def test_vision_agent_prompt_enrichment(tmp_path):
    """Prompt enrichment is appended when present and no prompt_builder."""
    from PIL import Image

    img = Image.new("RGB", (10, 10), color="green")
    img_path = tmp_path / "test.jpg"
    img.save(img_path)

    agent = VisionAgent()
    agent.client = MagicMock()
    agent.client.chat.completions.create = AsyncMock(
        return_value=_make_mock_openai_response(json.dumps(_MOCK_VISION_RESPONSE))
    )

    await agent.process(
        {
            "image_path": str(img_path),
            "prompt_enrichment": "Focus on the display digits",
        }
    )

    # Verify the system prompt includes the enrichment
    call_args = agent.client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    system_content = messages[0]["content"]
    assert "Focus on the display digits" in system_content


# -- DecisionAgent (mocked OpenAI) --------------------------------------------

_MOCK_SUFFICIENT_DECISION = {
    "routing": "sufficient",
    "reasoning": "All fields extracted with high confidence",
    "operator_message": "We read serial BH-2847, value 4521 kWh. Is this correct?",
    "issues_identified": [],
    "guidance_focus": None,
}

_MOCK_RETRY_DECISION = {
    "routing": "retry",
    "reasoning": "Display value unclear due to shadow",
    "operator_message": "The display is hard to read. Please use your flashlight.",
    "issues_identified": ["display_value unclear"],
    "guidance_focus": "lighting",
}


async def test_decision_agent_sufficient():
    agent = DecisionAgent()
    agent.client = MagicMock()
    agent.client.chat.completions.create = AsyncMock(
        return_value=_make_mock_openai_response(json.dumps(_MOCK_SUFFICIENT_DECISION))
    )

    result = await agent.process(
        {
            "extraction": _MOCK_VISION_RESPONSE,
            "turn_number": 1,
        }
    )

    assert result.output["routing"] == "sufficient"
    assert "BH-2847" in result.output["operator_message"]


async def test_decision_agent_retry():
    agent = DecisionAgent()
    agent.client = MagicMock()
    agent.client.chat.completions.create = AsyncMock(
        return_value=_make_mock_openai_response(json.dumps(_MOCK_RETRY_DECISION))
    )

    result = await agent.process(
        {
            "extraction": _MOCK_VISION_RESPONSE,
            "turn_number": 1,
        }
    )

    assert result.output["routing"] == "retry"
    assert "flashlight" in result.output["operator_message"].lower()


async def test_decision_agent_max_turns_escalate():
    """At max turns, retry should be forced to escalate."""
    agent = DecisionAgent()
    agent.client = MagicMock()
    agent.client.chat.completions.create = AsyncMock(
        return_value=_make_mock_openai_response(json.dumps(_MOCK_RETRY_DECISION))
    )

    result = await agent.process(
        {
            "extraction": _MOCK_VISION_RESPONSE,
            "turn_number": 5,
        }
    )

    assert result.output["routing"] == "escalate"


async def test_decision_agent_includes_context():
    """Decision context is included in LLM prompt when provided."""
    mock_builder = AsyncMock()
    mock_builder.build = AsyncMock(return_value="Device expects serial + reading")

    agent = DecisionAgent(decision_context_builder=mock_builder)
    agent.client = MagicMock()
    agent.client.chat.completions.create = AsyncMock(
        return_value=_make_mock_openai_response(json.dumps(_MOCK_SUFFICIENT_DECISION))
    )

    await agent.process(
        {
            "extraction": _MOCK_VISION_RESPONSE,
            "turn_number": 1,
        }
    )

    # Verify context builder was called
    mock_builder.build.assert_called_once()

    # Verify context appears in the user message
    call_args = agent.client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    user_content = messages[1]["content"]
    assert "Device expects serial + reading" in user_content


# -- Orchestrator (mocked agents) ---------------------------------------------


async def test_orchestrator_sufficient_flow(tmp_path):
    """Vision → Decision → sufficient routing."""
    from src.agents.orchestrator import Orchestrator

    from PIL import Image

    img = Image.new("RGB", (10, 10), color="green")
    img_path = tmp_path / "test.jpg"
    img.save(img_path)

    vision = VisionAgent()
    vision.client = MagicMock()
    vision.client.chat.completions.create = AsyncMock(
        return_value=_make_mock_openai_response(json.dumps(_MOCK_VISION_RESPONSE))
    )

    decision = DecisionAgent()
    decision.client = MagicMock()
    decision.client.chat.completions.create = AsyncMock(
        return_value=_make_mock_openai_response(json.dumps(_MOCK_SUFFICIENT_DECISION))
    )

    orch = Orchestrator(vision_agent=vision, decision_agent=decision)
    result = await orch.process({"image_path": str(img_path), "turn_number": 1})

    assert result.output["routing"] == "sufficient"
    assert result.output["extracted_fields"]["serial_number"]["value"] == "BH-2847"
    assert "BH-2847" in result.output["operator_message"]


async def test_orchestrator_retry_flow(tmp_path):
    """Vision → Decision → retry with guidance."""
    from src.agents.orchestrator import Orchestrator

    from PIL import Image

    img = Image.new("RGB", (10, 10), color="green")
    img_path = tmp_path / "test.jpg"
    img.save(img_path)

    vision = VisionAgent()
    vision.client = MagicMock()
    vision.client.chat.completions.create = AsyncMock(
        return_value=_make_mock_openai_response(json.dumps(_MOCK_VISION_RESPONSE))
    )

    decision = DecisionAgent()
    decision.client = MagicMock()
    decision.client.chat.completions.create = AsyncMock(
        return_value=_make_mock_openai_response(json.dumps(_MOCK_RETRY_DECISION))
    )

    orch = Orchestrator(vision_agent=vision, decision_agent=decision)
    result = await orch.process({"image_path": str(img_path), "turn_number": 1})

    assert result.output["routing"] == "retry"
    assert result.output["operator_message"] != ""
    assert len(result.output["issues_identified"]) > 0
