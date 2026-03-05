"""Tests for the Decision Agent — routing, guidance, escalation."""

import json
from unittest.mock import AsyncMock, MagicMock

from src.agents.decision import DecisionAgent


def _make_mock_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content))]
    return resp


_GOOD_EXTRACTION = {
    "device_type": "heat cost allocator",
    "extracted_fields": {
        "serial_number": {"value": "BH-2847", "confidence": 0.90},
        "display_value": {"value": "4521", "confidence": 0.92},
    },
    "image_quality": {
        "overall_usability": 0.85,
        "issues": [],
        "suggestions": [],
    },
}

_POOR_EXTRACTION = {
    "device_type": "unknown",
    "extracted_fields": {
        "serial_number": {"value": None, "confidence": 0.1},
    },
    "image_quality": {
        "overall_usability": 0.2,
        "issues": ["too dark", "blurry"],
        "suggestions": ["use flashlight", "hold steady"],
    },
}


async def test_sufficient_routing():
    """Good extraction → sufficient with confirmation message."""
    agent = DecisionAgent()
    agent.client = MagicMock()
    agent.client.chat.completions.create = AsyncMock(
        return_value=_make_mock_response(
            json.dumps(
                {
                    "routing": "sufficient",
                    "reasoning": "All fields high confidence",
                    "operator_message": "Serial BH-2847, value 4521. Correct?",
                    "issues_identified": [],
                    "guidance_focus": None,
                }
            )
        )
    )

    result = await agent.process(
        {
            "extraction": _GOOD_EXTRACTION,
            "turn_number": 1,
        }
    )

    assert result.output["routing"] == "sufficient"
    assert "BH-2847" in result.output["operator_message"]
    assert result.output["issues_identified"] == []


async def test_retry_routing():
    """Poor extraction → retry with guidance."""
    agent = DecisionAgent()
    agent.client = MagicMock()
    agent.client.chat.completions.create = AsyncMock(
        return_value=_make_mock_response(
            json.dumps(
                {
                    "routing": "retry",
                    "reasoning": "Image too dark, display not readable",
                    "operator_message": "Please turn on your flashlight and retake.",
                    "issues_identified": ["too dark", "display not readable"],
                    "guidance_focus": "lighting",
                }
            )
        )
    )

    result = await agent.process(
        {
            "extraction": _POOR_EXTRACTION,
            "turn_number": 1,
        }
    )

    assert result.output["routing"] == "retry"
    assert "flashlight" in result.output["operator_message"].lower()
    assert len(result.output["issues_identified"]) > 0


async def test_max_turns_forces_escalation():
    """At max turns, retry is forced to escalate."""
    agent = DecisionAgent()
    agent.client = MagicMock()
    agent.client.chat.completions.create = AsyncMock(
        return_value=_make_mock_response(
            json.dumps(
                {
                    "routing": "retry",
                    "reasoning": "Still can't read the display",
                    "operator_message": "Please try again with better lighting.",
                    "issues_identified": ["display unclear"],
                    "guidance_focus": "lighting",
                }
            )
        )
    )

    result = await agent.process(
        {
            "extraction": _POOR_EXTRACTION,
            "turn_number": 5,  # MAX_SESSION_TURNS
        }
    )

    assert result.output["routing"] == "escalate"
    assert (
        "escalation" in result.output["reasoning"].lower()
        or "forced" in result.output["reasoning"].lower()
    )


async def test_decision_context_in_prompt():
    """Decision context from knowledge base appears in LLM prompt."""
    mock_builder = AsyncMock()
    mock_builder.build = AsyncMock(
        return_value="Expect serial_number + display_value for this device type"
    )

    agent = DecisionAgent(decision_context_builder=mock_builder)
    agent.client = MagicMock()
    agent.client.chat.completions.create = AsyncMock(
        return_value=_make_mock_response(
            json.dumps(
                {
                    "routing": "sufficient",
                    "reasoning": "Fields match expectations",
                    "operator_message": "Reading looks good.",
                    "issues_identified": [],
                    "guidance_focus": None,
                }
            )
        )
    )

    await agent.process(
        {
            "extraction": _GOOD_EXTRACTION,
            "turn_number": 1,
        }
    )

    # Context builder should have been called
    mock_builder.build.assert_called_once()

    # The context should appear in the user message to the LLM
    call_args = agent.client.chat.completions.create.call_args
    user_content = call_args.kwargs["messages"][1]["content"]
    assert "Expect serial_number + display_value" in user_content


async def test_turn_history_in_prompt():
    """Turn history is included when provided."""
    agent = DecisionAgent()
    agent.client = MagicMock()
    agent.client.chat.completions.create = AsyncMock(
        return_value=_make_mock_response(
            json.dumps(
                {
                    "routing": "sufficient",
                    "reasoning": "Improved since last turn",
                    "operator_message": "Looks good now.",
                    "issues_identified": [],
                    "guidance_focus": None,
                }
            )
        )
    )

    await agent.process(
        {
            "extraction": _GOOD_EXTRACTION,
            "turn_number": 2,
            "turn_history": [
                {
                    "turn_number": 1,
                    "routing": "retry",
                    "issues_identified": ["too dark"],
                },
            ],
        }
    )

    call_args = agent.client.chat.completions.create.call_args
    user_content = call_args.kwargs["messages"][1]["content"]
    assert "Previous turns" in user_content
    assert "too dark" in user_content
