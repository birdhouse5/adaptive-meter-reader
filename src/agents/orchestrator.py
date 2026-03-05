"""Orchestrator — runs the 2-agent pipeline for a single turn.

Coordinates the Vision Agent (image extraction) and Decision Agent (routing
and operator messaging) into a single turn result.

See docs/architecture.md section 2 (Session Lifecycle) for how the
orchestrator manages multi-turn sessions.
"""

from typing import Any

from pydantic import BaseModel

from src.agents.base import AgentResult, BaseAgent
from src.agents.decision import DecisionAgent
from src.agents.vision import VisionAgent


class TurnResult(BaseModel):
    """Result of a single turn in a reading session."""

    routing: str  # "sufficient" / "retry" / "escalate"
    extracted_fields: dict[str, Any]
    image_quality: dict[str, Any]
    operator_message: str
    decision_reasoning: str
    issues_identified: list[str]
    description: str = ""
    steps: list[AgentResult] = []


class Orchestrator(BaseAgent):
    """Runs Vision Agent -> Decision Agent for one turn of a reading session.

    See docs/architecture.md section 1 (High-Level System Overview).
    """

    name = "orchestrator"

    def __init__(
        self,
        vision_agent: VisionAgent | None = None,
        decision_agent: DecisionAgent | None = None,
    ) -> None:
        super().__init__()
        self.vision = vision_agent or VisionAgent()
        self.decision = decision_agent or DecisionAgent()

    async def _run(self, payload: dict[str, Any]) -> AgentResult:
        steps: list[AgentResult] = []
        turn_number: int = payload.get("turn_number", 1)
        turn_history: list[dict] = payload.get("turn_history", [])

        # Step 1: Vision extraction
        vision_result = await self.vision.process(payload)
        steps.append(vision_result)

        vo = vision_result.output
        extracted_fields = vo.get("extracted_fields", {})
        image_quality = vo.get("image_quality", {})
        description = vo.get("description", "")

        # Step 2: Decision Agent evaluates extraction
        decision_payload = {
            "extraction": vo,
            "turn_number": turn_number,
            "turn_history": turn_history,
        }
        decision_result = await self.decision.process(decision_payload)
        steps.append(decision_result)

        do = decision_result.output
        routing = do.get("routing", "escalate")
        operator_message = do.get("operator_message", "")
        decision_reasoning = do.get("reasoning", "")
        issues_identified = do.get("issues_identified", [])

        turn = TurnResult(
            routing=routing,
            extracted_fields=extracted_fields,
            image_quality=image_quality,
            operator_message=operator_message,
            decision_reasoning=decision_reasoning,
            issues_identified=issues_identified,
            description=description,
            steps=steps,
        )

        return AgentResult(
            agent_name=self.name,
            output=turn.model_dump(),
            confidence=vision_result.confidence,
            metadata={"routing": routing, "turn_number": turn_number},
        )
