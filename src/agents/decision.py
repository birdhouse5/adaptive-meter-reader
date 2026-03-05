"""Decision Agent — evaluates extraction results and routes the session.

Uses GPT-4.1-mini (text-only, no image) to judge whether an extraction is
sufficient, needs retry with guidance, or should be escalated. Generates
operator-facing messages for each case.

See docs/architecture.md section 8 (Two-Agent Prompt Architecture) for how
the Decision Agent fits into the pipeline.
"""

import json
from typing import Any

from openai import AsyncOpenAI

from src.agents.base import AgentResult, BaseAgent
from src.config import DECISION_MODEL_NAME, MAX_SESSION_TURNS, OPENAI_API_KEY

_SYSTEM_PROMPT = """\
You evaluate meter reading extractions for completeness and usability.

Given the extraction result from a vision model, decide:
1. Is this sufficient for a usable reading? (sufficient / retry / escalate)
2. Generate an operator-facing message:
   - If sufficient: present the reading for confirmation
   - If retry: provide specific, actionable guidance
   - If escalate: acknowledge difficulty, summarize what was captured

Consider any provided context about device expectations, past interactions,
and consistency signals.

Return JSON only — no markdown fences, no explanation.

{
  "routing": "<sufficient | retry | escalate>",
  "reasoning": "<your internal reasoning>",
  "operator_message": "<message for the operator>",
  "issues_identified": ["<issue1>", ...],
  "guidance_focus": "<main area to improve, or null if sufficient>"
}
"""

_FALLBACK_RESULT: dict[str, Any] = {
    "routing": "escalate",
    "reasoning": "Could not parse decision model response",
    "operator_message": (
        "We had trouble evaluating this reading. "
        "A supervisor will review what was captured."
    ),
    "issues_identified": ["decision_parse_error"],
    "guidance_focus": None,
}


class DecisionAgent(BaseAgent):
    """Evaluates extraction results and decides routing + operator messaging.

    Accepts an optional decision_context_builder that assembles context from
    the knowledge base (device expectations, proven instructions, consistency
    signals, calibration data).

    See docs/architecture.md section 8.
    """

    name = "decision"

    def __init__(self, decision_context_builder=None) -> None:
        super().__init__()
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.decision_context_builder = decision_context_builder

    async def _run(self, payload: dict[str, Any]) -> AgentResult:
        extraction = payload.get("extraction", {})
        turn_number = payload.get("turn_number", 1)
        turn_history = payload.get("turn_history", [])

        # Build context for the decision
        user_content_parts = [
            f"Turn number: {turn_number} of {MAX_SESSION_TURNS}",
            f"Extraction result:\n{json.dumps(extraction, indent=2)}",
        ]

        # Add decision context from knowledge base if available
        if self.decision_context_builder:
            context = await self.decision_context_builder.build(
                extraction=extraction,
                turn_number=turn_number,
                turn_history=turn_history,
            )
            if context:
                user_content_parts.append(f"Context from knowledge base:\n{context}")

        # Add turn history summary
        if turn_history:
            history_lines = []
            for h in turn_history:
                history_lines.append(
                    f"  Turn {h.get('turn_number', '?')}: "
                    f"routing={h.get('routing', '?')}, "
                    f"issues={h.get('issues_identified', [])}"
                )
            user_content_parts.append("Previous turns:\n" + "\n".join(history_lines))

        # Force escalation if at max turns
        if turn_number >= MAX_SESSION_TURNS:
            user_content_parts.append(
                "IMPORTANT: This is the final allowed turn. "
                "If the reading is not clearly sufficient, you MUST escalate."
            )

        user_content = "\n\n".join(user_content_parts)

        response = await self.client.chat.completions.create(
            model=DECISION_MODEL_NAME,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=500,
            temperature=0.0,
        )

        raw = response.choices[0].message.content.strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            self.logger.warning(
                "Failed to parse decision response as JSON: %s", raw[:200]
            )
            parsed = {**_FALLBACK_RESULT}

        # Enforce escalation at max turns
        routing = parsed.get("routing", "escalate")
        if turn_number >= MAX_SESSION_TURNS and routing == "retry":
            parsed["routing"] = "escalate"
            parsed["reasoning"] = (
                f"Forced escalation: turn {turn_number} >= max {MAX_SESSION_TURNS}. "
                + parsed.get("reasoning", "")
            )

        return AgentResult(
            agent_name=self.name,
            output=parsed,
            confidence=1.0 if parsed.get("routing") == "sufficient" else 0.5,
            metadata={
                "model": DECISION_MODEL_NAME,
                "turn_number": turn_number,
                "raw_response": raw,
            },
        )
