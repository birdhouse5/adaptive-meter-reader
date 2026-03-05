"""Vision Agent — sends images to GPT-4.1 for open-ended meter data extraction.

Uses a multimodal LLM to extract all identifiable structured data from meter
images without prescribing specific fields. The extraction is open-ended so
the system discovers what each device type exposes.

See docs/architecture.md section 8 (Two-Agent Prompt Architecture) for how
the Vision Agent fits into the pipeline.
"""

import base64
import json
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from src.agents.base import AgentResult, BaseAgent
from src.config import OPENAI_API_KEY, VISION_MODEL_NAME

_BASE_PROMPT = """\
You are analyzing an image of a metering device. Extract all identifiable
structured data you can find. Report what you see — device type, any
identification numbers, display readings, units, and anything else visible.

For each extracted element, estimate your confidence (0.0-1.0).
Also assess the image quality and note any issues.

Important: If you can not properly read the displays or numbers, do not try to make them up. 
It's okay to report low confidence.

Return JSON only — no markdown fences, no explanation.

{
  "device_type": "<type of device, e.g. heat cost allocator, water meter>",
  "extracted_fields": {
    "<field_name>": {"value": "<value>", "confidence": <0.0-1.0>},
    ...
  },
  "image_quality": {
    "overall_usability": <0.0-1.0>,
    "issues": ["<issue1>", ...],
    "suggestions": ["<suggestion1>", ...]
  },
  "description": "<brief description of what you see>"
}
"""

_EMPTY_RESULT: dict[str, Any] = {
    "device_type": "unknown",
    "extracted_fields": {},
    "image_quality": {
        "overall_usability": 0.0,
        "issues": ["could not parse response"],
        "suggestions": [],
    },
    "description": "",
}


class VisionAgent(BaseAgent):
    """Extracts structured data from meter images using a vision-language model.

    Accepts an optional prompt_builder for few-shot enrichment on retries.
    See docs/architecture.md section 8.
    """

    name = "vision"

    def __init__(self, prompt_builder=None) -> None:
        super().__init__()
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.prompt_builder = prompt_builder

    async def _build_prompt(self, description: str = "") -> str:
        """Build the system prompt, optionally enriched by the prompt builder."""
        if self.prompt_builder:
            return await self.prompt_builder.build(description)
        return _BASE_PROMPT

    async def _run(self, payload: dict[str, Any]) -> AgentResult:
        image_path = Path(payload["image_path"])
        image_bytes = image_path.read_bytes()
        b64 = base64.b64encode(image_bytes).decode()
        mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"

        prompt_enrichment = payload.get("prompt_enrichment", "")
        system_prompt = await self._build_prompt(prompt_enrichment)

        if prompt_enrichment and self.prompt_builder is None:
            system_prompt += f"\n\nAdditional context:\n{prompt_enrichment}"

        response = await self.client.chat.completions.create(
            model=VISION_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                        {"type": "text", "text": "Read this meter."},
                    ],
                },
            ],
            max_tokens=500,
            temperature=0.0,
        )

        raw = response.choices[0].message.content.strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            self.logger.warning(
                "Failed to parse vision response as JSON: %s", raw[:200]
            )
            parsed = {**_EMPTY_RESULT, "description": raw}

        # Overall confidence = minimum of field confidences (conservative)
        extracted = parsed.get("extracted_fields", {})
        conf_values = []
        for field_data in extracted.values():
            if isinstance(field_data, dict):
                c = field_data.get("confidence", 0.0)
                if isinstance(c, (int, float)):
                    conf_values.append(c)
        overall_confidence = min(conf_values) if conf_values else 0.0

        return AgentResult(
            agent_name=self.name,
            output=parsed,
            confidence=overall_confidence,
            metadata={
                "model": VISION_MODEL_NAME,
                "image_path": str(image_path),
                "raw_response": raw,
            },
        )
