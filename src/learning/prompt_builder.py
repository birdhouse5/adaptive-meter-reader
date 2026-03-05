"""Vision Agent prompt builder — enriches prompts with few-shot examples.

Turn 1 uses the base prompt only. Turn 2+ gets enriched with similar confirmed
images and correction warnings from the knowledge base.

See docs/architecture.md section 8 (Two-Agent Prompt Architecture).
"""

from src.data.vector_store import KnowledgeBase

_BASE_PROMPT = """\
You are analyzing an image of a metering device. Extract all identifiable
structured data you can find. Report what you see — device type, any
identification numbers, display readings, units, and anything else visible.

For each extracted element, estimate your confidence (0.0-1.0).
Also assess the image quality and note any issues.

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

_FEW_SHOT_HEADER = (
    "\nHere are confirmed examples of similar devices to improve your accuracy:\n"
)

_CORRECTION_HEADER = "\nKnown errors to watch out for with similar devices:\n"


class VisionPromptBuilder:
    """Builds Vision Agent prompts, enriched with knowledge base on retries.

    See docs/architecture.md section 8.
    """

    def __init__(self, knowledge_base: KnowledgeBase | None = None) -> None:
        self.kb = knowledge_base or KnowledgeBase()

    async def build(self, description: str = "", n_examples: int = 3) -> str:
        """Build a prompt, enriched with few-shot examples if the KB has data."""
        prompt = _BASE_PROMPT

        if not description:
            return prompt

        # Few-shot examples from confirmed images
        similar_images = self.kb.find_similar_images(description, n_results=n_examples)
        if similar_images:
            prompt += _FEW_SHOT_HEADER
            for img in similar_images:
                device_type = img.get("device_type", "unknown")
                fields = img.get("confirmed_fields", "")
                prompt += f"- Device type: {device_type}, confirmed fields: {fields}\n"

        # Correction warnings
        corrections = self.kb.find_similar_corrections(
            description, n_results=n_examples
        )
        if corrections:
            prompt += _CORRECTION_HEADER
            for corr in corrections:
                field = corr.get("field_name", "unknown")
                original = corr.get("original_value", "?")
                corrected = corr.get("corrected_value", "?")
                prompt += (
                    f"- Field '{field}': model read '{original}' "
                    f"but correct was '{corrected}'\n"
                )

        return prompt
