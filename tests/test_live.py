"""Live LLM integration tests — requires a real OpenAI API key.

These tests send actual images to the Vision Agent and verify the response
structure. They are excluded from CI and run manually.

Usage:
    pytest tests/test_live.py -m live -v
"""

import os
from pathlib import Path

import pytest

from src.agents.vision import VisionAgent

pytestmark = pytest.mark.live

SAMPLE_DIR = Path("data/sample_images")


def _has_api_key() -> bool:
    key = os.getenv("OPENAI_API_KEY", "")
    return bool(key) and not key.startswith("sk-test")


def _find_sample_images() -> list[Path]:
    """Find sample images across all categories."""
    if not SAMPLE_DIR.exists():
        return []
    images = []
    for subdir in SAMPLE_DIR.iterdir():
        if subdir.is_dir():
            for img in sorted(subdir.glob("*.png"))[:1]:
                images.append(img)
            for img in sorted(subdir.glob("*.jpg"))[:1]:
                images.append(img)
    return images


@pytest.mark.skipif(not _has_api_key(), reason="No real API key available")
@pytest.mark.skipif(not _find_sample_images(), reason="No sample images found")
async def test_vision_agent_real_image():
    """Test the Vision Agent against a real sample image."""
    images = _find_sample_images()
    agent = VisionAgent()

    for image_path in images[:3]:
        result = await agent.process({"image_path": str(image_path)})

        assert result.agent_name == "vision"
        assert "device_type" in result.output
        assert "extracted_fields" in result.output
        assert "image_quality" in result.output
        assert isinstance(result.output["extracted_fields"], dict)
        assert isinstance(result.output["image_quality"], dict)


@pytest.mark.skipif(not _has_api_key(), reason="No real API key available")
@pytest.mark.skipif(not _find_sample_images(), reason="No sample images found")
async def test_vision_output_structure():
    """Verify the open-ended output structure from a real Vision Agent call."""
    images = _find_sample_images()
    if not images:
        pytest.skip("No images available")

    agent = VisionAgent()
    result = await agent.process({"image_path": str(images[0])})

    output = result.output
    # Must have these top-level keys
    assert "device_type" in output
    assert "extracted_fields" in output
    assert "image_quality" in output

    # Image quality must have overall_usability
    quality = output["image_quality"]
    assert "overall_usability" in quality

    # Extracted fields should have confidence per field
    for field_name, field_data in output["extracted_fields"].items():
        assert isinstance(field_data, dict), f"Field {field_name} should be a dict"
        assert "value" in field_data, f"Field {field_name} missing 'value'"
        assert "confidence" in field_data, f"Field {field_name} missing 'confidence'"
