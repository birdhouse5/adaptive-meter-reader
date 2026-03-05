"""Run the orchestrator against sample images and print the full story.

Demonstrates the 2-agent pipeline (Vision Agent + Decision Agent) against
curated scenarios showing different routing outcomes.

See docs/architecture.md section 2 (Session Lifecycle).
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.agents.decision import DecisionAgent
from src.agents.orchestrator import Orchestrator
from src.agents.vision import VisionAgent


async def run_story(image_path: Path, orchestrator: Orchestrator, turn: int = 1):
    """Run one image through the pipeline and print the story."""
    result = await orchestrator.process(
        {"image_path": str(image_path), "turn_number": turn}
    )
    o = result.output

    print(f"\n{'=' * 70}")
    print(f"IMAGE: {image_path.relative_to(Path('data/sample_images'))}")
    print(f"{'=' * 70}")

    # Extraction
    fields = o.get("extracted_fields", {})
    print("\n  EXTRACTION:")
    for field_name, field_data in fields.items():
        if isinstance(field_data, dict):
            val = field_data.get("value", "?")
            conf = field_data.get("confidence", 0)
            marker = "+" if conf >= 0.8 else "?" if conf >= 0.5 else "-"
            print(f"    {marker} {field_name:20s} = {str(val):30s} ({conf:.0%})")
        else:
            print(f"      {field_name:20s} = {field_data}")

    # Image quality
    quality = o.get("image_quality", {})
    print(f"\n  IMAGE QUALITY:")
    usability = quality.get("overall_usability", 0)
    print(f"    overall_usability = {usability:.2f}")
    issues = quality.get("issues", [])
    if issues:
        print(f"    issues: {', '.join(issues)}")
    suggestions = quality.get("suggestions", [])
    if suggestions:
        print(f"    suggestions: {', '.join(suggestions)}")

    # Description
    desc = o.get("description", "")
    if desc:
        print(f"\n  DESCRIPTION: {desc}")

    # Routing
    routing = o.get("routing", "?")
    print(f"\n  ROUTING: {routing.upper()}")

    # Decision reasoning
    reasoning = o.get("decision_reasoning", "")
    if reasoning:
        print(f"  REASONING: {reasoning}")

    # Issues identified
    issues = o.get("issues_identified", [])
    if issues:
        print(f"  ISSUES: {', '.join(issues)}")

    # Operator message
    message = o.get("operator_message", "")
    if message:
        print(f"  OPERATOR MESSAGE: {message}")

    return o


async def main():
    # Images to test — curated for interesting stories
    test_images = [
        # Story 1: Good image, should be sufficient
        "brunata_hca/good_01.png",
        # Story 2: Dark image, should need retry
        "brunata_hca/dark_01.png",
        # Story 3: Wrong device entirely
        "generic_radiator/thermostat_01.png",
        # Story 4: Good water meter
        "zenner_water_meter/good_01.png",
        # Story 5: Blurry image
        "minol_heat_allocator/blurry_01.png",
        # Story 6: Glare on display
        "brunata_hca/glare_01.png",
    ]

    # Filter to images passed as args, or use all
    if len(sys.argv) > 1:
        test_images = sys.argv[1:]

    vision = VisionAgent()
    decision = DecisionAgent()
    orchestrator = Orchestrator(vision_agent=vision, decision_agent=decision)

    base = Path("data/sample_images")
    for img_rel in test_images:
        img_path = base / img_rel
        if not img_path.exists():
            print(f"\nSKIP: {img_rel} not found")
            continue
        await run_story(img_path, orchestrator)

    print(f"\n{'=' * 70}")
    print("DONE")


if __name__ == "__main__":
    asyncio.run(main())
