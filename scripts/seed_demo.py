"""Seed the system with sample meter readings for a compelling demo.

Uploads a selection of good-quality sample images via the API, then confirms
each successful reading. After running, the dashboard shows populated sessions,
metrics, and knowledge base stats.

Requires the API server to be running: uvicorn src.main:app --port 8000

Usage:
    python scripts/seed_demo.py [--base-url http://localhost:8000]
"""

import argparse
import sys
import time
from pathlib import Path

import requests

# Sample images that produce good extractions (good quality, clear displays).
SAMPLE_IMAGES = [
    "data/sample_images/minol_heat_allocator/good_01.png",
    "data/sample_images/minol_heat_allocator/good_02.png",
    "data/sample_images/zenner_water_meter/good_01.png",
    "data/sample_images/zenner_water_meter/good_02.png",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def wait_for_server(base_url: str, timeout: int = 10) -> bool:
    """Wait for the API server to be reachable."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/api/health", timeout=2)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(1)
    return False


def seed(base_url: str) -> None:
    """Upload sample images and confirm successful readings."""
    print(f"Connecting to {base_url} ...")
    if not wait_for_server(base_url):
        print(f"ERROR: Server not reachable at {base_url}")
        print("Start it first: uvicorn src.main:app --port 8000")
        sys.exit(1)

    print(f"Server is up. Seeding {len(SAMPLE_IMAGES)} sample readings.\n")

    confirmed = 0
    for image_rel in SAMPLE_IMAGES:
        image_path = PROJECT_ROOT / image_rel
        if not image_path.exists():
            print(f"  SKIP  {image_rel} (file not found)")
            continue

        # Upload image
        print(f"  Upload  {image_rel} ... ", end="", flush=True)
        with open(image_path, "rb") as f:
            mime = "image/png" if image_path.suffix == ".png" else "image/jpeg"
            resp = requests.post(
                f"{base_url}/api/session/start",
                files={"file": (image_path.name, f, mime)},
                timeout=60,
            )

        if resp.status_code != 200:
            print(f"FAILED (HTTP {resp.status_code})")
            continue

        data = resp.json()
        session_id = data["session_id"]
        routing = data["routing"]
        print(f"session={session_id}  routing={routing}")

        if routing != "sufficient":
            print(f"          Skipping confirmation (routing={routing})")
            continue

        # Confirm the reading as-is (no corrections)
        extracted = data.get("extracted_fields", {})
        confirmed_fields = {
            k: v["value"] if isinstance(v, dict) else v
            for k, v in extracted.items()
        }

        print(f"  Confirm session {session_id} ... ", end="", flush=True)
        resp = requests.post(
            f"{base_url}/api/session/{session_id}/confirm",
            json={"confirmed_fields": confirmed_fields},
            timeout=30,
        )

        if resp.status_code == 200:
            print("OK")
            confirmed += 1
        else:
            print(f"FAILED (HTTP {resp.status_code})")

    # Summary
    print(f"\nDone. {confirmed}/{len(SAMPLE_IMAGES)} readings confirmed.")
    if confirmed > 0:
        print("Dashboard should now show sessions, metrics, and knowledge base stats.")


def main():
    parser = argparse.ArgumentParser(description="Seed demo data for the dashboard.")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="API server base URL (default: http://localhost:8000)",
    )
    args = parser.parse_args()
    seed(args.base_url)


if __name__ == "__main__":
    main()
