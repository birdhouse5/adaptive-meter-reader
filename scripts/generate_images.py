"""Generate sample meter images using Replicate's nano-banana-pro model."""

import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://api.replicate.com/v1/models/google/nano-banana-pro/predictions"
API_TOKEN = os.getenv("REPLICATE_API_KEY", "")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_images"

# ── Image definitions: (meter_type, filename, prompt) ────────────────────────

IMAGES: list[tuple[str, str, str]] = [
    # ── Brunata HCA ──────────────────────────────────────────────────────
    # Good
    (
        "brunata_hca",
        "good_01.png",
        "Close-up photograph of a Brunata heat cost allocator (HCA) mounted "
        "on a white radiator. LCD display showing digits 04521, serial number "
        "label visible below the display. Well-lit room, sharp focus, straight-on angle. "
        "Realistic indoor photo.",
    ),
    (
        "brunata_hca",
        "good_02.png",
        "Brunata electronic heat cost allocator on a radiator in a modern apartment. "
        "Small white plastic device with LCD screen showing numbers. Brand name Brunata "
        "visible. Good natural lighting from window, all details readable. Realistic photo.",
    ),
    # Dark
    (
        "brunata_hca",
        "dark_01.png",
        "Brunata heat cost allocator on a radiator photographed in a very dark basement "
        "room. Minimal lighting, the LCD display is barely visible, deep shadows around "
        "the device. Realistic low-light phone photo.",
    ),
    (
        "brunata_hca",
        "dark_02.png",
        "Heat cost allocator on a radiator in a dimly lit hallway at night. Only a faint "
        "glow from a distant light. The meter display is hard to read. Dark, underexposed "
        "phone photo.",
    ),
    # Blurry
    (
        "brunata_hca",
        "blurry_01.png",
        "Brunata heat cost allocator on a radiator, photographed with motion blur. "
        "The image is shaky and out of focus, digits on the display are smeared and "
        "unreadable. Realistic blurry phone photo.",
    ),
    # Angled
    (
        "brunata_hca",
        "angled_01.png",
        "Brunata heat cost allocator photographed at a steep 45-degree angle from the "
        "side. Strong perspective distortion, the display is hard to read due to the "
        "viewing angle. Realistic phone photo.",
    ),
    # Glare
    (
        "brunata_hca",
        "glare_01.png",
        "Brunata heat cost allocator with strong flash reflection on the LCD display. "
        "Bright white glare spot washing out half the digits. Phone flash reflecting off "
        "the plastic surface. Realistic photo.",
    ),
    # Cropped
    (
        "brunata_hca",
        "cropped_01.png",
        "Extreme close-up of just the LCD digits of a heat cost allocator. The serial "
        "number and brand label are cut off. Only numbers visible, no context about the "
        "device type. Tight crop, realistic phone photo.",
    ),
    # Occluded
    (
        "brunata_hca",
        "occluded_01.png",
        "Brunata heat cost allocator on a radiator with a finger partially blocking the "
        "display. Someone holding their phone too close. Part of the reading is hidden "
        "behind the thumb. Realistic phone photo.",
    ),
    # Condensation
    (
        "brunata_hca",
        "condensation_01.png",
        "Brunata heat cost allocator with condensation droplets on the display surface. "
        "The digits appear foggy and distorted through the moisture. Humid bathroom "
        "environment. Realistic photo.",
    ),

    # ── Zenner Water Meter ───────────────────────────────────────────────
    # Good
    (
        "zenner_water_meter",
        "good_01.png",
        "Close-up photograph of a Zenner water meter installed on a pipe. Circular dial "
        "face with cubic meter reading digits visible. Blue and silver metal housing. "
        "Well-lit utility room, sharp focus, straight-on view. Realistic photo.",
    ),
    (
        "zenner_water_meter",
        "good_02.png",
        "Zenner residential water meter showing the dial and counter wheels with numbers. "
        "Clean installation, pipes visible. Good lighting, serial number readable on the "
        "meter body. Realistic indoor photo.",
    ),
    # Dark
    (
        "zenner_water_meter",
        "dark_01.png",
        "Water meter in a dark basement utility closet. Very little ambient light. The "
        "meter dial is barely visible in the shadows. Pipes and valves surrounding it. "
        "Dark, underexposed phone photo.",
    ),
    # Blurry
    (
        "zenner_water_meter",
        "blurry_01.png",
        "Zenner water meter photographed with heavy motion blur. The circular dial and "
        "numbers are completely smeared. Shaky handheld phone photo in low light. "
        "Realistic blurry image.",
    ),
    # Glare
    (
        "zenner_water_meter",
        "glare_01.png",
        "Zenner water meter with strong glare reflection on the glass cover over the "
        "dial. Bright spot obscuring the reading numbers. Flash reflecting off the "
        "transparent dome. Realistic photo.",
    ),
    # Angled
    (
        "zenner_water_meter",
        "angled_01.png",
        "Water meter photographed from a low angle looking up, in a tight space between "
        "pipes. Perspective distortion makes the dial hard to read. Cramped utility "
        "closet. Realistic phone photo.",
    ),
    # Far away
    (
        "zenner_water_meter",
        "far_01.png",
        "Water meter photographed from two meters away in a utility room. The meter is "
        "small in the frame, surrounded by pipes and wall. Digits are tiny and hard to "
        "distinguish. Realistic photo.",
    ),
    # Dirty
    (
        "zenner_water_meter",
        "dirty_01.png",
        "Old water meter covered in dust, grime, and calcium deposits. The dial face is "
        "partially obscured by buildup. Years of use visible. Realistic close-up photo.",
    ),

    # ── Minol Heat Allocator ─────────────────────────────────────────────
    # Good
    (
        "minol_heat_allocator",
        "good_01.png",
        "Minol electronic heat cost allocator mounted on a radiator. Small white device "
        "with LCD display showing digits. Minol brand label visible. Well-lit room, "
        "sharp focus, all information readable. Realistic photo.",
    ),
    (
        "minol_heat_allocator",
        "good_02.png",
        "Minol minocal heat allocator close-up on a radiator. Digital display with "
        "current reading visible. Serial number sticker on the side. Good indoor "
        "lighting. Realistic phone photo.",
    ),
    # Dark
    (
        "minol_heat_allocator",
        "dark_01.png",
        "Minol heat allocator on a radiator in a very dark room. The small LCD display "
        "is barely glowing. No ambient light, deep shadows. Realistic dark phone photo.",
    ),
    # Blurry
    (
        "minol_heat_allocator",
        "blurry_01.png",
        "Minol heat allocator photographed out of focus. The small device and display "
        "are blurred, digits unreadable. Camera focused on the radiator behind it. "
        "Realistic phone photo.",
    ),
    # Behind furniture
    (
        "minol_heat_allocator",
        "behind_furniture_01.png",
        "Minol heat allocator on a radiator partially hidden behind a couch. The device "
        "is only partially visible, photographed at an awkward angle reaching behind the "
        "furniture. Realistic phone photo.",
    ),
    # Sticker covering
    (
        "minol_heat_allocator",
        "sticker_01.png",
        "Minol heat allocator with a decorative sticker partially covering the serial "
        "number area. The display reading is visible but the serial is obscured. "
        "Realistic close-up photo.",
    ),

    # ── Generic Radiator / Wrong Device ──────────────────────────────────
    # Thermostat (wrong device)
    (
        "generic_radiator",
        "thermostat_01.png",
        "Close-up of a radiator thermostatic valve knob with numbered dial from 1 to 5. "
        "White plastic knob on a metal valve. No heat allocator or meter visible. "
        "The tenant photographed the wrong device. Realistic photo.",
    ),
    (
        "generic_radiator",
        "thermostat_02.png",
        "Radiator thermostat valve in a living room. Numbers 1-5 on the dial. Modern "
        "white radiator. The photo shows the temperature control, not a meter or "
        "allocator. Realistic phone photo.",
    ),
    # Back of device
    (
        "generic_radiator",
        "back_side_01.png",
        "The back side of a heat cost allocator on a radiator, showing the mounting "
        "bracket, wires, and clips. No display visible. The tenant photographed the "
        "wrong side. Realistic photo.",
    ),
    # Two meters
    (
        "generic_radiator",
        "two_meters_01.png",
        "Two different heat cost allocators mounted on adjacent radiators. Both displays "
        "visible but it is unclear which one the tenant wants to read. Two small white "
        "electronic devices. Realistic photo.",
    ),
    # Just radiator
    (
        "generic_radiator",
        "radiator_only_01.png",
        "A plain white radiator in an apartment room with no meter or heat allocator "
        "visible. Just a bare heating radiator under a window. Realistic interior photo.",
    ),
]


def _extract_image_url(data: dict) -> str | None:
    """Extract image URL from Replicate response data."""
    image_url = data.get("output")
    if isinstance(image_url, list) and image_url:
        image_url = image_url[0]
    return image_url if isinstance(image_url, str) else None


def _poll_prediction(url: str, headers: dict, max_wait: int = 300) -> dict | None:
    """Poll a Replicate prediction URL until it completes or fails."""
    poll_headers = {"Authorization": headers["Authorization"]}
    elapsed = 0
    interval = 2

    while elapsed < max_wait:
        time.sleep(interval)
        elapsed += interval

        resp = requests.get(url, headers=poll_headers, timeout=30)
        if resp.status_code != 200:
            print(f"  POLL ERROR {resp.status_code}: {resp.text[:200]}")
            return None

        data = resp.json()
        status = data.get("status")

        if status == "succeeded":
            return data
        elif status in ("failed", "canceled"):
            print(f"  PREDICTION {status}: {data.get('error', 'unknown')}")
            return None
        else:
            print(f"  polling... status={status} ({elapsed}s)", end="\r")

    print(f"  TIMEOUT after {max_wait}s")
    return None


def generate_image(prompt: str, output_path: Path) -> bool:
    """Call Replicate API and save the result."""
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
        "Prefer": "wait",
    }

    payload = {
        "input": {
            "prompt": prompt,
            "resolution": "2K",
            "image_input": [],
            "aspect_ratio": "4:3",
            "output_format": "png",
            "safety_filter_level": "block_only_high",
            "allow_fallback_model": False,
        }
    }

    resp = requests.post(API_URL, json=payload, headers=headers, timeout=120)

    if resp.status_code not in (200, 201, 202):
        print(f"  ERROR {resp.status_code}: {resp.text[:200]}")
        return False

    data = resp.json()

    # Handle async (202) — poll until done
    if resp.status_code == 202 or data.get("status") in ("starting", "processing"):
        poll_url = data.get("urls", {}).get("get")
        if not poll_url:
            print(f"  ERROR: 202 but no poll URL in response: {data}")
            return False
        print(f"  async prediction — polling...")
        data = _poll_prediction(poll_url, headers)
        if not data:
            return False

    image_url = _extract_image_url(data)
    if not image_url:
        print(f"  ERROR: No image URL in response: {data}")
        return False

    # Download the image
    img_resp = requests.get(image_url, timeout=60)
    if img_resp.status_code != 200:
        print(f"  ERROR downloading image: {img_resp.status_code}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(img_resp.content)
    return True


def main():
    if not API_TOKEN:
        print("ERROR: REPLICATE_API_KEY not set in .env")
        return

    total = len(IMAGES)
    success = 0
    failed = 0

    print(f"Generating {total} images...\n")

    for i, (meter_type, filename, prompt) in enumerate(IMAGES, 1):
        output_path = OUTPUT_DIR / meter_type / filename
        print(f"[{i}/{total}] {meter_type}/{filename}")

        if output_path.exists():
            print("  SKIP (already exists)")
            success += 1
            continue

        ok = generate_image(prompt, output_path)
        if ok:
            print(f"  OK → {output_path}")
            success += 1
        else:
            failed += 1

        # Rate limiting — be kind to the API
        if i < total:
            time.sleep(1)

    print(f"\nDone: {success} succeeded, {failed} failed")
    print(f"Images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
