"""Central configuration loaded from environment variables.

See docs/architecture.md for how these settings map to the 2-agent pipeline.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", str(DATA_DIR / "processing.db")))
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", str(DATA_DIR / "chroma_data")))

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "gpt-4.1")
DECISION_MODEL_NAME = os.getenv("DECISION_MODEL_NAME", "gpt-4.1-mini")

# Session limits
MAX_SESSION_TURNS = int(os.getenv("MAX_SESSION_TURNS", "5"))

# Meter types the system recognises (reference only — the Vision Agent discovers types)
METER_TYPES = [
    "zenner_water_meter",
    "brunata_hca",
    "minol_heat_allocator",
    "generic_radiator",
    "unknown",
]
