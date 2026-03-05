"""Seed the database with fabricated session data for the presentation.

Creates ~60 sessions across 3 device types showing a clear learning trend:
- Early sessions: high turns-to-success (2-4 turns)
- Later sessions: low turns-to-success (mostly 1 turn)
- Growing knowledge base with confirmed readings
- Instruction effectiveness data

Does NOT require the API server. Writes directly to SQLite.

Usage:
    .venv/bin/python scripts/seed_presentation.py
"""

import json
import random
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processing.db"

DEVICE_TYPES = [
    "Brunata HCA",
    "Zenner Water Meter",
    "Minol Heat Allocator",
]

SERIAL_PREFIXES = {
    "Brunata HCA": "BH",
    "Zenner Water Meter": "ZW",
    "Minol Heat Allocator": "MH",
}

INSTRUCTIONS = {
    "Brunata HCA": [
        ("dark_image", "Use your flashlight to illuminate the display"),
        ("serial_unclear", "Tilt your phone down, the serial is on the bottom label"),
        ("blurry", "Hold steady and tap to focus before taking the photo"),
    ],
    "Zenner Water Meter": [
        ("glare", "Step to the side to avoid glare on the glass cover"),
        ("partial_view", "Move back slightly so the full dial is visible"),
        ("dark_image", "Use your flashlight to illuminate the display"),
    ],
    "Minol Heat Allocator": [
        ("obstructed", "Gently move the curtain or object blocking the device"),
        ("angle", "Photograph straight on, not from an angle"),
        ("dark_image", "Use your flashlight to illuminate the display"),
    ],
}


def init_db(conn: sqlite3.Connection) -> None:
    """Create tables (same schema as src/data/database.py)."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meter_type TEXT DEFAULT 'unknown',
            status TEXT DEFAULT 'active',
            validation_status TEXT DEFAULT 'pending',
            total_turns INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            completed_at TEXT DEFAULT NULL
        );
        CREATE TABLE IF NOT EXISTS session_turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            turn_number INTEGER NOT NULL,
            image_path TEXT DEFAULT '',
            extracted_fields TEXT DEFAULT '{}',
            image_quality TEXT DEFAULT '{}',
            routing TEXT NOT NULL,
            operator_message TEXT DEFAULT '',
            decision_reasoning TEXT DEFAULT '',
            issues_identified TEXT DEFAULT '[]',
            description TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
        CREATE TABLE IF NOT EXISTS confirmed_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            device_type TEXT NOT NULL,
            confirmed_fields TEXT DEFAULT '{}',
            original_fields TEXT DEFAULT '{}',
            was_corrected INTEGER DEFAULT 0,
            correction_details TEXT DEFAULT '{}',
            confirmed_by TEXT DEFAULT 'operator',
            confirmed_at TEXT NOT NULL,
            image_path TEXT DEFAULT '',
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
        CREATE TABLE IF NOT EXISTS device_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_identifier TEXT NOT NULL,
            device_type TEXT NOT NULL,
            reading_value TEXT,
            reading_unit TEXT,
            session_id INTEGER,
            recorded_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
        CREATE INDEX IF NOT EXISTS idx_device_history_identifier
            ON device_history(device_identifier);
        CREATE TABLE IF NOT EXISTS calibration_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_type TEXT NOT NULL,
            field_name TEXT NOT NULL,
            model_confidence REAL NOT NULL,
            was_correct INTEGER NOT NULL,
            recorded_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS instruction_effectiveness (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            situation_signature TEXT NOT NULL,
            instruction_text TEXT NOT NULL,
            times_used INTEGER DEFAULT 0,
            times_led_to_success INTEGER DEFAULT 0,
            avg_turns_after REAL DEFAULT 0,
            effectiveness_rate REAL DEFAULT 0,
            UNIQUE(situation_signature, instruction_text)
        );
        CREATE TABLE IF NOT EXISTS agent_activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT NOT NULL,
            action TEXT NOT NULL,
            details TEXT DEFAULT '',
            created_at TEXT NOT NULL
        );
    """)
    conn.commit()


def make_fields(device_type: str, session_num: int) -> tuple[dict, str, str]:
    """Generate realistic extracted fields for a device type."""
    prefix = SERIAL_PREFIXES[device_type]
    serial = f"{prefix}-{random.randint(1000, 9999)}"
    reading = str(random.randint(100, 99999)).zfill(5)
    fields = {
        "serial_number": {"value": serial, "confidence": round(random.uniform(0.75, 0.98), 2)},
        "display_value": {"value": reading, "confidence": round(random.uniform(0.70, 0.97), 2)},
    }
    if device_type == "Zenner Water Meter":
        fields["unit"] = {"value": "m³", "confidence": 0.95}
    return fields, serial, reading


def seed() -> None:
    """Generate fabricated sessions showing a learning trend."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Start fresh
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(str(DB_PATH))
    init_db(conn)

    start_time = datetime.now(timezone.utc) - timedelta(days=56)  # 8 weeks ago
    n_sessions = 60

    instruction_stats: dict[tuple[str, str], dict] = {}

    for i in range(n_sessions):
        device_type = DEVICE_TYPES[i % len(DEVICE_TYPES)]
        session_time = start_time + timedelta(days=56 * i / n_sessions, hours=random.randint(8, 17))

        # Learning curve: early sessions need more turns, later sessions mostly succeed first try
        progress = i / n_sessions  # 0.0 to 1.0
        if progress < 0.2:
            # Week 1-2: rough, 2-4 turns
            turns = random.choices([2, 3, 4], weights=[40, 40, 20])[0]
        elif progress < 0.5:
            # Week 3-4: improving, 1-2 turns
            turns = random.choices([1, 2, 3], weights=[40, 45, 15])[0]
        elif progress < 0.8:
            # Week 5-6: good, mostly 1 turn
            turns = random.choices([1, 2], weights=[70, 30])[0]
        else:
            # Week 7-8: mature, almost always 1 turn
            turns = random.choices([1, 2], weights=[85, 15])[0]

        # Create session
        cur = conn.execute(
            "INSERT INTO sessions (meter_type, status, validation_status, total_turns, created_at, completed_at) VALUES (?, ?, ?, ?, ?, ?)",
            (device_type, "completed", "user_confirmed", turns, session_time.isoformat(), (session_time + timedelta(minutes=turns * 2)).isoformat()),
        )
        session_id = cur.lastrowid

        fields, serial, reading = make_fields(device_type, i)

        # Create turns
        for t in range(1, turns + 1):
            turn_time = session_time + timedelta(minutes=(t - 1) * 2)
            is_last = t == turns
            routing = "sufficient" if is_last else "retry"

            if routing == "retry":
                # Pick a random instruction for this device type
                situation, instruction = random.choice(INSTRUCTIONS[device_type])
                quality = {"overall_usability": round(random.uniform(0.2, 0.5), 2), "issues": [situation]}
                message = instruction
                issues = [situation]

                # Track instruction effectiveness
                key = (situation, instruction)
                if key not in instruction_stats:
                    instruction_stats[key] = {"used": 0, "success": 0}
                instruction_stats[key]["used"] += 1
                instruction_stats[key]["success"] += 1  # all eventually succeed
            else:
                quality = {"overall_usability": round(random.uniform(0.7, 0.95), 2), "issues": []}
                message = f"We read serial {serial}, value {reading}. Correct?"
                issues = []

            conn.execute(
                """INSERT INTO session_turns
                   (session_id, turn_number, extracted_fields, image_quality,
                    routing, operator_message, decision_reasoning,
                    issues_identified, description, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id, t,
                    json.dumps(fields if is_last else {k: {**v, "confidence": round(v["confidence"] * 0.6, 2)} for k, v in fields.items()}),
                    json.dumps(quality),
                    routing, message,
                    "All fields high confidence" if is_last else "Insufficient extraction",
                    json.dumps(issues),
                    f"{device_type} reading",
                    turn_time.isoformat(),
                ),
            )

        # Confirmed reading (with occasional corrections early on)
        was_corrected = random.random() < (0.3 if progress < 0.3 else 0.05)
        flat_fields = {k: v["value"] for k, v in fields.items()}

        conn.execute(
            """INSERT INTO confirmed_readings
               (session_id, device_type, confirmed_fields, original_fields,
                was_corrected, confirmed_by, confirmed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id, device_type,
                json.dumps(flat_fields), json.dumps(flat_fields),
                int(was_corrected), "operator",
                (session_time + timedelta(minutes=turns * 2 + 1)).isoformat(),
            ),
        )

        # Device history
        conn.execute(
            """INSERT INTO device_history
               (device_identifier, device_type, reading_value, reading_unit,
                session_id, recorded_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (serial, device_type, reading, "units", session_id, session_time.isoformat()),
        )

    # Write instruction effectiveness
    for (situation, instruction), stats in instruction_stats.items():
        rate = stats["success"] / stats["used"] if stats["used"] > 0 else 0
        conn.execute(
            """INSERT OR REPLACE INTO instruction_effectiveness
               (situation_signature, instruction_text, times_used,
                times_led_to_success, avg_turns_after, effectiveness_rate)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (situation, instruction, stats["used"], stats["success"], 1.0, round(rate, 2)),
        )

    conn.commit()
    conn.close()

    print(f"Seeded {n_sessions} sessions into {DB_PATH}")
    print(f"  Device types: {', '.join(DEVICE_TYPES)}")
    print(f"  Instruction types: {len(instruction_stats)}")
    print(f"  Learning trend: high turns early -> low turns late")
    print(f"\nRun the dashboard:  .venv/bin/streamlit run dashboard/app.py")


if __name__ == "__main__":
    seed()
