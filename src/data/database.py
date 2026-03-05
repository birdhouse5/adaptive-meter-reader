"""SQLite operations — sessions, turns, confirmed readings, device history, calibration.

See docs/architecture.md section 4 (Ground Truth Validation Tiers) for how
confirmed readings and device history feed the learning loop.
"""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import DATABASE_PATH

_lock = threading.Lock()


def _get_conn(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or DATABASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(db_path: Path | None = None) -> None:
    """Create tables if they don't exist."""
    conn = _get_conn(db_path)
    conn.executescript(
        """
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
        """
    )
    conn.commit()
    conn.close()


# -- Session operations -------------------------------------------------------


def create_session(db_path: Path | None = None) -> int:
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        conn = _get_conn(db_path)
        cur = conn.execute(
            "INSERT INTO sessions (status, created_at) VALUES ('active', ?)",
            (now,),
        )
        session_id = cur.lastrowid
        conn.commit()
        conn.close()
    return session_id


def get_session(session_id: int, db_path: Path | None = None) -> dict[str, Any] | None:
    conn = _get_conn(db_path)
    row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_session(
    session_id: int,
    *,
    meter_type: str | None = None,
    status: str | None = None,
    validation_status: str | None = None,
    total_turns: int | None = None,
    db_path: Path | None = None,
) -> None:
    parts: list[str] = []
    values: list[Any] = []
    if meter_type is not None:
        parts.append("meter_type = ?")
        values.append(meter_type)
    if status is not None:
        parts.append("status = ?")
        values.append(status)
        if status in ("completed", "escalated", "abandoned"):
            parts.append("completed_at = ?")
            values.append(datetime.now(timezone.utc).isoformat())
    if validation_status is not None:
        parts.append("validation_status = ?")
        values.append(validation_status)
    if total_turns is not None:
        parts.append("total_turns = ?")
        values.append(total_turns)
    if not parts:
        return
    values.append(session_id)
    with _lock:
        conn = _get_conn(db_path)
        conn.execute(
            f"UPDATE sessions SET {', '.join(parts)} WHERE id = ?",
            values,
        )
        conn.commit()
        conn.close()


def list_sessions(limit: int = 50, db_path: Path | None = None) -> list[dict[str, Any]]:
    conn = _get_conn(db_path)
    rows = conn.execute(
        "SELECT * FROM sessions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# -- Turn operations -----------------------------------------------------------


def insert_turn(
    session_id: int,
    turn_number: int,
    routing: str,
    image_path: str = "",
    extracted_fields: dict | None = None,
    image_quality: dict | None = None,
    operator_message: str = "",
    decision_reasoning: str = "",
    issues_identified: list | None = None,
    description: str = "",
    db_path: Path | None = None,
) -> int:
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        conn = _get_conn(db_path)
        cur = conn.execute(
            """INSERT INTO session_turns
               (session_id, turn_number, image_path, extracted_fields,
                image_quality, routing, operator_message, decision_reasoning,
                issues_identified, description, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                turn_number,
                image_path,
                json.dumps(extracted_fields or {}),
                json.dumps(image_quality or {}),
                routing,
                operator_message,
                decision_reasoning,
                json.dumps(issues_identified or []),
                description,
                now,
            ),
        )
        turn_id = cur.lastrowid
        conn.commit()
        conn.close()
    return turn_id


def get_turns_for_session(
    session_id: int, db_path: Path | None = None
) -> list[dict[str, Any]]:
    conn = _get_conn(db_path)
    rows = conn.execute(
        "SELECT * FROM session_turns WHERE session_id = ? ORDER BY turn_number",
        (session_id,),
    ).fetchall()
    conn.close()
    results = []
    for r in rows:
        d = dict(r)
        for key in ("extracted_fields", "image_quality", "issues_identified"):
            if isinstance(d.get(key), str):
                d[key] = json.loads(d[key])
        results.append(d)
    return results


# -- Confirmed readings -------------------------------------------------------


def insert_confirmed_reading(
    session_id: int,
    device_type: str,
    confirmed_fields: dict,
    original_fields: dict,
    was_corrected: bool = False,
    correction_details: dict | None = None,
    confirmed_by: str = "operator",
    image_path: str = "",
    db_path: Path | None = None,
) -> int:
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        conn = _get_conn(db_path)
        cur = conn.execute(
            """INSERT INTO confirmed_readings
               (session_id, device_type, confirmed_fields, original_fields,
                was_corrected, correction_details, confirmed_by, confirmed_at, image_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                device_type,
                json.dumps(confirmed_fields),
                json.dumps(original_fields),
                int(was_corrected),
                json.dumps(correction_details or {}),
                confirmed_by,
                now,
                image_path,
            ),
        )
        reading_id = cur.lastrowid
        conn.commit()
        conn.close()
    return reading_id


def get_confirmed_readings(
    device_type: str | None = None,
    limit: int = 100,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    conn = _get_conn(db_path)
    if device_type:
        rows = conn.execute(
            "SELECT * FROM confirmed_readings WHERE device_type = ? "
            "ORDER BY id DESC LIMIT ?",
            (device_type, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM confirmed_readings ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    conn.close()
    results = []
    for r in rows:
        d = dict(r)
        for key in ("confirmed_fields", "original_fields", "correction_details"):
            if isinstance(d.get(key), str):
                d[key] = json.loads(d[key])
        results.append(d)
    return results


# -- Device history ------------------------------------------------------------


def insert_device_history(
    device_identifier: str,
    device_type: str,
    reading_value: str | None = None,
    reading_unit: str | None = None,
    session_id: int | None = None,
    db_path: Path | None = None,
) -> int:
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        conn = _get_conn(db_path)
        cur = conn.execute(
            """INSERT INTO device_history
               (device_identifier, device_type, reading_value, reading_unit,
                session_id, recorded_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                device_identifier,
                device_type,
                reading_value,
                reading_unit,
                session_id,
                now,
            ),
        )
        history_id = cur.lastrowid
        conn.commit()
        conn.close()
    return history_id


def get_device_history(
    device_identifier: str,
    limit: int = 50,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    conn = _get_conn(db_path)
    rows = conn.execute(
        "SELECT * FROM device_history WHERE device_identifier = ? "
        "ORDER BY recorded_at DESC LIMIT ?",
        (device_identifier, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# -- Calibration data ----------------------------------------------------------


def insert_calibration_data(
    device_type: str,
    field_name: str,
    model_confidence: float,
    was_correct: bool,
    db_path: Path | None = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        conn = _get_conn(db_path)
        conn.execute(
            """INSERT INTO calibration_data
               (device_type, field_name, model_confidence, was_correct, recorded_at)
               VALUES (?, ?, ?, ?, ?)""",
            (device_type, field_name, model_confidence, int(was_correct), now),
        )
        conn.commit()
        conn.close()


def get_calibration_data(
    device_type: str | None = None,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    conn = _get_conn(db_path)
    if device_type:
        rows = conn.execute(
            "SELECT * FROM calibration_data WHERE device_type = ? ORDER BY id DESC",
            (device_type,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM calibration_data ORDER BY id DESC"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# -- Stats ---------------------------------------------------------------------


def get_stats(db_path: Path | None = None) -> dict[str, Any]:
    conn = _get_conn(db_path)

    total_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    completed = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE status = 'completed'"
    ).fetchone()[0]
    escalated = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE status = 'escalated'"
    ).fetchone()[0]
    abandoned = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE status = 'abandoned'"
    ).fetchone()[0]

    avg_turns = (
        conn.execute(
            "SELECT AVG(total_turns) FROM sessions WHERE status = 'completed'"
        ).fetchone()[0]
        or 0.0
    )

    first_attempt = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE status = 'completed' AND total_turns = 1"
    ).fetchone()[0]
    first_attempt_rate = (first_attempt / completed * 100) if completed > 0 else 0.0

    by_meter_type = {}
    for row in conn.execute(
        """SELECT meter_type, COUNT(*) as cnt, AVG(total_turns) as avg_t
           FROM sessions WHERE status = 'completed'
           GROUP BY meter_type"""
    ).fetchall():
        by_meter_type[row["meter_type"]] = {
            "count": row["cnt"],
            "avg_turns": round(row["avg_t"], 2),
        }

    # Confirmation stats
    total_confirmed = conn.execute(
        "SELECT COUNT(*) FROM confirmed_readings"
    ).fetchone()[0]
    total_corrected = conn.execute(
        "SELECT COUNT(*) FROM confirmed_readings WHERE was_corrected = 1"
    ).fetchone()[0]
    correction_rate = (
        (total_corrected / total_confirmed * 100) if total_confirmed > 0 else 0.0
    )

    conn.close()
    return {
        "total_sessions": total_sessions,
        "completed": completed,
        "escalated": escalated,
        "abandoned": abandoned,
        "avg_turns_to_success": round(avg_turns, 2),
        "first_attempt_success_rate": round(first_attempt_rate, 1),
        "by_meter_type": by_meter_type,
        "confirmation_stats": {
            "total_confirmed": total_confirmed,
            "total_corrected": total_corrected,
            "correction_rate": round(correction_rate, 1),
        },
    }


# -- Instruction effectiveness -------------------------------------------------


def get_instruction_effectiveness(
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    conn = _get_conn(db_path)
    rows = conn.execute(
        "SELECT * FROM instruction_effectiveness ORDER BY effectiveness_rate DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def upsert_instruction_effectiveness(
    situation_signature: str,
    instruction_text: str,
    success: bool,
    turns_after: int = 0,
    db_path: Path | None = None,
) -> None:
    with _lock:
        conn = _get_conn(db_path)
        conn.execute(
            """INSERT INTO instruction_effectiveness
               (situation_signature, instruction_text, times_used,
                times_led_to_success, avg_turns_after, effectiveness_rate)
               VALUES (?, ?, 1, ?, ?, ?)
               ON CONFLICT(situation_signature, instruction_text) DO UPDATE SET
                   times_used = times_used + 1,
                   times_led_to_success = times_led_to_success + ?,
                   avg_turns_after = (avg_turns_after * times_used + ?) / (times_used + 1),
                   effectiveness_rate = CAST(times_led_to_success + ? AS REAL)
                       / (times_used + 1)""",
            (
                situation_signature,
                instruction_text,
                int(success),
                turns_after,
                float(success),
                int(success),
                turns_after,
                int(success),
            ),
        )
        conn.commit()
        conn.close()


# -- Agent activity ------------------------------------------------------------


def log_agent_activity(
    agent_name: str,
    action: str,
    details: str = "",
    db_path: Path | None = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        conn = _get_conn(db_path)
        conn.execute(
            """INSERT INTO agent_activity (agent_name, action, details, created_at)
               VALUES (?, ?, ?, ?)""",
            (agent_name, action, details, now),
        )
        conn.commit()
        conn.close()


def get_agent_activity(
    limit: int = 100, db_path: Path | None = None
) -> list[dict[str, Any]]:
    conn = _get_conn(db_path)
    rows = conn.execute(
        "SELECT * FROM agent_activity ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
