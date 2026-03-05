"""Shared API helpers for the dashboard pages.

See docs/architecture.md §1 for the full system topology.
"""

import requests
import streamlit as st

API_BASE = "http://localhost:8000/api"


def api_get(path: str):
    """GET *path* from the FastAPI backend. Returns JSON or ``None``."""
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_post(path: str, **kwargs):
    """POST to *path* on the FastAPI backend. Shows error via ``st.error``."""
    try:
        r = requests.post(f"{API_BASE}{path}", timeout=30, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None
