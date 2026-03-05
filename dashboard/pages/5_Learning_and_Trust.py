"""Page 5 — Learning & Trust."""

import streamlit as st

from diagrams import LEARNING_LOOP, render_mermaid

st.header("Learning & Trust")

st.markdown("### The flywheel")

render_mermaid(LEARNING_LOOP)

st.markdown(
    '<p style="font-size:1.15rem;">'
    "<b>Week 1:</b> generic guidance. "
    "<b>Week 4:</b> device-specific patterns. "
    "<b>Week 8:</b> per-device playbooks."
    "</p>",
    unsafe_allow_html=True,
)

st.divider()

st.markdown("### Three tiers of ground truth")

st.markdown(
    "The system never trusts its own confidence alone. "
    "Three independent sources of truth prevent circular learning:"
)

st.markdown(
    """
| Tier | How it works | Cost |
|------|-------------|------|
| **1. Operator confirmation** | Confirm or correct after every reading | Low (inline, one tap) |
| **2. Cross-session consistency** | Same device should have same serial; readings only go up | Free (automatic) |
| **3. Supervisor spot-checks** | Periodic sample verification | Medium (manual) |
"""
)
