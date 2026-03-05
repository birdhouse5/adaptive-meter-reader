"""Page 2 — The Solution."""

import streamlit as st

from diagrams import PIPELINE, render_mermaid

st.header("The Solution")

st.markdown(
    '<p style="font-size:1.3rem;">'
    "What if the phone told them, <b>right there</b>, how to fix the photo?"
    "</p>",
    unsafe_allow_html=True,
)

st.markdown("### An AI assistant that learns from every confirmed interaction")

render_mermaid(PIPELINE)

st.markdown(
    """
- **Vision Agent** extracts whatever it can see, no rigid schema
- **Decision Agent** judges if the reading is usable and generates guidance
- **Knowledge Base** grows from confirmed readings, better every time
"""
)
