"""Page 1 — The Problem."""

import streamlit as st

st.header("The Problem")

st.markdown(
    '<p style="font-size:1.4rem;">'
    "4 meters per apartment. Dark basements. Blurry photos. Wrong digits."
    "</p>",
    unsafe_allow_html=True,
)

st.markdown(
    """
- Operators photograph meters under time pressure, in poor conditions
- Errors discovered **weeks later** during billing, too late to fix
- No feedback loop: the same mistakes repeat every cycle
"""
)

st.markdown("---")
st.markdown(
    '<p style="font-size:1.5rem;font-weight:bold;">'
    "Nothing in this process gets better over time."
    "</p>",
    unsafe_allow_html=True,
)
