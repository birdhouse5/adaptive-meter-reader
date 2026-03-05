"""Page 7 — Vision."""

import streamlit as st

from diagrams import ROLLOUT, render_mermaid

st.header("Vision")

render_mermaid(ROLLOUT)

st.markdown(
    '<p style="font-size:1.15rem;">'
    "Operator usage builds the knowledge base. "
    "Tenants inherit a <b>mature system</b>."
    "</p>",
    unsafe_allow_html=True,
)

st.divider()

st.markdown(
    '<p style="font-size:1.3rem;text-align:center;font-weight:bold;">'
    "More confirmed readings &rarr; better guidance &rarr; "
    "fewer retries &rarr; more confirmed readings"
    "</p>",
    unsafe_allow_html=True,
)
