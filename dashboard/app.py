"""Adaptive Meter Reader — Dashboard entry point (Home page).

This is the landing page of an 8-page Streamlit presentation.
Sidebar navigation is automatic via the ``dashboard/pages/`` convention.

See docs/architecture.md for the full system architecture.
"""

import streamlit as st

st.set_page_config(page_title="Adaptive Meter Reader", layout="wide")

st.title("Adaptive Meter Reader")
st.markdown("### A self-improving AI system for utility meter readings")
st.markdown(
    '<p style="font-size:1.25rem;color:#555;">'
    "The system gets measurably better at its job with every interaction."
    "</p>",
    unsafe_allow_html=True,
)

st.divider()

st.markdown(
    """
**Navigate the pages in the sidebar** to walk through the system:

1. **The Problem** : why meter reading needs fixing
2. **The Solution** : the core idea
3. **Live Demo** : see it in action
4. **How It Works** : the 2-agent pipeline and knowledge base
5. **Learning & Trust** : how the system improves and validates
6. **Results** : the operational dashboard
7. **Trade-offs** : key technical decisions
8. **Vision** : where this goes next
"""
)
