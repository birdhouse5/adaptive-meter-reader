"""Page 4: How It Works.

Unified data flow diagram + concrete example walkthrough showing
exactly what data moves where during a complete session.
"""

import streamlit as st

from diagrams import DATA_FLOW, render_mermaid

st.header("How It Works")

st.markdown(
    "One complete session traced step by step: "
    "what each agent produces, what the knowledge base contributes, "
    "and what gets stored when the operator confirms."
)

# ---------------------------------------------------------------------------
# Unified data flow diagram
# ---------------------------------------------------------------------------
st.subheader("Data Flow Overview")

render_mermaid(DATA_FLOW, height=520)

st.markdown(
    "Dotted arrows are knowledge base queries. "
    "Solid arrows are data passing through the pipeline. "
    "On the first ever turn, no KB context exists. "
    "As confirmed readings accumulate, both agents receive richer context."
)

# ---------------------------------------------------------------------------
# Concrete example: a retry scenario
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Concrete Example: A Brunata HCA Reading")

st.markdown(
    "Walking through a real scenario. "
    "The operator photographs a heat cost allocator in a dim basement. "
    "It takes two attempts."
)

# --- Turn 1 ---
st.markdown("---")
st.markdown("#### Turn 1: First attempt (no KB context yet)")

st.markdown("**Step 1** — Operator uploads a blurry photo in low light.")

st.markdown("**Step 2** — Vision Agent extracts what it can:")
st.code(
    """{
  "device_type": "heat_cost_allocator",
  "extracted_fields": {
    "serial_number": { "value": "BH-284?", "confidence": 0.45 },
    "display_value": { "value": "1847",    "confidence": 0.82 }
  },
  "image_quality": {
    "overall_usability": 0.4,
    "issues": ["low_light", "partial_blur"]
  },
  "description": "Brunata HCA mounted on radiator, dim lighting,
                   serial label partially obscured by shadow"
}""",
    language="json",
)
st.caption(
    "No KB context on the first turn. The Vision Agent works from general knowledge only."
)

st.markdown("**Step 3** — Decision Agent receives the extraction (never the image):")
st.code(
    """{
  "routing": "retry",
  "operator_message": "The serial number is hard to read. Please use your
                       phone flashlight and retake the photo.",
  "issues_identified": ["low_light", "serial_unclear"],
  "guidance_focus": "serial_number"
}""",
    language="json",
)
st.caption(
    "No proven guidance in the KB yet, so the Decision Agent reasons from "
    "the extraction alone. It sees low confidence on the serial and asks for a retry."
)

# --- Turn 2 ---
st.markdown("---")
st.markdown("#### Turn 2: Retry (with KB context)")

st.markdown("**Step 1** — Operator uses flashlight and retakes the photo.")

st.markdown(
    "**Step 2** — Vision Agent extracts again, "
    "now with KB context injected into its prompt:"
)

with st.expander("KB retrieval: what the Vision Agent receives", expanded=True):
    st.markdown("**From confirmed_images** (cosine similarity search):")
    st.code(
        """Query: "Brunata HCA on radiator, dim lighting"
→ 3 similar confirmed readings found:

  1. "Brunata HCA in basement, serial on bottom label"
     confirmed_fields: { serial: "BH-1923", display: "0842" }

  2. "Brunata HCA near pipe, serial partially covered"
     confirmed_fields: { serial: "BH-3011", display: "2156" }

  3. "Brunata HCA in stairwell, low angle"
     confirmed_fields: { serial: "BH-0447", display: "1390" }

→ Injected as few-shot examples:
  "Here are 3 confirmed images of similar devices.
   Serial number was always on the bottom label."
""",
        language="text",
    )
    st.markdown("**From correction_patterns** (cosine similarity search):")
    st.code(
        """Query: "Brunata HCA, serial_number extraction"
→ 1 known error found:

  field: serial_number
  original: "BH-2841"  →  corrected: "BH-2847"

→ Injected as warning:
  "Known issue: final digit of serial often misread
   on Brunata HCAs in low light."
""",
        language="text",
    )

st.markdown("Vision Agent output (now more accurate):")
st.code(
    """{
  "device_type": "heat_cost_allocator",
  "extracted_fields": {
    "serial_number": { "value": "BH-2847", "confidence": 0.93 },
    "display_value": { "value": "1847",    "confidence": 0.91 }
  },
  "image_quality": {
    "overall_usability": 0.8,
    "issues": []
  },
  "description": "Brunata HCA on radiator, serial BH-2847 clearly
                   visible on bottom label, display reads 1847"
}""",
    language="json",
)

st.markdown("**Step 3** — Decision Agent judges, also with KB context:")

with st.expander("KB retrieval: what the Decision Agent receives", expanded=True):
    st.markdown("**From interaction_patterns** (proven guidance):")
    st.code(
        """Query: "heat_cost_allocator, low_light, serial_unclear"
→ 2 similar past interactions found:

  1. guidance: "Use phone flashlight"
     outcome: success, effectiveness: 74%

  2. guidance: "Tilt phone down to capture serial"
     outcome: success, effectiveness: 82%
""",
        language="text",
    )
    st.markdown("**From device_expectations** (aggregated from confirmed readings):")
    st.code(
        """device_type: "heat_cost_allocator"
total_confirmed: 87

  serial_number  — seen in 98% of readings → expected
  display_value  — seen in 95% of readings → expected
  unit           — seen in 32% of readings → optional
""",
        language="text",
    )
    st.markdown("**From device_history** (consistency check):")
    st.code(
        """device: BH-2847
last reading: 1802 (3 months ago)
current:      1847 (increase of 45)
→ Consistent: counter increased as expected.
""",
        language="text",
    )

st.markdown("Decision Agent output:")
st.code(
    """{
  "routing": "sufficient",
  "operator_message": "Please confirm: Serial BH-2847, Reading 1847",
  "issues_identified": [],
  "guidance_focus": null
}""",
    language="json",
)

# --- Confirmation ---
st.markdown("---")
st.markdown("#### Operator confirms: the system learns")

st.markdown(
    "The operator confirms the reading. "
    "The system distributes to **7 destinations** simultaneously:"
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**ChromaDB** (vector search)")
    st.markdown(
        """
| Collection | What gets stored |
|---|---|
| **confirmed_images** | Text description + device_type, confirmed fields, image_path as metadata |
| **interaction_patterns** | One entry per retry turn: situation, guidance text, outcome, turns to success |
| **correction_patterns** | Only if operator corrected a field: what was wrong vs. what was right |
"""
    )

with col2:
    st.markdown("**SQLite** (structured queries)")
    st.markdown(
        """
| Table | What gets stored |
|---|---|
| **confirmed_readings** | Full record: confirmed fields, original fields, corrections, image path |
| **device_history** | Device identifier + reading value for future consistency checks |
| **calibration_data** | Per field: model confidence vs. actual correctness |
| **instruction_effectiveness** | Aggregated: which guidance resolved which issues, success rate |
"""
    )

st.markdown("---")

st.markdown(
    "Every piece stored here **enriches future turns**. "
    "The confirmed image becomes a few-shot example. "
    "The interaction pattern becomes proven guidance. "
    "The device history enables consistency checks. "
    "The calibration data tracks whether the model's confidence "
    "matches its actual accuracy."
)

st.markdown(
    "**This is the core loop.** The system does not retrain or fine-tune. "
    "It learns by accumulating context that gets injected into future prompts."
)
