"""Page 4 — Live Demo.

Upload & test interface with turn history, prominent operator messages,
image quality badges, and session reset.

See docs/architecture.md §2 for the session lifecycle this page exercises.
"""

import streamlit as st

from api import api_post

st.header("Live Demo")

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "demo_session_id" not in st.session_state:
    st.session_state.demo_session_id = None
if "turn_history" not in st.session_state:
    st.session_state.turn_history = []
if "awaiting_confirmation" not in st.session_state:
    st.session_state.awaiting_confirmation = False
if "uploaded_bytes" not in st.session_state:
    st.session_state.uploaded_bytes = None
    st.session_state.uploaded_name = None

# ---------------------------------------------------------------------------
# Turn history display
# ---------------------------------------------------------------------------
if st.session_state.turn_history:
    st.markdown("#### Session History")
    for i, turn in enumerate(st.session_state.turn_history, 1):
        with st.container():
            tcol1, tcol2 = st.columns([1, 3])
            with tcol1:
                st.image(turn["image_bytes"], caption=f"Turn {i}", width=180)
            with tcol2:
                result = turn["result"]
                routing = result.get("routing", "?")
                if routing == "sufficient":
                    st.success(f"Turn {i}: Sufficient")
                elif routing == "retry":
                    st.warning(f"Turn {i}: Retry needed")
                else:
                    st.error(f"Turn {i}: Escalated")

                # Image quality badge
                iq = result.get("image_quality", {})
                usability = iq.get("overall_usability", 0) if isinstance(iq, dict) else 0
                if usability >= 0.7:
                    badge = ":green[Good]"
                elif usability >= 0.4:
                    badge = ":orange[Fair]"
                else:
                    badge = ":red[Poor]"
                st.markdown(f"Image quality: {badge}")

                msg = result.get("operator_message", "")
                if msg:
                    st.markdown(f"*{msg}*")
        st.divider()

# ---------------------------------------------------------------------------
# Upload area
# ---------------------------------------------------------------------------
st.markdown("#### Upload a Meter Photo")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded:
    st.session_state.uploaded_bytes = uploaded.getvalue()
    st.session_state.uploaded_name = uploaded.name
    st.session_state.uploaded_type = uploaded.type

col_start, col_retry, col_reset = st.columns(3)

with col_start:
    start_btn = st.button(
        "Start New Session",
        disabled=st.session_state.uploaded_bytes is None,
    )

with col_retry:
    retry_btn = st.button(
        "Upload Retry",
        disabled=(
            st.session_state.uploaded_bytes is None
            or st.session_state.demo_session_id is None
        ),
    )

with col_reset:
    if st.button("Reset Session"):
        st.session_state.demo_session_id = None
        st.session_state.turn_history = []
        st.session_state.awaiting_confirmation = False
        st.session_state.uploaded_bytes = None
        st.session_state.uploaded_name = None
        st.rerun()


def _process_result(result):
    """Store result in turn history and update session state."""
    if not result:
        return
    st.session_state.turn_history.append(
        {"image_bytes": st.session_state.uploaded_bytes, "result": result}
    )
    if result.get("routing") == "sufficient":
        st.session_state.awaiting_confirmation = True
    elif result.get("routing") in ("escalate",):
        st.session_state.demo_session_id = None


# ---------------------------------------------------------------------------
# Start / Retry actions
# ---------------------------------------------------------------------------
if start_btn and st.session_state.uploaded_bytes:
    st.session_state.awaiting_confirmation = False
    st.session_state.turn_history = []
    with st.spinner("Processing..."):
        result = api_post(
            "/session/start",
            files={
                "file": (
                    st.session_state.uploaded_name,
                    st.session_state.uploaded_bytes,
                    st.session_state.uploaded_type,
                )
            },
        )
    if result:
        st.session_state.demo_session_id = result.get("session_id")
        _process_result(result)
        st.rerun()

if retry_btn and st.session_state.uploaded_bytes and st.session_state.demo_session_id:
    st.session_state.awaiting_confirmation = False
    sid = st.session_state.demo_session_id
    with st.spinner("Processing retry..."):
        result = api_post(
            f"/session/{sid}/upload",
            files={
                "file": (
                    st.session_state.uploaded_name,
                    st.session_state.uploaded_bytes,
                    st.session_state.uploaded_type,
                )
            },
        )
    if result:
        _process_result(result)
        st.rerun()

# ---------------------------------------------------------------------------
# Current result display
# ---------------------------------------------------------------------------
if st.session_state.turn_history:
    latest = st.session_state.turn_history[-1]["result"]

    # Prominent operator message
    msg = latest.get("operator_message", "")
    if msg:
        st.markdown(f"### {msg}")

    # Extracted fields
    fields = latest.get("extracted_fields", {})
    if fields:
        st.markdown("**Extracted fields:**")
        for field_name, field_data in fields.items():
            if isinstance(field_data, dict):
                val = field_data.get("value", "?")
                conf = field_data.get("confidence", 0)
                color = (
                    "green" if conf >= 0.8 else "orange" if conf >= 0.5 else "red"
                )
                st.markdown(
                    f"&nbsp;&nbsp;**{field_name}**: {val} "
                    f"<span style='color:{color}'>({conf:.0%})</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"&nbsp;&nbsp;**{field_name}**: {field_data}")

    # Decision reasoning (collapsible)
    reasoning = latest.get("decision_reasoning", "")
    if reasoning:
        with st.expander("Decision reasoning"):
            st.write(reasoning)

    # -------------------------------------------------------------------
    # Confirmation UI
    # -------------------------------------------------------------------
    if st.session_state.awaiting_confirmation and latest.get("routing") == "sufficient":
        st.divider()
        st.subheader("Confirm Reading")
        st.write(
            "Please confirm the extracted reading is correct, or provide corrections."
        )

        confirm_col, correct_col = st.columns(2)

        with confirm_col:
            if st.button("Confirm as correct"):
                flat_fields = {}
                for k, v in fields.items():
                    flat_fields[k] = v.get("value") if isinstance(v, dict) else v

                confirm_result = api_post(
                    f"/session/{latest['session_id']}/confirm",
                    json={"confirmed_fields": flat_fields},
                )
                if confirm_result:
                    st.success("Reading confirmed! Knowledge base updated.")
                    st.session_state.awaiting_confirmation = False

        with correct_col:
            with st.form("correction_form"):
                st.write("Enter corrections:")
                corrections = {}
                for k, v in fields.items():
                    current = v.get("value") if isinstance(v, dict) else v
                    new_val = st.text_input(f"{k}", value=str(current or ""))
                    if new_val != str(current or ""):
                        corrections[k] = new_val

                if st.form_submit_button("Submit corrections"):
                    flat_fields = {}
                    for k, v in fields.items():
                        flat_fields[k] = v.get("value") if isinstance(v, dict) else v

                    confirm_result = api_post(
                        f"/session/{latest['session_id']}/confirm",
                        json={
                            "confirmed_fields": flat_fields,
                            "corrections": corrections if corrections else None,
                        },
                    )
                    if confirm_result:
                        st.success("Corrections submitted! Knowledge base updated.")
                        st.session_state.awaiting_confirmation = False
