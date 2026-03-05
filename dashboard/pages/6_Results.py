"""Page 6: Results.

Operational dashboard showing KPIs and the learning trend
from seeded data. This is what the system looks like during operation.
"""

import pandas as pd
import streamlit as st

from api import api_get

st.header("Results")

st.markdown(
    "This is the operational dashboard. "
    "These metrics update in real time as the system processes readings."
)

stats = api_get("/stats")

if not stats:
    st.info("Backend not reachable. Start the FastAPI server first.")
    st.stop()

# KPI row
kb_stats = stats.get("knowledge_base_stats", {})
kb_size = sum(kb_stats.get(k, 0) for k in kb_stats)
conf_stats = stats.get("confirmation_stats", {})

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Sessions", stats["total_sessions"])
c2.metric("Avg Turns to Success", stats["avg_turns_to_success"])
c3.metric("First-Attempt Rate", f"{stats['first_attempt_success_rate']:.0f}%")
c4.metric("Confirmed Readings", conf_stats.get("total_confirmed", 0))

st.divider()

# Turns-to-success trend (the key chart)
st.subheader("Turns-to-Success Over Time")
st.markdown("The core metric: as the knowledge base grows, readings take fewer attempts.")

sessions_all = api_get("/sessions?limit=500")

if sessions_all:
    completed = [
        s for s in sessions_all
        if s["status"] == "completed" and s["total_turns"] > 0
    ]
    if len(completed) > 1:
        completed.sort(key=lambda x: x["id"])
        df = pd.DataFrame(completed)
        df["rolling_avg"] = (
            df["total_turns"]
            .rolling(window=min(10, len(df)), min_periods=1)
            .mean()
        )
        st.line_chart(df.set_index("id")["rolling_avg"])
    else:
        st.info("Need more completed sessions to show trend.")

# Per device type
st.subheader("Per Device Type")
by_type = stats.get("by_meter_type", {})
if by_type:
    df_mt = pd.DataFrame(
        [
            {"Device Type": k, "Sessions": v["count"], "Avg Turns": v["avg_turns"]}
            for k, v in by_type.items()
        ]
    )
    st.dataframe(df_mt, use_container_width=True, hide_index=True)

# Guidance effectiveness
st.subheader("Guidance Effectiveness")
st.markdown("Which instructions actually resolve issues, and how often.")
effectiveness = stats.get("instruction_effectiveness", [])
if effectiveness:
    df_eff = pd.DataFrame(effectiveness)
    display_cols = [
        "situation_signature", "instruction_text",
        "times_used", "effectiveness_rate",
    ]
    existing_cols = [c for c in display_cols if c in df_eff.columns]
    st.dataframe(df_eff[existing_cols], use_container_width=True, hide_index=True)
else:
    st.info("No guidance data yet.")
