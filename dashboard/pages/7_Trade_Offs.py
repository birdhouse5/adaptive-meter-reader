"""Page 7: Trade-offs.

Key technical decisions with alternatives and migration paths.
"""

import streamlit as st

st.header("Technical Trade-offs")

st.markdown(
    "Every choice optimizes for **learning speed**: "
    "get the system running, get confirmed readings flowing, let the knowledge base grow. "
    "Production hardening comes after the concept is proven."
)

st.divider()

# 1. Vision Model
st.subheader("1. Vision Model: GPT-4.1")

st.markdown(
    """
**Why:** Best-in-class vision understanding, works immediately with zero training data.

| | GPT-4.1 (chosen) | Fine-tuned model | Self-hosted open model |
|---|---|---|---|
| Cold start | Works immediately | Needs labeled data | Works, lower quality |
| Accuracy | Excellent | Best for specific devices | Varies |
| Cost per call | ~$0.01-0.03 | Infrastructure cost | Infrastructure cost |
| Data privacy | Images sent to API | Fully on-premise | Fully on-premise |

**Migration path:** The confirmed images database is exactly the labeled dataset
you need to fine-tune a self-hosted model later. GPT-4.1 funds its own replacement.
"""
)

st.divider()

# 2. Learning Approach
st.subheader("2. Learning: In-Context, Not Fine-Tuning")

st.markdown(
    """
**Why:** Every confirmed reading immediately improves the next interaction.
No training pipeline, no GPU infrastructure, no retraining cycles.

| | In-context learning (chosen) | Fine-tuning | Online learning |
|---|---|---|---|
| Adaptation speed | Instant (next request) | Batch (retrain periodically) | Continuous |
| Cold start | Immediate | Needs ~1000+ examples | Needs examples |
| Transparency | High (examples visible) | Low (baked into weights) | Low |
| Infrastructure | API calls only | GPU for training | Complex |

**Migration path:** Fine-tuning becomes viable and cost-effective once the confirmed
database is large enough. Not a migration, a natural evolution.
"""
)

st.divider()

# 3. Two-Agent Split
st.subheader("3. Architecture: Two Agents, Not One")

st.markdown(
    """
**Why:** The expensive vision call happens once. The decision call is text-only and ~10x cheaper.
Clear separation: perception vs. reasoning.

| | Two agents (chosen) | Single LLM call | Rules + LLM |
|---|---|---|---|
| Cost per turn | Vision + cheap text | One expensive call | Rules + one call |
| Debuggability | Each output inspectable | One black box | Rules are rigid |
| Adaptability | Knowledge feeds each agent differently | One prompt does everything | Rules don't learn |

**The key insight:** Sufficiency judgment and guidance generation are the same reasoning.
Splitting them into rules + LLM is an artificial boundary.
"""
)

st.divider()

# 4. Data Layer
st.subheader("4. Data: SQLite + ChromaDB (Embedded)")

st.markdown(
    """
**Why:** Zero infrastructure. Runs in-process, persists to disk. Right for a prototype.

| | SQLite + ChromaDB (chosen) | PostgreSQL + pgvector | Cloud managed |
|---|---|---|---|
| Setup | Zero | Server needed | Account needed |
| Scale | Prototype | Production | Production |
| Portability | Single file, runs anywhere | Requires database server | Requires cloud |

**Migration path:** The abstraction layer makes this a swap, not a rewrite.
"""
)
