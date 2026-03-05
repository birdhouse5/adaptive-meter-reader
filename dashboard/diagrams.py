"""Simple presentation diagrams for the dashboard.

Three diagrams only — pipeline, learning loop, rollout.
Render with ``render_mermaid(DIAGRAM)`` in any page.
"""

import streamlit.components.v1 as components


def render_mermaid(code: str, height: int = 350) -> None:
    """Render a Mermaid diagram using the Mermaid JS CDN."""
    html = f"""
    <script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
    <style>
      .mermaid svg {{ max-width: none !important; height: auto !important; }}
    </style>
    <div class="mermaid">
    {code}
    </div>
    <script>mermaid.initialize({{startOnLoad:true, theme:'neutral', flowchart:{{curve:'basis'}}}});</script>
    """
    components.html(html, height=height, scrolling=True)


# Diagram 1 — The Pipeline
PIPELINE = """\
graph LR
    PHOTO["📷 Photo"] --> VIS["Vision Agent<br/>sees the image"]
    VIS --> DEC["Decision Agent<br/>judges the reading"]
    DEC --> OK["✅ Sufficient"]
    DEC --> RETRY["🔄 Retry + Guidance"]
    KB[("Knowledge Base")] -.->|enriches| VIS
    KB -.->|informs| DEC
"""

# Diagram 2 — The Learning Loop
LEARNING_LOOP = """\
graph LR
    A["Operator confirms reading"] --> B["Knowledge Base grows"]
    B --> C["Better guidance next time"]
    C --> D["Fewer retries needed"]
    D --> A
"""

# KB Diagrams — one per collection, showing store + retrieve flow

KB_CONFIRMED_IMAGES = """\
graph LR
    subgraph STORE["Stored when operator confirms"]
        direction TB
        S1["Device description<br/>(text, embedded)"]
        S2["device_type + confirmed_fields<br/>(metadata)"]
    end

    subgraph RETRIEVE["Retrieved by Vision Agent on retries"]
        direction TB
        R1["Query: current situation description"]
        R2["Result: top 3 similar devices<br/>(cosine similarity, max distance 0.4)"]
    end

    STORE --> DB[("confirmed_images<br/>ChromaDB")]
    DB --> RETRIEVE
"""

KB_INTERACTION_PATTERNS = """\
graph LR
    subgraph STORE["Stored when session completes"]
        direction TB
        S1["Situation description<br/>(text, embedded)"]
        S2["guidance_text + outcome<br/>+ turns_to_success<br/>(metadata)"]
    end

    subgraph RETRIEVE["Retrieved by Decision Agent every turn"]
        direction TB
        R1["Query: current situation description"]
        R2["Result: ranked guidance<br/>with success rates"]
    end

    STORE --> DB[("interaction_patterns<br/>ChromaDB")]
    DB --> RETRIEVE
"""

KB_CORRECTION_PATTERNS = """\
graph LR
    subgraph STORE["Stored when operator corrects a field"]
        direction TB
        S1["Error description<br/>(text, embedded)"]
        S2["field_name + original_value<br/>+ corrected_value<br/>(metadata)"]
    end

    subgraph RETRIEVE["Retrieved by Vision Agent on retries"]
        direction TB
        R1["Query: current extraction description"]
        R2["Result: known errors<br/>for similar devices"]
    end

    STORE --> DB[("correction_patterns<br/>ChromaDB")]
    DB --> RETRIEVE
"""

# Diagram — Unified data flow (How It Works page)
DATA_FLOW = """\
graph TD
    PHOTO["📷 Operator uploads photo"]
    PHOTO --> VIS["Vision Agent<br/>GPT-4.1 multimodal"]

    subgraph KB_READ_VIS [" "]
        direction TB
        KR1["confirmed_images<br/>few-shot examples"]
        KR2["correction_patterns<br/>error warnings"]
    end

    KB_READ_VIS -.->|"on retries"| VIS

    VIS -->|"extracted_fields<br/>image_quality<br/>description"| DEC["Decision Agent<br/>GPT-4.1-mini text-only"]

    subgraph KB_READ_DEC [" "]
        direction TB
        KR3["interaction_patterns<br/>proven guidance"]
        KR4["device_expectations<br/>field frequencies"]
        KR5["device_history<br/>consistency check"]
    end

    KB_READ_DEC -.->|"every turn"| DEC

    DEC -->|"sufficient"| CONFIRM["Operator confirms or corrects"]
    DEC -->|"retry + guidance"| PHOTO

    CONFIRM -->|"stores"| KB_WRITE["3 ChromaDB collections<br/>+ 4 SQLite tables"]
    KB_WRITE -.->|"enriches future turns"| KB_READ_VIS
    KB_WRITE -.->|"enriches future turns"| KB_READ_DEC
"""

# Diagram 3 — The Rollout
ROLLOUT = """\
graph LR
    P1["Phase 1<br/>Operators build<br/>knowledge base"] --> KB[("Knowledge Base<br/>devices, guidance,<br/>calibration")]
    KB --> P2["Phase 2<br/>Tenants inherit<br/>mature system"]
"""
