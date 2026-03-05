# System Architecture

## Architecture Overview

The Adaptive Meter Reader is an AI system that helps operators capture correct utility meter readings during on-site visits. The core challenge: meter devices vary enormously (heat cost allocators, water meters, gas meters), image quality is unpredictable (dark hallways, glare, obstructed devices), and the system must work reliably from day one while improving over time.

The system uses a **2-agent pipeline** — each upload triggers exactly two LLM calls. The **Vision Agent** (GPT-4.1, multimodal) performs open-ended extraction from the meter image: it discovers whatever fields are present (serial numbers, display values, device identifiers) without a prescribed schema, and reports per-field confidence scores alongside image quality assessment. The **Decision Agent** (GPT-4.1-mini, text-only) then evaluates the extraction and decides the routing: accept the reading as sufficient, request a retry with specific guidance, or escalate to a supervisor. This separation keeps the expensive vision call focused on perception while the cheaper text model handles judgment and communication.

The system learns from every confirmed reading through **three validation tiers**. First, operator confirmation: the user confirms or corrects the extracted values, producing labeled training data. Second, cross-session consistency: the system checks device identifier stability, reading monotonicity, and physical rate bounds across visits. Third, periodic supervisor verification of a sample of readings. All three tiers feed into a **knowledge base** of three ChromaDB collections — confirmed images (for few-shot prompting), interaction patterns (which guidance works), and correction patterns (known model errors). This knowledge enriches future prompts: the Vision Agent gets relevant examples on retries, and the Decision Agent gets device-type expectations, proven instructions, and calibrated confidence thresholds.

At **cold start** (week 1), the system operates on base prompts with conservative defaults. As confirmed readings accumulate (~200 by week 4), device-specific patterns emerge: the system learns what fields to expect per device type, which photo instructions actually help, and where the model tends to make errors. By week 8 (~500 readings), the system has mature per-device-type playbooks with calibrated confidence, proven guidance, and a rich few-shot library. This knowledge base then becomes the foundation for a Phase 2 tenant self-service extension — tenants start with a mature system, not a cold start.

The **optimization objective** is to minimize expected turns-to-success while keeping reading error rate below a threshold — measured against real ground truth from confirmations, not self-reported model confidence.

---

## 1. High-Level System Overview

The diagram below shows the full system topology: the 2-agent pipeline at the center, the knowledge base that grows from confirmed readings, the SQLite data layer that tracks sessions and device history, and the Streamlit dashboard for monitoring. Dashed lines represent knowledge enrichment — how confirmed data flows back into future prompts.

```mermaid
graph TB
    subgraph User["Operator / Tenant"]
        PHONE[Phone / Upload]
    end

    subgraph System["Adaptive Meter Reader"]
        direction TB
        ORCH[Orchestrator<br/><i>Session Manager</i>]

        subgraph Agents["Two-Agent Pipeline"]
            direction LR
            VIS[Vision Agent<br/><i>GPT-4.1 VLM</i><br/><i>open-ended extraction</i>]
            DEC[Decision Agent<br/><i>GPT-4.1-mini</i><br/><i>sufficiency + routing + guidance</i>]
        end

        subgraph Validation["Ground Truth"]
            direction LR
            CONFIRM[User Confirmation<br/><i>Confirm or correct</i>]
            CONSIST[Consistency Check<br/><i>Device history</i>]
        end

        subgraph Knowledge["Knowledge Base — grows from usage"]
            direction LR
            IMAGES[(Confirmed Images<br/><i>labeled by device type</i>)]
            PATTERNS[(Interaction Patterns<br/><i>what guidance works</i>)]
            CORRECTIONS[(Correction Patterns<br/><i>known model errors</i>)]
        end

        subgraph Data["Data Layer"]
            direction LR
            SQL[(SQLite<br/><i>Sessions, readings,<br/>device history</i>)]
        end
    end

    subgraph Dashboard["Streamlit Dashboard"]
        LIVE[Live Sessions]
        STATS[Metrics + Knowledge Growth]
        UPLOAD[Upload and Test]
    end

    PHONE -->|image upload| ORCH
    ORCH --> VIS
    VIS -->|extraction + quality| DEC
    DEC -->|retry + guidance| PHONE
    DEC -->|sufficient| CONFIRM
    CONFIRM -->|confirmed / corrected| SQL

    IMAGES -.->|few-shot examples| VIS
    CORRECTIONS -.->|known errors| VIS
    PATTERNS -.->|effective guidance| DEC
    CONSIST -.->|consistency signals| DEC
    SQL -.->|confirmed sessions| IMAGES
    SQL -.->|confirmed sessions| PATTERNS
    SQL -.->|corrections| CORRECTIONS
    SQL -.->|device history| CONSIST

    SQL --> Dashboard
```

## 2. Session Lifecycle

This sequence diagram traces a typical multi-turn interaction. Turn 1 produces an insufficient extraction (display value unclear), so the Decision Agent retrieves proven guidance from the knowledge base and instructs the operator to tilt the phone. Turn 2 succeeds with knowledge-enriched few-shot prompting, the operator confirms, and the confirmed session feeds back into the knowledge base — closing the learning loop.

```mermaid
sequenceDiagram
    participant U as Operator
    participant O as Orchestrator
    participant V as Vision Agent (VLM)
    participant D as Decision Agent (text)
    participant KB as Knowledge Base

    U->>O: Upload photo (turn 1)
    O->>V: Analyze image (base prompt)
    V->>V: Extract all identifiable data<br/>(open-ended, no prescribed fields)
    V->>O: Extraction result + confidences + quality scores

    O->>D: Evaluate extraction
    D->>KB: Device expectations?<br/>Consistency signals?<br/>What worked for similar situations?
    KB-->>D: Based on 47 confirmed readings:<br/>expect identifier + display value<br/>"Tilt phone down" resolved this 82% of the time
    D->>D: Judge sufficiency + generate response
    D->>O: Insufficient — display value unclear<br/>+ guidance for operator

    O->>U: "The display value is unclear.<br/>Tilt your phone down — the serial<br/>is usually on the bottom label."

    U->>O: Upload photo (turn 2)
    O->>V: Analyze image (enriched prompt)
    V->>KB: Retrieve similar confirmed images + error warnings
    KB-->>V: 3 similar devices, known digit confusion 1↔0
    V->>V: Extract with few-shot examples
    V->>O: All fields high confidence

    O->>D: Evaluate extraction
    D->>O: Sufficient — present for confirmation

    O->>U: "We read serial A7839201, value 04521. Correct?"
    U->>O: Confirmed ✓

    O->>KB: Store confirmed reading +<br/>labeled image + instruction outcome

    Note over KB: This confirmed session<br/>enriches future interactions:<br/>+1 labeled image<br/>+1 instruction effectiveness signal<br/>+1 device history entry
```

## 3. Knowledge Base Growth

Every confirmed session produces five distinct learning signals. These flow into four knowledge stores that directly improve system behavior: a device image library for few-shot prompting, learned field schemas for setting expectations, confidence calibration data for better accept/reject thresholds, and proven instruction templates for more effective guidance.

```mermaid
graph LR
    subgraph Input["Every Confirmed Session Produces"]
        direction TB
        A[Labeled Image<br/><i>photo + confirmed device type</i>]
        B[Field Structure<br/><i>what fields exist for this type</i>]
        C[Ground Truth Values<br/><i>the confirmed reading</i>]
        D[Extraction Quality<br/><i>model correct or wrong per field</i>]
        E[Instruction Outcome<br/><i>did the guidance work?</i>]
    end

    subgraph KB["Knowledge Base"]
        direction TB
        KB1[Device Image Library<br/><i>few-shot prompting<br/>→ Vision Agent</i>]
        KB2[Learned Field Schemas<br/><i>what to expect per type<br/>→ Decision Agent</i>]
        KB3[Confidence Calibration<br/><i>model confidence → actual accuracy<br/>→ Decision Agent</i>]
        KB4[Effective Instructions<br/><i>what works for what<br/>→ Decision Agent</i>]
    end

    subgraph Benefits["Feeds Back Into"]
        direction TB
        R1[Better device recognition]
        R2[Better extraction accuracy]
        R3[Better accept/reject decisions]
        R4[Better guidance]
    end

    Input --> KB
    KB --> Benefits
    Benefits -->|more confirmed readings| Input
```

## 4. Ground Truth Validation Tiers

The system never relies on model self-confidence alone. Three independent validation tiers provide ground truth. Tier 1 (user confirmation) is the primary source — every confirmed or corrected reading is a labeled data point. Tier 2 (cross-session consistency) catches physics violations for free by comparing readings across visits to the same device. Tier 3 (supervisor verification) provides periodic spot-checks that validate overall system accuracy.

```mermaid
graph TD
    subgraph Tier1["Tier 1 — User Confirmation"]
        direction TB
        T1A[Operator confirms or corrects]
        T1B["Confirmed image = labeled training data"]
        T1C["Correction = model error signal"]
    end

    subgraph Tier2["Tier 2 — Cross-Session Consistency"]
        direction TB
        T2A[Device identifier stability<br/>Reading monotonicity<br/>Rate bound checks]
        T2B[Physics violations = likely errors]
        T2C[Free, automatic, every session]
    end

    subgraph Tier3["Tier 3 — Supervisor Verification"]
        direction TB
        T3A[Spot-check sample of readings]
        T3B[High-confidence correction pairs]
        T3C[Validates overall system accuracy]
    end

    subgraph Learning["All Feed Into Knowledge Base"]
        direction TB
        L1[Confidence calibration]
        L2[Few-shot correction examples]
        L3[Instruction effectiveness]
        L4[Device type field expectations]
    end

    Tier1 --> Learning
    Tier2 --> Learning
    Tier3 --> Learning
```

## 5. Optimization Problem

The Decision Agent solves a constrained optimization problem every turn. The inputs are observed variables (image quality, extraction results, device context, user profile, session state, validation signals). The controls are the Decision Agent's choices: sufficiency judgment, issue prioritization, guidance content, and escalation timing. The objective is to minimize expected turns-to-success while keeping reading error rate below a threshold — where both metrics are measured against real ground truth, not model self-assessment.

```mermaid
graph TD
    subgraph Inputs["Input Variables — observed"]
        direction TB
        I1[Image Quality<br/><i>brightness, blur, angle,<br/>glare, occlusion</i>]
        I2[Extraction Results<br/><i>open-ended: whatever the<br/>model found + confidence</i>]
        I3[Device Context<br/><i>recognized type, number of<br/>similar confirmed images</i>]
        I4[User Profile<br/><i>operator/tenant, history,<br/>success rate, error patterns</i>]
        I5[Session State<br/><i>turn number, prior guidance,<br/>persisting issues</i>]
        I6[Validation Signals<br/><i>confirmation history,<br/>consistency checks,<br/>calibrated confidence</i>]
    end

    subgraph Controls["Control Variables — Decision Agent decides"]
        direction TB
        C1[Sufficiency Judgment<br/><i>is this reading usable?<br/>learned, not hardcoded</i>]
        C2[Issue Prioritization<br/><i>which problem first?<br/>based on effectiveness data</i>]
        C3[Guidance Content<br/><i>device-specific tips from<br/>confirmed examples</i>]
        C4[Escalation<br/><i>when to stop retrying</i>]
    end

    subgraph Objective["Objective"]
        OBJ["minimize E[turns to success]<br/>subject to error_rate < epsilon"]
    end

    Inputs --> OBJ
    Controls --> OBJ

    subgraph Metrics["Measurable Outcomes"]
        M1[Turns-to-success decreasing]
        M2[First-attempt success rate increasing]
        M3[Corrections per session decreasing]
        M4[Knowledge base growing]
        M5[Confidence calibration converging]
    end

    OBJ --> Metrics
```

## 6. System Evolution

This timeline shows how the system matures from cold start to production readiness. In week 1, the agents operate on base prompts with no examples. By week 4, device-specific patterns emerge from ~200 confirmed readings. By week 8, the system has mature per-type playbooks, calibrated confidence, and proven guidance. The accumulated knowledge base then enables Phase 2: tenant self-service, where tenants inherit a mature system rather than starting cold.

```mermaid
graph LR
    subgraph Week1["Week 1: Cold Start<br/>~0 confirmed readings"]
        W1A["Vision Agent: base prompt only<br/>Decision Agent: conservative defaults<br/>Generic guidance<br/>No few-shot examples"]
    end

    subgraph Week4["Week 4: Learning<br/>~200 confirmed readings"]
        W2A["Vision Agent: few-shot enriched on retries<br/>Decision Agent: device expectations emerging<br/>Instructions adapting to situations<br/>Confidence calibrating"]
    end

    subgraph Week8["Week 8: Mature<br/>~500 confirmed readings"]
        W3A["Vision Agent: rich few-shot library<br/>Decision Agent: per-type playbooks<br/>Proven guidance with effectiveness data<br/>Calibrated confidence"]
    end

    subgraph Phase2["Phase 2: Tenant Extension"]
        P2A["All knowledge transfers<br/>Tenants start with<br/>mature system<br/>Operator visit rate tracked"]
    end

    Week1 -->|"patterns emerge"| Week4
    Week4 -->|"per-type playbooks"| Week8
    Week8 -->|"knowledge base ready"| Phase2
```

## 7. Rollout Vision

The two-phase rollout strategy uses Phase 1 (operator tool) to organically build the knowledge base that Phase 2 (tenant self-service) depends on. Operators generate labeled training data through normal usage — no separate data collection effort needed. The knowledge base contents (device image library, discovered field schemas, proven instructions, calibration data, device history) all emerge from confirmed readings rather than manual configuration.

```mermaid
graph TB
    subgraph Phase1["Phase 1: Operator Tool"]
        direction TB
        P1A[Operators use system<br/>during on-site visits]
        P1B[System discovers device types<br/>from confirmed readings]
        P1C[Builds knowledge base<br/>organically from usage]
    end

    subgraph Knowledge["Knowledge Base — emergent, not pre-configured"]
        direction TB
        K1[Device image library<br/>labeled by type]
        K2[Field schemas per type<br/>discovered, not prescribed]
        K3[Effective instruction templates<br/>proven by outcomes]
        K4[Confidence calibration<br/>model accuracy per type]
        K5[Device reading history<br/>for consistency checks]
    end

    subgraph Phase2["Phase 2: Tenant Self-Service"]
        direction TB
        P2A["Tenants get a choice:<br/>Self-service (AI-guided)<br/>or schedule operator visit"]
        P2B[Starts with mature system<br/>not cold start]
        P2C[Reduces operator visits<br/>= cost savings]
    end

    Phase1 --> Knowledge
    Knowledge --> Phase2
```

## 8. Two-Agent Prompt Architecture

This diagram details the internal structure of one turn through the pipeline. The Vision Agent receives the image plus a base prompt (enriched with few-shot examples and error warnings on retries). It outputs open-ended extraction results with per-field confidences and image quality scores. The Decision Agent receives only the text extraction (not the image), enriched with device expectations, proven instructions, consistency signals, calibration data, and turn history. It outputs a routing decision and an operator-facing message.

```mermaid
graph TB
    subgraph Turn["One Turn"]
        direction TB

        subgraph Call1["Call 1: Vision Agent — GPT-4.1 (VLM)"]
            direction TB
            V_IN[Image + Base Prompt]
            V_KB["Knowledge Base enrichment<br/>(on retries only):<br/>• Similar confirmed images<br/>• Error pattern warnings"]
            V_OUT["Output:<br/>• Open-ended extraction<br/>• Per-field confidences<br/>• Image quality scores"]
        end

        subgraph Call2["Call 2: Decision Agent — GPT-4.1-mini (text-only)"]
            direction TB
            D_IN["Input: extraction result<br/>(not the image)"]
            D_KB["Knowledge Base enrichment<br/>(every turn):<br/>• Device expectations<br/>• Proven instructions<br/>• Consistency signals<br/>• Calibration data<br/>• Turn history"]
            D_OUT["Output:<br/>• Routing: sufficient / retry / escalate<br/>• Operator-facing message<br/>(confirmation or guidance)"]
        end
    end

    V_IN --> V_OUT
    V_KB -.-> V_OUT
    V_OUT --> D_IN
    D_KB -.-> D_OUT
    D_IN --> D_OUT
```
