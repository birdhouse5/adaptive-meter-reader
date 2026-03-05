# Technical Decisions & Trade-offs

Every architectural choice in this system optimizes for one thing: **get confirmed readings flowing as fast as possible so the knowledge base can grow**. Production hardening comes after the concept is proven. Each section below states what was chosen, why, what the alternatives are, and the upgrade path.

See `docs/architecture.md` for system diagrams.

---

## 1. Agent Architecture: 2-Agent Pipeline vs Alternatives

### Chosen: Vision Agent (GPT-4.1) + Decision Agent (GPT-4.1-mini)

Each turn makes exactly 2 LLM calls with clear separation — the Vision Agent *sees* (image → structured extraction), the Decision Agent *thinks* (extraction → routing + guidance). The Decision Agent never sees the image; it works from structured text only, which means it can use a cheaper, faster model.

### Why this split

- **Cost**: The expensive vision call (GPT-4.1, ~$0.01–0.03) happens once. The reasoning call (GPT-4.1-mini, ~10x cheaper) handles sufficiency judgment, routing, and guidance generation in one step.
- **Separation of concerns**: Perception and reasoning are independent capabilities. The Vision Agent's prompt can be enriched with few-shot examples without affecting routing logic. The Decision Agent's context can include device expectations and consistency signals without re-running the vision model.
- **Testability**: Each agent's output is independently inspectable and testable. Mocked LLM responses in CI tests verify the pipeline without API calls.

### Alternatives considered

| Approach | Description | Pro | Con |
|----------|-------------|-----|-----|
| **Single LLM call** | One prompt does extraction + routing + guidance | Fewer API calls, lower latency | Can't use a cheaper model for reasoning; prompt becomes unwieldy; harder to test and debug |
| **3-agent pipeline** (original design) | Vision + Quality Gate (rules) + Instruction Agent (LLM) | Explicit quality rules | Sufficiency and guidance are the same reasoning — splitting them is artificial. Rules can't handle "the serial looks plausible for this device type." |
| **Agent framework** (LangChain, CrewAI, AutoGen) | Framework-managed agent orchestration | Built-in state management, visualization | See Section 10 below |
| **Monolithic pipeline** | No agents — single function processes everything | Simplest to build | No modularity, can't swap components, poor demo value, doesn't match "agentic AI" requirement |

### Why not 3 agents?

The original design had a Quality Gate (rules-based sufficiency check) and an Instruction Agent (LLM-generated guidance) as separate steps. We merged them into the Decision Agent because:

1. Sufficiency judgment and guidance generation are the same reasoning — "what's wrong?" leads directly to "what should the operator do about it?"
2. Rules can't capture nuanced judgments like "the reading value is suspiciously round" or "given what worked last time, try this."
3. One LLM call handles both the accept and reject paths — it either confirms the reading or explains what to improve.
4. All knowledge base context goes to one place, composed coherently.

---

## 2. Vision Model: GPT-4.1

### Why GPT-4.1

- Best-in-class vision understanding for structured extraction from photos
- Works immediately with zero training data — critical for cold start
- Handles open-ended extraction ("tell me what you see") without a predefined schema
- JSON output mode makes structured extraction reliable

### Alternatives

| Consideration | GPT-4.1 (chosen) | Self-hosted open model (LLaVA, Qwen-VL) | Fine-tuned model | Traditional CV (YOLO + OCR) |
|---|---|---|---|---|
| Cold start | Works immediately | Works immediately, lower quality | Needs labeled data first | Needs training per device type |
| Extraction quality | Excellent | Good, varies by model | Potentially best for specific devices | Rigid, no reasoning |
| Cost per call | ~$0.01–0.03 | Infrastructure cost only | Infrastructure + training | Very low |
| Data privacy | Images sent to OpenAI API | Fully on-premise | Fully on-premise | Fully on-premise |
| Flexibility | Open-ended extraction | Often needs more rigid prompts | Locked to training schema | Each new type needs retraining |

### Data privacy note

API calls to GPT-4.1 mean images leave the infrastructure. For a prototype this is acceptable. For production:
- OpenAI doesn't train on API data by default
- Azure OpenAI provides EU data residency
- Self-hosted alternatives become viable when the confirmed database is large enough to fine-tune

### Upgrade path

The confirmed images database is exactly the labeled dataset needed to fine-tune a self-hosted model. GPT-4.1 funds its own replacement.

---

## 3. Extraction Strategy: Open-Ended vs Schema-Driven

### Chosen: Open-ended extraction

The Vision Agent extracts "whatever it sees" rather than filling a prescribed template. The prompt says: *"Extract all identifiable structured data you can find. Report what you see."*

### Why

- No assumptions about device types or field structures
- Works for any device type without reconfiguration
- The system discovers what exists rather than failing on what's missing
- Field schemas emerge from confirmed readings — not from configuration

### Trade-off

Open-ended extraction produces less predictable JSON shapes. The Decision Agent judges "sufficiency" holistically rather than checking a field checklist. This is harder to implement but more robust.

### How cold start works

On the first interaction with zero confirmed readings, GPT-4.1 works from its general training data. It already knows what meter readings look like and will extract serial numbers, display values, units, etc. without a checklist. The learning loop then reinforces what it discovered: after the first confirmed Brunata HCA reading, the `DeviceExpectations` module knows what fields to expect, and the Decision Agent gets that context.

### Mitigation

After enough confirmed readings of a device type, the `DecisionContextBuilder` injects implicit structure: *"Devices like this typically have fields: serial_number (100%), display_value (100%)."* The structure emerges from data, not from configuration.

---

## 4. Learning Approach: In-Context vs Fine-Tuning

### Chosen: In-context learning (few-shot prompting from knowledge base)

Every confirmed reading immediately improves the next interaction — no training pipeline, no GPU infrastructure, no retraining cycles.

### How it works

1. Operator confirms a reading → `SessionProcessor` distributes learning signals to 3 ChromaDB collections
2. Next similar image → `VisionPromptBuilder` retrieves confirmed examples and correction warnings → injects into the Vision Agent prompt
3. Next similar situation → `DecisionContextBuilder` retrieves proven guidance and device expectations → injects into the Decision Agent prompt

### Alternatives

| Consideration | In-context learning (chosen) | Fine-tuning | RAG + fine-tuning |
|---|---|---|---|
| Cold start | Immediate | Needs ~1000+ examples | Needs examples |
| Adaptation speed | Instant (next request) | Batch (retrain periodically) | Mix of both |
| Context window | Limited to ~3–5 examples | Unlimited training data | Hybrid |
| Transparency | High — examples are visible in the prompt | Low — baked into weights | Medium |
| Cost per request | Higher (more tokens) | Lower (shorter prompts) | Medium |
| Infrastructure | API calls only | GPU for training | Both |

### Upgrade path

In-context learning is right for 0–1000 confirmed readings. As the confirmed database grows, fine-tuning becomes viable. The confirmed readings database is exactly the training set needed. This isn't a migration — it's a natural evolution.

---

## 5. Knowledge Base Retrieval: Similarity Search

### How it works technically

ChromaDB uses HNSW (Hierarchical Navigable Small World) as its indexing algorithm with cosine distance. Documents are embedded using Sentence Transformers (`all-MiniLM-L6-v2`, 384-dimensional vectors). Similarity is semantic, not keyword-based — "poor lighting" and "underexposed" end up in similar regions of the vector space.

Three collections serve different purposes:

| Collection | Stored when | Retrieved by | Purpose |
|---|---|---|---|
| `confirmed_images` | Reading confirmed | `VisionPromptBuilder` | Few-shot examples for Vision Agent |
| `interaction_patterns` | Session completed | `DecisionContextBuilder` | Proven guidance for Decision Agent |
| `correction_patterns` | Reading corrected | `VisionPromptBuilder` | Error warnings for Vision Agent |

### The distance threshold problem

ChromaDB returns the top N nearest neighbors **regardless of how far away they are**. If the only stored interaction is about a water meter in bright sunlight, and the current situation is a heat cost allocator in a dark cellar, the cosine distance might be 0.85 (very dissimilar) — but it's still returned because it's the "closest" of one.

Injecting irrelevant context into an agent's prompt would confuse rather than help.

**Solution**: All retrieval methods apply a `max_distance=0.4` threshold. Results beyond this distance are filtered out. The knowledge base returns nothing rather than something misleading. This is correct behavior — on cold start with no similar examples, the agents work from their base prompts, which is the right fallback.

### The sequencing problem

The Vision Agent benefits from few-shot examples of similar devices. But on turn 1, the image hasn't been analyzed yet — we don't know the device type. How do you search for "similar" when you don't know what you're similar *to*?

| Approach | Pro | Con |
|---|---|---|
| **Two-pass extraction** — fast first pass classifies, second pass uses examples | Best of both worlds | Doubles cost/latency for Vision Agent |
| **Visual embedding (CLIP)** — compute image embedding before LLM call, search by visual similarity | Finds similar devices from appearance. Fast (~100ms). | Additional model dependency |
| **Enrich on retry only** (chosen) — turn 1 uses base prompt, turn 2+ gets enriched | Simplest. No extra models or calls. | First attempt doesn't benefit from knowledge base |

**Why "enrich on retry only" is correct**: First attempts on clear images already succeed from GPT-4.1's general capabilities. Knowledge base enrichment matters most on difficult cases — which are exactly the ones that trigger retries. As the knowledge base grows, adding CLIP embeddings for turn-1 enrichment becomes the natural next step.

### Scale

HNSW query time is O(log n). 10 entries and 100,000 entries have roughly similar latency. Only the top 3–5 results are retrieved per query, so prompt size stays bounded regardless of knowledge base size.

ChromaDB is embedded (single-process, file-based). It works well up to ~1M documents. Beyond that, swap to a managed vector database. The `KnowledgeBase` class abstracts the storage — changing the backend doesn't change the interface.

---

## 6. Relational Data: SQLite

### Why SQLite

- Zero-config, embedded, single file
- SQL for structured queries (stats, aggregations, history lookups)
- Easy to inspect and debug

### Alternatives

| Consideration | SQLite (chosen) | PostgreSQL | Cloud DB |
|---|---|---|---|
| Setup | Zero | Server needed | Account needed |
| Concurrent writes | Limited (single writer) | Excellent | Excellent |
| Scale | Prototype | Production | Production |
| Inspection | Open the file | Need client | Web UI |

### Upgrade path

SQLite is correct for a single-user prototype. For multi-operator production, move to PostgreSQL. Schema stays the same.

---

## 7. Vector Store: ChromaDB

### Why ChromaDB

- Zero infrastructure — runs in-process, persists to disk
- Python-native, cosine similarity search
- Sufficient for prototype scale (thousands of vectors)

### Alternatives

| Consideration | ChromaDB (chosen) | PostgreSQL + pgvector | Pinecone / Weaviate |
|---|---|---|---|
| Setup | Zero — pip install | Needs PostgreSQL | Cloud account or server |
| Scale | ~100K vectors | Millions | Millions+ |
| Query speed | Fast for prototype | Fast with indexes | Very fast, managed |
| Production ready | No — single process | Yes | Yes |

### Upgrade path

Migrate to pgvector (if already using PostgreSQL) or a managed vector service. The `KnowledgeBase` class makes this a swap, not a rewrite.

---

## 8. API Framework: FastAPI

### Why FastAPI

- Async by default — important when waiting on LLM API calls
- Auto-generated OpenAPI docs (useful for demo and testing)
- Pydantic integration for request/response validation
- Lightweight, well-documented

### Alternatives considered

- **Flask** — simpler but synchronous, poor for API-heavy workloads
- **Django** — too heavy for a prototype
- **Express/Node** — would split the stack (Python for ML, Node for API)

No significant trade-off. FastAPI is the right choice for a Python-based async API prototype.

---

## 9. Dashboard: Streamlit

### Why Streamlit

- Fastest path from data to interactive UI in Python
- Good charting, table, and file upload support
- Live refresh capability
- No frontend code needed

### Alternatives

| Consideration | Streamlit (chosen) | React/Next.js | Gradio |
|---|---|---|---|
| Development speed | Very fast | Slow (full frontend) | Fast |
| Interactivity | Good for demo | Full control | Good for ML demos |
| Production use | Limited | Yes | Limited |
| Custom UI | Limited | Full control | Limited |

### Upgrade path

Streamlit is right for the demo. A production operator-facing app would be mobile-first (React Native or Flutter). The API stays the same — only the frontend changes.

---

## 10. Agent Frameworks: LangChain, CrewAI, AutoGen

### Why not used

The current system has 2 agents with fixed topology — Vision → Decision, every turn, no exceptions. There's no dynamic routing, no agent-to-agent negotiation, no tool calling.

| Framework | What it adds | Why it doesn't fit |
|---|---|---|
| **LangChain / LangGraph** | Chain composition, memory, tool-use, streaming | Abstraction overhead. The learning system requires full control over what context reaches each LLM call — frameworks abstract this away. |
| **CrewAI** | Role-based agents, delegation | Opinionated structure, less prompt control |
| **AutoGen** | Multi-agent conversation | Designed for chat-based collaboration, poor fit for a pipeline |
| **OpenAI Assistants API** | Managed threads, tool-use | Vendor lock-in, less control, cost per stored thread |

The orchestrator is ~80 lines of code. A framework would add dependency churn and abstraction layers without proportional benefit.

### When frameworks become relevant

If the system grows beyond 3 agents with dynamic interactions (e.g., tenant self-service with separate routing logic, a supervisor review agent, integration agents for building management systems), a framework provides value. Until then, explicit orchestration is simpler and more transparent.

---

## 11. MCP (Model Context Protocol) and Tool-Use

### What MCP is

MCP is a standardized protocol (led by Anthropic) that sits between agents and their capabilities. MCP servers expose "tools" and "resources" that any MCP-compatible client can consume — similar to how HTTP standardized web communication.

### Why not used

| Dimension | Direct API (chosen) | MCP |
|---|---|---|
| Complexity | Minimal — just API calls | Additional protocol layer, server process |
| Provider switching | Change SDK + prompts | Swap MCP server, client stays the same |
| Edge deployment | Requires network to OpenAI | MCP server could wrap a local model |
| Maturity | Production-proven | Emerging standard |

For a prototype with one LLM provider and no external tool integrations, MCP adds complexity without benefit.

### Tool-use (function calling) vs prompt-based responses

The current agents return structured JSON via prompt instructions — no OpenAI function calling or tool-use patterns.

| Dimension | Prompt-based JSON (chosen) | Tool-use |
|---|---|---|
| Agent autonomy | None — pipeline is deterministic | Agent decides when to call tools |
| Learning control | Full — we inject context before the call | Partial — agent decides what to retrieve |
| Cost | 1 LLM call per agent per turn | Potentially N calls if agent loops |
| Predictability | High | Lower — agent may make unexpected tool calls |

The knowledge base enrichment is **deterministic** — `VisionPromptBuilder` and `DecisionContextBuilder` decide what context to inject, not the LLM. Tool-use would make agents autonomous in their retrieval, which introduces unpredictability and makes the learning loop harder to reason about.

### When MCP and tool-use become relevant

- **Model portability**: MCP enables swapping GPT-4.1 for a local model at the edge without changing client code
- **External integrations**: Tool-use makes sense when agents need to dynamically query external systems (meter registries, building management databases)
- **Tenant self-service**: A tenant-facing agent might need tool access to look up their building's device inventory

These are future concerns. The architecture supports adding them — the agent interfaces are clean abstractions — but including them now would be over-engineering.

---

## 12. Test Images: AI-Generated via Replicate

### Why

The system needs images to test against before any real operator data exists. Real meter photos are hard to obtain (privacy, physical access, volume), so we generate photorealistic device images using Replicate's nano-banana-pro model.

### What was generated

30 images across 4 device categories (heat cost allocators, water meters, gas meters, electricity meters) with deliberate quality degradations: low light, blur, glare, steep angle, partial occlusion. This lets us exercise the full pipeline — including retry guidance — without a single real photo.

### Trade-off

Generated images are realistic but not real. GPT-4.1's extraction performance on generated images may not perfectly match real-world performance. This is a prototype/demo concern — in production, the system uses real operator photos from day one, and the generated images become irrelevant.

### Why this matters for the demo

Without test images, the live demo would require photographing actual meters. Generated images let us demonstrate the full interaction loop — extraction, retry guidance, confirmation, knowledge base growth — in a controlled setting with predictable scenarios.

---

## 13. Ground Truth: Three Validation Tiers

### Why three tiers

A system that learns from its own output is circular. Validation from independent sources breaks the loop.

| Tier | Source | Reliability | Cost | Frequency |
|---|---|---|---|---|
| **User confirmation** | Operator confirms or corrects inline | High (~85% for operators) | Zero — part of the workflow | Every session |
| **Cross-session consistency** | Device history: serial stability, reading monotonicity, rate bounds | Medium — catches physics violations | Zero — automatic | Every session with history |
| **Supervisor verification** | Periodic spot-checks on a sample | Highest | Manual effort | Sampled |

Operator confirmations are not infallible — operators work under time pressure and make mistakes. The system accounts for this: cross-session consistency catches errors that operators miss, and supervisor spot-checks calibrate overall accuracy.

### Confidence calibration

The `CalibrationTracker` records model confidence vs actual correctness (from confirmed readings). This reveals systematic over- or under-confidence per device type and field. The Decision Agent can then make better sufficiency judgments — not based on raw model confidence, but on calibrated accuracy.

---

## 14. CI/CD Pipeline

### Prototype (current)

GitHub Actions runs on every push and pull request to `main`:

```
PR opened / push to main
  → Lint (ruff check)
  → Format check (ruff format --check)
  → Test (pytest -m "not live", 42 tests, all LLM calls mocked)
  → Matrix: Python 3.11 + 3.12
```

Live tests (`test_live.py`, 2 tests with real API calls) are excluded from CI via the `live` marker. They run manually during development to verify end-to-end behavior against the real OpenAI API.

**Why no CD step in the prototype**: There is no deployment target. The system runs locally. Adding a deploy step to nowhere would be theater — and would require secrets management, a container registry, and a target environment that don't exist yet.

### Production CI/CD pipeline

```
PR opened
  → Lint + format check (ruff)
  → Unit tests (mocked LLM, ~30s)
  → Integration tests (test containers for PostgreSQL + ChromaDB)
  → Build container image (Docker)
  → Push to staging registry

Merge to main
  → All of the above
  → Push to production registry
  → Deploy to staging environment
  → Run smoke tests against staging (health check, test upload, test confirm)
  → Manual approval gate
  → Deploy to production (rolling update)

Scheduled (weekly)
  → Run live tests against staging with real API key
  → Knowledge base backup verification
  → Dependency vulnerability scan
```

### CI/CD trade-offs

| Decision | Chosen | Alternative | Rationale |
|---|---|---|---|
| **CI platform** | GitHub Actions | GitLab CI, Jenkins, CircleCI | Already in GitHub. No additional infrastructure. YAML-based, good marketplace of actions. |
| **Deployment strategy** | Rolling update | Blue-green, canary | Rolling is simplest. The system is stateless (state lives in PostgreSQL + ChromaDB) so any instance can serve any request. Blue-green adds complexity for zero benefit at low traffic. Canary makes sense at scale when you want to test model changes on a subset of users. |
| **Container runtime** | Docker on Cloud Run / ECS | Kubernetes, bare VM, serverless functions | Cloud Run / ECS provide managed container hosting with auto-scaling and zero cluster management. Kubernetes is overkill for a single-service application. Serverless functions (Lambda) don't fit — LLM calls take 3-5 seconds, which clashes with cold starts and timeout limits. |
| **Deployment trigger** | Push to main + manual gate | GitOps (ArgoCD/Flux), scheduled releases | Push-to-main with a manual approval gate balances speed and safety. GitOps is appropriate for multi-service environments. Scheduled releases (weekly/monthly) are too slow for a system that needs to iterate on prompts and learning logic. |
| **Test strategy split** | Mocked CI + manual live tests | All live in CI, all mocked | Mocked tests are fast, free, deterministic. Live tests verify real LLM behavior but are slow (~10s per test), cost money, and can flake due to API variability. Running live tests in CI on every push is wasteful. Scheduled weekly runs catch regressions without burning API budget. |
| **Secrets management** | GitHub Actions secrets | HashiCorp Vault, AWS Secrets Manager | GitHub Actions secrets are sufficient for a single-repo deployment. Vault adds value in multi-service environments with secret rotation requirements. |

---

## 15. Deployment Topology

### Prototype (current)

```
Single machine:
  FastAPI server (port 8000)
  Streamlit dashboard (port 8501)
  SQLite file (data/processing.db)
  ChromaDB directory (data/chroma_data/)
```

Zero infrastructure, zero configuration beyond an API key. Clone, install, run.

### Production (Phase 1 — operator tool)

```
┌─────────────────────────────────────────────────────┐
│  Cloud Provider (AWS / GCP / Azure)                 │
│                                                     │
│  ┌──────────────┐     ┌──────────────────────────┐  │
│  │ Container     │     │ Managed Services          │  │
│  │ Runtime       │     │                           │  │
│  │ (Cloud Run /  │────▶│ PostgreSQL (RDS / Cloud   │  │
│  │  ECS Fargate) │     │   SQL) + pgvector         │  │
│  │               │     │                           │  │
│  │ FastAPI       │────▶│ Object Storage (S3 / GCS) │  │
│  │ application   │     │   for uploaded images     │  │
│  └──────┬───────┘     │                           │  │
│         │              │ Redis (ElastiCache)       │  │
│         │              │   for rate limiting       │  │
│         │              └──────────────────────────┘  │
│         │                                            │
│         ▼                                            │
│  ┌──────────────┐     ┌──────────────────────────┐  │
│  │ Mobile App    │     │ External APIs              │  │
│  │ (React Native │     │                           │  │
│  │  / Flutter)   │     │ OpenAI API                │  │
│  │               │     │   (or Azure OpenAI for    │  │
│  │ Operator-     │     │    EU data residency)     │  │
│  │ facing        │     │                           │  │
│  └──────────────┘     │ Later: self-hosted model  │  │
│                        └──────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### Key deployment decisions

| Decision | Chosen | Alternative | Rationale |
|---|---|---|---|
| **Cloud provider** | Provider-agnostic (Cloud Run / ECS / Azure Container Apps) | Single-provider lock-in | The application has no provider-specific dependencies. FastAPI + PostgreSQL + object storage run identically on any cloud. Choose based on the customer's existing infrastructure. |
| **Container vs serverless** | Container (Cloud Run / ECS Fargate) | AWS Lambda, Azure Functions | LLM API calls take 3-5 seconds. Serverless functions have cold start penalties and are designed for sub-second workloads. A container stays warm and handles concurrent requests naturally. |
| **Database** | PostgreSQL + pgvector | Separate vector DB (Pinecone, Weaviate) | PostgreSQL with pgvector extension handles both relational data and vector similarity search in one system. Eliminates operational complexity of running two databases. pgvector is production-proven and sufficient for the expected scale (~100K vectors). A managed vector DB becomes relevant only at millions of vectors with high-throughput requirements. |
| **Image storage** | Object storage (S3 / GCS) | Database BLOBs, local filesystem | Images are write-once, read-rarely. Object storage is cheap, durable, and scales without limit. Database BLOBs bloat backups. Local filesystem doesn't survive container restarts. |
| **LLM provider** | OpenAI API (default), Azure OpenAI (EU data residency) | Self-hosted open model | API-based for Phase 1 — immediate value, no GPU infrastructure. Azure OpenAI provides EU data residency for GDPR compliance. Self-hosted becomes viable when the confirmed database is large enough to fine-tune (~1000+ labeled images). The `VISION_MODEL_NAME` / `DECISION_MODEL_NAME` config makes switching a one-line change. |
| **Offline/upload support** | Async upload queue | Synchronous only | Operators photograph devices in cellars with no connectivity. The mobile app stores photos locally and uploads when connectivity returns. The API processes asynchronously — the operator doesn't wait for the LLM response in real-time. Results are available when they next open the app. |

### Production (Phase 2 — tenant self-service)

Phase 2 adds a tenant-facing interface but the backend architecture stays the same:

- **New frontend**: Tenant mobile app or PWA (simplified UX, no technical language)
- **Authentication**: Tenant identity + building/unit mapping
- **Same API**: The `/api/session/start` → `/api/session/{id}/confirm` flow is identical
- **Same knowledge base**: Tenants benefit from everything learned during operator usage
- **New metric**: Operator visit rate (every avoided visit = cost saving)

No backend changes are required. The knowledge base, validation tiers, and learning loop transfer directly.

### Transition from prototype to production

The architecture is designed so the transition is a series of swaps, not a rewrite:

| Component | Prototype | Production | Change required |
|---|---|---|---|
| Database | SQLite | PostgreSQL + pgvector | Swap connection string, same SQL schema |
| Vector store | ChromaDB (embedded) | pgvector (in PostgreSQL) | Swap `KnowledgeBase` client, same interface |
| Image storage | Local filesystem | S3 / GCS | Swap `_save_upload()` in `routes.py` |
| Frontend | Streamlit | Mobile app | New client, same API contract |
| LLM provider | OpenAI API | Azure OpenAI or self-hosted | Change `VISION_MODEL_NAME` env var |
| Deployment | `uvicorn` locally | Container on Cloud Run / ECS | Add Dockerfile, CI/CD pipeline |

---

## Decision Summary

| Decision | Choice | Key reason | Upgrade path |
|---|---|---|---|
| Agent architecture | 2-agent pipeline | Clear separation, Decision Agent is cheap | Add agents if scope grows |
| Vision model | GPT-4.1 | Best quality, zero training data | Fine-tune when KB is large |
| Extraction | Open-ended | No assumptions about device types | Implicit structure from KB |
| Learning | In-context (few-shot) | Instant, transparent | Fine-tune as data grows |
| Similarity search | ChromaDB + cosine + distance threshold | Semantic matching, no irrelevant results | Managed vector DB at scale |
| KB retrieval timing | Enrich on retry only | Simplest, no extra models | CLIP embeddings for turn-1 |
| Relational DB | SQLite | Zero setup | PostgreSQL + pgvector |
| Vector store | ChromaDB | Zero setup | pgvector (same DB) |
| API | FastAPI | Async, auto-docs | Stays |
| Dashboard | Streamlit | Fast to build | Mobile app for production |
| Agent framework | None (custom) | Full prompt control, transparency | Evaluate if >3 agents |
| MCP / tool-use | None | Deterministic learning loop | Model portability, external integrations |
| Test images | AI-generated (Replicate) | No real photos needed for demo | Real operator photos in production |
| Validation | 3 tiers | Honest learning, breaks circular feedback | Stays |
| CI | GitHub Actions | Already in GitHub, no extra infra | Stays |
| CD | Manual (prototype) | No deployment target yet | Rolling update with approval gate |
| Container runtime | None (prototype) | Runs locally | Cloud Run / ECS Fargate |
| Deployment | Single machine | Prototype simplicity | Containers + managed services |
