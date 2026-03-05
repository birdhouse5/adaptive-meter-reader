# Adaptive Meter Reader

A self-improving AI system that assists operators in capturing correct meter readings during on-site visits. The system extracts meter data from photos using open-ended vision analysis, provides specific guidance when images are insufficient, validates readings through operator confirmation, and learns from every interaction to optimize its guidance over time.

The knowledge base built from operator usage becomes the foundation for a future tenant self-service extension.

## Architecture

**2-Agent Pipeline** — each turn makes exactly 2 LLM calls:

```
Operator Upload
      |
  Orchestrator
      |
  +---+---+
  |       |
Vision  Decision
Agent   Agent
GPT-4.1 GPT-4.1-mini
(image) (text-only)
  |       |
  +---+---+
      |
  +---+---+---+
  |   |       |
 OK  Retry  Escalate
  |   |       |
Confirm Guidance Supervisor
  |
Knowledge Base
```

1. **Vision Agent** (GPT-4.1) — analyzes the meter image, extracts all identifiable data with per-field confidence scores. Open-ended extraction — no prescribed fields.
2. **Decision Agent** (GPT-4.1-mini, text-only) — evaluates the extraction, decides routing (sufficient/retry/escalate), generates operator-facing messages.

See `docs/architecture.md` for full Mermaid diagrams.

**The optimization problem**: minimize expected turns-to-success while keeping reading error rate below a threshold — calibrated against real ground truth, not self-reported model confidence.

**What adapts over time** (from confirmed readings):
- Vision prompts (few-shot examples from similar confirmed images + correction warnings)
- Decision context (device expectations, proven guidance, consistency signals, calibration)
- Knowledge base growth across 3 ChromaDB collections

**Three validation tiers**:
1. **User confirmation** — operator confirms or corrects the reading inline
2. **Cross-session consistency** — serial stability, reading monotonicity, rate bounds
3. **Supervisor verification** — periodic spot-checks on a sample of readings

## Setup

```bash
git clone <repo-url>
cd adaptive-meter-reader

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

> **Git safety**: `.env` is excluded by `.gitignore`. When initializing a new repo, verify `git status` does not list `.env` before your first commit.

## Running

### 1. Start the API server

```bash
python -m src.main
# or: uvicorn src.main:app --reload --port 8000
```

### 2. Start the dashboard

```bash
streamlit run dashboard/app.py
```

## Demo Flow (3-5 minutes)

1. **Start the API server** — `python -m src.main`
2. **Seed demo data** (recommended) — `python scripts/seed_demo.py` uploads sample meter images and confirms the readings, so the dashboard starts with data instead of empty
3. **Start the dashboard** — `streamlit run dashboard/app.py`
4. **Show architecture** — sidebar diagram explains the 2-agent pipeline + learning loop
5. **Upload a meter photo** — see open-ended extraction with per-field confidence, routing decision
6. **Confirm the reading** — operator confirms or corrects, knowledge base grows
7. **Show a retry flow** — upload a poor quality image, see the Decision Agent's guidance, upload a better one
8. **Show metrics** — knowledge base growing, guidance effectiveness, per-device-type breakdown
9. **Explain the vision** — Phase 1 builds the knowledge base with operators, Phase 2 extends to tenant self-service

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/session/start` | Start a new reading session (first photo) |
| POST | `/api/session/{id}/upload` | Upload a retry photo |
| POST | `/api/session/{id}/confirm` | Confirm or correct the extracted reading |
| GET | `/api/session/{id}` | Get session details with all turns |
| GET | `/api/sessions` | List recent sessions |
| GET | `/api/stats` | Optimization metrics + knowledge base stats |
| GET | `/api/activity` | Agent activity log |
| GET | `/api/health` | Health check |

## Development

```bash
pip install -r requirements-dev.txt

ruff check src/ tests/             # lint
ruff format src/ tests/ dashboard/  # format
pytest tests/ -v -m "not live"      # test (CI-safe)
pytest tests/test_live.py -m live   # live LLM tests (requires API key)
```
