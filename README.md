# picasso-rag-chat

Model-driven, retrieval-augmented conversational AI service for creative brief capture.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  React/Vite Debug UI  (ui/)                                     │
│  Chat window + State panel (captured / missing fields)          │
└────────────────────────┬────────────────────────────────────────┘
                         │ POST /conversation/message  (SSE)
┌────────────────────────▼────────────────────────────────────────┐
│  FastAPI  (app/main.py)                                         │
│    ConversationOrchestrator                                     │
│      ├── SessionRepository  →  Neon Postgres (sessions table)  │
│      ├── RAGRetriever       →  pgvector similarity search       │
│      ├── LLMProvider        →  OpenAI gpt-4.1 (streaming)      │
│      └── PromptBuilder      →  persona + chunks + missing fields │
└─────────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│  Neon Postgres                                                  │
│    conversation_sessions  (JSONB state)                         │
│    knowledge_chunks       (pgvector embeddings)                 │
└─────────────────────────────────────────────────────────────────┘
```

## Core Design Principles

1. **Zero domain strings in orchestrator code.** The orchestrator loads state, queries vectors, calls the LLM, and saves results. No field names, question scripts, or branching logic.

2. **All domain knowledge in profile YAML + knowledge_docs.** A new project = a new profile folder. Zero Python code changes.

3. **Forced tool-call schema.** Every LLM response returns structured JSON: `message`, `extracted_answers`, `suggested_next_topic`, `model_believes_complete`.

4. **Deterministic completion gate.** Only `state.py`'s `compute_missing_fields()` determines whether a brief is complete — not the LLM.

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Neon Postgres account (free tier works) with pgvector enabled
- OpenAI API key

### 1. Install Python dependencies

```bash
cd pythonProject
pip install -e ".[dev]"
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and DATABASE_URL
```

Your `DATABASE_URL` should look like:
```
DATABASE_URL=postgresql+asyncpg://user:password@ep-xxx.neon.tech/picasso_rag?ssl=require
```

### 3. Ingest knowledge documents

```bash
python scripts/ingest_knowledge.py --profile picasso_fusion
```

### 4. Start the API server

```bash
uvicorn app.main:app --reload
```

### 5. Start the UI

```bash
cd ui
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) — the debug UI proxies API calls to port 8000.

### 6. Run tests

```bash
pytest tests/ -v
```

---

## Folder Structure

```
pythonProject/
├── app/
│   ├── main.py                          # FastAPI app + lifespan
│   ├── core/
│   │   ├── config.py                    # pydantic-settings
│   │   └── logging.py                   # structured JSON logging
│   ├── api/
│   │   ├── deps.py                      # dependency injection wiring
│   │   └── routes/
│   │       ├── conversation.py          # POST /conversation/message (SSE)
│   │       └── health.py               # GET /health
│   ├── domain/
│   │   ├── conversation/
│   │   │   ├── orchestrator.py          # ← zero domain strings
│   │   │   ├── state.py                 # completion ledger
│   │   │   └── models.py               # SQLAlchemy ORM
│   │   ├── llm/
│   │   │   ├── provider.py             # LLMProvider protocol
│   │   │   ├── openai_provider.py      # OpenAI implementation
│   │   │   ├── prompt_builder.py       # prompt assembly
│   │   │   └── tool_schema.py          # forced tool-call schema
│   │   └── rag/
│   │       ├── embedder.py             # Embedder protocol + OpenAI impl
│   │       ├── retriever.py            # RAGRetriever
│   │       ├── vector_store.py         # VectorStore protocol
│   │       └── ingestion.py            # KnowledgeIngester
│   ├── infrastructure/
│   │   ├── persistence/
│   │   │   ├── session_repository.py   # interface + InMemory impl
│   │   │   └── postgres_session_repo.py
│   │   └── vector_db/
│   │       └── pgvector_client.py
│   └── project_profiles/
│       ├── base_profile.py             # ← reusability contract
│       └── picasso_fusion/
│           ├── profile.yaml            # domain config (no Python needed)
│           └── knowledge_docs/
│               ├── question_guidance.md
│               ├── domain_facts.md
│               └── examples.md
├── scripts/
│   ├── ingest_knowledge.py             # CLI ingestion runner
│   └── eval_retrieval.py              # RAG quality evaluation
├── tests/
│   ├── unit/
│   │   ├── test_state.py
│   │   ├── test_prompt_builder.py
│   │   └── test_retriever.py
│   └── integration/
│       └── test_conversation_turn.py
├── ui/                                 # React + Vite debug UI
├── .env.example
├── pyproject.toml
└── README.md
```

---

## Adding a New Project Profile

This is the core reusability feature. To add a new integration (e.g., "Orbit Agency"):

### Step 1: Create the profile folder

```
app/project_profiles/orbit_agency/
├── profile.yaml
└── knowledge_docs/
    ├── question_guidance.md
    ├── domain_facts.md
    └── examples.md
```

### Step 2: Write profile.yaml

```yaml
profile_id: "orbit_agency"
knowledge_namespace: "orbit_agency"
llm_temperature: 0.6

industries:
  - technology
  - healthcare

persona_prompt: |
  You are Orbit, a strategic consultant at Orbit Agency specialising in
  B2B SaaS go-to-market strategy. Your goal is to understand the client's
  product, market, and growth objectives to design a targeted campaign.

required_fields:
  - code: company_name
    description: >
      The name of the company or startup commissioning the campaign.
    required: true

  - code: product_category
    description: >
      The type of software product: SaaS platform, developer tool,
      data platform, marketplace, etc.
    required: true

  - code: ideal_customer_profile
    description: >
      Detailed ICP: company size, industry, job titles of decision-makers,
      pain points they experience.
    required: true

  - code: growth_stage
    description: >
      Current growth stage: pre-launch, seed, Series A, growth, enterprise.
    required: true

  - code: campaign_budget
    description: >
      Marketing budget for this campaign cycle.
    required: true
```

### Step 3: Write knowledge_docs/

Add markdown files explaining how to gather each field conversationally for this specific domain.

### Step 4: Ingest and activate

```bash
# Ingest the new profile's knowledge
python scripts/ingest_knowledge.py --profile orbit_agency

# Switch to the new profile
# In .env:
ACTIVE_PROFILE=orbit_agency

# Restart the API server
uvicorn app.main:app --reload
```

**That's it.** The orchestrator, retriever, state ledger, and UI require zero changes.

---

## API Reference

### `POST /conversation/message`

Start or continue a conversation.

**Request body:**
```json
{
  "session_id": "optional-existing-session-uuid",
  "user_message": "I need help with a brand campaign for my SaaS startup"
}
```

**SSE response stream:**

Token chunks during streaming:
```
data: {"chunk": "Great to hear about your startup! Tell me..."}
```

Final event with full session state:
```json
data: {
  "done": true,
  "snapshot": {
    "session_id": "...",
    "profile_id": "picasso_fusion",
    "status": "active",
    "extracted_answers": {
      "client_name": {"value": "Acme Corp", "confidence": 0.92, "turn_index": 2}
    },
    "missing_fields": [
      {"field_code": "project_type", "description": "..."},
      ...
    ],
    "model_believes_complete": false,
    "is_complete": false,
    "turn_count": 2
  }
}
```

### `GET /health`

```json
{"status": "ok", "db": "connected", "service": "picasso-rag-chat", "version": "0.1.0"}
```

### `GET /conversation/session/{session_id}`

Returns the current snapshot for an existing session (for UI page refresh).

---

## RAG Evaluation

To check retrieval quality for specific queries:

```bash
python scripts/eval_retrieval.py --profile picasso_fusion --query "how to ask about budget"
python scripts/eval_retrieval.py --profile picasso_fusion --query "target audience examples" --top-k 10
```

This shows top-k chunks with similarity scores without making any LLM calls.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | required | OpenAI API key |
| `DATABASE_URL` | required | asyncpg Postgres URL |
| `CHAT_MODEL` | `gpt-4.1` | OpenAI chat model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `ACTIVE_PROFILE` | `picasso_fusion` | Profile folder name |
| `RETRIEVAL_TOP_K` | `5` | Chunks per retrieval |
| `HISTORY_WINDOW` | `20` | History turns in prompt |
| `EXTRACTION_CONFIDENCE_THRESHOLD` | `0.7` | Min confidence to mark field captured |
| `LOG_LEVEL` | `INFO` | Log verbosity |
| `CORS_ORIGINS` | `http://localhost:5173,...` | Allowed CORS origins |
