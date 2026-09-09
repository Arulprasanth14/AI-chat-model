# picasso-rag-chat

Model-driven, retrieval-augmented conversational AI service for creative brief capture.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  React/Vite UI  (ui/)                                                    │
│  Chat window + State panel (captured / missing fields) + chip buttons   │
│  Vertical/template selection screen → sends vertical + template_key     │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │ POST /conversation/message  (SSE stream)
                             │ POST /session/{id}/direct_field_write
                             │ POST /session/{id}/logo
                             │ GET  /session/{id}/next_field_spec
┌────────────────────────────▼─────────────────────────────────────────────┐
│  FastAPI  (app/main.py)                                                  │
│    ConversationOrchestrator                                              │
│      ├── Phase A  — Silent LLM extraction (tool calls → state writes)   │
│      ├── Phase B  — Streamed response generation (SSE chunks)           │
│      ├── SessionRepository  →  Neon Postgres (sessions table, JSONB)    │
│      ├── RAGRetriever       →  pgvector cosine similarity search        │
│      ├── LLMProvider        →  OpenAI (streaming + tool calling)        │
│      └── PromptBuilder      →  persona + RAG chunks + missing fields    │
└──────────────────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────────────┐
│  Neon Postgres                                                           │
│    conversation_sessions  (JSONB state — full ConversationState)        │
│    knowledge_chunks       (pgvector embeddings for RAG retrieval)       │
└──────────────────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────────────┐
│  Cloudinary                                                              │
│    File/image uploads — logo, food photos, brand assets                 │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Core Design Principles

1. **Zero domain strings in orchestrator code.** The orchestrator loads state, queries vectors, calls the LLM, and saves results. No field names, question scripts, or branching logic. All domain knowledge lives in `profile.yaml` and `knowledge_docs/`.

2. **All domain knowledge in profile YAML + knowledge_docs.** A new project integration = a new profile folder. Zero Python code changes needed.

3. **Two-phase LLM architecture.** Every conversation turn makes two LLM calls: Phase A (silent extraction via named tool calls) and Phase B (streamed conversational response). Phase A writes are committed before Phase B generates its response, so the AI can only confirm what was actually saved.

4. **Deterministic completion gate.** Only `state.py`'s `compute_missing_fields()` determines whether a brief is complete — never the LLM. The LLM has an advisory `mark_session_complete` tool, but the system validates independently.

5. **Direct-write path for UI interactions.** Chip button clicks, file uploads, and structured UI selections bypass LLM extraction and write directly to the state ledger at `confidence=1.0`. This prevents ambiguity and is faster than re-routing through the LLM.

---

## Full Conversation Turn Flow

Each user message triggers the following sequence inside `orchestrator.process_turn()`:

```
User message received
        │
        ▼
① Load or create session state (Postgres)
        │
        ▼
② Detect special trigger tags (see "Special Trigger Tags" section below)
   - __start__        → skip history add, inject opening directive
   - __submit__       → send final confirmation, return snapshot, exit
   - __field_saved__: → skip Phase A, let Phase B acknowledge chip save
        │
        ▼
③ Obtain active profile (profile_provider resolves field-set if vertical/template set)
        │
        ▼
④ Compute missing fields from state ledger (deterministic)
        │
        ▼
⑤ Build brief summary if brief is now complete (render_brief → markdown)
        │
        ▼
⑥ Build Phase A prompt (PromptBuilder.build)
   System message: extraction persona + brief status + tool instructions + examples
   Conversation history (last N turns)
   Optional CURRENT CONTEXT hint (which field is actively being answered)
        │
        ├── Concurrent ──────────────────────────────────────────────────────┐
        │                                                                    │
        ▼                                                                    ▼
⑦ Phase A: LLM tool calling                         RAG Retrieval (parallel)
   - Calls named save tools (save_text_field,       - Vector search against
     save_enum_field, save_quantitative_field,        knowledge_chunks table
     save_custom_field, save_color_suggestion,      - Structural resolver
     mark_session_complete, set_next_topic)           guidance injected
   - Each tool call dispatched to state handler     - Returns list of chunks
   - Write results collected (saved / rejected)
        │
        └────────────────────────────────────────────────────────────────────┘
        │ (both complete)
        ▼
⑧ Recompute missing fields post-Phase-A
   Recompute completion status
   Re-render brief summary if brief just became complete
        │
        ▼
⑨ Build Phase B prompt (PromptBuilder.build_response_phase)
   - Replace system message with Phase B persona + brief status + response rules
   - Append synthetic assistant tool_calls message (Phase A calls)
   - Append one tool-result message per Phase A call (SAVED / NOT SAVED)
   - Append RAG context chunks
   - Append SYSTEM OVERRIDE directive for next field (or brief_confirmation stage)
        │
        ▼
⑩ Phase B: Streamed response generation
   - LLM calls generate_response tool with: intent, message, suggested_next_topic,
     model_believes_complete
   - message tokens streamed to client via SSE as: data: {"chunk": "..."}
        │
        ▼
⑪ Add assistant message to conversation history
   Persist final state (fire-and-forget background task)
        │
        ▼
⑫ Yield final SSE event: data: {"done": true, "snapshot": {...}}
```

---

## Special Trigger Tags

These are hidden text strings the frontend sends as the `user_message` field to trigger special backend behaviors without user-visible text.

| Tag | Sent when | What happens |
|-----|-----------|--------------|
| `__start__` | Frontend opens a new session and wants an opening greeting | User message is NOT added to history. Phase A is skipped. A special `OPENING TURN DIRECTIVE` is appended to the Phase A system prompt, asking the LLM to greet the user and ask for the brand name. |
| `__submit__` | User clicks the "🚀 Submit Brief" button | The orchestrator immediately returns a final confirmation message without calling the LLM at all. State is saved, a final SSE chunk is yielded, and the function returns. |
| `__field_saved__:<field_code>:<value>` | User selects an option via a chip button in the UI (after `direct_field_write` completes) | Phase A is skipped (field was already written at confidence=1.0 by `direct_field_write`). A `field_saved_note` string is injected into the Phase B prompt so the LLM acknowledges the chip selection naturally and asks the next question. |

**Example chip trigger:**
```
user_message = "__field_saved__:asset_availability:use_stock_images_where_suitable"
```
The backend extracts `field_code=asset_availability`, `value=use_stock_images_where_suitable` and tells Phase B to acknowledge it without re-saving.

---

## Two-Phase LLM Prompt Architecture

### Phase A — Extraction (silent, not streamed)

**System message structure** (`PromptBuilder.build` → `_build_system_message`):
1. Minimal extraction persona ("You are a backend data extraction process...")
2. `## Brief Status` block — lists remaining missing fields (or STATUS: BRIEF COMPLETE)
3. `## STEP 1 — EXTRACT AND SAVE FIELDS` — tool calling instructions
4. `## How to Call the Save Tools` — rules and confidence calibration guide
5. Few-shot extraction examples for the current field (primary) + 1-2 secondary fields
6. `## EXTRACTION GUARDRAILS` — 9 numbered hard rules (no guessing, enum validation, date validation, answer override, etc.)

**Current context hint** is appended as a final `system` role message identifying exactly which field is being asked about (prevents guessing for short/Tanglish answers).

### Phase B — Response (streamed to client)

**System message structure** (`PromptBuilder.build_response_phase` → `_build_phase_b_system_message`):
1. Full conversational persona (from `profile.yaml`'s `persona_prompt`)
2. `## Brief Status` block — post-Phase-A state (never re-asks saved fields)
3. `## Response Instructions` + `## RESPONSE STYLE RULES` + `## QUESTION QUALITY RULES`
4. `HONESTY INVARIANTS` — what the LLM must never claim
5. `## FRUSTRATION / FATIGUE HANDLER` — flow for when user is exhausted
6. `## Retrieved Knowledge` — RAG chunks (only injected in Phase B, not Phase A)

**After the system message**, Phase B receives:
- The Phase A tool calls as a synthetic `assistant` message with `tool_calls`
- One `tool` message per Phase A call (outcome: `✅ SAVED` or `❌ NOT SAVED`)
- A `WRITE OUTCOMES` summary note
- A `SYSTEM OVERRIDE — MANDATORY DIRECTIVE` message forcing the LLM to ask about the next specific missing field

**Special override directives** (appended last so they override everything):

| Situation | Directive injected |
|-----------|-------------------|
| Next field is `file_upload` input type | `[SYSTEM OVERRIDE — FILE UPLOAD REQUIRED]` — tells LLM to invite the user to upload, not ask a text question |
| Next field is `brief_confirmation` | `[SYSTEM OVERRIDE — REVIEW STAGE]` — tells LLM to present the full brief summary and ask for confirmation |
| Brief just became complete (`is_complete=True`) | Status block says `MANDATORY FIRST-COMPLETION RESPONSE RULES` — LLM must present full captured brief then ask user to click Submit |
| No missing fields, normal field | `[SYSTEM OVERRIDE — MANDATORY DIRECTIVE]` with field code, description, and enum options if applicable |

---

## Phase A Tool Schemas

Defined in `app/domain/llm/tool_schema.py`. Built dynamically per profile by `get_phase_a_tools(profile)`.

| Tool | When included | Args | What it does |
|------|---------------|------|--------------|
| `save_text_field` | Always | `field_code`, `value`, `confidence` | Save a free-text field to the ledger |
| `save_custom_field` | Always | `field_name`, `value`, `confidence` | Save extra info that has no predefined field; auto-prefixed with `custom_` |
| `save_enum_field` | When profile has enum fields | `field_code`, `value`, `confidence` | Save a field whose value must match the allowed enum list |
| `save_quantitative_field` | When profile has quantitative fields | `field_code`, `value`, `confidence` | Save a field that must contain a numeric/KPI signal |
| `mark_session_complete` | Always | _(none)_ | Advisory: LLM signals it believes all fields are captured |
| `set_next_topic` | Always | `topic` | Advisory: LLM declares which topic it plans to ask about next |
| `save_color_suggestion` | When profile has color-related fields | `name`, `hex`, `rationale` | Save a structured color recommendation (forces valid 6-digit HEX) |

**Confidence calibration rules** (applied to all save tools):
- `0.95–1.0`: User stated value verbatim or selected a chip directly
- `0.80–0.94`: Clear but required minor interpretation
- `0.60–0.79`: Paraphrased or partial answer
- `0.40–0.59`: Vague or hedged
- `0.10–0.39`: Inferred from indirect context — system will ask for clarification

**Field write bypass (confidence=1.0):** When the user explicitly confirms a previously rejected answer (e.g., for a date conflict), the LLM sends `confidence=1.0` which bypasses backend validation and forces the save.

---

## Phase B Tool Schema

`get_response_tool_schema()` returns a single `generate_response` tool.

**Returns:**
```json
{
  "intent": "PROVIDE_INFO | CORRECT_PREVIOUS | ASK_QUESTION | REQUEST_CLARIFICATION | CONFIRM | REJECT | OFF_TOPIC",
  "message": "The conversational reply to stream to the user",
  "suggested_next_topic": "advisory string",
  "model_believes_complete": false
}
```

The `message` field is streamed token-by-token to the client as SSE chunks.

---

## State Ledger (`state.py`)

`ConversationState` is the core domain object persisted as JSONB in Postgres.

**Key fields:**
| Field | Type | Purpose |
|-------|------|---------|
| `session_id` | `str` | UUID |
| `profile_id` | `str` | Which profile YAML to use |
| `status` | `str` | `"active"` or `"complete"` |
| `conversation_history` | `list[dict]` | Full turn history `{role, content, timestamp}` |
| `captured` | `dict[str, CapturedField]` | Completion ledger: field_code → best value |
| `resolved_vertical` | `str \| None` | Industry vertical selected in UI |
| `resolved_template_key` | `str \| None` | Template key selected in UI |
| `unmapped_signals` | `list[str]` | Values LLM tried to save to unknown fields |

**Write handlers** (each returns a `FieldWriteResult` with a `status`):

| Handler | Status codes possible |
|---------|----------------------|
| `handle_save_text_field` | `saved`, `rejected_low_confidence`, `rejected_unknown_field`, `rejected_lower_confidence`, `rejected_date_conflict` |
| `handle_save_enum_field` | `saved`, `rejected_low_confidence`, `rejected_unknown_field`, `rejected_enum` |
| `handle_save_quantitative_field` | `saved`, `rejected_low_confidence`, `rejected_unknown_field`, `rejected_qualitative` |
| `handle_save_custom_field` | `saved`, `rejected_low_confidence` |

**Date validation (bidirectional):**  
`handle_save_text_field` checks if both `project_deadline` and `launch_date_time` are captured. If `deadline >= launch`, the save is rejected with `WRITE_STATUS_REJECTED_DATE_CONFLICT` and a descriptive reason. Bypass: `confidence=1.0` skips this check (used when user explicitly confirms the dates).

**List field merging:**  
Fields with `input_type: list` (e.g. `deliverables`, `distribution_channels`) accumulate values across turns via deduplication merge. `confidence=1.0` triggers a full replace (explicit retraction/override).

---

## API Endpoints

### `POST /conversation/message`

Main SSE endpoint for conversation turns.

**Request:**
```json
{
  "session_id": "optional-uuid",
  "user_message": "message text (or __start__ / __submit__ / __field_saved__:...)",
  "vertical": "restaurant",
  "template_key": "restaurant_cafe_static_post"
}
```

**SSE stream:**
```
data: {"chunk": "To make the design stand out — what's the item you're launching?"}

data: {"done": true, "snapshot": {
  "session_id": "...",
  "profile_id": "picasso_fusion",
  "status": "active",
  "extracted_answers": {"brand_name": {"value": "Bhaii Kadai", "confidence": 0.95, "turn_index": 2}},
  "missing_fields": [{"field_code": "project_type", "description": "..."}],
  "is_complete": false,
  "turn_count": 4,
  "unmapped_signals": [],
  "has_unmapped_signals": false
}}
```

---

### `GET /conversation/session/{session_id}`

Returns current session snapshot (used for page refresh).

---

### `GET /conversation/session/{session_id}/brief`

Returns the deterministic brief summary as a markdown string.

**Response:**
```json
{
  "session_id": "...",
  "is_complete": true,
  "brief": "## 📋 Restaurant · Static Post — Captured Brief\n..."
}
```

---

### `POST /conversation/session/{session_id}/logo`

Upload one or more image files (logo, food photos, brand assets) to Cloudinary.

- Dynamically resolves which `file_upload` field to fill (priority: currently missing file_upload field → show_if-gated fields → fallback to `existing_assets`)
- Writes directly at `confidence=1.0` — bypasses LLM
- Returns `{status, field_code, filename, message, snapshot}`

---

### `POST /conversation/session/{session_id}/document`

Upload a text/PDF document to pre-fill brief fields.

- Text files are decoded and concatenated
- A prompt is constructed forcing the LLM to extract fields from the document text
- Runs through `orchestrator.process_turn()` — same path as a chat message
- Returns `{message, snapshot}`

---

### `GET /conversation/session/{session_id}/next_field_spec`

Returns the `FieldSpec` for the next missing required field. The UI uses this to render the correct input control (chip buttons, file upload button, date picker, free text).

**Response:**
```json
{"next_field": {"field_code": "asset_availability", "input_type": "enum", "enum_values": [...], "enum_options": [...]}}
```

---

### `POST /conversation/session/{session_id}/direct_field_write`

Write a field value directly from a UI selection (chip click), bypassing LLM extraction.

**Request body:**
```json
{"field_code": "asset_availability", "value": "use_stock_images_where_suitable"}
```

- Writes at `confidence=1.0` (never overwritten by LLM inference)
- Returns `{status, field_code, value, reason, snapshot}`
- After this returns, the UI sends `__field_saved__:<field_code>:<value>` as the next chat message so the LLM can acknowledge the selection naturally

---

### `GET /health`

```json
{"status": "ok", "db": "connected", "service": "picasso-rag-chat", "version": "0.1.0"}
```

---

## Profile YAML Structure

Every profile lives in `app/project_profiles/<profile_id>/profile.yaml`. The system is 100% profile-agnostic — the orchestrator, state ledger, and retriever operate on `BaseProfile` objects only.

```yaml
profile_id: "picasso_fusion"
knowledge_namespace: "picasso_fusion"   # Used for RAG vector search namespace
llm_temperature: 0.3

industries:
  - food_beverage
  - fashion_apparel
  # ...

persona_prompt: |
  You are Picasso, an expert creative strategist...

required_fields:
  - code: brand_name
    description: >
      The brand name or business commissioning the work.
    required: true

  - code: project_type
    description: The type of creative deliverable.
    required: true
    enum_values:
      - "Static Post"
      - "Carousel"
      # ...

  - code: distribution_channels
    description: Where the creative will be published.
    required: true
    input_type: list            # accumulates values across turns (merge not replace)
    enum_values:
      - "Instagram 1:1"
      - "Facebook"
      # ...

  - code: uploaded_files
    description: Food/product photos uploaded by the client.
    required: false
    input_type: file_upload     # triggers Upload button in UI
    show_if:
      field_code: asset_availability
      in_: ["ill_upload_food_photos", "ill_upload_restaurant_ambience_photos"]
```

**Field `input_type` values:**

| Value | UI renders | State behavior |
|-------|------------|---------------|
| `text` (default) | Free-text input | Single value, replaced on update |
| `list` | Chip multi-select or text | Values merged/deduplicated across turns |
| `enum` | Single-select chip buttons | Must match `enum_values`, validated in handler |
| `file_upload` | Upload button | Written directly by `/logo` endpoint |

**`show_if` conditional fields:** A field with `show_if` is only included in `compute_missing_fields()` when the dependency field (`show_if.field_code`) is captured above threshold AND the condition (`in_` or `not_in_`) is met. This drives dynamic field gating (e.g., file upload fields only appear after the user selects "I'll upload photos").

---

## Brief Renderer (`brief_renderer.py`)

`render_brief(captured, field_set_yaml_path, profile)` produces a deterministic markdown summary.

- If a field-set YAML exists (resolved from `vertical/template_key.yaml`), renders sections in V1 section order with emoji icons
- Falls back to flat base-profile list if no YAML
- Missing required fields shown as `*(not yet provided)*`
- Missing optional fields silently omitted
- Extra captured fields (not in YAML) shown in an `📌 Additional Captured Information` appendix
- Used in: orchestrator (when brief becomes complete), `/session/{id}/brief` GET endpoint, Phase B `brief_confirmation` override directive

---

## Folder Structure

```
pythonProject/
├── app/
│   ├── main.py                          # FastAPI factory + lifespan (keepalive task)
│   ├── core/
│   │   ├── config.py                    # pydantic-settings (all env vars)
│   │   └── logging.py                   # structured JSON logging
│   ├── api/
│   │   ├── deps.py                      # dependency injection wiring
│   │   └── routes/
│   │       ├── conversation.py          # All /conversation/* endpoints
│   │       └── health.py                # GET /health
│   ├── domain/
│   │   ├── conversation/
│   │   │   ├── orchestrator.py          # Central turn processor (zero domain strings)
│   │   │   ├── state.py                 # ConversationState + field write handlers
│   │   │   ├── brief_renderer.py        # Deterministic markdown brief summary
│   │   │   ├── models.py                # SQLAlchemy ORM model (sessions table)
│   │   │   ├── field_spec.py            # FieldSpec model for UI rendering hints
│   │   │   ├── field_spec_registry.py   # Maps profile field defs → FieldSpec
│   │   │   ├── field_set_loader.py      # Loads vertical/template YAML field-sets
│   │   │   ├── structural_resolver.py   # Injects structural section guidance into RAG
│   │   │   ├── suggestion_gate.py       # Gate: when can LLM make creative suggestions
│   │   │   ├── template_resolver.py     # Resolves vertical + template_key from state
│   │   │   └── color_suggestion.py      # Parses & validates save_color_suggestion calls
│   │   ├── llm/
│   │   │   ├── provider.py              # LLMProvider protocol (interface)
│   │   │   ├── openai_provider.py       # OpenAI implementation (streaming + tools)
│   │   │   ├── prompt_builder.py        # Phase A + Phase B prompt assembly
│   │   │   └── tool_schema.py           # All Phase A & Phase B tool JSON schemas
│   │   └── rag/
│   │       ├── embedder.py              # Embedder protocol + OpenAI implementation
│   │       ├── retriever.py             # RAGRetriever (query → top-k chunks)
│   │       ├── vector_store.py          # VectorStore protocol
│   │       └── ingestion.py             # KnowledgeIngester (CLI usage)
│   ├── infrastructure/
│   │   ├── persistence/
│   │   │   ├── session_repository.py    # Interface + InMemory implementation
│   │   │   └── postgres_session_repo.py # Postgres JSONB implementation
│   │   └── vector_db/
│   │       └── pgvector_client.py       # pgvector async client
│   └── project_profiles/
│       ├── base_profile.py              # BaseProfile dataclass (reusability contract)
│       └── picasso_fusion/
│           ├── profile.yaml             # All domain config — no Python needed
│           ├── field_sets/              # Per-vertical, per-template field-set YAMLs
│           │   └── restaurant/
│           │       └── restaurant_cafe_static_post.yaml
│           └── knowledge_docs/          # RAG source documents
│               └── restaurant/
│                   ├── restaurant_cafe_carousel__additional_notes.md
│                   └── ...
├── scripts/
│   ├── ingest_knowledge.py              # CLI ingestion runner
│   └── eval_retrieval.py               # RAG quality evaluation
├── tests/
│   ├── unit/
│   │   ├── test_state.py
│   │   ├── test_prompt_builder.py
│   │   └── test_retriever.py
│   └── integration/
│       └── test_conversation_turn.py
├── ui/                                  # React + Vite frontend
├── alembic/                             # DB migrations
├── .env
├── pyproject.toml
└── README.md
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Neon Postgres account (free tier works) with pgvector enabled
- OpenAI API key
- Cloudinary account (for file uploads)

### 1. Install Python dependencies

```bash
cd pythonProject
pip install -e ".[dev]"
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY, DATABASE_URL, CLOUDINARY_URL
```

### 3. Ingest knowledge documents

```bash
python scripts/ingest_knowledge.py --profile picasso_fusion
```

### 4. Start the API server

```bash
uvicorn app.main:app --reload --port 8000
```

### 5. Start the UI

```bash
cd ui
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) — the Vite dev server proxies API calls to port 8000.

### 6. Run tests

```bash
pytest tests/ -v
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | required | OpenAI API key |
| `DATABASE_URL` | required | asyncpg Postgres URL (Neon or local) |
| `CLOUDINARY_URL` | required | Full Cloudinary URL for asset uploads |
| `CHAT_MODEL` | `gpt-4.1` | OpenAI chat model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for RAG |
| `ACTIVE_PROFILE` | `picasso_fusion` | Profile folder name under `project_profiles/` |
| `RETRIEVAL_TOP_K` | `8` | Chunks returned per RAG retrieval |
| `HISTORY_WINDOW` | `20` | Max history messages included in prompt |
| `EXTRACTION_CONFIDENCE_THRESHOLD` | `0.7` | Min confidence to count a field as captured |
| `LOG_LEVEL` | `INFO` | Log verbosity |
| `CORS_ORIGINS` | `http://localhost:5173,...` | Allowed CORS origins |

---

## Adding a New Project Profile

To add a new integration (e.g., "Orbit Agency"):

### Step 1: Create the profile folder

```
app/project_profiles/orbit_agency/
├── profile.yaml
├── field_sets/
│   └── saas/
│       └── launch_campaign.yaml
└── knowledge_docs/
    └── saas/
        ├── question_guidance.md
        └── domain_facts.md
```

### Step 2: Write `profile.yaml`

```yaml
profile_id: "orbit_agency"
knowledge_namespace: "orbit_agency"
llm_temperature: 0.5

persona_prompt: |
  You are Orbit, a strategic B2B consultant at Orbit Agency...

required_fields:
  - code: company_name
    description: The company commissioning the campaign.
    required: true

  - code: ideal_customer_profile
    description: Target company size, industry, decision-maker titles.
    required: true
```

### Step 3: Ingest and activate

```bash
python scripts/ingest_knowledge.py --profile orbit_agency
# In .env:
ACTIVE_PROFILE=orbit_agency
uvicorn app.main:app --reload
```

**Zero Python changes required.** The orchestrator, state ledger, retriever, and UI all work unchanged.

---

## RAG Evaluation

```bash
python scripts/eval_retrieval.py --profile picasso_fusion --query "how to ask about budget"
python scripts/eval_retrieval.py --profile picasso_fusion --query "target audience examples" --top-k 10
```

Shows top-k chunks with cosine similarity scores without LLM calls.
