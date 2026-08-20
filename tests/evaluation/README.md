# Picasso AI Evaluation Framework

A dedicated evaluation suite that validates the RAG retriever, LLM orchestrator, and end-to-end system in the Picasso AI Model project.

## Location

```
tests/evaluation/
├── datasets/
│   └── evaluation_cases.json     ← 10 evaluation scenarios
├── metrics/
│   ├── test_cosine_similarity.py  ← Metric 1: Embedding quality
│   ├── test_context_precision.py  ← Metric 2: Retrieval precision
│   ├── test_context_recall.py     ← Metric 3: Retrieval recall
│   ├── test_extraction_accuracy.py← Metric 4: LLM field extraction
│   ├── test_faithfulness.py       ← Metric 5: Hallucination detection
│   ├── test_answer_relevance.py   ← Metric 6: Response relevance
│   ├── test_state_transition.py   ← Metric 7: State ledger correctness
│   └── test_turn_efficiency.py    ← Metric 8: Turn count / completion
├── conftest.py                    ← Shared fixtures + evaluator helpers
├── run_evaluation.py              ← Runner + formatted reporter
└── README.md                      ← This file
```

## Running the Evaluation Suite

### Full evaluation (includes live OpenAI API calls):
```bash
cd pythonProject
python tests/evaluation/run_evaluation.py
```

### Deterministic only (no API calls, no cost):
```bash
python tests/evaluation/run_evaluation.py --no-live
```

### Run directly with pytest:
```bash
pytest tests/evaluation/metrics/ -v -s
```

### Run only deterministic tests:
```bash
pytest tests/evaluation/metrics/ -v -s -m "not live"
```

### Save results to JSON:
```bash
python tests/evaluation/run_evaluation.py --output-json results.json
```

## Test Types

### Deterministic Tests
These tests do **not** call external APIs. They use:
- Stubbed LLM providers returning configured JSON
- Mock vector stores returning configured results
- In-memory session repositories

Deterministic tests validate the **application logic** around the AI system.

### Live Tests (`@pytest.mark.live`)
These tests call the **real OpenAI API** using your `.env` configuration. They validate:
- Real embedding quality (cosine similarity)
- Real LLM faithfulness and answer relevance (using LLM-as-evaluator)

## Thresholds

All thresholds are defined in a single place: `conftest.py` → `THRESHOLDS` dict.

| Threshold | Value | Description |
|-----------|-------|-------------|
| `min_relevant_similarity` | 0.30 | Min cosine score for a domain-relevant query |
| `max_irrelevant_similarity` | 0.65 | Max cosine score for an off-topic query |
| `extraction_confidence` | 0.70 | Min confidence to count a field as captured |
| `faithfulness_pass` | 0.60 | Min faithfulness score to pass |
| `answer_relevance_pass` | 0.60 | Min relevance score to pass |
| `max_efficient_turns` | 8 | Max turns before conversation is flagged as inefficient |

## Metrics Summary

| Metric | How Calculated | Type |
|--------|---------------|------|
| Cosine Similarity | `dot(A,B) / (\|A\| * \|B\|)` | Live (real embeddings) |
| Context Precision | `relevant_retrieved / total_retrieved` | Deterministic |
| Context Recall | `found_types / required_types` | Deterministic |
| Extraction Precision | `correct_extractions / total_extracted` | Deterministic |
| Extraction Recall | `correct_extractions / total_expected` | Deterministic |
| Faithfulness | Keyword scan + LLM evaluator score | Hybrid |
| Answer Relevance | Keyword acknowledgement + LLM evaluator score | Hybrid |
| State Transition | Binary per-check (assertion pass/fail) | Deterministic |
| Turn Efficiency | `turns <= max_efficient_turns` within `acceptable_turn_range` | Deterministic |

## Dataset

`evaluation_cases.json` contains 10 evaluation scenarios:

| ID | Description |
|----|-------------|
| `case_01_single_field` | User provides one field clearly |
| `case_02_multi_field` | User provides two fields in one message |
| `case_03_multi_turn` | Information spread across 4 turns |
| `case_04_missing_info` | User gives vague/incomplete answer |
| `case_05_irrelevant_info` | User talks off-topic |
| `case_06_ambiguous` | User gives partial/ambiguous answer |
| `case_07_rag_relevant` | Query expected to retrieve domain chunks |
| `case_08_rag_irrelevant` | Off-topic query, low similarity expected |
| `case_09_faithfulness` | Tests groundedness of responses |
| `case_10_complete_conversation` | Full happy-path: all 3 fields captured |

## Architecture Decisions

### Why deterministic over semantic where possible?
Picasso's design uses a **forced tool-call schema** and **deterministic Python ledger** (`state.py`). This makes state transitions, extraction, and completion purely testable without LLM evaluation. We only use LLM-as-evaluator for `faithfulness` and `answer_relevance`, which are inherently semantic.

### Why not a pre-built framework like Ragas?
- Ragas requires specific LangChain/Haystack integration patterns not used in Picasso.
- The existing test infrastructure (`StubLLMProvider`, `InMemorySessionRepository`, `MockVectorStore`) is idiomatic and complete.
- Implementing metrics from scratch makes every calculation explicit and auditable.
