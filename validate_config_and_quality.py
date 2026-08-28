"""
validate_config_and_quality.py
══════════════════════════════════════════════════════════════════════
End-to-end config correctness + quality validation script.

Checks:
  1. .env / Settings configuration audit
  2. Embedding dimension consistency (embedder → pgvector)
  3. Cosine similarity quality (≥ 75% required for pass)
  4. Faithfulness / correctness check via Ollama qwen2.5:7b

Run from the project root:
    python validate_config_and_quality.py
"""
from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
import textwrap
import time
from pathlib import Path

# ─── Colour helpers ───────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

PASS = f"{GREEN}✔ PASS{RESET}"
FAIL = f"{RED}✗ FAIL{RESET}"
WARN = f"{YELLOW}⚠ WARN{RESET}"
INFO = f"{CYAN}ℹ INFO{RESET}"

_results: list[tuple[str, bool, str]] = []   # (label, passed, detail)

def _record(label: str, passed: bool, detail: str = "") -> None:
    _results.append((label, passed, detail))
    status = PASS if passed else FAIL
    print(f"  {status}  {label}")
    if detail:
        for line in textwrap.wrap(detail, 90):
            print(f"           {line}")

def section(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'═'*70}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'═'*70}{RESET}")

# ══════════════════════════════════════════════════════════════════════════════
# 1.  .env / Settings configuration audit
# ══════════════════════════════════════════════════════════════════════════════

section("1. Configuration Audit (.env ↔ Settings ↔ Code)")

# Load .env manually so we can inspect raw values without pydantic coercion
env_path = Path(__file__).parent / ".env"
raw_env: dict[str, str] = {}
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        raw_env[k.strip()] = v.strip()
    _record(".env file found", True, str(env_path))
else:
    _record(".env file found", False, f"Not found at {env_path}")

# Load pydantic Settings
try:
    # Add project root to sys.path so imports work
    project_root = str(Path(__file__).parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from app.core.config import Settings
    settings = Settings()
    _record("Settings loads without error", True)
except Exception as e:
    _record("Settings loads without error", False, str(e))
    settings = None  # type: ignore

if settings:
    # ── Ollama ────────────────────────────────────────────────────────────────
    expected_ollama_model = "qwen2.5:7b"
    actual_ollama_model   = settings.ollama_model
    match_model = actual_ollama_model == expected_ollama_model
    _record(
        f"OLLAMA_MODEL = {expected_ollama_model}",
        match_model,
        f"Actual: '{actual_ollama_model}'"
        + ("" if match_model else f"  ← .env still has '{actual_ollama_model}', expected '{expected_ollama_model}'"),
    )

    ollama_url = settings.ollama_base_url
    _record(
        "OLLAMA_BASE_URL set",
        bool(ollama_url),
        f"Value: {ollama_url}",
    )

    # ── Embedding provider ─────────────────────────────────────────────────────
    emb_provider = settings.embedding_provider.lower()
    _record(
        "EMBEDDING_PROVIDER = local",
        emb_provider == "local",
        f"Actual: '{emb_provider}'",
    )

    local_model = settings.local_embedding_model
    expected_emb_model = "google/embeddinggemma-300m"
    match_emb = local_model == expected_emb_model
    _record(
        f"LOCAL_EMBEDDING_MODEL = {expected_emb_model}",
        match_emb,
        f"Actual: '{local_model}'"
        + ("" if match_emb else "  ← update LOCAL_EMBEDDING_MODEL in .env"),
    )

    # ── Similarity threshold ───────────────────────────────────────────────────
    sim_thresh = settings.vector_similarity_threshold
    # Correct range for embeddinggemma-300m is 0.20–0.35
    thresh_ok  = 0.15 <= sim_thresh <= 0.40
    _record(
        f"VECTOR_SIMILARITY_THRESHOLD in valid range [0.15–0.40] for embeddinggemma-300m",
        thresh_ok,
        f"Actual: {sim_thresh}" + ("" if thresh_ok else
            "  ← embeddinggemma-300m produces scores ~0.2–0.65; threshold of 0.7 (OpenAI default) will block ALL results"),
    )

    # ── Retrieval top-k ───────────────────────────────────────────────────────
    top_k = settings.retrieval_top_k
    _record(
        f"RETRIEVAL_TOP_K ≥ 5 (enough candidates for threshold filtering)",
        top_k >= 5,
        f"Actual: {top_k}",
    )

    # ── DB URL ─────────────────────────────────────────────────────────────────
    db_url = settings.database_url
    db_ok  = db_url.startswith("postgresql+asyncpg://")
    _record(
        "DATABASE_URL uses postgresql+asyncpg driver",
        db_ok,
        f"Prefix: {db_url[:40]}...",
    )

# ─── Cross-check code constants ───────────────────────────────────────────────
try:
    from app.infrastructure.vector_db.pgvector_client import EMBEDDING_DIM as PG_DIM
    _record(f"pgvector_client.EMBEDDING_DIM = 768", PG_DIM == 768, f"Actual: {PG_DIM}")
except Exception as e:
    _record("pgvector_client.EMBEDDING_DIM", False, str(e))

try:
    from app.infrastructure.rag.embedding_gemma_embedder import EMBEDDING_DIM as EMB_DIM
    _record(f"embedding_gemma_embedder.EMBEDDING_DIM = 768", EMB_DIM == 768, f"Actual: {EMB_DIM}")
except Exception as e:
    _record("embedding_gemma_embedder.EMBEDDING_DIM", False, str(e))

# ── deps.py wiring check ──────────────────────────────────────────────────────
try:
    deps_src = Path(__file__).parent / "app" / "api" / "deps.py"
    src = deps_src.read_text(encoding="utf-8")
    ollama_active = "return OllamaProvider(" in src and "# return OpenAIProvider" not in src
    gemma_active  = "EmbeddingGemmaEmbedder(" in src
    _record("deps.py: OllamaProvider is ACTIVE (not OpenAIProvider)", ollama_active)
    _record("deps.py: EmbeddingGemmaEmbedder is wired", gemma_active)
except Exception as e:
    _record("deps.py wiring check", False, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Embedding quality — cosine similarity ≥ 75 %
# ══════════════════════════════════════════════════════════════════════════════

section("2. Embedding Quality — Cosine Similarity (threshold ≥ 0.75)")

COSINE_PASS_THRESHOLD = 0.75   # required per user spec

def cosine(a: list[float], b: list[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)

# Probe pairs — semantically very similar (expected high cosine)
PROBE_PAIRS = [
    (
        "What is the budget for this marketing project?",
        "How much money is allocated for the marketing campaign?",
    ),
    (
        "The client wants a logo design for their brand.",
        "We need brand identity work including a new logo.",
    ),
    (
        "Target audience is young professionals aged 25-35.",
        "The demographic focus is millennials and young working adults.",
    ),
    (
        "Project timeline is 6 weeks from kick-off to delivery.",
        "The deadline is 6 weeks after project start.",
    ),
    (
        "Campaign should focus on social media channels like Instagram and LinkedIn.",
        "The marketing strategy targets Instagram and LinkedIn platforms.",
    ),
]

# Negative probe — semantically unrelated (expected low cosine)
NEGATIVE_PAIRS = [
    (
        "What is the budget for this marketing project?",
        "The weather in London is rainy and cold today.",
    ),
]

async def run_embedding_checks() -> dict[str, float]:
    scores: dict[str, float] = {}
    try:
        from sentence_transformers import SentenceTransformer
        model_id = settings.local_embedding_model if settings else "google/embeddinggemma-300m"
        print(f"\n  {INFO}  Loading model: {model_id}")
        t0 = time.time()
        model = SentenceTransformer(model_id)
        print(f"  {INFO}  Model loaded in {time.time()-t0:.1f}s\n")
    except ImportError:
        _record("sentence-transformers importable", False,
                "Run: pip install sentence-transformers")
        return {}
    except Exception as e:
        _record("SentenceTransformer model load", False, str(e))
        return {}

    _record("sentence-transformers importable", True)

    all_pass = True
    for i, (q, doc) in enumerate(PROBE_PAIRS, 1):
        t0 = time.time()
        vecs = model.encode([q, doc], normalize_embeddings=True)
        elapsed = time.time() - t0
        sim = float(vecs[0] @ vecs[1])   # dot product of L2-normalised = cosine
        scores[f"pair_{i}"] = sim
        passed = sim >= COSINE_PASS_THRESHOLD
        if not passed:
            all_pass = False
        label = f"Pair {i} cosine={sim:.4f} (≥{COSINE_PASS_THRESHOLD}) [{elapsed:.2f}s]"
        _record(label, passed,
                f"Q:   {q[:70]}\n"
                f"           DOC: {doc[:70]}")

    # Negative pair sanity check — should be LOW (< 0.70)
    for i, (q, doc) in enumerate(NEGATIVE_PAIRS, 1):
        vecs = model.encode([q, doc], normalize_embeddings=True)
        sim  = float(vecs[0] @ vecs[1])
        scores[f"neg_{i}"] = sim
        sane = sim < 0.70
        _record(
            f"Negative pair {i} cosine={sim:.4f} (expected < 0.70 — sanity check)",
            sane,
            f"Q:   {q[:70]}\n"
            f"           DOC: {doc[:70]}",
        )

    pos_scores = [v for k, v in scores.items() if k.startswith("pair_")]
    avg = sum(pos_scores) / len(pos_scores) if pos_scores else 0.0
    scores["average_positive"] = avg
    overall = avg >= COSINE_PASS_THRESHOLD
    _record(
        f"OVERALL average cosine similarity = {avg:.4f} (≥{COSINE_PASS_THRESHOLD} required)",
        overall,
    )
    return scores


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Faithfulness check via Ollama (qwen2.5:7b)
# ══════════════════════════════════════════════════════════════════════════════

section("3. Faithfulness Check — Ollama (qwen2.5:7b)")

FAITHFULNESS_PASS = 0.75  # 75 % required

FAITH_CASES = [
    {
        "context": (
            "The project budget is $50,000 USD. "
            "The timeline is 8 weeks. "
            "The client is TechCorp Inc."
        ),
        "question": "What is the project budget and who is the client?",
        "expected_facts": ["50,000", "TechCorp"],
    },
    {
        "context": (
            "The target audience is women aged 30-45 in urban areas. "
            "The brand personality is luxurious and sophisticated. "
            "Distribution channels include Instagram and Pinterest."
        ),
        "question": "What platforms will be used for distribution?",
        "expected_facts": ["Instagram", "Pinterest"],
    },
    {
        "context": (
            "The primary objective is to increase brand awareness by 30% "
            "within 6 months. Success metrics include social media reach "
            "and website traffic."
        ),
        "question": "What is the primary objective of this project?",
        "expected_facts": ["brand awareness", "30%"],
    },
]

async def run_faithfulness_checks() -> float:
    import httpx

    ollama_url   = (settings.ollama_base_url if settings else "http://localhost:11434").rstrip("/")
    ollama_model = settings.ollama_model if settings else "qwen2.5:7b"

    # Quick connectivity probe
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{ollama_url}/api/tags")
            r.raise_for_status()
        _record(f"Ollama server reachable at {ollama_url}", True)
    except Exception as e:
        _record(f"Ollama server reachable at {ollama_url}", False, str(e))
        print(f"\n  {WARN}  Faithfulness checks skipped — Ollama not reachable")
        return 0.0

    # Check if the required model is pulled
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{ollama_url}/api/tags")
            tags_data = r.json()
        model_names = [m.get("name", "") for m in tags_data.get("models", [])]
        model_available = any(ollama_model in n for n in model_names)
        _record(
            f"Model '{ollama_model}' available in Ollama",
            model_available,
            f"Available models: {', '.join(model_names[:8]) or 'none'}",
        )
        if not model_available:
            print(f"\n  {WARN}  Run: ollama pull {ollama_model}")
            return 0.0
    except Exception as e:
        _record("Ollama model availability check", False, str(e))
        return 0.0

    passed_cases = 0
    total_cases  = len(FAITH_CASES)

    for i, case in enumerate(FAITH_CASES, 1):
        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer ONLY based on the context provided. "
                    "Do NOT add information not present in the context."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{case['context']}\n\n"
                    f"Question: {case['question']}\n\n"
                    "Answer concisely in 1-2 sentences."
                ),
            },
        ]
        try:
            t0 = time.time()
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{ollama_url}/api/chat",
                    json={
                        "model": ollama_model,
                        "messages": prompt_messages,
                        "stream": False,
                        "options": {"temperature": 0.1},
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            elapsed = time.time() - t0
        except Exception as e:
            _record(f"Faithfulness case {i} — LLM call", False, str(e))
            continue

        answer = data.get("message", {}).get("content", "").strip()

        # Check that the expected facts appear in the answer (case-insensitive)
        facts_found = [
            f for f in case["expected_facts"]
            if f.lower() in answer.lower()
        ]
        facts_ok = len(facts_found) == len(case["expected_facts"])
        if facts_ok:
            passed_cases += 1

        label = (
            f"Faithfulness case {i}: {len(facts_found)}/{len(case['expected_facts'])} "
            f"facts present [{elapsed:.1f}s]"
        )
        detail = (
            f"Q:      {case['question']}\n"
            f"           Expected: {case['expected_facts']}\n"
            f"           Answer:   {answer[:120]}"
        )
        _record(label, facts_ok, detail)

    faithfulness_rate = passed_cases / total_cases if total_cases else 0.0
    overall_pass = faithfulness_rate >= FAITHFULNESS_PASS
    _record(
        f"OVERALL faithfulness = {faithfulness_rate:.0%} ({passed_cases}/{total_cases}) "
        f"(≥{FAITHFULNESS_PASS:.0%} required)",
        overall_pass,
    )
    return faithfulness_rate


# ══════════════════════════════════════════════════════════════════════════════
# Final summary
# ══════════════════════════════════════════════════════════════════════════════

async def main() -> None:
    print(f"\n{BOLD}{'═'*70}{RESET}")
    print(f"{BOLD}  Picasso AI Model — Config & Quality Validation{RESET}")
    print(f"{BOLD}{'═'*70}{RESET}")
    print(f"  Model stack: Ollama qwen2.5:7b + google/embeddinggemma-300m (768-dim)")
    print(f"  Pass threshold: cosine ≥ 0.75 | faithfulness ≥ 75%")

    emb_scores = await run_embedding_checks()
    faith_rate  = await run_faithfulness_checks()

    # ── Final report ──────────────────────────────────────────────────────────
    section("FINAL SUMMARY")

    total   = len(_results)
    passed  = sum(1 for _, ok, _ in _results if ok)
    failed  = total - passed
    pct     = passed / total * 100 if total else 0

    # Key quality metrics
    avg_cos = emb_scores.get("average_positive", 0.0)
    cos_ok  = avg_cos >= COSINE_PASS_THRESHOLD
    faith_ok = faith_rate >= FAITHFULNESS_PASS

    print(f"\n  Total checks : {total}")
    print(f"  Passed       : {GREEN}{passed}{RESET}")
    print(f"  Failed       : {RED}{failed}{RESET}")
    print(f"  Score        : {pct:.1f}%")

    print(f"\n  {BOLD}Quality Metrics:{RESET}")
    cosine_bar = f"{avg_cos:.4f}" if avg_cos else "N/A (model not loaded)"
    print(f"    Cosine similarity (avg) : {(GREEN if cos_ok else RED)}{cosine_bar}{RESET}  "
          f"(threshold ≥ {COSINE_PASS_THRESHOLD})")
    faith_bar = f"{faith_rate:.0%}" if faith_rate else "N/A (Ollama unreachable)"
    print(f"    Faithfulness rate       : {(GREEN if faith_ok else RED)}{faith_bar}{RESET}  "
          f"(threshold ≥ {FAITHFULNESS_PASS:.0%})")

    verdict_pass = failed == 0 and cos_ok and faith_ok
    if verdict_pass:
        print(f"\n  {GREEN}{BOLD}✔ VERDICT: ALL CHECKS PASSED — Model configuration is VALID ✔{RESET}")
    else:
        print(f"\n  {RED}{BOLD}✗ VERDICT: SOME CHECKS FAILED — Review issues above{RESET}")
        if failed > 0:
            print(f"\n  {BOLD}Failed checks:{RESET}")
            for label, ok, detail in _results:
                if not ok:
                    print(f"    {RED}✗{RESET}  {label}")
                    if detail:
                        print(f"       → {detail[:120]}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
