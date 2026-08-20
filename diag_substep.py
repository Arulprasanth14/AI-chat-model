"""
diag_substep.py
───────────────
Sub-step timing diagnostic for the two remaining bottlenecks:
  A. RAG retrieval (embedding API + pgvector query)
  B. save_session (connection acquisition + SQL execution)

Run with:
  .\.venv\Scripts\python.exe diag_substep.py

All timings are printed to stdout AND written to diag_substep.txt.
"""
from __future__ import annotations

import asyncio
import time
import json
from pathlib import Path

# ── Bootstrap app settings (reads .env) ───────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import settings
from app.domain.rag.embedder import OpenAIEmbedder
from app.infrastructure.vector_db.pgvector_client import PgVectorClient
from app.infrastructure.persistence.postgres_session_repo import PostgresSessionRepository
from app.domain.conversation.state import ConversationState
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text

RESULTS = []

def log(msg: str) -> None:
    print(msg)
    RESULTS.append(msg)


async def main() -> None:
    log("=" * 70)
    log("SUB-STEP LATENCY DIAGNOSTIC")
    log("=" * 70)

    # ── Create engine (same config as deps.py) ────────────────────────────────
    engine = create_async_engine(
        settings.database_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        echo=False,
    )
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    embedder = OpenAIEmbedder(api_key=settings.openai_api_key)
    vector_client = PgVectorClient(engine=engine, session_factory=session_factory)
    repo = PostgresSessionRepository(session_factory=session_factory)

    QUERY_TEXT = "Hi, I just want to understand the pricing and what services you offer for restaurants in New York."

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION A: RAG RETRIEVAL SUB-STEPS
    # ═══════════════════════════════════════════════════════════════════════════
    log("\n─── SECTION A: RAG Retrieval Sub-Steps ──────────────────────────────────")

    # ── A1: OpenAI Embedding API — COLD (first call, connection setup) ─────────
    log("\n[A1] OpenAI Embedding API — COLD (first call)")
    t = time.perf_counter()
    embedding = await embedder.embed(QUERY_TEXT)
    a1_cold = time.perf_counter() - t
    log(f"     Time: {a1_cold:.3f}s | vector dims: {len(embedding)}")

    # ── A2: OpenAI Embedding API — WARM (connection already open) ─────────────
    log("\n[A2] OpenAI Embedding API — WARM (second call, same process)")
    t = time.perf_counter()
    embedding2 = await embedder.embed(QUERY_TEXT)
    a2_warm = time.perf_counter() - t
    log(f"     Time: {a2_warm:.3f}s")

    # ── A3: OpenAI Embedding API — repeated (3rd call, same text) ─────────────
    log("\n[A3] OpenAI Embedding API — repeated same text (no caching)")
    t = time.perf_counter()
    embedding3 = await embedder.embed(QUERY_TEXT)
    a3_repeat = time.perf_counter() - t
    log(f"     Time: {a3_repeat:.3f}s")
    log(f"     FINDING: Embedding is called fresh every single turn. Same text = {a3_repeat:.3f}s wasted per repeat query.")

    # ── A4: pgvector similarity search — COLD connection ──────────────────────
    log("\n[A4] pgvector search — COLD DB connection (first query)")
    # Use a known profile_id from your setup
    profile_id = settings.active_profile
    t = time.perf_counter()
    try:
        results = await vector_client.search(
            query_embedding=embedding,
            profile_id=profile_id,
            top_k=5,
        )
        a4_cold = time.perf_counter() - t
        log(f"     Time: {a4_cold:.3f}s | results: {len(results)}")
    except Exception as e:
        a4_cold = time.perf_counter() - t
        log(f"     Time: {a4_cold:.3f}s | ERROR: {e}")

    # ── A5: pgvector similarity search — WARM (pool reused) ───────────────────
    log("\n[A5] pgvector search — WARM (connection from pool)")
    t = time.perf_counter()
    try:
        results2 = await vector_client.search(
            query_embedding=embedding,
            profile_id=profile_id,
            top_k=5,
        )
        a5_warm = time.perf_counter() - t
        log(f"     Time: {a5_warm:.3f}s | results: {len(results2)}")
    except Exception as e:
        a5_warm = time.perf_counter() - t
        log(f"     Time: {a5_warm:.3f}s | ERROR: {e}")

    # ── A6: DB connection acquisition only (raw ping) ─────────────────────────
    log("\n[A6] Raw DB connection acquisition time (pool checkout + 'SELECT 1')")
    async with engine.connect() as conn:  # warm the pool
        await conn.execute(text("SELECT 1"))

    # Now cold: dispose pool and reopen
    await engine.dispose()
    engine2 = create_async_engine(
        settings.database_url,
        pool_size=1,
        max_overflow=0,
        pool_pre_ping=False,  # no pre-ping so we measure raw connect
        echo=False,
    )
    t = time.perf_counter()
    async with engine2.connect() as conn:
        await conn.execute(text("SELECT 1"))
    a6_conn = time.perf_counter() - t
    log(f"     COLD new connection + SELECT 1: {a6_conn:.3f}s")
    await engine2.dispose()

    # Warm connection from the main pool:
    t = time.perf_counter()
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    a6_warm = time.perf_counter() - t
    log(f"     WARM pool checkout + SELECT 1:  {a6_warm:.3f}s")
    log(f"     FINDING: Cold-connect overhead = {a6_conn - a6_warm:.3f}s (Neon serverless suspends after inactivity)")

    # ── A7: Full end-to-end RAG (embed + search, serial) ─────────────────────
    log("\n[A7] Full RAG pipeline (embed → search), warm pool")
    t = time.perf_counter()
    t_embed = time.perf_counter()
    emb = await embedder.embed(QUERY_TEXT)
    d_embed = time.perf_counter() - t_embed
    t_search = time.perf_counter()
    try:
        _ = await vector_client.search(query_embedding=emb, profile_id=profile_id, top_k=5)
        d_search = time.perf_counter() - t_search
    except Exception as e:
        d_search = time.perf_counter() - t_search
        log(f"     search error: {e}")
    a7_total = time.perf_counter() - t
    log(f"     Embed: {d_embed:.3f}s | Search: {d_search:.3f}s | Total: {a7_total:.3f}s")
    log(f"     FINDING: Embed = {d_embed/a7_total*100:.0f}% of RAG time. Search = {d_search/a7_total*100:.0f}%.")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION B: save_session SUB-STEPS
    # ═══════════════════════════════════════════════════════════════════════════
    log("\n─── SECTION B: save_session Sub-Steps ───────────────────────────────────")

    # Create a real session state to upsert
    state = ConversationState(profile_id="test_diag")
    state.add_turn("user", "hello")
    state.add_turn("assistant", "hi there, how can I help?")

    # ── B1: Full save_session (first call, cold pool) ─────────────────────────
    log("\n[B1] save_session — COLD (first call)")
    t = time.perf_counter()
    await repo.save_session(state)
    b1_cold = time.perf_counter() - t
    log(f"     Time: {b1_cold:.3f}s")

    # ── B2: Full save_session — WARM (pool established) ───────────────────────
    log("\n[B2] save_session — WARM (pool reused)")
    t = time.perf_counter()
    await repo.save_session(state)
    b2_warm = time.perf_counter() - t
    log(f"     Time: {b2_warm:.3f}s")

    # ── B3: save_session with large conversation history ─────────────────────
    log("\n[B3] save_session — 20 turns of history (realistic long session)")
    for i in range(20):
        state.add_turn("user", f"User message turn {i} " + ("x" * 200))
        state.add_turn("assistant", f"Assistant reply turn {i} " + ("x" * 500))
    t = time.perf_counter()
    await repo.save_session(state)
    b3_large = time.perf_counter() - t
    log(f"     History size: {len(state.conversation_history)} messages | Payload chars: ~{sum(len(m.get('content','')) for m in state.conversation_history)}")
    log(f"     Time: {b3_large:.3f}s")
    log(f"     FINDING: Large conversation history adds {b3_large - b2_warm:.3f}s vs warm small write.")

    # ── B4: Break down connection acquisition vs SQL execution ────────────────
    log("\n[B4] Connection acquisition vs SQL execution (raw)")
    # Time connection checkout only
    t = time.perf_counter()
    async with session_factory() as sess:
        d_conn_acquire = time.perf_counter() - t
        t_sql = time.perf_counter()
        await sess.execute(text("SELECT 1"))
        d_sql = time.perf_counter() - t_sql
    log(f"     Connection acquire: {d_conn_acquire:.3f}s | SQL exec: {d_sql:.3f}s")

    # ── B5: Is save_session blocking response delivery? ───────────────────────
    log("\n[B5] Could save_session be fire-and-forget safely?")
    log("     Current state: save_session IS awaited synchronously AFTER yield,")
    log("     which means the user's SSE stream is BLOCKED until the DB write completes.")
    log("     The final yield (snapshot event) does NOT happen until after save_session.")
    log("     Fire-and-forget would require yielding the snapshot BEFORE the write.")
    log("     CORRECTNESS IMPACT: The snapshot event carries the final session state.")
    log("     If save_session fails fire-and-forget, the client still sees the correct")
    log("     snapshot (in memory), but crash/restart would lose that turn's data.")
    log("     This is a durability trade-off that requires EXPLICIT USER APPROVAL.")

    # ═══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("SUMMARY TABLE")
    log("=" * 70)
    log(f"{'Step':<45} {'Time':>8}")
    log("-" * 55)
    log(f"{'A1 OpenAI embed (cold, first call)':<45} {a1_cold:>7.3f}s")
    log(f"{'A2 OpenAI embed (warm, 2nd call)':<45} {a2_warm:>7.3f}s")
    log(f"{'A3 OpenAI embed (same text, no cache)':<45} {a3_repeat:>7.3f}s")
    log(f"{'A4 pgvector search (cold connection)':<45} {a4_cold:>7.3f}s")
    log(f"{'A5 pgvector search (warm pool)':<45} {a5_warm:>7.3f}s")
    log(f"{'A6 New TCP connection to Neon (cold)':<45} {a6_conn:>7.3f}s")
    log(f"{'A6 Pool checkout (warm)':<45} {a6_warm:>7.3f}s")
    log(f"{'A7 Full RAG: embed ({d_embed:.3f}s) + search ({d_search:.3f}s)':<45} {a7_total:>7.3f}s")
    log("-" * 55)
    log(f"{'B1 save_session (cold)':<45} {b1_cold:>7.3f}s")
    log(f"{'B2 save_session (warm, same state)':<45} {b2_warm:>7.3f}s")
    log(f"{'B3 save_session (20-turn history)':<45} {b3_large:>7.3f}s")
    log(f"{'B4 connection acquire (pool)':<45} {d_conn_acquire:>7.3f}s")
    log(f"{'B4 SQL exec time (SELECT 1)':<45} {d_sql:>7.3f}s")
    log("=" * 70)

    # Diagnose root cause
    log("\nROOT CAUSE ANALYSIS:")
    if a2_warm > 0.5:
        log(f"  ✗ EMBEDDING: Warm embed = {a2_warm:.3f}s. This is a network call to OpenAI")
        log(f"    on EVERY turn — no caching. Since the query text is unique per turn,")
        log(f"    this is largely unavoidable, but repeated identical queries waste {a3_repeat:.3f}s.")
    else:
        log(f"  ✓ EMBEDDING: Warm embed = {a2_warm:.3f}s — acceptable.")

    if a4_cold > 1.0 and a5_warm < 0.3:
        log(f"\n  ✗ NEON COLD-START: pgvector search cold={a4_cold:.3f}s vs warm={a5_warm:.3f}s.")
        log(f"    Cold-start penalty = {a4_cold - a5_warm:.3f}s. This is Neon's serverless compute")
        log(f"    resuming after inactivity. Cannot be fixed at the app level — only by:")
        log(f"    (a) keeping the connection pool warm, or (b) upgrading to Neon paid plan.")
    elif a5_warm > 0.5:
        log(f"\n  ✗ PGVECTOR QUERY: Even warm, search takes {a5_warm:.3f}s. Check IVFFlat index")
        log(f"    configuration or consider reducing top_k={settings.retrieval_top_k}.")
    else:
        log(f"\n  ✓ PGVECTOR SEARCH: Warm query = {a5_warm:.3f}s — acceptable.")

    if b2_warm > 0.5:
        log(f"\n  ✗ SAVE_SESSION WARM: Even with warm pool, write takes {b2_warm:.3f}s.")
        log(f"    This suggests Neon's compute or network RTT is the bottleneck,")
        log(f"    not connection acquisition. The upsert payload includes full conversation")
        log(f"    history serialized as JSONB. Larger history = bigger payload = more I/O.")
    else:
        log(f"\n  ✓ SAVE_SESSION WARM: {b2_warm:.3f}s — fast. Cold-start was the issue.")

    log("")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    await engine.dispose()

    # Write results to file
    with open("diag_substep.txt", "w") as f:
        f.write("\n".join(RESULTS))
    print("\n\nResults written to diag_substep.txt")


if __name__ == "__main__":
    asyncio.run(main())
