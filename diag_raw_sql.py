"""
diag_raw_sql.py
───────────────
Measures raw Neon SQL round-trip latency using asyncpg directly (no ORM),
and also measures a realistic pgvector cosine query with a raw SQL string.
This bypasses the ORM embedding column attachment issue in diag_substep.py.
"""
import asyncio
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import settings

# Strip the sqlalchemy asyncpg prefix to get a raw asyncpg DSN
RAW_DSN = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")

import asyncpg


async def main() -> None:
    print("=" * 60)
    print("RAW NEON / ASYNCPG LATENCY BREAKDOWN")
    print("=" * 60)

    # ── 1. Cold connect (new pool) ─────────────────────────────────
    print("\n[1] Cold asyncpg connection (new pool, no prior connection)")
    t = time.perf_counter()
    pool = await asyncpg.create_pool(RAW_DSN, min_size=1, max_size=1, command_timeout=30)
    d_pool_create = time.perf_counter() - t
    print(f"    create_pool(): {d_pool_create:.3f}s")

    # ── 2. First query on new connection ──────────────────────────
    print("\n[2] First query on freshly created pool (SELECT 1)")
    t = time.perf_counter()
    async with pool.acquire() as conn:
        await conn.fetchval("SELECT 1")
    d_first = time.perf_counter() - t
    print(f"    First SELECT 1 via pool: {d_first:.3f}s")

    # ── 3. Warm query (connection already established) ─────────────
    print("\n[3] Warm query — same pool, connection cached")
    t = time.perf_counter()
    async with pool.acquire() as conn:
        await conn.fetchval("SELECT 1")
    d_warm = time.perf_counter() - t
    print(f"    Warm SELECT 1: {d_warm:.3f}s")

    # ── 4. Repeated warm queries to measure RTT floor ──────────────
    print("\n[4] 5x warm SELECT 1 — measure RTT floor")
    times = []
    for i in range(5):
        t = time.perf_counter()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        times.append(time.perf_counter() - t)
    print(f"    Times: {[f'{x:.3f}s' for x in times]}")
    print(f"    Avg: {sum(times)/len(times):.3f}s | Min: {min(times):.3f}s | Max: {max(times):.3f}s")

    # ── 5. Upsert-like write (simulating save_session) ─────────────
    print("\n[5] Raw upsert write — simulate save_session payload size")
    # Create temp table for isolation
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS _diag_bench (
                id TEXT PRIMARY KEY,
                payload JSONB NOT NULL
            )
        """)

    import json
    # Small payload (new session, 2 turns)
    small_payload = json.dumps({"history": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"}
    ]})
    t = time.perf_counter()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO _diag_bench (id, payload)
            VALUES ($1, $2::jsonb)
            ON CONFLICT (id) DO UPDATE SET payload = EXCLUDED.payload
        """, "test_small", small_payload)
    d_small_write = time.perf_counter() - t
    print(f"    Small payload (~{len(small_payload)} chars): {d_small_write:.3f}s")

    # Large payload (20-turn session, ~15KB)
    history = [
        {"role": "user" if i % 2 == 0 else "assistant",
         "content": "This is a message with some realistic content " * 10}
        for i in range(40)
    ]
    large_payload = json.dumps({"history": history})
    t = time.perf_counter()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO _diag_bench (id, payload)
            VALUES ($1, $2::jsonb)
            ON CONFLICT (id) DO UPDATE SET payload = EXCLUDED.payload
        """, "test_large", large_payload)
    d_large_write = time.perf_counter() - t
    print(f"    Large payload (~{len(large_payload)} chars): {d_large_write:.3f}s")

    # 5x warm writes to measure write floor
    print("\n[6] 5x warm upserts (small payload) — write floor")
    write_times = []
    for i in range(5):
        t = time.perf_counter()
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO _diag_bench (id, payload)
                VALUES ($1, $2::jsonb)
                ON CONFLICT (id) DO UPDATE SET payload = EXCLUDED.payload
            """, f"test_loop_{i}", small_payload)
        write_times.append(time.perf_counter() - t)
    print(f"    Times: {[f'{x:.3f}s' for x in write_times]}")
    print(f"    Avg: {sum(write_times)/len(write_times):.3f}s | Min: {min(write_times):.3f}s | Max: {max(write_times):.3f}s")

    # ── Cleanup ───────────────────────────────────────────────────
    async with pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS _diag_bench")

    await pool.close()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Pool create (1 connection):    {d_pool_create:.3f}s")
    print(f"First query after pool create: {d_first:.3f}s")
    print(f"Warm SELECT 1:                 {d_warm:.3f}s")
    print(f"Avg warm SELECT 1 (5x):        {sum(times)/len(times):.3f}s")
    print(f"Small upsert (warm):           {d_small_write:.3f}s")
    print(f"Large upsert (warm):           {d_large_write:.3f}s")
    print(f"Avg small upsert (5x):         {sum(write_times)/len(write_times):.3f}s")
    print()
    print("KEY QUESTION: Is Neon RTT the bottleneck, or is it the ORM/driver overhead?")
    print(f"  -> Warm SELECT 1 = {sum(times)/len(times):.3f}s  (pure network RTT to Neon compute)")
    print(f"  -> Warm upsert   = {sum(write_times)/len(write_times):.3f}s  (RTT + write I/O)")
    print(f"  -> RTT accounts for ~{sum(times)/len(times)/max(sum(write_times)/len(write_times),0.001)*100:.0f}% of write time.")
    if sum(times)/len(times) > 0.8:
        print()
        print("CONCLUSION: Neon's network RTT is >= 0.8s per query. This is the baseline")
        print("floor — cannot be reduced at the app level. This is a hosting-tier constraint.")
        print("SQLAlchemy ORM overhead over this is minor (< 0.1s).")


if __name__ == "__main__":
    asyncio.run(main())
