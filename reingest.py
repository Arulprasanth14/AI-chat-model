"""
reingest.py
────────────
One-shot script to re-ingest the Picasso Fusion knowledge base using the
active embedding model (google/embeddinggemma-300m).

Run this whenever:
  - You switch the embedding model
  - You add or update knowledge_docs/ files
  - You want a clean slate

Usage:
    python reingest.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.api.deps import get_embedder, get_vector_store
from app.domain.rag.ingestion import KnowledgeIngester


async def main() -> None:
    print("=" * 60)
    print("  PICASSO AI — Knowledge Base Re-ingestion")
    print("=" * 60)
    print(f"  Embedding provider : {settings.embedding_provider}")
    print(f"  Embedding model    : {settings.local_embedding_model}")
    print(f"  Active profile     : {settings.active_profile}")
    print("=" * 60)

    # Locate the knowledge docs folder
    profile_dir = (
        Path(__file__).parent
        / "app"
        / "project_profiles"
        / settings.active_profile
    )
    knowledge_docs_path = profile_dir / "knowledge_docs"

    if not knowledge_docs_path.exists():
        print(f"\n[ERROR] knowledge_docs folder not found: {knowledge_docs_path}")
        sys.exit(1)

    md_files = list(knowledge_docs_path.rglob("*.md"))
    print(f"\n  Found {len(md_files)} markdown file(s) to ingest:")
    for f in md_files:
        print(f"    - {f.relative_to(profile_dir)}")

    print("\n  Loading embedding model (first run downloads from HuggingFace)...")
    embedder = get_embedder()

    print("  Connecting to pgvector database...")
    vector_store = get_vector_store()

    # Initialise tables and HNSW index
    await vector_store.initialise()
    print("  HNSW index ready.")

    ingester = KnowledgeIngester(embedder=embedder, vector_store=vector_store)

    print(f"\n  Starting ingestion for profile: '{settings.active_profile}'")
    print("  (clear_existing=True — old vectors will be deleted first)\n")

    stats = await ingester.ingest_profile(
        profile_id=settings.active_profile,
        knowledge_docs_path=knowledge_docs_path,
        clear_existing=True,
    )

    print("\n" + "=" * 60)
    print("  INGESTION COMPLETE")
    print("=" * 60)
    print(f"  Documents processed : {stats['docs_processed']}")
    print(f"  Chunks created      : {stats['chunks_created']}")
    print(f"  Chunks upserted     : {stats['chunks_upserted']}")
    print("=" * 60)
    print("\n  Done! Your knowledge base is now indexed with embeddinggemma-300m.")
    print("  Restart the FastAPI server to pick up the new configuration.\n")


if __name__ == "__main__":
    asyncio.run(main())
