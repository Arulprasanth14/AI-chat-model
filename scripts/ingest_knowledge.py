"""
scripts/ingest_knowledge.py
────────────────────────────
CLI script to ingest knowledge documents for a project profile.

Usage:
    python scripts/ingest_knowledge.py --profile picasso_fusion
    python scripts/ingest_knowledge.py --profile picasso_fusion --no-clear

This reads all .md files from the profile's knowledge_docs/ directory,
chunks them, embeds them via OpenAI, and upserts to the Neon Postgres
pgvector table.

Run this after:
  1. Setting up your .env file with OPENAI_API_KEY and DATABASE_URL
  2. Adding or updating documents in knowledge_docs/
  3. Any time you change the chunking strategy (use --no-clear to add only)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.core.logging import configure_logging


async def main(profile_name: str, clear_existing: bool) -> None:
    """Run the ingestion pipeline for the specified profile."""
    configure_logging(settings.log_level)
    logger = logging.getLogger("ingest_knowledge")

    from app.api.deps import get_embedder, get_vector_store
    from app.domain.rag.ingestion import KnowledgeIngester
    from app.infrastructure.vector_db.pgvector_client import PgVectorClient
    from app.project_profiles.base_profile import BaseProfile

    # Load profile
    profile_dir = Path(__file__).parent.parent / "app" / "project_profiles" / profile_name
    yaml_path = profile_dir / "profile.yaml"

    if not yaml_path.exists():
        logger.error(f"Profile YAML not found: {yaml_path}")
        sys.exit(1)

    profile = BaseProfile.from_yaml(yaml_path)
    knowledge_docs_path = profile_dir / "knowledge_docs"

    logger.info(
        f"Starting ingestion",
        extra={
            "profile_id": profile.profile_id,
            "knowledge_docs": str(knowledge_docs_path),
            "clear_existing": clear_existing,
        },
    )

    # Initialise vector store
    vector_store = get_vector_store()
    await vector_store.initialise()

    embedder = get_embedder()
    ingester = KnowledgeIngester(embedder=embedder, vector_store=vector_store)

    stats = await ingester.ingest_profile(
        profile_id=profile.profile_id,
        knowledge_docs_path=knowledge_docs_path,
        clear_existing=clear_existing,
    )

    print("\n[SUCCESS] Ingestion complete!")
    print(f"   Documents processed: {stats['docs_processed']}")
    print(f"   Chunks created:      {stats['chunks_created']}")
    print(f"   Chunks upserted:     {stats['chunks_upserted']}")
    print(f"   Profile:             {profile.profile_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest knowledge docs for a profile")
    parser.add_argument(
        "--profile",
        default=settings.active_profile,
        help=f"Profile name (default: {settings.active_profile})",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not clear existing chunks before ingesting (additive mode)",
    )
    args = parser.parse_args()

    asyncio.run(main(profile_name=args.profile, clear_existing=not args.no_clear))
