"""
scripts/eval_retrieval.py
──────────────────────────
CLI script for manual RAG retrieval quality evaluation.

Usage:
    python scripts/eval_retrieval.py --profile picasso_fusion --query "how should I ask about budget"
    python scripts/eval_retrieval.py --profile picasso_fusion --query "target audience examples" --top-k 10

Prints top-k retrieved chunks with similarity scores and metadata.
Use this to validate that your knowledge_docs are being retrieved correctly
before running full conversation tests.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.core.logging import configure_logging


async def main(profile_name: str, query: str, top_k: int) -> None:
    """Run a retrieval query and print results."""
    configure_logging("WARNING")  # quiet for eval output

    from app.api.deps import get_embedder, get_vector_store
    from app.domain.conversation.state import MissingField
    from app.domain.rag.retriever import RAGRetriever, RetrievalQuery
    from app.project_profiles.base_profile import BaseProfile

    profile_dir = Path(__file__).parent.parent / "app" / "project_profiles" / profile_name
    profile = BaseProfile.from_yaml(profile_dir / "profile.yaml")

    vector_store = get_vector_store()
    embedder = get_embedder()
    retriever = RAGRetriever(embedder=embedder, vector_store=vector_store, top_k=top_k)

    rq = RetrievalQuery(
        user_message=query,
        recent_history=[],
        missing_fields=[],
        profile_id=profile.profile_id,
    )

    print(f"\n🔍 Query: {query!r}")
    print(f"   Profile: {profile.profile_id} | Top-k: {top_k}\n")
    print("─" * 60)

    chunks = await retriever.retrieve(rq, top_k=top_k)

    if not chunks:
        print("⚠️  No chunks retrieved. Have you run ingest_knowledge.py?")
        return

    for i, chunk in enumerate(chunks, 1):
        print(f"\n[{i}] Score: {chunk.score:.4f}  |  Type: {chunk.doc_type}"
              + (f"  |  Field: {chunk.field_code}" if chunk.field_code else ""))
        print("─" * 60)
        # Print first 400 chars of content
        preview = chunk.content[:400].replace("\n", " ")
        print(f"    {preview}{'…' if len(chunk.content) > 400 else ''}")

    print(f"\n✅ {len(chunks)} chunks retrieved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality")
    parser.add_argument("--profile", default=settings.active_profile)
    parser.add_argument("--query", required=True, help="Query text to test")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    asyncio.run(main(profile_name=args.profile, query=args.query, top_k=args.top_k))
